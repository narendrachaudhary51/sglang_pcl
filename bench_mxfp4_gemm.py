#!/usr/bin/env python3
"""A/B harness for the dense MXFP4 GEMM path: mxfp4_scaled_mm_cpu.

Exercises ``torch.ops.sgl_kernel.mxfp4_scaled_mm_cpu`` (MXFP4 e2m1 weights with
per-32-element e8m0 micro-scales, BF16 activations). That path routes through the
libxsmm MXFP4 x BF16 GEMM when SGLANG_CPU_MXFP4_LIBXSMM=1, else the default path
(unpack e2m1->bf16 with the e8m0 scale + AMX bf16 brgemm).

------------------------------------------------------------------------------
Benchmarking methodology (identical to bench_fp8_gemm.py): allocate ``n_layers``
INDEPENDENT (act, weight, scales) sets so the resident footprint is
>= WORKING_SET_GB, making every GEMM genuinely cold / DRAM-bound; one warmup
pass; then ITERS timed passes over all layers.

The backend env var is read once per process, so this script runs ONE backend
per invocation and saves the result to a ``.pt`` file. bench_mxfp4_op.sh runs it
twice (off / on) and compares.

Usage (normally driven by bench_mxfp4_op.sh):
    SGLANG_CPU_MXFP4_LIBXSMM=0 python bench_mxfp4_gemm.py run /tmp/gemm_default.pt
    SGLANG_CPU_MXFP4_LIBXSMM=1 python bench_mxfp4_gemm.py run /tmp/gemm_libxsmm.pt
    python bench_mxfp4_gemm.py compare /tmp/gemm_default.pt /tmp/gemm_libxsmm.pt

Correctness reporting is controlled by BENCH_CHECK (default "cos"):
    BENCH_CHECK=cos   cosine similarity vs the torch reference (default)
    BENCH_CHECK=norm  libxsmm matdiff norm block (L1/L2/Linf + Check-norm) a la
                      sfc_ca_gemm, with a PASS/FAIL on the Check-norm
    BENCH_CHECK=both  cosine on the summary line + the full norm block
  BENCH_CHECK_NORM_TOL (default 5e-2) sets the Check-norm pass threshold.
"""
import os
import sys
import time

import torch

import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)

OPS = torch.ops.sgl_kernel
torch.manual_seed(1234)

# (M, N, K). N multiple of block_size_n()=32; K multiple of group_size=32.
# Override with BENCH_SHAPES="M,N,K;M,N,K;..." for shape sweeps.
SHAPES = [
    (128, 8192, 8192),
]
_env_shapes = os.environ.get("BENCH_SHAPES", "").strip()
if _env_shapes:
    parsed = []
    for grp in _env_shapes.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        vals = [v.strip() for v in grp.split(",") if v.strip()]
        if len(vals) != 3:
            raise SystemExit(
                f"Bad BENCH_SHAPES entry {grp!r} (got {len(vals)} value(s), need 3 as "
                f"M,N,K). Full BENCH_SHAPES={_env_shapes!r}. Did you forget to quote it? "
                'Use: BENCH_SHAPES="16,8192,8192;128,8192,8192"'
            )
        parsed.append(tuple(int(v) for v in vals))
    SHAPES = parsed

GROUP_SIZE = 32  # MXFP4 micro-scale block (e8m0 per 32 K-elements)

WORKING_SET_GB = float(os.environ.get("BENCH_WORKING_SET_GB", "5.0"))
WARMUP = 1
ITERS = int(os.environ.get("BENCH_ITERS", "10"))

_E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32)
_E2M1_MAX = 6.0


def quantize_mxfp4(weight, block=GROUP_SIZE):
    """bf16 weight [N, K] -> (packed_u8 [N, K/2], e8m0_u8 [N, K/32]).

    Mirrors MXFP4QuantizeUtil.quantize (NVIDIA ModelOpt convention): e2m1 nibbles
    packed 2-per-byte (even K-index in low nibble), per-32 e8m0 exponent scale.
    """
    N, K = weight.shape
    x = weight.float().view(-1, block)
    amax = x.abs().max(dim=-1, keepdim=True).values
    descale = amax / _E2M1_MAX
    e8m0 = torch.ceil(torch.maximum(torch.log2(descale), torch.tensor(-127.0)))
    xs = x / torch.exp2(e8m0)
    # cast to e2m1 code (sign bit + 3 magnitude bits)
    sign = torch.sign(xs)
    sign_bit = (2 - sign) // 2
    ord_ = torch.sum((xs.abs().unsqueeze(-1) - _E2M1_BOUNDS) > 0, dim=-1)
    fp4 = (sign_bit * 0b1000 + ord_).to(torch.uint8).view(N, K)
    # fuse 2 nibbles -> 1 byte (even idx low bits, odd idx high bits)
    lo = fp4[..., 0::2]
    hi = fp4[..., 1::2]
    packed = ((hi << 4) + lo).to(torch.uint8).contiguous()  # [N, K/2]
    e8m0 = (e8m0 + 127).to(torch.uint8).view(N, K // block).contiguous()
    return packed, e8m0


def dequantize_mxfp4(packed, e8m0, block=GROUP_SIZE):
    """Inverse of quantize_mxfp4 -> bf16 weight [N, K]."""
    N, Kh = packed.shape
    K = Kh * 2
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    code = torch.zeros(N, K, dtype=torch.long)
    code[..., 0::2] = lo
    code[..., 1::2] = hi
    sign = 1 - 2 * ((code & 0b1000) >> 3).float()
    mag = code & 0b0111
    mant = _E2M1_VALUES[mag.view(-1)].view(N, K)
    vals = (sign * mant).view(-1, block)
    scale = torch.exp2(e8m0.float() - 127).view(-1, 1)
    return (vals * scale).view(N, K).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Correctness reporting. By default we report cosine similarity vs the torch
# reference (BENCH_CHECK=cos). Set BENCH_CHECK=norm to instead report the
# libxsmm matdiff norm block (L1/L2/Linf + Check-norm), exactly as sfc_ca_gemm
# prints it, with a PASS/FAIL on the Check-norm; BENCH_CHECK=both prints cosine
# on the summary line AND the full norm block.
# ---------------------------------------------------------------------------
CHECK_MODE = os.environ.get("BENCH_CHECK", "cos").strip().lower()  # cos | norm | both
# Check-norm (relative Frobenius error) pass threshold for norm/both modes.
CHECK_NORM_TOL = float(os.environ.get("BENCH_CHECK_NORM_TOL", "5e-2"))


def matdiff_norms(ref, tst):
    """Replicate libxsmm_matdiff (the correctness block sfc_ca_gemm prints) in
    float64. Returns a dict of l1_ref, l1_tst, l2_abs, l2_rel, linf_abs,
    linf_rel and normf_rel. ``normf_rel`` is sfc_ca_gemm's "Check-norm": the
    relative Frobenius error ||ref - tst||_F / ||ref||_F, using the same
    zero-denominator fallback as libxsmm (fall back to min(||tst||_F^2,
    sum_di2) when ||ref||_F == 0). di = |ri - ti|; the per-element relative
    error is di/|ri| when |ri| > 0 else |ti|.
    """
    r = ref.detach().flatten().double()
    t = tst.detach().flatten().double()
    d = (r - t).abs()
    ra = r.abs()
    ta = t.abs()
    tiny = torch.finfo(torch.float64).tiny
    dri = torch.where(ra > 0, d / torch.clamp(ra, min=tiny), ta)
    sum_di2 = torch.dot(d, d)
    normfr = torch.dot(r, r)
    normft = torch.dot(t, t)
    if float(normfr) > 0:
        normf_rel = torch.sqrt(sum_di2 / normfr)
    else:
        normf_rel = torch.sqrt(torch.minimum(normft * normft, sum_di2))
    return {
        "l1_ref": float(ra.sum()),
        "l1_tst": float(ta.sum()),
        "l2_abs": float(torch.sqrt(sum_di2)),
        "l2_rel": float(torch.sqrt(torch.dot(dri, dri))),
        "linf_abs": float(d.max()) if d.numel() else 0.0,
        "linf_rel": float(dri.max()) if dri.numel() else 0.0,
        "normf_rel": float(normf_rel),
    }


def format_norm_block(norms, indent="      "):
    """Format a matdiff_norms() dict like sfc_ca_gemm's correctness output."""
    return "\n".join([
        f"{indent}L1 reference  : {norms['l1_ref']:.25g}",
        f"{indent}L1 test       : {norms['l1_tst']:.25g}",
        f"{indent}L2 abs.error  : {norms['l2_abs']:.24f}",
        f"{indent}L2 rel.error  : {norms['l2_rel']:.24f}",
        f"{indent}Linf abs.error: {norms['linf_abs']:.24f}",
        f"{indent}Linf rel.error: {norms['linf_rel']:.24f}",
        f"{indent}Check-norm    : {norms['normf_rel']:.24f}",
    ])


def report_correctness(label, norms, cos):
    """Print the per-shape correctness summary for the active BENCH_CHECK mode and
    return a (printed_tail, passed) tuple. ``passed`` is None in cos-only mode."""
    if CHECK_MODE == "norm":
        tail = f"Check-norm={norms['normf_rel']:.3e}"
    else:
        tail = f"cos_vs_torch={cos:.5f}"
    passed = None
    if CHECK_MODE in ("norm", "both"):
        print(f"  [{label}] correctness (libxsmm matdiff):")
        print(format_norm_block(norms))
        passed = norms["normf_rel"] <= CHECK_NORM_TOL
        verdict = "PASS" if passed else "FAIL"
        print(f"      -> {verdict} (Check-norm {norms['normf_rel']:.3e} <= tol {CHECK_NORM_TOL:.1e}? "
              f"{'yes' if passed else 'NO'})")
    return tail, passed


def sglang_to_libxsmm_vnni2(packed_w, N, K):
    """Re-shuffle the sglang 32-way mxfp4 weight pack into standard MXFP4-VNNI2,
    the layout libxsmm expects. This is an UPFRONT, load-time transform (the
    analog of convert_weight_packed for the default path) and is deliberately
    kept OUT of the timed GEMM loop.

    sglang packs each contiguous [K/2, BLOCK_N=32] tile as
        P[k2, n] = (unpacked[n+32] << 4) | unpacked[n]   (n = 0..31)
    with unpacked[2c] = even-K nibble of column c, unpacked[2c+1] = odd-K nibble.
    libxsmm wants  S[k2, col] = (oddK[col] << 4) | evenK[col]  (columns 0..31),
    cols 0..15 recovered from the low nibbles, cols 16..31 from the high nibbles.
    Mirrors what reshuffle_tile_to_vnni2 did per-tile in C++, but done once here.
    """
    BN = 32
    K2 = K // 2
    NB = N // BN
    buf = packed_w.contiguous().view(torch.uint8).reshape(NB, K2, BN)
    out = torch.empty_like(buf)
    even = buf[:, :, 0::2]  # [NB, K2, 16]
    odd = buf[:, :, 1::2]   # [NB, K2, 16]
    # columns 0..15 : low nibbles of the source row
    out[:, :, 0:16] = ((odd & 0x0F) << 4) | (even & 0x0F)
    # columns 16..31 : high nibbles of the source row
    out[:, :, 16:32] = (odd & 0xF0) | (even >> 4)
    return out.reshape(N, K2).contiguous()


def layer_bytes(M, N, K):
    a = M * K * 2          # activation bf16
    w = N * (K // 2) * 1   # weight mxfp4 (2 nibbles / byte)
    s = N * (K // 32) * 1  # e8m0 scale
    c = M * N * 2          # output bf16
    return a + w + s + c


def weight_bytes(M, N, K):
    return N * (K // 2) * 1   # streamed weight only ('Effective A BW')


def streamed_bytes(M, N, K):
    return M * K * 2 + N * (K // 2) + N * (K // 32) + M * N * 2


def auto_n_layers(M, N, K):
    per = layer_bytes(M, N, K)
    n = 1
    while (n * per) / 1024**3 < WORKING_SET_GB:
        n += 1
    return n


def make_layer(M, N, K):
    data = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16) * 0.1
    packed, e8m0 = quantize_mxfp4(weight, GROUP_SIZE)
    dq_weight = dequantize_mxfp4(packed, e8m0, GROUP_SIZE)
    # Pre-pack the weight (VNNI) + scale ONCE, as real serving does at load time,
    # so the packing never runs on the timed critical path. For the libxsmm
    # backend, fold the sglang -> standard-VNNI2 re-shuffle into this upfront
    # prepack as well (so the timed loop only runs the GEMM, no per-tile fixup).
    packed_w = OPS.convert_weight_packed(packed)
    if os.environ.get("SGLANG_CPU_MXFP4_LIBXSMM", "0") == "1":
        packed_w = sglang_to_libxsmm_vnni2(packed_w, N, K)
    packed_s = OPS.convert_scale_packed(e8m0)
    return data, packed_w, packed_s, dq_weight


def bench_one(M, N, K):
    n_layers = auto_n_layers(M, N, K)
    resident_gb = n_layers * layer_bytes(M, N, K) / 1024**3
    layers = [make_layer(M, N, K) for _ in range(n_layers)]

    # Correctness reference from layer 0 (dequantized weight @ activations).
    data0, _, _, dq_weight0 = layers[0]
    ref = torch.matmul(data0.to(torch.bfloat16), dq_weight0.T)

    def call(s):
        data, packed_w, packed_s, _ = s
        return OPS.mxfp4_scaled_mm_cpu(
            data,
            packed_w,
            packed_s,
            None,            # bias
            True,            # is_vnni / prepacked
        )

    for _ in range(WARMUP):
        for s in layers:
            call(s)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s in layers:
            call(s)
    t1 = time.perf_counter()

    per_pass_s = (t1 - t0) / ITERS
    per_gemm_s = per_pass_s / n_layers

    flops = 2.0 * M * N * K
    tflops = flops / per_gemm_s / 1e12
    wgt_gbps = weight_bytes(M, N, K) / per_gemm_s / 1e9
    tot_gbps = streamed_bytes(M, N, K) / per_gemm_s / 1e9

    out0 = call(layers[0])
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().float(), out0.flatten().float(), dim=0
    ).item()

    return {
        "us": per_gemm_s * 1e6,
        "tflops": tflops,
        "gbps": wgt_gbps,
        "total_gbps": tot_gbps,
        "n_layers": n_layers,
        "resident_gb": resident_gb,
        "cos_vs_torch": cos,
        "norms": matdiff_norms(ref, out0),
        "out": out0.clone(),
    }


def run(out_path):
    backend = "libxsmm" if os.environ.get("SGLANG_CPU_MXFP4_LIBXSMM", "0") == "1" else "default"
    print(f"[{backend}] threads={torch.get_num_threads()}  shapes={len(SHAPES)}  "
          f"working_set>={WORKING_SET_GB:.1f}GB  warmup={WARMUP} iters={ITERS}  check={CHECK_MODE}")
    results = {}
    all_passed = True
    for (M, N, K) in SHAPES:
        r = bench_one(M, N, K)
        key = f"{M}x{N}x{K}"
        results[key] = r
        tail, passed = report_correctness(key, r["norms"], r["cos_vs_torch"])
        if passed is False:
            all_passed = False
        print(f"  M={M:<5} N={N:<6} K={K:<5}  "
              f"{r['us']:9.1f} us  {r['tflops']:7.2f} TFLOP/s  "
              f"{r['gbps']:8.1f} GB/s(wgt)  {r['total_gbps']:8.1f} GB/s(tot)  "
              f"[{r['n_layers']} layers, {r['resident_gb']:.2f}GB resident]  "
              f"{tail}")
    torch.save({"backend": backend, "results": results}, out_path)
    print(f"[{backend}] saved -> {out_path}")
    if CHECK_MODE in ("norm", "both"):
        print(f"[{backend}] correctness: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} "
              f"(Check-norm tol {CHECK_NORM_TOL:.1e})")


def compare(path_a, path_b):
    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    ra, rb = a["results"], b["results"]
    print(f"\n{'shape (MxNxK)':<22} "
          f"{a['backend'] + ' GB/s':>14} {b['backend'] + ' GB/s':>14}  {'speedup':>8}  "
          f"{'cos(A,B)':>9}")
    print("-" * 74)
    for key in ra:
        if key not in rb:
            continue
        da, db = ra[key], rb[key]
        spd = da["us"] / db["us"]
        cab = torch.nn.functional.cosine_similarity(
            da["out"].flatten().float(), db["out"].flatten().float(), dim=0
        ).item()
        print(f"{key:<22} "
              f"{da['gbps']:14.1f} {db['gbps']:14.1f}  {spd:7.2f}x  {cab:9.6f}")
    print("\nGB/s = weight-only 'Effective A BW' (driver metric).")
    print("speedup > 1.0 means the second backend (B) is faster.")
    print("cos(A,B) ~ 1.0 confirms the two backends produce the same result.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
