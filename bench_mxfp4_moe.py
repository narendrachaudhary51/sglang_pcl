#!/usr/bin/env python3
"""A/B harness for the fused MXFP4 MoE path: fused_experts_cpu (CPUQuantMethod.MXFP4).

Exercises ``torch.ops.sgl_kernel.fused_experts_cpu`` with MXFP4 e2m1 weights and
per-32-element e8m0 micro-scales (BF16 activations). Both expert GEMM stages
route through the libxsmm MXFP4 x BF16 backend when SGLANG_CPU_MXFP4_LIBXSMM=1
(bf16-direct output: no F32 scratch + convert epilogue), else the default path
(unpack e2m1->bf16 with the e8m0 scale + AMX bf16 brgemm).

This is the MoE analog of bench_mxfp4_gemm.py and reuses its MXFP4 quantize /
dequantize / VNNI2-reshuffle helpers so the two harnesses stay in lockstep.

------------------------------------------------------------------------------
Benchmarking methodology (identical to bench_mxfp4_gemm.py): allocate
``n_layers`` INDEPENDENT (act, w1, w2, scales, routing) sets so the resident
footprint is >= WORKING_SET_GB, making every fused_experts call genuinely cold /
DRAM-bound; one warmup pass; then ITERS timed passes over all layers.

The backend env var is read once per process, so this script runs ONE backend
per invocation and saves the result to a ``.pt`` file. Run it twice (off / on)
and compare:

    SGLANG_CPU_MXFP4_LIBXSMM=0 python bench_mxfp4_moe.py run /tmp/moe_default.pt
    SGLANG_CPU_MXFP4_LIBXSMM=1 python bench_mxfp4_moe.py run /tmp/moe_libxsmm.pt
    python bench_mxfp4_moe.py compare /tmp/moe_default.pt /tmp/moe_libxsmm.pt

Correctness reporting is controlled by BENCH_CHECK (default "cos"; shared with
bench_mxfp4_gemm): "cos" = cosine vs torch reference, "norm" = libxsmm matdiff
norm block (L1/L2/Linf + Check-norm) a la sfc_ca_gemm with a PASS/FAIL, "both" =
cosine line + full norm block. BENCH_CHECK_NORM_TOL (default 5e-2) is the
Check-norm pass threshold.
"""
import os
import sys
import time

import torch

import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)

# Reuse the dense harness' MXFP4 quantize / dequantize / VNNI2-reshuffle so both
# benchmarks pack weights identically, plus the shared correctness reporting
# (cosine + libxsmm-matdiff norm block) so both stay in lockstep.
from bench_mxfp4_gemm import (  # noqa: E402
    CHECK_MODE,
    CHECK_NORM_TOL,
    GROUP_SIZE,
    dequantize_mxfp4,
    matdiff_norms,
    quantize_mxfp4,
    report_correctness,
)

from sglang.srt.layers.amx_utils import CPUQuantMethod  # noqa: E402

OPS = torch.ops.sgl_kernel
torch.manual_seed(1234)

LIBXSMM_ON = os.environ.get("SGLANG_CPU_MXFP4_LIBXSMM", "0") == "1"

# (M, N, K, E, topk): M tokens, N intermediate (per gate/up half), K hidden,
# E experts, topk experts/token. N & K must be multiples of 32. Default mirrors
# a gpt-oss-20b style block. Override with
#   BENCH_MOE_SHAPES="M,N,K,E,topk;M,N,K,E,topk;...".
SHAPES = [
    (128, 2880, 2880, 32, 4),
    (1, 2880, 2880, 32, 4),
]
_env_shapes = os.environ.get("BENCH_MOE_SHAPES", "").strip()
if _env_shapes:
    parsed = []
    for grp in _env_shapes.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        vals = [v.strip() for v in grp.split(",") if v.strip()]
        if len(vals) != 5:
            raise SystemExit(
                f"Bad BENCH_MOE_SHAPES entry {grp!r} (got {len(vals)} value(s), need 5 "
                f"as M,N,K,E,topk). Full BENCH_MOE_SHAPES={_env_shapes!r}. Did you forget "
                'to quote it? Use: BENCH_MOE_SHAPES="1,2880,2880,32,4;128,2880,2880,32,4"'
            )
        parsed.append(tuple(int(v) for v in vals))
    SHAPES = parsed

WORKING_SET_GB = float(os.environ.get("BENCH_WORKING_SET_GB", "5.0"))
WARMUP = 1
ITERS = int(os.environ.get("BENCH_ITERS", "10"))

BLOCK_N = 32  # libxsmm/sglang weight tile width (block_size_n())


def quantize_mxfp4_3d(weight):
    """[E, R, K] bf16 -> (packed_u8 [E, R, K/2], e8m0_u8 [E, R, K/32], dq [E, R, K])."""
    E = weight.shape[0]
    packs, scales, dqs = [], [], []
    for e in range(E):
        p, s = quantize_mxfp4(weight[e], GROUP_SIZE)
        packs.append(p)
        scales.append(s)
        dqs.append(dequantize_mxfp4(p, s, GROUP_SIZE))
    return torch.stack(packs), torch.stack(scales), torch.stack(dqs)


def reshuffle_moe_vnni2(packed_w, num_row_blocks, K):
    """Re-shuffle a per-expert sglang 32-way mxfp4 pack (already run through
    convert_weight_packed) into standard MXFP4-VNNI2, the layout libxsmm expects.

    convert_weight_packed lays the bytes out as [.., R/BN, K/2, BN]; this is the
    3D analog of bench_mxfp4_gemm.sglang_to_libxsmm_vnni2 -- the per-BN-tile
    nibble shuffle is identical, just flattened over (E * R / BN) tiles. Done
    ONCE at load time, never on the timed critical path.
    """
    BN = BLOCK_N
    K2 = K // 2
    buf = packed_w.contiguous().view(torch.uint8).reshape(num_row_blocks, K2, BN)
    out = torch.empty_like(buf)
    even = buf[:, :, 0::2]  # [blocks, K2, 16]
    odd = buf[:, :, 1::2]   # [blocks, K2, 16]
    out[:, :, 0:16] = ((odd & 0x0F) << 4) | (even & 0x0F)
    out[:, :, 16:32] = (odd & 0xF0) | (even >> 4)
    return out.reshape(packed_w.shape)


def layer_bytes(M, N, K, E, topk):
    # full resident footprint of one layer (all E experts kept in memory)
    w = E * (2 * N * (K // 2) + K * (N // 2))      # w1 + w2 mxfp4 bytes
    s = E * (2 * N * (K // 32) + K * (N // 32))     # e8m0 scales
    a = M * K * 2                                   # activation bf16
    return w + s + a


def active_weight_bytes(topk_ids, N, K):
    # DRAM weight traffic per layer: each *active* expert's w1+w2 read ~once
    # (blocks of the same expert reuse it from cache). topk_ids: [M, topk].
    n_active = int(torch.unique(topk_ids).numel())
    per_expert = 2 * N * (K // 2) + K * (N // 2)
    return n_active * per_expert


def streamed_bytes(M, N, K, E, topk, topk_ids):
    # weight traffic + activation in/out + the two intermediate caches the kernel
    # writes/reads (ic0 [M*topk, 2N], ic1 [M*topk, N], ic2 [M*topk, K]).
    w = active_weight_bytes(topk_ids, N, K)
    act = M * K * 2 + M * K * 2
    inter = M * topk * (2 * N + N + K) * 2
    return w + act + inter


def moe_flops(M, N, K, topk):
    # stage1: [M*topk] x (2N x K), stage2: [M*topk] x (K x N)
    return 2.0 * M * topk * (2 * N * K + K * N)


def auto_n_layers(M, N, K, E, topk):
    per = layer_bytes(M, N, K, E, topk)
    n = 1
    while (n * per) / 1024**3 < WORKING_SET_GB:
        n += 1
    return n


def torch_ref_moe(a, w1_dq, w2_dq, topk_weights, topk_ids):
    """Dense reference: silu_and_mul fused MoE. a [M,K], w1_dq [E,2N,K],
    w2_dq [E,K,N]. Matches the kernel: ic1 = silu(ic0[:N]) * ic0[N:]."""
    M, K = a.shape
    N = w1_dq.shape[1] // 2
    topk = topk_ids.shape[1]
    af = a.float()
    out = torch.zeros(M, K, dtype=torch.float32)
    for m in range(M):
        for j in range(topk):
            e = int(topk_ids[m, j])
            wgt = float(topk_weights[m, j])
            g = af[m] @ w1_dq[e].float().T          # [2N]
            act = torch.nn.functional.silu(g[:N]) * g[N:]
            out[m] += wgt * (act @ w2_dq[e].float().T)  # [K]
    return out.to(torch.bfloat16)


def make_layer(M, N, K, E, topk):
    a = torch.randn(M, K, dtype=torch.bfloat16) / 10
    w1_bf16 = torch.randn(E, 2 * N, K, dtype=torch.bfloat16) / 10
    w2_bf16 = torch.randn(E, K, N, dtype=torch.bfloat16) / 10

    w1q, w1s, w1dq = quantize_mxfp4_3d(w1_bf16)
    w2q, w2s, w2dq = quantize_mxfp4_3d(w2_bf16)

    # Pre-pack weights + scales ONCE, as real serving does at load time, so the
    # packing never runs on the timed critical path. For the libxsmm backend,
    # fold the sglang -> standard-VNNI2 re-shuffle into the upfront prepack.
    pw1 = OPS.convert_weight_packed(w1q)
    pw2 = OPS.convert_weight_packed(w2q)
    if LIBXSMM_ON:
        pw1 = reshuffle_moe_vnni2(pw1, E * (2 * N) // BLOCK_N, K)
        pw2 = reshuffle_moe_vnni2(pw2, E * K // BLOCK_N, N)
    ps1 = OPS.convert_scale_packed(w1s)
    ps2 = OPS.convert_scale_packed(w2s)

    # routing
    score = torch.softmax(torch.randn(M, E), dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    return {
        "a": a,
        "pw1": pw1,
        "pw2": pw2,
        "ps1": ps1,
        "ps2": ps2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1dq": w1dq,
        "w2dq": w2dq,
    }


def call(layer):
    return OPS.fused_experts_cpu(
        layer["a"],
        layer["pw1"],
        layer["pw2"],
        layer["topk_weights"],
        layer["topk_ids"],
        False,                  # inplace
        int(CPUQuantMethod.MXFP4),
        layer["ps1"],           # w1_scale
        layer["ps2"],           # w2_scale
        None,                   # w1_zp
        None,                   # w2_zp
        None,                   # block_size
        None,                   # w1_bias
        None,                   # w2_bias
        None,                   # alpha
        None,                   # limit
        True,                   # is_vnni (prepacked)
    )


def bench_one(M, N, K, E, topk):
    n_layers = auto_n_layers(M, N, K, E, topk)
    resident_gb = n_layers * layer_bytes(M, N, K, E, topk) / 1024**3
    layers = [make_layer(M, N, K, E, topk) for _ in range(n_layers)]

    # Correctness reference from layer 0.
    l0 = layers[0]
    ref = torch_ref_moe(l0["a"], l0["w1dq"], l0["w2dq"], l0["topk_weights"], l0["topk_ids"])

    for _ in range(WARMUP):
        for s in layers:
            call(s)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s in layers:
            call(s)
    t1 = time.perf_counter()

    per_pass_s = (t1 - t0) / ITERS
    per_call_s = per_pass_s / n_layers

    flops = moe_flops(M, N, K, topk)
    tflops = flops / per_call_s / 1e12
    # weight + total BW averaged over the (independently routed) layers
    total_wgt = sum(active_weight_bytes(s["topk_ids"], N, K) for s in layers)
    total_str = sum(streamed_bytes(M, N, K, E, topk, s["topk_ids"]) for s in layers)
    wgt_gbps = total_wgt / per_pass_s / 1e9
    tot_gbps = total_str / per_pass_s / 1e9

    out0 = call(l0)
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().float(), out0.flatten().float(), dim=0
    ).item()

    return {
        "us": per_call_s * 1e6,
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
    backend = "libxsmm" if LIBXSMM_ON else "default"
    print(f"[{backend}] threads={torch.get_num_threads()}  shapes={len(SHAPES)}  "
          f"working_set>={WORKING_SET_GB:.1f}GB  warmup={WARMUP} iters={ITERS}  check={CHECK_MODE}")
    results = {}
    all_passed = True
    for (M, N, K, E, topk) in SHAPES:
        r = bench_one(M, N, K, E, topk)
        key = f"{M}x{N}x{K}x{E}x{topk}"
        results[key] = r
        tail, passed = report_correctness(key, r["norms"], r["cos_vs_torch"])
        if passed is False:
            all_passed = False
        print(f"  M={M:<5} N={N:<6} K={K:<5} E={E:<3} topk={topk:<2} "
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
    print(f"\n{'shape (MxNxKxExtopk)':<26} "
          f"{a['backend'] + ' us':>14} {b['backend'] + ' us':>14}  {'speedup':>8}  "
          f"{'cos(A,B)':>9}")
    print("-" * 80)
    for key in ra:
        if key not in rb:
            continue
        da, db = ra[key], rb[key]
        spd = da["us"] / db["us"]
        cab = torch.nn.functional.cosine_similarity(
            da["out"].flatten().float(), db["out"].flatten().float(), dim=0
        ).item()
        print(f"{key:<26} "
              f"{da['us']:14.1f} {db['us']:14.1f}  {spd:7.2f}x  {cab:9.6f}")
    print("\nspeedup > 1.0 means the second backend (B) is faster.")
    print("cos(A,B) ~ 1.0 confirms the two backends produce the same result.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
