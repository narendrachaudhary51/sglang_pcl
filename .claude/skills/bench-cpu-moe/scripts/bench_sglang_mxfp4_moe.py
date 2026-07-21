import torch, time
import math
import sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops

grouped_topk_cpu = torch.ops.sgl_kernel.grouped_topk_cpu
fused_experts_cpu = torch.ops.sgl_kernel.fused_experts_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
convert_scale_packed = torch.ops.sgl_kernel.convert_scale_packed

# CPUQuantMethod.MXFP4 (see sglang/srt/layers/amx_utils.py)
MXFP4 = 4
# mxfp4 uses a fixed micro-scaling group size of 32 along K.
GROUP_SIZE = 32
# gpt-oss swiglu activation params (alpha, clamp limit).
ALPHA = 1.702
LIMIT = 7.0


def fused_moe(
    a, packed_w1, packed_w2, w1s, w2s, w1b, w2b, score, topk, renormalize, prepack
):

    G = 1
    topk_group = 1

    B, D = a.shape
    topk_weights = torch.empty(B, topk, dtype=torch.float32)
    topk_ids = torch.empty(B, topk, dtype=torch.int32)
    topk_weights, topk_ids = grouped_topk_cpu(
        a, score, topk, renormalize, G, topk_group, 0, None, None
    )

    unique_experts = topk_ids.unique().numel()
    inplace = False
    start = time.perf_counter()
    fused_output = fused_experts_cpu(
        a,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        inplace,
        MXFP4,
        w1s,  # w1_scale (uint8, e8m0)
        w2s,  # w2_scale (uint8, e8m0)
        None,  # w1_zp
        None,  # w2_zp
        None,  # block_size (mxfp4 group size is fixed at 32)
        w1b,  # w1 bias
        w2b,  # w2 bias
        ALPHA,  # alpha (swiglu)
        LIMIT,  # limit (swiglu)
        prepack,  # is_vnni
    )
    end = time.perf_counter()
    return fused_output, (end - start), unique_experts


def bench_mxfp4_moe(M, N, K, num_experts, topk=8):

    # Build enough distinct (a, b) sets so the combined working set exceeds cache.
    # Size a single layer's footprint, then pick num_layers to blow past a large LLC.
    CACHE_BYTES = 5 * 1024 * 1024 * 1024  # assume 5 GB working set

    active = M * topk if M * topk < num_experts else num_experts
    bytes_per_layer = (
        M * K * 2  # a: bf16
        + active * (2 * N) * (K // 2) * 1  # w1: mxfp4 packed uint8 (2 vals/byte)
        + active * K * (N // 2) * 1  # w2: mxfp4 packed uint8 (2 vals/byte)
        + active * (2 * N) * (K // GROUP_SIZE) * 1  # w1s: e8m0 uint8
        + active * K * (N // GROUP_SIZE) * 1  # w2s: e8m0 uint8
        + active * (2 * N) * 4  # w1 bias: fp32
        + active * K * 4  # w2 bias: fp32
    )

    num_layers = max(3, (CACHE_BYTES // bytes_per_layer) + 1)
    print(f"num_layers={num_layers} | working set ~{num_layers*bytes_per_layer/1e6:.1f} MB")

    a_list = []
    w1_list = []
    w2_list = []
    w1s_list = []
    w2s_list = []
    w1b_list = []
    w2b_list = []
    score_list = []
    for _ in range(num_layers):
        a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)

        # mxfp4 weights: uint8, 2 fp4 values packed per byte along K.
        w1q = torch.randint(0, 256, (num_experts, 2 * N, K // 2), dtype=torch.uint8)
        w2q = torch.randint(0, 256, (num_experts, K, N // 2), dtype=torch.uint8)

        # e8m0 block scales: uint8, one per group of 32 along K. Keep exponents
        # near 127 (scale ~1) to avoid overflow.
        w1s = torch.randint(
            120, 130, (num_experts, 2 * N, K // GROUP_SIZE), dtype=torch.uint8
        )
        w2s = torch.randint(
            120, 130, (num_experts, K, N // GROUP_SIZE), dtype=torch.uint8
        )

        # per-output-channel bias: fp32
        w1b = torch.randn(num_experts, 2 * N, dtype=torch.float32)
        w2b = torch.randn(num_experts, K, dtype=torch.float32)

        score = torch.randn((M, num_experts), device="cpu", dtype=torch.bfloat16)

        w1_packed = convert_weight_packed(w1q)  # pack to VNNI so packing isn't timed
        w2_packed = convert_weight_packed(w2q)
        w1s_packed = convert_scale_packed(w1s)
        w2s_packed = convert_scale_packed(w2s)

        a_list.append(a)
        w1_list.append(w1_packed)
        w2_list.append(w2_packed)
        w1s_list.append(w1s_packed)
        w2s_list.append(w2s_packed)
        w1b_list.append(w1b)
        w2b_list.append(w2b)
        score_list.append(score)

    prepack = True
    renormalize = False

    def run_all():
        total_infer_time = 0.0
        mean_unique_experts = 0.0
        for a, w1_packed, w2_packed, w1s, w2s, w1b, w2b, score in zip(
            a_list,
            w1_list,
            w2_list,
            w1s_list,
            w2s_list,
            w1b_list,
            w2b_list,
            score_list,
        ):
            _, moe_time, unique_experts = fused_moe(
                a,
                w1_packed,
                w2_packed,
                w1s,
                w2s,
                w1b,
                w2b,
                score,
                topk,
                renormalize,
                prepack,
            )
            total_infer_time += moe_time
            mean_unique_experts += unique_experts

        mean_unique_experts /= len(w1_list)
        return total_infer_time, mean_unique_experts

    # warmup
    for _ in range(3):
        run_all()

    iters = 10
    Total_time = 0.0
    mean_unique_experts = 0.0
    for _ in range(iters):
        iter_time, iter_mean_unique_experts = run_all()
        Total_time += iter_time
        mean_unique_experts += iter_mean_unique_experts
    mean_unique_experts /= iters

    print(f"mean_unique_experts={mean_unique_experts:.2f} / {num_experts}")

    ms = Total_time / iters / num_layers * 1e3

    tflops = 6 * M * topk * N * K / (ms * 1e-3) / 1e12

    # Calculate bandwidth (only count the unique experts actually touched).
    frac = mean_unique_experts / num_experts
    a_bytes = a_list[0].numel() * a_list[0].element_size()
    w1_bytes = w1_list[0].numel() * w1_list[0].element_size() * frac
    w2_bytes = w2_list[0].numel() * w2_list[0].element_size() * frac
    w1s_bytes = w1s_list[0].numel() * w1s_list[0].element_size() * frac
    w2s_bytes = w2s_list[0].numel() * w2s_list[0].element_size() * frac
    w1b_bytes = w1b_list[0].numel() * w1b_list[0].element_size() * frac
    w2b_bytes = w2b_list[0].numel() * w2b_list[0].element_size() * frac
    total_bytes = (
        a_bytes + w1_bytes + w2_bytes + w1s_bytes + w2s_bytes + w1b_bytes + w2b_bytes
    )
    gbps = total_bytes / (ms * 1e-3) / 1e9  # bytes/sec to GB/sec

    print(f"{ms:.3f} ms/iter | {tflops:.2f} TFLOPS | {gbps:.2f} GB/s")
    return ms, tflops, gbps


if __name__ == "__main__":
    assert len(sys.argv) > 5, f"usage: {sys.argv[0]} M N K E topk"
    M, N, K, E, topk = (
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
    )

    bench_mxfp4_moe(M, N, K, num_experts=E, topk=topk)
