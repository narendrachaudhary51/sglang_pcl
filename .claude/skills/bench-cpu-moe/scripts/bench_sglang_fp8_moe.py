import torch, time
import math
import sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops

grouped_topk_cpu = torch.ops.sgl_kernel.grouped_topk_cpu
fused_experts_cpu = torch.ops.sgl_kernel.fused_experts_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

# CPUQuantMethod.FP8_W8A16 (see sglang/srt/layers/amx_utils.py)
FP8_W8A16 = 2
# Block-wise fp8 quantization tiling (matches real model configs, e.g. DeepSeek-V3 / Qwen3)
BLOCK_N, BLOCK_K = 128, 128
factor_for_scale = 1e-3
fp8_max, fp8_min = 400, -400


def fused_moe(a, packed_w1, packed_w2, w1s, w2s, score, topk, renormalize, prepack):

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
        FP8_W8A16,
        w1s,  # w1_scale
        w2s,  # w2_scale
        None,  # w1_zp
        None,  # w2_zp
        [BLOCK_N, BLOCK_K],  # block_size
        None,  # w1 bias
        None,  # w2 bias
        None,  # alpha
        None,  # limit
        prepack,  # is_vnni
    )
    end = time.perf_counter()
    return fused_output, (end - start), unique_experts

def bench_fp8_moe(M, N, K, num_experts, topk=8):

    # Build enough distinct (a, b) sets so the combined working set exceeds cache.
    # Size a single layer's footprint, then pick num_layers to blow past a large LLC.
    CACHE_BYTES = 5 * 1024 * 1024 * 1024  # assume 5 GB working set

    bytes_per_layer = (
        M * K * 2      # a: bf16
        + (M*topk if M*topk < num_experts else num_experts) * (2*N) * K * 1  # up_packed: fp8
        + (M*topk if M*topk < num_experts else num_experts) * K * N * 1    # down_packed: fp8
        + (M*topk if M*topk < num_experts else num_experts) * math.ceil(2*N / BLOCK_N) * math.ceil(K / BLOCK_K) * 4  # w1s: fp32
        + (M*topk if M*topk < num_experts else num_experts) * math.ceil(K / BLOCK_N) * math.ceil(N / BLOCK_K) * 4  # w2s: fp32
    )

    num_layers = max(3, (CACHE_BYTES // bytes_per_layer) + 1)
    print(f"num_layers={num_layers} | working set ~{num_layers*bytes_per_layer/1e6:.1f} MB")

    a_list = []
    w1_list = []
    w2_list = []
    w1s_list = []
    w2s_list = []
    score_list = []
    for _ in range(num_layers):
        a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)

        w1_fp32 = torch.randn(num_experts, 2 * N, K)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(num_experts, K, N)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        # Block-wise scales
        w1s = (
            torch.randn(num_experts, math.ceil(2 * N / BLOCK_N), math.ceil(K / BLOCK_K))
            * factor_for_scale
        )
        w2s = (
            torch.randn(num_experts, math.ceil(K / BLOCK_N), math.ceil(N / BLOCK_K))
            * factor_for_scale
        )

        score = torch.randn((M, num_experts), device="cpu", dtype=torch.bfloat16)

        w1_packed = convert_weight_packed(w1)  # pack to VNNI so packing isn't timed
        w2_packed = convert_weight_packed(w2)

        a_list.append(a)
        w1_list.append(w1_packed)
        w2_list.append(w2_packed)
        w1s_list.append(w1s)
        w2s_list.append(w2s)
        score_list.append(score)

    prepack = True
    renormalize = False

    def run_all():
        total_infer_time = 0.0
        mean_unique_experts = 0.0
        for a, w1_packed, w2_packed, w1s, w2s, score in zip(
            a_list, w1_list, w2_list, w1s_list, w2s_list, score_list
        ):
            _, moe_time, unique_experts = fused_moe(
                a, w1_packed, w2_packed, w1s, w2s, score, topk, renormalize, prepack
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
    # ms = (time.perf_counter() - t0) / iters / num_layers * 1e3

    tflops = 6 * M * topk * N * K / (ms * 1e-3) / 1e12

    # Calculate bandwidth
    a_bytes = a_list[0].numel() * a_list[0].element_size()
    w1_bytes = w1_list[0].numel() * w1_list[0].element_size() * (mean_unique_experts / num_experts)  # only count the unique experts
    w2_bytes = w2_list[0].numel() * w2_list[0].element_size() * (mean_unique_experts / num_experts)
    w1s_bytes = w1s_list[0].numel() * w1s_list[0].element_size() * (mean_unique_experts / num_experts)
    w2s_bytes = w2s_list[0].numel() * w2s_list[0].element_size() * (mean_unique_experts / num_experts)

    total_bytes = a_bytes + w1_bytes + w2_bytes + w1s_bytes + w2s_bytes
    gbps = total_bytes / (ms * 1e-3) / 1e9  # bytes/sec to GB/sec

    print(f"{ms:.3f} ms/iter | {tflops:.2f} TFLOPS | {gbps:.2f} GB/s")
    return ms, tflops, gbps

if __name__ == "__main__":
    assert len(sys.argv) > 5, f"usage: {sys.argv[0]} M N K E topk"
    M, N, K, E, topk = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    bench_fp8_moe(M, N, K, num_experts=E, topk=topk)
