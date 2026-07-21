import torch, time
import sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops

int8_scaled_mm_cpu = torch.ops.sgl_kernel.int8_scaled_mm_cpu
per_token_quant_int8_cpu = torch.ops.sgl_kernel.per_token_quant_int8_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

def bench_int8(M, N, K):

    # Build enough distinct (a, b, scales) sets so the combined working set exceeds cache.
    # Size a single layer's footprint, then pick num_layers to blow past a large LLC.
    CACHE_BYTES = 5 * 1024 * 1024 * 1024  # assume 5 GB working set

    bytes_per_layer = (
        M * K          # a_q: uint8 (1 byte)
        + M * 4        # scales1: fp32 (per-token)
        + N * K        # b_packed: int8 (1 byte)
        + N * 4        # scales2: fp32 (per-channel)
    )
    num_layers = max(2, (CACHE_BYTES // bytes_per_layer) + 1)
    print(f"num_layers={num_layers} | working set ~{num_layers*bytes_per_layer/1e6:.1f} MB")

    a_list = []
    scales1_list = []
    b_list = []
    scales2_list = []
    for _ in range(num_layers):
        a = torch.randn(M, K, dtype=torch.bfloat16)
        a_q, scales1 = per_token_quant_int8_cpu(a)  # quantize activation up front

        b = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        b_packed = convert_weight_packed(b)  # pack to VNNI so packing isn't timed

        scales2 = torch.randn(N, dtype=torch.float32)

        a_list.append(a_q)
        scales1_list.append(scales1)
        b_list.append(b_packed)
        scales2_list.append(scales2)


    def run_all():
        for a_q, b_packed, scales1, scales2 in zip(a_list, b_list, scales1_list, scales2_list):
            int8_scaled_mm_cpu(a_q, b_packed, scales1, scales2, None, torch.bfloat16, True)


    # warmup
    for _ in range(3):
        run_all()

    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        run_all()
    ms = (time.perf_counter() - t0) / iters / num_layers * 1e3

    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12

    # Calculate bandwidth
    a_bytes = a_list[0].numel() * a_list[0].element_size()
    scales1_bytes = scales1_list[0].numel() * scales1_list[0].element_size()
    b_bytes = b_list[0].numel() * b_list[0].element_size()
    scales2_bytes = scales2_list[0].numel() * scales2_list[0].element_size()
    total_bytes = a_bytes + scales1_bytes + b_bytes + scales2_bytes
    gbps = total_bytes / (ms * 1e-3) / 1e9  # bytes/sec to GB/sec

    print(f"{ms:.3f} ms/iter | {tflops:.2f} TFLOPS | {gbps:.2f} GB/s")
    return ms, tflops, gbps

if __name__ == "__main__":
    assert len(sys.argv) > 3, f"usage: {sys.argv[0]} M N K"
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    bench_int8(M, N, K)
