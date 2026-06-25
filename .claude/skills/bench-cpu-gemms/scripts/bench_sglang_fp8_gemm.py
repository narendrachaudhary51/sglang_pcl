import torch, time
import sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops
import math

fp8_scaled_mm_cpu = torch.ops.sgl_kernel.fp8_scaled_mm_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

def bench_fp8(M, N, K):
    block_size = [128, 128]

    # Build enough distinct (a, b, scales) sets so the combined working set exceeds cache.
    # Size a single layer's footprint, then pick num_layers to blow past a large LLC.
    CACHE_BYTES = 5 * 1024 * 1024 * 1024  # assume 5 GB working set

    bytes_per_layer = (
        M * K * 2                                          # a: bf16
        + N * K                                            # b_packed: fp8 (1 byte)
        + math.ceil(N / block_size[0]) * math.ceil(K / block_size[1]) * 4  # scales: fp32
    )
    num_layers = max(2, (CACHE_BYTES // bytes_per_layer) + 1)
    print(f"num_layers={num_layers} | working set ~{num_layers*bytes_per_layer/1e6:.1f} MB")

    a_list = []
    b_list = []
    scales_list = []
    for _ in range(num_layers):
        a = torch.randn(M, K, dtype=torch.bfloat16)
        b_fp8 = torch.randn(N, K).to(torch.float8_e4m3fn)
        b_packed = convert_weight_packed(b_fp8)

        scales = torch.randn(math.ceil(N / block_size[0]),
                            math.ceil(K / block_size[1]),
                            dtype=torch.float32)

        a_list.append(a)
        b_list.append(b_packed)
        scales_list.append(scales)

    def run_all():
        for a, b_packed, scales in zip(a_list, b_list, scales_list):
            fp8_scaled_mm_cpu(a, b_packed, scales, block_size, None, torch.bfloat16, True)

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
    b_bytes = b_list[0].numel() * b_list[0].element_size()
    scales_bytes = scales_list[0].numel() * scales_list[0].element_size()
    total_bytes = a_bytes + b_bytes + scales_bytes
    gbps = total_bytes / (ms * 1e-3) / 1e9  # bytes/sec to GB/sec
    
    print(f"{ms:.3f} ms/iter | {tflops:.2f} TFLOPS | {gbps:.2f} GB/s")
    return ms, tflops, gbps

if __name__ == "__main__":
    assert len(sys.argv) > 3, f"usage: {sys.argv[0]} M N K"
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    bench_fp8(M, N, K)