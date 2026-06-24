import torch, time
import sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops

weight_packed_linear = torch.ops.sgl_kernel.weight_packed_linear
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

def bench_bf16(M, N, K):

    # Build enough distinct (a, b) sets so the combined working set exceeds cache.
    # Size a single layer's footprint, then pick num_layers to blow past a large LLC.
    CACHE_BYTES = 5 * 1024 * 1024 * 1024  # assume 5 GB working set

    bytes_per_layer = (
        M * K * 2      # a: bf16
        + N * K * 2    # b_packed: bf16
    )
    num_layers = max(2, (CACHE_BYTES // bytes_per_layer) + 1)
    print(f"num_layers={num_layers} | working set ~{num_layers*bytes_per_layer/1e6:.1f} MB")

    a_list = []
    b_list = []
    for _ in range(num_layers):
        a = torch.randn(M, K, dtype=torch.bfloat16)
        b = torch.randn(N, K, dtype=torch.bfloat16)
        b_packed = convert_weight_packed(b)  # pack to VNNI so packing isn't timed

        a_list.append(a)
        b_list.append(b_packed)


    def run_all():
        for a, b_packed in zip(a_list, b_list):
            weight_packed_linear(a, b_packed, None, True)


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
    total_bytes = a_bytes + b_bytes
    gbps = total_bytes / (ms * 1e-3) / 1e9  # bytes/sec to GB/sec

    print(f"{ms:.3f} ms/iter | {tflops:.2f} TFLOPS | {gbps:.2f} GB/s")
    return ms, tflops, gbps

assert len(sys.argv) > 3, f"usage: {sys.argv[0]} M N K"
M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

if __name__ == "__main__":
    bench_bf16(M, N, K)
