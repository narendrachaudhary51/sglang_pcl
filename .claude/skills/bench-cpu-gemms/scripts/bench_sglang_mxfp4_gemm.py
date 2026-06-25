import torch, time
import sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops

mxfp4_scaled_mm_cpu = torch.ops.sgl_kernel.mxfp4_scaled_mm_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

# mxfp4 weight is 4-bit (E2M1), packed 2 values per uint8 byte along K.
# scales are E8M0 (uint8) shared over groups of 32 along K.
GROUP_SIZE = 32

def bench_mxfp4(M, N, K):

    # Build enough distinct (a, b, scales) sets so the combined working set exceeds cache.
    # Size a single layer's footprint, then pick num_layers to blow past a large LLC.
    CACHE_BYTES = 5 * 1024 * 1024 * 1024  # assume 5 GB working set

    bytes_per_layer = (
        M * K * 2                      # a: bf16
        + N * (K // 2)                 # b_packed: uint8
        + N * (K // GROUP_SIZE)        # scales: uint8
    )
    num_layers = max(2, (CACHE_BYTES // bytes_per_layer) + 1)
    print(f"num_layers={num_layers} | working set ~{num_layers*bytes_per_layer/1e6:.1f} MB")

    a_list = []
    b_list = []
    scales_list = []
    for _ in range(num_layers):
        # activation stays bf16
        a = torch.randn(M, K, dtype=torch.bfloat16)

        # weight: [N, K/2] uint8, each byte holds two FP4 nibbles (any nibble is a valid code)
        b_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        b_packed = convert_weight_packed(b_fp4)  # pack to VNNI so packing isn't timed

        # scales: E8M0 exponents, [N, K/32] uint8 (numel == N*K/32). ~127 => scale ~1.0
        scales = torch.randint(126, 130, (N, K // GROUP_SIZE), dtype=torch.uint8)

        a_list.append(a)
        b_list.append(b_packed)
        scales_list.append(scales)

    def run_all():
        for a, b_packed, scales in zip(a_list, b_list, scales_list):
            mxfp4_scaled_mm_cpu(a, b_packed, scales, None, True)


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
    a_bytes = a.numel() * a.element_size()
    b_bytes = b_packed.numel() * b_packed.element_size()
    scales_bytes = scales.numel() * scales.element_size()

    total_bytes = a_bytes + b_bytes + scales_bytes
    gbps = total_bytes / (ms * 1e-3) / 1e9  # bytes/sec to GB/sec

    print(f"{ms:.3f} ms/iter | {tflops:.2f} TFLOPS | {gbps:.2f} GB/s")
    return ms, tflops, gbps

if __name__ == "__main__":
    assert len(sys.argv) > 3, f"usage: {sys.argv[0]} M N K"
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    assert K % GROUP_SIZE == 0, "K must be a multiple of the mxfp4 group size (32)"
    
    bench_mxfp4(M, N, K)