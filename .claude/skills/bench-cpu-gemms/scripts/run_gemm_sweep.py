import torch, time, math, os, sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops

fp8_scaled_mm_cpu = torch.ops.sgl_kernel.fp8_scaled_mm_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "references")
SHAPES = os.path.join(REF, "gemm_shapes.txt")
OUT = os.path.join(REF, "gemm_benchmark_results.csv")

M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
block_size = [128, 128]


def bench_fp8(M, N, K):
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b_fp8 = torch.randn(N, K).to(torch.float8_e4m3fn)
    b_packed = convert_weight_packed(b_fp8)
    scales = torch.randn(math.ceil(N / block_size[0]),
                         math.ceil(K / block_size[1]),
                         dtype=torch.float32)
    fn = lambda: fp8_scaled_mm_cpu(a, b_packed, scales, block_size, None,
                                   torch.bfloat16, True)
    for _ in range(10):
        fn()
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ms = (time.perf_counter() - t0) / iters * 1e3
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    total_bytes = (a.numel() * a.element_size()
                   + b_packed.numel() * b_packed.element_size()
                   + scales.numel() * scales.element_size())
    gbps = total_bytes / (ms * 1e-3) / 1e9
    return ms, tflops, gbps


def main():
    shapes = []
    with open(SHAPES) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tp, layer, prec, N, K = line.split()
            shapes.append((int(tp), layer, prec, int(N), int(K)))

    rows = []
    print(f"{'tp':>2} {'layer':<14} {'prec':<4} {'M':>5} {'N':>6} {'K':>6} "
          f"{'ms':>9} {'TFLOPS':>8} {'GB/s':>9}")
    for tp, layer, prec, N, K in shapes:
        for M in M_LIST:
            ms, tflops, gbps = bench_fp8(M, N, K)
            print(f"{tp:>2} {layer:<14} {prec:<4} {M:>5} {N:>6} {K:>6} "
                  f"{ms:>9.3f} {tflops:>8.2f} {gbps:>9.2f}")
            rows.append((tp, layer, prec, M, N, K, ms, tflops, gbps))

    with open(OUT, "w") as f:
        f.write("tp,layer,precision,M,N,K,ms_per_iter,tflops,bandwidth_gbps\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},"
                    f"{r[6]:.3f},{r[7]:.2f},{r[8]:.2f}\n")
    print(f"\nWrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
