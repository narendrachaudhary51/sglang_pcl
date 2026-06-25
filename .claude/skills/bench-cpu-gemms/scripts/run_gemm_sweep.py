import torch, time, math, os, sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops
from bench_sglang_fp8_gemm import bench_fp8
from bench_sglang_mxfp4_gemm import bench_mxfp4
from bench_sglang_bf16_gemm import bench_bf16

fp8_scaled_mm_cpu = torch.ops.sgl_kernel.fp8_scaled_mm_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "references")
SHAPES = os.path.join(REF, "gemm_shapes.txt")
OUT = os.path.join(REF, "gemm_benchmark_results.csv")

M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
block_size = [128, 128]


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
    bench_fns = {
        "fp8": bench_fp8,
        "mxfp4": bench_mxfp4,
        "bf16": bench_bf16,
    }
    for tp, layer, prec, N, K in shapes:
        bench_fn = bench_fns[prec]
        for M in M_LIST:
            ms, tflops, gbps = bench_fn(M, N, K)
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
