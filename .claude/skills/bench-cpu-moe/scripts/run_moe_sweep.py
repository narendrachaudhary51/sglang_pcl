import torch, time, math, os, sys
import sgl_kernel  # registers torch.ops.sgl_kernel.* CPU ops
from bench_sglang_bf16_moe import bench_bf16_moe


HERE = os.path.dirname(os.path.abspath(__file__))

def main(M_LIST, model_name):
    REF = os.path.join(HERE, "..", "references", model_name)
    SHAPES = os.path.join(REF, "moe_shapes.txt")
    OUT = os.path.join(REF, "moe_benchmark_results.csv")

    shapes = []
    with open(SHAPES) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tp, layer, prec, N, K, E, topk = line.split()
            shapes.append((int(tp), layer, prec, int(N), int(K), int(E), int(topk)))

    rows = []
    print(f"{'tp':>2} {'layer':<14} {'prec':<4} {'M':>5} {'N':>6} {'K':>6} "
          f"{'ms':>9} {'TFLOPS':>8} {'GB/s':>9}")
    bench_fns = {
        # "fp8": bench_fp8,
        # "mxfp4": bench_mxfp4,
        "bf16": bench_bf16_moe,
    }
    for tp, layer, prec, N, K, E, topk in shapes:
        bench_fn = bench_fns[prec]
        for M in M_LIST:
            ms, tflops, gbps = bench_fn(M, N, K, num_experts=E, topk=topk)
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
    M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    model_name = sys.argv[1]
    # Take M_LIST from command line if provided
    if len(sys.argv) > 2:
        M_LIST = [int(x) for x in sys.argv[2:]]
    main(M_LIST, model_name)
