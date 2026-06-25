#!/usr/bin/env python3
"""Plot TFLOPS and Bandwidth vs M from gemm_benchmark_results.csv.

Reads references/gemm_benchmark_results.csv and produces, for each TP value,
two line plots (TFLOPS and Bandwidth) where each curve is one GEMM layer.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "references")
CSV = os.path.join(REF, "gemm_benchmark_results.csv")


def load(path):
    # data[(tp, layer)] -> list of (M, tflops, gbps), N, K, precision
    data = defaultdict(list)
    meta = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tp = int(row["tp"])
            layer = row["layer"]
            key = (tp, layer)
            data[key].append(
                (int(row["M"]), float(row["tflops"]),
                 float(row["bandwidth_gbps"]))
            )
            meta[key] = (int(row["N"]), int(row["K"]), row["precision"])
    for key in data:
        data[key].sort(key=lambda r: r[0])
    return data, meta


def plot_metric(data, meta, tp, metric_idx, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for (t, layer), rows in sorted(data.items()):
        if t != tp:
            continue
        N, K, prec = meta[(t, layer)]
        ms = [r[0] for r in rows]
        ys = [r[metric_idx] for r in rows]
        ax.plot(ms, ys, marker="o", label=f"{layer} (N={N},K={K},{prec})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (batch / num tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen3-30B-A3B",
        help="Model name used in plot titles",
    )
    args = parser.parse_args()

    data, meta = load(CSV)
    tps = sorted({k[0] for k in data})
    for tp in tps:
        precs = sorted({m[2] for k, m in meta.items() if k[0] == tp})
        prec = "MIXED" if len(precs) > 1 else precs[0].upper()
        plot_metric(
            data, meta, tp, 1, "TFLOPS",
            f"{args.model}-{prec} GEMM TFLOPS (TP={tp})",
            os.path.join(REF, f"tflops_tp{tp}.png"),
        )
        plot_metric(
            data, meta, tp, 2, "Bandwidth (GB/s)",
            f"{args.model}-{prec} GEMM Bandwidth (TP={tp})",
            os.path.join(REF, f"bandwidth_tp{tp}.png"),
        )


if __name__ == "__main__":
    main()
