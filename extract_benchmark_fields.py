#!/usr/bin/env python3
"""Extract selected metrics from SGLang serving benchmark log file(s) and emit CSV.

Usage:
    python extract_benchmark_fields.py LOG [LOG ...] [-o results.csv]
    python extract_benchmark_fields.py 'gpt-oss-120b-GNR-TP1/sglang_client_*.log' -o out.csv
"""
import argparse
import csv
import glob
import re
import sys

# (column_name, label as it appears in the log between the ==== markers)
FIELDS = [
    # ("requests",         "Successful requests"),
    # ("duration_s",       "Benchmark duration (s)"),
    ("in_tokens",        "Total input tokens"),
    ("out_tokens",       "Total generated tokens"),
    # ("req_tput",         "Request throughput (req/s)"),
    ("in_tput",          "Input token throughput (tok/s)"),
    ("out_tput",         "Output token throughput (tok/s)"),
    ("peak_out_tput",    "Peak output token throughput (tok/s)"),
    ("total_tput",       "Total token throughput (tok/s)"),
    # ("concurrency",      "Concurrency"),
    # ("mean_e2e_ms",      "Mean E2E Latency (ms)"),
    # ("median_e2e_ms",    "Median E2E Latency (ms)"),
    ("mean_ttft_ms",     "Mean TTFT (ms)"),
    ("median_ttft_ms",   "Median TTFT (ms)"),
    ("p99_ttft_ms",      "P99 TTFT (ms)"),
    ("mean_tpot_ms",     "Mean TPOT (ms)"),
    ("median_tpot_ms",   "Median TPOT (ms)"),
    ("p99_tpot_ms",      "P99 TPOT (ms)"),
    # ("mean_itl_ms",      "Mean ITL (ms)"),
    # ("median_itl_ms",    "Median ITL (ms)"),
    # ("p95_itl_ms",       "P95 ITL (ms)"),
    # ("p99_itl_ms",       "P99 ITL (ms)"),
    # ("max_itl_ms",       "Max ITL (ms)"),
]

START_MARKER = "Serving Benchmark Result"
END_MARKER = "=================================================="


def parse_log(path):
    """Return a dict of metric values from the final benchmark result block, or None."""
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except OSError as e:
        print(f"# skip {path}: {e}", file=sys.stderr)
        return None

    idx = text.rfind(START_MARKER)
    if idx == -1:
        return None
    # Bound the search to the result block to avoid spurious matches elsewhere.
    end = text.find(END_MARKER, idx + len(START_MARKER))
    block = text[idx:end] if end != -1 else text[idx:]

    values = {}
    for key, label in FIELDS:
        # Match: "<label>:<whitespace><value>" where value is the first token.
        m = re.search(rf"^{re.escape(label)}:\s*(\S+)", block, re.MULTILINE)
        values[key] = m.group(1) if m else ""
    return values


def expand_paths(patterns):
    files = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        if matched:
            files.extend(matched)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="Log file path(s) or glob pattern(s).")
    ap.add_argument("-o", "--output", default="-",
                    help="Output CSV file (default: stdout).")
    ap.add_argument("--no-file", action="store_true",
                    help="Omit the source file column.")
    args = ap.parse_args()

    files = expand_paths(args.paths)
    header = ([] if args.no_file else ["file"]) + [k for k, _ in FIELDS]

    out_fh = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    try:
        writer = csv.writer(out_fh)
        writer.writerow(header)
        for path in files:
            values = parse_log(path)
            if values is None:
                print(f"# no benchmark result found in {path}", file=sys.stderr)
                continue
            row = ([] if args.no_file else [path]) + [values[k] for k, _ in FIELDS]
            writer.writerow(row)
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == "__main__":
    main()
