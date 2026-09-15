#!/usr/bin/env python3
"""Compute algorithmic arithmetic intensity and simple Roofline bounds."""

from __future__ import annotations

import argparse


def roofline(operations: float, bytes_moved: float, bandwidth_gbps: float,
             peak_tflops: float) -> tuple[float, float, float]:
    intensity = operations / bytes_moved
    memory_bound_gflops = intensity * bandwidth_gbps
    bound_gflops = min(memory_bound_gflops, peak_tflops * 1000.0)
    return intensity, memory_bound_gflops, bound_gflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth-gbps", type=float, required=True)
    parser.add_argument("--peak-tflops", type=float, required=True)
    args = parser.parse_args()
    if args.bandwidth_gbps <= 0 or args.peak_tflops <= 0:
        parser.error("bandwidth and peak must be positive")

    n = 2**25
    vector_ops = float(n)  # one add per element
    vector_bytes = float(n * (4 + 4 + 4))  # two FP32 reads and one write
    ai, memory_roof, bound = roofline(
        vector_ops, vector_bytes, args.bandwidth_gbps, args.peak_tflops
    )
    print("VectorAdd FP32")
    print(f"  elements={n} operations={vector_ops:.0f} bytes={vector_bytes:.0f}")
    print(f"  AI={ai:.6f} FLOP/byte memory_roof={memory_roof:.3f} GFLOP/s")
    print(f"  roofline_bound={bound:.3f} GFLOP/s")

    m, n, k = 8192, 6144, 4096
    gemm_ops = float(2 * m * n * k)
    fp32_bytes = float(4 * (m * n + n * k + m * k))
    fp16_storage_bytes = float(2 * (m * n + n * k + m * k))
    for label, bytes_moved in (("GEMM FP32 storage", fp32_bytes),
                               ("GEMM FP16 storage", fp16_storage_bytes)):
        ai, memory_roof, bound = roofline(
            gemm_ops, bytes_moved, args.bandwidth_gbps, args.peak_tflops
        )
        print(label)
        print(f"  shape=({m},{n})x({n},{k}) operations={gemm_ops:.0f}")
        print(f"  bytes={bytes_moved:.0f} AI={ai:.3f} FLOP/byte")
        print(f"  memory_roof={memory_roof:.3f} GFLOP/s roofline_bound={bound:.3f} GFLOP/s")

    print("Precision note: IEEE FP32 and TF32 are separate numerical contracts;")
    print("do not combine their timings into one roofline table.")


if __name__ == "__main__":
    main()
