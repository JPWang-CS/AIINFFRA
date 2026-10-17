"""Correctness-only validation for the two archived Reduction snapshots.

The CPU mode is an independent indexing/grouping model only; it does not
exercise Triton compilation, the compiler's tl.sum lowering, or GPU execution.
"""
from __future__ import annotations

import argparse
import math
import random
import sys


CASES = (1, 31, 255, 256, 257, 1023, 1024, 1025, 4097)
MULTI_LARGE_N = 1_048_576
PATTERNS = ("zeros", "ones", "alternating", "random")
ATOL = 2e-4
RTOL = 2e-5


def _values(n: int, pattern: str) -> list[float]:
    if pattern == "zeros":
        return [0.0] * n
    if pattern == "ones":
        return [1.0] * n
    if pattern == "alternating":
        return [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    rng = random.Random(20261017 + n)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _simulate(values: list[float], block_size: int, two_stage: bool) -> float:
    n = len(values)
    grid = (n + block_size - 1) // block_size
    partials = []
    for pid in range(grid):
        start = pid * block_size
        # Masked lanes contribute the same additive identity as tl.load(other=0).
        tile = values[start : min(start + block_size, n)]
        partials.append(math.fsum(tile))
    if not two_stage:
        return math.fsum(partials)
    second_tile = 1 << (len(partials) - 1).bit_length()
    return math.fsum(partials + [0.0] * (second_tile - len(partials)))


def cpu_check() -> int:
    checked = 0
    for n in CASES + (MULTI_LARGE_N,):
        patterns = PATTERNS
        for pattern in patterns:
            values = _values(n, pattern)
            reference = math.fsum(values)
            if n <= 4097:
                got = _simulate(values, 1 << (n - 1).bit_length(), False)
                assert math.isclose(got, reference, rel_tol=RTOL, abs_tol=ATOL), (n, pattern, "single")
                checked += 1
            block_size = 256
            got = _simulate(values, block_size, True)
            assert math.isclose(got, reference, rel_tol=RTOL, abs_tol=ATOL), (n, pattern, "multi")
            checked += 1
    print(f"CPU index/grouping simulation passed: {checked} cases; no GPU execution performed")
    return 0


def gpu_check() -> int:
    try:
        import torch
        import triton
        from reduction_single_block import solve as solve_single
        from reduction_multi_block import solve as solve_multi
    except Exception as exc:
        print(f"GPU validation unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("GPU validation unavailable: CUDA device is not available", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(
        f"GPU={properties.name}, UUID={getattr(properties, 'uuid', 'not exposed by this PyTorch build')}, "
        f"CC={properties.major}.{properties.minor}, CUDA runtime={torch.version.cuda}, "
        f"torch={torch.__version__}, triton={triton.__version__}"
    )
    jobs = [("single", solve_single, n) for n in CASES]
    jobs += [("multi", solve_multi, n) for n in CASES + (MULTI_LARGE_N,)]
    count = 0
    maxima: dict[tuple[str, int], float] = {}
    for label, solve, n in jobs:
        for pattern in PATTERNS:
            host = torch.tensor(_values(n, pattern), dtype=torch.float32)
            x = host.to(device).contiguous()
            output = torch.empty((1,), device=device, dtype=torch.float32)
            solve(x, output, n)
            expected = torch.sum(host.to(torch.float64), dtype=torch.float64).item()
            actual = output.item()
            error = abs(actual - expected)
            bound = ATOL + RTOL * abs(expected)
            exact_integer_pattern = pattern in ("zeros", "ones", "alternating")
            passed = math.isfinite(actual) and (
                actual == expected if exact_integer_pattern else error <= bound
            )
            print(
                f"{label} N={n} pattern={pattern} {'PASS' if passed else 'FAIL'} "
                f"actual={actual:.17g} fp64_reference={expected:.17g} abs_error={error:.9g}"
            )
            key = (label, n)
            maxima[key] = max(maxima.get(key, 0.0), error)
            if not passed:
                raise AssertionError(
                    f"{label} N={n} pattern={pattern}: actual={actual}, "
                    f"fp64_reference={expected}, abs_error={error}, "
                    f"exact_integer_pattern={exact_integer_pattern}, tolerance_bound={bound}"
                )
            count += 1
    torch.cuda.synchronize(device)
    for (label, n), error in sorted(maxima.items()):
        print(f"SUMMARY {label} N={n} patterns={len(PATTERNS)} max_abs_error={error:.9g}")
    print(f"GPU correctness passed: {count} cases; random atol={ATOL}, rtol={RTOL}; no benchmark performed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-check", action="store_true", help="stdlib-only indexing/grouping simulation")
    mode.add_argument("--correctness-only", action="store_true", help="run GPU correctness; never benchmark")
    args = parser.parse_args()
    if args.cpu_check:
        return cpu_check()
    return gpu_check()


if __name__ == "__main__":
    raise SystemExit(main())
