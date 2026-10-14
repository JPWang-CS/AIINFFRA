"""Correctness and CUDA-event benchmark for the archived Triton V2 transpose."""

import argparse
import math
import sys


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", type=positive_int, default=4096, help="input row count")
    parser.add_argument("--cols", type=positive_int, default=4096, help="input column count")
    parser.add_argument("--warmup", type=positive_int, default=20, help="untimed warmup iterations")
    parser.add_argument("--repeats", type=positive_int, default=100, help="timed iterations")
    parser.add_argument("--correctness-only", action="store_true", help="check outputs without benchmarking")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        import torch
        import triton
        if __package__:
            from .transpose_v2_leetgpu import solve
        else:
            from transpose_v2_leetgpu import solve
    except ImportError as exc:
        print(
            "ERROR: this command requires an environment with PyTorch and Triton installed.",
            file=sys.stderr,
        )
        print(f"Dependency detail: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable; run this command on a CUDA GPU host.", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {major}.{minor}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    torch.manual_seed(0)

    benchmark_shape = (args.rows, args.cols)
    shapes = [(1, 1), (3, 5), (32, 32), (37, 65), (65, 37)]
    if benchmark_shape not in shapes:
        shapes.append(benchmark_shape)

    def make_input(rows, cols):
        return torch.rand((rows, cols), device=device, dtype=torch.float32)

    def check(rows, cols, x, out):
        out.fill_(float("nan"))
        solve(x, out, rows, cols)
        torch.cuda.synchronize()
        if not out.is_contiguous():
            raise RuntimeError(f"output is not contiguous for shape ({rows}, {cols})")
        torch.testing.assert_close(out, x.T, rtol=0, atol=0)

    try:
        for rows, cols in shapes:
            x = make_input(rows, cols)
            out = torch.empty((cols, rows), device=device, dtype=torch.float32)
            check(rows, cols, x, out)
            print(f"correctness PASS: input=({rows},{cols}), output=({cols},{rows}), dtype=float32")

        if args.correctness_only:
            return 0

        rows, cols = benchmark_shape
        x = make_input(rows, cols)
        out = torch.empty((cols, rows), device=device, dtype=torch.float32)
        effective_bytes = 2 * x.numel() * x.element_size()
        print(
            f"benchmark input_shape={tuple(x.shape)}, output_shape={tuple(out.shape)}, "
            f"input_stride={x.stride()}, output_stride={out.stride()}, dtype={x.dtype}, "
            f"warmup={args.warmup}, repeats={args.repeats}, "
            f"effective_bytes={effective_bytes}"
        )

        def measure(label, operation):
            out.fill_(float("nan"))
            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            if not out.is_contiguous():
                raise RuntimeError(f"output is not contiguous after warmup: {label}")
            torch.testing.assert_close(out, x.T, rtol=0, atol=0)

            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                operation()
            stop.record()
            stop.synchronize()

            mean_ms = start.elapsed_time(stop) / args.repeats
            if not math.isfinite(mean_ms) or mean_ms <= 0:
                raise RuntimeError(f"invalid CUDA-event duration for {label}: {mean_ms}")
            torch.testing.assert_close(out, x.T, rtol=0, atol=0)
            effective_gb_per_second = effective_bytes / (mean_ms * 1.0e6)
            print(
                f"{label}: {mean_ms:.6f} ms, "
                f"{effective_gb_per_second:.3f} effective GB/s, "
                f"effective_bytes={effective_bytes}, correctness PASS"
            )

        print("timing_method=event-loop mean; same-buffer hot working set; CPU submission gaps may be included")
        measure("Triton V2", lambda: solve(x, out, rows, cols))
        measure("PyTorch out.copy_(input.T)", lambda: out.copy_(x.T))
    except Exception as exc:
        print(f"ERROR: transpose validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
