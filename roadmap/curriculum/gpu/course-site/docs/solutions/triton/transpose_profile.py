"""External Nsight Systems workload for the archived Triton V2 transpose."""

import argparse
import sys


def positive_int(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, default=4096)
    parser.add_argument("--cols", type=positive_int, default=4096)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=100)
    parser.add_argument("--impl", choices=("triton", "torch", "both"), default="both")
    parser.add_argument("--capture-profiler-api", action="store_true")
    return parser.parse_args(argv)


def run_capture(torch, operations, repeats, capture_profiler_api=False):
    """Run labeled workloads; optionally bracket them with CUDA Profiler API."""
    torch.cuda.synchronize()
    if capture_profiler_api:
        torch.cuda.profiler.start()
    try:
        for label, operation in operations:
            torch.cuda.nvtx.range_push(label)
            try:
                for _ in range(repeats):
                    operation()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
    finally:
        if capture_profiler_api:
            torch.cuda.profiler.stop()


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
        print(f"ERROR: PyTorch/Triton import failed: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable.", file=sys.stderr)
        return 2

    try:
        device = torch.device("cuda")
        torch.manual_seed(0)
        x = torch.rand((args.rows, args.cols), device=device, dtype=torch.float32)
        out = torch.empty((args.cols, args.rows), device=device, dtype=torch.float32)
        operations = []
        if args.impl in ("triton", "both"):
            operations.append(("transpose/triton", lambda: solve(x, out, args.rows, args.cols)))
        if args.impl in ("torch", "both"):
            operations.append(("transpose/pytorch", lambda: out.copy_(x.T)))

        # Compile, verify, and warm up every selected path before capture.
        for label, operation in operations:
            out.fill_(float("nan"))
            operation()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, x.T, rtol=0, atol=0)
            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, x.T, rtol=0, atol=0)

        major, minor = torch.cuda.get_device_capability(device)
        print(
            f"metadata input_shape={tuple(x.shape)} output_shape={tuple(out.shape)} "
            f"input_stride={x.stride()} output_stride={out.stride()} dtype={x.dtype} "
            f"device={torch.cuda.get_device_name(device)} capability={major}.{minor} "
            f"torch={torch.__version__} triton={triton.__version__} "
            f"cuda={torch.version.cuda} impl={args.impl} warmup={args.warmup} repeats={args.repeats}",
            flush=True,
        )
        run_capture(torch, operations, args.repeats, args.capture_profiler_api)
        for _, operation in operations:
            operation()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, x.T, rtol=0, atol=0)
        print("PROFILE_TARGET_COMPLETE correctness=PASS", flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: profiling workload failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
