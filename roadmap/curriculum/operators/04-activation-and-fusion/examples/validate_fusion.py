"""CUDA correctness and optional benchmark harness for the SwiGLU example.

Run this later in an environment with PyTorch, Triton, and CUDA.  The default
path checks finite inputs before comparison.  It does not put that scan in the
timed wrapper call.
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass

import torch
import triton

from fused_swiglu import fused_swiglu


@dataclass(frozen=True)
class Case:
    tokens: int
    hidden: int


CASES = (
    Case(1, 1),
    Case(3, 7),
    Case(5, 1025),
    Case(17, 33),
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 2e-5, 2e-5
    if dtype is torch.float16:
        return 2e-3, 1e-5
    if dtype is torch.bfloat16:
        return 1e-2, 1e-4
    raise ValueError(f"unsupported dtype: {dtype}")


def reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # One FP32 elementwise reference, then exactly one final cast.
    gate32 = gate.float()
    up32 = up.float()
    return (gate32 * torch.sigmoid(gate32) * up32).to(gate.dtype)


def assert_finite(*tensors: torch.Tensor) -> None:
    # This is a correctness preflight.  It is deliberately outside timing.
    for tensor in tensors:
        if not bool(torch.isfinite(tensor).all().item()):
            raise AssertionError("input, output, or reference contains a non-finite value")


def check_case(case: Case, dtype: torch.dtype, block: int) -> None:
    torch.manual_seed(case.tokens * 1000 + case.hidden)
    shape = (case.tokens, case.hidden)
    gate = (torch.randn(shape, device="cuda", dtype=dtype) * 1.5).contiguous()
    up = (torch.randn(shape, device="cuda", dtype=dtype) * 1.5).contiguous()
    gate[0, 0] = 0
    if gate.numel() > 2:
        gate.view(-1)[1] = -2.0
        up.view(-1)[2] = 0.75
    assert_finite(gate, up)

    output = torch.empty_like(gate)
    fused_swiglu(gate, up, out=output, block=block)
    torch.cuda.synchronize()
    expected = reference(gate, up)
    assert_finite(output, expected)
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol, equal_nan=False)
    print(f"correctness OK: shape={shape}, dtype={dtype}, numel={gate.numel()}, BLOCK={block}")


def split_swiglu_fp32(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
    activation: torch.Tensor,
) -> None:
    """Reference 2-kernel split path: silu.out, then mul.out."""
    try:
        silu_out = torch.ops.aten.silu.out
    except AttributeError as error:
        raise RuntimeError(
            "this PyTorch build lacks aten.silu.out; refusing to benchmark a "
            "silently different 3-kernel sigmoid+mul+mul baseline"
        ) from error
    silu_out(gate, out=activation)
    torch.mul(activation, up, out=out)


def time_ms(fn, warmup: int = 20, repeats: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def benchmark(tokens: int, hidden: int, block: int) -> None:
    if tokens <= 0 or hidden <= 0:
        raise ValueError("benchmark dimensions must be positive")
    torch.manual_seed(2026)
    dtype = torch.float32
    shape = (tokens, hidden)
    gate = (torch.randn(shape, device="cuda", dtype=dtype) * 1.5).contiguous()
    up = (torch.randn(shape, device="cuda", dtype=dtype) * 1.5).contiguous()
    assert_finite(gate, up)
    fused_out = torch.empty_like(gate)
    split_activation = torch.empty_like(gate)
    split_out = torch.empty_like(gate)

    # Check the actual timing shape before collecting any performance result.
    fused_swiglu(gate, up, out=fused_out, block=block)
    split_swiglu_fp32(gate, up, split_out, split_activation)
    expected = reference(gate, up)
    assert_finite(fused_out, split_out, expected)
    for output in (fused_out, split_out):
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5, equal_nan=False)

    fused_ms = time_ms(lambda: fused_swiglu(gate, up, out=fused_out, block=block))
    split_ms = time_ms(lambda: split_swiglu_fp32(gate, up, split_out, split_activation))
    torch.testing.assert_close(fused_out, split_out, rtol=2e-5, atol=2e-5, equal_nan=False)
    print(
        f"benchmark: device={torch.cuda.get_device_name(0)}, software=torch-{torch.__version__}, "
        f"triton={triton.__version__}, python={platform.python_version()}, "
        f"shape={shape}, numel={gate.numel()}, dtype={dtype}, "
        f"BLOCK={block}, num_warps=4, fused_ms={fused_ms:.6f}, split_2kernel_ms={split_ms:.6f}"
    )
    useful_bytes = 3 * gate.numel() * gate.element_size()
    split_bytes = 5 * gate.numel() * gate.element_size()
    print(f"algorithm_bytes: fused={useful_bytes}, split={split_bytes}; not measured DRAM traffic")
    print(
        f"useful_GB/s (common 3*n*s numerator): fused={useful_bytes / (fused_ms * 1e6):.3f}, "
        f"split={useful_bytes / (split_ms * 1e6):.3f}"
    )
    print("note: CUDA Event covers Python-launched work and launch gap; it is not pure kernel duration.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=4097)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this script")
    print(f"device={torch.cuda.get_device_name(0)}, torch={torch.__version__}, python={sys.version.split()[0]}")
    for dtype in DTYPES:
        for case in CASES:
            check_case(case, dtype, args.block)
    if args.benchmark:
        benchmark(args.tokens, args.hidden, args.block)


if __name__ == "__main__":
    main()
