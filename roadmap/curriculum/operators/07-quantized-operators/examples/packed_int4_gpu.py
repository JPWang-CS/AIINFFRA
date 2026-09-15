"""Triton teaching baseline for packed signed-INT4 linear layers.

The contract is deliberately narrow and visible:

* X[T, I] is contiguous CUDA FP16;
* packed[O, ceil(I / 2)] is CUDA uint8, with the low input nibble first;
* scales[ceil(I / G), O] is positive CUDA FP32;
* C[T, O] is CUDA FP16, with FP32 accumulation and alpha=1, beta=0.

This is a baseline for learning packed addressing and mixed-precision
semantics.  It is not an optimized kernel and it does not claim native INT4
MMA usage.  Run this file directly: without Torch, Triton, or CUDA it prints
SKIP and exits with status 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:  # Optional on purpose: the CPU repository remains runnable without it.
    import torch
except ImportError:  # pragma: no cover - depends on the host environment.
    torch = None  # type: ignore[assignment]

try:  # Optional on purpose: --help and SKIP must work without Triton.
    import triton
    import triton.language as tl
    import triton.testing
except ImportError:  # pragma: no cover - depends on the host environment.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from quantization_lab import (  # noqa: E402  (local teaching-module import)
    dequantize_int4_group,
    pack_signed_int4,
    quantize_int4_group,
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _check_group_size(group_size: int) -> int:
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)):
        raise ValueError("group_size must be a positive integer")
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    return int(group_size)


if triton is not None:

    @triton.jit
    def plain_kernel(
        x_ptr,
        packed_ptr,
        scales_ptr,
        c_ptr,
        t_size,
        i_size,
        o_size,
        group_size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One output tile, direct packed-byte addressing, no dequant buffer."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # Keep address arithmetic wide: a large flattened packed/scales
        # tensor must not overflow a 32-bit intermediate before pointer add.
        rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        cols = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        row_mask = rows < t_size
        col_mask = cols < o_size
        packed_bytes = (i_size + 1) // 2
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # The fixed reduction tile is 32.  Looping over K makes both I tails
        # and arbitrary I/G boundaries explicit in the same baseline.
        for k_block in range(0, tl.cdiv(i_size, BLOCK_K)):
            inputs = (k_block * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
            input_mask = inputs < i_size
            x_mask = row_mask[:, None] & input_mask[None, :]
            x = tl.load(
                x_ptr + rows[:, None] * i_size + inputs[None, :],
                mask=x_mask,
                other=0.0,
            )

            byte_index = inputs // 2
            packed_mask = col_mask[:, None] & input_mask[None, :]
            packed_byte = tl.load(
                packed_ptr + cols[:, None] * packed_bytes + byte_index[None, :],
                mask=packed_mask,
                other=0,
            )
            shift = ((inputs % 2) * 4).to(tl.int32)
            nibble = (packed_byte >> shift[None, :]) & 0xF
            nibble_i32 = nibble.to(tl.int32)
            signed_q = tl.where(nibble_i32 >= 8, nibble_i32 - 16, nibble_i32)

            scale_index = (inputs // group_size) * o_size + cols[:, None]
            scale = tl.load(scales_ptr + scale_index, mask=packed_mask, other=0.0)
            # Match the reference exactly: q*FP32 scale, cast to FP16, then
            # FP16 x FP16 dot with FP32 accumulation and final FP16 store.
            w_fp16 = (signed_q.to(tl.float32) * scale).to(tl.float16)
            acc += tl.dot(x, tl.trans(w_fp16), out_dtype=tl.float32)

        tl.store(c_ptr + rows[:, None] * o_size + cols[None, :], acc.to(tl.float16), mask=row_mask[:, None] & col_mask[None, :])

else:
    plain_kernel = None


def _require_gpu_stack() -> None:
    if torch is None:
        raise RuntimeError("SKIP: torch is unavailable")
    if triton is None or plain_kernel is None:
        raise RuntimeError("SKIP: triton is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("SKIP: CUDA is unavailable")


def _check_cuda_tensor(name: str, value, dtype, rank: int) -> None:
    if torch is None:  # pragma: no cover - guarded by _require_gpu_stack.
        raise RuntimeError("SKIP: torch is unavailable")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_contract(x, packed, scales, group_size: int) -> tuple[int, int, int, int]:
    _require_gpu_stack()
    group_size = _check_group_size(group_size)
    _check_cuda_tensor("X", x, torch.float16, 2)
    _check_cuda_tensor("packed", packed, torch.uint8, 2)
    _check_cuda_tensor("scales", scales, torch.float32, 2)
    t_size, i_size = map(int, x.shape)
    o_size = int(packed.shape[0])
    if t_size <= 0 or i_size <= 0 or o_size <= 0:
        raise ValueError("X and packed must have non-zero dimensions")
    if tuple(packed.shape) != (o_size, _ceil_div(i_size, 2)):
        raise ValueError("packed shape must be [O, ceil(I/2)]")
    expected_scales = (_ceil_div(i_size, group_size), o_size)
    if tuple(scales.shape) != expected_scales:
        raise ValueError(f"scales shape must be {expected_scales}")
    if packed.device != x.device or scales.device != x.device:
        raise ValueError("X, packed, and scales must be on the same CUDA device")
    # This is a contract check rather than a performance path; it catches bad
    # test fixtures before launching a kernel with invalid scale semantics.
    if not bool(torch.isfinite(scales).all().item()) or not bool((scales > 0).all().item()):
        raise ValueError("scales must be finite and positive")
    return t_size, i_size, o_size, group_size


def solve(x, packed, scales, group_size: int):
    """Launch the FP16-input/FP32-accumulator packed INT4 baseline."""
    t_size, i_size, o_size, group_size = _check_contract(x, packed, scales, group_size)
    return _solve_unchecked(x, packed, scales, group_size, t_size, i_size, o_size)


def _solve_unchecked(x, packed, scales, group_size: int, t_size: int, i_size: int, o_size: int):
    """Launch body after one-time wrapper validation (used by timed paths)."""
    output = torch.empty((t_size, o_size), device=x.device, dtype=torch.float16)
    grid = (_ceil_div(t_size, 16), _ceil_div(o_size, 32))
    # A tensor can live on a non-current CUDA device.  The launch context must
    # follow x.device rather than relying on process-global current_device.
    with torch.cuda.device(x.device):
        plain_kernel[grid](
            x,
            packed,
            scales,
            output,
            t_size,
            i_size,
            o_size,
            group_size,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
    return output


def _dequantize_torch_unchecked(packed, scales, input_size: int, group_size: int):
    """Timed-path body: assumes one-time shape/dtype/device/scale prechecks."""
    byte_indices = torch.arange(input_size, device=packed.device, dtype=torch.long) // 2
    shifts = (torch.arange(input_size, device=packed.device, dtype=torch.long) % 2) * 4
    byte_values = packed[:, byte_indices].to(torch.int16)
    nibble = (byte_values >> shifts[None, :]) & 0xF
    q_oi = torch.where(nibble >= 8, nibble - 16, nibble)
    q_io = q_oi.transpose(0, 1)
    scales_io = scales.repeat_interleave(group_size, dim=0)[:input_size, :]
    return (q_io.to(torch.float32) * scales_io).to(torch.float16)


def dequantize_torch(packed, scales, input_size: int, group_size: int):
    """Checked unpack to [I,O], with q*FP32-scale -> FP16 W semantics."""
    _require_gpu_stack()
    group_size = _check_group_size(group_size)
    _check_cuda_tensor("packed", packed, torch.uint8, 2)
    _check_cuda_tensor("scales", scales, torch.float32, 2)
    if isinstance(input_size, bool) or not isinstance(input_size, (int, np.integer)) or input_size <= 0:
        raise ValueError("input_size must be a positive integer")
    input_size = int(input_size)
    output_size = int(packed.shape[0])
    if tuple(packed.shape) != (output_size, _ceil_div(input_size, 2)):
        raise ValueError("packed shape does not match input_size")
    if tuple(scales.shape) != (_ceil_div(input_size, group_size), output_size):
        raise ValueError("scales shape does not match packed/input dimensions")
    if packed.device != scales.device:
        raise ValueError("packed and scales must be on the same CUDA device")
    if not bool(torch.isfinite(scales).all().item()) or not bool((scales > 0).all().item()):
        raise ValueError("scales must be finite and positive")
    return _dequantize_torch_unchecked(packed, scales, input_size, group_size)


def _reference_from_numpy(x_np: np.ndarray, q_np: np.ndarray, scales_np: np.ndarray, group_size: int):
    """Independent reference: NumPy q*float32 scale -> float16 W -> FP32 mm."""
    if torch is None:
        raise RuntimeError("SKIP: torch is unavailable")
    scales_fp32 = scales_np.astype(np.float32)
    w_np = (q_np.astype(np.float32) * scales_fp32.repeat(group_size, axis=0)[: q_np.shape[0]]).astype(np.float16)
    x_fp32 = torch.from_numpy(np.asarray(x_np, dtype=np.float16)).cuda().float()
    w_fp32 = torch.from_numpy(np.ascontiguousarray(w_np)).cuda().float()
    return torch.mm(x_fp32, w_fp32).to(torch.float16), torch.from_numpy(w_np).cuda()


def _make_case(t_size: int, i_size: int, o_size: int, group_size: int, seed: int):
    if torch is None:
        raise RuntimeError("SKIP: torch is unavailable")
    rng = np.random.default_rng(seed)
    x_np = np.ascontiguousarray(rng.normal(size=(t_size, i_size)).astype(np.float16))
    w_np = rng.normal(size=(i_size, o_size)).astype(np.float64)
    q_np, scales_np = quantize_int4_group(w_np, group_size)
    # Exercise storage code -8 even though the symmetric teaching quantizer
    # normally uses [-7,7]. Reference and all GPU paths consume the same q.
    q_np[0, 0] = -8
    if q_np.size > 1:
        q_np[-1, -1] = 7
    packed_np = pack_signed_int4(q_np)
    x = torch.from_numpy(x_np).cuda()
    packed = torch.from_numpy(np.ascontiguousarray(packed_np)).cuda()
    scales = torch.from_numpy(np.ascontiguousarray(scales_np.astype(np.float32))).cuda()
    reference, w_fp16 = _reference_from_numpy(x_np, q_np, scales_np, group_size)
    return x, packed, scales, reference, w_fp16


def _assert_close(name: str, actual, expected) -> None:
    if torch is None:
        raise RuntimeError("SKIP: torch is unavailable")
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError(f"{name} produced non-finite output")
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2, msg=name)


def run_correctness() -> None:
    """Run all required shapes against the independent FP32 reference."""
    print("correctness precision: X FP16 contiguous, q packed signed-nibble uint8, scale FP32, C FP16 (FP32 accumulator)")
    shapes = ((1, 1, 1, 1), (3, 5, 7, 3), (17, 65, 33, 16), (32, 128, 64, 32))
    for case_id, (t_size, i_size, o_size, group_size) in enumerate(shapes):
        x, packed, scales, reference, w_fp16 = _make_case(
            t_size, i_size, o_size, group_size, seed=20260911 + case_id
        )
        fused = solve(x, packed, scales, group_size)
        materialized = torch.mm(x, dequantize_torch(packed, scales, i_size, group_size))
        predequant = torch.mm(x, w_fp16)
        _assert_close(f"fused shape {(t_size, i_size, o_size, group_size)}", fused, reference)
        _assert_close("torch unpack + mm", materialized, reference)
        _assert_close("predequantized torch mm", predequant, reference)
        print(
            f"PASS shape T={t_size} I={i_size} O={o_size} G={group_size} "
            f"max_abs={float((fused - reference).abs().max().item()):.6g}"
        )


def _benchmark_case() -> None:
    """Benchmark T=1 and T=128 only after all three paths pass each shape."""
    if torch is None or triton is None:
        raise RuntimeError("SKIP: torch/triton is unavailable")
    print("benchmark precision: X FP16, q uint8 signed nibble, scales FP32, W/C FP16, FP32 accumulation")
    print("benchmark scopes: torch unpack+materialize+mm includes unpack; pre-dequantized mm excludes unpack and is reference-only")
    for case_id, t_size in enumerate((1, 128)):
        i_size, o_size, group_size = 1024, 768, 128
        x, packed, scales, reference, w_fp16 = _make_case(
            t_size, i_size, o_size, group_size, 20261001 + case_id
        )
        # Validate once before timing.  The three correctness outputs use the
        # same benchmark tensors and must all align with the independent ref.
        checked = _check_contract(x, packed, scales, group_size)
        fused = _solve_unchecked(x, packed, scales, group_size, *checked[:3])
        materialized_w = _dequantize_torch_unchecked(packed, scales, i_size, group_size)
        materialized = torch.mm(x, materialized_w)
        predequant = torch.mm(x, w_fp16)
        _assert_close("benchmark fused", fused, reference)
        _assert_close("benchmark torch unpack + mm", materialized, reference)
        _assert_close("benchmark pre-dequantized mm", predequant, reference)

        # Timed callables deliberately skip all public-input validation and
        # .item()-based scale checks.  do_bench still includes output
        # allocation for all three paths.
        fused_ms = float(
            triton.testing.do_bench(
                lambda: _solve_unchecked(x, packed, scales, group_size, *checked[:3]),
                warmup=50, rep=200,
            )
        )
        materialized_ms = float(
            triton.testing.do_bench(
                lambda: torch.mm(
                    x, _dequantize_torch_unchecked(packed, scales, i_size, group_size)
                ), warmup=50, rep=200,
            )
        )
        predequant_ms = float(triton.testing.do_bench(lambda: torch.mm(x, w_fp16), warmup=50, rep=200))
        print(f"benchmark shape T={t_size} I={i_size} O={o_size} G={group_size}")
        print(f"  packed fused baseline: {fused_ms:.6f} ms")
        print(f"  torch unpack + FP16 W + mm: {materialized_ms:.6f} ms")
        print(f"  pre-dequantized FP16 W + mm (unpack excluded; reference only): {predequant_ms:.6f} ms")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="run the correctness-gated benchmark shape")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        _require_gpu_stack()
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"GPU cuda:{device}: {props.name}, CC={props.major}.{props.minor}")
        print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, Triton={triton.__version__}")
        print("tile=(16,32,32), num_warps=4, num_stages=2; do_bench warmup=50ms rep=200ms")
        matmul_backend = torch.backends.cuda.matmul
        precision_attr = "fp32_precision" if hasattr(matmul_backend, "fp32_precision") else "allow_tf32"
        old_precision = getattr(matmul_backend, precision_attr)
        old_fp16_reduction = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        try:
            setattr(matmul_backend, precision_attr, "ieee" if precision_attr == "fp32_precision" else False)
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            run_correctness()
            if args.benchmark:
                _benchmark_case()
        finally:
            setattr(matmul_backend, precision_attr, old_precision)
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_fp16_reduction
    except RuntimeError as exc:
        if str(exc).startswith("SKIP:"):
            print(str(exc))
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
