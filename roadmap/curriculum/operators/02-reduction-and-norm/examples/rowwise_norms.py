"""Teaching implementation: one Triton program computes one contiguous row.

This file is a forward-only example. It is intentionally separate from any
platform submission and does not claim GPU execution in this repository.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x, weight, y, rows, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / cols
    inv_rms = tl.rsqrt(mean_square + eps)
    result = values * inv_rms * gamma
    tl.store(y + row * cols + offsets, result, mask=mask)


@triton.jit
def layer_norm_kernel(x, weight, bias, y, rows, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / cols
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / cols
    normalized = centered * tl.rsqrt(variance + eps)
    result = normalized * gamma + beta
    tl.store(y + row * cols + offsets, result, mask=mask)


_INT32_MAX = 2**31 - 1
_ALLOWED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _validate_common(x, weight, bias, eps):
    if x.ndim != 2 or weight.ndim != 1:
        raise ValueError("expected x=[rows, cols] and parameter=[cols]")
    if bias is not None and bias.ndim != 1:
        raise ValueError("expected bias=[cols]")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and parameters must be contiguous")
    if bias is not None and not bias.is_contiguous():
        raise ValueError("bias must be contiguous")
    if not x.is_cuda or not weight.is_cuda or (bias is not None and not bias.is_cuda):
        raise ValueError("this wrapper requires CUDA tensors")
    params = (weight,) if bias is None else (weight, bias)
    if any(parameter.device != x.device for parameter in params):
        raise ValueError("x and parameters must use the same device")
    if x.dtype not in _ALLOWED_DTYPES or any(parameter.dtype != x.dtype for parameter in params):
        raise ValueError("x and parameters must share float16, bfloat16, or float32 dtype")
    rows, cols = map(int, x.shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if rows > _INT32_MAX or cols > _INT32_MAX:
        raise ValueError("rows and cols must fit signed int32 kernel indexing")
    if rows * cols > _INT32_MAX:
        raise ValueError("rows * cols must fit signed int32 pointer indexing")
    if weight.shape[0] != cols or (bias is not None and bias.shape[0] != cols):
        raise ValueError("parameter length must equal x.shape[1]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    return rows, cols


def _block_size(cols: int) -> int:
    if cols <= 0:
        raise ValueError("cols must be positive")
    return triton.next_power_of_2(cols)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    rows, cols = _validate_common(x, weight, None, eps)
    y = torch.empty_like(x)
    block = _block_size(cols)
    rms_norm_kernel[(rows,)](
        x, weight, y, rows, cols, eps, BLOCK=block, num_warps=4
    )
    return y


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    rows, cols = _validate_common(x, weight, bias, eps)
    y = torch.empty_like(x)
    block = _block_size(cols)
    layer_norm_kernel[(rows,)](
        x, weight, bias, y, rows, cols, eps, BLOCK=block, num_warps=4
    )
    return y
