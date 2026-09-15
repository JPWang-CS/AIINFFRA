"""Forward-only row-wise Softmax teaching kernel.

The contract is a contiguous [rows, cols] tensor with finite values. Each
program owns one row, and all reductions use FP32 values and accumulation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_row_kernel(x, y, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(x + row * cols + offsets, mask=mask, other=-float("inf"))
    values = values.to(tl.float32)
    row_max = tl.max(values, axis=0)
    numerators = tl.exp(values - row_max)
    denom = tl.sum(numerators, axis=0)
    result = numerators / denom
    tl.store(y + row * cols + offsets, result, mask=mask)


def softmax(x: torch.Tensor, check_finite: bool = False):
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("expected a contiguous x=[rows, cols]")
    if not x.is_cuda:
        raise ValueError("this wrapper requires a CUDA tensor")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError("expected float16, bfloat16, or float32 input")
    rows, cols = map(int, x.shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if rows > 2**31 - 1 or cols > 2**31 - 1:
        raise ValueError("rows and cols must fit signed int32 indexing")
    if rows * cols > 2**31 - 1:
        raise ValueError("rows * cols must fit signed int32 pointer indexing")
    # This optional check is for a correctness preflight, not for a timed call.
    # The default keeps the kernel benchmark free of an extra device reduction.
    if check_finite and not bool(torch.isfinite(x).all()):
        raise ValueError("softmax input must be finite")
    block = triton.next_power_of_2(cols)
    y = torch.empty_like(x)
    softmax_row_kernel[(rows,)](x, y, rows, cols, BLOCK=block, num_warps=4)
    return y
