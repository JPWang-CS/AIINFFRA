"""A readable Triton pointwise SwiGLU kernel.

The kernel computes out = silu(gate) * up for two contiguous 2-D tensors.
It is an implementation example, not a claim of GPU validation.  The host
wrapper checks the tensor contract without doing a timed finite-value scan.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


DEFAULT_BLOCK = 1024
ALLOWED_BLOCKS = frozenset({128, 256, 512, 1024, 2048})


@triton.jit
def fused_swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """One Triton program owns a contiguous segment of the flattened tensor."""
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # exp(-abs(x)) avoids constructing exp(large positive) for finite x.
    # NaN and infinity still follow IEEE propagation; the wrapper's contract
    # is finite, reasonably scaled input rather than silent special-value use.
    exp_neg_abs = tl.exp(-tl.abs(gate))
    sigmoid = tl.where(
        gate >= 0.0,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs),
    )
    result = (gate * sigmoid) * up
    tl.store(out_ptr + offsets, result, mask=mask)


def _check_tensor_contract(gate: torch.Tensor, up: torch.Tensor) -> None:
    if gate.ndim != 2 or up.ndim != 2:
        raise ValueError("gate and up must both be 2-D [T, I] tensors")
    if gate.shape != up.shape:
        raise ValueError(f"shape mismatch: gate={tuple(gate.shape)}, up={tuple(up.shape)}")
    if gate.device != up.device or not gate.is_cuda:
        raise ValueError("gate and up must be on the same CUDA device")
    if gate.dtype != up.dtype or gate.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise ValueError("gate and up must share float16, bfloat16, or float32 dtype")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate and up must be contiguous row-major tensors")
    if gate.numel() > 2**63 - 1:
        raise ValueError("flattened element count does not fit signed int64")


def _ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.numel() == 0 or right.numel() == 0:
        return False
    left_begin = left.data_ptr()
    right_begin = right.data_ptr()
    left_end = left_begin + left.numel() * left.element_size()
    right_end = right_begin + right.numel() * right.element_size()
    return left_begin < right_end and right_begin < left_end


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` with one pointwise Triton launch.

    ``out`` is optional for readability; a benchmark should allocate it once
    and pass it on every call.  This function intentionally does not call
    ``torch.isfinite``: a per-call device reduction would add synchronization
    and is outside the kernel's hot path.  Validate finite inputs in a
    correctness preflight when that policy is required.
    """
    _check_tensor_contract(gate, up)
    if block not in ALLOWED_BLOCKS:
        raise ValueError(f"block must be one of {sorted(ALLOWED_BLOCKS)}")
    if out is None:
        out = torch.empty_like(gate)
    elif out.shape != gate.shape or out.device != gate.device or out.dtype != gate.dtype:
        raise ValueError("out must have gate's shape, device, and dtype")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous row-major")
    if _ranges_overlap(out, gate) or _ranges_overlap(out, up):
        raise ValueError("out must not overlap gate or up by byte range")

    n_elements = gate.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    fused_swiglu_kernel[grid](
        gate,
        up,
        out,
        n_elements,
        BLOCK=block,
        num_warps=4,
    )
    return out


__all__ = ["fused_swiglu", "fused_swiglu_kernel"]
