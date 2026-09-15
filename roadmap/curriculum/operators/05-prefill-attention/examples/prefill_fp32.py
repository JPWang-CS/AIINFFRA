"""教学用 FP32 prefill attention baseline.

This is intentionally a small, square [B, H, S, D] Triton kernel.  It is not
the current LeetGPU #6 rectangular solve(Q, K, V, out, M, N, d), and it does
not claim to implement GQA, varlen, dropout, or backward.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import triton
import triton.language as tl


BLOCK_Q: Final[int] = 16
BLOCK_KV: Final[int] = 32
SUPPORTED_D = (16, 32, 64, 128)


@triton.jit
def _prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, S, D,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1)
    q0 = pid_q * BLOCK_Q
    q_abs = q0 + tl.arange(0, BLOCK_Q)
    d = tl.arange(0, HEAD_DIM)
    q_valid = q_abs < S
    # All inputs are contiguous [B,H,S,D].  Use int64 arithmetic for the
    # logical base and row offsets so large shape arithmetic is not truncated.
    bh = pid_bh.to(tl.int64)
    base = (bh * S) * HEAD_DIM
    q_offsets = base + q_abs[:, None] * HEAD_DIM + d[None, :].to(tl.int64)
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)

    m = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    u = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    scale = SCALE

    for kv0 in range(0, S, BLOCK_KV):
        k_abs = kv0 + tl.arange(0, BLOCK_KV)
        key_valid = k_abs < S
        k_offsets = base + k_abs[:, None].to(tl.int64) * HEAD_DIM + d[None, :].to(tl.int64)
        k = tl.load(k_ptr + k_offsets, mask=key_valid[:, None], other=0.0)
        v = tl.load(v_ptr + k_offsets, mask=key_valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        valid_score = q_valid[:, None] & key_valid[None, :]
        if CAUSAL:
            valid_score = valid_score & (k_abs[None, :] <= q_abs[:, None])
        # Mask the score, not just K/V.  Otherwise padded keys enter softmax.
        scores = tl.where(valid_score, scores, float("-inf"))
        block_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m, block_m)
        safe_m = tl.where(m_new != float("-inf"), m_new, 0.0)
        alpha = tl.where(m != float("-inf"), tl.exp(m - safe_m), 0.0)
        p = tl.where(valid_score, tl.exp(scores - safe_m[:, None]), 0.0)
        u = alpha[:, None] * u + tl.dot(p, v, input_precision="ieee")
        l = alpha * l + tl.sum(p, axis=1)
        m = m_new

    denom = tl.where(l > 0.0, l, 1.0)
    out = u / denom[:, None]
    o_offsets = base + q_abs[:, None] * HEAD_DIM + d[None, :].to(tl.int64)
    tl.store(o_ptr + o_offsets, out, mask=q_valid[:, None])


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("prefill_fp32 requires CUDA")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected Q/K/V with shape [B,H,S,D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("this teaching baseline requires identical Q/K/V shapes")
    if q.dtype is not torch.float32 or k.dtype is not torch.float32 or v.dtype is not torch.float32:
        raise TypeError("this teaching baseline is FP32-only")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Q/K/V must be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("Q/K/V must be on the same CUDA device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous [B,H,S,D]")
    B, H, S, D = map(int, q.shape)
    if B <= 0 or H <= 0 or S <= 0:
        raise ValueError("B, H, and S must be positive")
    if D not in SUPPORTED_D:
        raise ValueError(f"D must be one of {SUPPORTED_D}")
    return B, H, S, D


def _ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_begin = left.data_ptr()
    right_begin = right.data_ptr()
    left_end = left_begin + left.numel() * left.element_size()
    right_end = right_begin + right.numel() * right.element_size()
    return left_begin < right_end and right_begin < left_end


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False,
                      out: torch.Tensor | None = None, check_finite: bool = False) -> torch.Tensor:
    B, H, S, D = _check_inputs(q, k, v)
    if check_finite and not (torch.isfinite(q).all() and torch.isfinite(k).all() and torch.isfinite(v).all()):
        raise ValueError("finite inputs are a preflight condition, not timed kernel work")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype is not torch.float32 or not out.is_cuda or not out.is_contiguous() or out.device != q.device:
        raise ValueError("out must be an independent contiguous FP32 CUDA tensor with Q's shape/device")
    elif any(_ranges_overlap(out, x) for x in (q, k, v)):
        raise ValueError("out must not overlap Q, K, or V by contiguous byte range")
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    with torch.cuda.device(q.device):
        _prefill_kernel[grid](q, k, v, out, B, H, S, D,
                              BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
                              HEAD_DIM=D, SCALE=1.0 / math.sqrt(D),
                              CAUSAL=causal, num_warps=4)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, S, D = 2, 2, 33, 32
    q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    print(prefill_attention(q, k, v, causal=True).shape)
