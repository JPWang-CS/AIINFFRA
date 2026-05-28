"""Triton Flash Attention — forward pass.

Based on the Flash Attention 2 algorithm:
- Q split into blocks along sequence dimension
- K/V split into blocks along sequence dimension
- Online softmax: incrementally update output with rescaling

Reference: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import triton
import triton.language as tl
import torch
import math


@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim,
    stride_q_seq, stride_q_dim,
    stride_k_seq, stride_k_dim,
    stride_v_seq, stride_v_dim,
    stride_o_seq, stride_o_dim,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # Program ID = which Q block
    pid = tl.program_id(0)
    q_start = pid * BLOCK_Q

    # Offsets within this Q block
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_dim = tl.arange(0, head_dim)

    # Load Q tile
    q = tl.load(
        q_ptr + offs_q[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim,
        mask=offs_q[:, None] < seq_len,
        other=0.0,
    )

    # Online softmax state
    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    # Loop over K/V blocks
    for kv_start in range(0, seq_len, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)

        # Load K tile
        k = tl.load(
            k_ptr + offs_kv[:, None] * stride_k_seq + offs_dim[None, :] * stride_k_dim,
            mask=offs_kv[:, None] < seq_len,
            other=0.0,
        )

        # Compute attention scores: Q @ K^T / sqrt(d)
        scores = tl.dot(q, tl.trans(k))
        scores = scores / math.sqrt(head_dim)

        # Update online softmax
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        # Load V tile and accumulate
        v = tl.load(
            v_ptr + offs_kv[:, None] * stride_v_seq + offs_dim[None, :] * stride_v_dim,
            mask=offs_kv[:, None] < seq_len,
            other=0.0,
        )

        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    tl.store(
        o_ptr + offs_q[:, None] * stride_o_seq + offs_dim[None, :] * stride_o_dim,
        acc,
        mask=offs_q[:, None] < seq_len,
    )


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Forward: O = softmax(QK^T / sqrt(d)) × V"""
    assert q.dim() == 3  # (batch, seq, dim)
    B, seq_len, head_dim = q.shape

    o = torch.empty_like(q)
    BLOCK_Q = 32
    BLOCK_KV = 64
    grid = (triton.cdiv(seq_len, BLOCK_Q),)

    flash_attn_kernel[grid](
        q, k, v, o,
        seq_len, head_dim,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
    )
    return o


if __name__ == "__main__":
    B, S, D = 1, 256, 64
    q = torch.randn((B, S, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, S, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, S, D), device="cuda", dtype=torch.float16)

    triton_out = flash_attention(q, k, v)

    # PyTorch reference
    scale = 1.0 / math.sqrt(D)
    attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    torch_out = attn @ v

    diff = (triton_out - torch_out).abs().max().item()
    print(f"Max diff: {diff:.4e}")
    print("PASS" if diff < 1e-2 else "FAIL")
