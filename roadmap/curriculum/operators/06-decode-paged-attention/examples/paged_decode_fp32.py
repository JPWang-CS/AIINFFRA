"""Teaching FP32 single-query PagedAttention baseline.

The kernel uses scalar dot products (``tl.sum(q * k)``), not a tiny-M dot
shortcut.  It is a correctness baseline, not FlashAttention-2 or a tuned
production backend.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import triton
import triton.language as tl

from paged_decode_host import (
    validate_head_counts,
    validate_memory_contract,
    validate_private_append_targets,
    validate_trusted_capacity,
)


BLOCK_T: Final[int] = 32
SUPPORTED_D = (16, 32, 64, 128)


@triton.jit
def _append_kv_kernel(
    new_k_ptr, new_v_ptr, cache_k_ptr, cache_v_ptr, lengths_ptr, table_ptr,
    NUM_PAGES, PAGE_SIZE, MAX_PAGES, KV_HEADS,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // KV_HEADS
    hk = pid % KV_HEADS
    d = tl.arange(0, HEAD_DIM)
    position = tl.load(lengths_ptr + b).to(tl.int64) - 1
    logical_page = position // PAGE_SIZE
    page_offset = position % PAGE_SIZE
    table_valid = (position >= 0) & (logical_page >= 0) & (logical_page < MAX_PAGES)
    physical_page = tl.load(
        table_ptr + b * MAX_PAGES + logical_page,
        mask=table_valid,
        other=0,
    ).to(tl.int64)
    valid = table_valid & (physical_page >= 0) & (physical_page < NUM_PAGES)
    cache_offset = (((physical_page * PAGE_SIZE + page_offset) * KV_HEADS + hk)
                    * HEAD_DIM + d)
    input_offset = ((b * KV_HEADS + hk) * HEAD_DIM + d)
    k = tl.load(new_k_ptr + input_offset, mask=valid, other=0.0)
    v = tl.load(new_v_ptr + input_offset, mask=valid, other=0.0)
    tl.store(cache_k_ptr + cache_offset, k, mask=valid)
    tl.store(cache_v_ptr + cache_offset, v, mask=valid)


@triton.jit
def _paged_decode_kernel(
    q_ptr, cache_k_ptr, cache_v_ptr, lengths_ptr, table_ptr, out_ptr,
    NUM_PAGES, PAGE_SIZE, MAX_PAGES, KV_HEADS,
    MAX_LEN: tl.constexpr, Q_HEADS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // Q_HEADS
    hq = pid % Q_HEADS
    d = tl.arange(0, HEAD_DIM)
    q_offset = ((b * Q_HEADS + hq) * HEAD_DIM + d)
    q = tl.load(q_ptr + q_offset)
    length = tl.load(lengths_ptr + b).to(tl.int64)
    kv_head = hq // GROUP_SIZE
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    u = tl.zeros((HEAD_DIM,), tl.float32)
    scale = 1.0 / tl.sqrt(float(HEAD_DIM))

    for kv0 in tl.range(0, MAX_LEN, BLOCK_T):
        logical = kv0 + tl.arange(0, BLOCK_T)
        logical_page = logical // PAGE_SIZE
        page_offset = logical % PAGE_SIZE
        logical_valid = (logical >= 0) & (logical < length)
        table_valid = logical_valid & (logical_page < MAX_PAGES)
        physical_page = tl.load(
            table_ptr + b * MAX_PAGES + logical_page,
            mask=table_valid,
            other=0,
        ).to(tl.int64)
        valid = table_valid & (physical_page >= 0) & (physical_page < NUM_PAGES)
        token_base = (((physical_page * PAGE_SIZE + page_offset) * KV_HEADS + kv_head)
                      * HEAD_DIM)[:, None]
        cache_offset = token_base + d[None, :]
        k = tl.load(cache_k_ptr + cache_offset, mask=valid[:, None], other=0.0)
        v = tl.load(cache_v_ptr + cache_offset, mask=valid[:, None], other=0.0)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        tile_m = tl.max(scores, axis=0)
        new_m = tl.maximum(m, tile_m)
        safe_m = tl.where(new_m != float("-inf"), new_m, 0.0)
        alpha = tl.where(m != float("-inf"), tl.exp(m - safe_m), 0.0)
        p = tl.where(valid, tl.exp(scores - safe_m), 0.0)
        u = alpha * u + tl.sum(p[:, None] * v, axis=0)
        l = alpha * l + tl.sum(p, axis=0)
        m = new_m

    denom = tl.where(l > 0.0, l, 1.0)
    tl.store(out_ptr + q_offset, u / denom)


def _check_tensor(tensor: torch.Tensor, name: str, ndim: int, device):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    if not tensor.is_cuda or tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the same CUDA device")


def _ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_begin = left.data_ptr()
    right_begin = right.data_ptr()
    left_end = left_begin + left.numel() * left.element_size()
    right_end = right_begin + right.numel() * right.element_size()
    return left_begin < right_end and right_begin < left_end


def _check_inputs(q, cache_k, cache_v, lengths, table, new_k, new_v, check_data):
    if not torch.cuda.is_available():
        raise RuntimeError("paged_decode_fp32 requires CUDA")
    _check_tensor(q, "q", 3, q.device)
    _check_tensor(cache_k, "cache_k", 4, q.device)
    _check_tensor(cache_v, "cache_v", 4, q.device)
    _check_tensor(lengths, "lengths", 1, q.device)
    _check_tensor(table, "table", 2, q.device)
    if q.dtype != torch.float32 or cache_k.dtype != torch.float32 or cache_v.dtype != torch.float32:
        raise TypeError("q/cache_k/cache_v must be FP32")
    if lengths.dtype not in (torch.int32, torch.int64) or table.dtype not in (torch.int32, torch.int64):
        raise TypeError("lengths/table must be int32 or int64 metadata")
    b, q_heads, dim = map(int, q.shape)
    pages, page_size, kv_heads, cache_dim = map(int, cache_k.shape)
    if kv_heads <= 0 or cache_v.shape != cache_k.shape or cache_dim != dim or q_heads % kv_heads:
        raise ValueError("incompatible q/cache shapes or non-divisible GQA heads")
    validate_head_counts(q_heads, kv_heads)
    if lengths.shape[0] != b or table.shape[0] != b or table.shape[1] <= 0:
        raise ValueError("lengths/table batch dimensions do not match")
    if dim not in SUPPORTED_D or b <= 0 or q_heads <= 0 or pages <= 0 or page_size <= 0:
        raise ValueError("empty batch/cache or unsupported head dimension")
    if (new_k is None) != (new_v is None):
        raise ValueError("new_k and new_v must be supplied together")
    if new_k is not None:
        _check_tensor(new_k, "new_k", 3, q.device)
        _check_tensor(new_v, "new_v", 3, q.device)
        if new_k.shape != (b, kv_heads, dim) or new_v.shape != new_k.shape:
            raise ValueError("new_k/new_v must be [B,Hkv,D]")
        if new_k.dtype != torch.float32 or new_v.dtype != torch.float32:
            raise TypeError("new_k/new_v must be FP32")
    if not check_data:
        return b, q_heads, dim, pages, page_size, kv_heads, int(table.shape[1]), None
    length_list = [int(x) for x in lengths.detach().cpu().tolist()]
    table_list = table.detach().cpu().tolist()
    if new_k is not None:
        validate_private_append_targets(length_list, table_list, pages, page_size)
    else:
        for row, length in zip(table_list, length_list):
            needed = (length + page_size - 1) // page_size
            if any(not 0 <= int(row[p]) < pages for p in range(needed)):
                raise ValueError("every logical page used by length must map to a valid physical page")
    return b, q_heads, dim, pages, page_size, kv_heads, int(table.shape[1]), max(length_list)


def paged_decode(q, cache_k, cache_v, lengths, table, *, new_k=None, new_v=None,
                 out=None, check_data=True, max_length=None):
    """Append an optional new KV row and compute one query per [B,Hq].

    ``lengths`` is the post-append logical length.  Append and attention are
    enqueued on the current CUDA stream; callers using another stream must
    establish event ownership before invoking this wrapper.  The ownership
    preflight sees only this batch: the allocator must guarantee global
    private/COW ownership and no concurrent reader before requesting append.
    This wrapper is not a complete allocator.
    """
    b, q_heads, dim, pages, page_size, kv_heads, max_pages, checked_max = _check_inputs(
        q, cache_k, cache_v, lengths, table, new_k, new_v, check_data)
    max_length = checked_max if check_data else max_length
    if max_length is None or max_length <= 0:
        raise ValueError("max_length is required when data checks are disabled")
    if out is None:
        out = torch.empty_like(q)
    elif (out.shape != q.shape or out.dtype != torch.float32 or not out.is_cuda
          or out.device != q.device or not out.is_contiguous()):
        raise ValueError("out must be an independent contiguous FP32 CUDA tensor")
    tensors = [q, cache_k, cache_v, lengths, table]
    if new_k is not None:
        tensors.extend([new_k, new_v])
    validate_memory_contract(q, cache_k, cache_v, lengths, table, new_k, new_v, out)
    if check_data and new_k is not None:
        # _check_inputs performed the ownership/COW preflight above.
        pass
    elif not check_data:
        validate_trusted_capacity(max_length, max_pages, page_size)
        if new_k is not None:
            raise ValueError("trusted path cannot append: run ownership preflight first")
    grid = (b * q_heads,)
    with torch.cuda.device(q.device):
        if new_k is not None:
            _append_kv_kernel[(b * kv_heads,)](
                new_k, new_v, cache_k, cache_v, lengths, table,
                pages, page_size, max_pages, kv_heads, HEAD_DIM=dim, num_warps=4)
        _paged_decode_kernel[grid](
            q, cache_k, cache_v, lengths, table, out,
            pages, page_size, max_pages, kv_heads,
            MAX_LEN=triton.cdiv(max_length, BLOCK_T) * BLOCK_T,
            Q_HEADS=q_heads, GROUP_SIZE=q_heads // kv_heads,
            HEAD_DIM=dim, BLOCK_T=BLOCK_T, num_warps=4)
    return out
