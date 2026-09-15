"""FP32 one-tile-lookahead attention and standalone validation/benchmark CLI.

The kernel uses explicit ordinary Triton loads for the next K/V tile. It is a
compiler-scheduled lookahead experiment, not a cp.async/TMA or FA3 kernel.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:  # Keep --help and the GPU skip contract usable on CPU hosts.
    torch = triton = tl = None
    _RUNTIME_IMPORT_ERROR = exc
else:
    _RUNTIME_IMPORT_ERROR = None


BLOCK_Q = 16
BLOCK_KV = 32
SUPPORTED_D = (16, 32, 64, 128)


if triton is not None:
    @triton.jit
    def _attention_lookahead_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        B, H, M, N, D,
        BLOCK_Q: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        PAD_D: tl.constexpr,
        SCALE: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        """Contiguous [B,H,M,D] x [B,H,N,D] FP32 attention, no GQA."""
        pid_q = tl.program_id(0).to(tl.int64)
        pid_bh = tl.program_id(1).to(tl.int64)
        q_abs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        kv_rel = tl.arange(0, BLOCK_KV).to(tl.int64)
        d = tl.arange(0, PAD_D)
        q_valid = q_abs < M
        d_valid = d < D

        q_base = pid_bh * M * D
        kv_base = pid_bh * N * D
        q_offsets = q_base + q_abs[:, None] * D + d[None, :]
        q = tl.load(q_ptr + q_offsets,
                    mask=q_valid[:, None] & d_valid[None, :], other=0.0)

        # Online state, one max/denominator per query row and one unnormalized
        # output vector per row. All three remain FP32 throughout this kernel.
        m = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
        l = tl.zeros((BLOCK_Q,), tl.float32)
        u = tl.zeros((BLOCK_Q, PAD_D), tl.float32)

        first_offsets = kv_base + kv_rel[:, None] * D + d[None, :]
        first_mask = (kv_rel < N)[:, None] & d_valid[None, :]
        k = tl.load(k_ptr + first_offsets, mask=first_mask, other=0.0)
        v = tl.load(v_ptr + first_offsets, mask=first_mask, other=0.0)

        for kv0 in range(0, N, BLOCK_KV):
            k_abs = kv0 + kv_rel
            key_valid = k_abs < N
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE

            # Look one tile ahead. These K/V loads are independent of the
            # current tile's softmax/P@V arithmetic and remain live for the
            # next loop iteration. They are ordinary loads, not async copies.
            next_abs = kv0.to(tl.int64) + BLOCK_KV + kv_rel
            next_offsets = kv_base + next_abs[:, None] * D + d[None, :]
            next_mask = (next_abs < N)[:, None] & d_valid[None, :]
            k_next = tl.load(k_ptr + next_offsets, mask=next_mask, other=0.0)
            v_next = tl.load(v_ptr + next_offsets, mask=next_mask, other=0.0)

            valid_score = q_valid[:, None] & key_valid[None, :]
            if CAUSAL:
                valid_score = valid_score & (k_abs[None, :] <= q_abs[:, None])
            scores = tl.where(valid_score, scores, float("-inf"))
            block_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m, block_m)
            safe_m = tl.where(m_new != float("-inf"), m_new, 0.0)
            alpha = tl.where(m != float("-inf"), tl.exp(m - safe_m), 0.0)
            p = tl.where(valid_score, tl.exp(scores - safe_m[:, None]), 0.0)

            # Recurrence order is mandatory: rebase old U, then add this tile's
            # P@V; l follows the same alpha and exponent base; m advances last.
            u = alpha[:, None] * u + tl.dot(p, v, input_precision="ieee")
            l = alpha * l + tl.sum(p, axis=1)
            m = m_new
            k, v = k_next, v_next

        out = u / tl.where(l > 0.0, l, 1.0)[:, None]
        o_offsets = q_base + q_abs[:, None] * D + d[None, :]
        tl.store(o_ptr + o_offsets, out,
                 mask=q_valid[:, None] & d_valid[None, :])


def _require_runtime() -> None:
    if torch is None or triton is None:
        raise RuntimeError(f"PyTorch/Triton import unavailable: {_RUNTIME_IMPORT_ERROR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")


def _ranges_overlap(left, right) -> bool:
    left_begin = left.data_ptr()
    right_begin = right.data_ptr()
    left_end = left_begin + left.numel() * left.element_size()
    right_end = right_begin + right.numel() * right.element_size()
    return left_begin < right_end and right_begin < left_end


def prefill_attention_pipeline(q, k, v, *, causal: bool = False, out=None):
    """Run the lookahead kernel for identical contiguous [B,H,S,D] tensors.

    Supports FP32 D in {16,32,64,128}, noncausal or same-length causal masks.
    It does not implement GQA, variable lengths, dropout, backward, or custom masks.
    """
    _require_runtime()
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("expected identical Q/K/V shapes [B,H,S,D]")
    if any(x.dtype is not torch.float32 for x in (q, k, v)):
        raise TypeError("pipeline kernel is FP32-only")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and q.device == k.device == v.device):
        raise ValueError("Q/K/V must be on the same CUDA device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous")
    B, H, S, D = map(int, q.shape)
    if B <= 0 or H <= 0 or S <= 0 or D not in SUPPORTED_D:
        raise ValueError(f"expected B,H,S>0 and D in {SUPPORTED_D}")
    if out is None:
        out = torch.empty_like(q)
    elif (out.shape != q.shape or out.dtype is not torch.float32 or not out.is_cuda
          or not out.is_contiguous() or out.device != q.device):
        raise ValueError("out must be independent contiguous FP32 with Q's shape/device")
    elif any(_ranges_overlap(out, x) for x in (q, k, v)):
        raise ValueError("out must not overlap Q, K, or V")
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    with torch.cuda.device(q.device):
        _attention_lookahead_kernel[grid](
            q, k, v, out, B, H, S, S, D,
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, PAD_D=D,
            SCALE=1.0 / math.sqrt(D), CAUSAL=causal, num_warps=4,
        )
    return out


def solve(Q, K, V, out, M: int, N: int, d: int):
    """LeetGPU #6 contract: FP32 noncausal Q[M,d], K/V[N,d] -> out[M,d]."""
    _require_runtime()
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q/K/V must be rank-2")
    if tuple(Q.shape) != (M, d) or tuple(K.shape) != (N, d) or tuple(V.shape) != (N, d):
        raise ValueError("tensor shapes do not match M, N, d")
    if tuple(out.shape) != (M, d):
        raise ValueError("out must have shape [M,d]")
    if M <= 0 or N <= 0 or d <= 0 or d > 128:
        raise ValueError("M, N, d must be positive and d <= 128")
    if any(x.dtype is not torch.float32 for x in (Q, K, V, out)):
        raise TypeError("solve is FP32-only")
    if not (Q.is_cuda and K.is_cuda and V.is_cuda and out.is_cuda):
        raise ValueError("Q/K/V/out must be CUDA tensors")
    if not (Q.device == K.device == V.device == out.device):
        raise ValueError("Q/K/V/out must use the same CUDA device")
    if not (Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and out.is_contiguous()):
        raise ValueError("Q/K/V/out must be contiguous")
    if any(_ranges_overlap(out, x) for x in (Q, K, V)):
        raise ValueError("out must not overlap Q/K/V")
    pad_d = max(16, triton.next_power_of_2(d))
    with torch.cuda.device(Q.device):
        _attention_lookahead_kernel[(triton.cdiv(M, BLOCK_Q), 1)](
            Q, K, V, out, 1, 1, M, N, d,
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, PAD_D=pad_d,
            SCALE=1.0 / math.sqrt(d), CAUSAL=False, num_warps=4,
        )
    return out


def _reference_prefill(q, k, v, causal: bool):
    torch.backends.cuda.matmul.allow_tf32 = False
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        S = q.shape[-2]
        valid = torch.arange(S, device=q.device)[:, None] >= torch.arange(S, device=q.device)[None, :]
        scores = scores.masked_fill(~valid, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _reference_solve(Q, K, V):
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.softmax(Q @ K.T / math.sqrt(Q.shape[-1]), dim=-1) @ V


def _sdpa_math(q, k, v, causal: bool):
    """PyTorch SDPA forced to its FP32 math backend for a strong library baseline."""
    import torch.nn.functional as F

    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        context = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
    with context:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _assert_finite(*xs):
    assert all(bool(torch.isfinite(x).all().item()) for x in xs)


def run_correctness() -> None:
    from prefill_fp32 import prefill_attention as baseline

    for causal in (False, True):
        for S in (1, 17, 33, 65):
            for D in (16, 32, 64):
                torch.manual_seed(1000 + S + D + int(causal))
                q = torch.randn((2, 2, S, D), device="cuda", dtype=torch.float32)
                k, v = torch.randn_like(q), torch.randn_like(q)
                ref = _reference_prefill(q, k, v, causal)
                sdpa = _sdpa_math(q, k, v, causal)
                base = baseline(q, k, v, causal=causal)
                got = prefill_attention_pipeline(q, k, v, causal=causal)
                _assert_finite(q, k, v, ref, sdpa, base, got)
                torch.testing.assert_close(sdpa, ref, rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(base, ref, rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
                print(f"prefill PASS: B=2 H=2 S={S} D={D} causal={causal}")

    # Rectangular challenge contract and both query/key tail masks.
    for M, N, D in ((2, 3, 4), (19, 35, 7)):
        torch.manual_seed(M * 100 + N)
        Q = torch.randn((M, D), device="cuda", dtype=torch.float32)
        K = torch.randn((N, D), device="cuda", dtype=torch.float32)
        V = torch.randn_like(K)
        out = torch.empty_like(Q)
        got = solve(Q, K, V, out, M, N, D)
        ref = _reference_solve(Q, K, V)
        _assert_finite(Q, K, V, got, ref)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
        print(f"solve PASS: M={M} N={N} d={D}")

    # A future-key mutation cannot affect earlier causal queries.
    S, D = 33, 16
    q = torch.randn((1, 1, S, D), device="cuda", dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    baseline = prefill_attention_pipeline(q, k, v, causal=True)
    cut = S // 2
    k2, v2 = k.clone(), v.clone()
    k2[:, :, cut + 1:] = 10000.0
    v2[:, :, cut + 1:] = -777.0
    mutated = prefill_attention_pipeline(q, k2, v2, causal=True)
    torch.testing.assert_close(mutated[:, :, :cut + 1], baseline[:, :, :cut + 1], rtol=1e-4, atol=1e-4)
    print("causal future-key isolation PASS")


def _wall_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def run_benchmark(args) -> None:
    from prefill_fp32 import prefill_attention as baseline

    B, H, S, D = args.batch, args.heads, args.seq_len, args.head_dim
    torch.manual_seed(2026)
    q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out_base, out_pipe = torch.empty_like(q), torch.empty_like(q)
    ref = _reference_prefill(q, k, v, args.causal)
    baseline(q, k, v, causal=args.causal, out=out_base)
    prefill_attention_pipeline(q, k, v, causal=args.causal, out=out_pipe)
    torch.testing.assert_close(out_base, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_pipe, ref, rtol=1e-4, atol=1e-4)
    sdpa = _sdpa_math(q, k, v, args.causal)
    torch.testing.assert_close(sdpa, ref, rtol=1e-4, atol=1e-4)

    base_ms = _wall_ms(lambda: baseline(q, k, v, causal=args.causal, out=out_base), args.warmup, args.iters)
    pipe_ms = _wall_ms(lambda: prefill_attention_pipeline(q, k, v, causal=args.causal, out=out_pipe), args.warmup, args.iters)
    sdpa_ms = _wall_ms(lambda: _sdpa_math(q, k, v, args.causal), args.warmup, args.iters)
    torch.backends.cuda.matmul.allow_tf32 = False
    eager_ms = _wall_ms(lambda: _reference_prefill(q, k, v, args.causal), args.warmup, args.iters)
    print(f"GPU={torch.cuda.get_device_name(q.device)} torch={torch.__version__} triton={triton.__version__} python={platform.python_version()}")
    print(f"shape=[{B},{H},{S},{D}] dtype=FP32 causal={args.causal} BLOCK_Q={BLOCK_Q} BLOCK_KV={BLOCK_KV}")
    print(f"existing Triton baseline wrapper_ms={base_ms:.5f} (preallocated output)")
    print(f"lookahead Triton wrapper_ms={pipe_ms:.5f} (preallocated output; no speedup claim)")
    print(f"PyTorch SDPA FP32 math backend_ms={sdpa_ms:.5f} (returned-output allocation included; separate boundary)")
    print(f"PyTorch dense FP32 correctness reference_ms={eager_ms:.5f} (SxS intermediates/output allocation included)")
    print("Only the two Triton values share the preallocated-output wrapper boundary and are directly comparable.")
    print("All timings are synchronized host wall time; JIT warmup excluded; no speedup is asserted by this program.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="time same-FP32 Triton baseline/lookahead and PyTorch SDPA math")
    parser.add_argument("--causal", action="store_true", help="use same-length causal attention for benchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, choices=SUPPORTED_D, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args(argv)
    if torch is None or triton is None:
        print(f"SKIP 77: PyTorch/Triton unavailable: {_RUNTIME_IMPORT_ERROR}", file=sys.stderr)
        return 77
    if not torch.cuda.is_available():
        print("SKIP 77: no CUDA device is available", file=sys.stderr)
        return 77
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be > 0")
    if args.benchmark:
        run_benchmark(args)
    else:
        run_correctness()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
