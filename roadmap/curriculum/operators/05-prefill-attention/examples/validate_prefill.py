"""Correctness-first GPU harness for the square teaching baseline and #6 shape math."""

from __future__ import annotations

import argparse
import math
import platform
import time

import torch
import triton

from prefill_fp32 import prefill_attention

SUPPORTED_D = (16, 32, 64, 128)


def reference_prefill(q, k, v, causal=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        s = q.shape[-2]
        scores = scores.masked_fill(~torch.tril(torch.ones((s, s), device=q.device, dtype=torch.bool)), float("-inf"))
    return torch.softmax(scores, dim=-1).matmul(v.float())


def reference_leetgpu_6(Q, K, V, M, N, d):
    """Current #6 contract: rectangular [M,d] x [N,d], noncausal, no batch."""
    assert tuple(Q.shape) == (M, d) and tuple(K.shape) == (N, d) and tuple(V.shape) == (N, d)
    return torch.softmax(Q.float() @ K.float().transpose(0, 1) / math.sqrt(d), dim=-1) @ V.float()


def finite(*xs):
    assert all(bool(torch.isfinite(x).all().item()) for x in xs)


def check(B, H, S, D, causal):
    torch.manual_seed(B * 10000 + H * 1000 + S + D)
    q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = prefill_attention(q, k, v, causal=causal, check_finite=True)
    ref = reference_prefill(q, k, v, causal)
    finite(q, k, v, out, ref)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    if causal:
        if S > 1:
            cut = S // 2
            future_k, future_v = k.clone(), v.clone()
            future_k[:, :, cut + 1:, :] = 10000
            future_v[:, :, cut + 1:, :] = -777
            changed = prefill_attention(q, future_k, future_v, causal=True, check_finite=True)
            torch.testing.assert_close(changed[:, :, :cut + 1], out[:, :, :cut + 1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out[:, :, 0], v[:, :, 0], rtol=1e-4, atol=1e-4)
    print(f"correctness OK: B={B} H={H} S={S} D={D} causal={causal}")


def check_bh_tail(causal):
    B, H, S, D = 2, 2, 33, 16
    q = torch.zeros((B, H, S, D), device="cuda", dtype=torch.float32)
    k = torch.zeros_like(q)
    v = torch.empty_like(q)
    for b in range(B):
        for h in range(H):
            v[b, h].fill_(10.0 + 100.0 * b + 10.0 * h)
    out = prefill_attention(q, k, v, causal=causal, check_finite=True)
    expected = torch.empty_like(out)
    for b in range(B):
        for h in range(H):
            expected[b, h].fill_(10.0 + 100.0 * b + 10.0 * h)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    print(f"tail-mask/BH-offset OK: B={B} H={H} S={S} D={D} causal={causal}")


def benchmark(B, H, S, D, causal):
    q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float32).contiguous()
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    prefill_attention(q, k, v, causal=causal, out=out, check_finite=True)
    ref = reference_prefill(q, k, v, causal)
    finite(q, k, v, out, ref)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    for _ in range(10):
        prefill_attention(q, k, v, causal=causal, out=out)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        prefill_attention(q, k, v, causal=causal, out=out)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000 / 50
    useful_flops = (2 * B * H * S * (S + 1) * D if causal else 4 * B * H * S * S * D)
    q_tiles = (S + 16 - 1) // 16
    kv_tiles = (S + 32 - 1) // 32
    padded_tile_work = 4 * B * H * q_tiles * 16 * kv_tiles * 32 * D
    print(f"benchmark: gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} python={platform.python_version()} "
          f"shape=[{B},{H},{S},{D}] precision=FP32 causal={causal} ms={ms:.4f} "
          f"unmasked_valid_domain_FLOPs={useful_flops} unmasked_domain_GFLOP/s={useful_flops/(ms*1e6):.3f} "
          f"padded_tile_work_estimate={padded_tile_work}")
    print("wall-clock average includes wrapper checks, launch, and synchronize; allocation/JIT warmup excluded")
    print("causal kernel still scans all KV tiles and masks future scores; useful FLOPs is not executed FLOPs")
    print(f"triton={triton.__version__} BLOCK_Q=16 BLOCK_KV=32 num_warps=4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, choices=SUPPORTED_D, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    for causal in (False, True):
        for D in (32, 64, 128):
            check(1, 1, 1, D, causal)
            check(2, 2, 17, D, causal)
            check(2, 2, 33, D, causal)
        for S in (33, 65):
            for B, H in ((2, 2), (2, 4)):
                check(B, H, S, 16, causal)
        check_bh_tail(causal)
    # Explicitly verify the current #6 rectangular math without claiming the
    # square prefill wrapper is a submission for that challenge.
    Q = torch.randn((2, 4), device="cuda", dtype=torch.float32)
    K = torch.randn((3, 4), device="cuda", dtype=torch.float32)
    V = torch.randn((3, 4), device="cuda", dtype=torch.float32)
    expected = torch.full((2, 4), 2.0, device="cuda", dtype=torch.float32)
    Q.zero_(); K.zero_(); V.copy_(torch.tensor([[1., 1., 1., 1.], [3., 3., 3., 3.], [2., 2., 2., 2.]], device="cuda"))
    torch.testing.assert_close(reference_leetgpu_6(Q, K, V, 2, 3, 4), expected)
    if args.benchmark:
        benchmark(args.batch, args.heads, args.seq_len, args.head_dim, args.causal)


if __name__ == "__main__":
    main()
