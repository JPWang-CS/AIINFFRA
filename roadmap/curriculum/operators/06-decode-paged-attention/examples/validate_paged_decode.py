"""GPU correctness harness for the teaching paged-decode baseline.

Correctness is the default. ``--benchmark`` is a deliberately small timing
demo; its dense references are not production baselines.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

try:
    import torch
except ModuleNotFoundError:  # Keep the script useful as a documented preflight on CPU hosts.
    torch = None


def dense_attention(q, k, v, lengths):
    _, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    mapping = torch.arange(q_heads, device=q.device) // (q_heads // kv_heads)
    kq = k[:, :, mapping, :].permute(0, 2, 1, 3)
    vq = v[:, :, mapping, :].permute(0, 2, 1, 3)
    scores = (q[:, :, None, :] * kq).sum(-1) / dim**0.5
    positions = torch.arange(k.shape[1], device=q.device)[None, None, :]
    valid = positions < lengths[:, None, None]
    probs = torch.softmax(scores.masked_fill(~valid, float("-inf")), dim=-1)
    return (probs[..., None] * vq).sum(dim=2)


def make_inputs(device, batch, q_heads, kv_heads, dim, lengths, seed):
    rng = random.Random(seed)
    page_size = 4
    max_len = max(lengths)
    max_pages = (max_len + page_size - 1) // page_size
    needed_total = sum((n + page_size - 1) // page_size for n in lengths)
    physical_pages = needed_total + 3
    chosen_pool = list(range(physical_pages))
    rng.shuffle(chosen_pool)
    table_host, cursor = [], 0
    for length in lengths:
        needed = (length + page_size - 1) // page_size
        chosen = chosen_pool[cursor:cursor + needed]
        cursor += needed
        table_host.append(chosen + [-1] * (max_pages - needed))
    lengths_t = torch.tensor(lengths, device=device, dtype=torch.int32)
    table = torch.tensor(table_host, device=device, dtype=torch.int32)
    logical_k = torch.randn((batch, max_len, kv_heads, dim), device=device, dtype=torch.float32)
    logical_v = torch.randn_like(logical_k)
    cache_k = torch.zeros((physical_pages, page_size, kv_heads, dim), device=device)
    cache_v = torch.zeros_like(cache_k)
    for b in range(batch):
        for token in range(lengths[b] - 1):
            page, offset = divmod(token, page_size)
            physical = table_host[b][page]
            cache_k[physical, offset].copy_(logical_k[b, token])
            cache_v[physical, offset].copy_(logical_v[b, token])
    new_k = torch.stack([logical_k[b, lengths[b] - 1] for b in range(batch)])
    new_v = torch.stack([logical_v[b, lengths[b] - 1] for b in range(batch)])
    q = torch.randn((batch, q_heads, dim), device=device, dtype=torch.float32)
    return q, cache_k, cache_v, lengths_t, table, new_k, new_v, logical_k, logical_v, max_len, page_size


def gather_dense(cache_k, cache_v, table, lengths, max_len, page_size):
    batch = int(table.shape[0])
    _, _, kv_heads, dim = cache_k.shape
    k = torch.zeros((batch, max_len, kv_heads, dim), device=cache_k.device)
    v = torch.zeros_like(k)
    for b in range(batch):
        for token in range(int(lengths[b].item())):
            page, offset = divmod(token, page_size)
            physical = table[b, page]
            k[b, token].copy_(cache_k[physical, offset])
            v[b, token].copy_(cache_v[physical, offset])
    return k, v


def expect_error(fn, label):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError(f"{label} was accepted")


def correctness_case(device, batch, q_heads, kv_heads, dim, lengths, seed):
    from paged_decode_fp32 import paged_decode

    data = make_inputs(device, batch, q_heads, kv_heads, dim, lengths, seed)
    q, cache_k, cache_v, lengths_t, table, new_k, new_v, logical_k, logical_v, max_len, page_size = data
    got = paged_decode(q, cache_k, cache_v, lengths_t, table,
                       new_k=new_k, new_v=new_v, check_data=True)
    expected = dense_attention(q, logical_k, logical_v, lengths_t)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)

    zero_q = torch.zeros_like(q)
    zero_k = torch.zeros_like(logical_k)
    data = make_inputs(device, batch, q_heads, kv_heads, dim, lengths, seed + 101)
    _, z_cache_k, z_cache_v, z_lengths, z_table, z_new_k, z_new_v, _, z_logical_v, _, _ = data
    zero_expected = dense_attention(zero_q, zero_k, z_logical_v, lengths_t)
    zero_got = paged_decode(zero_q, z_cache_k, z_cache_v, z_lengths, z_table,
                            new_k=z_new_k, new_v=z_new_v, check_data=True)
    torch.testing.assert_close(zero_got, zero_expected, rtol=1e-4, atol=1e-4)

    bad_table = table.clone()
    bad_table[0, 0] = cache_k.shape[0]
    expect_error(lambda: paged_decode(q, cache_k, cache_v, lengths_t, bad_table), "bad physical page")
    expect_error(lambda: paged_decode(q, cache_k, cache_k, lengths_t, table), "K/V alias")
    expect_error(lambda: paged_decode(q, cache_k, cache_v, lengths_t, table, out=q), "out alias")
    shared_table = table.clone()
    target_page_b1 = int(shared_table[1, (lengths[1] - 1) // page_size].item())
    shared_table[0, (lengths[0] - 1) // page_size] = target_page_b1
    expect_error(lambda: paged_decode(q, cache_k, cache_v, lengths_t, shared_table,
                                      new_k=new_k, new_v=new_v), "shared append page")
    print(f"PASS: B={batch} Hq={q_heads} Hkv={kv_heads} D={dim} lengths={lengths}")


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def benchmark(device, batch, q_heads, seq_len):
    from paged_decode_fp32 import paged_decode

    kv_heads = q_heads // 2 if q_heads % 2 == 0 else 1
    lengths = [seq_len] * batch
    data = make_inputs(device, batch, q_heads, kv_heads, 16, lengths, 211)
    q, cache_k, cache_v, lengths_t, table, new_k, new_v, logical_k, logical_v, max_len, page_size = data
    checked = paged_decode(q, cache_k, cache_v, lengths_t, table,
                           new_k=new_k, new_v=new_v, check_data=True)
    expected = dense_attention(q, logical_k, logical_v, lengths_t)
    torch.testing.assert_close(checked, expected, rtol=1e-4, atol=1e-4)
    out = torch.empty_like(q)
    reuse_us = timed(lambda: paged_decode(q, cache_k, cache_v, lengths_t, table, out=out,
                                         check_data=False, max_length=max_len))
    alloc_us = timed(lambda: paged_decode(q, cache_k, cache_v, lengths_t, table,
                                         check_data=False, max_length=max_len))
    dense_k, dense_v = gather_dense(cache_k, cache_v, table, lengths_t, max_len, page_size)
    torch.testing.assert_close(dense_k, logical_k, rtol=0, atol=0)
    torch.testing.assert_close(dense_v, logical_v, rtol=0, atol=0)
    torch.testing.assert_close(dense_attention(q, dense_k, dense_v, lengths_t),
                               expected, rtol=1e-4, atol=1e-4)
    dense_us = timed(lambda: dense_attention(q, dense_k, dense_v, lengths_t))
    gather_us = timed(lambda: dense_attention(q, *gather_dense(cache_k, cache_v, table,
                                                                lengths_t, max_len, page_size), lengths_t))
    triton_version = "unavailable"
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        pass
    print(f"device={torch.cuda.get_device_name(device)} torch={torch.__version__} triton={triton_version}")
    print(f"shape q=[{batch},{q_heads},16] kv=[{cache_k.shape[0]},{page_size},{kv_heads},16] dtype={q.dtype} lengths={lengths}")
    print(f"paged preallocated wrapper (no append, trusted metadata): {reuse_us:.2f} us")
    print(f"paged allocating output each call: {alloc_us:.2f} us")
    print(f"dense attention on pre-gathered KV: {dense_us:.2f} us")
    print(f"dense gather + attention (Python gather and allocation): {gather_us:.2f} us")
    print("Timing is a tiny demonstration with explicit boundaries, not a production performance claim.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--seq-len", type=int, default=9)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    if torch is None or not torch.cuda.is_available():
        print("SKIP: CUDA/PyTorch is unavailable; no GPU kernel was run")
        return
    device = torch.device("cuda")
    for dim in (16, 32, 64, 128):
        for q_heads, kv_heads in ((2, 2), (4, 2), (4, 1)):
            correctness_case(device, 5, q_heads, kv_heads, dim, [1, 4, 5, 33, 65], dim + q_heads)
    if args.benchmark:
        if args.seq_len <= 0 or args.batch <= 0 or args.heads <= 0:
            raise ValueError("benchmark dimensions must be positive")
        benchmark(device, args.batch, args.heads, args.seq_len)


if __name__ == "__main__":
    main()
