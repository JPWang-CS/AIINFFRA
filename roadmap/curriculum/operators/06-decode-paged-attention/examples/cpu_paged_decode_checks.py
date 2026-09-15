"""stdlib-only Decode/PagedAttention model and correctness checks."""

from __future__ import annotations

import math
import random


def gqa_head(q_head, q_heads, kv_heads):
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("q_heads must be a positive multiple of kv_heads")
    return q_head // (q_heads // kv_heads)


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def dense_attention(q, k, v, lengths, q_heads, kv_heads):
    """Independent dense reference; K/V are [B,S,Hkv,D]."""
    out, scale = [], 1.0 / math.sqrt(len(q[0][0]))
    for b, length in enumerate(lengths):
        batch = []
        for hq in range(q_heads):
            hk = gqa_head(hq, q_heads, kv_heads)
            scores = [dot(q[b][hq], k[b][t][hk]) * scale for t in range(length)]
            m = max(scores)
            weights = [math.exp(x - m) for x in scores]
            denom = sum(weights)
            batch.append([sum(weights[t] * v[b][t][hk][d] for t in range(length)) / denom
                          for d in range(len(q[b][hq]))])
        out.append(batch)
    return out


def iter_paged_rows(k_cache, v_cache, table, length, page_size, access_log):
    """Read only valid logical rows, checking the physical address first."""
    for logical in range(length):
        page, offset = divmod(logical, page_size)
        if page >= len(table):
            raise AssertionError("logical page exceeds table")
        physical = int(table[page])
        if not 0 <= physical < len(k_cache):
            raise AssertionError("invalid physical page was accessed")
        access_log.append((logical, page, offset, physical))
        yield k_cache[physical][offset], v_cache[physical][offset]


def paged_attention(case):
    q, k_cache, v_cache, tables, lengths, page_size, q_heads, kv_heads = case
    out, accesses, scale = [], [], 1.0 / math.sqrt(len(q[0][0]))
    for b, length in enumerate(lengths):
        rows = list(iter_paged_rows(k_cache, v_cache, tables[b], length, page_size, accesses))
        batch = []
        for hq in range(q_heads):
            hk = gqa_head(hq, q_heads, kv_heads)
            scores = [dot(q[b][hq], row[0][hk]) * scale for row in rows]
            m = max(scores)
            weights = [math.exp(x - m) for x in scores]
            denom = sum(weights)
            batch.append([sum(weights[t] * rows[t][1][hk][d] for t in range(length)) / denom
                          for d in range(len(q[b][hq]))])
        out.append(batch)
    return out, accesses


def state_for_rows(rows, q_row, hk, scale):
    if not rows:
        return float("-inf"), 0.0, [0.0] * len(q_row)
    scores = [dot(q_row, row[0][hk]) * scale for row in rows]
    m = max(scores)
    weights = [math.exp(score - m) for score in scores]
    return m, sum(weights), [sum(weights[t] * rows[t][1][hk][d] for t in range(len(rows)))
                             for d in range(len(q_row))]


def merge_states(left, right):
    """Exact online-softmax state merge for (m, l, U)."""
    lm, ll, lu = left
    rm, rl, ru = right
    m = max(lm, rm)
    if m == float("-inf"):
        return m, 0.0, [0.0] * len(lu)
    la = math.exp(lm - m) if lm != float("-inf") else 0.0
    ra = math.exp(rm - m) if rm != float("-inf") else 0.0
    return m, la * ll + ra * rl, [la * lu[d] + ra * ru[d] for d in range(len(lu))]


def split_attention(rows, q_row, hk, scale, splits):
    states = [state_for_rows(rows[begin:end], q_row, hk, scale) for begin, end in splits]
    state = states[0]
    for next_state in states[1:]:
        state = merge_states(state, next_state)
    return [x / state[1] for x in state[2]] if state[1] else [0.0] * len(q_row)


def close(left, right, tol=1e-10):
    if len(left) != len(right):
        return False
    return all(len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))
               for a, b in zip(left, right))


def make_case(seed=7):
    rng = random.Random(seed)
    batch, q_heads, kv_heads, dim = 3, 4, 2, 4
    page_size, max_pages, physical_pages = 3, 5, 17
    lengths = [1, 5, 10]
    q = [[[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(q_heads)]
         for _ in range(batch)]
    logical_k = [[[ [rng.uniform(-1, 1) for _ in range(dim)] for _ in range(kv_heads)]
                   for _ in range(lengths[b])] for b in range(batch)]
    logical_v = [[[ [rng.uniform(-1, 1) for _ in range(dim)] for _ in range(kv_heads)]
                   for _ in range(lengths[b])] for b in range(batch)]
    k_cache = [[[[0.0 for _ in range(dim)] for _ in range(kv_heads)]
                for _ in range(page_size)] for _ in range(physical_pages)]
    v_cache = [[[[0.0 for _ in range(dim)] for _ in range(kv_heads)]
                for _ in range(page_size)] for _ in range(physical_pages)]
    tables, available = [], list(range(physical_pages))
    for b, length in enumerate(lengths):
        needed = (length + page_size - 1) // page_size
        chosen = rng.sample(available, needed)
        available = [physical for physical in available if physical not in chosen]
        rng.shuffle(chosen)
        tables.append(chosen + [-1] * (max_pages - needed))
        for logical in range(length):
            page, offset = divmod(logical, page_size)
            physical = chosen[page]
            k_cache[physical][offset] = logical_k[b][logical]
            v_cache[physical][offset] = logical_v[b][logical]
    assert any(table[:2] != sorted(table[:2]) for table in tables if len(table) > 1)
    return q, k_cache, v_cache, tables, lengths, page_size, q_heads, kv_heads, logical_k, logical_v


def main():
    raw = make_case()
    q, k_cache, v_cache, tables, lengths, page_size, q_heads, kv_heads, k, v = raw
    case = (q, k_cache, v_cache, tables, lengths, page_size, q_heads, kv_heads)
    dense = dense_attention(q, k, v, lengths, q_heads, kv_heads)
    paged, accesses = paged_attention(case)
    assert all(close(dense[b], paged[b]) for b in range(len(lengths)))
    assert len(accesses) == sum(lengths)
    for b in range(len(lengths)):
        rows = list(zip(k[b], v[b]))
        scale = 1.0 / math.sqrt(len(q[b][0]))
        for hq in range(q_heads):
            n, hk = len(rows), gqa_head(hq, q_heads, kv_heads)
            splits = [(0, 0), (0, min(2, n)), (min(2, n), n), (n, n)]
            got = split_attention(rows, q[b][hq], hk, scale, splits)
            assert len(got) == len(dense[b][hq])
            assert all(abs(a - z) <= 1e-10 for a, z in zip(got, dense[b][hq]))
    assert split_attention([], [0.0] * 4, 0, 1.0, [(0, 0), (0, 0)]) == [0.0] * 4
    chunk_a = (0.0, 1.0, [2.0])
    chunk_b = (math.log(3.0), 1.0, [10.0])
    merged = merge_states(chunk_a, chunk_b)
    merged_output = merged[2][0] / merged[1]
    assert abs(merged_output - 8.0) <= 1e-12
    assert abs((2.0 + 10.0) / 2.0 - 6.0) <= 1e-12 and merged_output != 6.0
    for bad in (-1, len(k_cache)):
        try:
            list(iter_paged_rows(k_cache, v_cache, [bad], 1, page_size, []))
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid physical page did not fail")
    assert gqa_head(0, 4, 2) == 0 and gqa_head(3, 4, 2) == 1
    print("PASS: independent dense vs paged vs split-merge; scatter mapping; ragged lengths; random pages; GQA; empty segments; -1/>=P guards")


if __name__ == "__main__":
    main()
