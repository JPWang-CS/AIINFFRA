"""stdlib-only dense/tiled attention checks; no CUDA or GPU claim."""

from __future__ import annotations

import math
import random


def _shape(x):
    return len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])


def _heads(qh: int, kh: int):
    if qh % kh:
        raise ValueError("Hq must be divisible by Hkv")
    group = qh // kh
    return [h // group for h in range(qh)]


def dense_attention(q, k, v, *, causal=False, key_valid=None):
    B, Hq, S, D = _shape(q)
    Bk, Hkv, Sk, Dk = _shape(k)
    if (B, D) != (Bk, Dk) or _shape(v) != (B, Hkv, Sk, D):
        raise ValueError("incompatible B/H/S/D")
    mapping = _heads(Hq, Hkv)
    out = [[[[0.0 for _ in range(D)] for _ in range(S)] for _ in range(Hq)] for _ in range(B)]
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        for hq in range(Hq):
            hk = mapping[hq]
            for i in range(S):
                scores = []
                for j in range(Sk):
                    valid = key_valid is None or key_valid[b][hk][j]
                    valid = valid and (not causal or j <= i)
                    if not valid:
                        scores.append(float("-inf"))
                        continue
                    scores.append(sum(q[b][hq][i][d] * k[b][hk][j][d] for d in range(D)) * scale)
                finite = [x for x in scores if x != float("-inf")]
                if not finite:
                    continue
                m = max(finite)
                p = [0.0 if x == float("-inf") else math.exp(x - m) for x in scores]
                z = sum(p)
                for d in range(D):
                    out[b][hq][i][d] = sum(p[j] * v[b][hk][j][d] for j in range(Sk)) / z
    return out


def tiled_online_attention(q, k, v, *, block_kv=32, causal=False, key_valid=None):
    B, Hq, S, D = _shape(q)
    _, Hkv, Sk, _ = _shape(k)
    mapping = _heads(Hq, Hkv)
    out = [[[[0.0 for _ in range(D)] for _ in range(S)] for _ in range(Hq)] for _ in range(B)]
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        for hq in range(Hq):
            hk = mapping[hq]
            for i in range(S):
                m, l, u = float("-inf"), 0.0, [0.0] * D
                for start in range(0, Sk, block_kv):
                    # Model the GPU tile: compute a vector of scores first,
                    # reduce its block max, then merge one whole tile.
                    scores, values = [], []
                    for j in range(start, start + block_kv):
                        valid = j < Sk and (key_valid is None or key_valid[b][hk][j])
                        valid = valid and (not causal or j <= i)
                        if valid:
                            scores.append(sum(q[b][hq][i][d] * k[b][hk][j][d] for d in range(D)) * scale)
                            values.append(v[b][hk][j])
                        else:
                            scores.append(float("-inf"))
                            values.append([0.0] * D)
                    block_m = max(scores)
                    new_m = max(m, block_m)
                    safe_m = 0.0 if new_m == float("-inf") else new_m
                    alpha = math.exp(m - safe_m) if m != float("-inf") else 0.0
                    p = [0.0 if score == float("-inf") else math.exp(score - safe_m) for score in scores]
                    tile_u = [sum(p[j] * values[j][d] for j in range(block_kv)) for d in range(D)]
                    u = [alpha * old + tile_u[d] for d, old in enumerate(u)]
                    l = alpha * l + sum(p)
                    m = new_m
                if l > 0.0:
                    out[b][hq][i] = [x / l for x in u]
    return out


def make_data(B, H, S, D, seed=0):
    rng = random.Random(seed)
    return [[[[rng.uniform(-1, 1) for _ in range(D)] for _ in range(S)] for _ in range(H)] for _ in range(B)]


def close(a, b, tol=1e-10):
    def flat(x):
        if isinstance(x, list):
            for child in x:
                yield from flat(child)
        else:
            yield x
    left, right = list(flat(a)), list(flat(b))
    if len(left) != len(right) or not all(math.isfinite(x) for x in left + right):
        return False
    return max(abs(x - y) for x, y in zip(left, right)) <= tol


def main():
    for S in (1, 3, 5, 33, 65):
        q = make_data(2, 4, S, 4, S)
        k = make_data(2, 2, S, 4, S + 1)
        v = make_data(2, 2, S, 4, S + 2)
        for causal in (False, True):
            ref = dense_attention(q, k, v, causal=causal)
            for block in (1, 2, 3, 4):
                got = tiled_online_attention(q, k, v, block_kv=block, causal=causal)
                assert close(ref, got), (S, causal, block)
    q = make_data(1, 2, 3, 4)
    k = make_data(1, 1, 3, 4, 1)
    v = make_data(1, 1, 3, 4, 2)
    valid = [[[False, False, False]]]
    assert dense_attention(q, k, v, key_valid=valid) == tiled_online_attention(q, k, v, block_kv=2, key_valid=valid)
    assert _heads(4, 2) == [0, 0, 1, 1]
    # Tail-mask counterexample: padded K/V rows must not enter the denominator.
    q = [[[[0.0] * 4 for _ in range(3)] for _ in range(4)] for _ in range(2)]
    k = [[[[0.0] * 4 for _ in range(3)] for _ in range(2)] for _ in range(2)]
    v = [[[[float(10 + b * 100 + hk * 10)] * 4 for _ in range(3)] for hk in range(2)] for b in range(2)]
    expected = dense_attention(q, k, v)
    assert all(row == [10.0, 10.0, 10.0, 10.0] for row in expected[0][0])
    assert all(row == [20.0, 20.0, 20.0, 20.0] for row in expected[0][2])
    assert all(row == [110.0, 110.0, 110.0, 110.0] for row in expected[1][0])
    for block in (1, 2, 3, 4):
        assert close(expected, tiled_online_attention(q, k, v, block_kv=block))
    print("PASS: dense/tiled online, S=1/3/5/33/65 tails, B/H offsets, causal, all-masked zero state, GQA mapping")


if __name__ == "__main__":
    main()
