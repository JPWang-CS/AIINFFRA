"""Mathematical teaching models, not DeepSeek kernels or a model reproduction."""
import numpy as np


def hierarchical_select(first_scores, later_scores, visible, block_size, blocks, topk):
    first = np.asarray(first_scores, dtype=np.float64)
    later = np.asarray(later_scores, dtype=np.float64)
    if first.ndim != 1 or later.shape != first.shape:
        raise ValueError("two equal-length score vectors required")
    if not (0 < visible <= len(first)) or min(block_size, blocks, topk) <= 0:
        raise ValueError("positive visible length and budgets required")
    if not np.isfinite(first[:visible]).all() or not np.isfinite(later[:visible]).all():
        raise ValueError("visible scores must be finite")
    nblocks = (visible + block_size - 1) // block_size
    padded = np.full(nblocks * block_size, -np.inf)
    padded[:visible] = first[:visible]
    block_scores = padded.reshape(nblocks, block_size).max(axis=1)
    chosen_blocks = np.argsort(-block_scores, kind="stable")[:blocks]
    candidates = (chosen_blocks[:, None] * block_size
                  + np.arange(block_size)[None, :]).reshape(-1)
    candidates = np.sort(candidates[candidates < visible])
    first_top = np.argsort(-first[:visible], kind="stable")[:topk]
    later_top = candidates[np.argsort(-later[candidates], kind="stable")[:topk]]
    return first_top, candidates, later_top


def prefix_survival(conditional_acceptance):
    p = np.asarray(conditional_acceptance, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("finite conditional probabilities in [0,1] required")
    return np.cumprod(p)


def choose_length(conditional_acceptance, cycle_ms):
    survival = prefix_survival(conditional_acceptance)
    cost = np.asarray(cycle_ms, dtype=np.float64)
    if cost.shape != (len(survival) + 1,) or not np.isfinite(cost).all() or np.any(cost <= 0):
        raise ValueError("positive measured cycle cost for k=0..K required")
    # Assumes one correction/bonus token per cycle, no EOS/budget truncation.
    committed = 1.0 + np.r_[0.0, np.cumsum(survival)]
    rate = committed / cost
    return int(np.argmax(rate)), committed, rate


def ngram_addresses(tokens, order, table_sizes):
    # Deliberately simple teaching hash; NOT the author's hashing algorithm.
    tokens = [int(t) for t in tokens]
    if order < 1 or len(tokens) < order or any(t < 0 for t in tokens):
        raise ValueError("nonnegative tokens and a complete n-gram required")
    ids = []
    for head, size in enumerate(table_sizes):
        if size <= 1:
            raise ValueError("table size must exceed one")
        value = head + 1
        for token in tokens[-order:]:
            value = (value * 257 + token + 1) % size
        ids.append(value)
    return ids


def sinkhorn_step(weight, gradient, momentum, beta=0.95, lr=1e-3,
                  gamma=0.18, steps=11, tau=1e-3, eps=1e-20):
    w, g, old_m = [np.asarray(x, dtype=np.float64)
                   for x in (weight, gradient, momentum)]
    if w.ndim != 2 or w.shape != g.shape or w.shape != old_m.shape:
        raise ValueError("W, G and M must have equal matrix shapes")
    if steps < 1 or steps % 2 != 1 or eps <= 0 or tau < 0 or not 0 <= beta < 1:
        raise ValueError("odd positive steps and valid numerical constants required")
    if not all(np.isfinite(x).all() for x in (w, g, old_m)):
        raise ValueError("finite inputs required")
    m = beta * old_m + (1 - beta) * g
    g_hat = beta * m + (1 - beta) * g
    rho = np.linalg.norm(g_hat, axis=1)
    active = rho > tau * rho.mean()
    u = g_hat.copy()
    u[~active] = 0.0
    for k in range(1, steps + 1):
        axis = 1 if k % 2 else 0
        u = u / (np.linalg.norm(u, axis=axis, keepdims=True) + eps)
    delta = np.sqrt(w.shape[1]) * u
    return w - gamma * lr * delta, m, delta, active


def effort_penalty(length, effort, k0=1.0, bmin=25.0, tau=25.0,
                   length_norm=1000.0, cap=4.0):
    if length < 0 or tau <= 0 or length_norm <= 0 or min(k0, cap) < 0:
        raise ValueError("invalid length or penalty configuration")
    k = k0 * np.exp(-(effort - bmin) / tau)
    return -min(cap, k * length / length_norm)


def fp4_block_bytes(elements, scale_group, scale_bytes=1):
    if elements <= 0 or scale_group <= 0 or scale_bytes <= 0:
        raise ValueError("positive element, group, and scale sizes required")
    if elements % 2 or elements % scale_group:
        raise ValueError("FP4 packing and scale groups must divide the block")
    return elements // 2 + (elements // scale_group) * scale_bytes


def cache_budget_example():
    main_bytes = fp4_block_bytes(512, 16)  # 288 bytes
    index_bytes = fp4_block_bytes(128, 32)  # 68 bytes
    return main_bytes, index_bytes, 2.5 * (main_bytes + index_bytes)


def check():
    a = [9, 8, 1, 0, 2]
    b = [0, 1, 8, 7, 100]
    first, pool, later = hierarchical_select(a, b, 5, 2, 1, 1)
    assert first.tolist() == [0] and pool.tolist() == [0, 1]
    assert later.tolist() == [1] and np.argmax(b) == 4  # Candidate miss.
    _, pool, later = hierarchical_select(a, b, 5, 2, 10, 10)
    assert set(pool) == set(range(5)) and later[0] == 4
    _, pool, _ = hierarchical_select(a, b, 3, 2, 10, 10)
    assert set(pool) == {0, 1, 2}  # Padding/future positions cannot enter.
    np.testing.assert_allclose(prefix_survival([.9, .8, .5]), [.9, .72, .36])
    best_light, committed, _ = choose_length([.9, .8, .5], [1., 1.1, 1.2, 1.3])
    best_busy, _, _ = choose_length([.9, .8, .5], [1., 2., 3., 4.])
    assert best_light == 3 and best_busy == 0
    np.testing.assert_allclose(committed, [1., 1.9, 2.62, 2.98])
    assert ngram_addresses([1, 2, 3], 2, [17, 19]) == ngram_addresses([9, 2, 3], 2, [17, 19])
    rng = np.random.default_rng(4)
    w = rng.normal(size=(8, 3)); g = rng.normal(size=w.shape); g[0] = 0
    new_w, m, delta, active = sinkhorn_step(w, g, np.zeros_like(w))
    assert not active[0] and np.all(delta[0] == 0)
    np.testing.assert_allclose(np.sqrt(np.mean(delta[active] ** 2, axis=1)), 1.)
    np.testing.assert_allclose(new_w, w - .18 * 1e-3 * delta)
    _, _, zero_delta, zero_active = sinkhorn_step(w, np.zeros_like(w), np.zeros_like(w))
    assert not zero_active.any() and np.all(zero_delta == 0)
    assert abs(effort_penalty(1000, 50) / effort_penalty(1000, 25) - np.exp(-1)) < 1e-12
    assert effort_penalty(100000, 25) == -4.0
    assert cache_budget_example() == (288, 68, 890.0)
    try:
        fp4_block_bytes(513, 16)
        raise AssertionError("odd FP4 block must be rejected")
    except ValueError:
        pass
    print("PASS: causal candidate pool and miss; conditional survival and load tradeoff; "
          "deterministic toy hash; Sinkhorn row RMS/zero mask; capped effort penalty; "
          "FP4 cache-byte accounting")


if __name__ == "__main__":
    check()
