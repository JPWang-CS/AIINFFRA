"""Independent stdlib-only tests for the online m/l/U recurrence and tails."""

from __future__ import annotations

import math


def update_online(scores, values, valid, block_size):
    """Return final m, l and vector U after masked score/value tiles."""
    width = len(values[0]) if values else 0
    m = float("-inf")
    l = 0.0
    u = [0.0] * width
    for start in range(0, len(scores), block_size):
        tile_scores, tile_values, tile_valid = [], [], []
        for j in range(start, start + block_size):
            ok = j < len(scores) and valid[j]
            tile_valid.append(ok)
            tile_scores.append(scores[j] if ok else float("-inf"))
            tile_values.append(values[j] if ok else [0.0] * width)
        finite_scores = [score for score, ok in zip(tile_scores, tile_valid) if ok]
        block_m = max(finite_scores) if finite_scores else float("-inf")
        m_new = max(m, block_m)
        safe_m = 0.0 if m_new == float("-inf") else m_new
        alpha = math.exp(m - safe_m) if m != float("-inf") else 0.0
        p = [math.exp(score - safe_m) if ok else 0.0
             for score, ok in zip(tile_scores, tile_valid)]
        u = [alpha * old + sum(p[j] * tile_values[j][d] for j in range(len(p)))
             for d, old in enumerate(u)]
        l = alpha * l + sum(p)
        m = m_new
    return m, l, u


def dense_state(scores, values, valid):
    kept = [i for i, ok in enumerate(valid) if ok]
    if not kept:
        return float("-inf"), 0.0, [0.0] * len(values[0])
    m = max(scores[i] for i in kept)
    weights = [math.exp(scores[i] - m) for i in kept]
    l = sum(weights)
    u = [sum(weights[j] * values[i][d] for j, i in enumerate(kept))
         for d in range(len(values[0]))]
    return m, l, u


def assert_close(left, right, tol=1e-12):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert abs(a - b) <= tol * max(1.0, abs(a), abs(b)), (a, b)


def check_numeric_recurrence():
    scores = [0.0, math.log(2.0), math.log(4.0)]
    values = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
    valid = [True, True, True]
    m, l, u = update_online(scores, values, valid, block_size=2)
    assert_close([m, l], [math.log(4.0), 1.75])
    assert_close(u, [6.75, 42.5])
    assert_close([x / l for x in u], [27.0 / 7.0, 170.0 / 7.0])


def check_tails_and_masks():
    for length in (1, 3, 5, 33, 65):
        scores = [math.sin(i * 0.37) * 4.0 for i in range(length)]
        values = [[math.cos(i * 0.19), float(i % 7 - 3), float(i) / 9.0]
                  for i in range(length)]
        for causal_end in (length - 1, max(0, length // 2)):
            valid = [i <= causal_end for i in range(length)]
            expected = dense_state(scores, values, valid)
            for block in (1, 2, 3, 4, 32):
                got = update_online(scores, values, valid, block)
                assert_close([got[0], got[1]], [expected[0], expected[1]])
                assert_close(got[2], expected[2])

    # Padded lanes contain adversarial data but are invalid, so they must not
    # affect either the running maximum or denominator/numerator.
    scores = [0.0, math.log(2.0), 900.0, -800.0]
    values = [[1.0], [3.0], [1e30], [-1e30]]
    valid = [True, True, False, False]
    reference = dense_state(scores, values, valid)
    for block in (1, 2, 3, 4, 8):
        got = update_online(scores, values, valid, block)
        assert_close([got[0], got[1]], [reference[0], reference[1]])
        assert_close(got[2], reference[2])

    # An all-masked row remains a defined zero state, matching the kernel guard.
    m, l, u = update_online([1.0, 2.0], [[7.0], [9.0]], [False, False], 2)
    assert m == float("-inf") and l == 0.0 and u == [0.0]


def main():
    check_numeric_recurrence()
    check_tails_and_masks()
    print("PASS: exact m/l/U recurrence, online-vs-dense, causal masks, partial tails, all-masked row")


if __name__ == "__main__":
    main()
