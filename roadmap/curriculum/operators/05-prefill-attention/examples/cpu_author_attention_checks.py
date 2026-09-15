"""CPU-only checks for stride, ownership, masks, recurrence, and lifetimes.

The tests model address and dependency contracts discussed alongside FA2/FA3
source. They do not emulate a CUDA warp, TMA, WGMMA, shared-memory bank, or
performance behavior.
"""

from math import exp, isclose
import unittest


def ceil_div(x, y):
    return (x + y - 1) // y


def offset4(index, strides):
    if len(index) != 4 or len(strides) != 4:
        raise ValueError("rank-4 coordinates and strides are required")
    if any(type(i) is not int or i < 0 for i in index):
        raise ValueError("indices must be non-negative integers")
    if any(type(s) is not int or s < 0 for s in strides):
        raise ValueError("this model accepts non-negative element strides")
    return sum(i * s for i, s in zip(index, strides))


def q_tile_owner(batch, head, query_row, block_m):
    if block_m <= 0 or query_row < 0:
        raise ValueError("block_m must be positive and query row non-negative")
    return batch, head, query_row // block_m


def bottom_right_causal_valid(q_local, k_local, q_len, k_len):
    """FA2/FA3 unequal-length causal alignment, with bounds included."""
    return (0 <= q_local < q_len and 0 <= k_local < k_len
            and k_local <= q_local + (k_len - q_len))


def online_weighted_sum(scores, values, valid, block_n):
    """Return output and (m,l,u) using blockwise stable natural-exp updates."""
    if len(scores) != len(values) or len(scores) != len(valid):
        raise ValueError("scores, values, and mask lengths must agree")
    if block_n <= 0:
        raise ValueError("block_n must be positive")
    width = len(values[0]) if values else 0
    m = float("-inf")
    l = 0.0
    u = [0.0] * width
    for start in range(0, len(scores), block_n):
        ids = [i for i in range(start, min(start + block_n, len(scores))) if valid[i]]
        if not ids:
            continue
        block_m = max(scores[i] for i in ids)
        m_new = max(m, block_m)
        alpha = 0.0 if m == float("-inf") else exp(m - m_new)
        p = {i: exp(scores[i] - m_new) for i in ids}
        u = [alpha * old + sum(p[i] * values[i][d] for i in ids)
             for d, old in enumerate(u)]
        l = alpha * l + sum(p.values())
        m = m_new
    out = [value / l for value in u] if l else [0.0] * width
    return out, (m, l, u)


def dense_weighted_sum(scores, values, valid):
    ids = [i for i, keep in enumerate(valid) if keep]
    if not ids:
        return [0.0] * (len(values[0]) if values else 0)
    m = max(scores[i] for i in ids)
    weights = {i: exp(scores[i] - m) for i in ids}
    denom = sum(weights.values())
    return [sum(weights[i] * values[i][d] for i in ids) / denom
            for d in range(len(values[0]))]


EVENT_ORDER = {"load": 0, "qk_complete": 1, "softmax_complete": 2, "pv_complete": 3}
LAST_USE = {"K": "qk_complete", "P": "pv_complete", "V": "pv_complete"}


def require_release_after_last_use(buffer_name, release_event):
    if buffer_name not in LAST_USE or release_event not in EVENT_ORDER:
        raise ValueError("unknown buffer or lifecycle event")
    if EVENT_ORDER[release_event] < EVENT_ORDER[LAST_USE[buffer_name]]:
        raise RuntimeError(f"{buffer_name} is released before its last reader")


class AuthorAttentionCpuChecks(unittest.TestCase):
    def test_contiguous_and_transposed_views_use_their_own_strides(self):
        batch, seq, heads, dim = 2, 3, 4, 5
        bshd = (seq * heads * dim, heads * dim, dim, 1)
        bhsd_contiguous = (heads * seq * dim, seq * dim, dim, 1)
        bhsd_transposed_view = (seq * heads * dim, dim, heads * dim, 1)
        for b in range(batch):
            for h in range(heads):
                for s in range(seq):
                    for d in range(dim):
                        from_view = offset4((b, h, s, d), bhsd_transposed_view)
                        in_source = offset4((b, s, h, d), bshd)
                        self.assertEqual(from_view, in_source)
                        self.assertEqual(
                            offset4((b, h, s, d), bhsd_contiguous),
                            (((b * heads + h) * seq + s) * dim + d),
                        )

    def test_ragged_query_rows_have_exactly_one_cta_owner(self):
        block_m = 16
        for seq in (1, 3, 17, 33, 65):
            owners = {}
            for b in range(2):
                for h in range(3):
                    for q0 in range(0, seq, block_m):
                        owner = (b, h, q0 // block_m)
                        for q in range(q0, min(q0 + block_m, seq)):
                            key = (b, h, q)
                            self.assertNotIn(key, owners)
                            owners[key] = owner
            self.assertEqual(len(owners), 2 * 3 * seq)
            self.assertTrue(all(owner == q_tile_owner(*key, block_m)
                                for key, owner in owners.items()))

    def test_causal_mask_and_kv_tail_are_applied_to_scores(self):
        for seq in (1, 3, 5, 33, 65):
            block_m, block_n = 2, 4
            for q0 in range(0, seq, block_m):
                for q in range(q0, min(q0 + block_m, seq)):
                    allowed = [k for k in range(seq) if k <= q]
                    observed = []
                    for n0 in range(0, seq, block_n):
                        for k in range(n0, n0 + block_n):
                            key_valid = k < seq
                            causal_valid = key_valid and k <= q
                            if causal_valid:
                                observed.append(k)
                    self.assertEqual(observed, allowed)
                    self.assertTrue(all(k < seq for k in observed))

    def test_unequal_sequence_causal_mask_is_bottom_right_aligned(self):
        self.assertEqual(
            [[k for k in range(5) if bottom_right_causal_valid(q, k, 2, 5)]
             for q in range(2)],
            [[0, 1, 2, 3], [0, 1, 2, 3, 4]],
        )
        self.assertEqual(
            [[k for k in range(2) if bottom_right_causal_valid(q, k, 5, 2)]
             for q in range(5)],
            [[], [], [], [0], [0, 1]],
        )

    def test_online_recurrence_matches_dense_for_tail_and_causal_rows(self):
        scores = [-2.0, 0.5, 3.0, -1.0, 1.5]
        values = [[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0], [2.0, 3.0], [5.0, -1.0]]
        for query_row in range(len(scores)):
            valid = [key <= query_row for key in range(len(scores))]
            reference = dense_weighted_sum(scores, values, valid)
            for block_n in (1, 2, 4, 8):
                got, (m, l, u) = online_weighted_sum(scores, values, valid, block_n)
                self.assertGreater(l, 0.0)
                self.assertAlmostEqual(m, max(scores[:query_row + 1]))
                self.assertTrue(all(isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
                                    for x, y in zip(got, reference)))
                self.assertTrue(all(isclose(got[d], u[d] / l, rel_tol=1e-12, abs_tol=1e-12)
                                    for d in range(len(got))))

    def test_fully_masked_row_returns_zero_without_nan(self):
        out, (m, l, u) = online_weighted_sum(
            [1.0, 2.0], [[4.0], [8.0]], [False, False], 1
        )
        self.assertEqual(out, [0.0])
        self.assertEqual(m, float("-inf"))
        self.assertEqual(l, 0.0)
        self.assertEqual(u, [0.0])

    def test_k_p_and_v_cannot_be_released_before_last_reader(self):
        require_release_after_last_use("K", "qk_complete")
        require_release_after_last_use("P", "pv_complete")
        require_release_after_last_use("V", "pv_complete")
        for buffer_name, early in (("K", "load"), ("P", "softmax_complete"),
                                   ("V", "qk_complete")):
            with self.assertRaises(RuntimeError):
                require_release_after_last_use(buffer_name, early)

    def test_invalid_rank_and_negative_stride_are_rejected(self):
        with self.assertRaises(ValueError):
            offset4((0, 0, 1), (1, 1, 1, 1))
        with self.assertRaises(ValueError):
            offset4((0, 0, 0, 0), (1, 1, -1, 1))


if __name__ == "__main__":
    unittest.main()
