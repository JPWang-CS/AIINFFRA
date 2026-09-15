"""CPU-only checks for the published CUTLASS 3.8.0 example-49 contracts.

This checks logical coordinates, row/column-major addresses, cluster output
ownership, and an abstract TMA/WGMMA stage protocol. It does not emulate CuTe
layouts, TMA, WGMMA, CUDA scheduling, or hardware performance.
"""

import unittest


CTA_M, CTA_N, CTA_K = 128, 128, 64
CLUSTER_M, CLUSTER_N = 2, 1
THREADS_PER_WARPGROUP = 4 * 32


def row_major_offset(row, col, leading_dim):
    return row * leading_dim + col


def column_major_offset(row, col, leading_dim):
    return row + col * leading_dim


def cluster_cta_output_tiles(cluster_m=CLUSTER_M, cluster_n=CLUSTER_N):
    """Return each CTA's distinct 128x128 C tile in one cluster tile."""
    return [
        (cta_m * CTA_M, cta_n * CTA_N)
        for cta_n in range(cluster_n)
        for cta_m in range(cluster_m)
    ]


def require_tma_ready(arrivals):
    """Reject consumer access until both modeled operand loads are complete."""
    if arrivals.get("A") is not True or arrivals.get("B") is not True:
        raise RuntimeError("A and B TMA transactions must both complete")


def require_release_safe(stage, outstanding_mmas):
    """Reject releasing a stage while a modeled MMA may still read it."""
    if any(inflight_stage == stage for _, inflight_stage in outstanding_mmas):
        raise RuntimeError("shared stage is still referenced by an outstanding MMA")


def simulate_stage_reuse(k_tiles, stages, max_mmas_in_flight=1):
    """Check ring-stage ownership and A+B readiness in an abstract schedule.

    Completions are ordered model events, not simulated GPU concurrency.
    """
    if k_tiles < 0 or stages < 1 or max_mmas_in_flight < 0:
        raise ValueError("invalid tile, stage, or in-flight count")

    slots = [None] * stages
    outstanding = []  # (k_tile, stage) with a WGMMA group still using the slot
    issued = []
    acquire_waits = 0

    def complete_oldest():
        tile, stage = outstanding.pop(0)
        if slots[stage] != tile:
            raise AssertionError("completion refers to a reused generation")
        require_release_safe(stage, outstanding)
        slots[stage] = None  # consumer release makes this generation reusable

    for tile in range(k_tiles):
        stage = tile % stages
        while slots[stage] is not None:
            # producer_acquire cannot overwrite a stage before consumer_release
            complete_oldest()
            acquire_waits += 1

        arrivals = {"A": False, "B": False}
        slots[stage] = tile
        arrivals["A"] = True
        arrivals["B"] = True
        require_tma_ready(arrivals)

        issued.append((tile, stage))
        outstanding.append((tile, stage))
        # Abstract warpgroup_wait<N>: retain at most N newest MMA groups.
        while len(outstanding) > max_mmas_in_flight:
            complete_oldest()

    while outstanding:
        complete_oldest()  # mma_tail: drain all groups and release their slots
    if any(slot is not None for slot in slots):
        raise AssertionError("pipeline tail left a shared stage live")
    return issued, acquire_waits


class Cutlass49CpuChecks(unittest.TestCase):
    def test_operand_and_output_layout_addresses(self):
        # A[M,K] row-major; B[K,N], C[M,N] column-major.
        a = [[1, 2], [3, 4]]
        b = [[5, 6, 7], [8, 9, 10]]
        m, n, k = 2, 3, 2
        a_flat = [0] * (m * k)
        b_flat = [0] * (k * n)
        for i in range(m):
            for r in range(k):
                a_flat[row_major_offset(i, r, k)] = a[i][r]
        for r in range(k):
            for j in range(n):
                b_flat[column_major_offset(r, j, k)] = b[r][j]

        got = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                got[i][j] = sum(
                    a_flat[row_major_offset(i, r, k)]
                    * b_flat[column_major_offset(r, j, k)]
                    for r in range(k)
                )
        self.assertEqual(got, [[21, 24, 27], [47, 54, 61]])
        c_column_major = [got[i][j] for j in range(n) for i in range(m)]
        self.assertEqual(c_column_major, [21, 47, 24, 54, 27, 61])

    def test_cluster_ctas_have_unique_output_ownership(self):
        tiles = cluster_cta_output_tiles()
        self.assertEqual(tiles, [(0, 0), (CTA_M, 0)])
        owned = set()
        for m0, n0 in tiles:
            for m in range(m0, m0 + CTA_M):
                for n in range(n0, n0 + CTA_N):
                    self.assertNotIn((m, n), owned)
                    owned.add((m, n))
        self.assertEqual(len(owned), CLUSTER_M * CTA_M * CLUSTER_N * CTA_N)

    def test_warpgroup_lanes_are_four_complete_warps(self):
        for group in range(3):
            threads = range(group * THREADS_PER_WARPGROUP,
                            (group + 1) * THREADS_PER_WARPGROUP)
            ownership = [(t // THREADS_PER_WARPGROUP,
                          t % THREADS_PER_WARPGROUP) for t in threads]
            self.assertEqual(len(set(ownership)), THREADS_PER_WARPGROUP)
            self.assertEqual(ownership[0][1], 0)
            self.assertEqual(ownership[-1][1], THREADS_PER_WARPGROUP - 1)

    def test_stage_generations_and_tail_release(self):
        for stages in (1, 2, 3, 4):
            issued, waits = simulate_stage_reuse(11, stages, 1)
            self.assertEqual([tile for tile, _ in issued], list(range(11)))
            self.assertEqual(
                [stage for _, stage in issued], [tile % stages for tile in range(11)]
            )
            self.assertGreaterEqual(waits, 0)

    def test_invalid_pipeline_configuration_is_rejected(self):
        for args in ((-1, 2, 1), (1, 0, 1), (1, 2, -1)):
            with self.assertRaises(ValueError):
                simulate_stage_reuse(*args)

    def test_incomplete_tma_pair_cannot_be_consumed(self):
        for arrivals in ({"A": True, "B": False}, {"A": False, "B": True}):
            with self.assertRaises(RuntimeError):
                require_tma_ready(arrivals)

    def test_stage_cannot_be_released_before_mma_completion(self):
        with self.assertRaises(RuntimeError):
            require_release_safe(0, [(4, 0)])
        require_release_safe(0, [(3, 1)])


if __name__ == "__main__":
    unittest.main()
