#!/usr/bin/env python3
"""CPU ownership model for the two-CTA DSM histogram example."""

from collections import Counter

CLUSTER_BLOCKS = 2
THREADS = 128
BINS = 32
BINS_PER_BLOCK = BINS // CLUSTER_BLOCKS


def generated_input(n):
    return [(i * 17 + (i // 7) * 3 + 1) % BINS for i in range(n)]


def emulate_cluster_ownership(values):
    n = len(values)
    blocks_needed = (n + THREADS - 1) // THREADS
    grid_blocks = ((blocks_needed + CLUSTER_BLOCKS - 1) // CLUSTER_BLOCKS) * CLUSTER_BLOCKS
    cluster_count = grid_blocks // CLUSTER_BLOCKS
    stride = cluster_count * CLUSTER_BLOCKS * THREADS

    # One private shared-memory array per CTA in each cluster.
    cluster_bins = [[0] * BINS for _ in range(cluster_count)]
    visits = [0] * n
    remote_updates = 0

    for cluster_rank in range(cluster_count):
        for block_rank in range(CLUSTER_BLOCKS):
            first = (cluster_rank * CLUSTER_BLOCKS * THREADS
                     + block_rank * THREADS)
            for lane in range(THREADS):
                for index in range(first + lane, n, stride):
                    visits[index] += 1
                    bin_id = values[index]
                    owner_rank = bin_id // BINS_PER_BLOCK
                    owner_offset = bin_id % BINS_PER_BLOCK
                    assert 0 <= owner_rank < CLUSTER_BLOCKS
                    assert 0 <= owner_offset < BINS_PER_BLOCK
                    assert owner_rank * BINS_PER_BLOCK + owner_offset == bin_id
                    if owner_rank != block_rank:
                        remote_updates += 1
                    cluster_bins[cluster_rank][bin_id] += 1

    assert visits == [1] * n, "each input element must have exactly one CTA owner"
    if n >= 127:
        assert remote_updates > 0, "test data must exercise cross-CTA DSM updates"

    actual = [sum(partial[bin_id] for partial in cluster_bins)
              for bin_id in range(BINS)]
    expected = [Counter(values)[bin_id] for bin_id in range(BINS)]
    assert actual == expected, (actual, expected)
    assert sum(actual) == n
    return grid_blocks, cluster_count, remote_updates


def main():
    # Includes a one-element minimum, CTA/cluster boundaries, and tails.
    for n in (1, 127, 128, 129, 255, 256, 257, 4099):
        blocks, clusters, remote_updates = emulate_cluster_ownership(generated_input(n))
        print(f"PASS n={n} gridBlocks={blocks} clusters={clusters} "
              f"remoteDSMUpdates={remote_updates}")


if __name__ == "__main__":
    main()
