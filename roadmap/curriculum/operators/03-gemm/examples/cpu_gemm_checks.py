"""纯标准库的 GEMM 语义与资源账本检查。

这个脚本不编译 CUDA、不导入 NumPy/PyTorch，也不生成缓存文件；它只把课程中的
索引、tile、mask、FLOPs 和理想输入复用公式变成可以重复运行的断言。
"""

from __future__ import annotations

import math
import struct
from typing import Iterable


def ceil_div(n: int, tile: int) -> int:
    if n < 0 or tile <= 0:
        raise ValueError("n must be non-negative and tile must be positive")
    return (n + tile - 1) // tile


def canonical_addresses(m: int, k: int, n_values: Iterable[int], n: int, k_extent: int) -> tuple[list[int], list[int], int]:
    """A[M,N] @ B[N,K] = C[M,K]，返回一个输出元素的三组元素偏移。"""
    indices = tuple(n_values)
    a = [m * n + n_index for n_index in indices]
    b = [n_index * k_extent + k for n_index in indices]
    c = m * k_extent + k
    return a, b, c


def conventional_addresses(m: int, n: int, k_values: Iterable[int], k_extent: int, n_extent: int) -> tuple[list[int], list[int], int]:
    """A[M,K] @ B[K,N] = C[M,N]，归约轴改名为 K。"""
    indices = tuple(k_values)
    a = [m * k_extent + k_index for k_index in indices]
    b = [k_index * n_extent + n for k_index in indices]
    c = m * n_extent + n
    return a, b, c


def output_store_counts(m_extent: int, k_extent: int, bm: int, bk: int) -> dict[tuple[int, int], int]:
    """模拟二维 program grid 的 store mask，返回所有有效 C 坐标。"""
    counts: dict[tuple[int, int], int] = {}
    for pid_m in range(ceil_div(m_extent, bm)):
        for pid_k in range(ceil_div(k_extent, bk)):
            for m in range(pid_m * bm, (pid_m + 1) * bm):
                for k in range(pid_k * bk, (pid_k + 1) * bk):
                    if m < m_extent and k < k_extent:
                        coordinate = (m, k)
                        counts[coordinate] = counts.get(coordinate, 0) + 1
    return counts


def load_coverage(m_extent: int, n_extent: int, k_extent: int, bm: int, bn: int, bk: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """模拟每个 output tile 在所有归约 tile 上的 A/B load mask。"""
    a_seen: set[tuple[int, int]] = set()
    b_seen: set[tuple[int, int]] = set()
    for pid_m in range(ceil_div(m_extent, bm)):
        for pid_k in range(ceil_div(k_extent, bk)):
            for n0 in range(0, n_extent, bn):
                for m in range(pid_m * bm, (pid_m + 1) * bm):
                    for n in range(n0, min(n0 + bn, n_extent)):
                        if m < m_extent:
                            a_seen.add((m, n))
                for n in range(n0, min(n0 + bn, n_extent)):
                    for k in range(pid_k * bk, (pid_k + 1) * bk):
                        if k < k_extent:
                            b_seen.add((n, k))
    return a_seen, b_seen


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ordered_f32_sum(values: Iterable[float]) -> float:
    total = f32(0.0)
    for value in values:
        total = f32(total + f32(value))
    return total


def pairwise_f32_sum(values: list[float]) -> float:
    current = [f32(value) for value in values]
    while len(current) > 1:
        current = [f32(current[i] + current[i + 1]) for i in range(0, len(current) - 1, 2)] + current[len(current) - (len(current) % 2):]
    return current[0] if current else f32(0.0)


def split_k_partials(a, b, bm: int, bk: int, bn: int, parts: int):
    """CPU schedule model for split-K: split the reduction N, not output K."""
    m_extent, n_extent = len(a), len(a[0])
    assert len(b) == n_extent and len(b[0])
    k_extent = len(b[0])
    partials = []
    for part in range(parts):
        n0 = (n_extent * part) // parts
        n1 = (n_extent * (part + 1)) // parts
        tiles = {}
        for tm in range(ceil_div(m_extent, bm)):
            for tk in range(ceil_div(k_extent, bk)):
                tile = [[0.0 for _ in range(bk)] for _ in range(bm)]
                for mi in range(bm):
                    for ki in range(bk):
                        m, k = tm * bm + mi, tk * bk + ki
                        if m < m_extent and k < k_extent:
                            tile[mi][ki] = sum(a[m][n] * b[n][k] for n in range(n0, n1))
                tiles[(tm, tk)] = tile
        partials.append(tiles)
    return partials


def epilogue_once(partials, m_extent, k_extent, bm, bk, beta=0.0, old_c=None, bias=None, activation=None):
    """Merge split-K partials, then apply beta/bias/activation exactly once."""
    out = [[0.0 for _ in range(k_extent)] for _ in range(m_extent)]
    # The explicit extents support multiple output tiles and a ragged tail;
    # the index formula is the point, not a GPU performance estimate.
    for m in range(m_extent):
        for k in range(k_extent):
            value = sum(part[(m // bm, k // bk)][m % bm][k % bk] for part in partials)
            if beta:
                value += beta * old_c[m][k]
            if bias is not None:
                value += bias[k]
            out[m][k] = activation(value) if activation else value
    return out


def persistent_tile_ids(num_tiles: int, num_programs: int):
    if num_tiles < 0 or num_programs <= 0:
        raise ValueError("invalid persistent schedule")
    return [(pid, tile_id) for pid in range(num_programs) for tile_id in range(pid, num_tiles, num_programs)]


def grouped_tile_map(groups, bm: int, bk: int):
    """Prefix-sum dispatch for ragged groups; each group owns its tile range."""
    prefix, mapping = 0, []
    for group_id, (m_extent, k_extent) in enumerate(groups):
        count = ceil_div(m_extent, bm) * ceil_div(k_extent, bk)
        for local in range(count):
            mapping.append((prefix + local, group_id, local))
        prefix += count
    return mapping


def main() -> None:
    # 1) 两套符号系统必须给出同一组物理地址：m=1,k=2 的归约元素是 0 和 1。
    a, b, c = canonical_addresses(1, 2, (index for index in (0, 1)), 2, 5)
    assert (a, b, c) == ([2, 3], [2, 7], 7)
    a2, b2, c2 = conventional_addresses(1, 2, (index for index in (0, 1)), 2, 5)
    assert (a2, b2, c2) == ([2, 3], [2, 7], 7)

    # 2) 非整除形状：M=65,N=33,K=67，BM=64,BN=32,BK=64。
    m_extent, n_extent, k_extent = 65, 33, 67
    bm, bn, bk = 64, 32, 64
    assert (ceil_div(m_extent, bm), ceil_div(k_extent, bk)) == (2, 2)
    assert ceil_div(n_extent, bn) == 2
    store_counts = output_store_counts(m_extent, k_extent, bm, bk)
    assert len(store_counts) == m_extent * k_extent
    assert all(count == 1 for count in store_counts.values())
    a_seen, b_seen = load_coverage(m_extent, n_extent, k_extent, bm, bn, bk)
    assert len(a_seen) == m_extent * n_extent
    assert len(b_seen) == n_extent * k_extent

    # 3) 算法量与理想输入复用 AI（不计 C/out 与 beta 的 old-C 读取）。
    flops = 2 * m_extent * n_extent * k_extent
    assert flops == 287430
    element_bytes = 4
    input_bytes_per_tile = element_bytes * bn * (bm + bk)
    tile_flops = 2 * bm * bn * bk
    reuse_ai = tile_flops / input_bytes_per_tile
    assert tile_flops == 262144
    assert input_bytes_per_tile == 16384
    assert math.isclose(reuse_ai, 16.0)
    assert math.isclose((2 * bm * 64 * bk) / (element_bytes * 64 * (bm + bk)), 16.0)
    assert ceil_div(n_extent, bn) == 2 and ceil_div(n_extent, 64) == 1
    large_ai = (2 * 128 * 32 * 256) / (4 * 32 * (128 + 256))
    assert math.isclose(large_ai, 42.666666666666664)
    assert 4 * 128 * 256 == 131072

    # 5) split-K actually partitions N; partials are merged before one epilogue.
    aa = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    bb = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    partial = split_k_partials(aa, bb, bm=2, bk=2, bn=1, parts=2)
    got = epilogue_once(partial, 2, 2, 2, 2, bias=[10.0, 20.0], activation=lambda x: max(x, 0.0))
    assert got == [[32.0, 48.0], [59.0, 84.0]]

    # Multiple output tiles and a real tail: M=3,N=5,K=7,BM=2,BK=4,3 split-N parts.
    m, n, k, bm, bk = 3, 5, 7, 2, 4
    aa = [[float(1 + row * n + col) for col in range(n)] for row in range(m)]
    bb = [[float(1 + row * k + col) for col in range(k)] for row in range(n)]
    partial = split_k_partials(aa, bb, bm=bm, bk=bk, bn=2, parts=3)
    old = [[float(row * k + col) for col in range(k)] for row in range(m)]
    bias = [float(col) for col in range(k)]
    got = epilogue_once(partial, m, k, bm, bk, beta=0.25, old_c=old, bias=bias, activation=lambda x: x * 2.0)
    expected = []
    for row in range(m):
        expected.append([2.0 * (sum(aa[row][r] * bb[r][col] for r in range(n)) + 0.25 * old[row][col] + bias[col]) for col in range(k)])
    assert got == expected
    nan_old = [[float("nan") for _ in range(k)] for _ in range(m)]
    assert epilogue_once(partial, m, k, bm, bk, beta=0.0, old_c=nan_old) == [[sum(aa[row][r] * bb[r][col] for r in range(n)) for col in range(k)] for row in range(m)]

    # 6) Persistent mapping revisits tiles with tile_id=pid+k*num_programs;
    # it changes work assignment, not the number of kernel launches by itself.
    ids = persistent_tile_ids(7, 3)
    assert sorted(tile for _, tile in ids) == list(range(7))
    assert ids == [(0, 0), (0, 3), (0, 6), (1, 1), (1, 4), (2, 2), (2, 5)]

    # 7) Grouped prefix sums use each ragged group's own ceil(M/BM)*ceil(K/BK).
    mapping = grouped_tile_map([(65, 67), (1, 129), (33, 33)], 64, 64)
    assert [(g, local) for _, g, local in mapping] == [(0, i) for i in range(4)] + [(1, 0), (1, 1), (1, 2), (2, 0)]
    assert [tile for tile, _, _ in mapping] == list(range(8))

    # 4) FP32 累加顺序属于语义上的数值边界，不把不同树形归约混成同一个结果。
    values = [1.0e8, 1.0, -1.0e8, 1.0]
    left_to_right = ordered_f32_sum(values)
    pairwise = pairwise_f32_sum(values)
    assert left_to_right != pairwise

    print("PASS: canonical/conventional addresses, grid=(2,2), reduction_tiles=2, masks, FLOPs=", flops)
    print(f"PASS: tile FLOPs={tile_flops}, FP32 input bytes/tile={input_bytes_per_tile}, ideal input reuse AI={reuse_ai:.6f}")
    print(f"PASS: BN=64 keeps ideal input AI=16.000000 but reduces loops to {ceil_div(n_extent, 64)}")
    print("PASS: BM=128,BK=256,FP32 ideal input AI=42.666667, logical acc bytes=131072")
    print(f"NOTE: FP32 reduction order differs in CPU model ({left_to_right} vs {pairwise})")
    print("PASS: split-N partial merge/epilogue, persistent tile-id stride, ragged grouped prefix sum")


if __name__ == "__main__":
    main()
