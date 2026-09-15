"""Coordinate contract for dense PTX mma.m16n8k16 f16, not CUDA WMMA ABI."""
from collections import Counter
import unittest


def mma16816_f16_coords(lane):
    if not isinstance(lane, int) or not 0 <= lane < 32:
        raise ValueError("lane must be an integer in [0,31]")
    group, member = divmod(lane, 4)
    a = [(group + 8 * ((i // 2) % 2),
          2 * member + i % 2 + 8 * (i // 4)) for i in range(8)]
    b = [(2 * member + i % 2 + 8 * (i // 2), group) for i in range(4)]
    c = [(group + 8 * (i // 2), 2 * member + i % 2) for i in range(4)]
    return {"A": a, "B": b, "C": c}


def register_tile_gemm(a, b, thread_rows=2, thread_cols=3):
    """A list-based outer-product microtile model; not a CUDA register layout."""
    m, n, k = len(a), len(b), len(b[0])
    if thread_rows <= 0 or thread_cols <= 0:
        raise ValueError("positive microtile dimensions required")
    if any(len(row) != n for row in a) or any(len(row) != k for row in b):
        raise ValueError("A[M,N] and B[N,K] required")
    c = [[0.0 for _ in range(k)] for _ in range(m)]
    for row0 in range(0, m, thread_rows):
        for col0 in range(0, k, thread_cols):
            acc = [[0.0] * thread_cols for _ in range(thread_rows)]
            for reduction in range(n):
                ar = [a[row0 + i][reduction] if row0 + i < m else 0.0
                      for i in range(thread_rows)]
                br = [b[reduction][col0 + j] if col0 + j < k else 0.0
                      for j in range(thread_cols)]
                for i in range(thread_rows):
                    for j in range(thread_cols):
                        acc[i][j] += ar[i] * br[j]
            for i in range(thread_rows):
                for j in range(thread_cols):
                    if row0 + i < m and col0 + j < k:
                        c[row0 + i][col0 + j] = acc[i][j]
    return c


class LayoutTests(unittest.TestCase):
    def test_exact_instruction_cover(self):
        for operand, rows, cols in (("A", 16, 16), ("B", 16, 8), ("C", 16, 8)):
            coordinates = [xy for lane in range(32) for xy in mma16816_f16_coords(lane)[operand]]
            self.assertEqual(Counter(coordinates), Counter((r, c) for r in range(rows) for c in range(cols)))

    def test_selected_lanes(self):
        self.assertEqual(mma16816_f16_coords(0)["C"], [(0, 0), (0, 1), (8, 0), (8, 1)])
        self.assertEqual(mma16816_f16_coords(31)["C"], [(7, 6), (7, 7), (15, 6), (15, 7)])
        self.assertEqual(len(mma16816_f16_coords(0)["A"]), 8)
        self.assertEqual(len(mma16816_f16_coords(0)["B"]), 4)
        with self.assertRaises(ValueError):
            mma16816_f16_coords(32)

    def test_outer_product_microtiles(self):
        for m, n, k in ((1, 1, 1), (3, 5, 7), (8, 3, 4)):
            a = [[((r * n + c) % 7 - 3) / 4 for c in range(n)] for r in range(m)]
            b = [[((r * k + c) % 5 - 2) / 4 for c in range(k)] for r in range(n)]
            expected = [[sum(a[r][t] * b[t][c] for t in range(n)) for c in range(k)] for r in range(m)]
            for tr, tc in ((1, 1), (2, 3), (4, 2)):
                self.assertEqual(register_tile_gemm(a, b, tr, tc), expected)


if __name__ == "__main__":
    unittest.main()
