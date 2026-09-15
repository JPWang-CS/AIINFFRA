"""CPU semantic checks for a simple, unswizzled tensor-map layout.

This is a small Python model for layout, tile coverage, and logical boundary
behavior. It is not a GPU or TMA emulator.

Run with the standard library only::

    python tensor_map_checks.py
"""

from __future__ import annotations

import unittest
from collections import Counter
from typing import Iterator, NamedTuple, Sequence


TILE_ROWS = 8
TILE_COLS = 32
ELEMENT_BYTES = 4  # int32


class TensorMapLayout(NamedTuple):
    rows: int
    cols: int
    pitch: int
    global_dim: tuple[int, int]
    stride_bytes: int


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def layout(rows: int, cols: int) -> TensorMapLayout:
    """Describe row-major int32 storage with each row padded to 16 bytes.

    ``pitch`` is measured in int32 elements. Tensor-map dimensions are given
    fastest-first, so ``global_dim`` is ``[cols, rows]``. The byte stride is
    the physical distance between row starts.
    """
    _require_positive_int("rows", rows)
    _require_positive_int("cols", cols)
    pitch = ((cols + 3) // 4) * 4
    return TensorMapLayout(
        rows=rows,
        cols=cols,
        pitch=pitch,
        global_dim=(cols, rows),
        stride_bytes=pitch * ELEMENT_BYTES,
    )


def tile_coords(
    rows: int,
    cols: int,
    tile_rows: int = TILE_ROWS,
    tile_cols: int = TILE_COLS,
) -> Iterator[tuple[int, int, bool]]:
    """Yield every tile-local coordinate as ``(row, col, is_logically_valid)``.

    Coordinates are global logical coordinates, including out-of-bounds
    coordinates in edge tiles. Tiles and their local elements are traversed
    in row-major order.
    """
    _require_positive_int("rows", rows)
    _require_positive_int("cols", cols)
    _require_positive_int("tile_rows", tile_rows)
    _require_positive_int("tile_cols", tile_cols)
    for tile_row in range(0, rows, tile_rows):
        for tile_col in range(0, cols, tile_cols):
            for local_row in range(tile_rows):
                for local_col in range(tile_cols):
                    row = tile_row + local_row
                    col = tile_col + local_col
                    yield row, col, row < rows and col < cols


def model_gather(
    physical_input: Sequence[int], rows: int, cols: int, bm: int, bn: int
) -> list[list[int]]:
    """Gather an 8x32 tile from flat, row-pitched int32 storage.

    ``bm`` and ``bn`` are the tile's logical top-left row and column. Storage
    must contain exactly ``rows * pitch`` elements, including physical column
    padding. Coordinates outside the logical ``rows x cols`` domain produce
    zero, even if the corresponding physical padding contains another value.
    No swizzle or device behavior is modeled.
    """
    desc = layout(rows, cols)
    if isinstance(bm, bool) or not isinstance(bm, int) or bm < 0:
        raise ValueError("bm must be a non-negative integer")
    if isinstance(bn, bool) or not isinstance(bn, int) or bn < 0:
        raise ValueError("bn must be a non-negative integer")
    if len(physical_input) != rows * desc.pitch:
        raise ValueError(
            f"physical_input must contain exactly {rows * desc.pitch} elements"
        )

    tile: list[list[int]] = []
    for local_row in range(TILE_ROWS):
        row = bm + local_row
        values: list[int] = []
        for local_col in range(TILE_COLS):
            col = bn + local_col
            if row < rows and col < cols:
                values.append(physical_input[row * desc.pitch + col])
            else:
                values.append(0)
        tile.append(values)
    return tile


class TensorMapChecks(unittest.TestCase):
    def test_layout_rounding_and_validation(self) -> None:
        for rows, cols, expected_pitch in (
            (1, 1, 4),
            (8, 32, 32),
            (9, 33, 36),
            (17, 65, 68),
        ):
            with self.subTest(rows=rows, cols=cols):
                desc = layout(rows, cols)
                self.assertEqual(desc.pitch, expected_pitch)
                self.assertEqual(desc.stride_bytes, expected_pitch * 4)
                self.assertEqual(desc.stride_bytes % 16, 0)
                self.assertEqual(desc.global_dim, (cols, rows))

        for bad_rows, bad_cols in ((0, 1), (1, 0), (-1, 2), (True, 2), (2, 1.5)):
            with self.subTest(rows=bad_rows, cols=bad_cols):
                with self.assertRaises(ValueError):
                    layout(bad_rows, bad_cols)

    def test_tile_coords_cover_each_valid_coordinate_exactly_once(self) -> None:
        for rows, cols in ((1, 1), (8, 32), (9, 33), (17, 65)):
            with self.subTest(rows=rows, cols=cols):
                coords = list(tile_coords(rows, cols))
                valid = [(row, col) for row, col, is_valid in coords if is_valid]
                expected = [(row, col) for row in range(rows) for col in range(cols)]
                self.assertEqual(Counter(valid), Counter(expected))
                self.assertEqual(len(valid), rows * cols)

    def test_gather_returns_fixed_tile_and_zeroes_logical_oob(self) -> None:
        rows, cols = 9, 33
        desc = layout(rows, cols)
        physical = [-77] * (rows * desc.pitch)
        for row in range(rows):
            for col in range(cols):
                physical[row * desc.pitch + col] = row * 1000 + col

        tile = model_gather(physical, rows, cols, bm=8, bn=32)
        self.assertEqual(len(tile), 8)
        self.assertTrue(all(len(row) == 32 for row in tile))
        self.assertEqual(tile[0][0], 8032)
        self.assertEqual(tile[0][1], 0)  # Logical col 33, not padded -77.
        self.assertTrue(all(value == 0 for value in tile[1][0:]))

        # A logically valid last-column value survives despite adjacent poison.
        previous_tile = model_gather(physical, rows, cols, bm=8, bn=0)
        self.assertEqual(previous_tile[0][31], 8031)  # This tile ends at col 31.
        edge_tile = model_gather(physical, rows, cols, bm=0, bn=32)
        self.assertEqual(edge_tile[0][0], 32)
        self.assertEqual(edge_tile[0][1], 0)

    def test_gather_matches_every_local_element_for_all_shapes(self) -> None:
        for rows, cols in ((1, 1), (8, 32), (9, 33), (17, 65)):
            with self.subTest(rows=rows, cols=cols):
                desc = layout(rows, cols)
                physical = [-77] * (rows * desc.pitch)
                for row in range(rows):
                    for col in range(cols):
                        physical[row * desc.pitch + col] = 1000 * row + col + 1

                for bm in range(0, rows, TILE_ROWS):
                    for bn in range(0, cols, TILE_COLS):
                        tile = model_gather(physical, rows, cols, bm=bm, bn=bn)
                        self.assertEqual(len(tile), TILE_ROWS)
                        for local_row in range(TILE_ROWS):
                            self.assertEqual(len(tile[local_row]), TILE_COLS)
                            for local_col in range(TILE_COLS):
                                row = bm + local_row
                                col = bn + local_col
                                expected = (
                                    1000 * row + col + 1
                                    if row < rows and col < cols
                                    else 0
                                )
                                self.assertEqual(
                                    tile[local_row][local_col],
                                    expected,
                                    msg=(
                                        f"shape={rows}x{cols}, tile=({bm},{bn}), "
                                        f"local=({local_row},{local_col})"
                                    ),
                                )

    def test_wrong_byte_stride_used_as_element_stride_is_wrong(self) -> None:
        desc = layout(2, 5)
        physical = [10, 11, 12, 13, 14, -77, -77, -77,
                    20, 21, 22, 23, 24, -77, -77, -77]
        correct_row1_col0 = physical[desc.pitch + 0]
        wrong_element_index = desc.stride_bytes + 0
        self.assertEqual(desc.pitch, 8)  # Element units.
        self.assertEqual(desc.stride_bytes, 32)  # Byte units.
        self.assertEqual(correct_row1_col0, 20)
        self.assertGreaterEqual(wrong_element_index, len(physical))

    def test_missing_edge_tiles_are_detected(self) -> None:
        rows, cols = 17, 65
        expected = rows * cols
        only_first_tile = [
            (row, col)
            for row, col, valid in tile_coords(rows, cols)
            if valid and row < TILE_ROWS and col < TILE_COLS
        ]
        complete = [
            (row, col)
            for row, col, valid in tile_coords(rows, cols)
            if valid
        ]
        self.assertEqual(len(only_first_tile), TILE_ROWS * TILE_COLS)
        self.assertNotEqual(len(only_first_tile), expected)
        self.assertEqual(len(complete), expected)


if __name__ == "__main__":
    unittest.main()
