#!/usr/bin/env python3
"""Check cuTile view coordinates and shape transforms without CUDA."""


def main() -> None:
    array = [[row * 4 + col for col in range(4)] for row in range(4)]

    def read_window(tile_row: int) -> list[list[int]]:
        origin_row = tile_row * 1  # traversal_steps[0]
        origin_col = 0 * 4
        return [
            array[row][origin_col:origin_col + 4]
            for row in range(origin_row, origin_row + 2)
        ]

    first = read_window(0)
    second = read_window(1)
    assert first == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert second == [[4, 5, 6, 7], [8, 9, 10, 11]]
    assert first[1] == second[0]  # the one-row traversal step overlaps

    flattened = [value for row in second for value in row]
    assert flattened == [4, 5, 6, 7, 8, 9, 10, 11]
    transposed = [second[row][col] for col in range(4) for row in range(2)]
    assert transposed == [4, 8, 5, 9, 6, 10, 7, 11]

    cube = [[[i * 4 + j * 2 + k for k in range(2)]
             for j in range(2)] for i in range(2)]
    permuted = [cube[j][k][i] for i in range(2)
                for j in range(2) for k in range(2)]
    assert permuted == [0, 2, 4, 6, 1, 3, 5, 7]

    # Eight blocks each add four to one counter; this is only arithmetic,
    # not a CPU simulation of CUDA's concurrent atomic implementation.
    assert sum(4 for _ in range(8)) == 32
    print("CPU tile-view coordinate model: PASS")


if __name__ == "__main__":
    main()
