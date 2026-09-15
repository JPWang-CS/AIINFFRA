#!/usr/bin/env python3
"""Exercise cuTile tiled views, shape operations, and reduction atomics."""

import sys


def main() -> int:
    try:
        import torch
        import cuda.tile as ct
    except ImportError as error:
        print(f"SKIP: PyTorch or cuTile is unavailable: {error}")
        return 77

    if not torch.cuda.is_available():
        print("SKIP: no usable CUDA device")
        return 77

    @ct.kernel
    def transform(source, flat_out, transpose_out, permuted_out):
        bid = ct.bid(0)
        source_view = source.tiled_view(
            (2, 4),
            padding_mode=ct.PaddingMode.ZERO,
            traversal_steps=(1, 4),
        )
        tile = source_view.load((bid, 0))
        flat_view = flat_out.tiled_view((1, 8))
        transposed_view = transpose_out.tiled_view((1, 8))
        flat_view.store((bid, 0), tile.reshape((1, 8)))
        transposed_view.store((bid, 0), ct.transpose(tile).reshape((1, 8)))

        if bid == 0:
            cube = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
            permuted = ct.permute(cube, (2, 0, 1))
            permuted_out.tiled_view((8,)).store(0, permuted.reshape((8,)))

    @ct.kernel
    def add_one_tile_per_block(counter):
        view = counter.tiled_view((1,))
        increment = ct.full((1,), 4, dtype=ct.int32)
        # No old value is needed: use TiledView's reduction-style atomic form.
        view.atomic_store_add(0, increment)

    source = torch.arange(16, dtype=torch.int32, device="cuda").reshape(4, 4)
    flat_out = torch.full((2, 8), -1, dtype=torch.int32, device="cuda")
    transpose_out = torch.full((2, 8), -1, dtype=torch.int32, device="cuda")
    permuted_out = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream()

    ct.launch(stream, (2,), transform,
              (source, flat_out, transpose_out, permuted_out))
    ct.launch(stream, (8,), add_one_tile_per_block, (counter,))
    stream.synchronize()

    expected_flat = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    expected_transpose = torch.tensor(
        [[0, 4, 1, 5, 2, 6, 3, 7], [4, 8, 5, 9, 6, 10, 7, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    expected_permute = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(flat_out, expected_flat, rtol=0, atol=0)
    torch.testing.assert_close(transpose_out, expected_transpose, rtol=0, atol=0)
    torch.testing.assert_close(permuted_out, expected_permute, rtol=0, atol=0)
    torch.testing.assert_close(
        counter, torch.tensor([32], dtype=torch.int32, device="cuda"), rtol=0, atol=0
    )
    print("view/reshape/transpose/permute/atomic checks: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
