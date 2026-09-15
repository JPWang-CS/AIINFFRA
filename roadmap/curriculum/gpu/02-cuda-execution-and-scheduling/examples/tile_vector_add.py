#!/usr/bin/env python3
"""Minimal cuTile vector add with explicit tail-storage validation."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a cuTile FP32 vector add for several tail sizes."
    )
    return parser.parse_args()


def run() -> int:
    # Keep optional GPU-stack imports after argparse so --help works on a CPU-only
    # machine and does not require torch or cuda.tile to be importable.
    try:
        import torch
        import cuda.tile as ct
    except ImportError as error:
        # A missing top-level module is an expected local skip.  Broken imports
        # inside an installed package remain errors and must not be masked.
        missing = getattr(error, "name", "")
        if isinstance(error, ModuleNotFoundError) and missing in (
            "torch",
            "cuda",
            "cuda.tile",
        ):
            print(f"SKIP: cuTile/torch stack unavailable: {error}")
            return 77
        raise

    if not torch.cuda.is_available():
        print("SKIP: no usable CUDA device")
        return 77

    from importlib.metadata import PackageNotFoundError, version

    try:
        tile_version = version("cuda-tile")
    except PackageNotFoundError:
        tile_version = "unknown"
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"cuda.tile={tile_version} gpu={torch.cuda.get_device_name(0)}"
    )

    @ct.kernel
    def vec_add(a, b, c, TILE: ct.Constant[int]):
        bid = ct.bid(0)
        x = ct.load(
            a,
            index=(bid,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        y = ct.load(
            b,
            index=(bid,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ct.store(c, index=(bid,), tile=x + y)

    device = torch.device("cuda")
    sentinel = -12345.25

    def run_case(n: int) -> None:
        # Integer-like FP32 values keep the reference comparison exact.
        indices = torch.arange(n, dtype=torch.float32, device=device)
        a = indices - 7.0
        b = torch.remainder(indices, 13.0) + 2.0
        output_storage = torch.full(
            (n + 7,), sentinel, dtype=torch.float32, device=device
        )
        c = output_storage[:n]

        if n == 0:
            # Never submit a zero-sized grid; the empty case is host-side only.
            print("case n=0: bypassed launch (no zero grid)")
            return

        if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
            raise AssertionError(f"case n={n} did not produce contiguous arrays")
        pointers = (a.data_ptr(), b.data_ptr(), output_storage.data_ptr())
        if len(set(pointers)) != len(pointers):
            raise AssertionError(f"case n={n} unexpectedly aliases input/output")

        grid = ((n + 127) // 128,)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            vec_add,
            (a, b, c, 128),
        )
        torch.cuda.current_stream().synchronize()

        expected = a + b
        torch.testing.assert_close(c, expected, rtol=0, atol=0)
        if not torch.all(output_storage[n:] == sentinel).item():
            raise AssertionError(f"case n={n} overwrote the output tail")
        print(f"case n={n}: PASS")

    for n in (0, 1, 127, 128, 129, 1025):
        run_case(n)
    return 0


def main() -> int:
    parse_args()
    return run()


if __name__ == "__main__":
    sys.exit(main())
