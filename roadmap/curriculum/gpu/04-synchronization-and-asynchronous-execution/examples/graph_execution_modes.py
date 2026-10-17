"""Compare Eager, torch.compile, and CUDA Graph for one pointwise chain.

The default command checks correctness on CUDA. ``--cpu-check`` checks only
the CPU mathematical reference and never queries CUDA device availability.
This example intentionally contains no timing or profiling code.
"""

from __future__ import annotations

import argparse
import sys

import torch


ALPHA = 1.25
BETA = 0.25


def pointwise_chain(x: torch.Tensor) -> torch.Tensor:
    """Apply the same three elementwise operations in every execution mode."""
    u = x * ALPHA
    v = u + BETA
    return torch.relu(v)


def cpu_cases() -> list[torch.Tensor]:
    return [
        torch.tensor([-2.0, -1.0, 0.0, 2.0], dtype=torch.float32),
        torch.tensor([-4.0, -0.5, 1.0, 3.0], dtype=torch.float32),
        torch.tensor([0.25, -3.0, 2.5, -1.25], dtype=torch.float32),
    ]


def run_cpu_check() -> None:
    """Check hand-computed values and shape without touching CUDA state."""
    x = cpu_cases()[0]
    expected = torch.tensor([0.0, 0.0, 0.25, 2.75], dtype=torch.float32)
    actual = pointwise_chain(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)
    print("PASS: CPU reference values and shape")


def check_same_shape_dtype_device(
    new_x: torch.Tensor, static_x: torch.Tensor
) -> None:
    """Reject input changes that cannot safely update a captured buffer."""
    if not isinstance(new_x, torch.Tensor):
        raise TypeError("new_x must be a torch.Tensor")
    if new_x.shape != static_x.shape:
        raise ValueError(
            f"shape changed: got {tuple(new_x.shape)}, "
            f"expected {tuple(static_x.shape)}"
        )
    if new_x.dtype != static_x.dtype:
        raise TypeError(f"dtype changed: got {new_x.dtype}, expected {static_x.dtype}")
    if new_x.device != static_x.device:
        raise ValueError(
            f"device changed: got {new_x.device}, expected {static_x.device}"
        )


def gpu_cases(device: torch.device) -> list[torch.Tensor]:
    """Build three distinct inputs with one static shape for graph replay."""
    base = torch.arange(257, dtype=torch.float32, device=device)
    return [
        (base - 128.0) / 32.0,
        (64.0 - base) / 16.0,
        (base.remainder(17.0) - 8.0) / 4.0,
    ]


def run_gpu_check() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --cpu-check for the CPU reference check"
        )

    device = torch.device("cuda")
    batches = gpu_cases(device)

    # This isolated compile comparison disables Inductor's automatic CUDA
    # Graph path, so compilation/fusion is not mixed with graph replay.
    compiled = torch.compile(
        pointwise_chain,
        backend="inductor",
        fullgraph=True,
        options={"triton.cudagraphs": False},
    )

    # Trigger the first compile before graph capture. The calls are correctness
    # setup only; this program does not measure their duration.
    compiled(batches[0])

    # Keep these fixed-address tensors alive as long as the graph can replay.
    static_x = batches[0].clone()
    current_stream = torch.cuda.current_stream(device)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            pointwise_chain(static_x)
    current_stream.wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Capture the original Eager function, not the compiled wrapper.
        static_y = pointwise_chain(static_x)

    for batch_index, new_x in enumerate(batches):
        eager_y = pointwise_chain(new_x)
        compiled_y = compiled(new_x)

        check_same_shape_dtype_device(new_x, static_x)
        # copy_ and replay are ordered on the same current stream. Rebinding
        # static_x would not change the pointer captured by graph.
        static_x.copy_(new_x)
        graph.replay()

        torch.testing.assert_close(compiled_y, eager_y)
        torch.testing.assert_close(static_y, eager_y)
        print(f"PASS: CUDA batch {batch_index + 1}/3")

    # graph, static_x, and static_y remain referenced through the final replay.
    # A later replay will overwrite static_y; clone it first if a caller needs
    # to preserve a previous result across replays.


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-check",
        action="store_true",
        help="check the CPU reference values and shape without querying CUDA",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.cpu_check:
        run_cpu_check()
        return 0
    run_gpu_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
