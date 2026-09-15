"""Pure-stdlib activation, packing, mask, and byte-ledger checks.

This file is intentionally independent of CUDA, Triton, NumPy, and PyTorch.
It checks the mathematical contract and the address arithmetic used by the
chapter; it does not execute a GPU kernel.
"""

from __future__ import annotations

import math
import struct


F32_MAX = 3.4028234663852886e38
F32_EXP_OVERFLOW_LOG = math.log(F32_MAX)


def stable_sigmoid(x: float) -> float:
    """Avoid exp(large positive) while retaining IEEE NaN propagation."""
    e = math.exp(-abs(x))
    return 1.0 / (1.0 + e) if x >= 0.0 else e / (1.0 + e)


def relu(x: float) -> float:
    return max(0.0, x)


def gelu_erf(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_tanh(x: float) -> float:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x**3)))


def silu(x: float) -> float:
    return x * stable_sigmoid(x)


def swiglu(gate: float, up: float) -> float:
    return silu(gate) * up


def derivatives(x: float, up: float) -> dict[str, float]:
    s = stable_sigmoid(x)
    silu_dx = s + x * s * (1.0 - s)
    return {
        "relu": 1.0 if x > 0.0 else 0.0,
        "gelu_erf": 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        + x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi),
        "silu": silu_dx,
        "swiglu_dgate": up * silu_dx,
        "swiglu_dup": silu(x),
    }


def finite_difference(fn, x: float, step: float = 1e-5) -> float:
    return (fn(x + step) - fn(x - step)) / (2.0 * step)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def packed_offset(token: int, feature: int, width: int, branch: str) -> int:
    """Element offset in a row-major [T, 2I] row [gate | up] packing."""
    if token < 0 or feature < 0 or feature >= width:
        raise IndexError("token/feature outside logical [T, I]")
    if branch == "gate":
        return token * (2 * width) + feature
    if branch == "up":
        return token * (2 * width) + width + feature
    raise ValueError("branch must be gate or up")


def split_flattened_packed(values: list[float], tokens: int, width: int) -> tuple[list[float], list[float]]:
    """Read rows, proving that global first-half/second-half slicing is wrong."""
    if len(values) != tokens * 2 * width:
        raise ValueError("packed buffer has the wrong number of elements")
    gate, up = [], []
    for token in range(tokens):
        row = values[token * 2 * width : (token + 1) * 2 * width]
        gate.extend(row[:width])
        up.extend(row[width:])
    return gate, up


def pointwise_bytes(numel: int, element_bytes: int) -> dict[str, int]:
    if numel < 0 or element_bytes <= 0:
        raise ValueError("numel must be non-negative and element_bytes positive")
    # Separate SiLU: read G + write S, then read S + read U + write Z.
    return {"separate": 5 * numel * element_bytes, "fused": 3 * numel * element_bytes}


def index_mask(numel: int, block: int, program: int) -> list[tuple[int, bool]]:
    if numel < 0 or block <= 0 or program < 0:
        raise ValueError("invalid index-mask arguments")
    base = int(program) * int(block)
    return [(base + lane, base + lane < numel) for lane in range(block)]


def main() -> None:
    for value in (-4.0, -1.0, 0.0, 1.0, 4.0):
        assert math.isclose(derivatives(value, 1.7)["silu"], finite_difference(silu, value), rel_tol=2e-8, abs_tol=2e-8)
        assert math.isclose(derivatives(value, 1.7)["gelu_erf"], finite_difference(gelu_erf, value), rel_tol=2e-7, abs_tol=2e-7)
        assert math.isclose(
            derivatives(value, 1.7)["swiglu_dgate"],
            finite_difference(lambda gate: swiglu(gate, 1.7), value),
            rel_tol=2e-8,
            abs_tol=2e-8,
        )
        assert math.isclose(
            derivatives(value, 1.7)["swiglu_dup"],
            finite_difference(lambda value_up: swiglu(value, value_up), 1.7),
            rel_tol=2e-8,
            abs_tol=2e-8,
        )

    # ReLU is not centrally differenced at zero: its mathematical derivative
    # is undefined there, while an implementation must choose a convention.
    assert derivatives(-1.0, 1.0)["relu"] == 0.0
    assert derivatives(1.0, 1.0)["relu"] == 1.0

    assert gelu_erf(1.25) != gelu_tanh(1.25)
    assert stable_sigmoid(-1000.0) == 0.0
    assert stable_sigmoid(1000.0) == 1.0
    assert math.isnan(stable_sigmoid(float("nan")))
    assert F32_EXP_OVERFLOW_LOG > 88.0

    packed = [float(index) for index in range(2 * 2 * 3)]
    gate, up = split_flattened_packed(packed, tokens=2, width=3)
    assert gate == [0.0, 1.0, 2.0, 6.0, 7.0, 8.0]
    assert up == [3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
    assert packed_offset(1, 2, 3, "gate") == 8
    assert packed_offset(1, 2, 3, "up") == 11

    assert pointwise_bytes(7, 2) == {"separate": 70, "fused": 42}
    mask = index_mask(10, block=8, program=1)
    assert [offset for offset, valid in mask if valid] == [8, 9]
    assert all(isinstance(offset, int) for offset, _ in mask)

    print("PASS: activation derivatives, stable sigmoid, IEEE boundaries, row-packed offsets, bytes, and tail masks")


if __name__ == "__main__":
    main()
