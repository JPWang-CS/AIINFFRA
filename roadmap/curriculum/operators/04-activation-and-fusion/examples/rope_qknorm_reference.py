"""Pure-Python RoPE/QK-Norm reference and edge-case checks.

The main implementation here follows the split-half convention used by
LeetGPU #61: [x_0..x_(D/2-1), x_(D/2)..x_(D-1)] -> [-x_2, x_1].  cos/sin are
inputs, not values recomputed by this reference.
"""

from __future__ import annotations

import math


def rotate_half_split(x: list[float]) -> list[float]:
    if len(x) % 2:
        raise ValueError("rotation dimension must be even")
    half = len(x) // 2
    return [-x[half + i] for i in range(half)] + [x[i] for i in range(half)]


def rope_from_table_row(x: list[float], cos_row: list[float], sin_row: list[float], drot: int | None = None) -> list[float]:
    """Low-level rotation using one already selected cos/sin row."""
    if drot is None:
        drot = len(x)
    if drot <= 0 or drot > len(x) or drot % 2 or len(cos_row) < drot or len(sin_row) < drot:
        raise ValueError("drot must be a positive even prefix covered by cos/sin")
    prefix = x[:drot]
    rotated_prefix = rotate_half_split(prefix)
    rotated = [prefix[i] * cos_row[i] + rotated_prefix[i] * sin_row[i] for i in range(drot)]
    return rotated + x[drot:]


def rope_split_half(x: list[float], cos_table: list[list[float]], sin_table: list[list[float]], position: int, drot: int | None = None) -> list[float]:
    """Select the absolute table row, including any prefix tokens."""
    if position < 0 or position >= len(cos_table) or position >= len(sin_table):
        raise IndexError("absolute position outside cos/sin table")
    return rope_from_table_row(x, cos_table[position], sin_table[position], drot)


def adjacent_to_split_half(x: list[float]) -> list[float]:
    if len(x) % 2:
        raise ValueError("dimension must be even")
    return x[0::2] + x[1::2]


def split_half_to_adjacent(x: list[float]) -> list[float]:
    if len(x) % 2:
        raise ValueError("dimension must be even")
    half = len(x) // 2
    return [value for pair in zip(x[:half], x[half:]) for value in pair]


def rms_norm(x: list[float], gamma: list[float] | None = None, eps: float = 1e-6) -> list[float]:
    if gamma is not None and len(gamma) != len(x):
        raise ValueError("gamma must be per head-dimension")
    scale = 1.0 / math.sqrt(sum(value * value for value in x) / len(x) + eps)
    return [value * scale * (gamma[i] if gamma is not None else 1.0) for i, value in enumerate(x)]


def layer_norm(x: list[float], gamma: list[float] | None = None, beta: list[float] | None = None, eps: float = 1e-5) -> list[float]:
    mean = sum(x) / len(x)
    var = sum((value - mean) ** 2 for value in x) / len(x)
    return [((value - mean) / math.sqrt(var + eps)) * (gamma[i] if gamma else 1.0) + (beta[i] if beta else 0.0) for i, value in enumerate(x)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def trig_row(phase: float, half: int = 4) -> tuple[list[float], list[float]]:
    """Test-only repeated-half table; model code consumes checkpoint tables."""
    angles = [phase * (i + 1) for i in range(half)]
    return [math.cos(a) for a in angles] * 2, [math.sin(a) for a in angles] * 2


def main() -> None:
    cos, sin = trig_row(0.37)
    table_c = [trig_row(0.37 * p)[0] for p in range(6)]
    table_s = [trig_row(0.37 * p)[1] for p in range(6)]
    x = [0.4, -1.2, 2.0, 0.7, -0.3, 1.1, 0.8, -2.0]
    y = rope_split_half(x, table_c, table_s, position=3)
    assert math.isclose(sum(a * a for a in x), sum(a * a for a in y), rel_tol=1e-12, abs_tol=1e-12)

    # A non-zero position is essential: position zero is the identity and can
    # hide an adjacent-vs-split-half implementation error.
    zero_cos = [1.0] * 8
    zero_sin = [0.0] * 8
    assert rope_split_half(x, table_c, table_s, position=0) == x
    c4, s4 = trig_row(0.37, half=2)  # separately organized [c0,c1|c0,c1], not a full-table slice
    partial = rope_from_table_row(x, c4, s4, drot=4)
    assert len(partial) == len(x) and partial[4:] == x[4:]
    assert partial[:4] == [x[0] * c4[0] - x[2] * s4[0], x[1] * c4[1] - x[3] * s4[1], x[2] * c4[0] + x[0] * s4[0], x[3] * c4[1] + x[1] * s4[1]]
    assert math.isclose(sum(a * a for a in x[:4]), sum(a * a for a in partial[:4]), rel_tol=1e-12, abs_tol=1e-12)
    assert split_half_to_adjacent(adjacent_to_split_half(x)) == x

    # Relative-position invariance: rotating both vectors by the same absolute
    # shift preserves their dot product for a real orthogonal table.
    q, k = x, [0.2, 1.3, -0.4, 0.9, 1.2, -0.7, 0.6, 0.1]
    c0, s0 = trig_row(0.11)
    c1, s1 = trig_row(0.11 * 4)
    p, qpos = 1, 4
    lhs = dot(rope_from_table_row(q, c0, s0), rope_from_table_row(k, c1, s1))
    rel_c, rel_s = trig_row(0.11 * (qpos - p))
    rhs = dot(q, rope_from_table_row(k, rel_c, rel_s))
    assert math.isclose(lhs, rhs, rel_tol=1e-10, abs_tol=1e-10)

    # RMS without gamma and pair-shared gamma commute with a true orthogonal
    # rotation.  Per-dimension gamma generally does not; position 3 avoids the
    # identity-rotation false positive.
    unit_c, unit_s = trig_row(0.2)
    pair_gamma = [1.5, 0.7, 1.2, 0.9, 1.5, 0.7, 1.2, 0.9]
    uneven_gamma = [1.5, 0.8, 0.7, 1.4, 1.2, 0.6, 0.9, 1.7]
    assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(
        rope_from_table_row(rms_norm(x), unit_c, unit_s), rms_norm(rope_from_table_row(x, unit_c, unit_s))))
    assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(
        rope_from_table_row(rms_norm(x, pair_gamma), unit_c, unit_s), rms_norm(rope_from_table_row(x, unit_c, unit_s), pair_gamma)))
    assert any(abs(a - b) > 1e-5 for a, b in zip(
        rope_from_table_row(rms_norm(x, uneven_gamma), unit_c, unit_s), rms_norm(rope_from_table_row(x, unit_c, unit_s), uneven_gamma)))
    assert any(abs(a - b) > 1e-5 for a, b in zip(
        layer_norm(rope_from_table_row(x, unit_c, unit_s)), rope_from_table_row(layer_norm(x), unit_c, unit_s)))

    # A K cache stores K_norm -> RoPE(position) once; a later read must not
    # rotate the already-positioned cache entry again.
    k_raw = [0.6, -0.2, 1.1, 0.9, -0.7, 0.4, 0.3, -1.0]
    k_norm = rms_norm(k_raw)
    cache = {}
    cache[3] = rope_from_table_row(k_norm, unit_c, unit_s)
    assert k_raw == [0.6, -0.2, 1.1, 0.9, -0.7, 0.4, 0.3, -1.0]
    double_rope = rope_from_table_row(cache[3], unit_c, unit_s)
    assert any(abs(a - b) > 1e-5 for a, b in zip(double_rope, cache[3]))
    print("PASS: split-half RoPE, prefix/drot, norm/relative-position, layout roundtrip, gamma counterexample, K-cache single rotation")


if __name__ == "__main__":
    main()
