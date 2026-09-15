"""Toy dependency counterexample and report traffic arithmetic, not mHC implementation."""
import math


def mix(coefficients, streams):
    if not streams or len(coefficients) != len(streams):
        raise ValueError("one coefficient per residual stream required")
    dim = len(streams[0])
    if dim == 0 or any(len(row) != dim for row in streams):
        raise ValueError("rectangular nonempty streams required")
    return [sum(a * row[d] for a, row in zip(coefficients, streams)) for d in range(dim)]


def toy_coefficients(streams):
    # An input-dependent predictor used only to exhibit the dependency.
    scores = [sum(row) for row in streams]
    weights = [math.exp(x - max(scores)) for x in scores]
    return [x / sum(weights) for x in weights]


if __name__ == "__main__":
    streams = [[1.0, 0.0], [0.0, 2.0]]
    current = mix(toy_coefficients(streams), streams)
    shifted = mix([0.5, 0.5], streams)
    assert any(abs(a - b) > 1e-3 for a, b in zip(current, shifted))
    n, d = 4, 5120
    old, two_pass, single = (4*n + 4)*d, (3*n + 2)*d, (2*n + 2)*d
    assert (old, two_pass, single) == (102400, 71680, 51200)
    main_entry = 512 * 4 // 8 + 512 // 16
    index_entry = 128 * 4 // 8 + 128 // 32
    assert (main_entry, index_entry) == (288, 68)
    assert (main_entry + index_entry) * (3 / 2 + 1) == 890
    assert 61 * (512 + 64) * 2 == 70272
    print("PASS: dependency change is not identity; mHC traffic; 890B and MLA cache arithmetic")
