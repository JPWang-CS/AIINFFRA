"""CPU-only semantic checks for the reduction and normalization chapter.

These checks deliberately do not import CUDA, Triton, NumPy, or PyTorch. They
check indexing and formulas; they are not a compiler or GPU execution test.
"""

from math import exp, isclose, isfinite, sqrt


INT32_MAX = 2**31 - 1


def row_sum_coverage(cols, block):
    rounded = ((cols + block - 1) // block) * block
    seen = []
    for tid in range(block):
        for col in range(tid, rounded, block):
            if col < cols:
                seen.append(col)
    return seen


def partial_softmax(values):
    if not values or any(not isfinite(value) for value in values):
        raise ValueError("partial input must be finite and non-empty")
    local_max = max(values)
    local_sum = sum(exp(value - local_max) for value in values)
    return local_max, local_sum


def merge_softmax_partials(partials):
    if not partials:
        raise ValueError("at least one partial is required")
    global_max = max(local_max for local_max, _ in partials)
    global_sum = sum(
        local_sum * exp(local_max - global_max)
        for local_max, local_sum in partials
    )
    return global_max, global_sum


def merge_welford(a, b):
    n_a, mean_a, m2_a = a
    n_b, mean_b, m2_b = b
    if n_a == 0:
        return b
    if n_b == 0:
        return a
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    m2 = m2_a + m2_b + delta * delta * n_a * n_b / n
    return n, mean, m2


def welford(values):
    state = (0, 0.0, 0.0)
    for value in values:
        state = merge_welford(state, (1, value, 0.0))
    return state


def masked_layer_norm(values, cols, block, eps=1e-5):
    padded = list(values) + [0.0] * (block - cols)
    mask = [index < cols for index in range(block)]
    mean = sum(value for value, valid in zip(padded, mask) if valid) / cols
    centered = [value - mean if valid else 0.0 for value, valid in zip(padded, mask)]
    variance = sum(value * value for value in centered) / cols
    inv_std = 1.0 / sqrt(variance + eps)
    return [centered[index] * inv_std for index in range(cols)], mean, variance


def layer_norm_forward(values, gamma, beta, eps=1e-5):
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    variance = sum(value * value for value in centered) / len(values)
    inv_std = 1.0 / sqrt(variance + eps)
    return [centered[i] * inv_std * gamma[i] + beta[i] for i in range(len(values))]


def layer_norm_dx(values, gamma, upstream, eps=1e-5):
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    variance = sum(value * value for value in centered) / len(values)
    inv_std = 1.0 / sqrt(variance + eps)
    hat = [value * inv_std for value in centered]
    u = [upstream[i] * gamma[i] for i in range(len(values))]
    mean_u = sum(u) / len(values)
    mean_uhat = sum(u[i] * hat[i] for i in range(len(values))) / len(values)
    return [inv_std * (u[i] - mean_u - hat[i] * mean_uhat) for i in range(len(values))]


def reject_softmax_input(values):
    if not values:
        raise ValueError("rows and cols must be positive")
    if any(not isfinite(value) for value in values):
        raise ValueError("softmax input must be finite")


def main():
    # D=1000, BLOCK=256: 232 threads have four valid elements; 24 have three.
    coverage = row_sum_coverage(1000, 256)
    assert sorted(coverage) == list(range(1000))
    assert len(coverage) == len(set(coverage)) == 1000

    left = partial_softmax([3.0, 2.0, 1.0])
    right = partial_softmax([1.0, 0.0])
    global_max, global_sum = merge_softmax_partials([left, right])
    expected = sum(exp(value - global_max) for value in [3.0, 2.0, 1.0, 1.0, 0.0])
    assert global_max == 3.0
    assert isclose(global_sum, expected, rel_tol=1e-12)

    merged = merge_welford((2, 2.0, 2.0), (2, 6.0, 2.0))
    assert merged == (4, 4.0, 20.0)
    assert merge_welford((0, 0.0, 0.0), merged) == merged
    assert merge_welford(merged, (0, 0.0, 0.0)) == merged
    n, mean, m2 = welford([1.0, 3.0, 5.0, 7.0])
    assert (n, mean) == (4, 4.0)
    assert isclose(m2 / n, 5.0, rel_tol=1e-12)

    normalized, mean, variance = masked_layer_norm([3.0] * 257, 257, 512)
    assert isclose(mean, 3.0, rel_tol=1e-12)
    assert isclose(variance, 0.0, abs_tol=1e-12)
    assert max(abs(value) for value in normalized) < 1e-12
    padded_values = [3.0, 4.0, 5.0, 0.0]
    wrong_centered_sum = sum((value - 4.0) ** 2 for value in padded_values)
    assert wrong_centered_sum / 3.0 == 6.0

    negative, _, _ = masked_layer_norm([-8.0, -4.0, 0.0, 4.0], 4, 4)
    assert isclose(sum(negative), 0.0, abs_tol=1e-12)
    scaled, _, _ = masked_layer_norm([1e10, 1e10 + 1.0], 2, 2)
    assert all(isfinite(value) for value in scaled)

    x = [3.0, 4.0, 5.0]
    gamma = [1.0, 2.0, 3.0]
    upstream = [0.3, -0.1, 0.2]
    analytic = layer_norm_dx(x, gamma, upstream)
    step = 1e-5
    numeric = []
    for index in range(len(x)):
        plus = x[:]
        minus = x[:]
        plus[index] += step
        minus[index] -= step
        loss_plus = sum(a * b for a, b in zip(layer_norm_forward(plus, gamma, [0.0] * 3), upstream))
        loss_minus = sum(a * b for a, b in zip(layer_norm_forward(minus, gamma, [0.0] * 3), upstream))
        numeric.append((loss_plus - loss_minus) / (2.0 * step))
    assert max(abs(a - b) for a, b in zip(analytic, numeric)) < 1e-6

    for bad in ([], [float("inf")], [float("-inf")], [float("nan")]):
        try:
            reject_softmax_input(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite or empty input was accepted")

    assert 0 < INT32_MAX
    print("PASS: row-tail coverage, stable partial merge, Welford n=0, masked LayerNorm, and rejection boundaries")


if __name__ == "__main__":
    main()
