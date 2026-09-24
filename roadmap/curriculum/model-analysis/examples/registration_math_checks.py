"""NumPy-only finite-difference check of the RMSNorm gradient formula.

This validates mathematics only; it does not test torch.library, FakeTensor,
autograd registration, Triton compilation, or GPU execution.
"""
import numpy as np


def norm(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps)


def analytic_grad(x, upstream, eps):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps)
    projection = np.mean(upstream * x, axis=1, keepdims=True)
    return upstream * r - x * (r ** 3) * projection


def loss(x, upstream, eps):
    return np.sum(norm(x, eps) * upstream)


def finite_difference(x, upstream, eps, step=1e-6):
    result = np.empty_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (loss(plus, upstream, eps) - loss(minus, upstream, eps)) / (2 * step)
    return result


def main():
    rng = np.random.default_rng(20260922)
    for shape in ((1, 1), (2, 3), (3, 65)):
        x = rng.normal(size=shape)
        upstream = rng.normal(size=shape)
        expected = finite_difference(x, upstream, 1e-6)
        actual = analytic_grad(x, upstream, 1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        print(f"PASS formula-only finite difference shape={shape}; max_abs={np.max(np.abs(actual-expected)):.3g}")
    print("NOTE: no PyTorch registration, Triton, or GPU behavior was tested.")


if __name__ == "__main__":
    main()
