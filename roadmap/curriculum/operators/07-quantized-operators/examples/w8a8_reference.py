"""Symmetric per-token/per-output W8A8 numerical reference (NumPy CPU)."""
import numpy as np


def w8a8_reference(x, w):
    x, w = np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0] or min(*x.shape, *w.shape) <= 0:
        raise ValueError("nonempty X[T,I] and W[I,O] required")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("finite inputs required")
    sx = np.max(np.abs(x), axis=1, keepdims=True)
    sw = np.max(np.abs(w), axis=0, keepdims=True)
    sx = np.where(sx == 0, 1., np.maximum(sx / 127, np.finfo(np.float64).tiny))
    sw = np.where(sw == 0, 1., np.maximum(sw / 127, np.finfo(np.float64).tiny))
    qx = np.clip(np.rint(x / sx), -127, 127).astype(np.int8)
    qw = np.clip(np.rint(w / sw), -127, 127).astype(np.int8)
    # int64 is the CPU mathematical reference, not a claim about GPU instructions.
    acc = qx.astype(np.int64) @ qw.astype(np.int64)
    y = acc.astype(np.float64) * sx * sw
    return y, qx, qw, sx, sw


if __name__ == "__main__":
    from quantization_lab import apply_smoothquant
    z, *_ = w8a8_reference(np.zeros((3, 4)), np.zeros((4, 2)))
    np.testing.assert_array_equal(z, 0)
    rng = np.random.default_rng(21)
    x = rng.normal(size=(16, 8)); x[:, 0] *= 100
    w = rng.normal(size=(8, 5)); w[0] *= .01
    before, qx, qw, sx, sw = w8a8_reference(x, w)
    for t in range(16):
        for o in range(5):
            expected = sum(int(qx[t, i]) * int(qw[i, o]) for i in range(8)) * sx[t, 0] * sw[0, o]
            np.testing.assert_allclose(before[t, o], expected)
    xs, ws, _ = apply_smoothquant(x, w, .5)
    np.testing.assert_allclose(xs @ ws, x @ w, rtol=1e-12, atol=1e-12)
    after, *_ = w8a8_reference(xs, ws)
    assert np.isfinite(after).all()
    print("PASS: zero channels; integer dot and scales; SmoothQuant transform")
    print("Illustrative MSE before/after:",
          float(np.mean((before - x @ w)**2)), float(np.mean((after - x @ w)**2)))
