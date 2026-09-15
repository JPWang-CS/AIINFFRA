"""A small, NumPy-only quantization laboratory.

The convention in this file is ``X[T, I] @ W[I, O]``.  Quantization groups
run along the input axis ``I``.  This is deliberately a teaching
implementation: it favors visible axes, checks, and reference calculations
over speed, and it does not require Torch, a GPU, or an author library.

Run ``python quantization_lab.py`` for the built-in tests or
``python quantization_lab.py --demo`` for a deterministic numerical demo.
"""

from __future__ import annotations

import argparse
import math
import sys
import unittest
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


_QMIN = -7                 # Deliberately leave storage code -8 unused.
_QMAX = 7
_STORAGE_QMIN = -8         # A signed int4 nibble can represent -8..7.
_STORAGE_QMAX = 7


def _as_float_array(name: str, value: np.ndarray, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _check_group_size(group_size: int) -> int:
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)):
        raise ValueError("group_size must be a positive integer")
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    return int(group_size)


def _valid_rows(
    x: np.ndarray,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a validated row mask; True means a real calibration token.

    ``token_mask`` uses the common 1/True=valid convention.  A
    ``padding_mask`` uses the common key-padding convention: 1/True means
    padding and is therefore excluded.
    """
    x = _as_float_array("X", x, 2)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must have a non-zero token and channel dimension")
    mask = np.ones(x.shape[0], dtype=bool)
    for name, supplied in (("token_mask", token_mask), ("padding_mask", padding_mask)):
        if supplied is None:
            continue
        candidate = np.asarray(supplied)
        if candidate.shape != (x.shape[0],):
            raise ValueError(f"{name} must have shape ({x.shape[0]},)")
        if candidate.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be boolean or numeric 0/1")
        if not np.all(np.isfinite(candidate.astype(np.float64, copy=False))):
            raise ValueError(f"{name} contains NaN or infinity")
        if not np.all(np.isin(candidate, (0, 1))):
            raise ValueError(f"{name} must contain only 0/1 values")
        if name == "token_mask":
            mask &= candidate.astype(bool)
        else:
            mask &= ~candidate.astype(bool)
    if not np.any(mask):
        raise ValueError("the token/padding masks exclude every calibration token")
    return mask


@dataclass(frozen=True)
class CalibrationStats:
    """Statistics calculated only from unmasked calibration rows."""

    meanabs: np.ndarray
    maxabs: np.ndarray
    gram: np.ndarray
    valid_tokens: int


def calibration_statistics(
    x: np.ndarray,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> CalibrationStats:
    """Compute mean-absolute, max-absolute, and second-order Gram statistics.

    ``token_mask`` is 1=valid, while ``padding_mask`` is 1=padding and is
    excluded.  The Gram matrix is the unnormalised second-order sum, so
    callers can choose its normalization.
    """
    x = _as_float_array("X", x, 2)
    mask = _valid_rows(x, token_mask, padding_mask)
    valid = x[mask]
    try:
        with np.errstate(over="raise", invalid="raise"):
            gram = valid.T @ valid
            meanabs = np.mean(np.abs(valid), axis=0)
            maxabs = np.max(np.abs(valid), axis=0)
    except FloatingPointError as exc:
        raise ValueError("calibration statistics overflowed; reduce input magnitude") from exc
    if not (np.all(np.isfinite(meanabs)) and np.all(np.isfinite(maxabs)) and np.all(np.isfinite(gram))):
        raise ValueError("calibration statistics are non-finite after reduction")
    return CalibrationStats(
        meanabs=meanabs,
        maxabs=maxabs,
        gram=gram,
        valid_tokens=int(valid.shape[0]),
    )


def _check_weight(w: np.ndarray) -> np.ndarray:
    w = _as_float_array("W", w, 2)
    if w.shape[0] == 0 or w.shape[1] == 0:
        raise ValueError("W must have a non-zero input and output dimension")
    return w


def _group_count(input_size: int, group_size: int) -> int:
    return (input_size + group_size - 1) // group_size


def quantize_int4_group(w: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Round-to-nearest-even symmetric INT4 per ``[input-group, output]``.

    The returned ``q`` is int8 for convenient teaching/debugging, but its
    values are restricted to [-7, 7].  A packed signed nibble still has the
    representable storage range [-8, 7]; -8 is intentionally never produced
    by this symmetric scale=amax/7 quantizer.
    """
    w = _check_weight(w)
    group_size = _check_group_size(group_size)
    groups = _group_count(w.shape[0], group_size)
    scales = np.ones((groups, w.shape[1]), dtype=np.float64)
    q = np.empty(w.shape, dtype=np.int8)
    for g in range(groups):
        start, end = g * group_size, min((g + 1) * group_size, w.shape[0])
        amax = np.max(np.abs(w[start:end]), axis=0)
        # A finite subnormal can underflow during division.  Keep a positive
        # representable scale; an exactly zero channel still uses identity.
        tiny = np.finfo(np.float64).tiny
        scales[g] = np.where(amax == 0.0, 1.0, np.maximum(amax / _QMAX, tiny))
        # np.rint is ties-to-even, and clipping is explicit rather than an
        # accidental consequence of casting to int8.
        q[start:end] = np.clip(
            np.rint(w[start:end] / scales[g][None, :]), _QMIN, _QMAX
        ).astype(np.int8)
    return q, scales


def dequantize_int4_group(q: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """Materialize a teaching reference from q[I,O] and scales[G,O]."""
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("q must be a non-empty 2-D array")
    if not np.issubdtype(q.dtype, np.integer):
        raise ValueError("q must be an integer array")
    if np.any(q < _QMIN) or np.any(q > _QMAX):
        raise ValueError("quantizer q must be in [-7, 7]")
    scales = _as_float_array("scales", scales, 2)
    group_size = _check_group_size(group_size)
    expected = (_group_count(q.shape[0], group_size), q.shape[1])
    if scales.shape != expected:
        raise ValueError(f"scales must have shape {expected}, got {scales.shape}")
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")
    out = np.empty(q.shape, dtype=np.float64)
    for g in range(expected[0]):
        start, end = g * group_size, min((g + 1) * group_size, q.shape[0])
        out[start:end] = q[start:end].astype(np.float64) * scales[g][None, :]
    return out


def _check_x_w(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float_array("X", x, 2)
    w = _check_weight(w)
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"X[I={x.shape[1]}] and W[I={w.shape[0]}] disagree")
    return x, w


def smoothquant_scales(x: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """SmoothQuant's separate amax formula, not the AWQ candidate family."""
    x, w = _check_x_w(x, w)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    x_amax = np.max(np.abs(x), axis=0)
    w_amax = np.max(np.abs(w), axis=1)
    # A channel that is zero on both sides has no scale information; identity
    # is the stable convention.  The tiny floor avoids zero or infinity when
    # only one side is zero.
    numerator = np.maximum(x_amax, 1e-12) ** alpha
    denominator = np.maximum(w_amax, 1e-12) ** (1.0 - alpha)
    scales = numerator / denominator
    scales = np.where((x_amax == 0) & (w_amax == 0), 1.0, scales)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("SmoothQuant derived scales are non-finite or non-positive")
    return scales


def apply_smoothquant(x: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X/s``, ``W*s``, and ``s``; the float layer is invariant."""
    x, w = _check_x_w(x, w)
    scales = smoothquant_scales(x, w, alpha)
    x_scaled = x / scales[None, :]
    w_scaled = w * scales[:, None]
    if not np.allclose(x @ w, x_scaled @ w_scaled, rtol=1e-12, atol=1e-12):
        raise AssertionError("SmoothQuant transform broke X@W invariance")
    return x_scaled, w_scaled, scales


def _awq_scale_candidates(activation_meanabs: np.ndarray) -> list[tuple[Optional[float], np.ndarray]]:
    """AWQ-style teaching subset: explicit identity plus the 20 ratio scales."""
    a = np.asarray(activation_meanabs, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("activation_meanabs must be a finite non-negative vector")
    candidates: list[tuple[Optional[float], np.ndarray]] = [(None, np.ones_like(a))]
    for ratio in np.linspace(0.0, 0.95, 20):
        powered = np.ones_like(a) if ratio == 0.0 else np.power(a, ratio)
        powered = np.maximum(powered, 1e-4)
        # sqrt-before-multiply avoids an otherwise easy overflow in max*min.
        geometric = math.sqrt(float(np.max(powered))) * math.sqrt(float(np.min(powered)))
        candidates.append((float(ratio), powered / geometric))
    return candidates


@dataclass(frozen=True)
class AWQSearchResult:
    scale: np.ndarray
    ratio: Optional[float]
    baseline_mse: float
    best_mse: float
    candidate_mse: tuple[float, ...]
    calibration_output: np.ndarray


def awq_style_search(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> AWQSearchResult:
    """Select an AWQ-style activation-aware scale using calibration output MSE.

    This is only a linear-layer, symmetric-group-quantization teaching subset:
    it does not claim the complete AWQ algorithm (which can also involve
    module outputs, architecture transformations, clipping, and other model
    details).  There is intentionally no held-out input to leak into the
    selection.  The explicit identity candidate makes the selected calibration
    MSE no worse than this function's RTN baseline by construction.
    """
    x, w = _check_x_w(x, w)
    mask = _valid_rows(x, token_mask, padding_mask)
    x_cal = x[mask]
    y_cal = x_cal @ w
    stats = calibration_statistics(x, token_mask, padding_mask)
    candidates = _awq_scale_candidates(stats.meanabs)
    errors: list[float] = []
    for _, scale in candidates:
        q, q_scales = quantize_int4_group(w * scale[:, None], group_size)
        w_q = dequantize_int4_group(q, q_scales, group_size) / scale[:, None]
        errors.append(float(np.mean((x_cal @ w_q - y_cal) ** 2)))
    best_index = int(np.argmin(errors))
    ratio, scale = candidates[best_index]
    return AWQSearchResult(
        scale=scale.copy(),
        ratio=ratio,
        baseline_mse=errors[0],
        best_mse=errors[best_index],
        candidate_mse=tuple(errors),
        calibration_output=y_cal,
    )


def _prepare_gptq(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    damping: float,
    token_mask: Optional[np.ndarray],
    padding_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs and construct fixed scales plus upper ``chol(inv(H))``."""
    x, w = _check_x_w(x, w)
    group_size = _check_group_size(group_size)
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be a finite positive scalar")
    mask = _valid_rows(x, token_mask, padding_mask)
    x_cal = x[mask]
    _, scales = quantize_int4_group(w, group_size)
    t = x_cal.shape[0]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hessian = (2.0 / t) * (x_cal.T @ x_cal) + float(damping) * np.eye(x.shape[1])
    except FloatingPointError as exc:
        raise ValueError("GPTQ Hessian overflowed; reduce calibration input magnitude") from exc
    if not np.all(np.isfinite(hessian)):
        raise ValueError("GPTQ Hessian is non-finite after reduction")
    try:
        inv_hessian = np.linalg.inv(hessian)
        # np.linalg.cholesky returns lower L for A=L@L.T.  U=L.T is the
        # upper Cholesky factor of inv(H), matching the paper-style update.
        u = np.linalg.cholesky(inv_hessian).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("GPTQ Hessian could not be inverted/Cholesky-factorized") from exc
    if not np.all(np.isfinite(u)) or np.any(np.diag(u) <= 0):
        raise ValueError("GPTQ inverse-Hessian Cholesky is non-finite or non-positive")
    return w, scales, u


def _quantize_work_column(
    work: np.ndarray,
    j: int,
    scales: np.ndarray,
    group_size: int,
    scale_column: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    scale = scales[(j if scale_column is None else scale_column) // group_size]
    q_j = np.clip(np.rint(work[:, j] / scale), _QMIN, _QMAX).astype(np.int8)
    return q_j, q_j.astype(np.float64) * scale


def gptq_quantize(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    damping: float = 0.01,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full sequential GPTQ teaching update, with no lazy blocking/act-order.

    ``W_work`` is [O,I].  For ``H=2 X.T@X/T + damping*I`` and upper
    ``U=chol(inv(H))``, each column uses ``e=(w-q)/U[j,j]`` and updates
    ``W_work[:,j:] -= e[:,None]*U[j,j:]``.  Scales are fixed from original W.
    Positive damping keeps a calibration-zero input channel invertible.
    """
    w, scales, u = _prepare_gptq(x, w, group_size, damping, token_mask, padding_mask)
    group_size = _check_group_size(group_size)
    work = w.T.copy()                  # [O, I]
    q = np.empty(w.shape, dtype=np.int8)
    w_hat = np.empty_like(w)
    for j in range(w.shape[0]):
        q_j, quantized_j = _quantize_work_column(work, j, scales, group_size)
        q[j] = q_j
        w_hat[j] = quantized_j
        error = (work[:, j] - quantized_j) / u[j, j]
        work[:, j:] -= error[:, None] * u[j, j:][None, :]
    return q, scales, w_hat


def gptq_quantize_suffix_reference(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    damping: float = 0.01,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent suffix-H-inverse reference for checking the U orientation.

    At each input column j this recomputes ``F=inv(H[j:,j:])`` and applies
    ``(current-q) * F[0,:]/F[0,0]`` to the remaining columns.  It is slower
    than the U implementation but makes the Schur/suffix relation explicit.
    """
    w, scales, _ = _prepare_gptq(x, w, group_size, damping, token_mask, padding_mask)
    x = _as_float_array("X", x, 2)
    group_size = _check_group_size(group_size)
    mask = _valid_rows(x, token_mask, padding_mask)
    x_cal = x[mask]
    hessian = 2.0 * (x_cal.T @ x_cal) / x_cal.shape[0] + float(damping) * np.eye(w.shape[0])
    work = w.T.copy()
    q = np.empty(w.shape, dtype=np.int8)
    w_hat = np.empty_like(w)
    for j in range(w.shape[0]):
        q_j, quantized_j = _quantize_work_column(work, j, scales, group_size)
        q[j] = q_j
        w_hat[j] = quantized_j
        suffix = hessian[j:, j:]
        try:
            with np.errstate(over="raise", invalid="raise"):
                f = np.linalg.inv(suffix)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("suffix GPTQ Hessian inverse is invalid") from exc
        direction = f[0, :] / f[0, 0]
        work[:, j:] -= (work[:, j] - quantized_j)[:, None] * direction[None, :]
    return q, scales, w_hat


def gptq_quantize_blocked(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    block_size: int,
    damping: float = 0.01,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocked lazy GPTQ: update W1/Err inside a block, then update its tail.

    For block ``[i1:i2]`` the external update is exactly
    ``work[:,i2:] -= Err @ U[i1:i2,i2:]``.  block_size=1 is the sequential
    route; larger blocks expose the lazy-GPTQ bookkeeping without act-order.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    w, scales, u = _prepare_gptq(x, w, group_size, damping, token_mask, padding_mask)
    group_size = _check_group_size(group_size)
    block_size = int(block_size)
    work = w.T.copy()
    q = np.empty(w.shape, dtype=np.int8)
    w_hat = np.empty_like(w)
    for i1 in range(0, w.shape[0], block_size):
        i2 = min(i1 + block_size, w.shape[0])
        w1 = work[:, i1:i2].copy()
        err = np.zeros_like(w1)
        for local, j in enumerate(range(i1, i2)):
            q_j, quantized_j = _quantize_work_column(
                w1, local, scales, group_size, scale_column=j
            )
            q[j] = q_j
            w_hat[j] = quantized_j
            e = (w1[:, local] - quantized_j) / u[j, j]
            err[:, local] = e
            w1[:, local:] -= e[:, None] * u[j, j:i2][None, :]
        if i2 < w.shape[0]:
            work[:, i2:] -= err @ u[i1:i2, i2:]
    return q, scales, w_hat


def _validate_storage_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("q must be a non-empty 2-D array")
    if not np.issubdtype(q.dtype, np.integer):
        raise ValueError("q must be an integer array")
    if np.any(q < _STORAGE_QMIN) or np.any(q > _STORAGE_QMAX):
        raise ValueError("stored signed INT4 values must be in [-8, 7]")
    return q.astype(np.int8, copy=False)


def pack_signed_int4(q: np.ndarray) -> np.ndarray:
    """Pack q[I,O] as packed[O, ceil(I/2)], low input nibble first."""
    q = _validate_storage_q(q)
    input_size, output_size = q.shape
    packed = np.zeros((output_size, (input_size + 1) // 2), dtype=np.uint8)
    for i in range(input_size):
        nibble = (q[i].astype(np.int16) & 0xF).astype(np.uint8)
        if i % 2 == 0:
            packed[:, i // 2] |= nibble
        else:
            packed[:, i // 2] |= nibble << 4
    return packed


def _sign_extend_nibble(nibble: np.ndarray) -> np.ndarray:
    nibble = np.asarray(nibble, dtype=np.int16)
    return np.where(nibble >= 8, nibble - 16, nibble).astype(np.int8)


def unpack_signed_int4(packed: np.ndarray, input_size: int) -> np.ndarray:
    """Unpack the low-then-high input nibbles and sign-extend them."""
    packed = np.asarray(packed)
    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError("packed must be a 2-D uint8 array")
    if isinstance(input_size, bool) or not isinstance(input_size, (int, np.integer)) or input_size <= 0:
        raise ValueError("input_size must be a positive integer")
    expected_bytes = (int(input_size) + 1) // 2
    if packed.shape[1] != expected_bytes:
        raise ValueError(f"packed must have {expected_bytes} bytes per output channel")
    out = np.empty((int(input_size), packed.shape[0]), dtype=np.int8)
    for i in range(int(input_size)):
        nibble = packed[:, i // 2] & (0xF if i % 2 == 0 else 0xF0)
        if i % 2:
            nibble = nibble >> 4
        out[i] = _sign_extend_nibble(nibble)
    return out


def packed_int4_matmul(x: np.ndarray, packed: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """Compute X @ dequant(W) from packed storage without materializing W.

    The intentionally plain loops expose scale addressing at group boundaries:
    each input column contributes to one group's scale row and one output
    channel.  Only a single input-column q vector is decoded at a time.
    """
    x = _as_float_array("X", x, 2)
    packed = np.asarray(packed)
    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError("packed must be a 2-D uint8 array")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must be non-empty")
    input_size = x.shape[1]
    output_size = packed.shape[0]
    expected_bytes = (input_size + 1) // 2
    if packed.shape[1] != expected_bytes:
        raise ValueError("packed byte count does not match X input size")
    scales = _as_float_array("scales", scales, 2)
    group_size = _check_group_size(group_size)
    expected_scales = (_group_count(input_size, group_size), output_size)
    if scales.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, got {scales.shape}")
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")
    output = np.zeros((x.shape[0], output_size), dtype=np.float64)
    for group in range(expected_scales[0]):
        start, end = group * group_size, min((group + 1) * group_size, input_size)
        for i in range(start, end):
            nibble = packed[:, i // 2] & (0xF if i % 2 == 0 else 0xF0)
            if i % 2:
                nibble = nibble >> 4
            q_i = _sign_extend_nibble(nibble).astype(np.float64)
            output += x[:, i, None] * (q_i[None, :] * scales[group][None, :])
    return output


def _rtn_output(x: np.ndarray, w: np.ndarray, group_size: int) -> np.ndarray:
    q, scales = quantize_int4_group(w, group_size)
    return x @ dequantize_int4_group(q, scales, group_size)


class QuantizationLabTests(unittest.TestCase):
    def test_calibration_masks_and_statistics(self) -> None:
        x = np.array([[1, 2], [100, 100], [3, -4]], dtype=np.float64)
        stats = calibration_statistics(x, token_mask=[1, 0, 1])
        np.testing.assert_allclose(stats.meanabs, [2, 3])
        np.testing.assert_allclose(stats.maxabs, [3, 4])
        np.testing.assert_allclose(stats.gram, np.array([[10, -10], [-10, 20]]))
        self.assertEqual(stats.valid_tokens, 2)
        with self.assertRaises(ValueError):
            calibration_statistics(x, padding_mask=[1, 1, 1])
        with self.assertRaises(ValueError):
            calibration_statistics(np.empty((2, 0)))

    def test_group_quantization_tail_zero_and_ties_even(self) -> None:
        w = np.array([[0, 1], [0, -1], [2, 0], [0, 0], [3, -3]], dtype=np.float64)
        q, scales = quantize_int4_group(w, 2)
        self.assertEqual(q.shape, (5, 2))
        self.assertEqual(scales.shape, (3, 2))
        self.assertTrue(np.all((q >= -7) & (q <= 7)))
        np.testing.assert_allclose(scales[0], [1, 1 / 7])
        np.testing.assert_allclose(scales[1], [2 / 7, 1])
        np.testing.assert_allclose(scales[2], [3 / 7, 3 / 7])
        zero_q, zero_s = quantize_int4_group(np.zeros((3, 2)), 2)
        np.testing.assert_array_equal(zero_q, 0)
        np.testing.assert_array_equal(zero_s, 1)
        tie_q, _ = quantize_int4_group(
            np.array([[0.5 / 7], [1.0]], dtype=np.float64), 2
        )
        self.assertEqual(int(tie_q[0, 0]), 0)  # np.rint uses ties-to-even.

    def test_validation_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            quantize_int4_group(np.array([[np.nan]]), 1)
        with self.assertRaises(ValueError):
            calibration_statistics(np.array([[np.inf]]))
        with self.assertRaises(ValueError):
            dequantize_int4_group(np.array([[-8]], dtype=np.int8), np.ones((1, 1)), 1)
        with self.assertRaises(ValueError):
            pack_signed_int4(np.array([[8]], dtype=np.int8))
        with self.assertRaises(ValueError):
            packed_int4_matmul(np.ones((1, 1)), np.zeros((1, 1), dtype=np.uint8), np.ones((1, 1)), 0)
        with self.assertRaises(ValueError):
            calibration_statistics(np.array([[1e308, 1e308], [1e308, -1e308]]))
        subnormal = np.array([[np.nextafter(0.0, 1.0)]])
        _, subnormal_scale = quantize_int4_group(subnormal, 1)
        self.assertGreater(subnormal_scale[0, 0], 0.0)

    def test_smoothquant_is_distinct_and_invariant(self) -> None:
        x = np.array([[1, -2, 0], [2, 1, 0]], dtype=np.float64)
        w = np.array([[1, 2], [3, -1], [0, 0]], dtype=np.float64)
        xs, ws, scale = apply_smoothquant(x, w, 0.5)
        self.assertEqual(scale.shape, (3,))
        np.testing.assert_allclose(x @ w, xs @ ws)
        self.assertEqual(scale[2], 1.0)
        with self.assertRaises(ValueError):
            smoothquant_scales(x, w, 1.1)

    def test_awq_style_is_calibration_only_and_has_identity(self) -> None:
        x = np.array([[1, 2], [2, -1], [50, 0]], dtype=np.float64)
        w = np.array([[1.3, -0.4], [0.7, 2.2]], dtype=np.float64)
        result = awq_style_search(x, w, 1)
        self.assertEqual(len(result.candidate_mse), 21)  # ones + 20 ratios
        self.assertLessEqual(result.best_mse, result.baseline_mse + 1e-15)
        np.testing.assert_allclose(result.calibration_output, x @ w)
        self.assertGreaterEqual(result.ratio is None or result.ratio >= 0, True)
        # A huge held-out row is not accepted by the API and cannot affect the
        # selected calibration score.
        result_again = awq_style_search(x, w, 1)
        self.assertEqual(result.ratio, result_again.ratio)

    def test_gptq_matches_rtn_for_scalar_diagonal_h(self) -> None:
        x = np.eye(3, dtype=np.float64)
        w = np.array([[1.1, -2.2], [0.6, 1.7], [-1.8, 0.3]], dtype=np.float64)
        q_rtn, s_rtn = quantize_int4_group(w, 3)
        q_gptq, s_gptq, w_hat = gptq_quantize(x, w, 3, damping=0.25)
        np.testing.assert_array_equal(q_gptq, q_rtn)
        np.testing.assert_allclose(s_gptq, s_rtn)
        np.testing.assert_allclose(w_hat, dequantize_int4_group(q_gptq, s_gptq, 3))

    def test_gptq_matches_independent_suffix_inverse_in_3_and_5d(self) -> None:
        # This catches a transposed/lower-Cholesky update: both q and W_hat
        # must agree with an independently recomputed suffix inverse.
        rng = np.random.default_rng(44)
        for input_size, group_size in ((3, 2), (5, 3)):
            x = rng.normal(size=(8, input_size))
            if input_size > 1:
                x[:, 1] = 0.8 * x[:, 0] + 0.2 * rng.normal(size=8)
            w = rng.normal(size=(input_size, 3))
            sequential = gptq_quantize(x, w, group_size, damping=0.07)
            suffix = gptq_quantize_suffix_reference(x, w, group_size, damping=0.07)
            np.testing.assert_array_equal(sequential[0], suffix[0])
            np.testing.assert_allclose(sequential[1], suffix[1], rtol=0, atol=0)
            np.testing.assert_allclose(sequential[2], suffix[2], rtol=1e-12, atol=1e-12)
            self.assertGreater(np.count_nonzero(w - sequential[2]), 0)

    def test_gptq_blocked_matches_sequential_for_all_block_sizes(self) -> None:
        rng = np.random.default_rng(45)
        x = rng.normal(size=(9, 5))
        x[:, 1] = 0.7 * x[:, 0] + 0.3 * rng.normal(size=9)
        w = rng.normal(size=(5, 4))
        sequential = gptq_quantize(x, w, group_size=3, damping=0.05)
        for block_size in (1, 2, 3, 5):
            blocked = gptq_quantize_blocked(
                x, w, group_size=3, block_size=block_size, damping=0.05
            )
            np.testing.assert_array_equal(blocked[0], sequential[0])
            np.testing.assert_allclose(blocked[1], sequential[1], rtol=0, atol=0)
            np.testing.assert_allclose(blocked[2], sequential[2], rtol=1e-12, atol=1e-12)

    def test_gptq_two_dimensional_constraint_is_not_rtn_assertion(self) -> None:
        # Correlated X gives a non-diagonal Hessian.  This fixed example checks
        # the independent quadratic objective, not a claim that GPTQ wins
        # every task (RTN/search/GPTQ are deliberately not ordered globally).
        x = np.array([
            [1.3650570351779499, 1.2695821029678758],
            [0.3034397358351804, 0.27941770249447323],
            [-0.5302055914390346, -0.415195246916246],
            [-1.2703985159995006, -1.2004126047698815],
        ])
        w = np.array([[0.4637638409193796, -0.4237360698338129],
                      [-0.7219123202213226, 0.03326465564619614]])
        q_rtn, s_rtn = quantize_int4_group(w, 2)
        q_gptq, s_gptq, _ = gptq_quantize(x, w, 2, damping=0.1)
        h = 2 * (x.T @ x) / x.shape[0] + 0.1 * np.eye(2)
        delta_rtn = (w - dequantize_int4_group(q_rtn, s_rtn, 2)).T
        delta_gptq = (w - dequantize_int4_group(q_gptq, s_gptq, 2)).T
        objective_rtn = float(np.trace(delta_rtn @ h @ delta_rtn.T))
        objective_gptq = float(np.trace(delta_gptq @ h @ delta_gptq.T))
        self.assertLessEqual(objective_gptq, objective_rtn + 1e-12)
        self.assertTrue(np.all((q_gptq >= -7) & (q_gptq <= 7)))

    def test_packing_all_256_signed_nibble_pairs(self) -> None:
        pairs = np.array(
            [[a, b] for a in range(-8, 8) for b in range(-8, 8)], dtype=np.int8
        ).T  # q[I=2, O=256]: all 16*16 adjacent nibble pairs.
        packed = pack_signed_int4(pairs)
        unpacked = unpack_signed_int4(packed, 2)
        np.testing.assert_array_equal(unpacked, pairs)

    def test_packed_matmul_odd_cross_group_negative_and_tail(self) -> None:
        # Five input columns, group tail of one, negative values, and both
        # nibble positions are exercised here.
        q = np.array(
            [[-7, 1], [3, -2], [-1, 7], [6, -6], [-4, 2]], dtype=np.int8
        )
        scales = np.array([[0.5, 0.25], [1.0, 2.0], [0.75, 1.5]])
        x = np.array([[1, -2, 3, -4, 5], [-1, 0.5, 0, 2, -3]], dtype=np.float64)
        packed = pack_signed_int4(q)
        got = packed_int4_matmul(x, packed, scales, 2)
        expected = x @ dequantize_int4_group(q, scales, 2)
        np.testing.assert_allclose(got, expected)
        self.assertEqual(packed.shape, (2, 3))

    def test_zero_calibration_and_zero_input_channel_strategy(self) -> None:
        # i=2 and i=3 share a byte but use different scales when G=3.
        boundary_q = np.array([[-1], [2], [-7], [7], [3]], dtype=np.int8)
        boundary_scales = np.array([[.5], [2.]])
        result = packed_int4_matmul(np.ones((1, 5)), pack_signed_int4(boundary_q), boundary_scales, 3)
        np.testing.assert_array_equal(result, [[17.]])
        x = np.zeros((4, 3), dtype=np.float64)
        w = np.array([[1, -1], [0, 2], [3, 0]], dtype=np.float64)
        stats = calibration_statistics(x)
        np.testing.assert_array_equal(stats.gram, 0)
        q, scales, w_hat = gptq_quantize(x, w, 2, damping=0.5)
        self.assertTrue(np.all(np.isfinite(w_hat)))
        self.assertTrue(np.all((q >= -7) & (q <= 7)))
        # The zero activation channel is retained; positive damping, rather
        # than dropping/reordering the channel, keeps H invertible.
        np.testing.assert_allclose(w_hat, dequantize_int4_group(q, scales, 2))


def demo() -> None:
    rng = np.random.default_rng(20260911)
    t_cal, t_eval, i, o, group = 7, 6, 5, 3, 2
    x_cal = rng.normal(size=(t_cal, i))
    x_eval = rng.normal(size=(t_eval, i))
    w = rng.normal(size=(i, o))
    token_mask = np.array([1, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    x_valid = x_cal[token_mask.astype(bool)]
    stats = calibration_statistics(x_cal, token_mask=token_mask)

    q_rtn, scales_rtn = quantize_int4_group(w, group)
    w_rtn = dequantize_int4_group(q_rtn, scales_rtn, group)
    search = awq_style_search(x_cal, w, group, token_mask=token_mask)
    q_awq, scales_awq = quantize_int4_group(w * search.scale[:, None], group)
    w_awq = dequantize_int4_group(q_awq, scales_awq, group) / search.scale[:, None]
    q_gptq, scales_gptq, w_gptq = gptq_quantize(x_valid, w, group, damping=0.05)

    y_cal = x_valid @ w
    y_eval = x_eval @ w
    mse = lambda prediction, target: float(np.mean((prediction - target) ** 2))
    packed = pack_signed_int4(q_gptq)
    y_packed = packed_int4_matmul(x_eval, packed, scales_gptq, group)
    print("Quantization lab demo (NumPy CPU; fixed seed 20260911)")
    print(f"shape Xcal[T,I]={x_cal.shape}, Xeval[T,I]={x_eval.shape}, W[I,O]={w.shape}, group_size={group}")
    print(f"valid calibration tokens={stats.valid_tokens}, zero activation channels={int(np.sum(stats.maxabs == 0))}")
    print(f"RTN MSE: calibration={mse(x_valid @ w_rtn, y_cal):.8f}, heldout={mse(x_eval @ w_rtn, y_eval):.8f}")
    print(f"AWQ-style teaching subset ratio={search.ratio}, MSE: calibration={mse(x_valid @ w_awq, y_cal):.8f}, heldout={mse(x_eval @ w_awq, y_eval):.8f}")
    print(f"GPTQ MSE: calibration={mse(x_valid @ w_gptq, y_cal):.8f}, heldout={mse(x_eval @ w_gptq, y_eval):.8f}")
    print(f"packed-vs-GPTQ heldout max_abs_diff={np.max(np.abs(y_packed - x_eval @ w_gptq)):.8f}")
    print(f"storage bytes: packed-q={packed.nbytes}, scales-float64={scales_gptq.nbytes}, total={packed.nbytes + scales_gptq.nbytes}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="print deterministic numerical errors and storage bytes")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.demo:
        demo()
        return 0
    unittest.main(module=__name__, argv=[sys.argv[0]], verbosity=2)
    return 0


if __name__ == "__main__":
    main()
