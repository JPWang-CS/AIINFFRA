"""CPU-only end-to-end PTQ lab for a small parallel-residual decoder.

All linear weights use W[input, output], matching quantization_lab.py.
The archived INT4 layout is output-major packed[output, ceil(input/2)].
This is a teaching reference, not a production LLM quantizer or GPU kernel.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

import quantization_lab as qlab


FORMAT_NAME = "aiinffra-mini-decoder-int4"
FORMAT_VERSION = 1
QUANT_SCHEME = "symmetric-groupwise-int4-rtn-with-calibration-clipping-search"
CLIP_RATIOS = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 23
    hidden_size: int = 12
    intermediate_size: int = 20
    num_heads: int = 3
    num_layers: int = 2
    max_seq_len: int = 8

    def validate(self) -> None:
        values = asdict(self)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values.values()):
            raise ValueError("all model dimensions must be positive integers")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


def _matrix_names(config: ModelConfig) -> list[str]:
    names: list[str] = []
    for layer in range(config.num_layers):
        prefix = f"layers.{layer}."
        names.extend(prefix + suffix for suffix in (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ))
    names.append("lm_head")
    return names


def _expected_matrix_shape(name: str, config: ModelConfig) -> tuple[int, int]:
    if name == "lm_head":
        return config.hidden_size, config.vocab_size
    _, layer_text, suffix = name.split(".")
    if int(layer_text) >= config.num_layers:
        raise ValueError(f"weight name has invalid layer index: {name}")
    if suffix in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return config.hidden_size, config.hidden_size
    if suffix in ("gate_proj", "up_proj"):
        return config.hidden_size, config.intermediate_size
    if suffix == "down_proj":
        return config.intermediate_size, config.hidden_size
    raise ValueError(f"unknown matrix tensor: {name}")


class MiniDecoder:
    """Small causal decoder with parallel attention/MLP residual branches.

    RMSNorm output is shared by Q/K/V and gate/up. One diagonal input fold must
    therefore update the norm gain and every one of those five consumers.
    """

    def __init__(self, config: ModelConfig, params: dict[str, np.ndarray]):
        config.validate()
        self.config = config
        self.params = params

    @classmethod
    def random(cls, config: ModelConfig, seed: int = 7) -> "MiniDecoder":
        config.validate()
        rng = np.random.default_rng(seed)
        h, m = config.hidden_size, config.intermediate_size
        params: dict[str, np.ndarray] = {
            "embedding": (rng.standard_normal((config.vocab_size, h)) * 0.20).astype(np.float64),
            "final_gamma": np.ones(h, dtype=np.float64),
        }
        params["embedding"][0] = 0.0
        for layer in range(config.num_layers):
            p = f"layers.{layer}."
            params[p + "norm_gamma"] = np.ones(h, dtype=np.float64)
            for name, shape in (
                ("q_proj", (h, h)), ("k_proj", (h, h)), ("v_proj", (h, h)),
                ("o_proj", (h, h)), ("gate_proj", (h, m)), ("up_proj", (h, m)),
                ("down_proj", (m, h)),
            ):
                params[p + name] = (rng.standard_normal(shape) / math.sqrt(shape[0])).astype(np.float64)
        params["lm_head"] = (rng.standard_normal((h, config.vocab_size)) / math.sqrt(h)).astype(np.float64)
        return cls(config, params)

    def clone(self) -> "MiniDecoder":
        return MiniDecoder(self.config, {name: value.copy() for name, value in self.params.items()})

    @staticmethod
    def _rms_norm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * gamma

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def forward(
        self,
        token_ids: np.ndarray,
        token_mask: np.ndarray,
        padding_mask: np.ndarray,
        capture: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        token_ids = np.asarray(token_ids)
        token_mask = np.asarray(token_mask, dtype=bool)
        padding_mask = np.asarray(padding_mask, dtype=bool)
        if token_ids.ndim != 2 or token_mask.shape != token_ids.shape or padding_mask.shape != token_ids.shape:
            raise ValueError("token_ids, token_mask, and padding_mask must have the same [batch, seq] shape")
        if not np.issubdtype(token_ids.dtype, np.integer) or np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):
            raise ValueError("token_ids are outside the vocabulary")
        if token_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        valid = token_mask & ~padding_mask
        if np.any(np.sum(valid, axis=1) == 0):
            raise ValueError("each sequence must contain at least one valid token")
        h = self.params["embedding"][token_ids].copy()
        captured: dict[str, np.ndarray] = {}
        batch, seq = token_ids.shape
        head_dim = self.config.hidden_size // self.config.num_heads
        causal = np.tril(np.ones((seq, seq), dtype=bool))

        for layer in range(self.config.num_layers):
            p = f"layers.{layer}."
            normed = self._rms_norm(h, self.params[p + "norm_gamma"])
            if capture:
                captured[p + "shared_in"] = normed.reshape(-1, self.config.hidden_size).copy()
            q = normed @ self.params[p + "q_proj"]
            k = normed @ self.params[p + "k_proj"]
            v = normed @ self.params[p + "v_proj"]
            q = q.reshape(batch, seq, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, seq, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, seq, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
            allowed = causal[None, None, :, :] & valid[:, None, None, :]
            scores = np.where(allowed, scores, -1.0e30)
            scores -= np.max(scores, axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= np.sum(probs, axis=-1, keepdims=True)
            probs *= valid[:, None, :, None]
            context = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, seq, self.config.hidden_size)
            if capture:
                captured[p + "o_proj"] = context.reshape(-1, self.config.hidden_size).copy()
            attention = context @ self.params[p + "o_proj"]

            gate = self._silu(normed @ self.params[p + "gate_proj"])
            up = normed @ self.params[p + "up_proj"]
            mlp_hidden = gate * up
            if capture:
                captured[p + "down_proj"] = mlp_hidden.reshape(-1, self.config.intermediate_size).copy()
            mlp = mlp_hidden @ self.params[p + "down_proj"]
            h = h + attention + mlp

        final = self._rms_norm(h, self.params["final_gamma"])
        if capture:
            captured["lm_head"] = final.reshape(-1, self.config.hidden_size).copy()
        logits = final @ self.params["lm_head"]
        if capture:
            return logits, captured
        return logits


@dataclass
class TokenBatch:
    token_ids: np.ndarray
    token_mask: np.ndarray
    padding_mask: np.ndarray


def make_tokens(
    config: ModelConfig,
    seed: int,
    count: int = 16,
    sequence_length: int = 7,
    shifted: bool = False,
) -> TokenBatch:
    """Generate deterministic, independently seeded calibration/eval tokens."""
    if sequence_length > config.max_seq_len or count <= 0:
        raise ValueError("count must be positive and sequence_length must fit the model")
    rng = np.random.default_rng(seed)
    tokens = np.zeros((count, sequence_length), dtype=np.int64)
    token_mask = np.zeros_like(tokens, dtype=bool)
    for row in range(count):
        length = int(rng.integers(2, sequence_length + 1))
        if shifted:
            low = max(1, config.vocab_size // 2)
            values = rng.integers(low, config.vocab_size, size=length)
        else:
            values = rng.integers(1, config.vocab_size, size=length)
        tokens[row, :length] = values
        token_mask[row, :length] = True
    return TokenBatch(tokens, token_mask, ~token_mask)


def collect_calibration_inputs(
    model: MiniDecoder, data: TokenBatch, batch_size: int = 4,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Collect every layer's inputs in one set of original-FP-model passes.

    This is not sequential quantization replay: deeper layers are observed on
    the unquantized model's activations, not on errors accumulated from already
    quantized earlier layers.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows: dict[str, list[np.ndarray]] = {}
    token_rows: list[np.ndarray] = []
    padding_rows: list[np.ndarray] = []
    for start in range(0, data.token_ids.shape[0], batch_size):
        end = min(start + batch_size, data.token_ids.shape[0])
        _, captured = model.forward(
            data.token_ids[start:end], data.token_mask[start:end], data.padding_mask[start:end], capture=True
        )
        for name, x in captured.items():
            rows.setdefault(name, []).append(x)
        token_rows.append(data.token_mask[start:end].reshape(-1))
        padding_rows.append(data.padding_mask[start:end].reshape(-1))
    token_mask = np.concatenate(token_rows)
    padding_mask = np.concatenate(padding_rows)
    if not np.any(token_mask & ~padding_mask):
        raise ValueError("calibration data contains no valid tokens")
    return {
        name: (np.concatenate(values), token_mask.copy(), padding_mask.copy())
        for name, values in rows.items()
    }


@dataclass
class ClipChoice:
    q: np.ndarray
    scales: np.ndarray
    ratio: float
    baseline_mse: float
    best_mse: float
    candidate_mse: tuple[float, ...]


def _clip_and_quantize(w: np.ndarray, group_size: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.empty_like(w, dtype=np.float64)
    for start in range(0, w.shape[0], group_size):
        end = min(start + group_size, w.shape[0])
        limit = np.max(np.abs(w[start:end]), axis=0, keepdims=True) * ratio
        clipped[start:end] = np.clip(w[start:end], -limit, limit)
    return qlab.quantize_int4_group(clipped, group_size)


def search_clipping(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    token_mask: np.ndarray,
    padding_mask: np.ndarray,
    ratios: tuple[float, ...] = CLIP_RATIOS,
) -> ClipChoice:
    """Teaching-only per-layer group clipping search against calibration MSE."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("expected X[T,I] and W[I,O]")
    valid = np.asarray(token_mask, dtype=bool).reshape(-1) & ~np.asarray(padding_mask, dtype=bool).reshape(-1)
    if valid.size != x.shape[0] or not np.any(valid):
        raise ValueError("mask shape mismatch or no valid calibration tokens")
    if not ratios or any(not np.isfinite(r) or not 0.0 < r <= 1.0 for r in ratios):
        raise ValueError("clip ratios must be finite values in (0, 1]")
    if not np.isclose(ratios[0], 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("the first clip ratio must be 1.0 so baseline_mse is the unclipped candidate")
    x_valid = x[valid]
    reference = x_valid @ w
    errors: list[float] = []
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for ratio in ratios:
        q, scales = _clip_and_quantize(w, group_size, ratio)
        restored = qlab.dequantize_int4_group(q, scales, group_size)
        errors.append(float(np.mean((x_valid @ restored - reference) ** 2)))
        candidates.append((q, scales))
    best = int(np.argmin(errors))
    return ClipChoice(candidates[best][0], candidates[best][1], float(ratios[best]), errors[0], errors[best], tuple(errors))


@dataclass
class QuantizationBundle:
    folded_model: MiniDecoder
    qweights: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    group_size: int
    fold_scales: dict[str, np.ndarray]
    clip_ratios: dict[str, float]
    calibration_valid_tokens: int
    layer_calibration_mse: dict[str, dict[str, float]]


def quantize_full_model(
    model: MiniDecoder,
    calibration: TokenBatch,
    group_size: int = 4,
    clip_ratios: tuple[float, ...] = CLIP_RATIOS,
) -> QuantizationBundle:
    """Calibrate all layers, fold shared inputs, then pack-ready INT4 weights."""
    original_inputs = collect_calibration_inputs(model, calibration)
    folded = model.clone()
    fold_scales: dict[str, np.ndarray] = {}
    clip_choices: dict[str, ClipChoice] = {}

    for layer in range(model.config.num_layers):
        prefix = f"layers.{layer}."
        x, token_mask, padding_mask = original_inputs[prefix + "shared_in"]
        consumer_names = [prefix + n for n in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")]
        joint_w = np.concatenate([model.params[name] for name in consumer_names], axis=1)
        # One AWQ-style scale is searched over every consumer of this shared norm.
        selection = qlab.awq_style_search(x, joint_w, group_size, token_mask, padding_mask)
        scale = selection.scale
        folded.params[prefix + "norm_gamma"] = model.params[prefix + "norm_gamma"] / scale
        x_folded = x / scale[None, :]
        fold_scales[prefix + "shared_in"] = scale.copy()
        for name in consumer_names:
            folded.params[name] = model.params[name] * scale[:, None]
            clip_choices[name] = search_clipping(
                x_folded, folded.params[name], group_size, token_mask, padding_mask, clip_ratios
            )

        for suffix in ("o_proj", "down_proj"):
            name = prefix + suffix
            activation_key = name
            x_single, tm, pm = original_inputs[activation_key]
            clip_choices[name] = search_clipping(x_single, model.params[name], group_size, tm, pm, clip_ratios)
        # The shared norm also feeds attention and MLP in this parallel-residual decoder.

    x, tm, pm = original_inputs["lm_head"]
    clip_choices["lm_head"] = search_clipping(x, model.params["lm_head"], group_size, tm, pm, clip_ratios)
    qweights = {name: choice.q for name, choice in clip_choices.items()}
    scales = {name: choice.scales for name, choice in clip_choices.items()}
    metrics = {
        name: {"baseline_mse": choice.baseline_mse, "best_mse": choice.best_mse}
        for name, choice in clip_choices.items()
    }
    valid_tokens = int(np.count_nonzero(calibration.token_mask & ~calibration.padding_mask))
    return QuantizationBundle(
        folded, qweights, scales, group_size, fold_scales,
        {name: choice.ratio for name, choice in clip_choices.items()}, valid_tokens, metrics,
    )


def materialize_quantized_model(bundle: QuantizationBundle) -> MiniDecoder:
    model = bundle.folded_model.clone()
    for name in _matrix_names(model.config):
        model.params[name] = qlab.dequantize_int4_group(bundle.qweights[name], bundle.scales[name], bundle.group_size)
    return model


def _archive_key(prefix: str, name: str) -> str:
    return prefix + "__" + name.replace(".", "__")


def _float_tensor_names(config: ModelConfig) -> list[str]:
    names = ["embedding", "final_gamma"]
    for layer in range(config.num_layers):
        names.append(f"layers.{layer}.norm_gamma")
    return names


def save_quantized_checkpoint(bundle: QuantizationBundle, path: str | Path) -> Path:
    """Write an NPZ checkpoint without overwriting an existing user file."""
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent directory does not exist: {target.parent}")
    config = bundle.folded_model.config
    matrices = _matrix_names(config)
    metadata = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model": asdict(config),
        "quantization": {
            "scheme": QUANT_SCHEME,
            "group_size": bundle.group_size,
            "weight_orientation": "W[input,output]",
            "packed_orientation": "packed[output,ceil(input/2)]",
            "nibble_order": "low-input-index-first",
            "scale_shape": "[ceil(input/group_size),output]",
            "calibration_valid_tokens": bundle.calibration_valid_tokens,
            "clipping_is_full_awq": False,
        },
        "tensors": {
            name: {
                "logical_shape": list(_expected_matrix_shape(name, config)),
                "packed_key": _archive_key("packed", name),
                "scales_key": _archive_key("scales", name),
            }
            for name in matrices
        },
        "float_tensors": {
            name: {"key": _archive_key("float", name), "shape": list(bundle.folded_model.params[name].shape)}
            for name in _float_tensor_names(config)
        },
        "fold_scales": {name: value.tolist() for name, value in bundle.fold_scales.items()},
        "clip_ratios": bundle.clip_ratios,
        "layer_calibration_mse": bundle.layer_calibration_mse,
    }
    arrays: dict[str, np.ndarray] = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    for name in matrices:
        arrays[_archive_key("packed", name)] = qlab.pack_signed_int4(bundle.qweights[name])
        arrays[_archive_key("scales", name)] = bundle.scales[name]
    for name in _float_tensor_names(config):
        arrays[_archive_key("float", name)] = bundle.folded_model.params[name]
    # Exclusive create guarantees that an existing destination is never replaced.
    # On write failure, leave cleanup to the caller rather than unlinking a path
    # that another process could have replaced after this file was opened.
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return target


def load_quantized_checkpoint(path: str | Path, expected_config: ModelConfig | None = None) -> MiniDecoder:
    """Validate packed storage, then materialize dequantized floating weights.

    The returned NumPy model is a correctness reference. It does not execute
    packed INT4 dot products and makes no low-bit inference speed claim.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("checkpoint metadata is missing")
        try:
            metadata = json.loads(str(archive["metadata"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint metadata is not valid JSON") from exc
        if metadata.get("format_name") != FORMAT_NAME:
            raise ValueError("unsupported checkpoint format_name")
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format_version: {metadata.get('format_version')}")
        try:
            config = ModelConfig(**metadata["model"])
            config.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint model metadata is invalid") from exc
        if expected_config is not None and asdict(config) != asdict(expected_config):
            raise ValueError("checkpoint model shape does not match expected_config")
        quant = metadata.get("quantization", {})
        expected_quant_fields = {
            "scheme": QUANT_SCHEME,
            "weight_orientation": "W[input,output]",
            "packed_orientation": "packed[output,ceil(input/2)]",
            "nibble_order": "low-input-index-first",
            "scale_shape": "[ceil(input/group_size),output]",
            "clipping_is_full_awq": False,
        }
        for field, expected_value in expected_quant_fields.items():
            if quant.get(field) != expected_value:
                raise ValueError(f"unsupported or invalid quantization metadata: {field}")
        group_size = quant.get("group_size")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("checkpoint group_size must be a positive integer")
        matrices = _matrix_names(config)
        tensor_meta = metadata.get("tensors")
        if not isinstance(tensor_meta, dict) or set(tensor_meta) != set(matrices):
            raise ValueError("checkpoint tensor metadata names do not match model architecture")
        float_names = _float_tensor_names(config)
        float_meta = metadata.get("float_tensors")
        if not isinstance(float_meta, dict) or set(float_meta) != set(float_names):
            raise ValueError("checkpoint float tensor metadata names do not match model architecture")
        expected_keys = {"metadata"}
        expected_keys.update(_archive_key("packed", name) for name in matrices)
        expected_keys.update(_archive_key("scales", name) for name in matrices)
        expected_keys.update(_archive_key("float", name) for name in float_names)
        if set(archive.files) != expected_keys:
            raise ValueError("checkpoint array keys do not match metadata")

        params: dict[str, np.ndarray] = {}
        for name in float_names:
            info = float_meta[name]
            shape = tuple(info.get("shape", ()))
            expected_shape = (config.vocab_size, config.hidden_size) if name == "embedding" else (config.hidden_size,)
            if name.endswith("norm_gamma"):
                expected_shape = (config.hidden_size,)
            if shape != expected_shape or info.get("key") != _archive_key("float", name):
                raise ValueError(f"invalid float tensor metadata for {name}")
            value = np.asarray(archive[info["key"]], dtype=np.float64)
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid float tensor shape or values for {name}")
            params[name] = value.copy()

        for name in matrices:
            info = tensor_meta[name]
            logical_shape = _expected_matrix_shape(name, config)
            if tuple(info.get("logical_shape", ())) != logical_shape:
                raise ValueError(f"logical tensor shape mismatch for {name}")
            packed_key = _archive_key("packed", name)
            scales_key = _archive_key("scales", name)
            if info.get("packed_key") != packed_key or info.get("scales_key") != scales_key:
                raise ValueError(f"invalid packed tensor keys for {name}")
            input_size, output_size = logical_shape
            packed = np.asarray(archive[packed_key])
            expected_packed = (output_size, (input_size + 1) // 2)
            if packed.dtype != np.uint8 or packed.shape != expected_packed:
                raise ValueError(f"packed tensor shape/dtype mismatch for {name}: expected uint8{expected_packed}")
            q = qlab.unpack_signed_int4(packed, input_size)
            if np.any(q < -7) or np.any(q > 7):
                raise ValueError(f"quantized values for {name} are outside this quantizer's [-7,7] range")
            scales = np.asarray(archive[scales_key], dtype=np.float64)
            expected_scales = ((input_size + group_size - 1) // group_size, output_size)
            if scales.shape != expected_scales or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError(f"scale tensor shape/values mismatch for {name}: expected {expected_scales}")
            params[name] = qlab.dequantize_int4_group(q, scales, group_size)
    return MiniDecoder(config, params)


def next_token_metrics(model: MiniDecoder, data: TokenBatch) -> dict[str, float]:
    """Shift logits/targets and score only adjacent valid, non-padding tokens."""
    logits = model.forward(data.token_ids, data.token_mask, data.padding_mask)
    valid = data.token_mask & ~data.padding_mask
    predict = valid[:, :-1] & valid[:, 1:]
    if not np.any(predict):
        raise ValueError("next-token evaluation needs at least one adjacent valid token pair")
    shifted_logits = logits[:, :-1, :][predict]
    targets = data.token_ids[:, 1:][predict]
    row_max = np.max(shifted_logits, axis=-1, keepdims=True)
    logsumexp = row_max[:, 0] + np.log(np.sum(np.exp(shifted_logits - row_max), axis=-1))
    target_logits = shifted_logits[np.arange(targets.size), targets]
    nll = float(np.mean(logsumexp - target_logits))
    return {"next_token_count": float(targets.size), "next_token_nll": nll, "perplexity": float(np.exp(nll))}


def evaluate_against(reference: MiniDecoder, candidate: MiniDecoder, data: TokenBatch) -> dict[str, float]:
    ref = reference.forward(data.token_ids, data.token_mask, data.padding_mask)
    got = candidate.forward(data.token_ids, data.token_mask, data.padding_mask)
    valid = data.token_mask & ~data.padding_mask
    diff = got[valid] - ref[valid]
    ref_valid = ref[valid]
    ref_lm = next_token_metrics(reference, data)
    candidate_lm = next_token_metrics(candidate, data)
    return {
        "valid_tokens": float(np.count_nonzero(valid)),
        "logit_mse": float(np.mean(diff * diff)),
        "logit_max_abs": float(np.max(np.abs(diff))),
        "logit_relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref_valid), 1e-15)),
        "top1_agreement": float(np.mean(np.argmax(got[valid], axis=-1) == np.argmax(ref_valid, axis=-1))),
        "reference_next_token_nll": ref_lm["next_token_nll"],
        "candidate_next_token_nll": candidate_lm["next_token_nll"],
        "reference_perplexity": ref_lm["perplexity"],
        "candidate_perplexity": candidate_lm["perplexity"],
        "next_token_count": candidate_lm["next_token_count"],
    }


def shared_branch_counterexample() -> tuple[float, float]:
    """Return exact-fold error and error when only Q is folded in a shared QKV input."""
    rng = np.random.default_rng(91)
    x = rng.standard_normal((5, 4))
    scale = np.array([0.5, 1.5, 0.75, 2.0])
    matrices = [rng.standard_normal((4, 4)) for _ in range(3)]
    q, k, v = [x @ w for w in matrices]
    x_scaled = x / scale
    q_good, k_good, v_good = [x_scaled @ (scale[:, None] * w) for w in matrices]
    q_bad = x_scaled @ (scale[:, None] * matrices[0])
    k_bad, v_bad = x_scaled @ matrices[1], x_scaled @ matrices[2]

    def attend(qq: np.ndarray, kk: np.ndarray, vv: np.ndarray) -> np.ndarray:
        scores = qq @ kk.T / math.sqrt(qq.shape[-1])
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return weights @ vv

    original = attend(q, k, v)
    correct = attend(q_good, k_good, v_good)
    wrong = attend(q_bad, k_bad, v_bad)
    return float(np.max(np.abs(original - correct))), float(np.max(np.abs(original - wrong)))


def run_demo(output_path: str | None = None) -> None:
    config = ModelConfig()
    fp_model = MiniDecoder.random(config, seed=7)
    calibration = make_tokens(config, seed=101, count=16, sequence_length=7)
    heldout = make_tokens(config, seed=707, count=12, sequence_length=7, shifted=True)
    bundle = quantize_full_model(fp_model, calibration, group_size=4)
    folded_metrics = evaluate_against(fp_model, bundle.folded_model, heldout)
    quantized = materialize_quantized_model(bundle)

    if output_path is None:
        with tempfile.TemporaryDirectory(prefix="aiinffra-mini-quant-") as temp_dir:
            path = Path(temp_dir) / "mini_decoder_int4.npz"
            save_quantized_checkpoint(bundle, path)
            loaded = load_quantized_checkpoint(path, expected_config=config)
            show_demo(calibration, heldout, fp_model, folded_metrics, loaded, bundle, path, temporary=True)
    else:
        path = save_quantized_checkpoint(bundle, output_path)
        loaded = load_quantized_checkpoint(path, expected_config=config)
        show_demo(calibration, heldout, fp_model, folded_metrics, loaded, bundle, path, temporary=False)
    # Ensure the in-memory path and serialized roundtrip use the same packed values.
    roundtrip = evaluate_against(quantized, loaded, heldout)
    if roundtrip["logit_max_abs"] > 1e-12:
        raise AssertionError("checkpoint roundtrip changed dequantized model outputs")


def show_demo(
    calibration: TokenBatch,
    heldout: TokenBatch,
    fp_model: MiniDecoder,
    folded_metrics: dict[str, float],
    loaded: MiniDecoder,
    bundle: QuantizationBundle,
    path: Path,
    temporary: bool,
) -> None:
    print(f"model: {bundle.folded_model.config.num_layers}-layer parallel Mini Decoder; W[input,output]; group={bundle.group_size}")
    print(f"calibration valid tokens: {bundle.calibration_valid_tokens}; heldout valid tokens: {int(np.count_nonzero(heldout.token_mask & ~heldout.padding_mask))}")
    print(f"fold-only heldout: max_abs={folded_metrics['logit_max_abs']:.3e} (floating function should be preserved)")
    print(f"quantized heldout (random untrained model; metric-path demo only): {json.dumps(evaluate_against(fp_model, loaded, heldout), sort_keys=True)}")
    print(f"quantized calibration (random untrained model; metric-path demo only): {json.dumps(evaluate_against(fp_model, loaded, calibration), sort_keys=True)}")
    print(f"mean selected clip ratio: {np.mean(list(bundle.clip_ratios.values())):.3f}; per-layer clipping is not full AWQ")
    correct, wrong = shared_branch_counterexample()
    print(f"shared QKV fold: correct max_abs={correct:.3e}; Q-only wrong-fold max_abs={wrong:.3e}")
    print(f"checkpoint: {path}{' (temporary; removed on exit)' if temporary else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run deterministic CPU model calibration/export/load/evaluation")
    parser.add_argument("--output", type=str, help="optional new checkpoint path; existing files are never overwritten")
    args = parser.parse_args()
    run_demo(args.output)


if __name__ == "__main__":
    main()
