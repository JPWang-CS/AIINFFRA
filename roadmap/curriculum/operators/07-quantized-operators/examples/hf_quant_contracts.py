"""NumPy-only contracts shared by the local Hugging Face Llama lab."""

from __future__ import annotations

from typing import Any

import numpy as np


FORMAT_NAME = "aiinffra.llama-int4-overlay"
FORMAT_VERSION = 1


def llama_fold_groups(num_layers: int) -> list[tuple[str, tuple[str, ...]]]:
    """Return the supported HF Llama RMSNorm-to-Linear consumer groups."""
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    groups: list[tuple[str, tuple[str, ...]]] = []
    for index in range(num_layers):
        prefix = f"model.layers.{index}."
        groups.append((prefix + "input_layernorm", tuple(
            prefix + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj")
        )))
        groups.append((prefix + "post_attention_layernorm", tuple(
            prefix + "mlp." + name for name in ("gate_proj", "up_proj")
        )))
    return groups


def valid_token_rows(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Flatten [batch, seq, hidden] and retain exactly non-padding positions."""
    values = np.asarray(hidden)
    mask = np.asarray(attention_mask)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("hidden must be [batch,seq,hidden] and mask [batch,seq]")
    if mask.dtype.kind not in "biuf" or not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must contain only boolean or numeric 0/1 values")
    keep = mask.astype(bool, copy=False)
    if not np.any(keep):
        raise ValueError("attention_mask contains no valid tokens")
    rows = values[keep]
    if not np.all(np.isfinite(rows)):
        raise ValueError("valid hidden states must be finite")
    return rows


def fold_shared_linear_group(
    norm_weight: np.ndarray,
    weights_io: dict[str, np.ndarray],
    scale: np.ndarray,
    norm_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    """Apply gamma/s and W[I,O]*s to every consumer of one normalized tensor."""
    gamma = np.asarray(norm_weight)
    s = np.asarray(scale)
    if gamma.ndim != 1 or s.shape != gamma.shape:
        raise ValueError("norm_weight and scale must be equal-length vectors")
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("fold scales must be positive and finite")
    if not weights_io:
        raise ValueError("a fold group must have at least one consumer")
    folded: dict[str, np.ndarray] = {}
    for name, value in weights_io.items():
        weight = np.asarray(value)
        if weight.ndim != 2 or weight.shape[0] != gamma.size:
            raise ValueError(f"{name}: expected W[I,O] with I={gamma.size}")
        folded[name] = weight * s[:, None]
    beta_out = None
    if norm_bias is not None:
        beta = np.asarray(norm_bias)
        if beta.shape != gamma.shape:
            raise ValueError("norm_bias must match norm_weight")
        beta_out = beta / s
    return gamma / s, beta_out, folded


def symmetric_int4_storage_bytes(
    input_size: int, output_size: int, group_size: int, scale_bytes: int = 4
) -> int:
    """Count packed signed nibbles plus one stored scale per [group, output]."""
    values = (input_size, output_size, group_size, scale_bytes)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError("dimensions, group_size and scale_bytes must be positive")
    packed = output_size * ((input_size + 1) // 2)
    scales = ((input_size + group_size - 1) // group_size) * output_size * scale_bytes
    return packed + scales


def validate_manifest(value: Any) -> dict[str, Any]:
    """Reject incomplete or incompatible teaching-overlay metadata."""
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    required = {
        "format", "format_version", "architecture", "base_model_fingerprint",
        "transformers_version", "tokenizer_fingerprint", "quantization",
        "tensors", "float_exceptions", "fold_groups", "data_fingerprints", "metrics",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"manifest missing fields: {missing}")
    if value["format"] != FORMAT_NAME or value["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported teaching overlay format/version")
    if value["architecture"] != "LlamaForCausalLM":
        raise ValueError("only the declared LlamaForCausalLM topology is supported")
    if not all(isinstance(value[k], str) and value[k] for k in (
        "base_model_fingerprint", "transformers_version", "tokenizer_fingerprint"
    )):
        raise ValueError("base model, Transformers and tokenizer fingerprints are required")
    quant = value["quantization"]
    if not isinstance(quant, dict) or quant.get("bits") != 4:
        raise ValueError("quantization must declare 4-bit weights")
    if quant.get("logical_weight_orientation") != "O,I" or quant.get("math_weight_orientation") != "I,O":
        raise ValueError("unsupported logical/math weight orientation")
    if quant.get("packed_orientation") != "O,ceil(I/2)" or quant.get("nibble_order") != "low-index-first":
        raise ValueError("unsupported packed layout")
    if isinstance(quant.get("group_size"), bool) or not isinstance(quant.get("group_size"), int) or quant["group_size"] <= 0:
        raise ValueError("group_size must be positive")
    tensors = value["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("manifest must describe at least one quantized tensor")
    names: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError("tensor records must be objects")
        for key in ("name", "logical_shape", "packed_shape", "scale_shape", "source_dtype", "fold_group", "packed_key", "scale_key"):
            if key not in tensor:
                raise ValueError(f"tensor record missing {key}")
        if any(not isinstance(tensor[key], str) or not tensor[key] for key in ("packed_key", "scale_key", "fold_group")):
            raise ValueError("tensor array keys and fold_group must be non-empty strings")
        if not isinstance(tensor["name"], str) or tensor["name"] in names:
            raise ValueError("tensor names must be unique strings")
        names.add(tensor["name"])
        logical = tensor["logical_shape"]
        if not isinstance(logical, list) or len(logical) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in logical):
            raise ValueError(f"{tensor['name']}: logical_shape must be [O,I]")
        out_features, in_features = logical
        if tensor["packed_shape"] != [out_features, (in_features + 1) // 2]:
            raise ValueError(f"{tensor['name']}: packed shape disagrees with O,I")
        expected_scales = [(in_features + quant["group_size"] - 1) // quant["group_size"], out_features]
        if tensor["scale_shape"] != expected_scales:
            raise ValueError(f"{tensor['name']}: scale shape disagrees with group_size")
    if not isinstance(value["float_exceptions"], list):
        raise ValueError("float_exceptions must list tensors intentionally left in the base dtype")
    if not isinstance(value["metrics"], dict):
        raise ValueError("metrics must be an object")
    if not isinstance(value["fold_groups"], list) or not isinstance(value["data_fingerprints"], dict):
        raise ValueError("fold_groups and data_fingerprints must be recorded")
    if "calibration" not in value["data_fingerprints"] or "heldout" not in value["data_fingerprints"]:
        raise ValueError("separate calibration and heldout fingerprints are required")
    if value["data_fingerprints"]["calibration"] == value["data_fingerprints"]["heldout"]:
        raise ValueError("calibration and heldout data must be distinct")
    layer_ids = set()
    tensor_norms = {record["name"][:-len(".weight")]: record["fold_group"] for record in tensors}
    for record in tensors:
        parts = record["name"].split(".")
        if len(parts) < 5 or parts[:2] != ["model", "layers"] or not parts[2].isdigit():
            raise ValueError(f"unsupported Llama tensor name: {record['name']}")
        layer_ids.add(int(parts[2]))
    if not layer_ids or layer_ids != set(range(max(layer_ids) + 1)):
        raise ValueError("quantized Llama layers must start at zero and be contiguous")
    if any(not isinstance(group, dict) or not isinstance(group.get("norm"), str)
           or not isinstance(group.get("consumers"), list)
           or not isinstance(group.get("scale_key"), str) or not group.get("scale_key")
           for group in value["fold_groups"]):
        raise ValueError("fold groups require norm, consumer list and scale_key")
    groups_by_norm = {group["norm"]: tuple(group["consumers"]) for group in value["fold_groups"]}
    if len(groups_by_norm) != len(value["fold_groups"]):
        raise ValueError("fold group norms must be unique object records")
    all_array_keys = [group["scale_key"] for group in value["fold_groups"]]
    all_array_keys += [key for record in tensors for key in (record["packed_key"], record["scale_key"])]
    if any(not isinstance(key, str) or not key for key in all_array_keys) or len(set(all_array_keys)) != len(all_array_keys):
        raise ValueError("overlay array keys must be non-empty and unique")
    expected_groups = llama_fold_groups(max(layer_ids) + 1)
    if len(groups_by_norm) != len(expected_groups):
        raise ValueError("manifest must record both standard Llama fold groups for every layer")
    for norm_name, consumers in expected_groups:
        if groups_by_norm.get(norm_name) != consumers:
            raise ValueError(f"{norm_name}: shared fold must list every standard Llama consumer")
        if any(tensor_norms.get(name) != norm_name for name in consumers):
            raise ValueError(f"{norm_name}: every shared consumer must have a quantized weight record")
    return value


def validate_overlay_arrays(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    """Check packed/scales/fold payloads without importing Torch or mutating a model."""
    validate_manifest(manifest)
    expected_keys: set[str] = set()
    tensors_by_name = {row["name"][:-len(".weight")]: row for row in manifest["tensors"]}
    for group in manifest["fold_groups"]:
        key = group["scale_key"]
        expected_keys.add(key)
        widths = {tensors_by_name[name]["logical_shape"][1] for name in group["consumers"]}
        if len(widths) != 1:
            raise ValueError(f"{group['norm']}: fold consumers do not share the same input width")
        scale = arrays.get(key)
        width = next(iter(widths))
        if not isinstance(scale, np.ndarray) or scale.dtype.kind != "f" or list(scale.shape) != [width]:
            raise ValueError(f"{group['norm']}: fold scale must be a float vector of width {width}")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"{group['norm']}: fold scale must be positive and finite")

    for record in manifest["tensors"]:
        packed_key, scale_key = record["packed_key"], record["scale_key"]
        expected_keys.update((packed_key, scale_key))
        packed, scale = arrays.get(packed_key), arrays.get(scale_key)
        if not isinstance(packed, np.ndarray) or packed.dtype != np.dtype(np.uint8) or list(packed.shape) != record["packed_shape"]:
            raise ValueError(f"{record['name']}: packed data must be uint8 with the declared shape")
        if not isinstance(scale, np.ndarray) or scale.dtype.kind != "f" or list(scale.shape) != record["scale_shape"]:
            raise ValueError(f"{record['name']}: quantization scales must be float arrays with the declared shape")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"{record['name']}: quantization scales must be positive and finite")
        in_features = record["logical_shape"][1]
        if np.any((packed & 15) == 8):
            raise ValueError(f"{record['name']}: reserved symmetric INT4 code -8 was found")
        if in_features % 2 == 0:
            if np.any((packed >> 4) == 8):
                raise ValueError(f"{record['name']}: reserved symmetric INT4 code -8 was found")
        elif packed.shape[1] > 1 and np.any((packed[:, :-1] >> 4) == 8):
            raise ValueError(f"{record['name']}: reserved symmetric INT4 code -8 was found")
        if in_features % 2 and np.any(packed[:, -1] >> 4):
            raise ValueError(f"{record['name']}: unused high nibble must be zero")

    if set(arrays) != expected_keys:
        raise ValueError("overlay arrays do not exactly match the manifest keys")
