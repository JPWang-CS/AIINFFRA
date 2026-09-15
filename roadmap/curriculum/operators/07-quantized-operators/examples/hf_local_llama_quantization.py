"""Local-only Llama W4 teaching overlay: calibration, heldout evaluation, save/load.

Requires a local safetensors LlamaForCausalLM checkpoint, its local tokenizer,
PyTorch, Transformers and NumPy. It never downloads model files or remote code.
The packed overlay is not a Transformers or serving-engine checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quantization_lab as qlab  # existing teaching implementation; not AWQ/GPTQ author code
from hf_quant_contracts import (
    FORMAT_NAME, FORMAT_VERSION, llama_fold_groups, validate_manifest, validate_overlay_arrays,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted({p.resolve() for p in paths})
    if not files:
        raise ValueError("cannot fingerprint an empty local file set")
    for path in files:
        digest.update(path.relative_to(root.resolve()).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def local_model_fingerprint(model_dir: Path) -> str:
    config = model_dir / "config.json"
    weights = sorted(model_dir.rglob("*.safetensors"))
    if not config.is_file() or not weights:
        raise FileNotFoundError("local config.json and at least one safetensors weight file are required")
    return fingerprint([config, *weights], model_dir)


def local_tokenizer_fingerprint(model_dir: Path) -> str:
    names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "tokenizer.model", "spiece.model",
    }
    paths = [p for p in model_dir.rglob("*") if p.is_file() and p.name in names]
    if not paths:
        raise FileNotFoundError("no local tokenizer files were found beside the model")
    return fingerprint(paths, model_dir)


def file_fingerprint(path: Path) -> str:
    return sha256_file(path.resolve())


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"local text file not found: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"text file has no non-empty lines: {path}")
    return rows


def check_disjoint(calibration: list[str], heldout: list[str]) -> None:
    overlap = set(calibration) & set(heldout)
    if overlap:
        raise ValueError(f"calibration and heldout contain {len(overlap)} identical non-empty lines")


def load_runtime(model_dir: Path, device_arg: str, dtype_arg: str):
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install PyTorch, Transformers, NumPy and safetensors in the user's model environment.") from exc

    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"--model-dir must be an existing local directory: {model_dir}")
    config = AutoConfig.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    if getattr(config, "model_type", None) != "llama":
        raise ValueError(f"expected config.model_type='llama', got {getattr(config, 'model_type', None)!r}")
    if getattr(config, "quantization_config", None):
        raise ValueError("start from an unquantized local Llama checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a local pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    device = device_arg
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports no CUDA device")
    if dtype_arg == "auto":
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_arg]
    major = int(transformers.__version__.split(".", 1)[0])
    dtype_key = "dtype" if major >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        **{dtype_key: dtype},
    )
    if model.__class__.__name__ != "LlamaForCausalLM":
        raise ValueError(f"expected HF LlamaForCausalLM, got {model.__class__.__name__}")
    model.to(device)
    model.eval()
    return torch, transformers, tokenizer, model, device, dtype


def validate_topology(model, torch) -> list[tuple[str, tuple[str, ...]]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("expected the standard HF LlamaForCausalLM model.layers topology")
    layers = model.model.layers
    groups = llama_fold_groups(len(layers))
    modules = dict(model.named_modules())
    for norm_name, consumers in groups:
        norm = modules.get(norm_name)
        if norm is None or not isinstance(getattr(norm, "weight", None), torch.Tensor):
            raise ValueError(f"missing expected normalization module {norm_name}")
        if norm.weight.ndim != 1:
            raise ValueError(f"{norm_name}.weight must be one-dimensional")
        if getattr(norm, "bias", None) is not None and norm.bias.shape != norm.weight.shape:
            raise ValueError(f"{norm_name}.bias shape does not match its weight")
        for name in consumers:
            linear = modules.get(name)
            if not isinstance(linear, torch.nn.Linear):
                raise ValueError(f"{name} must be a separate torch.nn.Linear projection")
            if linear.in_features != norm.weight.numel():
                raise ValueError(f"{name}.in_features does not match {norm_name}")
    for layer in layers:
        attn = layer.self_attn
        if any(getattr(attn, name, None) is not None for name in ("q_norm", "k_norm", "q_layernorm", "k_layernorm")):
            raise ValueError("Q/K-normalized Llama variants need an explicit topology adapter")
    return groups


def encoded_batches(tokenizer, texts: list[str], batch_size: int, max_seq_len: int, device):
    import torch

    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        yield {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }


class NormCapture:
    def __init__(self, model, group_names: Iterable[str], limit: int):
        self.parts: dict[str, list[np.ndarray]] = {name: [] for name in group_names}
        self.counts = {name: 0 for name in group_names}
        self.limit = limit
        self.mask = None
        modules = dict(model.named_modules())
        self.handles = []
        for name in self.parts:
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def collect(_module, _inputs, output):
            import torch

            if self.mask is None or not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError(f"{name}: expected [batch,sequence,hidden] output and active attention_mask")
            if tuple(output.shape[:2]) != tuple(self.mask.shape):
                raise ValueError(f"{name}: normalization output and attention_mask shapes differ")
            remain = self.limit - self.counts[name]
            if remain <= 0:
                return
            rows = output.detach().float()[self.mask.bool()]
            if rows.numel() == 0:
                return
            rows = rows[:remain].cpu().numpy()
            if not np.all(np.isfinite(rows)):
                raise ValueError(f"{name}: valid calibration/evaluation activations are non-finite")
            self.parts[name].append(rows)
            self.counts[name] += rows.shape[0]
        return collect

    def set_mask(self, mask):
        self.mask = mask

    def arrays(self) -> dict[str, np.ndarray]:
        result = {}
        for name, pieces in self.parts.items():
            if not pieces:
                raise ValueError(f"no valid tokens were captured at {name}")
            result[name] = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def collect_norms(torch, model, tokenizer, texts, groups, batch_size, max_seq_len, device, limit):
    capture = NormCapture(model, (norm for norm, _ in groups), limit)
    try:
        with torch.inference_mode():
            for batch in encoded_batches(tokenizer, texts, batch_size, max_seq_len, device):
                capture.set_mask(batch["attention_mask"])
                model(**batch, use_cache=False)
        return capture.arrays()
    finally:
        capture.close()


def read_task_pairs(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    pairs = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in ("prompt", "target")):
            raise ValueError(f"{path}:{line_no}: expected JSON object with string prompt and target")
        pairs.append((item["prompt"], item["target"]))
    if path is not None and not pairs:
        raise ValueError("task file contains no prompt/target pairs")
    return pairs


def evaluate(torch, model, tokenizer, texts, batch_size, max_seq_len, device, groups,
             metric_tokens: int, logit_samples: int, capture_limit: int | None = None):
    import torch.nn.functional as F

    capture = NormCapture(model, (norm for norm, _ in groups), capture_limit) if capture_limit else None
    total_nll = 0.0
    total_targets = 0
    correct = 0
    sample_logits: list[np.ndarray] = []
    sample_targets: list[np.ndarray] = []
    seen_metric_tokens = 0
    try:
        with torch.inference_mode():
            for batch in encoded_batches(tokenizer, texts, batch_size, max_seq_len, device):
                if seen_metric_tokens >= metric_tokens and (
                    capture is None or all(n >= capture.limit for n in capture.counts.values())
                ):
                    break
                mask = batch["attention_mask"].bool()
                if capture is not None:
                    capture.set_mask(batch["attention_mask"])
                logits = model(**batch, use_cache=False).logits.float()
                target_mask = mask[:, :-1] & mask[:, 1:]
                remaining = metric_tokens - seen_metric_tokens
                if remaining <= 0:
                    continue
                positions = target_mask.nonzero(as_tuple=False)
                if positions.numel() == 0:
                    continue
                positions = positions[:remaining]
                rows, cols = positions[:, 0], positions[:, 1]
                predicted = logits[rows, cols]
                labels = batch["input_ids"][rows, cols + 1]
                total_nll += float(F.cross_entropy(predicted, labels, reduction="sum").item())
                correct += int((predicted.argmax(dim=-1) == labels).sum().item())
                total_targets += int(labels.numel())
                seen_metric_tokens += int(labels.numel())
                if sum(x.shape[0] for x in sample_logits) < logit_samples:
                    take = min(logit_samples - sum(x.shape[0] for x in sample_logits), predicted.shape[0])
                    sample_logits.append(predicted[:take].detach().cpu().numpy())
                    sample_targets.append(labels[:take].detach().cpu().numpy())
        if total_targets == 0:
            raise ValueError("heldout data needs at least one adjacent valid next-token pair")
        nll = total_nll / total_targets
        result = {
            "next_token_count": total_targets,
            "next_token_nll": nll,
            "perplexity": math.exp(min(nll, 700.0)),
            "next_token_top1_accuracy": correct / total_targets,
            "metric_token_cap": metric_tokens,
        }
        if capture is not None:
            result["norm_activations"] = capture.arrays()
        if sample_logits:
            result["sample_logits"] = np.concatenate(sample_logits, axis=0)
            result["sample_targets"] = np.concatenate(sample_targets, axis=0)
        return result
    finally:
        if capture is not None:
            capture.close()


def evaluate_exact_match(torch, model, tokenizer, pairs, device, prompt_token_limit, max_new_tokens):
    if not pairs:
        return None
    hits = 0
    for prompt, target in pairs:
        tokens = tokenizer(prompt, add_special_tokens=True, truncation=True,
                           max_length=prompt_token_limit, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items() if key in ("input_ids", "attention_mask")}
        prompt_len = tokens["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(**tokens, do_sample=False, max_new_tokens=max_new_tokens)
        answer = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        normalize = lambda s: " ".join(s.casefold().split())
        hits += int(normalize(answer) == normalize(target))
    return {"exact_match": hits / len(pairs), "examples": len(pairs), "max_new_tokens": max_new_tokens}


def make_overlay(torch, transformers, model, groups, calibration_acts,
                 group_size, base_fp, calibration_path, heldout_path, tokenizer_fp):
    modules = dict(model.named_modules())
    packed_arrays: dict[str, np.ndarray] = {}
    fold_records = []
    tensor_records = []
    key_index = 0
    for norm_name, consumers in groups:
        x_cal = calibration_acts[norm_name].astype(np.float64, copy=False)
        original = {
            name: modules[name].weight.detach().float().cpu().numpy().T.astype(np.float64)
            for name in consumers
        }
        joined = np.concatenate([original[name] for name in consumers], axis=1)
        selection = qlab.awq_style_search(x_cal, joined, group_size)
        # Use the exact float32 scale that will be stored and later loaded.
        fold = selection.scale.astype(np.float32).astype(np.float64)
        joined_scaled = joined * fold[:, None]
        _, joined_scales = qlab.quantize_int4_group(joined_scaled, group_size)
        joined_scales = joined_scales.astype(np.float32)
        q_joined = np.empty(joined_scaled.shape, dtype=np.int8)
        for group_index in range(joined_scales.shape[0]):
            start = group_index * group_size
            end = min(start + group_size, joined_scaled.shape[0])
            q_joined[start:end] = np.clip(
                np.rint(joined_scaled[start:end] / joined_scales[group_index][None, :]), -7, 7
            ).astype(np.int8)
        joined_restored = qlab.dequantize_int4_group(q_joined, joined_scales, group_size) / fold[:, None]
        stored_scale_mse = float(np.mean(np.square(x_cal @ joined_restored - x_cal @ joined)))
        fold_key = f"fold_{key_index:04d}"
        packed_arrays[fold_key] = fold.astype(np.float32)
        fold_records.append({"norm": norm_name, "consumers": list(consumers), "scale_key": fold_key,
                             "calibration_mse_before": selection.baseline_mse,
                             "calibration_mse_after": stored_scale_mse,
                             "candidate_ratio": selection.ratio})
        for name in consumers:
            scaled_weight = original[name] * fold[:, None]
            _, scales = qlab.quantize_int4_group(scaled_weight, group_size)
            scales = scales.astype(np.float32)
            q = np.empty(scaled_weight.shape, dtype=np.int8)
            for group_index in range(scales.shape[0]):
                start = group_index * group_size
                end = min(start + group_size, scaled_weight.shape[0])
                q[start:end] = np.clip(
                    np.rint(scaled_weight[start:end] / scales[group_index][None, :]), -7, 7
                ).astype(np.int8)
            packed = qlab.pack_signed_int4(q)
            restored = qlab.dequantize_int4_group(q, scales, group_size)
            module = modules[name]
            weight_key, scale_key = f"packed_{key_index:04d}", f"scales_{key_index:04d}"
            packed_arrays[weight_key] = packed
            packed_arrays[scale_key] = scales.astype(np.float32)
            tensor_records.append({
                "name": name + ".weight",
                "logical_shape": [int(original[name].shape[1]), int(original[name].shape[0])],
                "packed_shape": [int(packed.shape[0]), int(packed.shape[1])],
                "scale_shape": [int(scales.shape[0]), int(scales.shape[1])],
                "source_dtype": str(module.weight.dtype).replace("torch.", ""),
                "fold_group": norm_name,
                "packed_key": weight_key,
                "scale_key": scale_key,
            })
            key_index += 1

    quantized_names = {record["name"] for record in tensor_records}
    folded_names = {record["norm"] + ".weight" for record in fold_records}
    folded_names.update(record["norm"] + ".bias" for record in fold_records
                        if getattr(modules[record["norm"]], "bias", None) is not None)
    float_exceptions = []
    for name, param in model.named_parameters():
        if name not in quantized_names and name not in folded_names:
            float_exceptions.append({"name": name, "shape": list(param.shape),
                                     "dtype": str(param.dtype).replace("torch.", ""),
                                     "storage": "unchanged in the local base checkpoint"})
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    tied = bool(input_embedding is not None and output_embedding is not None and
                input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr())
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "artifact_kind": "non-deployable, base-checkpoint-bound teaching overlay",
        "architecture": "LlamaForCausalLM",
        "model_type": model.config.model_type,
        "model_config": {name: getattr(model.config, name) for name in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "rope_theta", "tie_word_embeddings",
            "attention_bias", "mlp_bias",
        ) if getattr(model.config, name, None) is not None},
        "base_model_fingerprint": base_fp,
        "tokenizer_fingerprint": tokenizer_fp,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "quantization": {
            "bits": 4,
            "scheme": "symmetric_groupwise_round_to_nearest_even_after_shared_activation-aware fold",
            "group_size": group_size,
            "logical_weight_orientation": "O,I",
            "math_weight_orientation": "I,O",
            "packed_orientation": "O,ceil(I/2)",
            "nibble_order": "low-index-first",
            "scale_orientation": "ceil(I/group_size),O",
            "runtime_execution": "dequantized floating torch.nn.Linear reference",
        },
        "tensors": tensor_records,
        "float_exceptions": float_exceptions,
        "weight_sharing": {"input_embedding_and_lm_head_tied": tied,
                            "lm_head_quantized": False,
                            "reason": "preserve embedding/output tying and keep the output projection as an explicit FP exception"},
        "fold_groups": fold_records,
        "data_fingerprints": {"calibration": file_fingerprint(calibration_path),
                              "heldout": file_fingerprint(heldout_path)},
        "preprocessing": {"format": "one UTF-8 text example per non-empty line",
                          "tokenizer": "local AutoTokenizer; add_special_tokens=True; right padding",
                          "activation_mask": "attention_mask == 1 only",
                          "heldout_layer_mse": "valid normalized rows captured from the original FP model"},
        "metrics": {"calibration_shared_group_search": [
            {"norm": row["norm"], "mse_before": row["calibration_mse_before"],
             "mse_after": row["calibration_mse_after"], "candidate_ratio": row["candidate_ratio"]}
            for row in fold_records
        ]},
        "layout_transform": "W[O,I] -> W_math[I,O] -> packed[O,ceil(I/2)]; dequantized back to W[O,I]",
    }
    return packed_arrays, validate_manifest(manifest)


def load_overlay(path: Path, base_fp: str, tokenizer_fp: str):
    if not path.is_file():
        raise FileNotFoundError(f"overlay not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "manifest" not in archive.files:
            raise ValueError("NPZ has no manifest")
        manifest = validate_manifest(json.loads(str(archive["manifest"].item())))
        if manifest["base_model_fingerprint"] != base_fp:
            raise ValueError("overlay was created from a different local model checkpoint")
        if manifest["tokenizer_fingerprint"] != tokenizer_fp:
            raise ValueError("overlay was created with a different local tokenizer")
        arrays = {name: archive[name].copy() for name in archive.files if name != "manifest"}
    validate_overlay_arrays(arrays, manifest)
    return arrays, manifest


def validate_overlay_for_model(torch, model, arrays, manifest):
    """Validate every array and target before apply_overlay mutates any parameter."""
    validate_overlay_arrays(arrays, manifest)
    modules = dict(model.named_modules())
    seen_norms: set[str] = set()
    for group in manifest["fold_groups"]:
        norm_name = group["norm"]
        if norm_name in seen_norms:
            raise ValueError("duplicate norm or fold-scale key in overlay")
        seen_norms.add(norm_name)
        norm = modules.get(norm_name)
        if norm is None or not isinstance(getattr(norm, "weight", None), torch.Tensor):
            raise ValueError(f"overlay fold target is missing: {norm_name}")
        fold = arrays[group["scale_key"]]
        if tuple(fold.shape) != tuple(norm.weight.shape):
            raise ValueError(f"{norm_name}: fold scale width does not match norm.weight")

    seen_targets: set[str] = set()
    for record in manifest["tensors"]:
        name = record["name"][:-len(".weight")]
        if name in seen_targets:
            raise ValueError(f"duplicate quantized target: {name}")
        seen_targets.add(name)
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"overlay target is not a Linear module: {name}")
        out_features, in_features = record["logical_shape"]
        if list(module.weight.shape) != [out_features, in_features]:
            raise ValueError(f"base model shape mismatch for {name}.weight")


def measure_heldout_layer_mse(model, heldout_acts, arrays, manifest):
    """Compare each packed/dequantized Linear to FP on original heldout norm rows."""
    modules = dict(model.named_modules())
    fold_by_norm = {row["norm"]: arrays[row["scale_key"]].astype(np.float64) for row in manifest["fold_groups"]}
    result = {}
    group_size = manifest["quantization"]["group_size"]
    for record in manifest["tensors"]:
        module_name = record["name"][:-len(".weight")]
        norm_name = record["fold_group"]
        x = heldout_acts[norm_name].astype(np.float64, copy=False)
        original = modules[module_name].weight.detach().float().cpu().numpy().T.astype(np.float64)
        q = qlab.unpack_signed_int4(arrays[record["packed_key"]], original.shape[0])
        restored = qlab.dequantize_int4_group(q, arrays[record["scale_key"]], group_size)
        reference_y = x @ original
        candidate_y = (x / fold_by_norm[norm_name][None, :]) @ restored
        bias = modules[module_name].bias
        if bias is not None:
            b = bias.detach().float().cpu().numpy().astype(np.float64)
            reference_y += b[None, :]
            candidate_y += b[None, :]
        result[module_name] = float(np.mean(np.square(candidate_y - reference_y)))
    return result


def apply_overlay(torch, model, arrays, manifest):
    validate_overlay_for_model(torch, model, arrays, manifest)
    modules = dict(model.named_modules())
    group_size = manifest["quantization"]["group_size"]
    with torch.no_grad():
        for group in manifest["fold_groups"]:
            norm = modules[group["norm"]]
            fold = torch.as_tensor(arrays[group["scale_key"]], device=norm.weight.device, dtype=norm.weight.dtype)
            norm.weight.div_(fold)
            if getattr(norm, "bias", None) is not None:
                norm.bias.div_(fold)
        for record in manifest["tensors"]:
            name = record["name"][:-len(".weight")]
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"overlay target is not a Linear module: {name}")
            out_features, in_features = record["logical_shape"]
            if list(module.weight.shape) != [out_features, in_features]:
                raise ValueError(f"base model shape mismatch for {name}.weight")
            q = qlab.unpack_signed_int4(arrays[record["packed_key"]], in_features)
            restored_io = qlab.dequantize_int4_group(q, arrays[record["scale_key"]], group_size)
            value = torch.as_tensor(restored_io.T.copy(), device=module.weight.device, dtype=module.weight.dtype)
            with torch.no_grad():
                module.weight.copy_(value)


def update_manifest_metrics(manifest, reference, candidate, task_reference, task_candidate):
    ref_logits = reference.get("sample_logits")
    cand_logits = candidate.get("sample_logits")
    compare = {}
    if ref_logits is not None and cand_logits is not None:
        n = min(ref_logits.shape[0], cand_logits.shape[0])
        delta = cand_logits[:n].astype(np.float64) - ref_logits[:n].astype(np.float64)
        denom = float(np.linalg.norm(ref_logits[:n].astype(np.float64)))
        compare = {
            "sampled_heldout_logits_tokens": n,
            "sampled_heldout_logits_mse": float(np.mean(delta * delta)),
            "sampled_heldout_logits_max_abs": float(np.max(np.abs(delta))),
            "sampled_heldout_logits_relative_l2": float(np.linalg.norm(delta) / max(denom, 1e-30)),
            "sampled_heldout_top1_agreement": float(np.mean(
                ref_logits[:n].argmax(axis=-1) == cand_logits[:n].argmax(axis=-1)
            )),
        }
    manifest["metrics"].update({
        "heldout_reference": {k: v for k, v in reference.items() if not k.startswith("sample_") and k != "norm_activations"},
        "heldout_quantized": {k: v for k, v in candidate.items() if not k.startswith("sample_") and k != "norm_activations"},
        "heldout_logit_comparison": compare,
        "optional_task_reference": task_reference,
        "optional_task_quantized": task_candidate,
    })


def save_overlay(path: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any]):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**arrays, "manifest": np.asarray(json.dumps(manifest, sort_keys=True))}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()


def run_task(torch, model, tokenizer, pairs, device, max_seq_len, max_new_tokens):
    model.eval()
    return evaluate_exact_match(torch, model, tokenizer, pairs, device, max_seq_len, max_new_tokens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="existing local HF Llama checkpoint directory")
    parser.add_argument("--heldout-file", type=Path, required=True, help="UTF-8 file: one independent example per non-empty line")
    parser.add_argument("--calibration-file", type=Path, help="required when creating a new overlay; never used for evaluation")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--calibration-tokens", type=int, default=256, help="valid activation rows captured per norm site")
    parser.add_argument("--layer-metric-tokens", type=int, default=32, help="heldout valid rows used for per-layer MSE")
    parser.add_argument("--metric-tokens", type=int, default=4096, help="maximum heldout next-token targets")
    parser.add_argument("--logit-samples", type=int, default=8, help="bounded heldout vocab-logit rows for MSE comparison")
    parser.add_argument("--task-file", type=Path, help="optional local JSONL, each row has prompt and target for greedy exact match")
    parser.add_argument("--task-max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit torch device")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--save", type=Path, help="write a new NPZ overlay; existing paths are not overwritten")
    parser.add_argument("--load", type=Path, help="load an overlay against this exact local base model/tokenizer")
    args = parser.parse_args(argv)
    if args.group_size <= 0 or args.batch_size <= 0 or args.max_seq_len < 2:
        parser.error("group-size/batch-size must be positive and max-seq-len at least 2")
    if min(args.calibration_tokens, args.layer_metric_tokens, args.metric_tokens,
           args.logit_samples, args.task_max_new_tokens) <= 0:
        parser.error("token limits and task generation length must be positive")
    if args.save and args.load:
        parser.error("choose --save for a newly calibrated overlay or --load for an existing overlay")
    if not args.load and not args.calibration_file:
        parser.error("--calibration-file is required unless --load is used")

    torch, transformers, tokenizer, model, device, dtype = load_runtime(args.model_dir, args.device, args.dtype)
    groups = validate_topology(model, torch)
    model_dir = args.model_dir.resolve()
    base_fp = local_model_fingerprint(model_dir)
    tokenizer_fp = local_tokenizer_fingerprint(model_dir)
    heldout = read_lines(args.heldout_file)
    heldout_fp = file_fingerprint(args.heldout_file)
    calibration = read_lines(args.calibration_file) if args.calibration_file else None
    calibration_fp = file_fingerprint(args.calibration_file) if args.calibration_file else None
    if calibration is not None:
        check_disjoint(calibration, heldout)
        if calibration_fp == heldout_fp:
            raise ValueError("calibration and heldout files must have different fingerprints")
    task_pairs = read_task_pairs(args.task_file)
    calibration_set = set(calibration or ())
    heldout_set = set(heldout)
    for prompt, target in task_pairs:
        if prompt in heldout_set or prompt in calibration_set or target in calibration_set:
            raise ValueError("task examples must not duplicate calibration/heldout lines")

    max_positions = getattr(model.config, "max_position_embeddings", args.max_seq_len)
    if args.max_seq_len > max_positions:
        raise ValueError(f"max-seq-len {args.max_seq_len} exceeds model limit {max_positions}")
    task_prompt_limit = min(args.max_seq_len, max_positions - args.task_max_new_tokens)
    if task_pairs and task_prompt_limit < 1:
        raise ValueError("task-max-new-tokens leaves no room for a prompt within the model context limit")
    print(json.dumps({"architecture": model.__class__.__name__, "transformers": transformers.__version__,
                      "torch": torch.__version__, "device": str(device), "dtype": str(dtype),
                      "local_files_only": True, "trust_remote_code": False,
                      "base_model_fingerprint": base_fp}, sort_keys=True))

    if args.load:
        arrays, manifest = load_overlay(args.load, base_fp, tokenizer_fp)
        if heldout_fp == manifest["data_fingerprints"]["calibration"]:
            raise ValueError("heldout evaluation file fingerprint matches the overlay's calibration data")
        if args.task_file and file_fingerprint(args.task_file) == manifest["data_fingerprints"]["calibration"]:
            raise ValueError("task evaluation file fingerprint matches the overlay's calibration data")
        old_mse = manifest["metrics"].pop("heldout_layer_mse_current", None)
        old_mse_fp = manifest["metrics"].pop("heldout_layer_mse_data_fingerprint", None)
        if old_mse is None:
            old_mse = manifest["metrics"].pop("heldout_layer_mse", None)
        if old_mse is not None:
            manifest["metrics"]["original_artifact_heldout_layer_mse"] = {
                "data_fingerprint": old_mse_fp or manifest["data_fingerprints"]["heldout"],
                "by_tensor": old_mse,
            }
        validate_overlay_for_model(torch, model, arrays, manifest)

    reference = evaluate(torch, model, tokenizer, heldout, args.batch_size, args.max_seq_len,
                         device, groups, args.metric_tokens, args.logit_samples,
                         capture_limit=args.layer_metric_tokens)
    task_reference = run_task(torch, model, tokenizer, task_pairs, device, task_prompt_limit,
                              args.task_max_new_tokens) if task_pairs else None

    if not args.load:
        calibration_acts = collect_norms(torch, model, tokenizer, calibration, groups,
                                         args.batch_size, args.max_seq_len, device, args.calibration_tokens)
        arrays, manifest = make_overlay(
            torch, transformers, model, groups, calibration_acts,
            args.group_size, base_fp,
            args.calibration_file, args.heldout_file, tokenizer_fp,
        )
        manifest["data_fingerprints"]["task"] = file_fingerprint(args.task_file) if args.task_file else None
        # Preserve base-file fingerprints before mutating the in-memory teaching model.
        manifest["source"] = {"model_dir_label": model_dir.name,
                              "calibration_file_fingerprint": calibration_fp,
                              "heldout_file_fingerprint": heldout_fp}
    validate_overlay_for_model(torch, model, arrays, manifest)
    layer_mse = measure_heldout_layer_mse(
        model, reference["norm_activations"], arrays, manifest
    )

    apply_overlay(torch, model, arrays, manifest)
    candidate = evaluate(torch, model, tokenizer, heldout, args.batch_size, args.max_seq_len,
                         device, groups, args.metric_tokens, args.logit_samples)
    task_candidate = run_task(torch, model, tokenizer, task_pairs, device, task_prompt_limit,
                              args.task_max_new_tokens) if task_pairs else None
    update_manifest_metrics(manifest, reference, candidate, task_reference, task_candidate)
    manifest["metrics"]["heldout_layer_mse_current"] = layer_mse
    manifest["metrics"]["heldout_layer_mse_data_fingerprint"] = heldout_fp
    manifest["metrics"]["evaluation_heldout_file_fingerprint"] = heldout_fp
    manifest["metrics"]["evaluation_task_file_fingerprint"] = file_fingerprint(args.task_file) if args.task_file else None
    validate_manifest(manifest)

    print(json.dumps({"heldout_reference": {k: v for k, v in reference.items() if k != "norm_activations" and not k.startswith("sample_")},
                      "heldout_quantized": {k: v for k, v in candidate.items() if not k.startswith("sample_")},
                      "heldout_layer_mse_current_data": heldout_fp,
                      "heldout_layer_mse_current": layer_mse,
                      "heldout_logit_comparison": manifest["metrics"]["heldout_logit_comparison"],
                      "task_reference_exact_match": task_reference,
                      "task_quantized_exact_match": task_candidate}, sort_keys=True))
    if args.save:
        save_overlay(args.save, arrays, manifest)
        print(f"saved teaching overlay: {args.save.resolve()}")
    if args.load:
        print(f"loaded teaching overlay: {args.load.resolve()}")
    print("Execution used dequantized floating Linear weights; no packed INT4 kernel/performance claim is made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
