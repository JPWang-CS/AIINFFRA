"""Small local contracts: NumPy only, no Torch, Transformers, weights or network."""

from __future__ import annotations

import unittest

import numpy as np
import quantization_lab as qlab

from hf_quant_contracts import (
    FORMAT_NAME,
    FORMAT_VERSION,
    fold_shared_linear_group,
    llama_fold_groups,
    symmetric_int4_storage_bytes,
    valid_token_rows,
    validate_manifest,
    validate_overlay_arrays,
)


def valid_manifest() -> dict:
    projection_names = [
        (f"model.layers.0.self_attn.{name}.weight", "model.layers.0.input_layernorm")
        for name in ("q_proj", "k_proj", "v_proj")
    ] + [
        (f"model.layers.0.mlp.{name}.weight", "model.layers.0.post_attention_layernorm")
        for name in ("gate_proj", "up_proj")
    ]
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "architecture": "LlamaForCausalLM",
        "base_model_fingerprint": "sha256:base",
        "transformers_version": "5.x-recorded-at-runtime",
        "tokenizer_fingerprint": "sha256:tokenizer",
        "quantization": {
            "bits": 4,
            "group_size": 4,
            "logical_weight_orientation": "O,I",
            "math_weight_orientation": "I,O",
            "packed_orientation": "O,ceil(I/2)",
            "nibble_order": "low-index-first",
        },
        "tensors": [{
            "name": name,
            "logical_shape": [4, 5],
            "packed_shape": [4, 3],
            "scale_shape": [2, 4],
            "source_dtype": "float16",
            "fold_group": norm,
            "packed_key": f"packed_{index}",
            "scale_key": f"scales_{index}",
        } for index, (name, norm) in enumerate(projection_names)],
        "float_exceptions": ["lm_head.weight"],
        "metrics": {},
        "fold_groups": [
            {"norm": "model.layers.0.input_layernorm", "consumers": [
                f"model.layers.0.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj")
            ], "scale_key": "fold_0"},
            {"norm": "model.layers.0.post_attention_layernorm", "consumers": [
                f"model.layers.0.mlp.{name}" for name in ("gate_proj", "up_proj")
            ], "scale_key": "fold_1"},
        ],
        "data_fingerprints": {"calibration": "sha256:cal", "heldout": "sha256:held"},
    }


def valid_arrays(manifest: dict) -> dict[str, np.ndarray]:
    arrays = {}
    for index, record in enumerate(manifest["tensors"]):
        arrays[record["packed_key"]] = np.zeros(record["packed_shape"], dtype=np.uint8)
        arrays[record["scale_key"]] = np.ones(record["scale_shape"], dtype=np.float32)
    for index, group in enumerate(manifest["fold_groups"]):
        arrays[group["scale_key"]] = np.ones(5, dtype=np.float32)
    return arrays


class HFQuantContractTests(unittest.TestCase):
    def test_llama_topology_folds_qkv_and_gate_up_as_shared_groups(self):
        groups = llama_fold_groups(2)
        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0][0], "model.layers.0.input_layernorm")
        self.assertEqual(groups[0][1], tuple(
            f"model.layers.0.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj")
        ))
        self.assertEqual(groups[1][0], "model.layers.0.post_attention_layernorm")
        self.assertEqual(groups[1][1], tuple(
            f"model.layers.0.mlp.{name}" for name in ("gate_proj", "up_proj")
        ))
        with self.assertRaises(ValueError):
            llama_fold_groups(0)
        with self.assertRaises(ValueError):
            llama_fold_groups(True)

    def test_attention_mask_excludes_padding_from_calibration_rows(self):
        hidden = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        mask = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.int64)
        rows = valid_token_rows(hidden, mask)
        np.testing.assert_array_equal(rows, hidden[[0, 0, 1], [0, 1, 1]])
        with self.assertRaises(ValueError):
            valid_token_rows(hidden, np.zeros((2, 3), dtype=np.int64))
        with self.assertRaises(ValueError):
            valid_token_rows(hidden, np.ones((2, 2), dtype=np.int64))
        for invalid in (np.array([[1, 0, -1], [0, 1, 0]]),
                        np.array([[1, 0, 2], [0, 1, 0]]),
                        np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]])):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "attention_mask"):
                    valid_token_rows(hidden, invalid)

    def test_folding_all_consumers_is_function_preserving_and_bias_stays(self):
        rng = np.random.default_rng(31)
        gamma = rng.normal(size=4)
        beta = rng.normal(size=4)
        scale = np.array([0.5, 0.8, 1.4, 2.0])
        x = rng.normal(size=(7, 4))
        weights = {name: rng.normal(size=(4, 3)) for name in ("q", "k", "v", "gate", "up")}
        linear_bias = {name: rng.normal(size=3) for name in weights}
        gamma2, beta2, folded = fold_shared_linear_group(gamma, weights, scale, beta)
        z = x * gamma + beta
        z2 = x * gamma2 + beta2
        for name in weights:
            before = z @ weights[name] + linear_bias[name]
            after = z2 @ folded[name]
            # z2=(x*gamma+beta)/s and every consumer weight is W*s.
            np.testing.assert_allclose(after + linear_bias[name], before, rtol=1e-12, atol=1e-12)

    def test_omitting_a_shared_consumer_breaks_equivalence(self):
        x = np.array([[2.0, 1.0]])
        scale = np.array([2.0, 0.5])
        q = np.array([[1.0], [3.0]])
        k = np.array([[2.0], [-1.0]])
        _, _, folded = fold_shared_linear_group(np.ones(2), {"q": q, "k": k}, scale)
        correct_x = x / scale
        np.testing.assert_allclose(correct_x @ folded["q"], x @ q)
        np.testing.assert_allclose(correct_x @ folded["k"], x @ k)
        wrong_k = correct_x @ k
        self.assertFalse(np.allclose(wrong_k, x @ k))

    def test_weight_layout_storage_counts_packed_payload_and_scales(self):
        # O=3, I=5, G=4: 3*ceil(5/2) bytes + 2*3 float32 scales.
        self.assertEqual(symmetric_int4_storage_bytes(5, 3, 4), 33)
        with self.assertRaises(ValueError):
            symmetric_int4_storage_bytes(5, 3, True)

    def test_existing_int4_helpers_roundtrip_math_I_by_O_to_packed_O_by_bytes(self):
        q_io = np.array([[-7, 7], [0, 1], [2, -3]], dtype=np.int8)
        packed_obytes = qlab.pack_signed_int4(q_io)
        self.assertEqual(packed_obytes.shape, (2, 2))
        self.assertEqual(packed_obytes.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(qlab.unpack_signed_int4(packed_obytes, input_size=3), q_io)

    def test_manifest_accepts_declared_overlay_and_rejects_shape_or_split_mismatch(self):
        original = valid_manifest()
        self.assertIs(validate_manifest(original), original)
        bad_shape = valid_manifest()
        bad_shape["tensors"][0]["packed_shape"] = [4, 2]
        with self.assertRaisesRegex(ValueError, "packed shape"):
            validate_manifest(bad_shape)
        same_split = valid_manifest()
        same_split["data_fingerprints"]["heldout"] = "sha256:cal"
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            validate_manifest(same_split)
        fake_backend = valid_manifest()
        fake_backend["format"] = "vllm-compatible"
        with self.assertRaisesRegex(ValueError, "unsupported teaching overlay"):
            validate_manifest(fake_backend)

    def test_overlay_array_validation_precedes_any_runtime_or_model_dependency(self):
        manifest = valid_manifest()
        arrays = valid_arrays(manifest)
        validate_overlay_arrays(arrays, manifest)

        bad_dtype = valid_arrays(manifest)
        bad_dtype[manifest["tensors"][0]["packed_key"]] = bad_dtype[
            manifest["tensors"][0]["packed_key"]
        ].astype(np.int8)
        with self.assertRaisesRegex(ValueError, "uint8"):
            validate_overlay_arrays(bad_dtype, manifest)

        bad_scale = valid_arrays(manifest)
        bad_scale[manifest["tensors"][0]["scale_key"]][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            validate_overlay_arrays(bad_scale, manifest)

        bad_fold = valid_arrays(manifest)
        bad_fold[manifest["fold_groups"][0]["scale_key"]] = np.ones(4, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "fold scale"):
            validate_overlay_arrays(bad_fold, manifest)

        duplicate_consumer = valid_manifest()
        duplicate_consumer["fold_groups"][0]["consumers"][1] = duplicate_consumer["fold_groups"][0]["consumers"][0]
        with self.assertRaisesRegex(ValueError, "every standard Llama consumer"):
            validate_overlay_arrays(valid_arrays(duplicate_consumer), duplicate_consumer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
