"""Tests for the CPU full-model quantization teaching path."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_quantization_lab as lab


class ModelQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = lab.ModelConfig(
            vocab_size=13, hidden_size=8, intermediate_size=12,
            num_heads=2, num_layers=1, max_seq_len=6,
        )
        cls.reference = lab.MiniDecoder.random(cls.config, seed=17)
        cls.calibration = lab.make_tokens(cls.config, seed=31, count=6, sequence_length=5)
        cls.heldout = lab.make_tokens(cls.config, seed=89, count=5, sequence_length=5, shifted=True)
        cls.bundle = lab.quantize_full_model(
            cls.reference, cls.calibration, group_size=4, clip_ratios=(1.0, 0.9)
        )

    def _write_mutated(self, source: Path, destination: Path, mutate) -> None:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(arrays["metadata"].item()))
        mutate(metadata)
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(destination, **arrays)

    def test_calibration_and_heldout_are_separate_and_masked(self) -> None:
        self.assertEqual(self.calibration.token_ids.shape[0], 6)
        self.assertEqual(self.heldout.token_ids.shape[0], 5)
        self.assertFalse(np.array_equal(self.calibration.token_ids[:5], self.heldout.token_ids))
        captured = lab.collect_calibration_inputs(self.reference, self.calibration, batch_size=2)
        x, token_mask, padding_mask = captured["layers.0.shared_in"]
        self.assertEqual(x.shape[0], token_mask.size)
        stats = lab.qlab.calibration_statistics(x, token_mask, padding_mask)
        expected = int(np.count_nonzero(self.calibration.token_mask & ~self.calibration.padding_mask))
        self.assertEqual(stats.valid_tokens, expected)
        self.assertLess(expected, x.shape[0], "the fixture must contain masked padding rows")

    def test_all_parallel_shared_consumers_are_folded_together(self) -> None:
        name = "layers.0.shared_in"
        scale = self.bundle.fold_scales[name]
        np.testing.assert_allclose(
            self.bundle.folded_model.params["layers.0.norm_gamma"],
            self.reference.params["layers.0.norm_gamma"] / scale,
        )
        for suffix in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            weight_name = "layers.0." + suffix
            np.testing.assert_allclose(
                self.bundle.folded_model.params[weight_name],
                self.reference.params[weight_name] * scale[:, None],
            )
        float_fold_error = lab.evaluate_against(
            self.reference, self.bundle.folded_model, self.heldout
        )["logit_max_abs"]
        self.assertLess(float_fold_error, 1e-10)

    def test_single_branch_fold_is_not_equivalent(self) -> None:
        correct_error, wrong_error = lab.shared_branch_counterexample()
        self.assertLess(correct_error, 1e-12)
        self.assertGreater(wrong_error, 1e-3)

    def test_clipping_search_includes_unclipped_baseline(self) -> None:
        for result in self.bundle.layer_calibration_mse.values():
            self.assertLessEqual(result["best_mse"], result["baseline_mse"] + 1e-15)

    def test_next_token_nll_shifts_targets_and_excludes_padding(self) -> None:
        class FixedLogitModel:
            def forward(self, token_ids, token_mask, padding_mask):
                logits = np.array([[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, np.log(2.0)],
                    [100.0, 0.0, 0.0],  # would dominate loss if padded target were included
                    [0.0, 0.0, 0.0],
                ]])
                return logits

        data = lab.TokenBatch(
            token_ids=np.array([[0, 1, 2, 0]]),
            token_mask=np.array([[True, True, True, False]]),
            padding_mask=np.array([[False, False, False, True]]),
        )
        measured = lab.next_token_metrics(FixedLogitModel(), data)
        expected_nll = (np.log(3.0) + np.log(2.0)) / 2.0
        self.assertEqual(measured["next_token_count"], 2.0)
        self.assertAlmostEqual(measured["next_token_nll"], expected_nll, places=14)
        self.assertAlmostEqual(measured["perplexity"], np.sqrt(6.0), places=14)

    def test_packed_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aiinffra-model-quant-test-") as directory:
            path = Path(directory) / "checkpoint.npz"
            lab.save_quantized_checkpoint(self.bundle, path)
            restored = lab.load_quantized_checkpoint(path, expected_config=self.config)
            error = lab.evaluate_against(
                lab.materialize_quantized_model(self.bundle), restored, self.heldout
            )["logit_max_abs"]
            self.assertEqual(error, 0.0)

    def test_existing_destination_is_never_modified_or_deleted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aiinffra-existing-file-test-") as directory:
            path = Path(directory) / "user-owned.npz"
            sentinel = b"user-owned bytes; do not replace or delete\x00"
            path.write_bytes(sentinel)
            with self.assertRaises(FileExistsError):
                lab.save_quantized_checkpoint(self.bundle, path)
            self.assertEqual(path.read_bytes(), sentinel)

    def test_loader_rejects_unsupported_quantization_format_labels(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aiinffra-schema-test-") as directory:
            root = Path(directory)
            source = root / "valid.npz"
            lab.save_quantized_checkpoint(self.bundle, source)
            fields = (
                "scheme", "weight_orientation", "packed_orientation", "nibble_order", "scale_shape",
            )
            for field in fields:
                with self.subTest(field=field):
                    bad = root / f"bad-{field}.npz"
                    def mutate(metadata, field=field):
                        metadata["quantization"][field] = "unsupported"
                    self._write_mutated(source, bad, mutate)
                    with self.assertRaisesRegex(ValueError, "quantization metadata"):
                        lab.load_quantized_checkpoint(bad)

    def test_loader_rejects_version_and_logical_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aiinffra-version-shape-test-") as directory:
            root = Path(directory)
            source = root / "valid.npz"
            lab.save_quantized_checkpoint(self.bundle, source)
            wrong_version = root / "wrong-version.npz"
            self._write_mutated(source, wrong_version, lambda m: m.update(format_version=999))
            with self.assertRaisesRegex(ValueError, "format_version"):
                lab.load_quantized_checkpoint(wrong_version)
            wrong_shape = root / "wrong-shape.npz"
            def mutate_shape(metadata):
                metadata["tensors"]["layers.0.q_proj"]["logical_shape"] = [1, 1]
            self._write_mutated(source, wrong_shape, mutate_shape)
            with self.assertRaisesRegex(ValueError, "logical tensor shape"):
                lab.load_quantized_checkpoint(wrong_shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
