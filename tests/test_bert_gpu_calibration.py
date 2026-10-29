"""CPU-only contract tests for a supervised DistilBERT GPU calibration."""

import importlib
import importlib.util
import inspect
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM


class CalibrationObservationTests(unittest.TestCase):
    def test_asset_staging_rejects_changed_bytes_and_never_overwrites(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "stage_assets", None)),
                        "verified asset staging is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, stage = root / "source", root / "stage"
            source.mkdir()
            stage.mkdir()
            data = b"known pinned bytes"
            (source / "config.json").write_bytes(data)
            assets = {"config.json": (len(data), hashlib.sha256(data).hexdigest())}
            with patch.object(subject, "ASSETS", assets):
                subject.stage_assets(source, stage)
                self.assertEqual((stage / "config.json").read_bytes(), data)
                with self.assertRaises(FileExistsError):
                    subject.stage_assets(source, stage)
                (source / "config.json").write_bytes(b"changed pinned byte")
                other = root / "other"
                other.mkdir()
                with self.assertRaises(ValueError):
                    subject.stage_assets(source, other)

    def test_owner_rejects_low_free_memory_before_loading_the_model(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        gib = 1024 ** 3
        class FakeCuda:
            def is_initialized(self): return False
            def device_count(self): return 1
            def get_allocator_backend(self): return "native"
            def get_device_properties(self, index):
                return SimpleNamespace(uuid=object(), total_memory=8 * gib)
            def set_per_process_memory_fraction(self, fraction, index): pass
            def get_per_process_memory_fraction(self, index): return .25
            def reset_peak_memory_stats(self, index): pass
            def synchronize(self, index): pass
            def mem_get_info(self, index): return 2 * gib, 8 * gib
            def memory_allocated(self, index): return 0
            def memory_reserved(self, index): return 0
            def max_memory_allocated(self, index): return 0
            def max_memory_reserved(self, index): return 0
        session = object.__new__(subject.OwnerSession)
        lease = object.__new__(subject.DeviceLease)
        stages = []
        with patch.object(subject.OwnerSession, "check"), \
             patch.object(subject.OwnerSession, "verify_runtime_binding"), \
             patch.object(subject, "_runtime_uuid", return_value="GPU-fixed", create=True), \
             patch.object(subject, "_load_model", create=True) as load:
            with self.assertRaisesRegex(RuntimeError, "free|reserve|memory"):
                subject.measure_owned(session, lease, SimpleNamespace(cuda=FakeCuda()), "unused",
                                      record_stage=lambda name, data: stages.append(name))
        load.assert_not_called()
        self.assertIn("before_model_load", stages)

    def test_owner_failure_retains_diagnostic_without_clean_attestation(self):
        # An interrupted or OOM owner must never publish a clean candidate.
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "owner_main", None)),
                        "the supervised owner is unavailable")
        session = SimpleNamespace(device_lease_path="unused", identity=object(), run_id="owner-run")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(subject.OwnerSession, "from_environment", return_value=session), \
                 patch.object(subject.DeviceLease, "acquire", return_value=object()), \
                 patch.object(subject.OwnerSession, "check"), \
                 patch.object(subject.OwnerSession, "attest_clean") as attest, \
                 patch.object(subject, "_verify_startup"), \
                 patch.object(subject, "_require_pristine_dependencies"), \
                 patch.object(subject, "_import_torch", return_value=object()), \
                 patch.object(subject, "measure_owned", side_effect=MemoryError("simulated OOM")):
                status = subject.owner_main(directory, directory, "test-run")
            self.assertEqual(status, 1)
            self.assertFalse((Path(directory) / "candidate.json").exists())
            failure = json.loads((Path(directory) / "failure.json").read_bytes())
            self.assertEqual((failure["status"], failure["error_type"]), ("quarantined", "MemoryError"))
            attest.assert_not_called()

    def test_live_memory_guard_preserves_commitment_and_allocator_headroom(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "check_memory", None)),
                        "the live memory guard is unavailable")
        gib, mib = 1024 ** 3, 1024 ** 2
        good = {"device_free_bytes": 6 * gib, "device_total_bytes": 8 * gib,
                "allocated_bytes": 400 * mib, "reserved_bytes": 500 * mib,
                "peak_allocated_bytes": 450 * mib, "peak_reserved_bytes": 550 * mib}
        subject.check_memory(good, expected_total=8 * gib, cap=2 * gib,
                             commitment=1536 * mib)
        for changed in (
            {"device_free_bytes": 2 * gib},
            {"reserved_bytes": 2 * gib - 255 * mib, "peak_reserved_bytes": 2 * gib - 255 * mib},
            {"device_total_bytes": 7 * gib},
            {"peak_allocated_bytes": 501 * mib, "peak_reserved_bytes": 500 * mib},
        ):
            with self.subTest(changed=changed), self.assertRaises((ValueError, RuntimeError)):
                subject.check_memory({**good, **changed}, expected_total=8 * gib,
                                     cap=2 * gib, commitment=1536 * mib)

    def test_output_stays_alive_until_synchronization_then_is_disposed(self):
        # A GPU output must not disappear before its completion boundary.
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertIn("synchronize", inspect.signature(subject.observe).parameters)
        torch.set_num_threads(1)
        model = DistilBertForMaskedLM(DistilBertConfig(
            vocab_size=64, dim=16, hidden_dim=32, n_layers=6, n_heads=4,
            dropout=0.0, attention_dropout=0.0, max_position_embeddings=32,
        )).eval()
        original, refs, stages = model.forward, [], []
        def forward(*args, **kwargs):
            output = original(*args, **kwargs)
            refs.append(weakref.ref(output.logits))
            return output
        model.forward = forward
        def sync(stage):
            stages.append((stage, refs[-1]() is not None))
        subject.observe(model, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
                        is_id=20, are_id=21, synchronize=sync)
        self.assertEqual(stages, [("output_live", True), ("output_disposed", False)])

    def test_failed_live_output_sync_stops_before_disposal_sync_or_next_forward(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertIn("synchronize", inspect.signature(subject.run_sequence).parameters)
        torch.set_num_threads(1)
        model = DistilBertForMaskedLM(DistilBertConfig(
            vocab_size=64, dim=16, hidden_dim=32, n_layers=6, n_heads=4,
            dropout=0.0, attention_dropout=0.0, max_position_embeddings=32,
        )).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        checkpoints, syncs = [], []
        def sync(stage):
            syncs.append(stage)
            raise RuntimeError("simulated synchronization failure")
        with self.assertRaisesRegex(RuntimeError, "simulated synchronization failure"):
            subject.run_sequence(model, ids, ids, is_id=20, are_id=21,
                                 checkpoint=lambda stage, count: checkpoints.append((stage, count)),
                                 synchronize=sync)
        self.assertEqual(syncs, ["output_live"])
        self.assertEqual(checkpoints, [("before_target", 0)])
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks
                             for module in model.modules()))

    def test_revocation_after_forward_prevents_output_validation_or_copy(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertIn("check_owner", inspect.signature(subject.observe).parameters)
        torch.set_num_threads(1)
        model = DistilBertForMaskedLM(DistilBertConfig(
            vocab_size=64, dim=16, hidden_dim=32, n_layers=6, n_heads=4,
            dropout=0.0, attention_dropout=0.0, max_position_embeddings=32,
        )).eval()
        original_forward, original_finite = model.forward, torch.isfinite
        returned, late_finiteness = [], []
        def forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            returned.append(True)
            return output
        def check_owner():
            if returned:
                raise RuntimeError("owner revoked")
        def finite(value):
            if returned:
                late_finiteness.append(True)
            return original_finite(value)
        model.forward = forward
        with patch.object(torch, "isfinite", side_effect=finite):
            with self.assertRaisesRegex(RuntimeError, "owner revoked"):
                subject.observe(model, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
                                is_id=20, are_id=21, check_owner=check_owner)
        self.assertEqual(late_finiteness, [])
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks
                             for module in model.modules()))

    def test_same_forward_self_and_cross_patch_preserve_selected_bits(self):
        # A wrong source, wrong site, or failed hook cleanup must change this
        # observable result. The tiny model uses only fabricated CPU inputs.
        spec = importlib.util.find_spec("looking_glass.bert_gpu_calibration")
        self.assertIsNotNone(spec, "the calibration measurement is unavailable")
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        torch.set_num_threads(1)
        torch.manual_seed(731)
        model = DistilBertForMaskedLM(DistilBertConfig(
            vocab_size=64, dim=16, hidden_dim=32, n_layers=6, n_heads=4,
            dropout=0.0, attention_dropout=0.0, max_position_embeddings=32,
        )).eval()
        target = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        donor = torch.tensor([[1, 2, 11, 4, 5, 6, 7, 8, 9, 10]])

        baseline = subject.observe(model, target, is_id=20, are_id=21)
        source = subject.observe(model, donor, is_id=20, are_id=21)
        same = subject.observe(model, target, is_id=20, are_id=21, self_patch=True)
        cross = subject.observe(model, target, is_id=20, are_id=21, source=source.before)

        self.assertEqual(same.mask_logits_bytes, baseline.mask_logits_bytes)
        self.assertEqual(same.before_bits, same.after_bits)
        self.assertEqual(cross.after_bits, source.before_bits)
        self.assertNotEqual(cross.before_bits, cross.after_bits)
        self.assertEqual([row["operation"] for row in cross.events],
                         ["capture", "patch", "capture"])
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks
                             for module in model.modules()))

    def test_sequence_retains_four_observations_and_checks_landing(self):
        # Omitting a stage or accepting a mismatched self/cross source must
        # fail. All inputs and the model are synthetic; no CUDA is used.
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "run_sequence", None)),
                        "the four-stage calibration sequence is unavailable")
        torch.set_num_threads(1)
        torch.manual_seed(731)
        model = DistilBertForMaskedLM(DistilBertConfig(
            vocab_size=64, dim=16, hidden_dim=32, n_layers=6, n_heads=4,
            dropout=0.0, attention_dropout=0.0, max_position_embeddings=32,
        )).eval()
        target = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        donor = torch.tensor([[1, 2, 11, 4, 5, 6, 7, 8, 9, 10]])
        stages = []

        result = subject.run_sequence(model, target, donor, is_id=20, are_id=21,
                                      checkpoint=lambda stage, count: stages.append((stage, count)))

        self.assertEqual(stages, [("before_target", 0), ("after_target", 1),
                                  ("before_donor", 1), ("after_donor", 2),
                                  ("before_self", 2), ("after_self", 3),
                                  ("before_cross", 3), ("after_cross", 4)])
        self.assertEqual(result["forward_count"], 4)
        self.assertTrue(result["self_mask_logits_bitwise_equal"])
        self.assertNotIn("self_logits_equal", result)
        self.assertTrue(result["cross_source_landed"])
        self.assertTrue(result["source_differs_from_target"])
        self.assertEqual([row["stage"] for row in result["observations"]],
                         ["target", "donor", "self", "cross"])


if __name__ == "__main__":
    unittest.main()
