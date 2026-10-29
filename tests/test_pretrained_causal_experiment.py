"""Development-only tests: no pretrained weights or held-out model forwards."""
import copy
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import json
import multiprocessing
from pathlib import Path
import base64
import struct
import tempfile
import threading
import unittest
from unittest import mock

import torch

from looking_glass import pretrained_causal_experiment as study
from looking_glass import pretrained_protocol as protocol


def fake_public():
    # These token IDs/nonces are synthetic records, not a private study split.
    public, _ = protocol.seal_manifest(
        [[i + 1] for i in range(8)], [[i + 101] for i in range(8)],
        site=2, site_nonce=b"a" * 32, split_nonce=b"b" * 32,
        source_commit="1" * 40, protocol_sha256="2" * 64,
        runtime={key: ("cpu" if key == "device" else "development-only") for key in
                 ("python", "torch", "transformers", "device", "driver", "deterministic_settings")},
        created_utc="2026-10-28T00:00:00Z")
    return public


def _evaluation_process(path, public_digest, prediction_digest, entered, release, output):
    """Subprocess target using synthetic observations, never a model forward."""
    def callback(index):
        if index == 0:
            entered.set()
            if not release.wait(20):
                raise TimeoutError("test coordination timed out")
        return 0, PublicStudyTests().development_case(index)
    try:
        result = study.run_development_evaluation(path, public_digest, prediction_digest, callback)
        output.put(("success", result["completed_cases"]))
    except Exception as exc:
        output.put(("error", type(exc).__name__))


def _evaluation_writer_to_kill(path, public_digest, prediction_digest, published):
    original = study.registry.ExperimentRun.write_artifact
    wait_forever = threading.Event()
    def pause_after_data(run, name, data):
        original(run, name, data)
        if name == "evaluation.json":
            published.set()
            wait_forever.wait(60)
    with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=pause_after_data):
        study.run_development_evaluation(path, public_digest, prediction_digest,
                                         lambda index: (0, PublicStudyTests().development_case(index)))


class PublicStudyTests(unittest.TestCase):
    def test_manifest_accepts_only_exact_public_schema_and_fixed_protocol_fields(self):
        public = fake_public()
        self.assertEqual(study.validate_public_manifest(public), protocol.canonical_sha256(public))
        variants = []
        for key, value in [("seed", 1), ("heldout_count", True), ("random_baseline_layer", 4),
                           ("site_commitment", "X" * 64), ("protocol", "other")]:
            record = copy.deepcopy(public)
            record[key] = value
            variants.append(record)
        for key, value in [("metric", {"logit_token_ids": [257, 262]}),
                           ("model", {"id": "other"}), ("injection", {})]:
            record = copy.deepcopy(public)
            record[key] = value
            variants.append(record)
        record = copy.deepcopy(public)
        record["discovery"][1]["token_ids"] = record["discovery"][0]["token_ids"][:]
        record["discovery_data_sha256"] = protocol.canonical_sha256(record["discovery"])
        variants.append(record)
        record = copy.deepcopy(public)
        record["private_site"] = 2
        variants.append(record)
        for record in variants:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    study.validate_public_manifest(record)

    def test_direction_has_exact_hash_bits_and_float32_scale(self):
        delta = study.injection_delta()
        self.assertEqual(delta.shape, (768,))
        self.assertEqual(delta.dtype, torch.float32)
        bits = hashlib.sha256(b"MLG-v1-direction\0" + protocol.SEED.to_bytes(8, "big") + bytes(4)).digest()
        expected_signs = [1 if bits[j // 8] & (1 << (7 - j % 8)) else -1 for j in range(256)]
        self.assertEqual(torch.sign(delta[:256]).int().tolist(), expected_signs)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(delta)), 0.5, places=6)

    def test_injection_integrity_requires_exact_float32_add_and_realized_change(self):
        pre = torch.linspace(-1.0, 1.0, 768, dtype=torch.float32)
        post = pre + study.injection_delta()
        integrity = study.verify_injection_integrity(pre, post)
        self.assertTrue(torch.equal(integrity["realized_delta"], post - pre))
        self.assertAlmostEqual(integrity["realized_norm"], 0.5, places=5)
        bad = post.clone()
        bad[17] += 0.1
        with self.assertRaises(ValueError):
            study.verify_injection_integrity(pre, bad)
        cancellation_pre = -study.injection_delta()
        cancellation_post = cancellation_pre + study.injection_delta()
        signed_zero = cancellation_post.clone()
        signed_zero[0] = -0.0
        with self.assertRaises(ValueError):
            study.verify_injection_integrity(cancellation_pre, signed_zero)
        huge = torch.full((768,), 1e10, dtype=torch.float32)
        with self.assertRaises(ValueError):
            study.verify_injection_integrity(huge, huge + study.injection_delta())

    def test_first_difference_scoring_abstains_and_breaks_ties_by_layer(self):
        scores = study.rank_sites([None, 3, 2, 3, 2, None, None, None],
                                  [[0, 0, 0, 0, 0, 0] for _ in range(8)])
        self.assertEqual(scores["ranking"], [2, 3, 1, 4])
        self.assertEqual(scores["selected"], 2)
        self.assertEqual(scores["magnitude"], 1)
        self.assertEqual(scores["counts"], {"1": 0, "2": 2, "3": 2, "4": 0})
        scores = study.rank_sites([None] * 8, [[0] * 6 for _ in range(8)])
        self.assertIsNone(scores["selected"])
        self.assertEqual(scores["ranking"], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            study.rank_sites([0] + [None] * 7, [[0] * 6 for _ in range(8)])

    def test_effect_summary_uses_prespecified_zero_and_sign_threshold(self):
        actual = [0.0, 0.00001, -0.000011, 0.1, -0.2, 0.3, 0.4, -0.5]
        result = study.summarize_effects(actual, prediction=0.0)
        self.assertEqual(result["sign_excluded"], 2)
        self.assertEqual(result["sign_eligible"], 6)
        self.assertEqual(result["sign_accuracy"], 0.0)
        self.assertEqual(result["mae"], result["zero_predictor_mae"])
        self.assertAlmostEqual(result["rmse"], math.sqrt(sum(x*x for x in actual)/8))
        self.assertIsNone(study.summarize_effects([0.0] * 8, prediction=1.0)["sign_accuracy"])
        with self.assertRaises(ValueError):
            study.summarize_effects([float("nan")] * 8, prediction=0.0)

    @staticmethod
    def observation(token=0, *, injected=False, changed_layer=None, logits=None):
        vectors = []
        for layer in range(6):
            value = torch.zeros(768, dtype=torch.float32)
            if changed_layer is not None and layer >= changed_layer:
                value[0] = 0.25
            vectors.append(study.TraceEvent(layer=layer, module_path=f"transformer.h.{layer}.mlp",
                                            invocation=0, batch_index=0, token_index=token,
                                            output_selector=(), source_shape=(1, token + 1, 768),
                                            source_dtype="torch.float32", source_device="cpu",
                                            forward_index=0, sequence=layer, value=value))
        vectors.append(study.TraceEvent(layer=5, module_path="transformer.h.5", invocation=0,
                                        batch_index=0, token_index=token, output_selector=(0,),
                                        source_shape=(1, token + 1, 768), source_dtype="torch.float32",
                                        source_device="cpu", forward_index=0, sequence=6,
                                        value=torch.zeros(768, dtype=torch.float32)))
        if logits is None:
            logits = torch.zeros(50257, dtype=torch.float32)
        return study.Observation(events=tuple(vectors), logits=logits,
                                 injection_integrity_verified=injected,
                                 private_tensor_bytes=3 * 768 * 4 if injected else 0)

    @staticmethod
    def charge_conditions(conditions):
        charged = {}
        for name, observation in conditions.items():
            injected = name not in ("clean", "clean_repeat") and not name.startswith("reverse_")
            patch = (name in ("self_patch", "upstream", "late") or
                     name.startswith("patch_") or name.startswith("reverse_"))
            post_action = name not in ("clean", "clean_repeat", "injected")
            charged[name] = replace(observation, extra_tensor_bytes=(int(injected) + int(patch) + int(post_action)) * 768 * 4)
        return charged

    def test_discovery_case_validates_controls_and_records_raw_conditions(self):
        clean = self.observation()
        corrupted_logits = torch.zeros(50257, dtype=torch.float32)
        corrupted_logits[262] = 1.0
        injected = self.observation(injected=True, changed_layer=2, logits=corrupted_logits)
        patch_logits = corrupted_logits.clone()
        patch_logits[262] = 0.5
        patched = self.observation(injected=True, changed_layer=2, logits=patch_logits)
        late = self.observation(injected=True, changed_layer=2)
        conditions = {"clean": clean, "clean_repeat": clean, "injected": injected,
                      "null": injected, "self_patch": injected, "upstream": injected,
                      "late": late}
        conditions.update({f"patch_{site}": patched for site in range(1, 5)})
        conditions.update({f"reverse_{site}": clean for site in range(1, 5)})
        conditions = self.charge_conditions(conditions)
        cleanup = {name: clean for name in conditions if name not in ("clean", "clean_repeat")}
        case = study.DiscoveryCase(index=0, conditions=conditions, post_clean=cleanup)
        analysis = study.analyze_discovery_case(case, token_index=0)
        self.assertEqual(analysis["first_difference"], 2)
        self.assertEqual(analysis["effects"], {str(site): -0.5 for site in range(1, 5)})
        self.assertEqual(analysis["reverse_effects"], {str(site): 0.0 for site in range(1, 5)})
        bad = dict(conditions)
        bad["null"] = replace(patched, extra_tensor_bytes=2 * 768 * 4)
        with self.assertRaisesRegex(ValueError, "null"):
            study.analyze_discovery_case(study.DiscoveryCase(0, bad, cleanup), token_index=0)
        with self.assertRaisesRegex(ValueError, "post-clean"):
            study.analyze_discovery_case(study.DiscoveryCase(0, conditions, {}), token_index=0)
        damaged = list(injected.events)
        damaged[1], damaged[2] = damaged[2], damaged[1]
        bad["injected"] = study.Observation(tuple(damaged), injected.logits, True, 3 * 768 * 4)
        with self.assertRaises(ValueError):
            study.validate_observation(bad["injected"], token_index=0, injected=True)
        event = replace(injected.events[0], source_shape=(1, 2, 768))
        invalid_events = (event,) + injected.events[1:]
        with self.assertRaises(ValueError):
            study.validate_observation(replace(injected, events=invalid_events), token_index=0,
                                       injected=True, source_device="cpu")
        with self.assertRaises(ValueError):
            study.validate_observation(injected, token_index=0, injected=True, source_device="cuda:0")
        for broken_event in (replace(injected.events[0], source_shape=(True, 1, 768)),
                             replace(injected.events[6], output_selector=(False,))):
            events = list(injected.events)
            events[0 if broken_event.module_path.endswith("mlp") else 6] = broken_event
            with self.assertRaises(ValueError):
                study.validate_observation(replace(injected, events=tuple(events)), token_index=0, injected=True)
        bad = dict(conditions)
        bad["null"] = replace(bad["null"], extra_tensor_bytes=0)
        with self.assertRaisesRegex(ValueError, "site-independent"):
            study.analyze_discovery_case(study.DiscoveryCase(0, bad, cleanup), token_index=0)

    def test_restoration_is_separate_and_near_zero_denominator_is_undefined(self):
        self.assertEqual(study.summarize_restoration([0.0] * 8, [0.0] * 8)["restoration"], None)
        result = study.summarize_restoration([1.0] * 8, [0.25] * 8)
        self.assertEqual(result["denominator"], 8.0)
        self.assertEqual(result["numerator"], 2.0)
        self.assertEqual(result["restoration"], 0.75)
        self.assertEqual(result["numerator_per_prompt"], [0.25] * 8)
        self.assertEqual(study.summarize_restoration([1.0] * 8, [2.0] * 8)["restoration"], -1.0)
        self.assertIsNone(study.summarize_restoration([1.25e-9] * 8, [1.0] * 8)["restoration"])

    def test_layer_five_only_difference_is_recorded_without_an_eligible_vote(self):
        case = self.development_case(0)
        clean = case.conditions["clean"]
        logits = torch.zeros(50257, dtype=torch.float32)
        changed = self.observation(injected=True, changed_layer=5, logits=logits)
        conditions = dict(case.conditions)
        for name in ("injected", "null", "self_patch", "upstream", "late"):
            conditions[name] = changed
        for site in range(1, 5):
            conditions[f"patch_{site}"] = changed
        conditions = self.charge_conditions(conditions)
        analyzed = study.analyze_discovery_case(study.DiscoveryCase(0, conditions, case.post_clean), token_index=0)
        self.assertEqual(analyzed["first_observed_layer"], 5)
        self.assertIsNone(analyzed["first_difference"])

    def test_revealed_rank_never_reports_zero_score_tie_as_success_when_abstaining(self):
        selectors = study.rank_sites([None] * 8, [[0] * 6 for _ in range(8)])
        self.assertEqual(study.score_revealed_site(selectors, 1),
                         {"site": 1, "rank": None, "top_1": False, "reciprocal_rank": 0.0,
                          "baseline_top_1": {"magnitude": True, "random": selectors["random"] == 1,
                                             "earliest": True}})
        selectors = study.rank_sites([2] * 8, [[0] * 6 for _ in range(8)])
        self.assertEqual(study.score_revealed_site(selectors, 2)["rank"], 1)
        self.assertTrue(study.score_revealed_site(selectors, 2)["top_1"])
        self.assertEqual(study.score_revealed_site(selectors, 4)["reciprocal_rank"], 0.25)
        selectors = study.rank_sites([1] * 8, [[0] * 6 for _ in range(8)])
        selectors["selected"] = True
        with self.assertRaises(ValueError):
            study.score_revealed_site(selectors, 1)

    def development_case(self, index):
        clean = self.observation()
        corrupt_logits = torch.zeros(50257, dtype=torch.float32)
        corrupt_logits[262] = 1.0
        corrupt = self.observation(injected=True, changed_layer=2, logits=corrupt_logits)
        patch_logits = corrupt_logits.clone()
        patch_logits[262] = 0.5
        patched = self.observation(injected=True, changed_layer=2, logits=patch_logits)
        late = self.observation(injected=True, changed_layer=2)
        conditions = {"clean": clean, "clean_repeat": clean, "injected": corrupt,
                      "null": corrupt, "self_patch": corrupt, "upstream": corrupt, "late": late}
        conditions.update({f"patch_{site}": patched for site in range(1, 5)})
        conditions.update({f"reverse_{site}": clean for site in range(1, 5)})
        conditions = self.charge_conditions(conditions)
        return study.DiscoveryCase(index, conditions,
                                   {key: clean for key in conditions if key not in ("clean", "clean_repeat")})

    def test_manifest_is_published_before_any_development_callback_and_predictions_freeze(self):
        public = fake_public()
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "new-run"
            def callback(index, token_ids):
                manifest = path / "artifacts" / "public_manifest.json"
                self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                 protocol.canonical_sha256(public))
                self.assertFalse((path / "artifacts" / "predictions.json").exists())
                self.assertEqual(token_ids, tuple(public["discovery"][index]["token_ids"]))
                calls.append(index)
                return self.development_case(index)
            result = study.run_development_discovery(path, public, callback)
            self.assertEqual(calls, list(range(8)))
            frozen_path = path / "artifacts" / "predictions.json"
            raw = frozen_path.read_bytes()
            frozen = json.loads(raw)
            self.assertEqual(result["predictions_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(frozen["mode"], "development-only")
            self.assertEqual(frozen["development_accounting_schema"], study.DEVELOPMENT_ACCOUNTING_SCHEMA)
            self.assertEqual(frozen["development_sources_sha256"], study._read_development_sources())
            self.assertEqual(frozen["public_manifest_sha256"], protocol.canonical_sha256(public))
            self.assertEqual(frozen["selectors"]["selected"], 2)
            self.assertEqual(frozen["predictions"], {str(i): -0.5 for i in range(1, 5)})
            self.assertEqual(len(frozen["cases"]), 8)
            self.assertTrue(frozen["tensors"])
            self.assertIn("logits", frozen["cases"][0]["conditions"]["null"])
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            self.assertFalse((path / "artifacts" / "evaluation.json").exists())
            with self.assertRaises(FileExistsError):
                study.registry.open_run(path).write_artifact("predictions.json", b"overwrite")

    def test_later_phases_reject_other_development_schema_or_source(self):
        public, reveal = protocol.seal_manifest(
            [[i + 1] for i in range(8)], [[i + 101] for i in range(8)],
            site=2, site_nonce=b"a" * 32, split_nonce=b"b" * 32,
            source_commit="1" * 40, protocol_sha256="2" * 64,
            runtime={k: ("cpu" if k == "device" else "development-only") for k in
                     ("python", "torch", "transformers", "device", "driver", "deterministic_settings")},
            created_utc="2026-10-28T00:00:00Z")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, public,
                                                    lambda index, ids: self.development_case(index))
            calls = []
            def callback(index):
                calls.append(index)
                return 0, self.development_case(index)
            changed = dict(study._LOADED_DEVELOPMENT_SOURCES)
            changed["analysis"] = "0" * 64
            with mock.patch.object(study, "DEVELOPMENT_ACCOUNTING_SCHEMA", "other-schema"):
                with self.assertRaises(ValueError):
                    study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"], callback)
            # Another process can load different source consistently, but it
            # must still match the source frozen in the discovery artifact.
            with mock.patch.object(study, "_read_development_sources", return_value=changed), \
                    mock.patch.object(study, "_LOADED_DEVELOPMENT_SOURCES", changed):
                with self.assertRaises(ValueError):
                    study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"], callback)
            # A source edit after import is rejected as well.
            with mock.patch.object(study, "_read_development_sources", return_value=changed):
                with self.assertRaises(ValueError):
                    study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"], callback)
            self.assertEqual(calls, [])
            self.assertFalse((path / "artifacts" / "evaluation_claim.json").exists())
            evaluated = study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                         frozen["predictions_sha256"], callback)
            self.assertEqual(calls, list(range(8)))
            with mock.patch.object(study, "_read_development_sources", return_value=changed), \
                    mock.patch.object(study, "_LOADED_DEVELOPMENT_SOURCES", changed):
                with self.assertRaises(ValueError):
                    study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                        frozen["predictions_sha256"], evaluated["evaluation_sha256"], reveal)
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                frozen["predictions_sha256"], evaluated["evaluation_sha256"], reveal)

    def test_failed_callback_leaves_a_non_success_record_without_private_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "failed-run"
            def callback(index, token_ids):
                if index == 1:
                    raise RuntimeError("PRIVATE-SITE-DO-NOT-PUBLISH")
                return self.development_case(index)
            with self.assertRaises(study.StudyExecutionError) as caught:
                study.run_development_discovery(path, fake_public(), callback)
            self.assertNotIn("PRIVATE-SITE", str(caught.exception))
            state = study.registry.verify_run(path)
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["outcome"]["details"]["completed_cases"], 1)
            self.assertNotIn("PRIVATE-SITE", json.dumps(state))
            self.assertTrue((path / "artifacts" / "partial_discovery.json").exists())
            self.assertFalse((path / "artifacts" / "predictions.json").exists())

    def test_interrupt_before_commit_is_recorded_and_surfaced_as_cancelled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cancelled-run"
            def callback(index, token_ids):
                raise KeyboardInterrupt("PRIVATE BACKEND DETAILS")
            with self.assertRaises(study.StudyCancelled) as caught:
                study.run_development_discovery(path, fake_public(), callback)
            self.assertEqual(caught.exception.stage, "discovery")
            self.assertEqual(caught.exception.terminal_state, "cancelled")
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertEqual(study.registry.verify_run(path)["state"], "cancelled")
            self.assertFalse((path / "artifacts" / "discovery_complete.json").exists())

    def test_cancellation_during_failed_finalization_reports_existing_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "failed-run"
            original = study.registry.ExperimentRun.finalize
            def faulty_finalize(run, state, details):
                original(run, state, details)
                raise KeyboardInterrupt()
            def callback(index, ids):
                raise RuntimeError("PRIVATE BACKEND DETAILS")
            with mock.patch.object(study.registry.ExperimentRun, "finalize", new=faulty_finalize):
                with self.assertRaises(study.StudyCancelled) as caught:
                    study.run_development_discovery(path, fake_public(), callback)
            self.assertEqual(caught.exception.terminal_state, "failed")
            self.assertIn("failed", str(caught.exception))
            self.assertEqual(study.registry.verify_run(path)["state"], "failed")

    def test_interrupt_with_unverifiable_commit_preserves_cancellation_and_incomplete_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "uncertain-run"
            original = study.registry.ExperimentRun.write_artifact
            def interrupt_after_marker(run, name, data):
                original(run, name, data)
                if name == "discovery_complete.json":
                    raise KeyboardInterrupt()
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=interrupt_after_marker):
                with mock.patch.object(study, "_completion_matches", side_effect=OSError("unavailable")):
                    with self.assertRaises(study.StudyCancellationUncertain) as caught:
                        study.run_development_discovery(path, fake_public(),
                                                       lambda index, ids: self.development_case(index))
            self.assertEqual(caught.exception.stage, "discovery")
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            self.assertTrue((path / "artifacts" / "discovery_complete.json").exists())

    def test_interrupt_during_reconciliation_is_not_hidden_by_original_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "uncertain-run"
            original = study.registry.ExperimentRun.write_artifact
            def marker_error(run, name, data):
                original(run, name, data)
                if name == "discovery_complete.json":
                    raise OSError("post-publication failure")
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=marker_error):
                with mock.patch.object(study, "_completion_matches", side_effect=KeyboardInterrupt()):
                    with self.assertRaises(study.StudyCancellationUncertain):
                        study.run_development_discovery(path, fake_public(),
                                                       lambda index, ids: self.development_case(index))
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")

    def test_interrupt_writing_reconciliation_warning_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            original = study.registry.ExperimentRun.write_artifact
            def write(run, name, data):
                if name == "discovery_warning.json":
                    raise KeyboardInterrupt()
                original(run, name, data)
                if name == "discovery_complete.json":
                    raise OSError("post-publication failure")
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=write):
                with self.assertRaises(study.StudyCommittedCancellation):
                    study.run_development_discovery(path, fake_public(),
                                                    lambda index, ids: self.development_case(index))
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")

    def test_interrupt_after_phase_helper_returns_reconciles_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            path, public = Path(temp) / "run", fake_public()
            original = study._publish_phase
            def interrupt_after_return(*args, **kwargs):
                original(*args, **kwargs)
                raise KeyboardInterrupt()
            with mock.patch.object(study, "_publish_phase", side_effect=interrupt_after_return):
                with self.assertRaises(study.StudyCommittedCancellation) as caught:
                    study.run_development_discovery(path, public,
                                                    lambda index, ids: self.development_case(index))
            self.assertEqual(caught.exception.stage, "discovery")
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            predictions_digest = caught.exception.artifact_sha256
            with mock.patch.object(study, "_publish_phase", side_effect=interrupt_after_return):
                with self.assertRaises(study.StudyCommittedCancellation) as caught:
                    study.run_development_evaluation(path, protocol.canonical_sha256(public), predictions_digest,
                                                     lambda index: (0, self.development_case(index)))
            self.assertEqual(caught.exception.stage, "evaluation")
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            self.assertTrue((path / "artifacts" / "evaluation_complete.json").exists())

    def test_callback_mutation_after_analysis_cannot_change_serialized_observation(self):
        public = fake_public()
        submitted = []
        mutate = threading.Event()
        done = threading.Event()
        def callback(index, ids):
            case = self.development_case(index)
            submitted.append(case)
            return case
        def retained_backend_thread():
            if mutate.wait(10):
                submitted[0].conditions["injected"].logits[262] = 42.0
                done.set()
        original = study.analyze_discovery_case
        def analyze(case, **kwargs):
            result = original(case, **kwargs)
            if case.index == 0:
                mutate.set()
                if not done.wait(10):
                    raise TimeoutError("mutation thread did not run")
            return result
        thread = threading.Thread(target=retained_backend_thread)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "run"
                with mock.patch.object(study, "analyze_discovery_case", side_effect=analyze):
                    study.run_development_discovery(path, public, callback)
                data = json.loads((path / "artifacts" / "predictions.json").read_bytes())
                case = data["cases"][0]
                ref = case["conditions"]["injected"]["logits"]["tensor_sha256"]
                raw = base64.b64decode(data["tensors"][ref]["base64"], validate=True)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), ref)
                self.assertEqual(struct.unpack_from("<f", raw, 262 * 4)[0], 1.0)
                self.assertEqual(case["analysis"]["injected_metric"], 1.0)
        finally:
            mutate.set()
            thread.join(timeout=10)

    def test_development_evaluation_checks_external_frozen_digests_before_callback(self):
        public, reveal = protocol.seal_manifest(
            [[i + 1] for i in range(8)], [[i + 101] for i in range(8)],
            site=2, site_nonce=b"a" * 32, split_nonce=b"b" * 32,
            source_commit="1" * 40, protocol_sha256="2" * 64,
            runtime={k: ("cpu" if k == "device" else "development-only") for k in
                     ("python", "torch", "transformers", "device", "driver", "deterministic_settings")},
            created_utc="2026-10-28T00:00:00Z")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, public,
                                                    lambda index, ids: self.development_case(index))
            seen = []
            def eval_case(index):
                seen.append(index)
                return 0, self.development_case(index)
            with self.assertRaises(ValueError):
                study.run_development_evaluation(path, frozen["public_manifest_sha256"], "0" * 64, eval_case)
            self.assertEqual(seen, [])
            with self.assertRaises(ValueError):
                study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                                                 frozen["predictions_sha256"], "0" * 64, reveal)
            self.assertEqual(seen, [])
            result = study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"], eval_case)
            self.assertEqual(seen, list(range(8)))
            evaluation = json.loads((path / "artifacts" / "evaluation.json").read_bytes())
            self.assertEqual(evaluation["metrics"]["2"]["mae"], 0)
            self.assertEqual(evaluation["restoration"]["2"]["restoration"], 0.75)
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            final = study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"],
                                                     result["evaluation_sha256"], reveal)
            self.assertEqual(final["revealed_site_score"]["rank"], 1)
            self.assertEqual(study.registry.verify_run(path)["state"], "succeeded")

    def test_simultaneous_evaluators_claim_once_without_loser_finalizing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, fake_public(),
                                                    lambda index, ids: self.development_case(index))
            barrier = threading.Barrier(2)
            original = study._checked_development_stage
            calls = []
            lock = threading.Lock()
            def checked(*args, **kwargs):
                result = original(*args, **kwargs)
                barrier.wait(timeout=10)
                return result
            def callback(index):
                with lock:
                    calls.append(index)
                return 0, self.development_case(index)
            def evaluate():
                try:
                    return study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                            frozen["predictions_sha256"], callback)
                except (FileExistsError, ValueError):
                    return None
            with mock.patch.object(study, "_checked_development_stage", side_effect=checked):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(lambda _: evaluate(), range(2)))
            self.assertEqual(sum(result is not None for result in results), 1)
            self.assertEqual(calls, list(range(8)))
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")

    def test_process_claim_blocks_second_evaluator_while_first_is_running(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, fake_public(),
                                                    lambda index, ids: self.development_case(index))
            ctx = multiprocessing.get_context("spawn")
            entered, release = ctx.Event(), ctx.Event()
            first_output, second_output = ctx.Queue(), ctx.Queue()
            args = (str(path), frozen["public_manifest_sha256"], frozen["predictions_sha256"], entered, release)
            first = ctx.Process(target=_evaluation_process, args=(*args, first_output))
            second = ctx.Process(target=_evaluation_process, args=(*args, second_output))
            try:
                first.start()
                self.assertTrue(entered.wait(20), "first evaluator did not start")
                second.start()
                self.assertEqual(second_output.get(timeout=20)[0], "error")
                self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
                release.set()
                self.assertEqual(first_output.get(timeout=20), ("success", 8))
            finally:
                release.set()
                first.join(timeout=20)
                if second.pid is not None:
                    second.join(timeout=20)
                if first.is_alive():
                    first.terminate()
                if second.pid is not None and second.is_alive():
                    second.terminate()
            self.assertEqual(first.exitcode, 0)
            self.assertEqual(second.exitcode, 0)

    def test_killed_process_after_data_publication_does_not_commit_or_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, fake_public(),
                                                    lambda index, ids: self.development_case(index))
            ctx = multiprocessing.get_context("spawn")
            published = ctx.Event()
            writer = ctx.Process(target=_evaluation_writer_to_kill,
                                 args=(str(path), frozen["public_manifest_sha256"],
                                       frozen["predictions_sha256"], published))
            try:
                writer.start()
                self.assertTrue(published.wait(30), "evaluation data were not published")
                writer.terminate()
                writer.join(timeout=20)
                self.assertFalse(writer.is_alive())
                self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
                self.assertTrue((path / "artifacts" / "evaluation.json").exists())
                self.assertFalse((path / "artifacts" / "evaluation_complete.json").exists())
                callbacks = []
                with self.assertRaises(ValueError):
                    study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"],
                                                     lambda index: callbacks.append(index))
                self.assertEqual(callbacks, [])
                evaluation_digest = study.registry.sha256_file(path / "artifacts" / "evaluation.json")
                with self.assertRaisesRegex(ValueError, "evaluation completion marker is missing"):
                    study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"], evaluation_digest, {})
            finally:
                if writer.pid is not None and writer.is_alive():
                    writer.terminate()
                    writer.join(timeout=20)

    def test_reveal_cannot_race_a_failed_evaluation_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, fake_public(),
                                                    lambda index, ids: self.development_case(index))
            published, release = threading.Event(), threading.Event()
            original = study.registry.ExperimentRun.write_artifact
            def faulty_write(run, name, data):
                original(run, name, data)
                if name == "evaluation.json":
                    published.set()
                    if not release.wait(10):
                        raise TimeoutError("test coordination timed out")
                    raise OSError("simulated post-publication cleanup failure")
            def evaluate():
                return study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                        frozen["predictions_sha256"],
                                                        lambda index: (0, self.development_case(index)))
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=faulty_write):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(evaluate)
                    try:
                        self.assertTrue(published.wait(10))
                        raw = (path / "artifacts" / "evaluation.json").read_bytes()
                        digest = hashlib.sha256(raw).hexdigest()
                        with self.assertRaises(ValueError):
                            study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                                                             frozen["predictions_sha256"], digest, {})
                    finally:
                        release.set()
                    with self.assertRaises(study.StudyExecutionError):
                        future.result(timeout=10)
            self.assertEqual(study.registry.verify_run(path)["state"], "failed")
            self.assertFalse((path / "artifacts" / "evaluation_complete.json").exists())

    def test_evaluation_cannot_race_a_failed_prediction_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            path, public = Path(temp) / "run", fake_public()
            published, release = threading.Event(), threading.Event()
            original = study.registry.ExperimentRun.write_artifact
            calls = []
            def faulty_write(run, name, data):
                original(run, name, data)
                if name == "predictions.json":
                    published.set()
                    if not release.wait(10):
                        raise TimeoutError("test coordination timed out")
                    raise OSError("simulated post-publication cleanup failure")
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=faulty_write):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(study.run_development_discovery, path, public,
                                             lambda index, ids: self.development_case(index))
                    try:
                        self.assertTrue(published.wait(10))
                        raw = (path / "artifacts" / "predictions.json").read_bytes()
                        digest = hashlib.sha256(raw).hexdigest()
                        with self.assertRaises(ValueError):
                            study.run_development_evaluation(path, protocol.canonical_sha256(public), digest,
                                                             lambda index: calls.append(index))
                    finally:
                        release.set()
                    with self.assertRaises(study.StudyExecutionError):
                        future.result(timeout=10)
            self.assertEqual(calls, [])
            self.assertEqual(study.registry.verify_run(path)["state"], "failed")
            self.assertFalse((path / "artifacts" / "discovery_complete.json").exists())

    def test_postpublication_completion_error_is_reconciled_as_committed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            original = study.registry.ExperimentRun.write_artifact
            def faulty_write(run, name, data):
                original(run, name, data)
                if name == "discovery_complete.json":
                    raise OSError("simulated post-publication completion cleanup failure")
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=faulty_write):
                frozen = study.run_development_discovery(path, fake_public(),
                                                        lambda index, ids: self.development_case(index))
            self.assertTrue(frozen["publication_warning"])
            self.assertTrue((path / "artifacts" / "discovery_warning.json").exists())
            result = study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                     frozen["predictions_sha256"],
                                                     lambda index: (0, self.development_case(index)))
            self.assertEqual(result["completed_cases"], 8)

    def test_terminal_finalize_error_after_publication_returns_committed_warning(self):
        public, reveal = protocol.seal_manifest(
            [[i + 1] for i in range(8)], [[i + 101] for i in range(8)],
            site=2, site_nonce=b"a" * 32, split_nonce=b"b" * 32,
            source_commit="1" * 40, protocol_sha256="2" * 64,
            runtime={k: ("cpu" if k == "device" else "development-only") for k in
                     ("python", "torch", "transformers", "device", "driver", "deterministic_settings")},
            created_utc="2026-10-28T00:00:00Z")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, public,
                                                    lambda index, ids: self.development_case(index))
            evaluation = study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                         frozen["predictions_sha256"],
                                                         lambda index: (0, self.development_case(index)))
            original = study.registry.ExperimentRun.finalize
            def faulty_finalize(run, state, details):
                original(run, state, details)
                raise OSError("simulated terminal cleanup failure")
            with mock.patch.object(study.registry.ExperimentRun, "finalize", new=faulty_finalize):
                result = study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                                                         frozen["predictions_sha256"],
                                                         evaluation["evaluation_sha256"], reveal)
            self.assertEqual(result["state"], "succeeded")
            self.assertTrue(result["publication_warning"])
            self.assertEqual(study.registry.verify_run(path)["state"], "succeeded")
            warning = json.loads((path / ".phase" / "reveal_warning.json").read_bytes())
            self.assertEqual(warning["stage"], "reveal")
            self.assertNotIn("simulated", json.dumps(warning))

    def test_interrupt_after_completion_marker_surfaces_committed_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            path, public = Path(temp) / "run", fake_public()
            original = study.registry.ExperimentRun.write_artifact
            def interrupt_after_marker(run, name, data):
                original(run, name, data)
                if name == "discovery_complete.json":
                    raise KeyboardInterrupt()
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=interrupt_after_marker):
                with self.assertRaises(study.StudyCommittedCancellation) as caught:
                    study.run_development_discovery(path, public,
                                                    lambda index, ids: self.development_case(index))
            self.assertEqual(caught.exception.stage, "discovery")
            self.assertEqual(caught.exception.artifact_sha256,
                             study.registry.sha256_file(path / "artifacts" / "predictions.json"))
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            self.assertTrue((path / "artifacts" / "discovery_complete.json").exists())
            self.assertFalse((path / "artifacts" / "evaluation_claim.json").exists())
            predictions_digest = caught.exception.artifact_sha256
            def interrupt_evaluation(run, name, data):
                original(run, name, data)
                if name == "evaluation_complete.json":
                    raise KeyboardInterrupt()
            with mock.patch.object(study.registry.ExperimentRun, "write_artifact", new=interrupt_evaluation):
                with self.assertRaises(study.StudyCommittedCancellation) as caught:
                    study.run_development_evaluation(path, protocol.canonical_sha256(public), predictions_digest,
                                                     lambda index: (0, self.development_case(index)))
            self.assertEqual(caught.exception.stage, "evaluation")
            self.assertEqual(caught.exception.artifact_sha256,
                             study.registry.sha256_file(path / "artifacts" / "evaluation.json"))
            self.assertEqual(study.registry.verify_run(path)["state"], "incomplete")
            self.assertFalse((path / "artifacts" / "reveal.json").exists())

    def test_interrupt_after_terminal_reveal_surfaces_committed_cancellation(self):
        public, reveal = protocol.seal_manifest(
            [[i + 1] for i in range(8)], [[i + 101] for i in range(8)],
            site=2, site_nonce=b"a" * 32, split_nonce=b"b" * 32,
            source_commit="1" * 40, protocol_sha256="2" * 64,
            runtime={k: ("cpu" if k == "device" else "development-only") for k in
                     ("python", "torch", "transformers", "device", "driver", "deterministic_settings")},
            created_utc="2026-10-28T00:00:00Z")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, public,
                                                    lambda index, ids: self.development_case(index))
            evaluation = study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                         frozen["predictions_sha256"],
                                                         lambda index: (0, self.development_case(index)))
            original = study.registry.ExperimentRun.finalize
            def interrupt_after_finalize(run, state, details):
                original(run, state, details)
                raise KeyboardInterrupt()
            with mock.patch.object(study.registry.ExperimentRun, "finalize", new=interrupt_after_finalize):
                with self.assertRaises(study.StudyCommittedCancellation) as caught:
                    study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                                                    frozen["predictions_sha256"],
                                                    evaluation["evaluation_sha256"], reveal)
            self.assertEqual(caught.exception.stage, "reveal")
            self.assertEqual(caught.exception.artifact_sha256,
                             study.registry.sha256_file(path / "artifacts" / "reveal.json"))
            self.assertEqual(study.registry.verify_run(path)["state"], "succeeded")

    def test_interrupt_during_terminal_reconciliation_remains_cancellation(self):
        public, reveal = protocol.seal_manifest(
            [[i + 1] for i in range(8)], [[i + 101] for i in range(8)],
            site=2, site_nonce=b"a" * 32, split_nonce=b"b" * 32,
            source_commit="1" * 40, protocol_sha256="2" * 64,
            runtime={k: ("cpu" if k == "device" else "development-only") for k in
                     ("python", "torch", "transformers", "device", "driver", "deterministic_settings")},
            created_utc="2026-10-28T00:00:00Z")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            frozen = study.run_development_discovery(path, public,
                                                    lambda index, ids: self.development_case(index))
            evaluation = study.run_development_evaluation(path, frozen["public_manifest_sha256"],
                                                         frozen["predictions_sha256"],
                                                         lambda index: (0, self.development_case(index)))
            finalized = False
            original_finalize = study.registry.ExperimentRun.finalize
            original_status = study.registry.ExperimentRun.status
            def faulty_finalize(run, state, details):
                nonlocal finalized
                original_finalize(run, state, details)
                finalized = True
                raise OSError("terminal cleanup failed")
            def interrupted_status(run):
                if finalized:
                    raise KeyboardInterrupt()
                return original_status(run)
            with mock.patch.object(study.registry.ExperimentRun, "finalize", new=faulty_finalize):
                with mock.patch.object(study.registry.ExperimentRun, "status", new=interrupted_status):
                    with self.assertRaises(study.StudyCancellationUncertain) as caught:
                        study.publish_development_reveal(path, frozen["public_manifest_sha256"],
                                                        frozen["predictions_sha256"],
                                                        evaluation["evaluation_sha256"], reveal)
            self.assertEqual(caught.exception.stage, "reveal")
            self.assertEqual(study.registry.verify_run(path)["state"], "succeeded")


if __name__ == "__main__":
    unittest.main()
