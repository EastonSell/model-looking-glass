"""Fabricated records only: no synthetic v2 inputs, schedules, or model runs."""
import copy
import json
import pickle
from pathlib import Path
import tempfile
import unittest

from looking_glass import registry
from looking_glass.v2_evidence import create_evidence
from looking_glass.causal_v2_evidence_adapter import (
    CELL_CHUNK_LIMIT, CellEvidence, CellEvidenceError,
)


def encoded(value):
    return registry._json_bytes(value)


def tensor(width, *, bits=None):
    return {"dtype": "torch.float32", "device": "cpu", "shape": [1, width],
            "bits": list(bits if bits is not None else [0] * width)}


def trace(label):
    return [{"metadata": {"run_id": label, "node_id": f"block.{i}.output",
                           "sequence": i, "occurrence": 0, "shape": [1, 8],
                           "dtype": "torch.float32", "device": "cpu", "tensor_bytes": 32},
             "value": tensor(8, bits=[0x80000000, 1] + [0] * 6)} for i in range(4)]


def record(index, strength, *, phase="discovery"):
    plans = []
    for kind, sites in (("forward", (1, 2, 3)), ("reverse", (1, 2, 3)),
                        ("null", (1, 2, 3)), ("self", (1, 2, 3)), ("upstream", (0,))):
        for site in sites:
            captures = {"post_action": tensor(8)}
            if kind == "self":
                captures["self_source"] = tensor(8, bits=[1] + [0] * 7)
            plans.append({"kind": kind, "site": site,
                          "metadata": {"steps": [{"flag": False, "order": 0}],
                                       "capture_bytes": 64 if kind == "self" else 32},
                          "captures": captures,
                          "source_id": (None if kind == "null" else
                                        "same_forward_self_source" if kind == "self" else "source-ref"),
                          "base_id": "base-ref"})
    l2 = lambda: tensor(2, bits=[0x80000000, 1])
    return {"schema": "causal-v2-condition-record-v1", "cell_id": "opaque-cell",
            "phase": phase, "case_id": f"case-{index:02d}", "case_index": index,
            "condition_id": f"condition-{index:02d}-{strength}", "model_seed": 17,
            "model_state_sha256": "a" * 64, "input_sha256": f"{index:064x}",
            "strength": strength,
            "prediction_ref": None if phase == "discovery" else "committed-prediction-ref",
            "traces": {"clean": trace("clean"), "repeat_clean": trace("repeat"),
                       "perturbed": trace("perturbed")},
            "logits": {"clean": l2(), "repeat_clean": l2(), "perturbed": l2(),
                       "forward": [l2() for _ in range(3)], "reverse": [l2() for _ in range(3)],
                       "null": [l2() for _ in range(3)], "self": [l2() for _ in range(3)],
                       "upstream": l2(), "post_hook_clean": l2()},
            "plans": plans,
            "private_integrity": {"target": "private-location", "delta": strength,
                                  "clean_target": tensor(8), "perturbed_target": tensor(8),
                                  "clean_capture_metadata": {"invocation": 0},
                                  "perturbed_capture_metadata": {"invocation": 0}},
            "comparison": {"status": "fabricated", "first_unequal": None}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run"
        self.writer = create_evidence(self.path, protocol={}, inputs={}, model={}, code={}, metadata={},
                                      chunk_limits={"cell.json": CELL_CHUNK_LIMIT}, extra_limits={},
                                      required_extra=[], failure_limit=4096,
                                      disk_limit_bytes=16 * 1024 * 1024)
        self.cell = CellEvidence(self.writer, name="cell.json", cell_id="opaque-cell",
                                 phase="discovery", model_seed=17, model_state_sha256="a" * 64)

    def test_full_cell_preserves_exact_private_records(self):
        expected = []
        for index in range(16):
            for strength in (0.25, 0.75):
                row = record(index, strength)
                expected.append(row)
                self.cell.append(encoded(row))
        self.cell.publish_complete()
        self.writer.finalize("succeeded", {"conditions": 32})
        raw = (self.path / "artifacts" / "cell.json").read_bytes()
        value = json.loads(raw)
        self.assertEqual(raw, encoded(value))
        self.assertEqual(value["state"], "complete")
        self.assertEqual(value["records"], expected)
        self.assertEqual(value["records"][0]["traces"]["clean"][0]["value"]["bits"][:2],
                         [0x80000000, 1])
        self.assertEqual(value["records"][0]["plans"][9]["source_id"],
                         "same_forward_self_source")
        self.assertEqual(self.writer.status()["state"], "succeeded")

    def test_invalid_next_record_retains_completed_conditions_as_incomplete(self):
        good = record(0, 0.25)
        self.cell.append(encoded(good))
        bad = record(0, 0.75)
        bad["plans"][0]["site"] = True
        with self.assertRaises(CellEvidenceError):
            self.cell.append(encoded(bad))
        with self.assertRaises(CellEvidenceError):
            self.cell.append(encoded(record(0, 0.75)))
        self.cell.publish_incomplete()
        raw = (self.path / "artifacts" / "cell.json").read_bytes()
        value = json.loads(raw)
        self.assertEqual(value["state"], "incomplete")
        self.assertEqual(value["records"], [good])
        failure = {"schema": "causal-v2-failure-v1", "outcome": "failed", "stage": "fake",
                   "error_type": "FakeError", "message": "complete detail", "submissions": 3,
                   "root_forwards": 2, "cell_id": "opaque-cell", "phase": "discovery"}
        self.writer.publish_failure(encoded(failure))
        self.writer.finalize("failed", {"completed_conditions": 1})
        self.assertEqual(self.writer.status()["state"], "failed")

    def test_partial_record_is_retained_without_claiming_completion(self):
        self.cell.append(encoded(record(0, 0.25)))
        partial = {"schema": "causal-v2-partial-record-v1", "cell_id": "opaque-cell",
                   "phase": "discovery", "model_seed": 17, "stage": "plan",
                   "evidence": {"completed_steps": [{"snapshot_bits": [0x80000000, 1]}]}}
        self.cell.publish_incomplete(encoded(partial))
        value = json.loads((self.path / "artifacts" / "cell.json").read_bytes())
        self.assertEqual(value["state"], "incomplete")
        self.assertEqual(value["records"][-1], partial)

    def test_bad_partial_can_salvage_prefix_with_explicit_unavailable_marker(self):
        good = record(0, 0.25)
        self.cell.append(encoded(good))
        with self.assertRaises(CellEvidenceError):
            self.cell.publish_incomplete(b"{bad")
        with self.assertRaises(CellEvidenceError):
            self.cell.publish_complete()
        with self.assertRaises(CellEvidenceError):
            self.cell.publish_incomplete(b"{bad")
        self.cell.publish_incomplete()
        value = json.loads((self.path / "artifacts" / "cell.json").read_bytes())
        self.assertEqual(value["state"], "incomplete")
        self.assertEqual(value["records"][0], good)
        self.assertEqual(value["records"][1],
                         {"schema": "causal-v2-partial-unavailable-v1", "cell_id": "opaque-cell",
                          "phase": "discovery", "model_seed": 17,
                          "reason": "invalid_or_oversize_partial"})

    def test_rejects_noncanonical_oversize_and_wrong_identity(self):
        cases = []
        cases.append(b" " + encoded(record(0, 0.25)))
        cases.append(b"x" * (65536 + 1))
        wrong = record(0, 0.25); wrong["input_sha256"] = "not-a-digest"
        cases.append(encoded(wrong))
        wrong = record(0, 0.25); wrong["model_seed"] = True
        cases.append(encoded(wrong))
        wrong = record(0, 0.25); wrong["logits"]["post_hook_clean"]["bits"][0] = 0x7f800000
        cases.append(encoded(wrong))
        for i, data in enumerate(cases):
            with self.subTest(index=i):
                cell = CellEvidence(self.writer, name="cell.json", cell_id="opaque-cell",
                                    phase="discovery", model_seed=17, model_state_sha256="a" * 64)
                with self.assertRaises(CellEvidenceError):
                    cell.append(data)

    def test_missing_conditions_cannot_publish_complete(self):
        self.cell.append(encoded(record(0, 0.25)))
        with self.assertRaises(CellEvidenceError):
            self.cell.publish_complete()
        self.cell.publish_incomplete()
        self.assertEqual(self.writer.status()["state"], "incomplete")

    def test_same_case_pair_and_distinct_cases_are_required(self):
        self.cell.append(encoded(record(0, 0.25)))
        bad = record(0, 0.75)
        bad["input_sha256"] = "e" * 64
        with self.assertRaises(CellEvidenceError):
            self.cell.append(encoded(bad))
        other = CellEvidence(self.writer, name="cell.json", cell_id="opaque-cell",
                             phase="discovery", model_seed=17, model_state_sha256="a" * 64)
        other.append(encoded(record(0, 0.25)))
        other.append(encoded(record(0, 0.75)))
        repeated = record(1, 0.25)
        repeated["input_sha256"] = record(0, 0.25)["input_sha256"]
        with self.assertRaises(CellEvidenceError):
            other.append(encoded(repeated))

    def test_identity_is_read_only_and_buffer_is_not_copyable(self):
        for field in ("name", "cell_id", "phase", "model_seed", "model_state_sha256"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(self.cell, field, "changed")
        for method in (copy.copy, copy.deepcopy, pickle.dumps):
            with self.assertRaises(TypeError):
                method(self.cell)

    def test_fabricated_heldout_record_requires_a_prediction_reference(self):
        cell = CellEvidence(self.writer, name="cell.json", cell_id="opaque-cell",
                            phase="heldout", model_seed=17, model_state_sha256="a" * 64)
        valid = record(0, 0.25, phase="heldout")
        cell.append(encoded(valid))
        invalid = record(0, 0.75, phase="heldout")
        invalid["prediction_ref"] = None
        with self.assertRaises(CellEvidenceError):
            cell.append(encoded(invalid))
        cell.publish_incomplete()
        output = json.loads((self.path / "artifacts" / "cell.json").read_bytes())
        self.assertEqual(output["records"], [valid])

    def test_publication_error_is_fail_stop_even_after_link(self):
        class RaiseAfterPublication:
            def publish_chunk(inner, name, data):
                self.writer.publish_chunk(name, data)
                raise OSError("ambiguous acknowledgement")
        cell = CellEvidence(RaiseAfterPublication(), name="cell.json", cell_id="opaque-cell",
                            phase="discovery", model_seed=17, model_state_sha256="a" * 64)
        with self.assertRaises(OSError):
            cell.publish_incomplete()
        with self.assertRaises(CellEvidenceError):
            cell.publish_incomplete()
        state = self.writer.status()
        self.assertEqual(state["state"], "incomplete")
        self.assertIn("cell.json", state["artifacts"])

    def test_full_layout_declarations_fit_registry_and_host_envelope(self):
        chunk_limits = {f"cell-{cell:02d}-{phase}.json": CELL_CHUNK_LIMIT
                        for cell in range(54) for phase in ("d", "h")}
        self.assertEqual(len(chunk_limits), 108)
        self.assertEqual(sum(chunk_limits.values()), 234012672)
        path = Path(self.temp.name) / "full-layout"
        run = create_evidence(path, protocol={}, inputs={}, model={}, code={}, metadata={},
                              chunk_limits=chunk_limits, extra_limits={"freeze.json": 16 * 1024 * 1024},
                              required_extra=["freeze.json"], failure_limit=65536,
                              external_reserve_bytes=256 * 1024 * 1024)
        self.assertEqual(run.status()["state"], "incomplete")
        self.assertEqual(len(run.config["chunks"]), 108)


if __name__ == "__main__":
    unittest.main()
