"""Fake executor values only; no v2 inputs or schedules are generated."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from looking_glass import registry
from looking_glass.causal_v2_selection import DiscoveryCondition
from looking_glass.probes import TraceEvent
from looking_glass.v2_evidence import create_evidence
from looking_glass.causal_v2_evidence_adapter import CELL_CHUNK_LIMIT, CellEvidence, CellEvidenceError
from looking_glass.causal_v2_evidence_encoder import append_executor_result


def tensor(width):
    raw = [-2147483648, 1] + [0] * (width - 2)
    return torch.tensor([raw], dtype=torch.int32).view(torch.float32)


def trace(label):
    return tuple(TraceEvent(label, f"block.{i}.output", i, 0, (1, 8),
                            "torch.float32", "cpu", tensor(8)) for i in range(4))


def result(strength=0.25):
    kinds = [(kind, site) for kind in ("forward", "reverse", "null", "self")
             for site in (1, 2, 3)] + [("upstream", 0)]
    plans = []
    for kind, site in kinds:
        source = {"forward": "clean_sparse_trace", "reverse": "perturbed_sparse_trace",
                  "null": None, "self": "same_forward_self_source", "upstream": "clean_sparse_trace"}[kind]
        comparison = "perturbed_sparse_trace" if kind in ("reverse", "null", "self") else "clean_sparse_trace"
        captures = {"post_action": [0x80000000, 1] + [0] * 6}
        if kind == "self":
            captures["self_source"] = [0x80000000, 1] + [0] * 6
        plans.append({"kind": kind, "site": site, "metadata": {"steps": [{"flag": False, "order": 0}]},
                      "captures": captures, "source_identity": source, "comparison_identity": comparison,
                      "base_identity": "clean" if kind == "reverse" else "perturbed"})
    condition = DiscoveryCondition(strength, trace("perturbed"), tensor(2), tuple(tensor(2) for _ in range(3)),
                                   tuple(tensor(2) for _ in range(3)), tuple(tensor(2) for _ in range(3)),
                                   tuple(tensor(2) for _ in range(3)), tensor(2), tensor(2))
    return {"clean_trace": trace("clean"), "repeat_trace": trace("repeat"), "clean_logits": tensor(2),
            "repeat_logits": tensor(2), "condition": condition, "plans": plans,
            "private_integrity": {"target": "private-location", "delta": strength,
                                  "clean_target_bits": [0x80000000, 1] + [0] * 6,
                                  "perturbed_target_bits": [0x80000000, 1] + [0] * 6,
                                  "clean_capture_metadata": {"invocation": 0},
                                  "perturbed_capture_metadata": {"invocation": 0}},
            "comparison": {"status": "fake"}, "input_sha256": "b" * 64,
            "model_state_sha256": "a" * 64}


REFERENCES = {"clean_sparse_trace": "clean-trace", "perturbed_sparse_trace": "changed-trace",
              "clean": "clean-base", "perturbed": "changed-base"}


class EncoderTests(unittest.TestCase):
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

    def append(self, data, *, case_id="case-0", case_index=0, condition_id="condition-0", refs=REFERENCES):
        return append_executor_result(self.cell, data, case_id=case_id, case_index=case_index,
                                      condition_id=condition_id, prediction_ref=None, references=refs)

    def test_fake_executor_round_trip_preserves_all_plan_roles_and_bits(self):
        data = result()
        digest = self.append(data)
        self.cell.publish_incomplete()
        record = json.loads((self.path / "artifacts" / "cell.json").read_bytes())["records"][0]
        self.assertEqual(hashlib.sha256(registry._json_bytes(record)).hexdigest(), digest)
        self.assertEqual(record["schema"], "causal-v2-condition-record-v2")
        self.assertEqual(record["traces"]["clean"][0]["value"]["bits"][:2], [0x80000000, 1])
        self.assertEqual(record["logits"]["post_hook_clean"]["bits"][:2], [0x80000000, 1])
        self.assertEqual(record["private_integrity"]["clean_target"]["bits"][:2], [0x80000000, 1])
        self.assertEqual(record["private_integrity"]["perturbed_capture_metadata"], {"invocation": 0})
        self.assertEqual(record["plans"][0]["metadata"], data["plans"][0]["metadata"])
        self.assertEqual(record["plans"][0]["source_id"], "clean-trace/block.1.output")
        self.assertEqual(record["plans"][9]["source_role"], "same_forward_self_source")
        self.assertEqual(record["plans"][9]["comparison_role"], "perturbed_sparse_trace")
        self.assertEqual(record["plans"][9]["comparison_id"], "changed-trace/block.1.output")
        self.assertEqual(record["plans"][9]["base_id"], "changed-base")
        self.assertFalse(torch.cuda.is_initialized())

    def test_complete_fake_cell_encodes_all_32_ordered_conditions(self):
        for index in range(16):
            for strength in (0.25, 0.75):
                data = result(strength)
                data["input_sha256"] = f"{index:064x}"
                self.append(data, case_id=f"fake-{index}", case_index=index,
                            condition_id=f"fake-{index}-{strength}")
        self.cell.publish_complete()
        self.writer.finalize("succeeded", {"conditions": 32})
        saved = json.loads((self.path / "artifacts" / "cell.json").read_bytes())
        self.assertEqual(saved["state"], "complete")
        self.assertEqual(len(saved["records"]), 32)
        self.assertEqual({row["schema"] for row in saved["records"]}, {"causal-v2-condition-record-v2"})
        self.assertEqual(self.writer.status()["state"], "succeeded")

    def test_encoder_failure_marks_cell_failed_and_retains_prefix(self):
        self.append(result())
        invalid = result(0.75)
        invalid["plans"][0]["comparison_identity"] = "unrecognized"
        with self.assertRaises(ValueError):
            self.append(invalid, condition_id="condition-1")
        with self.assertRaises(CellEvidenceError):
            self.append(result(0.75), condition_id="condition-1")
        self.cell.publish_incomplete()
        saved = json.loads((self.path / "artifacts" / "cell.json").read_bytes())
        self.assertEqual(saved["state"], "incomplete")
        self.assertEqual(len(saved["records"]), 1)

    def test_unknown_executor_fields_are_not_silently_dropped(self):
        invalid = result()
        invalid["extra_unretained_evidence"] = {"important": True}
        with self.assertRaises(ValueError):
            self.append(invalid)
        with self.assertRaises(CellEvidenceError):
            self.cell.publish_complete()
        self.cell.publish_incomplete()

    def test_interrupt_during_encoding_is_sticky(self):
        with patch("looking_glass.causal_v2_evidence_encoder._tensor", side_effect=KeyboardInterrupt("stop")):
            with self.assertRaises(KeyboardInterrupt):
                self.append(result())
        with self.assertRaises(CellEvidenceError):
            self.append(result())
        self.cell.publish_incomplete()

    def test_references_must_be_distinct_and_tensor_shape_valid(self):
        with self.assertRaises(ValueError):
            self.append(result(), refs={**REFERENCES, "clean": "changed-base"})
        other = CellEvidence(self.writer, name="cell.json", cell_id="opaque-cell",
                             phase="discovery", model_seed=17, model_state_sha256="a" * 64)
        bad = result(); bad["clean_logits"] = torch.zeros((1, 3), dtype=torch.float32)
        with self.assertRaises(ValueError):
            append_executor_result(other, bad, case_id="case-0", case_index=0,
                                   condition_id="condition-0", prediction_ref=None, references=REFERENCES)
        other.publish_incomplete()

    def test_strided_subnormal_and_signed_zero_bits_survive_both_flush_modes(self):
        # Use a separate process because the flush setting is thread-local and
        # must not leak into other tests or a real study's numeric policy.
        code = r'''
import torch
from looking_glass.causal_v2_evidence_encoder import _tensor
for enabled in (False, True):
    if not torch.set_flush_denormal(enabled):
        raise SystemExit(77)
    raw = [1, 10, -2147483647, 10, -2147483648, 10, 0, 10,
           1, 10, -2147483647, 10, -2147483648, 10, 0, 10]
    value = torch.tensor([raw], dtype=torch.int32).view(torch.float32)[:, ::2]
    assert not value.is_contiguous()
    assert _tensor(value, 8)['bits'] == [1, 0x80000001, 0x80000000, 0] * 2
assert not torch.cuda.is_initialized()
'''
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                               text=True, timeout=30)
        if child.returncode == 77:
            self.skipTest("this host cannot configure denormal flush mode")
        self.assertEqual(child.returncode, 0, child.stderr)


if __name__ == "__main__":
    unittest.main()
