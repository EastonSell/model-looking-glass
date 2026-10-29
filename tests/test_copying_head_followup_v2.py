"""Synthetic CPU checks for the expanded public head follow-up runner."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from looking_glass.copy_head_followup import ARMS, action_order
from scripts import copying_head_eval_v1 as base
from scripts import copying_head_followup_v2 as followup


def synthetic_cases():
    groups, cases = [], []
    for group in range(8):
        words = [f"fake{group}token{index}" for index in range(6)]
        tokens = [32000 + 6 * group + index for index in range(6)]
        groups.append({"words": words, "token_ids": tokens})
        for variant, order in enumerate(base.PERMUTATIONS):
            indices = order + (0, 1, 2, 3, 4)
            cases.append({"case_id": f"copy-followup-{group + 1:02d}-{variant}",
                          "group": group + 1, "variant": variant,
                          "text": "".join(" " + words[index] for index in indices),
                          "token_ids": [tokens[index] for index in indices], "target_id": tokens[5]})
    return {"schema": "distilgpt2-copy-head-followup-cases-v2", "model_id": base.MODEL_ID,
            "model_revision": base.MODEL_REVISION,
            "protocol_revision": followup.PROTOCOL_REVISION, "protocol_sha256": followup.PROTOCOL_SHA256,
            "known_exposure_baseline_v7_sha256": base.EXPOSURE_SHA256,
            "known_exposure_current_version": followup.V8_EXPOSURE_VERSION,
            "known_exposure_current_sha256": followup.V8_EXPOSURE_SHA256,
            "tokenizer_sha256": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
            "arms": [{"layer": 3, "head": head, "position": position} for _, head, position in ARMS],
            "rotation": "(zero_based_family + 2 * variant) % 6, left",
            "expected_forwards": 288, "raw_tensor_bytes": followup.EXPECTED_RAW_BYTES,
            "permutations": [list(row) for row in base.PERMUTATIONS], "groups": groups, "cases": cases}


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        self.transformer.h[3].attn = torch.nn.Module()
        self.transformer.h[3].attn.c_proj = torch.nn.Linear(768, 768, bias=False)

    def forward(self, *, input_ids, **kwargs):
        value = (torch.arange(11 * 768, dtype=torch.float32).reshape(1, 11, 768) + 1) / 1000
        projected = self.transformer.h[3].attn.c_proj(value)
        logits = torch.zeros((1, 11, base.VOCAB), dtype=torch.float32)
        logits[:, :, :768] = projected
        return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)


class FollowupRunnerTests(unittest.TestCase):
    def test_cpu_backend_environment_drift_is_rejected(self):
        absent = {name: {"present": False, "value": None} for name in followup.CPU_ENV_NAMES}
        with patch.object(followup, "CPU_ENV_AT_TORCH_IMPORT", absent), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(followup.check_cpu_backend_environment(), absent)
            for name in followup.CPU_ENV_NAMES:
                with patch.dict(os.environ, {name: "unexpected"}):
                    with self.assertRaisesRegex(RuntimeError, "CPU backend environment"):
                        followup.check_cpu_backend_environment()

    def test_previous_saved_evidence_selects_both_prespecified_heads(self):
        previous = followup.verify_previous()
        self.assertEqual(previous["discovery"]["selected"]["head"], 10)
        self.assertEqual(previous["prior_source_revision"], followup.PRIOR_SOURCE_REVISION)
        self.assertAlmostEqual(previous["prior_candidate_mean"], -0.6381090603940153)

    def test_case_validation_and_unpublished_gate(self):
        data = synthetic_cases()
        self.assertEqual(len(followup.validate_cases(data)), 32)
        data["cases"][0]["token_ids"][0] += 1
        with self.assertRaises(ValueError):
            followup.validate_cases(data)
        for key, bad in (("protocol_revision", "different"), ("protocol_sha256", "different"),
                         ("tokenizer_sha256", "different"), ("rotation", "different"),
                         ("expected_forwards", 287), ("raw_tensor_bytes", 1),
                         ("known_exposure_baseline_v7_sha256", "different"),
                         ("known_exposure_current_version", "different"),
                         ("known_exposure_current_sha256", "different"), ("arms", [])):
            candidate = copy.deepcopy(synthetic_cases())
            candidate[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                followup.validate_cases(candidate)
        with patch.object(followup, "PROTOCOL_SHA256", None):
            with self.assertRaisesRegex(RuntimeError, "not all frozen"):
                followup.inventory()

    def test_rejects_known_and_internal_prefixes_in_both_directions(self):
        followup.reject_prefix_overlap(((1, 2), (3, 4)), ((9, 9),))
        for proposed, old in ((((1, 2, 3),), ((1, 2),)),
                              (((1, 2),), ((1, 2, 3),)),
                              (((1, 2), (1, 2, 3)), ((9,),)),
                              (((1, 2),), ((),))):
            with self.subTest(proposed=proposed, old=old), self.assertRaises(RuntimeError):
                followup.reject_prefix_overlap(proposed, old)

    def test_published_cases_match_v9_and_reject_wrong_version(self):
        from looking_glass import pretrained_exposures_v9 as v9

        cases = followup.inventory()
        self.assertEqual(len(cases), 32)
        self.assertEqual(followup.check_exposure(cases)["v9"], followup.V9_EXPOSURE_SHA256)
        with patch.object(v9, "INVENTORY_VERSION", "wrong-version"):
            with self.assertRaisesRegex(RuntimeError, "exposure reservation digest changed"):
                followup.check_exposure(cases)

    def test_every_arm_has_exact_payload_and_unhooked_controls(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(19)
            model = SyntheticModel().eval()
        ids = torch.arange(11, dtype=torch.int64).reshape(1, 11)
        payloads = {action: followup.run_forward(model, ids, action) for action in action_order(0, 0)}
        self.assertEqual(payloads["clean"]["logits"], payloads["noop"]["logits"])
        self.assertEqual(payloads["clean"]["logits"], payloads["clean_after"]["logits"])
        self.assertEqual(set(payloads["clean"]), {"logits"})
        for action, head, position in ARMS:
            payload = payloads[action]
            self.assertEqual(payload["cproj_before"], payloads["noop"]["cproj_before"])
            witness = followup.arm_witness(action, payload)
            self.assertEqual((witness["head"], witness["position"]), (head, position))
            self.assertEqual(witness["changed_float32_bit_patterns"], 64)
            self.assertEqual(sum(map(len, payload.values())), 302404)

    def test_failed_control_saves_raw_bytes_before_rejection(self):
        case = synthetic_cases()["cases"][0]
        payload = {"logits": bytes(base.LOGITS_BYTES), "cproj_before": bytes(base.WITNESS_BYTES),
                   "cproj_after": bytes(base.WITNESS_BYTES), "cproj_output": bytes(base.WITNESS_BYTES)}
        marker = {"recorded": False}
        with tempfile.TemporaryDirectory() as directory:
            raw_path, log_path = Path(directory) / "raw.f32", Path(directory) / "attempt.jsonl"
            with raw_path.open("xb") as raw, log_path.open("xb") as log:
                with self.assertRaisesRegex(RuntimeError, "logit bits differ"):
                    followup.persist_action(log, raw, case, "noop", payload, b"different", None,
                                            lambda: None, marker)
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertTrue(marker["recorded"])
            self.assertEqual([event["event"] for event in events], ["raw_recorded"])
            saved = raw_path.read_bytes()
            for name, item in events[0]["raw"].items():
                self.assertEqual(saved[item["offset"]:item["offset"] + item["bytes"]], payload[name])


if __name__ == "__main__":
    unittest.main()
