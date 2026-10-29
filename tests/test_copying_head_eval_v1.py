"""Pure and synthetic tests for the public head-ablation runner."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.copying_head_eval_v1 import (
    ACTIONS, CONTROL_DIGEST, HEADS, LENGTH, LOGITS_BYTES, MODEL_ID, MODEL_REVISION,
    PERMUTATIONS, SCREEN_RESULT_SHA256, SCREEN_REVISION, VOCAB, WIDTH,
    WITNESS_BYTES, _write_payload, action_witness, buffer_hash, inventory, persist_action, run_forward,
    selected_control, summarize, validate_inventory, verify_discovery,
)


def synthetic_inventory():
    groups, cases = [], []
    for group in range(8):
        words = [f"synthetic{group}word{index}" for index in range(6)]
        tokens = [30000 + 6 * group + index for index in range(6)]
        groups.append({"words": words, "token_ids": tokens})
        for variant, order in enumerate(PERMUTATIONS):
            indices = order + (0, 1, 2, 3, 4)
            cases.append({"case_id": f"copy-eval-{group + 1:02d}-{variant}",
                          "group": group + 1, "variant": variant,
                          "text": "".join(" " + words[index] for index in indices),
                          "token_ids": [tokens[index] for index in indices], "target_id": tokens[5]})
    return {"schema": "distilgpt2-copy-head-eval-cases-v1", "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "screen_source_revision": SCREEN_REVISION,
            "screen_result_sha256": SCREEN_RESULT_SHA256,
            "candidate": {"layer": 3, "head": 10, "channel_start": 640, "channel_end_exclusive": 704},
            "layer_matched_random_control": {"layer": 3, "head": 7, "channel_start": 448,
                                              "channel_end_exclusive": 512,
                                              "seed": 20261029, "sha256": CONTROL_DIGEST},
            "actions": list(ACTIONS), "expected_forwards": 160,
            "permutations": [list(row) for row in PERMUTATIONS], "groups": groups, "cases": cases}


class SyntheticProjectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        self.transformer.h[3].attn = torch.nn.Module()
        self.transformer.h[3].attn.c_proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, *, input_ids, **kwargs):
        value = (torch.arange(LENGTH * WIDTH, dtype=torch.float32).reshape(1, LENGTH, WIDTH) + 1) / 1000
        projected = self.transformer.h[3].attn.c_proj(value)
        logits = torch.zeros((1, LENGTH, VOCAB), dtype=torch.float32)
        logits[:, :, :WIDTH] = projected
        return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)


class CopyingHeadEvaluationTests(unittest.TestCase):
    def test_published_discovery_and_cases_replay_to_selected_head(self):
        _, cases = inventory()
        evidence = verify_discovery()
        self.assertEqual(len(cases), 32)
        self.assertEqual((evidence["selected"]["layer"], evidence["selected"]["head"]), (3, 10))

    def test_nonpersistent_buffer_mutation_changes_witness(self):
        model = torch.nn.Module()
        model.register_buffer("mask", torch.tensor([True, False]), persistent=False)
        before = buffer_hash(model)
        model.mask[0] = False
        self.assertNotEqual(before, buffer_hash(model))

    def test_fixed_control_and_case_validation_reject_changes(self):
        self.assertEqual(selected_control(), (CONTROL_DIGEST, 7))
        data = synthetic_inventory()
        self.assertEqual(len(validate_inventory(data)), 32)
        data["cases"][1]["token_ids"][0] += 1
        with self.assertRaises(ValueError):
            validate_inventory(data)
        data = synthetic_inventory()
        data["groups"][1]["token_ids"][0] = data["groups"][0]["token_ids"][0]
        with self.assertRaises(ValueError):
            validate_inventory(data)

    def test_unhooked_noop_zero_and_raw_offsets(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(9)
            model = SyntheticProjectionModel().eval()
        ids = torch.arange(LENGTH, dtype=torch.int64).reshape(1, LENGTH)
        payloads = {action: run_forward(model, ids, action) for action in ACTIONS}
        self.assertEqual(payloads["clean"]["logits"], payloads["noop"]["logits"])
        self.assertEqual(payloads["clean"]["logits"], payloads["clean_after"]["logits"])
        self.assertEqual(set(payloads["clean"]), {"logits"})
        self.assertEqual(len(payloads["clean"]["logits"]), LOGITS_BYTES)
        for action in ("candidate_zero", "control_zero"):
            self.assertEqual(payloads["noop"]["cproj_before"], payloads[action]["cproj_before"])
            self.assertEqual(len(payloads[action]["cproj_after"]), WITNESS_BYTES)
            witness = action_witness(action, payloads[action])
            self.assertEqual(witness["changed_float32_bit_patterns"], 64)
            self.assertGreater(witness["pre_head_l2"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.f32"
            with path.open("xb") as stream:
                entries = _write_payload(stream, payloads["candidate_zero"])
            raw = path.read_bytes()
            for name, entry in entries.items():
                self.assertEqual(raw[entry["offset"]:entry["offset"] + entry["bytes"]],
                                 payloads["candidate_zero"][name])

    def test_family_contrast_preserves_negative_results(self):
        records = []
        for _group in range(8):
            for variant in range(4):
                candidate = -2 if variant == 0 else -1
                control = -0.5 if variant == 0 else 0
                for action, value in zip(ACTIONS, (0, 0, candidate, control, 0)):
                    records.append({"action": action, "score": {"target_log_probability": value}})
        summary = summarize(records)
        self.assertEqual(summary["mean_candidate"], -1)
        self.assertEqual(summary["mean_control"], -0.5)
        self.assertEqual(summary["mean_candidate_minus_control"], -0.5)
        self.assertEqual(len(summary["families"]), 8)
        self.assertEqual(summary["families"][0]["candidate"]["case_deltas"], [-2, -1, -1, -1])

    def test_mismatched_noop_is_recorded_before_rejection(self):
        payload = {"logits": bytes(LOGITS_BYTES), "cproj_before": bytes(WITNESS_BYTES),
                   "cproj_after": bytes(WITNESS_BYTES), "cproj_output": bytes(WITNESS_BYTES)}
        case = synthetic_inventory()["cases"][0]
        marker = {"recorded": False}
        with tempfile.TemporaryDirectory() as directory:
            raw_path, log_path = Path(directory) / "raw.f32", Path(directory) / "attempt.jsonl"
            with raw_path.open("xb") as raw, log_path.open("xb") as log:
                with self.assertRaisesRegex(RuntimeError, "logit bits differ"):
                    persist_action(log, raw, case, "noop", payload, b"different", None,
                                   lambda: None, marker)
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertTrue(marker["recorded"])
            self.assertEqual([event["event"] for event in events], ["raw_recorded"])
            saved = raw_path.read_bytes()
            for name, item in events[0]["raw"].items():
                self.assertEqual(saved[item["offset"]:item["offset"] + item["bytes"]], payload[name])


if __name__ == "__main__":
    unittest.main()
