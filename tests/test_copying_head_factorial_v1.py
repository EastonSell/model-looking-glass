"""Synthetic CPU checks for the public matched-dose factorial runner."""
import copy
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from looking_glass.copy_head_factorial import HEADS, head_slice, plan_family
from scripts import copying_head_factorial_v1 as runner


def synthetic_cases():
    groups, cases = [], []
    for group in range(8):
        words = [f"fake{group}token{index}" for index in range(6)]
        tokens = [32000 + 6 * group + index for index in range(6)]
        groups.append({"words": words, "token_ids": tokens})
        for variant, order in enumerate(runner.base.PERMUTATIONS):
            indices = order + (0, 1, 2, 3, 4)
            cases.append({"case_id": f"copy-factorial-{group + 1:02d}-{variant}", "group": group + 1,
                          "variant": variant, "text": "".join(" " + words[index] for index in indices),
                          "token_ids": [tokens[index] for index in indices], "target_id": tokens[5]})
    return {"schema": "distilgpt2-copy-head-factorial-cases-v1", "model_id": runner.base.MODEL_ID,
            "model_revision": runner.base.MODEL_REVISION, "protocol_revision": runner.PROTOCOL_REVISION,
            "protocol_sha256": runner.PROTOCOL_SHA256, "known_exposure_version": runner.BASELINE_VERSION,
            "known_exposure_sha256": runner.BASELINE_SHA256,
            "selection_rule": "First six eligible words in each ordered public pool: single spaced token, no reuse of prior copying tokens or selected IDs, and no one-token prefix in the bound v11 inventory; no model forward.",
            "tokenizer_sha256": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
            "arms": [{"layer": 3, "heads": heads, "position": 10} for heads in ([10], [2], [7], [2, 10])],
            "baseline_order": "four variants: clean then noop; after family dose calculation, each variant runs four rotated arms then clean_after",
            "rotation": "(zero_based_family + variant) % 4, left", "expected_forwards": 224,
            "raw_tensor_bytes": runner.RAW_BYTES,
            "permutations": [list(row) for row in runner.base.PERMUTATIONS], "groups": groups, "cases": cases}


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
        logits = torch.zeros((1, 11, runner.base.VOCAB), dtype=torch.float32)
        logits[:, :, :768] = projected
        return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)


class FactorialRunnerTests(unittest.TestCase):
    def test_main_rejects_torch_preimport_and_environment_drift(self):
        self.assertIn("torch", sys.modules)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            with self.assertRaisesRegex(RuntimeError, "torch must not be imported"):
                runpy.run_path(str(Path(runner.__file__)), run_name="__main__")
        absent = {name: {"present": False, "value": None} for name in runner.ENV_NAMES}
        with patch.object(runner, "ENV_AT_IMPORT", absent), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runner.environment(), absent)
            for name in runner.ENV_NAMES:
                with patch.dict(os.environ, {name: "unexpected"}), self.assertRaises(RuntimeError):
                    runner.environment()

    def test_case_metadata_and_unpublished_gate(self):
        data = synthetic_cases()
        self.assertEqual(len(runner.validate_cases(data)), 32)
        for key, bad in (("protocol_revision", "bad"), ("known_exposure_version", "bad"),
                         ("known_exposure_sha256", "bad"), ("tokenizer_sha256", "bad"), ("selection_rule", "bad"),
                         ("arms", []), ("baseline_order", "bad"), ("rotation", "bad"),
                         ("expected_forwards", 223), ("raw_tensor_bytes", 0)):
            changed = copy.deepcopy(data); changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_cases(changed)
        changed = copy.deepcopy(data); changed["cases"][0]["token_ids"][0] += 1
        with self.assertRaises(ValueError):
            runner.validate_cases(changed)
        with patch.object(runner, "CASE_SHA256", None), self.assertRaisesRegex(RuntimeError, "not all frozen"):
            runner.inventory()

    def test_synthetic_witnesses_and_joint_match_single_bits(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(23)
            model = SyntheticModel().eval()
        ids = torch.arange(11, dtype=torch.int64).reshape(1, 11)
        clean = runner.run_forward(model, ids, "clean")
        noop = runner.run_forward(model, ids, "noop")
        self.assertEqual(clean["logits"], noop["logits"])
        self.assertEqual(noop["cproj_before"], noop["cproj_after"])
        plan = plan_family((noop["cproj_before"],) * 4)
        row = plan.variants[0]
        outputs = {}
        for name, heads in (("h10", (10,)), ("h2", (2,)), ("h7", (7,)), ("joint", (2, 10))):
            payload = runner.run_forward(model, ids, name, expected_before=noop["cproj_before"],
                                         deliveries={head: row[head].delivered for head in heads})
            self.assertEqual(payload["cproj_before"], noop["cproj_before"])
            outputs[name] = payload["cproj_after"]
        for head in (2, 10):
            self.assertEqual(head_slice(outputs["joint"], head), head_slice(outputs[f"h{head}"], head))
        self.assertEqual(clean["logits"], runner.run_forward(model, ids, "clean_after")["logits"])

    def test_failed_noop_control_saves_returned_bytes_first(self):
        case = synthetic_cases()["cases"][0]
        payload = {"logits": bytes(runner.base.LOGITS_BYTES), "cproj_before": bytes(runner.base.WITNESS_BYTES),
                   "cproj_after": bytes(runner.base.WITNESS_BYTES), "cproj_output": bytes(runner.base.WITNESS_BYTES)}
        marker = {"recorded": False}
        with tempfile.TemporaryDirectory() as directory:
            raw_path, log_path = Path(directory) / "raw.f32", Path(directory) / "attempt.jsonl"
            with raw_path.open("xb") as raw, log_path.open("xb") as log:
                with self.assertRaisesRegex(RuntimeError, "logits differ"):
                    runner.persist_action(log, raw, case, "noop", payload, b"different", None, lambda: None, marker)
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertTrue(marker["recorded"])
            self.assertEqual([event["event"] for event in events], ["raw_recorded"])
            saved = raw_path.read_bytes()
            for name, item in events[0]["raw"].items():
                self.assertEqual(saved[item["offset"]:item["offset"] + item["bytes"]], payload[name])

    def test_nonfinite_logits_and_projection_outputs_are_saved_before_failure(self):
        case = synthetic_cases()["cases"][0]
        class NonfiniteLogits(SyntheticModel):
            def forward(self, **kwargs):
                result = super().forward(**kwargs)
                result.logits[0, -1, 0] = float("nan")
                return result
        returned = runner.run_forward(NonfiniteLogits().eval(), torch.arange(11).reshape(1, 11), "clean")
        self.assertEqual(returned["logits"][:4], bytes.fromhex("0000c07f"))
        payload = {"logits": bytes(runner.base.LOGITS_BYTES), "cproj_before": bytes(runner.base.WITNESS_BYTES),
                   "cproj_after": bytes(runner.base.WITNESS_BYTES), "cproj_output": bytes(runner.base.WITNESS_BYTES)}
        for field in ("logits", "cproj_output"):
            changed = dict(payload)
            changed[field] = bytes.fromhex("0000c07f") + changed[field][4:]
            marker = {"recorded": False}
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                raw_path, log_path = Path(directory) / "raw.f32", Path(directory) / "attempt.jsonl"
                with raw_path.open("xb") as raw, log_path.open("xb") as log:
                    with self.assertRaisesRegex(RuntimeError, f"nonfinite {field}"):
                        runner.persist_action(log, raw, case, "noop", changed, payload["logits"], None,
                                              lambda: None, marker)
                event = json.loads(log_path.read_text().splitlines()[0])
                self.assertTrue(marker["recorded"])
                item = event["raw"][field]
                self.assertEqual(raw_path.read_bytes()[item["offset"]:item["offset"] + item["bytes"]], changed[field])

    def test_model_return_then_cleanup_failure_retains_bounded_logits(self):
        model = SyntheticModel().eval()
        ids = torch.arange(11, dtype=torch.int64).reshape(1, 11)
        observation = {}
        with patch.object(runner, "assert_no_forward_hooks", side_effect=[None, RuntimeError("cleanup")]):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                runner.run_forward(model, ids, "clean", observation=observation)
        self.assertTrue(observation["model_call_return_observed"])
        self.assertEqual(len(observation["payload"]["logits"]), runner.base.LOGITS_BYTES)
        marker = {"partial": False}
        with tempfile.TemporaryDirectory() as directory:
            raw_path, log_path = Path(directory) / "raw.f32", Path(directory) / "attempt.jsonl"
            with log_path.open("xb") as log:
                runner.persist_partial(log, raw_path, "case", "clean", observation, marker)
            event = json.loads(log_path.read_text())
            self.assertTrue(marker["partial"])
            self.assertEqual(event["event"], "raw_recorded_partial")
            self.assertEqual(raw_path.read_bytes(), observation["payload"]["logits"])

    def test_nonfinite_projection_output_returns_bounded_witness(self):
        class BadProjection(torch.nn.Module):
            def forward(self, value):
                return torch.full_like(value, float("nan"))
        class BadOutputModel(SyntheticModel):
            def __init__(self):
                super().__init__()
                self.transformer.h[3].attn.c_proj = BadProjection()
            def forward(self, *, input_ids, **kwargs):
                self.transformer.h[3].attn.c_proj(torch.ones((1, 11, 768), dtype=torch.float32))
                return SimpleNamespace(logits=torch.zeros((1, 11, runner.base.VOCAB)),
                                       past_key_values=None, attentions=None, hidden_states=None)
        payload = runner.run_forward(BadOutputModel().eval(), torch.arange(11).reshape(1, 11), "noop")
        self.assertEqual(payload["cproj_output"][:4], bytes.fromhex("0000c07f"))
        case = synthetic_cases()["cases"][0]
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "raw.f32").open("xb") as raw, (Path(directory) / "log").open("xb") as log:
                with self.assertRaisesRegex(RuntimeError, "nonfinite cproj_output"):
                    runner.persist_action(log, raw, case, "noop", payload, payload["logits"], None,
                                          lambda: None, {"recorded": False})

    def test_nonfinite_projection_input_is_retained_before_model_return(self):
        class BadInputModel(SyntheticModel):
            def forward(self, *, input_ids, **kwargs):
                value = torch.zeros((1, 11, 768), dtype=torch.float32)
                value[0, 10, 0] = float("nan")
                self.transformer.h[3].attn.c_proj(value)
                raise AssertionError("the projection pre-hook must reject this input")
        model = BadInputModel().eval()
        ids = torch.arange(11, dtype=torch.int64).reshape(1, 11)
        observation = {}
        with self.assertRaisesRegex(RuntimeError, "unexpected projection tensor"):
            runner.run_forward(model, ids, "noop", observation=observation)
        self.assertFalse(observation["model_call_return_observed"])
        self.assertEqual(set(observation["payload"]), {"cproj_before"})
        self.assertEqual(len(observation["payload"]["cproj_before"]), runner.base.WITNESS_BYTES)

    def test_full_synthetic_schedule_writes_all_records(self):
        class FastModel(SyntheticModel):
            def __init__(self):
                super().__init__()
                self.transformer.h[3].attn.c_proj = torch.nn.Identity()
        data = synthetic_cases()
        old_threads, old_dtype = torch.get_num_threads(), torch.get_default_dtype()
        old_deterministic, old_precision = torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision()
        try:
            torch.set_num_threads(1)
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
                root = Path(directory)
                repository, pycache, cache = root / "repo", root / "pycache", root / "cache"
                repository.mkdir(); pycache.mkdir(); cache.mkdir()
                (repository / "cases.json").write_text(json.dumps(data))
                output = root / "run"
                bundle = SimpleNamespace(model=FastModel().eval(), tokenizer=None,
                                         metadata={"weights_sha256": runner.base.WEIGHTS_SHA256})
                patches = [patch.object(runner, "ROOT", repository), patch.object(runner, "CASE_PATH", "cases.json"),
                           patch.object(runner, "inventory", return_value=data["cases"]),
                           patch.object(runner, "source_identity", return_value={"revision": "synthetic"}),
                           patch.object(runner, "environment", return_value=runner.EXPECTED_ENV),
                           patch.object(runner, "verify_previous", return_value={}),
                           patch.object(runner, "check_exposure", return_value={}),
                           patch.object(runner, "verify_tokenization"), patch.object(runner, "runtime", return_value={}),
                           patch.object(runner.base, "dependency_identity", return_value={}),
                           patch.object(runner.base, "numeric_policy", return_value={}),
                           patch.object(runner.base, "load_distilgpt2", return_value=bundle),
                           patch.object(runner.base, "state_hash", return_value="state"),
                           patch.object(runner.base, "buffer_hash", return_value="buffers"),
                           patch.object(sys, "argv", ["runner", "--cache-dir", str(cache), "--output-dir", str(output)]),
                           patch.object(sys, "dont_write_bytecode", True), patch.object(sys, "pycache_prefix", str(pycache)),
                           patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})]
                with contextlib.ExitStack() as stack:
                    for item in patches:
                        stack.enter_context(item)
                    with contextlib.redirect_stdout(io.StringIO()):
                        runner.main()
                result = json.loads((output / "result.json").read_bytes())
                self.assertEqual((result["submitted_forwards"], result["model_call_returns_observed"],
                                  result["returned_forwards"]), (224, 224, 224))
                self.assertEqual(result["raw_payload_bytes"], 61_250_432)
                self.assertEqual((len(result["family_doses"]), len(result["case_diagnostics"])), (8, 32))
                self.assertFalse((output / "failure.json").exists())
        finally:
            torch.set_num_threads(old_threads)
            torch.set_default_dtype(old_dtype)
            torch.use_deterministic_algorithms(old_deterministic)
            torch.set_float32_matmul_precision(old_precision)


if __name__ == "__main__":
    unittest.main()
