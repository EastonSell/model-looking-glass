"""Synthetic-only unit checks; no pretrained model or reserved inputs run."""

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import py_compile
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from looking_glass import agreement_preflight_v1 as study


class FakeTokenizer:
    def __init__(self):
        self.subjects = {subject: 10 + 4 * g + i
                         for g, (subjects, _, _) in enumerate(study.GROUPS)
                         for i, subject in enumerate(subjects)}
        self.contexts = {context: [100 + i, 200 + i]
                         for i, (_, context, _) in enumerate(study.GROUPS)}

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("no special tokens allowed")
        if text.startswith(" "):
            return list(self.contexts[text[1:]])
        if text == "The":
            return [1]
        target = None
        for suffix, token in study.TARGETS.items():
            if text.endswith(suffix):
                text, target = text[:-len(suffix)], token
                break
        parts = text.split(" ", 2)
        if len(parts) == 2:
            return [1, self.subjects[parts[1]]]
        _, subject, context = parts
        result = [1, self.subjects[subject], *self.contexts[context]]
        return result + ([target] if target is not None else [])


def fake_rows():
    return study.tokenize_inputs(FakeTokenizer())[0]


class AgreementPreflightTests(unittest.TestCase):
    def test_fixed_rows_and_alignment(self):
        rows, groups = study.tokenize_inputs(FakeTokenizer())
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0]["text"], "The cat near the windows")
        self.assertEqual(rows[10]["text"], "The writer beside the rivers")
        self.assertEqual(sum(row["subject_number"] == "singular" for row in rows), 8)
        self.assertEqual(sum(row["congruent"] for row in rows), 8)
        self.assertTrue(all(group["same_length"] and group["same_context_suffix"] for group in groups))
        self.assertTrue(all(row["attention_mask"] == [1] * row["length"] for row in rows))
        self.assertTrue(all(row["subject_index"] == 1 and row["subject_token_id"] == row["input_ids"][1]
                            for row in rows))

    def test_alignment_failure_prevents_manifest(self):
        class Broken(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if text.endswith(" are"):
                    ids[-1] = 123
                return ids

        with self.assertRaisesRegex(ValueError, "next-token alignment"):
            study.tokenize_inputs(Broken())

    def test_full_text_context_mismatch_is_rejected(self):
        class Split(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if "writers " in text:
                    ids.insert(2, 88)
                return ids

        with self.assertRaisesRegex(ValueError, "subject and context token alignment"):
            study.tokenize_inputs(Split())

    def test_multi_token_subject_is_rejected(self):
        class Split(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if text == "The writers":
                    ids.append(88)
                return ids

        with self.assertRaisesRegex(ValueError, "exactly one token"):
            study.tokenize_inputs(Split())

    def test_threshold_applies_every_prespecified_stratum(self):
        rows = fake_rows()
        for row in rows:
            row["margin"] = float(row["grammatical_sign"])
        summary = study.summarize(rows)
        self.assertTrue(summary["informative"])
        self.assertEqual(summary["counts"]["overall"], {"correct": 16, "total": 16})
        for row in [row for row in rows if row["subject_number"] == "singular"][:4]:
            row["margin"] = -1.0
        summary = study.summarize(rows)
        self.assertEqual(summary["counts"]["overall"]["correct"], 12)
        self.assertEqual(summary["counts"]["singular"]["correct"], 4)
        self.assertFalse(summary["informative"])

    def test_zero_margin_is_incorrect(self):
        rows = fake_rows()
        for row in rows:
            row["margin"] = float(row["grammatical_sign"])
        rows[0]["margin"] = 0.0
        self.assertEqual(study.summarize(rows)["counts"]["overall"]["correct"], 15)

    def test_raw_float32_targets_are_decoded_without_losing_bits(self):
        raw = bytearray(study.VOCAB_SIZE * 4)
        values = {" is": 1.25, " are": -2.5, " was": -0.0, " were": 0.125}
        for name, value in values.items():
            struct.pack_into("<f", raw, study.TARGETS[name] * 4, value)
        scores = study.extract_targets(bytes(raw))
        for name, value in values.items():
            self.assertEqual(scores[name]["value"], value)
            self.assertEqual(scores[name]["bits_le"], struct.pack("<f", value).hex())

    def test_subnormal_and_signed_zero_decode_under_float32_flush_modes(self):
        code = '''
import math, torch
from looking_glass import agreement_preflight_v1 as study
for enabled in (False, True):
    if not torch.set_flush_denormal(enabled):
        continue
    raw = bytearray(study.VOCAB_SIZE * 4)
    for name, bits in {" is": 1, " are": 0x80000001, " was": 0, " were": 0x80000000}.items():
        token = study.TARGETS[name]
        raw[token * 4:token * 4 + 4] = bits.to_bytes(4, "little")
    score = study.extract_targets(bytes(raw))
    assert score[" is"]["value"] == math.ldexp(1, -149)
    assert score[" are"]["value"] == -math.ldexp(1, -149)
    assert math.copysign(1, score[" was"]["value"]) == 1
    assert math.copysign(1, score[" were"]["value"]) == -1
    assert study._prepare_denormals(torch) == {"subnormal_input_bits": 1, "subnormal_output_bits": 0x00400000}
'''
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "no-bytecode"
            result = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={prefix}",
                                     "-c", code], cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_nonfinite_target_bits_are_rejected(self):
        raw = bytearray(study.VOCAB_SIZE * 4)
        token = study.TARGETS[" is"]
        raw[token * 4:token * 4 + 4] = (0x7f800000).to_bytes(4, "little")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            study.extract_targets(bytes(raw))

    def test_canonical_comparison_distinguishes_json_number_and_boolean(self):
        self.assertEqual({"token": True}, {"token": 1})
        self.assertNotEqual(study._canonical_json({"token": True}), study._canonical_json({"token": 1}))

    def test_source_first_bootstrap_rejects_ordinary_imports_and_bypasses_stale_pyc(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "run_agreement_preflight_v1.py"
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "no-bytecode"
            failed = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("pycache_prefix", failed.stderr)
            command = [sys.executable, "-B", "-X", f"pycache_prefix={prefix}"]
            accepted = subprocess.run([*command, str(script), "--help"], capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            source = Path(tmp) / "stale_probe.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            py_compile.compile(str(source), doraise=True)
            stat = source.stat()
            source.write_text("VALUE = 2\n", encoding="utf-8")
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            probe = "import stale_probe; print(stale_probe.VALUE)"
            stale = subprocess.run([sys.executable, "-B", "-c", probe], cwd=tmp,
                                   capture_output=True, text=True, check=True)
            fresh = subprocess.run([*command, "-c", probe], cwd=tmp,
                                   capture_output=True, text=True, check=True)
            self.assertEqual(stale.stdout.strip(), "1")
            self.assertEqual(fresh.stdout.strip(), "2")
            self.assertFalse(prefix.exists())

    def test_supervisor_records_child_failure_without_model_execution(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "supervise_agreement_preflight_v1.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            process = subprocess.run([sys.executable, str(script), "--cache", str(root),
                                      "--manifest", str(root / "nonexistent.json"),
                                      "--inventory-sha256", "0" * 64,
                                      "--pycache-prefix", str(root / "no-pyc"),
                                      "--output", str(out)], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            record = json.loads((out / "supervisor.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertNotEqual(record["exit_code"], 0)
            self.assertEqual(record["stderr_sha256"], hashlib.sha256((out / "stderr.txt").read_bytes()).hexdigest())
            self.assertFalse((out / "run").exists())

    def test_inventory_requires_planned_exact_rows_and_identity(self):
        rows = fake_rows()
        identity = {"model_id": study.MODEL_ID, "model_revision": study.REVISION,
                    "weights_sha256": study.ASSETS["model.safetensors"][1],
                    "tokenizer_id": study.MODEL_ID, "tokenizer_revision": study.REVISION,
                    "tokenizer_sha256": study.ASSETS["tokenizer.json"][1]}
        payload = {"identity": identity,
                   "records": [{"category": "planned", "token_ids": row["input_ids"],
                                "public_text": row["text"]} for row in rows]}
        study._verify_inventory(payload, rows)
        changed = copy.deepcopy(payload)
        changed["records"][0]["category"] = "observed-saved"
        with self.assertRaisesRegex(ValueError, "absent"):
            study._verify_inventory(changed, rows)
        changed = copy.deepcopy(payload)
        changed["identity"]["tokenizer_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            study._verify_inventory(changed, rows)

    def test_state_digest_detects_parameter_mutation(self):
        model = nn.Linear(2, 2)
        before = study._state_digest(model)
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(before, study._state_digest(model))

    def _run_fake(self, fail_at=None, fail_logging=False):
        """Exercise archive/attempt handling with fake token IDs and random-free weights."""
        rows, groups = study.tokenize_inputs(FakeTokenizer())
        manifest = {"rows": rows, "groups": groups, "source_revision": "synthetic"}
        identity = {"model_id": study.MODEL_ID, "model_revision": study.REVISION,
                    "weights_sha256": study.ASSETS["model.safetensors"][1],
                    "tokenizer_id": study.MODEL_ID, "tokenizer_revision": study.REVISION,
                    "tokenizer_sha256": study.ASSETS["tokenizer.json"][1]}
        payload = {"identity": identity, "records": [
            {"category": "planned", "token_ids": row["input_ids"], "public_text": row["text"]}
            for row in rows]}
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False).encode("utf-8")
        inventory_sha = hashlib.sha256(payload_bytes).hexdigest()

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(()))
                self.calls = 0

            def forward(self, input_ids, attention_mask):
                call = self.calls
                self.calls += 1
                if call == fail_at:
                    raise ValueError("simulated forward failure")
                logits = torch.zeros((1, input_ids.shape[1], study.VOCAB_SIZE))
                logits[:, :, study.TARGETS[" is"]] = 1.25
                logits[:, :, study.TARGETS[" are"]] = -0.5
                logits[:, :, study.TARGETS[" was"]] = 0.75
                return SimpleNamespace(logits=logits, past_key_values=None)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            out = Path(tmp) / "result"
            model = Toy().eval()
            bundle = SimpleNamespace(model=model, metadata={"model": "fake"})
            patches = [
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}),
                mock.patch.object(study, "_require_source_first"),
                mock.patch.object(study, "_verify_manifest", return_value=(manifest, "manifest", "commit")),
                mock.patch.object(study, "_tokenizer", return_value=FakeTokenizer()),
                mock.patch.object(study, "load_distilgpt2", return_value=bundle),
                mock.patch.object(study, "_prepare_denormals", return_value={"subnormal_input_bits": 1, "subnormal_output_bits": 0x00400000}),
                mock.patch.object(study, "_probe_denormals", return_value={"subnormal_input_bits": 1, "subnormal_output_bits": 0x00400000}),
                mock.patch("looking_glass.resident_worker.registered_model_state", return_value="stable"),
                mock.patch("torch.set_num_interop_threads"),
                mock.patch("torch.use_deterministic_algorithms"),
            ]
            with ExitStack() as stack:
                stack.callback(torch.set_num_threads, torch.get_num_threads())
                for patch in patches:
                    stack.enter_context(patch)
                if fail_logging:
                    original_event = study._event
                    def event(stream, payload):
                        if payload["event"] == "failed":
                            raise OSError("simulated log failure")
                        original_event(stream, payload)
                    stack.enter_context(mock.patch.object(study, "_event", side_effect=event))
                with torch.random.fork_rng(devices=[]):
                    try:
                        result = study.run(root, root, root / "manifest.json", payload,
                                           inventory_sha, "inventory", "source", out)
                    except ValueError as error:
                        if fail_at is None:
                            raise
                        if fail_logging:
                            self.assertIsInstance(error.__cause__, OSError)
                        result = None
            events = [json.loads(line) for line in (out / "attempt.jsonl").read_text().splitlines()]
            archive = (out / "last_logits.f32").read_bytes()
            self.assertEqual(model.calls, 16 if fail_at is None else fail_at + 1)
            self.assertEqual(len(archive), (16 if fail_at is None else fail_at) * study.VOCAB_SIZE * 4)
            if fail_at is None:
                self.assertEqual(events[-1]["event"], "complete")
                self.assertEqual(len(result["rows"]), 16)
                self.assertEqual(result["rows"][0]["margin"], 1.75)
                self.assertEqual(result["rows"][0]["past_margin"], 0.75)
                self.assertEqual(hashlib.sha256(archive).hexdigest(), result["archive_sha256"])
                self.assertTrue((out / "result.json").is_file())
            else:
                last_attempt = events[-1] if fail_logging else events[-2]
                self.assertEqual(last_attempt["event"], "forward_started")
                self.assertEqual(last_attempt["index"], fail_at)
                if not fail_logging:
                    self.assertEqual(events[-1]["event"], "failed")
                self.assertFalse((out / "result.json").exists())

    def test_fake_end_to_end_saves_complete_bounded_archive(self):
        self._run_fake()

    def test_fake_forward_failure_retains_attempt_and_partial_archive(self):
        self._run_fake(fail_at=3)

    def test_log_failure_does_not_mask_original_forward_error(self):
        self._run_fake(fail_at=3, fail_logging=True)
