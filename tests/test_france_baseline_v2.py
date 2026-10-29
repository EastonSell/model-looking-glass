"""Synthetic-only durable-baseline tests; no pretrained model or study inputs."""
import json
import hashlib
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

_root = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_root / "src" if (_root / "src" / "looking_glass").is_dir() else _root))
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_v2_owner as custody
from looking_glass import targeted_edit_france_baseline_v2 as subject


class TinyModel(torch.nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("fixed", torch.arange(3))
        self.fault = fault
        self.calls = 0
        self.mutate = None

    def forward(self, *, input_ids, attention_mask, use_cache):
        index = self.calls
        self.calls += 1
        if self.mutate is not None and index == 0:
            self.mutate()
        if self.fault == "raise" and index == 3:
            raise RuntimeError("synthetic forward error")
        if self.fault == "interrupt" and index == 3:
            raise KeyboardInterrupt()
        width = 39 if self.fault == "bad_shape" and index == 3 else 40
        logits = torch.zeros((1, input_ids.shape[1], width), dtype=torch.float32)
        if self.fault == "nonfinite" and index == 3:
            logits[0, -1, 2] = float("nan")
        if index < 7:
            logits[0, -1, 1] = 4
        elif index < 11:
            logits[0, -1, 3] = 4
        elif index < 19:
            logits[0, -1, 0] = 4
        else:
            logits[0, -1, 4] = 4
        return SimpleNamespace(logits=logits)


def inputs():
    ids = ["fit", "calibration-1", "calibration-2"]
    ids += [f"T{i:02d}" for i in range(1, 9)]
    ids += [f"L{i:02d}" for i in range(1, 17)]
    rows = [{"case_id": name, "token_ids": [1, i + 2], "attention_mask": [1, 1]}
            for i, name in enumerate(ids)]
    locality = [{"case_id": f"L{i:02d}", "accepted_ids": [0]} for i in range(1, 17)]
    return rows, locality


def guard(fault=None):
    model = TinyModel(fault).eval()
    return custody._ModelGuard(model, model.proj, model.proj.weight, torch, v1)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_flush_denormal(False)

    def run_one(self, target, directory, **kwargs):
        rows, locality = inputs()
        return subject.run_rows(target, rows, locality, 1, 2, pathlib.Path(directory) / "screen",
                                vocabulary=40, identity={"fixture": True}, **kwargs)

    def test_complete_27_ordered_forwards_preserve_actual_entry_and_state(self):
        owned = guard()
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_one(owned, directory)
            output = pathlib.Path(directory) / "screen"
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 54)
            self.assertEqual([event["kind"] for event in events], ["forward_start", "forward_complete"] * 27)
            rows, _ = inputs()
            for index, row in enumerate(rows):
                start, complete = events[2 * index:2 * index + 2]
                self.assertEqual(start["case_id"], row["case_id"])
                self.assertEqual(complete["actual_input_ids"], row["token_ids"])
                self.assertEqual(complete["actual_attention_mask"], row["attention_mask"])
                self.assertEqual(complete["index"], index)
                raw = (output / "screen.f32").read_bytes()
                self.assertEqual(hashlib.sha256(raw[index * 160:(index + 1) * 160]).hexdigest(),
                                 complete["sha256"])
            self.assertEqual(result["baseline"]["status"], "eligible")
            self.assertEqual((result["baseline"]["E"], result["baseline"]["H"], result["baseline"]["H_new"]), (4, 8, 5))
            self.assertEqual((output / "screen.f32").stat().st_size, 27 * 40 * 4)
            self.assertTrue((output / "completed.json").is_file())
            self.assertFalse((output / "incomplete.json").exists())
            self.assertFalse(owned.poisoned)
            self.assertFalse(custody._has_hooks(owned.model, torch))
            owned.verify()

    def test_bad_return_retains_bounded_invalid_and_no_complete_for_that_call(self):
        owned = guard("bad_shape")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subject.BaselineError):
                self.run_one(owned, directory)
            output = pathlib.Path(directory) / "screen"
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual([entry["kind"] for entry in events[-2:]], ["forward_start", "forward_returned_invalid"])
            self.assertEqual(events[-1]["index"], 3)
            self.assertEqual((output / "invalid-0003.f32").stat().st_size, 39 * 4)
            self.assertTrue((output / "incomplete.json").is_file())
            self.assertFalse((output / "completed.json").exists())
            self.assertTrue(owned.poisoned)
            self.assertFalse(custody._has_hooks(owned.model, torch))

    def test_nonfinite_return_is_saved_as_invalid_and_never_scored(self):
        owned = guard("nonfinite")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subject.BaselineError):
                self.run_one(owned, directory)
            output = pathlib.Path(directory) / "screen"
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["kind"], "forward_returned_invalid")
            self.assertEqual(events[-1]["raw_last_row"]["bytes"], 160)
            self.assertFalse((output / "provisional.json").exists())
            self.assertTrue((output / "incomplete.json").exists())

    def test_copied_rows_and_labels_cannot_change_during_the_run(self):
        owned = guard()
        rows, locality = inputs()
        identity = {"fixture": True}
        owned.model.mutate = lambda: (rows[1]["token_ids"].__setitem__(0, 30),
                                      locality[0]["accepted_ids"].__setitem__(0, 4),
                                      identity.__setitem__("fixture", False))
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "screen"
            result = subject.run_rows(owned, rows, locality, 1, 2, output,
                                      vocabulary=40, identity=identity)
            self.assertEqual(result["baseline"]["H"], 8)
            self.assertEqual(result["identity"], {"fixture": True})
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[3]["actual_input_ids"], [1, 3])

    def test_forward_failure_and_interrupt_are_preserved_without_hook_leak(self):
        for fault, exception in (("raise", RuntimeError), ("interrupt", KeyboardInterrupt)):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                owned = guard(fault)
                with self.assertRaises(exception):
                    self.run_one(owned, directory)
                output = pathlib.Path(directory) / "screen"
                events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
                self.assertEqual([entry["kind"] for entry in events[-2:]], ["forward_start", "forward_failed"])
                self.assertTrue((output / "incomplete.json").is_file())
                self.assertTrue(owned.poisoned)
                self.assertFalse(custody._has_hooks(owned.model, torch))

    def test_post_return_write_failure_never_publishes_completion(self):
        owned = guard()
        append = custody._append
        def fail_complete(handle, data):
            if b'"kind":"forward_complete"' in data:
                raise OSError("synthetic event failure")
            return append(handle, data)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(custody, "_append", fail_complete):
            with self.assertRaises(OSError):
                self.run_one(owned, directory)
            output = pathlib.Path(directory) / "screen"
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual([entry["kind"] for entry in events], ["forward_start", "forward_returned_invalid"])
            self.assertEqual((output / "invalid-0000.f32").stat().st_size, 40 * 4)
            self.assertTrue((output / "incomplete.json").is_file())
            self.assertFalse((output / "completed.json").exists())
            self.assertTrue(owned.poisoned)

    def test_started_write_failure_retains_marker_and_never_forwards(self):
        owned = guard()
        original = custody._json_new
        def fail_start(path, value):
            if pathlib.Path(path).name == "started.json":
                pathlib.Path(path).write_bytes(b"partial")
                raise OSError("synthetic short start")
            return original(path, value)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(custody, "_json_new", fail_start):
            with self.assertRaises(OSError):
                self.run_one(owned, directory)
            output = pathlib.Path(directory) / "screen"
            self.assertEqual((output / "started.json").read_bytes(), b"partial")
            marker = json.loads((output / "incomplete.json").read_bytes())
            self.assertEqual(marker["phase"], "started_record")
            self.assertEqual(owned.model.calls, 0)
            self.assertTrue(owned.poisoned)

    def test_wrong_fixed_bytes_and_boolean_mask_are_rejected_before_model(self):
        owned = guard()
        rows, locality = inputs()
        rows[0]["attention_mask"][0] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subject.BaselineError):
                subject.run_rows(owned, rows, locality, 1, 2, pathlib.Path(directory) / "screen",
                                 vocabulary=40, identity={"fixture": True})
            with self.assertRaises(subject.BaselineError):
                subject.run_fixed(owned, b"{}", {}, pathlib.Path(directory) / "screen")
            self.assertEqual(owned.model.calls, 0)
            self.assertFalse((pathlib.Path(directory) / "screen").exists())


if __name__ == "__main__":
    unittest.main()
