"""No-model owner admission tests on fabricated public receipts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[1]
script = root / "src" / "scripts" / "france_baseline_cpu_owner_v2.py"
if not script.exists():
    script = root / "scripts" / "france_baseline_cpu_owner_v2.py"
spec = importlib.util.spec_from_file_location("_france_cpu_owner_test", script)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def public_report():
    return {"schema": "france-v4-public-exposure-independent-v2", "status": "no_collision_in_named_scope",
            "pins_sha256": {"france": subject.CASE_SHA256}, "study_rows": 27,
            "projection_basis_rows": 4, "study_sequences": 73, "unexempted_inventory_groups": 0,
            "additional_bundle_results": {"adapter": {"matches": [], "sequences": 80},
                                           "causal": {"matches": [], "sequences": 612}}}


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


class OwnerAdmissionTests(unittest.TestCase):
    def test_windows_drive_and_separator_escapes_are_rejected(self):
        for value in ("C:/outside/item.txt", "C:outside/item.txt", "x\\..\\file", "../elsewhere", "//host/share"):
            with self.assertRaises(subject.OwnerError):
                subject._source_relative(value)
        self.assertEqual(subject._source_relative("looking_glass/example.py").as_posix(),
                         "looking_glass/example.py")

    def test_checked_public_receipt_binds_case_scope_and_counts(self):
        payload = raw(public_report())
        observed = subject.check_public_report(payload, hashlib.sha256(payload).hexdigest())
        self.assertEqual(observed["status"], "no_collision_in_named_scope")

    def test_rehashed_wrong_types_collision_and_case_are_rejected(self):
        for mutate in (lambda x: x.__setitem__("unexempted_inventory_groups", True),
                       lambda x: x.__setitem__("unexempted_inventory_groups", 1),
                       lambda x: x["pins_sha256"].__setitem__("france", "0" * 64),
                       lambda x: x["additional_bundle_results"]["causal"]["matches"].append("bad")):
            report = public_report()
            mutate(report)
            payload = raw(report)
            with self.assertRaises(subject.OwnerError):
                subject.check_public_report(payload, hashlib.sha256(payload).hexdigest())

    def test_capacity_requires_c_and_e_and_physical_memory_margins(self):
        threshold = subject.CAPACITY
        self.assertTrue(subject.check_capacity_values(threshold["c_free"], threshold["e_free"], threshold["physical_free"]))
        for key in threshold:
            values = dict(threshold)
            values[key] -= 1
            with self.assertRaises(subject.OwnerError):
                subject.check_capacity_values(values["c_free"], values["e_free"], values["physical_free"])
        with self.assertRaises(subject.OwnerError):
            subject.check_capacity_values(True, threshold["e_free"], threshold["physical_free"])

    def test_tokenizer_checks_every_declared_boundary(self):
        class Tokenizer:
            def __call__(self, text, **_kwargs):
                return {"input_ids": [1, 2], "attention_mask": [1, 1]}

            def encode(self, text, **_kwargs):
                return {"prefix": [1, 2], " next": [3], "prefix next": [1, 2, 3]}[text]

        row = {"text": "prefix", "token_ids": [1, 2], "attention_mask": [1, 1],
               "continuations": [{"text": " next", "token_id": 3, "appended_token_ids": [1, 2, 3]}]}
        subject.verify_tokenizer_rows(Tokenizer(), [row])
        changed = {**row, "continuations": [{**row["continuations"][0], "appended_token_ids": [1, 2, 4]}]}
        with self.assertRaises(subject.OwnerError):
            subject.verify_tokenizer_rows(Tokenizer(), [changed])

    def test_result_acceptance_requires_reaped_zero_exit_and_matching_identity(self):
        class Result:
            pid = 23
            run_id = "a" * 32
            exit_code = 0
            clean = True
            timed_out = False
        class Job:
            _handle = None
            _confirmed = True
        child = {"status": "complete", "run_id": "a" * 32, "pid": 23,
                 "case_sha256": subject.CASE_SHA256, "source_sha256": "b" * 64,
                 "source_revision": "e" * 40, "mode": "baseline", "prefix_empty": True,
                 "model_released": True, "details": {"identity": {"run_id": "a" * 32,
                 "source_revision": "e" * 40, "case_sha256": subject.CASE_SHA256,
                 "source_sha256": "b" * 64}, "screen_sha256": "f" * 64, "journal_sha256": "a" * 64}}
        subject.check_result(Result(), Job(), child, "b" * 64, "baseline", "e" * 40)
        for key, value in (("pid", 24), ("run_id", "c" * 32), ("prefix_empty", False),
                           ("model_released", False), ("source_sha256", "d" * 64),
                           ("source_revision", "f" * 40)):
            forged = dict(child); forged[key] = value
            with self.assertRaises(subject.OwnerError):
                subject.check_result(Result(), Job(), forged, "b" * 64, "baseline", "e" * 40)
        result = Result(); result.exit_code = 1
        with self.assertRaises(subject.OwnerError):
            subject.check_result(result, Job(), child, "b" * 64, "baseline", "e" * 40)
        rehearsal = {**child, "mode": "rehearsal", "details": {"checkpoint_loaded": False, "model_forwarded": False}}
        subject.check_result(Result(), Job(), rehearsal, "b" * 64, "rehearsal", "e" * 40)
        for key in ("checkpoint_loaded", "model_forwarded"):
            forged = {**rehearsal, "details": {**rehearsal["details"], key: True}}
            with self.assertRaises(subject.OwnerError):
                subject.check_result(Result(), Job(), forged, "b" * 64, "rehearsal", "e" * 40)


if __name__ == "__main__":
    unittest.main()
