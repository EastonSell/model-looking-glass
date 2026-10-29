"""Behavioral tests for matched-study exposure reservations; no model calls."""

import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from looking_glass.cpu_study_supervision import run_cpu_study
from looking_glass.supervisor import SupervisionResult

try:
    from looking_glass.bert_matched_owner import MatchedLedger
except ModuleNotFoundError as error:
    if error.name != "looking_glass.bert_matched_owner":
        raise
    MatchedLedger = None


def _binding():
    protocol = "b" * 64
    manifest = "c" * 64
    return {
        "source_commit": "a" * 40,
        "source_sha256": {
            "docs/BERT_MATCHED_MAGNITUDE_V1.md": protocol,
            "experiments/bert_matched_public_cases_v1.json": manifest,
            "looking_glass/bert_matched_owner.py": "e" * 64,
        },
        "protocol_sha256": protocol,
        "manifest_sha256": manifest,
        "prior_exposure_sha256": "d" * 64,
        "model": {"model": "distilbert/distilbert-base-uncased",
                  "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
                  "assets": {name: {"size": 1, "sha256": "f" * 64} for name in
                             ("config.json", "tokenizer_config.json", "vocab.txt",
                              "tokenizer.json", "model.safetensors")}},
        "rows": [{"row_id": f"evaluation-matched-{i:02d}",
                  "input_ids": [101, 1996, 2000 + i, 3875, 1996, 3000, 103, 3201, 1012, 102],
                  "attention_mask": [1] * 10} for i in range(64)],
    }


class MatchedLedgerTests(unittest.TestCase):
    def test_source_inventory_rejects_windows_drive_escape(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            binding = _binding()
            binding["source_sha256"]["C:/outside.py"] = "1" * 64
            try:
                ledger = MatchedLedger.acquire(ledger_root, custody_root, binding)
            except ValueError:
                pass
            else:
                ledger.release()
                self.fail("a Windows drive source path was accepted")

    def test_source_inventory_rejects_case_insensitive_path_aliases(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            binding = _binding()
            binding["source_sha256"]["looking_glass/Example.py"] = "1" * 64
            binding["source_sha256"]["looking_glass/example.py"] = "2" * 64
            try:
                ledger = MatchedLedger.acquire(ledger_root, custody_root, binding)
            except ValueError:
                pass
            else:
                ledger.release()
                self.fail("ambiguous source paths were accepted")

    def test_identity_cannot_be_spoofed_by_a_string_subclass(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")

        class Spoof(str):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            binding = _binding()
            binding["model"]["model"] = Spoof("not-the-pinned-model")
            try:
                ledger = MatchedLedger.acquire(ledger_root, custody_root, binding)
            except ValueError:
                pass
            else:
                ledger.release()
                self.fail("a spoofed model identity was accepted")

    def test_input_mutation_during_validation_cannot_enter_the_reservation(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            binding = _binding()
            changed = []
            previous = sys.gettrace()

            def trace(frame, event, arg):
                if (not changed and event == "line" and frame.f_code.co_name == "_binding"
                        and "row" in frame.f_locals and "inputs" in frame.f_locals):
                    row, seen = frame.f_locals["row"], frame.f_locals["inputs"]
                    if type(row) is dict and tuple(row["input_ids"]) in seen:
                        binding["rows"][0]["input_ids"][0] = -7
                        changed.append(True)
                return trace

            try:
                sys.settrace(trace)
                ledger = MatchedLedger.acquire(ledger_root, custody_root, binding)
            finally:
                sys.settrace(previous)
            try:
                self.assertTrue(changed, "test did not exercise the post-validation mutation window")
                path = ledger.reserve("0" * 32)
                record = json.loads((path / "reservation.json").read_bytes())
                self.assertEqual(record["binding"]["rows"][0]["input_ids"][0], 101)
            finally:
                ledger.release()

    def test_reservation_records_the_whole_workload_before_any_owner(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            binding = _binding()
            ledger = MatchedLedger.acquire(ledger_root, custody_root, binding)
            try:
                path = ledger.reserve("1" * 32)
                record = json.loads((path / "reservation.json").read_bytes())
                self.assertEqual(record["binding"], binding)
                self.assertEqual(len(record["binding"]["rows"]), 64)
                self.assertEqual(record["prior_attempts"], [])
                self.assertFalse(record["prior_may_have_exposed"])
                self.assertFalse((path / "owner-started.json").exists())
            finally:
                ledger.release()

    def test_public_binding_mutation_cannot_change_the_reserved_workload(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            original = _binding()
            ledger = MatchedLedger.acquire(ledger_root, custody_root, original)
            try:
                original["rows"][0]["input_ids"][2] = 123
                ledger.binding["rows"][1]["input_ids"][2] = 124
                path = ledger.reserve("2" * 32)
                record = json.loads((path / "reservation.json").read_bytes())
                self.assertEqual(record["binding"], _binding())
            finally:
                ledger.release()

    def test_unknown_prior_attempt_and_sticky_custody_block_new_reservation(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            first = MatchedLedger.acquire(ledger_root, custody_root, _binding())
            try:
                first.reserve("3" * 32)
            finally:
                first.release()
            second = MatchedLedger.acquire(ledger_root, custody_root, _binding())
            try:
                with self.assertRaises(RuntimeError):
                    second.reserve("4" * 32)
                self.assertFalse((ledger_root / ("4" * 32)).exists())
            finally:
                second.release()
            fresh_root = root / "fresh-ledger"
            fresh_root.mkdir()
            (custody_root / "cpu-study.unsafe.json").write_text("unknown prior owner")
            third = MatchedLedger.acquire(fresh_root, custody_root, _binding())
            try:
                with self.assertRaises(RuntimeError):
                    third.reserve("5" * 32)
                self.assertFalse((fresh_root / ("5" * 32)).exists())
            finally:
                third.release()

    def test_orphan_attestation_blocks_reservation_before_exposure(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            (custody_root / "cpu-study.unsafe.json.attestation").write_text("orphan")
            ledger = MatchedLedger.acquire(ledger_root, custody_root, _binding())
            try:
                with self.assertRaises(RuntimeError):
                    ledger.reserve("9" * 32)
                self.assertFalse((ledger_root / ("9" * 32)).exists())
            finally:
                ledger.release()

    def test_reaped_incomplete_attempt_remains_exposed_on_a_later_reservation(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            first = MatchedLedger.acquire(ledger_root, custody_root, _binding())
            try:
                first_path = first.reserve("6" * 32)
                (custody_root / "cpu-study.unsafe.json").write_text("unclean owner")
                observation = SupervisionResult("a" * 32, 1234, 7, False, False, "owner failed")
                first.record_incomplete(observation)
                self.assertTrue((first_path / "result.json").is_file())
                recorded = json.loads((first_path / "result.json").read_bytes())
                self.assertEqual(recorded["supervision"]["scope"], "cooperative-cpu-study-custody-only")
            finally:
                first.release()
            # Only the test's temporary, externally reconciled marker is cleared.
            (custody_root / "cpu-study.unsafe.json").unlink()
            second = MatchedLedger.acquire(ledger_root, custody_root, _binding())
            try:
                path = second.reserve("7" * 32)
                record = json.loads((path / "reservation.json").read_bytes())
                self.assertTrue(record["prior_may_have_exposed"])
                self.assertEqual([item["attempt_id"] for item in record["prior_attempts"]], ["6" * 32])
            finally:
                second.release()

    def test_supervised_child_claims_the_exact_reserved_workload(self):
        self.assertIsNotNone(MatchedLedger, "matched-study ledger is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root = root / "ledger", root / "custody"
            ledger_root.mkdir()
            custody_root.mkdir()
            ledger = MatchedLedger.acquire(ledger_root, custody_root, _binding())
            try:
                path = ledger.reserve("8" * 32)
                import hashlib
                reservation_sha = hashlib.sha256((path / "reservation.json").read_bytes()).hexdigest()
                expected = root / "expected.json"
                expected.write_text(json.dumps(_binding()), encoding="utf-8")
                verified = root / "verified"
                code = textwrap.dedent("""
                    import json, sys
                    from pathlib import Path
                    from looking_glass.cpu_study_supervision import claim_cpu_study
                    session, lease = claim_cpu_study(Path(sys.argv[2]))
                    from looking_glass.bert_matched_owner import verify_owner_claim
                    binding = json.loads(Path(sys.argv[5]).read_text(encoding='utf-8'))
                    verify_owner_claim(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
                                       sys.argv[4], binding, session=session, lease=lease)
                    Path(sys.argv[6]).touch()
                    session.attest_clean(lease)
                """)
                env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "TEMP", "TMP")
                       if key in os.environ}
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
                observed = run_cpu_study([sys._base_executable, "-c", code, str(ledger_root), str(custody_root),
                                          "8" * 32, reservation_sha, str(expected), str(verified)],
                                         study_root=custody_root, timeout=10, cwd=root, env=env)
                self.assertEqual(observed.exit_code, 0)
                self.assertTrue(observed.clean)
                self.assertTrue(verified.is_file())
            finally:
                ledger.release()


if __name__ == "__main__":
    unittest.main()
