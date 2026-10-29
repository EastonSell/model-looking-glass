"""No-checkpoint process/ledger integration for the matched CPU controller."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from test_bert_matched_owner import _binding
from looking_glass.bert_matched_owner import MatchedLedger
from looking_glass.supervisor import SupervisionResult

try:
    from looking_glass.bert_matched_controller import run_rehearsal
except ModuleNotFoundError as error:
    if error.name != "looking_glass.bert_matched_controller":
        raise
    run_rehearsal = None


class MatchedControllerTests(unittest.TestCase):
    def test_bad_or_ambiguous_child_cannot_promote_or_retry(self):
        for exit_code, clean in ((1, True), (False, True), (0, False)):
            with self.subTest(exit_code=exit_code, clean=clean), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                ledger_root, custody_root, work = root / 'ledger', root / 'custody', root / 'work'
                for path in (ledger_root, custody_root, work):
                    path.mkdir()
                ledger = MatchedLedger.acquire(ledger_root, custody_root, _binding())
                observed = SupervisionResult('b' * 32, 123, exit_code, clean, False, 'fixture')
                try:
                    with patch('looking_glass.bert_matched_controller.run_cpu_study', return_value=observed):
                        with self.assertRaises(RuntimeError):
                            run_rehearsal(ledger, 'a' * 32, command=lambda *_: ['fixture'],
                                          cwd=work, env={}, timeout=1)
                    attempt = ledger_root / ('a' * 32)
                    self.assertTrue((attempt / 'rehearsal-failure.json').is_file())
                    self.assertFalse((attempt / 'rehearsal-result.json').exists())
                finally:
                    ledger.release()
                again = MatchedLedger.acquire(ledger_root, custody_root, _binding())
                try:
                    with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                        again.reserve('c' * 32)
                finally:
                    again.release()

    def test_interrupt_after_reservation_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('ledger', 'custody', 'work'):
                (root / name).mkdir()
            ledger = MatchedLedger.acquire(root / 'ledger', root / 'custody', _binding())
            try:
                def interrupted(*_):
                    raise KeyboardInterrupt()
                with self.assertRaises(KeyboardInterrupt):
                    run_rehearsal(ledger, 'a' * 32, command=interrupted,
                                  cwd=root / 'work', env={}, timeout=1)
                self.assertTrue((root / 'ledger' / ('a' * 32) / 'rehearsal-failure.json').is_file())
            finally:
                ledger.release()

    def test_no_checkpoint_supervised_rehearsal_links_reservation_and_closed_journal(self):
        self.assertIsNotNone(run_rehearsal, "matched CPU rehearsal is not implemented")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_root, custody_root, work = root / "ledger", root / "custody", root / "work"
            for path in (ledger_root, custody_root, work):
                path.mkdir()
            expected = root / "expected.json"
            expected.write_text(json.dumps(_binding()), encoding="utf-8")
            ledger = MatchedLedger.acquire(ledger_root, custody_root, _binding())
            try:
                code = textwrap.dedent("""
                    import hashlib, json, os, sys
                    from pathlib import Path
                    from looking_glass.cpu_study_supervision import claim_cpu_study
                    session, lease = claim_cpu_study(Path(sys.argv[2]))
                    from looking_glass.bert_matched_owner import verify_owner_claim
                    from looking_glass.bert_evaluation_ledger import write_record
                    binding = json.loads(Path(sys.argv[5]).read_text(encoding='utf-8'))
                    verify_owner_claim(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
                                       sys.argv[4], binding, session=session, lease=lease)
                    destination = Path(sys.argv[1]) / sys.argv[3]
                    raw = b'{"kind":"fabricated-no-model"}\\n'
                    with (destination / 'events.jsonl').open('xb') as out:
                        out.write(raw); out.flush(); os.fsync(out.fileno())
                    write_record(destination / 'owner-candidate.json', {
                        'schema': 'bert-matched-rehearsal-candidate-v1',
                        'attempt_id': sys.argv[3], 'pid': os.getpid(),
                        'reservation_sha256': sys.argv[4],
                        'journal_sha256': hashlib.sha256(raw).hexdigest(),
                        'forward_count': 0,
                    })
                    session.attest_clean(lease)
                """)
                env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "TEMP", "TMP")
                       if key in os.environ}
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

                def command(path, digest):
                    self.assertTrue((path / "reservation.json").is_file())
                    return [sys._base_executable, "-c", code, str(ledger_root), str(custody_root),
                            path.name, digest, str(expected)]

                outcome = run_rehearsal(ledger, "a" * 32, command=command, cwd=work, env=env, timeout=10)
                self.assertEqual(outcome["status"], "rehearsal_complete")
                self.assertEqual(outcome["forward_count"], 0)
                self.assertEqual(outcome["journal_sha256"], hashlib.sha256(b'{"kind":"fabricated-no-model"}\n').hexdigest())
                self.assertTrue(outcome["child_reaped"])
                self.assertTrue((ledger_root / ("a" * 32) / "rehearsal-result.json").is_file())
            finally:
                ledger.release()


if __name__ == "__main__":
    unittest.main()
