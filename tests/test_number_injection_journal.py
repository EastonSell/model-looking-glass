"""Synthetic CPU checks for the private discovery-pair journal."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
from looking_glass.pretrained_number_adapter import to_selector_observation
from looking_glass.pretrained_number_archive import _CONDITIONS, _ROLES, serialize_discovery_observation
from looking_glass.pretrained_number_commitments import derive_site_layer, seal_site
from looking_glass.pretrained_number_injection import NumberInjectionExecutor
from looking_glass.pretrained_number_journal import DiscoveryJournal, inspect_discovery_journal
from looking_glass.pretrained_number_requests import create_family_request_plan, create_request_plan
from looking_glass.pretrained_number_witness import encode_private_forward
from tests.test_pretrained_number_injection import ToyModel


SOURCE = "a" * 40
RUNTIME = "b" * 64
PLAN = create_family_request_plan(bytes(range(32)), SOURCE)
PLAN_HASH = hashlib.sha256(PLAN).hexdigest()
NONCE = bytes(range(32, 64))  # Fabricated; never a study secret.
SITE = seal_site(NONCE, SOURCE)


def make_pairs(count):
    requests = json.loads(PLAN)["requests"]
    pairs = []
    with NumberInjectionExecutor(ToyModel(), site=derive_site_layer(NONCE)) as executor:
        baselines = {}
        for ordinal in range(count):
            case, role = requests[ordinal]["case_index"], requests[ordinal]["role"]
            clean, injected = baselines.get(case, (None, None))
            ids = PUBLIC_DISCOVERY[(case // 2) * 4 + case % 2][1]
            kwargs = {}
            if role.startswith("forward_"):
                layer = int(role[-1])
                kwargs = {"layer": layer, "donor": clean.mlp_donor(layer)}
            elif role.startswith("reverse_"):
                layer = int(role[-1])
                kwargs = {"layer": layer, "donor": injected.mlp_donor(layer)}
            elif role == "upstream":
                kwargs = {"donor": clean.mlp_donor(0)}
            elif role == "downstream":
                kwargs = {"donor": clean.block_donor()}
            result = executor.forward(ids, request_id=requests[ordinal]["request_id"],
                                      condition=_CONDITIONS[role], **kwargs)
            if role == "clean":
                baselines[case] = (result, None)
            elif role == "injected":
                baselines[case] = (clean, result)
            elif role == "verify_clean":
                del baselines[case]
            public = to_selector_observation(result)
            proof = tuple(e for e in executor.private_evidence() if e.request_id == public.request_id)
            pairs.append((serialize_discovery_observation(case, role, public, execution_device="cpu"),
                          encode_private_forward(case, role, public, proof)))
    return pairs


class DiscoveryJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.pairs = make_pairs(32)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "attempt"
        self.binding = {"plan_bytes": PLAN, "expected_source_commit": SOURCE,
                        "expected_plan_sha256": PLAN_HASH, "runtime_sha256": RUNTIME,
                        "site_nonce": NONCE, "site_record_bytes": SITE.record_bytes,
                        "site_commitment": SITE.commitment_sha256}
        self.journal = DiscoveryJournal(self.path, **self.binding, max_disk_bytes=30_000_000,
                                        max_retained_payload_bytes=25_000_000)

    def inspect(self, acks):
        return inspect_discovery_journal(self.path, **self.binding, acknowledgments=tuple(acks),
                                         max_read_bytes=30_000_000, externally_quiescent=True)

    def test_ordered_pairs_are_reaudited_and_externally_acknowledged(self):
        acks = [self.journal.append(*pair) for pair in self.pairs[:4]]
        result = self.inspect(acks)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(len(result.verified_pairs), 4)
        self.assertEqual(tuple((x.public, x.private) for x in result.verified_pairs), tuple(self.pairs[:4]))
        self.assertEqual(result.tail, "empty")
        self.assertEqual(self.inspect(acks[:-1]).tail, "unacknowledged")

    def test_bad_private_evidence_is_staged_before_terminal_failure(self):
        acks = [self.journal.append(*pair) for pair in self.pairs[:3]]
        public, private = self.pairs[3]
        changed = json.loads(private)
        changed["request_id"] = "0" * 32
        raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaises(ValueError):
            self.journal.append(public, raw)
        with self.assertRaises(RuntimeError):
            self.journal.append(*self.pairs[3])
        scan = self.inspect(acks)
        self.assertEqual((scan.status, scan.tail), ("incomplete", "unacknowledged"))
        self.assertEqual(len(scan.verified_pairs), 3)
        self.assertEqual((scan.tail_pair.public, scan.tail_pair.private), (public, raw))
        self.assertTrue((self.path / "terminal.json").is_file())

    def test_wrong_order_or_external_binding_is_terminal(self):
        with self.assertRaises(ValueError):
            self.journal.append(*self.pairs[1])
        self.assertEqual(self.inspect(()).tail, "unacknowledged")
        wrong = dict(self.binding, expected_plan_sha256="c" * 64)
        with self.assertRaises(ValueError):
            inspect_discovery_journal(self.path, **wrong, acknowledgments=(), max_read_bytes=10_000_000,
                                      externally_quiescent=True)

    def test_fsync_failure_keeps_complete_looking_tail_unacknowledged(self):
        real_fsync = os.fsync
        calls = 0
        def fail_first(fd):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected fsync failure")
            return real_fsync(fd)
        with patch("looking_glass.pretrained_number_journal.os.fsync", side_effect=fail_first):
            with self.assertRaises(OSError):
                self.journal.append(*self.pairs[0])
        scan = self.inspect(())
        self.assertEqual((scan.status, scan.tail), ("incomplete", "unacknowledged"))
        self.assertEqual((scan.tail_pair.public, scan.tail_pair.private), self.pairs[0])

    def test_ack_fsync_failure_requires_external_receipt(self):
        real_fsync = os.fsync
        calls = 0
        def fail_second(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected acknowledgment fsync failure")
            return real_fsync(fd)
        with patch("looking_glass.pretrained_number_journal.os.fsync", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.journal.append(*self.pairs[0])
        self.assertTrue((self.path / "000.ok").is_file())
        self.assertEqual(self.inspect(()).tail, "unacknowledged")

    def test_short_write_progress_and_stalled_write(self):
        real_write = os.write
        def short(fd, data):
            return real_write(fd, data[:min(4096, len(data))])
        with patch("looking_glass.pretrained_number_journal.os.write", side_effect=short):
            ack = self.journal.append(*self.pairs[0])
        self.assertEqual(len(self.inspect([ack]).verified_pairs), 1)
        attempts = 0
        def stall_once(fd, data):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return 0
            return real_write(fd, data)
        with patch("looking_glass.pretrained_number_journal.os.write", side_effect=stall_once):
            with self.assertRaises(OSError):
                self.journal.append(*self.pairs[1])
        self.assertEqual((len(self.inspect([ack]).verified_pairs), self.inspect([ack]).tail), (1, "invalid"))

    def test_two_case_baselines_survive_until_both_family_final_checks(self):
        acks = [self.journal.append(*pair) for pair in self.pairs]
        scan = self.inspect(acks)
        self.assertEqual((scan.status, scan.tail, len(scan.verified_pairs)), ("incomplete", "empty", 32))

    def test_early_family_clean_check_and_v1_plan_are_rejected(self):
        acks = [self.journal.append(*pair) for pair in self.pairs[:15]]
        with self.assertRaises(ValueError):
            self.journal.append(*self.pairs[30])
        self.assertEqual((len(self.inspect(acks).verified_pairs), self.inspect(acks).tail), (15, "unacknowledged"))
        old = create_request_plan(bytes(range(32)), SOURCE)
        wrong = dict(self.binding, plan_bytes=old, expected_plan_sha256=hashlib.sha256(old).hexdigest())
        with self.assertRaises(ValueError):
            DiscoveryJournal(Path(self.temp.name) / "v1", **wrong, max_disk_bytes=10_000_000,
                             max_retained_payload_bytes=8_000_000)

    def test_equal_valued_wrong_request_donor_is_rejected(self):
        acks = [self.journal.append(*pair) for pair in self.pairs[:3]]
        clean = json.loads(self.pairs[0][0])
        repeat = json.loads(self.pairs[1][0])
        self.assertEqual(clean["payload"], repeat["payload"])
        public, private = map(json.loads, self.pairs[3])
        public["action"]["donor"]["request_id"] = repeat["request_id"]
        patch_action = next(item for item in private["actions"] if item["operation"] == "patch")
        patch_action["donor_identity"]["request_id"] = repeat["request_id"]
        encode = lambda obj: json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaises(ValueError):
            self.journal.append(encode(public), encode(private))
        self.assertEqual(self.inspect(acks).tail, "unacknowledged")

    def test_abort_retains_bounded_opaque_partial_evidence(self):
        ack = self.journal.append(*self.pairs[0])
        payload = b"synthetic private partial evidence"
        self.journal.abort(phase="forward", partial_evidence=payload)
        result = self.inspect([ack])
        self.assertEqual((result.status, result.tail, result.partial_evidence), ("incomplete", "partial", payload))
        self.assertTrue(result.terminal_present)
        self.assertTrue(self.journal.reservation().failed)
        with self.assertRaises(RuntimeError):
            self.journal.append(*self.pairs[1])

    def test_bad_payload_budget_records_failure_without_writing_a_frame(self):
        tiny_path = Path(self.temp.name) / "tiny"
        journal = DiscoveryJournal(tiny_path, **self.binding, max_disk_bytes=10_000_000,
                                   max_retained_payload_bytes=1)
        with self.assertRaises(ValueError):
            journal.append(*self.pairs[0])
        self.assertFalse((tiny_path / "000.frame").exists())
        self.assertTrue((tiny_path / "terminal.json").is_file())
        self.assertEqual(journal.reservation().payload_bytes, 0)

    def test_relative_attempt_path_is_rejected(self):
        with self.assertRaises(ValueError):
            DiscoveryJournal(Path("relative-attempt"), **self.binding, max_disk_bytes=10_000_000,
                             max_retained_payload_bytes=1_000_000)

    def test_site_nonce_cannot_equal_published_request_entropy(self):
        public_entropy = bytes(range(32))
        public_site = seal_site(public_entropy, SOURCE)
        bad = dict(self.binding, site_nonce=public_entropy, site_record_bytes=public_site.record_bytes,
                   site_commitment=public_site.commitment_sha256)
        target = Path(self.temp.name) / "revealed-site"
        with self.assertRaises(ValueError):
            DiscoveryJournal(target, **bad, max_disk_bytes=10_000_000, max_retained_payload_bytes=8_000_000)
        self.assertFalse(target.exists())

    def test_fresh_cancellation_during_failure_recording_wins(self):
        def fail(path, parts):
            if str(path).endswith(".frame"):
                raise ValueError("original private error")
            raise KeyboardInterrupt()
        with patch("looking_glass.pretrained_number_journal._write_parts", side_effect=fail):
            with self.assertRaises(KeyboardInterrupt) as raised:
                self.journal.append(*self.pairs[0])
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertTrue(self.journal.reservation().failed)

    def test_active_cancellation_is_preserved_when_terminal_write_fails(self):
        def fail(path, parts):
            if str(path).endswith(".frame"):
                raise SystemExit(73)
            raise OSError("terminal unavailable")
        with patch("looking_glass.pretrained_number_journal._write_parts", side_effect=fail):
            with self.assertRaises(SystemExit) as raised:
                self.journal.append(*self.pairs[0])
        self.assertEqual(raised.exception.code, 73)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertTrue(self.journal.reservation().failed)

    def test_owner_thread_identity_and_unexpected_inventory(self):
        errors = []
        def foreign():
            try:
                self.journal.append(*self.pairs[0])
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        ack = self.journal.append(*self.pairs[0])
        (self.path / "128.frame").write_bytes(b"unexpected")
        self.assertEqual(self.inspect([ack]).tail, "invalid")

    def test_tampered_acknowledged_frame_does_not_validate(self):
        ack = self.journal.append(*self.pairs[0])
        frame_path = self.path / "000.frame"
        raw = bytearray(frame_path.read_bytes())
        raw[-10] ^= 1
        frame_path.write_bytes(raw)
        result = self.inspect([ack])
        self.assertEqual((result.status, result.tail, len(result.verified_pairs)), ("incomplete", "invalid", 0))

    def test_read_budget_and_external_quiescence_are_enforced(self):
        ack = self.journal.append(*self.pairs[0])
        result = inspect_discovery_journal(self.path, **self.binding, acknowledgments=(ack,),
                                           max_read_bytes=5, externally_quiescent=True)
        self.assertEqual(result.tail, "capacity")
        with self.assertRaises(ValueError):
            inspect_discovery_journal(self.path, **self.binding, acknowledgments=(ack,),
                                      max_read_bytes=10_000_000, externally_quiescent=False)

    def test_complete_128_pair_schedule(self):
        # Fabricated source, nonce, toy model and public discovery inputs only.
        pairs = make_pairs(128)
        complete_path = Path(self.temp.name) / "complete"
        journal = DiscoveryJournal(complete_path, **self.binding, max_disk_bytes=80_000_000,
                                   max_retained_payload_bytes=60_000_000)
        acks = [journal.append(*pair) for pair in pairs]
        scan = inspect_discovery_journal(complete_path, **self.binding, acknowledgments=tuple(acks),
                                         max_read_bytes=80_000_000, externally_quiescent=True)
        self.assertEqual((scan.status, scan.tail, len(scan.verified_pairs)), ("complete-prefix", "empty", 128))
        (complete_path / "128.frame").write_bytes(b"unexpected")
        scan = inspect_discovery_journal(complete_path, **self.binding, acknowledgments=tuple(acks),
                                         max_read_bytes=80_000_000, externally_quiescent=True)
        self.assertEqual((scan.status, scan.tail), ("incomplete", "invalid"))


if __name__ == "__main__":
    unittest.main()
