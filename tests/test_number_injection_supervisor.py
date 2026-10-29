"""CPU-only end-to-end tests of discovery evidence and its custody handoff."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import torch

from looking_glass.pretrained_number_commitments import seal_site
from looking_glass.pretrained_number_incomplete import decode_incomplete_forward
from looking_glass.pretrained_number_journal import DiscoveryJournal, PairReceipt, inspect_discovery_journal
from looking_glass.pretrained_number_injection import NumberInjectionExecutor
from looking_glass.pretrained_number_requests import create_family_request_plan
from tests.test_number_injection_coordinate_oracle import Model

try:
    from looking_glass.pretrained_number_supervisor import run_discovery
except ModuleNotFoundError as exc:
    if exc.name != 'looking_glass.pretrained_number_supervisor':
        raise
    run_discovery = None


class DiscoverySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        tiny = torch.tensor([1], dtype=torch.int32).view(torch.float32)
        previous = int((tiny * 1.0).view(torch.int32).item()) == 0
        if not torch.set_flush_denormal(False):
            self.skipTest('CPU denormal mode unavailable')
        self.addCleanup(torch.set_flush_denormal, previous)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.attempt = Path(self.temp.name).resolve() / 'attempt'
        self.source = 'a' * 40
        self.nonce = bytes(range(32, 64))  # Fabricated; not a study secret.
        self.site = seal_site(self.nonce, self.source)
        self.plan = create_family_request_plan(bytes(range(32)), self.source)

    def run_model(self, model, **changes):
        arguments = dict(model=model, attempt_dir=self.attempt, plan_bytes=self.plan,
                         source_commit=self.source, plan_sha256=hashlib.sha256(self.plan).hexdigest(),
                         runtime_sha256='b' * 64, site_nonce=self.nonce,
                         site_record_bytes=self.site.record_bytes,
                         site_commitment=self.site.commitment_sha256,
                         logical_limit_bytes=48 * 1024**2, disk_limit_bytes=128 * 1024**2)
        arguments.update(changes)
        return run_discovery(**arguments)

    def test_complete_run_preserves_receipts_and_replayable_public_observations(self):
        self.assertTrue(callable(run_discovery), 'discovery supervisor is missing')
        result = self.run_model(Model())
        self.assertEqual(result.status, 'complete-local')
        self.assertEqual(result.record_count, 128)
        receipts = (self.attempt / 'private' / 'receipts.jsonl').read_bytes().splitlines()
        self.assertEqual(len(receipts), 128)
        self.assertEqual([json.loads(row)['ordinal'] for row in receipts], list(range(128)))
        public = self.attempt / 'public'
        records = sorted(public.glob('observation-*.json'))
        self.assertEqual(len(records), 128)
        self.assertEqual(hashlib.sha256((public / 'prediction.json').read_bytes()).hexdigest(),
                         result.prediction_sha256)
        self.assertNotIn(self.nonce.hex().encode(), b''.join(p.read_bytes() for p in records))
        self.assertNotIn(b'inject_before', b''.join(p.read_bytes() for p in records))

    def test_insufficient_logical_budget_stops_before_a_forward(self):
        self.assertTrue(callable(run_discovery), 'discovery supervisor is missing')
        model = Model()
        def forbidden(*args, **kwargs):
            raise AssertionError('model forward must not run')
        model.forward = forbidden
        with self.assertRaises(ValueError):
            self.run_model(model, logical_limit_bytes=1)

    def test_receipt_failure_stops_before_the_next_forward(self):
        calls = []
        original = Model.forward
        def counted(model, *args, **kwargs):
            calls.append(1)
            return original(model, *args, **kwargs)
        with patch.object(Model, 'forward', new=counted), patch(
                'looking_glass.pretrained_number_supervisor._append_receipt', side_effect=OSError('private')):
            with self.assertRaises(OSError):
                self.run_model(Model())
        self.assertEqual(len(calls), 1)
        journal = self.attempt / 'private' / 'journal'
        self.assertTrue((journal / '000.frame').exists())
        self.assertFalse((journal / '001.frame').exists())
        self.assertFalse((journal / '001.partial').exists())
        self.assertEqual((self.attempt / 'private' / 'receipts.jsonl').read_bytes(), b'')
        failed = json.loads((self.attempt / 'private' / 'failed.json').read_bytes())
        self.assertEqual((failed['status'], failed['partial_evidence']), ('incomplete', 'retained'))
        sidecar = decode_incomplete_forward((self.attempt / 'private' / 'incomplete-forward.bin').read_bytes())
        scan = inspect_discovery_journal(journal, plan_bytes=self.plan, expected_source_commit=self.source,
            expected_plan_sha256=hashlib.sha256(self.plan).hexdigest(), runtime_sha256='b' * 64,
            site_nonce=self.nonce, site_record_bytes=self.site.record_bytes,
            site_commitment=self.site.commitment_sha256, acknowledgments=(),
            max_read_bytes=128 * 1024**2, externally_quiescent=True)
        self.assertEqual((scan.status, scan.tail, scan.terminal_present), ('incomplete', 'unacknowledged', True))
        self.assertEqual(sidecar.manifest['observation']['request_id'], json.loads(scan.tail_pair.public)['request_id'])
        self.assertEqual(sidecar.manifest['observation']['request_id'], json.loads(self.plan)['requests'][0]['request_id'])

    def test_interrupted_append_return_keeps_advanced_pair_as_unacknowledged(self):
        original = DiscoveryJournal.append
        def append_then_interrupt(journal, *args):
            original(journal, *args)
            raise KeyboardInterrupt('interrupt after journal advanced')
        with patch.object(DiscoveryJournal, 'append', new=append_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_model(Model())
        journal = self.attempt / 'private' / 'journal'
        sidecar = decode_incomplete_forward((self.attempt / 'private' / 'incomplete-forward.bin').read_bytes())
        self.assertFalse((journal / '001.partial').exists())
        self.assertTrue((journal / 'terminal.json').exists())
        self.assertEqual(sidecar.manifest['observation']['request_id'], json.loads(self.plan)['requests'][0]['request_id'])

    def test_last_receipt_failure_after_write_still_marks_journal_incomplete(self):
        from looking_glass.pretrained_number_supervisor import _append_receipt, _read_receipts
        def append_then_fail_last(path, receipt):
            _append_receipt(path, receipt)
            if receipt.ordinal == 127:
                raise OSError('failure after final receipt was persisted')
        with patch('looking_glass.pretrained_number_supervisor._append_receipt', new=append_then_fail_last):
            with self.assertRaises(OSError):
                self.run_model(Model())
        private = self.attempt / 'private'
        journal = private / 'journal'
        receipts = _read_receipts(private / 'receipts.jsonl')
        self.assertEqual(len(receipts), 128)
        scan = inspect_discovery_journal(journal, plan_bytes=self.plan, expected_source_commit=self.source,
            expected_plan_sha256=hashlib.sha256(self.plan).hexdigest(), runtime_sha256='b' * 64,
            site_nonce=self.nonce, site_record_bytes=self.site.record_bytes,
            site_commitment=self.site.commitment_sha256, acknowledgments=receipts,
            max_read_bytes=128 * 1024**2, externally_quiescent=True)
        self.assertEqual((scan.status, scan.terminal_present), ('incomplete', True))
        sidecar = decode_incomplete_forward((private / 'incomplete-forward.bin').read_bytes())
        self.assertEqual(sidecar.manifest['observation']['request_id'], json.loads(self.plan)['requests'][-1]['request_id'])

    def test_sidecar_write_failure_does_not_prevent_early_journal_terminal(self):
        from looking_glass.pretrained_number_supervisor import _write_new
        def reject_sidecar(path, data):
            if path.name == 'incomplete-forward.bin':
                raise OSError('sidecar failed')
            return _write_new(path, data)
        with patch('looking_glass.pretrained_number_supervisor._append_receipt', side_effect=OSError('receipt failed')), \
             patch('looking_glass.pretrained_number_supervisor._write_new', new=reject_sidecar):
            with self.assertRaisesRegex(OSError, 'receipt failed'):
                self.run_model(Model())
        private = self.attempt / 'private'
        self.assertTrue((private / 'journal' / 'terminal.json').exists())
        self.assertFalse((private / 'journal' / '001.partial').exists())
        self.assertEqual(json.loads((private / 'failed.json').read_bytes())['partial_evidence'], 'unavailable')

    def test_sidecar_write_cancellation_still_terminalizes_before_propagating(self):
        from looking_glass.pretrained_number_supervisor import _write_new
        def cancel_sidecar(path, data):
            if path.name == 'incomplete-forward.bin':
                raise KeyboardInterrupt('stop')
            return _write_new(path, data)
        with patch('looking_glass.pretrained_number_supervisor._append_receipt', side_effect=OSError('receipt failed')), \
             patch('looking_glass.pretrained_number_supervisor._write_new', new=cancel_sidecar):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_model(Model())
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertTrue((self.attempt / 'private' / 'journal' / 'terminal.json').exists())

    def test_sidecar_write_failure_after_final_receipt_still_terminalizes_journal(self):
        from looking_glass.pretrained_number_supervisor import _append_receipt, _read_receipts, _write_new
        def append_then_fail_last(path, receipt):
            _append_receipt(path, receipt)
            if receipt.ordinal == 127:
                raise OSError('receipt failed')
        def reject_sidecar(path, data):
            if path.name == 'incomplete-forward.bin':
                raise OSError('sidecar failed')
            return _write_new(path, data)
        with patch('looking_glass.pretrained_number_supervisor._append_receipt', new=append_then_fail_last), \
             patch('looking_glass.pretrained_number_supervisor._write_new', new=reject_sidecar):
            with self.assertRaisesRegex(OSError, 'receipt failed'):
                self.run_model(Model())
        private = self.attempt / 'private'
        receipts = _read_receipts(private / 'receipts.jsonl')
        scan = inspect_discovery_journal(private / 'journal', plan_bytes=self.plan,
            expected_source_commit=self.source, expected_plan_sha256=hashlib.sha256(self.plan).hexdigest(),
            runtime_sha256='b' * 64, site_nonce=self.nonce, site_record_bytes=self.site.record_bytes,
            site_commitment=self.site.commitment_sha256, acknowledgments=receipts,
            max_read_bytes=128 * 1024**2, externally_quiescent=True)
        self.assertEqual((len(receipts), scan.status, scan.terminal_present), (128, 'incomplete', True))
        self.assertEqual(json.loads((private / 'failed.json').read_bytes())['partial_evidence'], 'unavailable')

    def test_disk_allowance_covers_final_prediction_before_first_forward(self):
        calls = []
        original = Model.forward
        def counted(model, *args, **kwargs):
            calls.append(1)
            return original(model, *args, **kwargs)
        with patch.object(Model, 'forward', new=counted):
            with self.assertRaises(ValueError):
                self.run_model(Model(), disk_limit_bytes=8 * 1024**2)
        self.assertEqual(calls, [])

    def test_corrupt_oversized_receipt_sidecar_is_read_with_a_memory_bound(self):
        from looking_glass.pretrained_number_supervisor import _read_receipts
        path = Path(self.temp.name) / 'oversized-receipts'
        with path.open('wb') as stream:
            stream.truncate(5 * 1024**2)
        tracemalloc.start()
        try:
            with self.assertRaises(ValueError):
                _read_receipts(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 1024**2)

    def test_serialization_failure_preserves_available_raw_forward_privately(self):
        from looking_glass.pretrained_number_archive import serialize_discovery_observation
        captured = {}
        def reject_injected(index, role, observation, **kwargs):
            if role == 'injected':
                captured['observation'] = observation
                raise ValueError('private serializer failure')
            return serialize_discovery_observation(index, role, observation, **kwargs)
        with patch('looking_glass.pretrained_number_supervisor.serialize_discovery_observation',
                   side_effect=reject_injected):
            with self.assertRaises(ValueError):
                self.run_model(Model())
        journal = self.attempt / 'private' / 'journal'
        self.assertTrue((journal / '002.partial').exists())
        self.assertFalse((journal / '002.frame').exists())
        status = json.loads((self.attempt / 'private' / 'failed.json').read_bytes())
        self.assertEqual(status['partial_evidence'], 'retained')
        rows = (self.attempt / 'private' / 'receipts.jsonl').read_bytes().splitlines()
        acknowledgments = tuple(PairReceipt(**json.loads(row)) for row in rows)
        scan = inspect_discovery_journal(journal, plan_bytes=self.plan, expected_source_commit=self.source,
            expected_plan_sha256=hashlib.sha256(self.plan).hexdigest(), runtime_sha256='b' * 64,
            site_nonce=self.nonce, site_record_bytes=self.site.record_bytes,
            site_commitment=self.site.commitment_sha256, acknowledgments=acknowledgments,
            max_read_bytes=128 * 1024**2, externally_quiescent=True)
        self.assertEqual((len(scan.verified_pairs), scan.tail), (2, 'partial'))
        decoded = decode_incomplete_forward(scan.partial_evidence)
        self.assertEqual(decoded.payloads['observation.logits'], captured['observation'].logits)
        self.assertEqual(decoded.payloads['observation.mlp.2'], captured['observation'].mlp[2])
        self.assertNotEqual(decoded.payloads['actions.0.before'], decoded.payloads['actions.0.after'])

    def test_adapter_failure_keeps_available_private_action_witnesses(self):
        from looking_glass.pretrained_number_adapter import to_selector_observation
        def reject_injected(observation):
            if observation.condition == 'injected':
                raise ValueError('private adapter failure')
            return to_selector_observation(observation)
        with patch('looking_glass.pretrained_number_supervisor.to_selector_observation', side_effect=reject_injected):
            with self.assertRaises(ValueError):
                self.run_model(Model())
        journal = self.attempt / 'private' / 'journal'
        rows = (self.attempt / 'private' / 'receipts.jsonl').read_bytes().splitlines()
        acknowledgments = tuple(PairReceipt(**json.loads(row)) for row in rows)
        scan = inspect_discovery_journal(journal, plan_bytes=self.plan, expected_source_commit=self.source,
            expected_plan_sha256=hashlib.sha256(self.plan).hexdigest(), runtime_sha256='b' * 64,
            site_nonce=self.nonce, site_record_bytes=self.site.record_bytes,
            site_commitment=self.site.commitment_sha256, acknowledgments=acknowledgments,
            max_read_bytes=128 * 1024**2, externally_quiescent=True)
        decoded = decode_incomplete_forward(scan.partial_evidence)
        self.assertNotEqual(decoded.payloads['actions.0.before'], decoded.payloads['actions.0.after'])

    def test_interrupted_forward_return_preserves_the_completed_observation(self):
        previous = sys.gettrace()
        def trace(frame, event, value):
            if (frame.f_code is NumberInjectionExecutor.forward.__code__ and event == 'return'
                    and getattr(value, 'condition', None) == 'injected'):
                sys.settrace(None)
                raise KeyboardInterrupt('private interruption')
            return trace
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                self.run_model(Model())
        finally:
            sys.settrace(previous)
        journal = self.attempt / 'private' / 'journal'
        rows = (self.attempt / 'private' / 'receipts.jsonl').read_bytes().splitlines()
        acknowledgments = tuple(PairReceipt(**json.loads(row)) for row in rows)
        scan = inspect_discovery_journal(journal, plan_bytes=self.plan, expected_source_commit=self.source,
            expected_plan_sha256=hashlib.sha256(self.plan).hexdigest(), runtime_sha256='b' * 64,
            site_nonce=self.nonce, site_record_bytes=self.site.record_bytes,
            site_commitment=self.site.commitment_sha256, acknowledgments=acknowledgments,
            max_read_bytes=128 * 1024**2, externally_quiescent=True)
        decoded = decode_incomplete_forward(scan.partial_evidence)
        self.assertEqual(len(decoded.payloads['observation.logits']), 50257 * 4)
        self.assertNotEqual(decoded.payloads['actions.0.before'], decoded.payloads['actions.0.after'])

    def test_incomplete_action_list_is_reported_as_an_omission(self):
        original = NumberInjectionExecutor.private_evidence
        def missing_self_patch(executor):
            return tuple(item for item in original(executor)
                         if not (item.condition == 'self' and item.operation == 'patch'))
        with patch.object(NumberInjectionExecutor, 'private_evidence', new=missing_self_patch):
            with self.assertRaises(ValueError):
                self.run_model(Model())
        status = json.loads((self.attempt / 'private' / 'failed.json').read_bytes())
        self.assertEqual(status['status'], 'incomplete')
        self.assertEqual(status['partial_evidence'], 'retained-with-omissions')

    def test_available_partial_survives_when_journal_status_is_unavailable(self):
        from looking_glass.pretrained_number_adapter import to_selector_observation
        failed = [False]
        original_reservation = DiscoveryJournal.reservation
        def adapter(observation):
            if observation.condition == 'injected':
                failed[0] = True
                raise ValueError('private adapter failure')
            return to_selector_observation(observation)
        def reservation(journal):
            if failed[0]:
                raise RuntimeError('journal status unavailable')
            return original_reservation(journal)
        with patch('looking_glass.pretrained_number_supervisor.to_selector_observation', side_effect=adapter), \
             patch.object(DiscoveryJournal, 'reservation', new=reservation):
            with self.assertRaises(ValueError):
                self.run_model(Model())
        raw = (self.attempt / 'private' / 'incomplete-forward.bin').read_bytes()
        decoded = decode_incomplete_forward(raw)
        self.assertEqual(len(decoded.payloads['observation.logits']), 50257 * 4)
        self.assertNotEqual(decoded.payloads['actions.0.before'], decoded.payloads['actions.0.after'])
        status = json.loads((self.attempt / 'private' / 'failed.json').read_bytes())
        self.assertEqual(status['partial_evidence'], 'retained')

    def test_serializer_failure_still_terminates_journal_and_preserves_original_error(self):
        with patch('looking_glass.pretrained_number_supervisor.to_selector_observation',
                   side_effect=ValueError('original adapter failure')), \
             patch('looking_glass.pretrained_number_supervisor.encode_incomplete_forward',
                   side_effect=RuntimeError('evidence serializer failed')):
            with self.assertRaisesRegex(ValueError, 'original adapter failure'):
                self.run_model(Model())
        private = self.attempt / 'private'
        self.assertTrue((private / 'journal' / 'terminal.json').exists())
        self.assertEqual(json.loads((private / 'failed.json').read_bytes())['partial_evidence'], 'unavailable')
        self.assertFalse((private / 'incomplete-forward.bin').exists())

    def test_serializer_cancellation_terminalizes_before_propagating(self):
        with patch('looking_glass.pretrained_number_supervisor.to_selector_observation',
                   side_effect=ValueError('original adapter failure')), \
             patch('looking_glass.pretrained_number_supervisor.encode_incomplete_forward',
                   side_effect=KeyboardInterrupt('stop')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_model(Model())
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertTrue((self.attempt / 'private' / 'journal' / 'terminal.json').exists())

    def test_original_cancellation_survives_evidence_close_failure(self):
        from looking_glass.pretrained_number_supervisor import _write_new
        close = os.close
        def close_then_fail(fd):
            close(fd)
            raise OSError('private close error')
        with patch('looking_glass.pretrained_number_supervisor.os.write', side_effect=KeyboardInterrupt('stop')), \
             patch('looking_glass.pretrained_number_supervisor.os.close', side_effect=close_then_fail):
            with self.assertRaises(KeyboardInterrupt) as caught:
                _write_new(Path(self.temp.name) / 'interrupt', b'data')
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_new_cancellation_during_close_keeps_the_original_write_error(self):
        from looking_glass.pretrained_number_supervisor import _write_new
        close = os.close
        def close_then_interrupt(fd):
            close(fd)
            raise KeyboardInterrupt('stop')
        with patch('looking_glass.pretrained_number_supervisor.os.write', side_effect=OSError('private write error')), \
             patch('looking_glass.pretrained_number_supervisor.os.close', side_effect=close_then_interrupt):
            with self.assertRaises(KeyboardInterrupt) as caught:
                _write_new(Path(self.temp.name) / 'close-interrupt', b'data')
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_receipt_read_cancellation_survives_close_failure(self):
        from looking_glass.pretrained_number_supervisor import _read_receipts
        path = Path(self.temp.name) / 'receipt'
        path.write_bytes(b'{}\n')
        close = os.close
        def close_then_fail(fd):
            close(fd)
            raise OSError('private close error')
        with patch('looking_glass.pretrained_number_supervisor.os.read', side_effect=KeyboardInterrupt('stop')), \
             patch('looking_glass.pretrained_number_supervisor.os.close', side_effect=close_then_fail):
            with self.assertRaises(KeyboardInterrupt) as caught:
                _read_receipts(path)
        self.assertIsInstance(caught.exception.__cause__, OSError)


if __name__ == '__main__':
    unittest.main()
