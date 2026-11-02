"""Synthetic cross-track ordering; no study split or model execution."""
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    gate = importlib.import_module('looking_glass.agreement_split_gate')
except ModuleNotFoundError as error:
    if error.name != 'looking_glass.agreement_split_gate':
        raise
    gate = None


class SplitGateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gate, 'shared split gate is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.natural = {'track': 'natural', 'freeze_sha256': 'a' * 64,
                        'heldout_commitment': 'b' * 64,
                        'inventory_sha256': 'd' * 64}
        self.injection = {'track': 'injection', 'freeze_sha256': 'e' * 64,
                          'heldout_commitment': 'f' * 64,
                          'inventory_sha256': '2' * 64}
        for value in (self.natural, self.injection):
            raw = self.commitment_bytes(value)
            value['commitment_file_sha256'] = hashlib.sha256(raw).hexdigest()

    def commitment_bytes(self, values):
        value = {'schema': 'mlg-' + values['track'] + '-heldout-public-commitment-v1',
                 'prediction_freeze_sha256': values['freeze_sha256'],
                 'heldout_commitment': values['heldout_commitment'],
                 'exposure_inventory_sha256': values['inventory_sha256'],
                 'operational_binding_sha256': '4' * 64}
        if values['track'] == 'natural':
            value.update({'wrapper_sha256': '5' * 64, 'deviation_sha256': '6' * 64})
        else:
            value['custody_source_sha256'] = '7' * 64
        return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii')

    def commit_track(self, locked, values):
        locked.announce_allocation(values['track'], values['freeze_sha256'])
        locked.announce_commitment(**values, commitment_bytes=self.commitment_bytes(values),
                                   publication_check=lambda raw: True)

    def fake_receipt(self):
        return json.dumps({'schema': 'mlg-number-disjoint-split-check-v1',
            'natural_prediction_freeze_sha256': self.natural['freeze_sha256'],
            'natural_heldout_commitment': self.natural['heldout_commitment'],
            'natural_exposure_inventory_sha256': self.natural['inventory_sha256'],
            'natural_commitment_file_sha256': self.natural['commitment_file_sha256'],
            'injection_prediction_freeze_sha256': self.injection['freeze_sha256'],
            'injection_heldout_commitment': self.injection['heldout_commitment'],
            'injection_exposure_inventory_sha256': self.injection['inventory_sha256'],
            'injection_commitment_file_sha256': self.injection['commitment_file_sha256'],
            'comparator_source_sha256': '3' * 64, 'natural_case_count': 16,
            'injection_case_count': 16, 'status': 'clear'}, sort_keys=True,
            separators=(',', ':')).encode('ascii')

    def test_unallocated_other_is_unchecked_and_later_split_needs_clearance(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            self.commit_track(locked, self.natural)
            first = locked.announce_forward('natural')
            self.assertEqual(first['comparison'], 'other_not_allocated_unchecked')
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            self.commit_track(locked, self.injection)
            with self.assertRaises(gate.GateError):
                locked.announce_forward('injection')
            second = locked.announce_forward('injection', receipt_bytes=self.fake_receipt(),
                                             receipt_check=lambda raw: True)
            self.assertEqual(second['comparison'], 'clear')
        self.assertNotIn('token', json.dumps(second))

    def test_peer_allocation_without_commitment_is_pending_not_clear(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            self.commit_track(locked, self.natural)
            locked.announce_allocation('injection', self.injection['freeze_sha256'])
            first = locked.announce_forward('natural')
            self.assertEqual(first['comparison'], 'other_allocation_pending_unchecked')
            locked.announce_commitment(**self.injection, commitment_bytes=self.commitment_bytes(self.injection),
                                       publication_check=lambda raw: True)
            with self.assertRaises(gate.GateError):
                locked.announce_forward('injection')
            second = locked.announce_forward('injection', receipt_bytes=self.fake_receipt(),
                                             receipt_check=lambda raw: True)
            self.assertEqual(second['comparison'], 'clear')

    def test_forged_receipt_cannot_authorize_forward(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            self.commit_track(locked, self.natural)
            self.commit_track(locked, self.injection)
            bad = self.fake_receipt().replace(b'"status":"clear"', b'"status":"collision"')
            with self.assertRaises(gate.GateError):
                locked.announce_forward('natural', receipt_bytes=bad, receipt_check=lambda raw: True)
            with self.assertRaises(gate.GateError):
                locked.announce_forward('natural', receipt_bytes=self.fake_receipt(),
                                        receipt_check=lambda raw: False)

    def test_lock_excludes_another_process(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1):
            program = ('from looking_glass.agreement_split_gate import acquire; '
                       'import sys; acquire(sys.argv[1], expected_root=sys.argv[1], timeout=0)')
            result = subprocess.run([sys.executable, '-B', '-c', program, str(self.root)],
                                    capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)

    def test_unobserved_or_changed_commitment_cannot_be_announced(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            locked.announce_allocation('natural', self.natural['freeze_sha256'])
            with self.assertRaises(gate.GateError):
                locked.announce_commitment(**self.natural, commitment_bytes=self.commitment_bytes(self.natural),
                                           publication_check=lambda raw: False)
            with self.assertRaises(gate.GateError):
                locked.announce_commitment(**self.natural, commitment_bytes=b'{}',
                                           publication_check=lambda raw: True)

    def test_corrupt_peer_forward_marker_is_not_treated_as_clear(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            self.commit_track(locked, self.natural)
            self.commit_track(locked, self.injection)
            (self.root / 'injection-forward.json').write_bytes(b'{"corrupt":true}')
            with self.assertRaises(gate.GateError):
                locked.announce_forward('natural', receipt_bytes=self.fake_receipt(),
                                        receipt_check=lambda raw: True)

    def test_revoked_lease_after_marker_does_not_authorize(self):
        locked = gate.acquire(self.root, expected_root=self.root, timeout=1)
        real_write = gate._write_new
        def revoke_after_write(path, data):
            real_write(path, data)
            locked.lease.release()
        with mock.patch.object(gate, '_write_new', side_effect=revoke_after_write):
            with self.assertRaises(Exception):
                locked.announce_allocation('natural', self.natural['freeze_sha256'])
        self.assertTrue((self.root / 'natural-allocation.json').exists())

    def test_interrupted_unlock_never_leaves_a_stale_gate_authorized(self):
        child = ('import sys; from looking_glass.agreement_split_gate import acquire; '
                 'g=acquire(sys.argv[1],expected_root=sys.argv[1],timeout=2); print("ready",flush=True); '
                 'sys.stdin.readline(); g.lease.release()')
        probe = '''
import subprocess, sys
from looking_glass import device_lease
from looking_glass.agreement_split_gate import acquire
g = acquire(sys.argv[1], expected_root=sys.argv[1], timeout=2)
real_unlock = device_lease._unlock
def interrupted(fd):
    real_unlock(fd)
    raise KeyboardInterrupt('synthetic post-unlock interruption')
device_lease._unlock = interrupted
try:
    g.lease.release()
except KeyboardInterrupt:
    pass
else:
    raise RuntimeError('interruption was not delivered')
device_lease._unlock = real_unlock
child = subprocess.Popen([sys.executable, '-B', '-c', sys.argv[2], sys.argv[1]],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
try:
    if child.stdout.readline().strip() != 'ready':
        raise RuntimeError('successor did not acquire')
    try:
        g.lease.check()
    except Exception:
        rejected = True
    else:
        rejected = False
finally:
    child.stdin.write('exit\\n'); child.stdin.flush()
    code = child.wait(timeout=10)
if code != 0 or not rejected:
    raise RuntimeError('stale lease remained authorized')
'''
        result = subprocess.run([sys.executable, '-B', '-c', probe, str(self.root), child],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unmatched_canonical_gate_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            with self.assertRaises(gate.GateError):
                gate.acquire(self.root, expected_root=Path(other), timeout=1)

    def test_unexpected_gate_file_fails_closed(self):
        (self.root / 'unreviewed-extra.bin').write_bytes(b'x')
        with self.assertRaises(gate.GateError):
            gate.acquire(self.root, expected_root=self.root, timeout=1)

    def test_peer_marker_mutation_during_receipt_check_is_rejected(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            self.commit_track(locked, self.natural)
            self.commit_track(locked, self.injection)
            def mutate(raw):
                path = self.root / 'injection-commitment.json'
                value = json.loads(path.read_bytes())
                value['inventory_sha256'] = '8' * 64
                path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')), encoding='utf-8')
                return True
            with self.assertRaises(gate.GateError):
                locked.announce_forward('natural', receipt_bytes=self.fake_receipt(), receipt_check=mutate)

    def test_previously_forwarded_commitment_cannot_be_replaced_under_same_freeze(self):
        with gate.acquire(self.root, expected_root=self.root, timeout=1) as locked:
            self.commit_track(locked, self.natural)
            locked.announce_forward('natural')
            self.commit_track(locked, self.injection)
            path = self.root / 'natural-commitment.json'
            original = json.loads(path.read_bytes())
            for field in ('heldout_commitment', 'commitment_file_sha256', 'inventory_sha256'):
                with self.subTest(field=field):
                    marker = dict(original)
                    marker[field] = '9' * 64
                    path.write_text(json.dumps(marker, sort_keys=True, separators=(',', ':')), encoding='utf-8')
                    previous = self.natural[field]
                    self.natural[field] = marker[field]
                    try:
                        with self.assertRaises(gate.GateError):
                            locked.announce_forward('injection', receipt_bytes=self.fake_receipt(),
                                                    receipt_check=lambda raw: True)
                    finally:
                        self.natural[field] = previous


if __name__ == '__main__':
    unittest.main()
