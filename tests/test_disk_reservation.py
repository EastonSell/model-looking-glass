"""Small CPU-only tests for cooperative disk admission and owner custody."""
import json
import inspect
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass.disk_reservation import (
    DiskCapacityError, DiskLedger, DiskLedgerError, DiskOwnershipUncertain,
)

_CHILD_PYTHON = getattr(sys, '_base_executable', None) or sys.executable


def fake_free(value):
    return patch('looking_glass.disk_reservation.shutil.disk_usage',
                 return_value=SimpleNamespace(free=value))


class DiskReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def ledger(self, cushion=100):
        return DiskLedger(self.root, target_dir=self.root, cushion_bytes=cushion)

    def test_claims_share_one_fixed_cushion_and_recheck_live_free(self):
        with fake_free(1000):
            ledger = self.ledger()
            first = ledger.reserve(output_bytes=300, staging_bytes=100)
            second = ledger.reserve(output_bytes=200, staging_bytes=100)
            snap = ledger.snapshot()
            self.assertEqual((snap.free_bytes, snap.committed_bytes, snap.cushion_bytes), (1000, 700, 100))
            self.assertEqual([row.status for row in snap.claims], ['held', 'held'])
            with self.assertRaises(DiskCapacityError):
                ledger.reserve(output_bytes=201, staging_bytes=0)
        with fake_free(799):
            with self.assertRaises(DiskCapacityError):
                first.check()
        with fake_free(800):
            first.check()
            first.release()
            second.release()
            self.assertEqual(ledger.snapshot().committed_bytes, 0)

    def test_admission_and_recheck_read_the_actual_target_volume(self):
        real_usage = shutil.disk_usage
        readings = []
        def observe(path):
            result = real_usage(path)
            readings.append((Path(path), result.free))
            return result
        with patch('looking_glass.disk_reservation.shutil.disk_usage', side_effect=observe):
            ledger = self.ledger(cushion=0)
            lease = ledger.reserve(output_bytes=1, staging_bytes=1)
            snap = lease.check()
            self.assertEqual(snap.free_bytes, readings[-1][1])
            self.assertGreaterEqual(len(readings), 2)
            self.assertTrue(all(path == self.root.resolve() for path, _ in readings))
            lease.release()

    def test_rejects_bad_arithmetic_and_mismatched_cushion_or_volume(self):
        with fake_free(1000):
            ledger = self.ledger()
            for output, staging in ((True, 0), (-1, 0), (0, 0), (2**63, 0), (2**63-1, 1)):
                with self.subTest(output=output, staging=staging):
                    with self.assertRaises((ValueError, DiskLedgerError)):
                        ledger.reserve(output_bytes=output, staging_bytes=staging)
            with self.assertRaises(DiskLedgerError):
                self.ledger(cushion=99)
            other = self.root / 'other'
            other.mkdir()
            from looking_glass import disk_reservation as module
            identity = module._volume_identity
            def changed(path):
                return 'different-volume' if Path(path) == other.resolve() else identity(path)
            lease = ledger.reserve(output_bytes=10, staging_bytes=1)
            with patch.object(module, '_volume_identity', side_effect=changed):
                with self.assertRaises(DiskLedgerError):
                    ledger.reserve(output_bytes=10, staging_bytes=0, target_dir=other)
                with self.assertRaises(DiskLedgerError):
                    ledger.reserve(output_bytes=10, staging_bytes=1, staging_dir=other)
                with self.assertRaises(DiskLedgerError):
                    lease.check(staging_dir=other)
            lease.release()

    def test_stale_owner_remains_charged_until_explicit_reconciliation(self):
        with fake_free(1000):
            ledger = self.ledger()
        script = r'''import json,sys
from types import SimpleNamespace
from unittest.mock import patch
from looking_glass.disk_reservation import DiskLedger
with patch('looking_glass.disk_reservation.shutil.disk_usage', return_value=SimpleNamespace(free=1000)):
    ledger=DiskLedger(sys.argv[1],target_dir=sys.argv[1],cushion_bytes=100)
    lease=ledger.reserve(output_bytes=400,staging_bytes=0)
    print(json.dumps({'claim_id':lease.claim_id}),flush=True)
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
        child = subprocess.run([_CHILD_PYTHON, '-B', '-c', script, str(self.root)],
                               cwd=Path(__file__).resolve().parents[1], env=env,
                               text=True, capture_output=True, timeout=20, check=True)
        claim_id = json.loads(child.stdout)['claim_id']
        with fake_free(1000):
            snap = ledger.snapshot()
            self.assertEqual((snap.committed_bytes, snap.claims[0].status), (400, 'stale'))
            with self.assertRaises(DiskOwnershipUncertain):
                ledger.reserve(output_bytes=1, staging_bytes=0)
            ledger.reconcile_stale(claim_id, reason='operator verified all writers exited')
            self.assertEqual(ledger.snapshot().committed_bytes, 0)
            new = ledger.reserve(output_bytes=1, staging_bytes=0)
            new.release()

    def test_held_or_missing_lock_cannot_be_reconciled(self):
        with fake_free(1000):
            ledger = self.ledger()
            lease = ledger.reserve(output_bytes=100, staging_bytes=0)
            with self.assertRaises(DiskOwnershipUncertain):
                ledger.reconcile_stale(lease.claim_id, reason='not yet safe')
            lease.release()

    def test_missing_lock_stays_unknown_and_blocks_admission(self):
        with fake_free(1000):
            ledger = self.ledger()
        script = r'''import json,sys
from types import SimpleNamespace
from unittest.mock import patch
from looking_glass.disk_reservation import DiskLedger
with patch('looking_glass.disk_reservation.shutil.disk_usage', return_value=SimpleNamespace(free=1000)):
    ledger=DiskLedger(sys.argv[1],target_dir=sys.argv[1],cushion_bytes=100)
    lease=ledger.reserve(output_bytes=400,staging_bytes=0)
    print(json.dumps({'claim_id':lease.claim_id}),flush=True)
'''
        child = subprocess.run([_CHILD_PYTHON, '-B', '-c', script, str(self.root)],
                               cwd=Path(__file__).resolve().parents[1], text=True,
                               capture_output=True, timeout=20, check=True)
        claim_id = json.loads(child.stdout)['claim_id']
        ledger._owner_path(claim_id).unlink()
        with fake_free(1000):
            snap = ledger.snapshot()
            self.assertEqual((snap.committed_bytes, snap.claims[0].status), (400, 'unknown'))
            with self.assertRaises(DiskOwnershipUncertain):
                ledger.reserve(output_bytes=1, staging_bytes=0)
            with self.assertRaises(DiskOwnershipUncertain):
                ledger.reconcile_stale(claim_id, reason='missing lock is not proof of safety')

    def test_interrupted_stale_probe_stays_unknown_across_processes(self):
        with fake_free(1000):
            ledger = self.ledger()
        script = r'''import json,os,sys
from types import SimpleNamespace
from unittest.mock import patch
from looking_glass.disk_reservation import DiskLedger
with patch('looking_glass.disk_reservation.shutil.disk_usage', return_value=SimpleNamespace(free=1000)):
    ledger=DiskLedger(sys.argv[1],target_dir=sys.argv[1],cushion_bytes=100)
    if sys.argv[2] == 'claim':
        lease=ledger.reserve(output_bytes=400,staging_bytes=0)
        print(lease.claim_id,flush=True)
        os._exit(0)
    print(json.dumps([c.status for c in ledger.snapshot().claims]),flush=True)
'''
        def child(mode):
            return subprocess.run([_CHILD_PYTHON, '-B', '-c', script, str(self.root), mode],
                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=20, check=True).stdout.strip()
        claim_id = child('claim')
        lines, first = inspect.getsourcelines(DiskLedger._probe_owner_locked)
        target = first + next(i for i, line in enumerate(lines) if line.strip() == 'os.close(closing)')
        leaked = []
        def interrupt(frame, event, _arg):
            if frame.f_code is DiskLedger._probe_owner_locked.__code__ and event == 'line' and frame.f_lineno == target:
                leaked.append(frame.f_locals['closing'])
                sys.settrace(None)
                raise KeyboardInterrupt('interrupted owner probe close')
            return interrupt
        try:
            try:
                sys.settrace(interrupt)
                with fake_free(1000), self.assertRaises(KeyboardInterrupt):
                    ledger.snapshot()
            finally:
                sys.settrace(None)
            self.assertEqual(json.loads(child('status')), ['unknown'])
            with fake_free(1000), self.assertRaises(DiskOwnershipUncertain):
                ledger.reserve(output_bytes=1, staging_bytes=0)
        finally:
            for fd in leaked:
                os.close(fd)
        with fake_free(1000):
            ledger.reconcile_stale(claim_id, reason='test verified all owners and writers exited')
            self.assertEqual(ledger.snapshot().committed_bytes, 0)

    def test_owner_probe_holds_the_fork_gate_until_closed(self):
        from looking_glass import disk_reservation as module
        with fake_free(1000):
            ledger = self.ledger()
            lease = ledger.reserve(output_bytes=100)
            entered, resume = threading.Event(), threading.Event()
            outcomes = queue.Queue()
            real_try_lock = module._try_lock
            def pause_probe(fd):
                acquired = real_try_lock(fd)
                entered.set()
                if not resume.wait(10):
                    raise TimeoutError('test probe was not resumed')
                return acquired
            def snapshot():
                try:
                    outcomes.put(ledger.snapshot())
                except BaseException as error:
                    outcomes.put(error)
            with patch.object(module, '_try_lock', side_effect=pause_probe):
                worker = threading.Thread(target=snapshot, daemon=True)
                worker.start()
                try:
                    self.assertTrue(entered.wait(10))
                    acquired = module._ROOT_LOCK.acquire(timeout=0.1)
                    if acquired:
                        module._ROOT_LOCK.release()
                    self.assertFalse(acquired)
                finally:
                    resume.set()
                    worker.join(timeout=10)
                self.assertFalse(worker.is_alive())
                outcome = outcomes.get(timeout=1)
                if isinstance(outcome, BaseException):
                    raise outcome
                self.assertEqual(outcome.claims[0].status, 'held')
            lease.release()

    @unittest.skipUnless(os.name == 'nt', 'uses a real Windows directory junction')
    def test_owner_directory_junction_is_rejected_at_init_and_recheck(self):
        owner = self.root / '.disk-reservation-owners'
        target = self.root / 'junction-target'
        target.mkdir()
        def create_junction():
            result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(owner), str(target)],
                                    capture_output=True, text=True, timeout=20)
            if result.returncode:
                self.skipTest('this account cannot create a directory junction')
        def remove_junction():
            self.assertTrue(os.lstat(owner).st_file_attributes & 0x400)
            os.rmdir(owner)  # Remove only the junction, never its target.
        create_junction()
        try:
            with self.assertRaises(DiskLedgerError):
                self.ledger()
        finally:
            remove_junction()
        ledger = self.ledger()
        backup = self.root / 'original-owner-directory'
        os.rename(owner, backup)
        try:
            create_junction()
            try:
                with self.assertRaises(DiskLedgerError):
                    ledger.snapshot()
            finally:
                remove_junction()
        finally:
            os.rename(backup, owner)

    def test_two_processes_cannot_both_admit_the_same_free_pool(self):
        script = r'''import json,sys
from types import SimpleNamespace
from unittest.mock import patch
from looking_glass.disk_reservation import DiskLedger,DiskCapacityError
with patch('looking_glass.disk_reservation.shutil.disk_usage', return_value=SimpleNamespace(free=1000)):
    ledger=DiskLedger(sys.argv[1],target_dir=sys.argv[1],cushion_bytes=100)
    print(json.dumps({'kind':'ready'}),flush=True)
    sys.stdin.readline()
    try:
        lease=ledger.reserve(output_bytes=600,staging_bytes=0)
    except DiskCapacityError:
        print(json.dumps({'kind':'denied'}),flush=True)
    else:
        print(json.dumps({'kind':'admitted','id':lease.claim_id}),flush=True)
        sys.stdin.readline()
        lease.release()
'''
        children, queues = [], []
        def read_lines(stream, messages):
            for line in stream:
                messages.put(json.loads(line))
        try:
            for _ in range(2):
                child = subprocess.Popen([_CHILD_PYTHON, '-B', '-c', script, str(self.root)],
                    cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                children.append(child)
                messages = queue.Queue()
                queues.append(messages)
                threading.Thread(target=read_lines, args=(child.stdout, messages), daemon=True).start()
            for messages in queues:
                self.assertEqual(messages.get(timeout=15), {'kind':'ready'})
            for child in children:
                child.stdin.write('start\n')
                child.stdin.flush()
            outcomes = [messages.get(timeout=15) for messages in queues]
            self.assertEqual(sorted(outcome['kind'] for outcome in outcomes), ['admitted', 'denied'])
            with fake_free(1000):
                self.assertEqual(self.ledger().snapshot().committed_bytes, 600)
            for child, outcome in zip(children, outcomes):
                if outcome['kind'] == 'admitted':
                    child.stdin.write('release\n')
                    child.stdin.flush()
                self.assertEqual(child.wait(timeout=15), 0, child.stderr.read())
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()

    def test_lost_reservation_return_keeps_the_live_claim_rooted(self):
        from looking_glass import disk_reservation as module
        with fake_free(1000):
            ledger = self.ledger()
            before = set(module._ROOTS)
            lines, first = inspect.getsourcelines(DiskLedger.reserve)
            target = first + next(i for i, line in enumerate(lines) if line.strip() == 'return lease')
            def interrupt(frame, event, _arg):
                if frame.f_code is DiskLedger.reserve.__code__ and event == 'line' and frame.f_lineno == target:
                    sys.settrace(None)
                    raise KeyboardInterrupt('lost reservation return')
                return interrupt
            try:
                sys.settrace(interrupt)
                with self.assertRaises(KeyboardInterrupt):
                    ledger.reserve(output_bytes=100, staging_bytes=0)
            finally:
                sys.settrace(None)
            claimed = set(module._ROOTS) - before
            self.assertEqual(len(claimed), 1)
            self.assertEqual(ledger.snapshot().committed_bytes, 100)
            module._ROOTS[claimed.pop()].release()

    def test_interrupted_release_revokes_the_lease_and_can_be_reconciled_by_owner(self):
        from looking_glass.disk_reservation import DiskReservation
        with fake_free(1000):
            ledger = self.ledger()
            lease = ledger.reserve(output_bytes=100, staging_bytes=0)
            lines, first = inspect.getsourcelines(DiskReservation.release)
            target = first + next(i for i, line in enumerate(lines) if line.strip() == 'self._close_handle()')
            def interrupt(frame, event, _arg):
                if frame.f_code is DiskReservation.release.__code__ and event == 'line' and frame.f_lineno == target:
                    sys.settrace(None)
                    raise KeyboardInterrupt('lost release return')
                return interrupt
            try:
                sys.settrace(interrupt)
                with self.assertRaises(KeyboardInterrupt):
                    lease.release()
            finally:
                sys.settrace(None)
            with self.assertRaises(DiskLedgerError):
                lease.check()
            self.assertEqual(ledger.snapshot().committed_bytes, 0)
            lease.release()

    def test_release_revokes_further_checks_and_is_idempotent(self):
        with fake_free(1000):
            ledger = self.ledger()
            lease = ledger.reserve(output_bytes=100, staging_bytes=0)
            lease.release()
            lease.release()
            with self.assertRaises(DiskLedgerError):
                lease.check()

    def test_unknown_persisted_state_cannot_hide_a_claim(self):
        with fake_free(1000):
            ledger = self.ledger()
            lease = ledger.reserve(output_bytes=100, staging_bytes=0)
            def set_state(state):
                connection = sqlite3.connect(ledger._db)
                try:
                    connection.execute('UPDATE claims SET state=? WHERE claim_id=?', (state, lease.claim_id))
                    connection.commit()
                finally:
                    connection.close()
            set_state('unrecognized')
            with self.assertRaises(DiskLedgerError):
                ledger.snapshot()
            with self.assertRaises(DiskLedgerError):
                ledger.reserve(output_bytes=1, staging_bytes=0)
            set_state('active')
            lease.release()


if __name__ == '__main__':
    unittest.main()
