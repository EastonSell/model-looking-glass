"""CPU-only synthetic checks of the cooperative study reservation ledger."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from looking_glass import agreement_phase_ledger as ledger


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    return path.read_bytes()


class AgreementPhaseLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        payload = b'synthetic public preflight payload'
        self.preflight = self.root / 'preflight.f32'
        self.preflight.write_bytes(payload)
        count_patch = patch.object(ledger, 'PREFLIGHT_BYTES', len(payload))
        hash_patch = patch.object(ledger, 'PREFLIGHT_SHA256', hashlib.sha256(payload).hexdigest())
        count_patch.start()
        hash_patch.start()
        self.addCleanup(count_patch.stop)
        self.addCleanup(hash_patch.stop)
        self.out = self.root / 'natural'
        attempt = self.out / 'owner' / 'attempt'
        (attempt / 'tensors').mkdir(parents=True)
        hashes = {}
        for name in ('started.json', 'events.jsonl', 'terminal.json'):
            data = ('synthetic-' + name).encode()
            (attempt / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        tensor = b'T'
        tensor_hash = hashlib.sha256(tensor).hexdigest()
        (attempt / 'tensors' / (tensor_hash + '.f32le')).write_bytes(tensor)
        owner = {'status': 'evidence_complete', 'exit_code': 0, 'child_reaped': True,
                 'job_drained': True, 'job_closed': True, 'cleanup_verified': True,
                 'inventory': {'retained_tensor_bytes': 1, 'tensor_sha256': [tensor_hash],
                               'started_sha256': hashes['started.json'],
                               'events_sha256': hashes['events.jsonl'],
                               'terminal_sha256': hashes['terminal.json']}}
        owner_bytes = _write(self.out / 'owner' / 'supervisor.json', owner)
        _write(self.out / 'launcher-terminal.json', {
            'status': 'evidence_complete', 'child_started': True, 'cleanup_verified': True,
            'owner_record': 'owner/supervisor.json', 'owner_supervisor_sha256': hashlib.sha256(owner_bytes).hexdigest(),
        })
        (self.out / 'unreferenced.partial').write_bytes(b'also charged')
        self.extra = self.root / 'natural-public-copy.zip'
        self.extra.write_bytes(b'a physical copy')
        self.audit_source_hash = 'a' * 64
        self.audit = self.root / 'independent-audit.json'
        _write(self.audit, {'schema': 'mlg-independent-natural-discovery-audit-v1',
                            'status': 'independent_saved_evidence_pass', 'audit_source_sha256': self.audit_source_hash,
                            'tensor_count': 1, 'tensor_bytes': 1,
                            'journal_started_sha256': hashes['started.json'],
                            'journal_event_sha256': hashes['events.jsonl'],
                            'journal_terminal_sha256': hashes['terminal.json']})
        self.budget = {
            'study_limit_bytes': ledger.STUDY_LOGICAL_LIMIT,
            'host_disk_allowance_bytes': ledger.STUDY_DISK_LIMIT,
            'prior_unique_tensor_bytes': len(payload),
            'required_free_disk_bytes': ledger.STUDY_DISK_LIMIT,
            'phase_reservations_bytes': {p: v[0] for p, v in ledger.PHASE_LIMITS.items()},
            'host_disk_phase_reservations_bytes': {p: v[1] for p, v in ledger.PHASE_LIMITS.items()},
        }
        self.binding = self.root / 'binding.json'
        b = _write(self.binding, {'study_budget': self.budget, 'independent_natural_audit_source_sha256': self.audit_source_hash})
        self.claim = self.root / 'natural.claim'
        _write(self.claim, {'schema': 'mlg-natural-study-claim-v1', 'budget': self.budget,
                            'binding_sha256': hashlib.sha256(b).hexdigest(), 'output': str(self.out)})
        self.db = self.root / 'study.sqlite3'
        self.args = {'natural_claim_path': self.claim, 'natural_binding_path': self.binding,
                     'prior_paths': (self.preflight,), 'natural_extra_paths': (self.extra,),
                     'prior_unique_tensor_bytes': len(payload), 'natural_unique_tensor_bytes': 1,
                     'natural_audit_path': self.audit, 'expected_natural_audit_sha256': hashlib.sha256(self.audit.read_bytes()).hexdigest(),
                     'expected_natural_terminal_sha256': hashlib.sha256((self.out / 'launcher-terminal.json').read_bytes()).hexdigest(),
                     'expected_natural_claim_sha256': hashlib.sha256(self.claim.read_bytes()).hexdigest(),
                     'expected_natural_binding_sha256': hashlib.sha256(self.binding.read_bytes()).hexdigest()}

    def initialize(self):
        ledger.initialize_study_ledger(self.db, **self.args)

    def claim_phase(self, phase='injection_discovery', name='injection'):
        return ledger.claim_phase(self.db, phase, output_path=self.root / name)

    def test_import_exact_limits_physical_copies_and_unique_payload(self):
        self.initialize()
        with closing(sqlite3.connect(self.db)) as db:
            meta_text, inv_text = db.execute('SELECT data, inventory FROM meta').fetchone()
            claims = db.execute('SELECT phase, logical, disk FROM claims').fetchall()
        meta, inv = json.loads(meta_text), json.loads(inv_text)
        self.assertEqual(claims, [('natural_discovery', *ledger.PHASE_LIMITS['natural_discovery'])])
        self.assertEqual(meta['natural_physical_bytes'], sum(v[0] for v in inv['natural'].values()))
        self.assertEqual(meta['prior_physical_bytes'], sum(v[0] for v in inv['prior'].values()))
        self.assertEqual(meta['prior_unique_tensor_bytes'], len(self.preflight.read_bytes()))
        self.assertIn(os.path.normcase(str(self.extra)), inv['natural'])
        self.assertIn(os.path.normcase(str(self.out / 'unreferenced.partial')), inv['natural'])
        grant = self.claim_phase()
        self.assertEqual((grant.logical_limit_bytes, grant.disk_limit_bytes), (48 * 1024**2, 128 * 1024**2))
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase(name='another-attempt')
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase('natural_discovery', 'cannot-reissue')

    def test_one_winner_for_concurrent_claims(self):
        self.initialize()
        def run(i):
            try:
                return self.claim_phase(name=f'attempt-{i}')
            except ledger.LedgerError:
                return None
        with ThreadPoolExecutor(max_workers=6) as pool:
            grants = list(pool.map(run, range(6)))
        self.assertEqual(sum(g is not None for g in grants), 1)

    def test_failed_or_interrupted_caller_cannot_reuse_committed_claim(self):
        self.initialize()
        with patch.object(ledger, 'PhaseGrant', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.claim_phase()
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase(name='retry')

    def test_separate_processes_cannot_reissue_a_phase(self):
        self.initialize()
        code = ('import sys; import looking_glass.agreement_phase_ledger as m; m.PREFLIGHT_BYTES=int(sys.argv[3]); '
                'from looking_glass.agreement_phase_ledger import claim_phase, LedgerError; '
                'exec("try:\\n claim_phase(sys.argv[1], \'injection_discovery\', output_path=sys.argv[2])\\n'
                ' print(\'CLAIMED\')\\nexcept LedgerError:\\n print(\'DENIED\')")')
        processes = [subprocess.Popen([sys.executable, '-B', '-c', code, str(self.db), str(self.root / f'child-{i}'), str(ledger.PREFLIGHT_BYTES)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for i in range(2)]
        result = [p.communicate(timeout=30) for p in processes]
        self.assertEqual([p.returncode for p in processes], [0, 0], result)
        self.assertEqual(sorted(out.strip() for out, _ in result), ['CLAIMED', 'DENIED'])

    def test_changed_process_budget_cannot_enlarge_a_phase(self):
        self.initialize()
        changed = dict(ledger.PHASE_LIMITS)
        changed['injection_discovery'] = (400 * 1024**2, 128 * 1024**2)
        with patch.object(ledger, 'PHASE_LIMITS', changed):
            with self.assertRaises(ledger.LedgerError):
                self.claim_phase()
        self.assertEqual(self.claim_phase().logical_limit_bytes, 48 * 1024**2)

    def test_unfinished_or_forged_natural_owner_is_not_imported(self):
        terminal = self.out / 'launcher-terminal.json'
        t = json.loads(terminal.read_text())
        t['cleanup_verified'] = False
        _write(terminal, t)
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.assertFalse(self.db.exists())

    def test_claim_and_binding_require_independent_expected_digests(self):
        self.args['expected_natural_claim_sha256'] = '0' * 64
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.args['expected_natural_claim_sha256'] = hashlib.sha256(self.claim.read_bytes()).hexdigest()
        self.args['expected_natural_binding_sha256'] = '0' * 64
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.args['expected_natural_binding_sha256'] = hashlib.sha256(self.binding.read_bytes()).hexdigest()
        self.args['expected_natural_audit_sha256'] = '0' * 64
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.assertFalse(self.db.exists())

    def test_changed_natural_journal_or_tensor_fails_before_import(self):
        event = self.out / 'owner' / 'attempt' / 'events.jsonl'
        original = event.read_bytes()
        event.write_bytes(b'X' * len(original))
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        event.write_bytes(original)
        tensor = next((self.out / 'owner' / 'attempt' / 'tensors').iterdir())
        tensor.write_bytes(b'X')
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.assertFalse(self.db.exists())

    def test_rewritten_owner_and_terminal_cannot_rebind_the_audit(self):
        owner_path = self.out / 'owner' / 'supervisor.json'
        owner = json.loads(owner_path.read_text())
        owner['inventory']['tensor_sha256'] = ['b' * 64]
        owner_bytes = _write(owner_path, owner)
        terminal_path = self.out / 'launcher-terminal.json'
        terminal = json.loads(terminal_path.read_text())
        terminal['owner_supervisor_sha256'] = hashlib.sha256(owner_bytes).hexdigest()
        _write(terminal_path, terminal)
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.assertFalse(self.db.exists())

    def test_audit_changed_between_pin_and_inventory_is_rejected(self):
        scan = ledger._files
        def changed(roots):
            self.audit.write_bytes(b'changed after pinned read')
            return scan(roots)
        with patch.object(ledger, '_files', side_effect=changed):
            with self.assertRaises(ledger.LedgerError):
                self.initialize()
        self.assertFalse(self.db.exists())

    def test_changed_or_omitted_evidence_blocks_grants(self):
        self.initialize()
        (self.out / 'late-orphan').write_bytes(b'not in reconciled inventory')
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase()
        (self.out / 'late-orphan').unlink()
        self.preflight.write_bytes(b'altered')
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase()

    def test_prior_and_natural_are_not_double_counted(self):
        self.args['prior_paths'] = (self.preflight, self.extra)
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.assertFalse(self.db.exists())

    def test_no_free_disk_or_overlapping_output(self):
        self.initialize()
        with patch.object(ledger.shutil, 'disk_usage', return_value=type('D', (), {'free': 1})()):
            with self.assertRaises(ledger.LedgerError):
                self.claim_phase()
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase(name='natural/new-attempt')
        self.assertIsNotNone(self.claim_phase())

    def test_disk_is_rechecked_after_inventory_before_commit(self):
        self.initialize()
        ample = type('D', (), {'free': ledger.STUDY_DISK_LIMIT + 1})()
        low = type('D', (), {'free': ledger.STUDY_DISK_LIMIT - 1})()
        with patch.object(ledger.shutil, 'disk_usage', side_effect=[ample, low]):
            with self.assertRaises(ledger.LedgerError):
                self.claim_phase()
        self.assertIsNotNone(self.claim_phase())

    def test_fixed_partition_and_prior_bounds_fail_closed(self):
        self.args['prior_unique_tensor_bytes'] = ledger.STUDY_LOGICAL_LIMIT
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.args['prior_unique_tensor_bytes'] = ledger.PREFLIGHT_BYTES
        with patch.object(ledger, 'LEDGER_DISK_RESERVE', ledger.STUDY_DISK_LIMIT):
            with self.assertRaises(ledger.LedgerError):
                self.initialize()
        self.assertFalse(self.db.exists())

    def test_natural_unique_and_physical_copies_have_separate_limits(self):
        # A small synthetic logical share makes this an inexpensive regression:
        # the natural originals and copies exceed it physically but contain
        # one unique tensor, so check copies against the physical share.
        limits = dict(ledger.PHASE_LIMITS)
        limits['natural_discovery'] = (1, 224 * 1024**2)
        with patch.object(ledger, 'PHASE_LIMITS', limits):
            self.budget['phase_reservations_bytes']['natural_discovery'] = 1
            binding_bytes = _write(self.binding, {'study_budget': self.budget, 'independent_natural_audit_source_sha256': self.audit_source_hash})
            claim_bytes = _write(self.claim, {'schema': 'mlg-natural-study-claim-v1', 'budget': self.budget,
                                              'binding_sha256': hashlib.sha256(binding_bytes).hexdigest(),
                                              'output': str(self.out)})
            self.args['expected_natural_claim_sha256'] = hashlib.sha256(claim_bytes).hexdigest()
            self.args['expected_natural_binding_sha256'] = hashlib.sha256(binding_bytes).hexdigest()
            self.initialize()
            with closing(sqlite3.connect(self.db)) as db:
                meta = json.loads(db.execute('SELECT data FROM meta').fetchone()[0])
            self.assertGreater(meta['natural_physical_bytes'], 1)
            self.assertEqual(meta['natural_unique_tensor_bytes'], 1)
            self.assertIsNotNone(self.claim_phase())

    def test_natural_unique_subtotal_cannot_undercut_owner_or_exceed_share(self):
        self.args['natural_unique_tensor_bytes'] = ledger.PHASE_LIMITS['natural_discovery'][0] + 1
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.args['natural_unique_tensor_bytes'] = 0
        owner_path = self.out / 'owner' / 'supervisor.json'
        owner = json.loads(owner_path.read_text())
        owner['inventory']['retained_tensor_bytes'] = 2
        owner_bytes = _write(owner_path, owner)
        terminal_path = self.out / 'launcher-terminal.json'
        terminal = json.loads(terminal_path.read_text())
        terminal['owner_supervisor_sha256'] = hashlib.sha256(owner_bytes).hexdigest()
        terminal_bytes = _write(terminal_path, terminal)
        self.args['expected_natural_terminal_sha256'] = hashlib.sha256(terminal_bytes).hexdigest()
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        self.assertFalse(self.db.exists())

    def test_incomplete_database_is_not_silently_reinitialized(self):
        self.db.touch()
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase()

    def test_old_ledger_schema_is_not_silently_upgraded(self):
        self.initialize()
        with closing(sqlite3.connect(self.db)) as db:
            meta = json.loads(db.execute('SELECT data FROM meta').fetchone()[0])
            meta['schema'] = 'mlg-agreement-phase-ledger-v1'
            db.execute('UPDATE meta SET data=? WHERE id=1', (json.dumps(meta),))
            db.commit()
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase()
        with self.assertRaises(ledger.LedgerError):
            self.initialize()

    def test_initialization_interrupt_keeps_exclusive_file(self):
        with patch.object(ledger, '_connect', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.initialize()
        self.assertTrue(self.db.is_file())
        with self.assertRaises(ledger.LedgerError):
            self.initialize()
        with self.assertRaises(ledger.LedgerError):
            self.claim_phase()

    def test_inventory_stops_at_bounds_and_rejects_file_growth(self):
        many = self.root / 'many'
        many.mkdir()
        for i in range(5):
            (many / str(i)).touch()
        with patch.object(ledger, '_MAX_FILES', 3):
            with self.assertRaises(ledger.LedgerError):
                ledger._files([many])
        short = self.root / 'short'
        short.write_bytes(b'a')
        with patch.object(Path, 'open', return_value=io.BytesIO(b'ab')):
            with self.assertRaises(ledger.LedgerError):
                ledger._files([short])


if __name__ == '__main__':
    unittest.main()
