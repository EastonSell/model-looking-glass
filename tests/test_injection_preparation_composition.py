"""Fabricated cross-component handoff; no study claim, site or model."""
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from unittest import mock

import pytest

from looking_glass import injection_phase_accounting as accounting
from looking_glass import injection_preclaim_guard as guard
from looking_glass import injection_split_custody as custody
from tools import injection_preparation_operation as operation
from tools import injection_replay_claim as claim
from tools import injection_runtime_replay_child as replay_child
from tools import injection_runtime_replay_owner as replay_owner


@pytest.mark.parametrize('failure', (None, 'receipt', 'diagnostic'))
def test_real_receipt_guard_and_accounting_compose_before_fabricated_draw(tmp_path, failure):
    fixture_path = Path(__file__).with_name('test_injection_split_custody.py')
    spec = importlib.util.spec_from_file_location('_injection_composition_fixtures', fixture_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    fixture = fixtures.InjectionCustodyTests('test_claim_precedes_one_draw_and_private_commitment_verifies')
    fixture.setUp()
    try:
        public = fixture.v3_public()
        workspace = tmp_path / 'workspace'
        work = workspace / 'work'
        parent = work / 'number-injection-heldout-cpu-v1'
        diagnostic = work / 'injection-heldout-runtime-rehearsal-v1'
        ledger = work / 'agreement-study-ledger-v5' / 'ledger.sqlite3'
        parent.mkdir(parents=True)
        diagnostic.mkdir()
        ledger.parent.mkdir()
        (diagnostic / 'diagnostic.bin').write_bytes(b'fabricated diagnostic')
        phase = parent / ('injection-heldout-' + fixture.freeze.sha256)
        pilot = parent / 'runtime-compatibility-replay-v1'
        pilot.mkdir()
        (pilot / 'replay.bin').write_bytes(b'fabricated replay')

        original_rows = [('natural_discovery', str(work / 'natural-discovery'),
                          128 * 1024**2, 4096 * 1024**2, 0),
                         ('injection_discovery', str(work / 'injection-discovery'),
                          48 * 1024**2, 128 * 1024**2, 13 * 1024**3)]
        row = ('injection_heldout', str(phase), 72 * 1024**2, 192 * 1024**2, 13 * 1024**3)
        history = fixtures.digest(b'reconciliations:present\nretired_evidence:present\n')
        scope = '1' * 64
        meta = fixtures.canonical({'schema': 'mlg-agreement-phase-ledger-v6',
            'current_history_sha256': history, 'evidence_scope_amendment_sha256': scope}).decode()
        inventory = '{"prior":{},"natural":{}}'
        with closing(sqlite3.connect(ledger)) as db:
            db.execute('CREATE TABLE meta (id INTEGER PRIMARY KEY, data TEXT NOT NULL, inventory TEXT NOT NULL)')
            db.execute('CREATE TABLE claims (phase TEXT PRIMARY KEY, output TEXT NOT NULL, logical INTEGER NOT NULL, disk INTEGER NOT NULL, free INTEGER NOT NULL)')
            db.execute('CREATE TABLE reconciliations (seq INTEGER, manifest_sha TEXT, gap_sha TEXT, claims_sha TEXT, previous_meta TEXT, previous_inventory TEXT)')
            db.execute('CREATE TABLE retired_evidence (side TEXT, path_sha TEXT, size INTEGER, content_sha TEXT)')
            db.execute('INSERT INTO meta VALUES(1, ?, ?)', (meta, inventory))
            db.executemany('INSERT INTO claims VALUES(?, ?, ?, ?, ?)', original_rows + [row])
            db.commit()
        state = hashlib.sha256()
        for item in (meta.encode(), inventory.encode()):
            state.update(len(item).to_bytes(8, 'big'))
            state.update(item)
        identity = lambda path: (path.stat().st_dev, path.stat().st_ino)
        parent_id, ledger_id = identity(parent), identity(ledger)
        anchor = guard.ClaimAnchor(str(ledger), *ledger_id, str(phase), state.hexdigest(), history, scope,
            fixtures.digest(json.dumps(sorted(original_rows), ensure_ascii=True, separators=(',', ':')).encode()),
            fixtures.digest(json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')))
        plan_sha = fixtures.digest(public.runtime_replay_plan_bytes)
        count, diagnostic_sha = claim._diagnostic_bytes(diagnostic)
        intent = {'schema': 'mlg-injection-replay-claim-intent-v1', 'plan_sha256': plan_sha,
                  'ledger_path': str(ledger), 'phase_path': str(phase), 'pilot_path': str(pilot),
                  'diagnostic_path': str(diagnostic), 'parent_identity': list(parent_id),
                  'ledger_identity': list(ledger_id), 'diagnostic_actual_bytes': count,
                  'diagnostic_inventory_sha256': diagnostic_sha}
        intent_raw = fixtures.canonical(intent)
        (parent / 'replay-claim-intent.json').write_bytes(intent_raw)
        receipt = {'schema': 'mlg-injection-replay-claim-receipt-v1', 'status': 'preclaimed',
                   'plan_sha256': plan_sha, 'intent_sha256': fixtures.digest(intent_raw),
                   'ledger_path': str(ledger), 'phase_path': str(phase), 'pilot_path': str(pilot),
                   'diagnostic_path': str(diagnostic), 'parent_identity': list(parent_id),
                   'ledger_identity': list(ledger_id), **replay_child._BUDGETS,
                   'diagnostic_actual_bytes': count, 'diagnostic_inventory_sha256': diagnostic_sha,
                   'injection_claim_row_sha256': anchor.heldout_claim_sha256}
        receipt_raw = fixtures.canonical(receipt)
        (parent / 'replay-claim-receipt.json').write_bytes(receipt_raw)
        config = {'attempt_dir': str(pilot / 'attempt'), 'postcheck_file': str(pilot / 'child-postcheck.json'),
                  'claim_receipt_file': str(parent / 'replay-claim-receipt.json')}
        config_raw = fixtures.canonical(config)
        handoff = operation.ClaimHandoff(str(workspace), config_raw, plan_sha, fixtures.digest(receipt_raw),
                                         parent_id, ledger_id)
        roots = accounting.PhaseRoots(str(parent), *parent_id, str(phase), str(pilot), *identity(pilot),
                                      str(diagnostic), *identity(diagnostic))
        pins = {'_STUDY_FREEZE': fixture.freeze.sha256, '_STUDY_SOURCE': fixture.source,
                '_STUDY_RUNTIME': fixtures.digest(fixture.runtime),
                '_STUDY_PREDICTION': fixtures.digest(fixture.prediction),
                '_STUDY_PLAN': fixtures.digest(fixture.plan),
                '_STUDY_SITE_COMMITMENT': fixture.site.commitment_sha256}
        if failure == 'receipt':
            (parent / 'replay-claim-receipt.json').write_bytes(b'changed')
        elif failure == 'diagnostic':
            (diagnostic / 'diagnostic.bin').write_bytes(b'changed')

        def draw(size):
            assert size == 32
            with closing(sqlite3.connect(ledger, timeout=0, isolation_level=None)) as other:
                try:
                    other.execute('BEGIN IMMEDIATE')
                except sqlite3.OperationalError:
                    pass
                else:
                    raise AssertionError('competing writer entered during the draw')
            return fixtures.SPLIT_NONCE

        with mock.patch.multiple(custody, **pins), mock.patch.object(replay_child, '_FREEZE', fixture.freeze.sha256), \
             mock.patch.object(replay_owner, 'verify_private_parent', return_value=True), \
             mock.patch.object(custody.secrets, 'token_bytes', side_effect=draw) as draw_mock:
            options = dict(
                custody=custody, guard=guard, accounting=accounting, replay_owner=replay_owner,
                replay_child=replay_child, replay_claim=claim, public=public, private_parent=parent,
                ledger_path=ledger, anchor=anchor, roots=roots, handoff=handoff, replay_tensor_bytes=1024,
                config_bytes=b'fabricated-preparation-config', site_nonce=fixtures.SITE_NONCE,
                site_record_bytes=fixture.site.record_bytes, tokenizer=fixtures.FakeTokenizer(),
                observe_claimant_exit=lambda value: value == fixtures.digest(receipt_raw),
                observe_replay_config=lambda: config_raw,
                verify_replay_evidence=lambda plan, result, size: (
                    plan == public.runtime_replay_plan_bytes and result == public.runtime_replay_evidence_bytes
                    and size == 1024),
                publication_check=lambda name, raw: True, source_check=lambda binding: True,
                prediction_check=lambda raw, sha: True, gate_allocation=lambda freeze: True,
                observe_inventory=lambda: fixture.inventory_sha,
                runtime_check=lambda runtime, deviation, plan, result, binding: True)
            if failure is not None:
                with pytest.raises((ValueError, replay_owner.ReplayOwnerError)):
                    operation.prepare_from_handoff(**options)
                draw_mock.assert_not_called()
                assert not phase.exists()
                assert not (parent / (phase.name + '.preparation-started.json')).exists()
                return
            prepared = operation.prepare_from_handoff(**options)
        assert prepared.phase_root == phase
        assert (phase / 'preparation-complete.json').is_file()
        with closing(sqlite3.connect(ledger, timeout=0, isolation_level=None)) as other:
            other.execute('BEGIN IMMEDIATE')
            other.rollback()
    finally:
        fixture.doCleanups()
