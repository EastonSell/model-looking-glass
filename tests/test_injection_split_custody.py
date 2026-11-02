"""Synthetic-only tests: no real split, checkpoint, model or site."""
import hashlib
import importlib
import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from contextlib import contextmanager

from looking_glass import pretrained_exposures_v3 as exposures
from looking_glass import pretrained_number_commitments as commitments

try:
    custody = importlib.import_module('looking_glass.injection_split_custody')
except ModuleNotFoundError as error:
    if error.name != 'looking_glass.injection_split_custody':
        raise
    custody = None


NOUNS = ('alpha/alphas', 'beta/betas', 'gamma/gammas', 'delta/deltas',
         'epsilon/epsilons', 'zeta/zetas', 'eta/etas', 'theta/thetas')
PLURAL = ('near the boxes', 'behind the cars', 'above the lamps', 'below the desks')
SINGULAR = ('near the box', 'behind the car', 'above the lamp', 'below the desk')
SUBJECTS = [form for pair in NOUNS for form in pair.split('/')]
CONTEXTS = list(PLURAL + SINGULAR)
SITE_NONCE = bytes(reversed(range(32)))
SPLIT_NONCE = bytes(range(32))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


@contextmanager
def simulated_protection(*args, **kwargs):
    yield True


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == 'The':
            return [10]
        for context in CONTEXTS:
            if text == ' ' + context:
                return [21000 + CONTEXTS.index(context)]
        for subject in SUBJECTS:
            prefix = 'The ' + subject
            ids = [10, 20000 + SUBJECTS.index(subject)]
            if text == prefix:
                return ids
            for context in CONTEXTS:
                whole = prefix + ' ' + context
                if text == whole:
                    return ids + [21000 + CONTEXTS.index(context)]
                for suffix, target in ((' is', 318), (' are', 389), (' was', 373), (' were', 547)):
                    if text == whole + suffix:
                        return ids + [21000 + CONTEXTS.index(context), target]
        raise ValueError('unexpected fabricated text')


class OneClaim:
    def __init__(self):
        self.claimed = False
        self.events = []

    def __call__(self, ledger_path, phase, *, output_path):
        if self.claimed:
            raise ValueError('phase already claimed')
        self.claimed = True
        self.events.append('claim')
        return SimpleNamespace(phase=phase, output_path=output_path,
                               logical_limit_bytes=72 * 1024**2, disk_limit_bytes=192 * 1024**2)


class InjectionCustodyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(custody, 'injection split custody is not implemented')
        patch = mock.patch.multiple(commitments, NOUNS=NOUNS, PLURAL_CONTEXTS=PLURAL,
                                    SINGULAR_CONTEXTS=SINGULAR)
        patch.start()
        self.addCleanup(patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = 'd' * 40
        self.runtime = b'fabricated runtime'
        self.prediction = b'fabricated prediction'
        self.freeze = commitments.make_prediction_freeze({'injection': {
            'source_commit': self.source, 'runtime_sha256': digest(self.runtime),
            'prediction_sha256': digest(self.prediction)}})
        self.site = commitments.seal_site(SITE_NONCE, self.source)
        self.inventory = exposures.canonical_inventory_bytes()
        self.inventory_sha = exposures.inventory_digest()
        self.plan = canonical({'schema': 'mlg-number-injection-family-request-plan-v2',
                               'protocol_sha256': commitments.PROTOCOL_SHA256,
                               'checkpoint_revision': commitments.CHECKPOINT_REVISION,
                               'source_commit': self.source, 'public_entropy': '33' * 32,
                               'requests': []})
        self.operational = canonical({'schema': 'mlg-injection-heldout-operational-binding-v1',
            'protocol_sha256': commitments.PROTOCOL_SHA256,
            'checkpoint_revision': commitments.CHECKPOINT_REVISION,
            'discovery_source_commit': self.source, 'prediction_freeze_sha256': self.freeze.sha256,
            'runtime_sha256': digest(self.runtime), 'prediction_sha256': digest(self.prediction),
            'plan_sha256': digest(self.plan), 'site_commitment': self.site.commitment_sha256,
            'inventory_sha256': self.inventory_sha,
            'custody_source_sha256': digest(Path(custody.__file__).read_bytes()),
            'executor_source_sha256': '2' * 64, 'comparator_source_sha256': '3' * 64,
            'ledger_source_sha256': '4' * 64, 'tokenizer_sha256': '5' * 64,
            'supplement_sha256': '6' * 64, 'owner_source_sha256': '7' * 64,
            'journal_source_sha256': '8' * 64, 'supervisor_source_sha256': '9' * 64,
            'auditor_source_sha256': 'a' * 64, 'gate_source_sha256': 'b' * 64,
            'inventory_source_sha256': 'c' * 64, 'inventory_source_commit': 'e' * 40,
            'lease_source_sha256': 'f' * 64,
            'accounting_amendment_sha256': 'd' * 64})
        self.inputs = custody.PublicInputs(
            freeze_bytes=self.freeze.record_bytes, freeze_sha256=self.freeze.sha256,
            runtime_bytes=self.runtime, prediction_bytes=self.prediction,
            plan_bytes=self.plan, plan_sha256=digest(self.plan),
            site_commitment=self.site.commitment_sha256, inventory_bytes=self.inventory,
            inventory_sha256=self.inventory_sha, operational_binding_bytes=self.operational,
            operational_binding_sha256=digest(self.operational))
        self.claim = OneClaim()
        self.published = []

    def prepare(self, *, nonce=SPLIT_NONCE, tokenizer=None, observe=None,
                source_check=lambda binding: True, prediction_check=lambda raw, sha: True):
        def observe_public(name, expected):
            self.published.append(name)
            return True
        def draw(size):
            self.claim.events.append('draw')
            self.assertEqual(size, 32)
            return nonce
        with mock.patch.object(custody.secrets, 'token_bytes', side_effect=draw):
            return custody.prepare_once(self.inputs, self.root, self.root / 'fake-ledger',
                site_nonce=SITE_NONCE, site_record_bytes=self.site.record_bytes,
                tokenizer=tokenizer or FakeTokenizer(), claim_phase=self.claim,
                publication_check=observe_public,
                source_check=source_check, prediction_check=prediction_check,
                gate_allocation=lambda freeze: self.claim.events.append('gate') or True,
                observe_inventory=observe or (lambda: self.inventory_sha))

    def v2_public(self):
        binding = json.loads(self.operational)
        binding.update({'schema': 'mlg-injection-heldout-operational-binding-v2',
            'preparation_child_source_sha256': '1' * 64,
            'preparation_audit_source_sha256': '2' * 64,
            'outer_observer_source_sha256': '3' * 64,
            'gate_allocation_caller_source_sha256': '4' * 64,
            'gate_transition_caller_source_sha256': '5' * 64})
        raw = canonical(binding)
        return replace(self.inputs, operational_binding_bytes=raw, operational_binding_sha256=digest(raw))

    def v3_public(self):
        binding = json.loads(self.v2_public().operational_binding_bytes)
        heldout_runtime = canonical({'schema': 'fabricated-heldout-runtime-v1', 'system_environment_sha256': '7' * 64})
        deviation = b'fabricated published runtime deviation'
        replay_plan = canonical({'schema': 'fabricated-compatibility-plan-v1', 'fixed_roles': 16})
        replay = canonical({'schema': 'fabricated-compatibility-replay-v1', 'status': 'passed'})
        binding.update({'schema': 'mlg-injection-heldout-operational-binding-v3',
            'runtime_deviation_sha256': digest(deviation), 'heldout_runtime_sha256': digest(heldout_runtime),
            'runtime_replay_plan_sha256': digest(replay_plan), 'runtime_replay_source_sha256': '6' * 64,
            'runtime_replay_claim_source_sha256': '7' * 64, 'runtime_replay_owner_source_sha256': '8' * 64,
            'preclaim_guard_source_sha256': '9' * 64, 'phase_accounting_source_sha256': 'a' * 64,
            'runtime_replay_evidence_sha256': digest(replay)})
        raw = canonical(binding)
        return replace(self.inputs, operational_binding_bytes=raw, operational_binding_sha256=digest(raw),
                       runtime_deviation_bytes=deviation, heldout_runtime_bytes=heldout_runtime,
                       runtime_replay_plan_bytes=replay_plan,
                       runtime_replay_evidence_bytes=replay)

    def fixed_options(self):
        config = canonical({'schema': 'fabricated-preparation-config-v1'})
        return dict(site_nonce=SITE_NONCE, site_record_bytes=self.site.record_bytes,
            tokenizer=FakeTokenizer(), verify_preclaim=lambda path, phase, **kw: True,
            protect_preclaim=simulated_protection,
            reconcile_phase=lambda path, **kw: True, config_bytes=config, config_sha256=digest(config),
            publication_check=lambda name, raw: True, source_check=lambda binding: True,
            prediction_check=lambda raw, sha: True, gate_allocation=lambda freeze: True,
            observe_inventory=lambda: self.inventory_sha,
            runtime_check=lambda runtime, deviation, plan, replay, binding: True)

    def test_claim_precedes_one_draw_and_private_commitment_verifies(self):
        prepared = self.prepare()
        self.assertEqual(self.claim.events, ['claim', 'gate', 'draw'])
        self.assertEqual(set(self.published), {'freeze', 'runtime', 'prediction', 'plan',
                                               'site_commitment', 'inventory', 'operational_binding'})
        phase = prepared.phase_root
        self.assertEqual((phase / 'nonce.bin').read_bytes(), SPLIT_NONCE)
        record = (phase / 'heldout-record.json').read_bytes()
        public = json.loads((phase / 'commitment.json').read_bytes())
        self.assertNotIn('nonce', public)
        self.assertNotIn('families', public)
        self.assertTrue(commitments.verify_heldout_reveal(self.freeze, SPLIT_NONCE, record,
            public['heldout_commitment'], inventory_snapshot=commitments.load_inventory_snapshot(
                self.inventory, self.inventory_sha), site_nonce=SITE_NONCE,
            site_record_bytes=self.site.record_bytes, site_commitment=self.site.commitment_sha256))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_reused_public_entropy_aborts_after_single_consumed_draw(self):
        with self.assertRaises(custody.CustodyError):
            self.prepare(nonce=bytes.fromhex('33' * 32))
        self.assertEqual(self.claim.events, ['claim', 'gate', 'draw'])
        phase = self.root / ('injection-heldout-' + self.freeze.sha256)
        self.assertTrue((phase / 'preparation-failure.json').is_file())
        self.assertFalse((phase / 'commitment.json').exists())
        with self.assertRaises(ValueError):
            self.prepare()

    def test_changed_inventory_stops_before_claim_or_nonce(self):
        with self.assertRaises(custody.CustodyError):
            self.prepare(observe=lambda: '0' * 64)
        self.assertEqual(self.claim.events, [])

    def test_unverified_source_or_prediction_stops_before_claim(self):
        for options in ({'source_check': lambda binding: False},
                        {'prediction_check': lambda raw, sha: False}):
            with self.subTest(options=tuple(options)), self.assertRaises(custody.CustodyError):
                self.prepare(**options)
            self.assertEqual(self.claim.events, [])

    def test_real_study_entry_rejects_a_self_consistent_alternate_freeze(self):
        self.assertTrue(hasattr(custody, 'prepare_study_once'), 'fixed-study entry is not implemented')
        with self.assertRaises(custody.CustodyError):
            custody.prepare_study_once(self.v3_public(), self.root, self.root / 'fake-ledger',
                                       **self.fixed_options())
        self.assertEqual(self.claim.events, [])

    def test_fixed_study_requires_v2_preparation_sources_before_claim(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        options = self.fixed_options()
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', return_value=SPLIT_NONCE):
            with self.assertRaises(ValueError):
                custody.prepare_study_once(self.inputs, self.root, self.root / 'fake-ledger', **options)
            self.assertEqual(self.claim.events, [])
            for field in ('preparation_child_source_sha256', 'preparation_audit_source_sha256',
                          'outer_observer_source_sha256', 'gate_allocation_caller_source_sha256',
                          'gate_transition_caller_source_sha256', 'runtime_replay_claim_source_sha256',
                          'runtime_replay_owner_source_sha256', 'preclaim_guard_source_sha256',
                          'phase_accounting_source_sha256'):
                with self.subTest(field=field):
                    altered = json.loads(self.v3_public().operational_binding_bytes)
                    altered[field] = 'NOT-A-SHA256'
                    raw = canonical(altered)
                    bad = replace(self.v3_public(), operational_binding_bytes=raw, operational_binding_sha256=digest(raw))
                    with tempfile.TemporaryDirectory(dir=self.root) as temp:
                        with self.assertRaises(ValueError):
                            custody.prepare_study_once(bad, Path(temp), self.root / 'fake-ledger', **options)
                        self.assertTrue((Path(temp) / ('injection-heldout-' + self.freeze.sha256
                                                       + '.preparation-started.json')).exists())
                    self.assertEqual(self.claim.events, [])
            result = custody.prepare_study_once(self.v3_public(), self.root, self.root / 'fake-ledger', **options)
            self.assertTrue((result.phase_root / 'preparation-complete.json').is_file())
            self.assertEqual(self.claim.events, [])

    def test_fixed_study_requires_published_runtime_deviation_before_claim(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        options = self.fixed_options()
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', return_value=SPLIT_NONCE):
            with self.assertRaises(ValueError):
                custody.prepare_study_once(self.v2_public(), self.root, self.root / 'fake-ledger', **options)
            self.assertEqual(self.claim.events, [])
            public = self.v3_public()
            for name in ('runtime_deviation', 'heldout_runtime', 'runtime_replay_plan', 'runtime_replay_evidence'):
                with self.subTest(name=name):
                    options['publication_check'] = lambda key, raw, refused=name: key != refused
                    with tempfile.TemporaryDirectory(dir=self.root) as temp:
                        with self.assertRaises(ValueError):
                            custody.prepare_study_once(public, Path(temp), self.root / 'fake-ledger', **options)
                        self.assertTrue((Path(temp) / ('injection-heldout-' + self.freeze.sha256
                                                       + '.preparation-started.json')).exists())
                    self.assertEqual(self.claim.events, [])
            options['publication_check'] = lambda name, raw: True
            options['runtime_check'] = lambda runtime, deviation, plan, replay, binding: False
            with tempfile.TemporaryDirectory(dir=self.root) as temp:
                with self.assertRaises(ValueError):
                    custody.prepare_study_once(public, Path(temp), self.root / 'fake-ledger', **options)
            self.assertEqual(self.claim.events, [])
            options['runtime_check'] = lambda runtime, deviation, plan, replay, binding: True
            result = custody.prepare_study_once(public, self.root, self.root / 'fake-ledger', **options)
            self.assertTrue((result.phase_root / 'preparation-complete.json').exists())

    def test_preclaimed_study_uses_one_durable_handoff_without_reclaiming(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        phase = self.root / ('injection-heldout-' + self.freeze.sha256)
        ledger = self.root / 'fake-ledger'
        calls = []
        def verify_preclaim(path, phase_name, *, output_path):
            calls.append('verify')
            return path == ledger and phase_name == 'injection_heldout' and output_path == phase
        def reconcile_phase(output_path, *, pilot_logical_bytes, pilot_disk_bytes, rehearsal_disk_bytes, stage):
            calls.append('reconcile:' + stage)
            return (output_path == phase and pilot_logical_bytes == 8 * 1024**2
                    and pilot_disk_bytes == 16 * 1024**2 and rehearsal_disk_bytes == 4 * 1024**2)
        options = dict(site_nonce=SITE_NONCE, site_record_bytes=self.site.record_bytes,
            tokenizer=FakeTokenizer(), verify_preclaim=verify_preclaim, protect_preclaim=simulated_protection,
            reconcile_phase=reconcile_phase,
            publication_check=lambda name, raw: True, source_check=lambda binding: True,
            prediction_check=lambda raw, sha: True, gate_allocation=lambda freeze: True,
            observe_inventory=lambda: self.inventory_sha,
            runtime_check=lambda runtime, deviation, plan, replay, binding: True,
            config_bytes=b'fabricated-config', config_sha256=digest(b'fabricated-config'))
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', return_value=SPLIT_NONCE) as draw:
            result = custody.prepare_study_preclaimed(self.v3_public(), self.root, ledger, **options)
            marker = self.root / (phase.name + '.preparation-started.json')
            self.assertTrue(marker.is_file())
            claim = json.loads((result.phase_root / 'claim.json').read_bytes())
            self.assertEqual(claim['preclaim_handoff_sha256'], digest(marker.read_bytes()))
            self.assertIn('verify', calls)
            self.assertEqual([item for item in calls if item.startswith('reconcile:')],
                             ['reconcile:before_phase_creation', 'reconcile:before_nonce'])
            with self.assertRaises(ValueError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root, ledger, **options)
            self.assertEqual(draw.call_count, 1)
            self.assertFalse(self.claim.claimed)

    def test_v3_rejects_altered_replay_plan_and_result_before_draw(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
            for field in ('runtime_replay_plan_bytes', 'runtime_replay_evidence_bytes'):
                with self.subTest(field=field), tempfile.TemporaryDirectory(dir=self.root) as temp:
                    bad = replace(self.v3_public(), **{field: b'altered published bytes'})
                    with self.assertRaises(custody.CustodyError):
                        custody.prepare_study_preclaimed(bad, Path(temp), self.root / 'fake-ledger',
                                                         **self.fixed_options())
            draw.assert_not_called()

    def test_preclaim_or_phase_reconciliation_failure_is_terminal_before_draw(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
            for failed_callback in ('verify_preclaim', 'reconcile_phase'):
                with self.subTest(callback=failed_callback), tempfile.TemporaryDirectory(dir=self.root) as temp:
                    parent = Path(temp)
                    options = self.fixed_options()
                    options[failed_callback] = lambda *args, **kwargs: False
                    for _ in range(2):
                        with self.assertRaises(custody.CustodyError):
                            custody.prepare_study_preclaimed(self.v3_public(), parent,
                                                              self.root / 'fake-ledger', **options)
                    phase = parent / ('injection-heldout-' + self.freeze.sha256)
                    self.assertFalse(phase.exists())
                    self.assertTrue((parent / (phase.name + '.preparation-started.json')).is_file())
            draw.assert_not_called()

    def test_interrupted_handoff_is_retained_and_cannot_be_retried(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        def interrupted_write(path, data):
            path.write_bytes(data[:13])
            raise KeyboardInterrupt()
        options = self.fixed_options()
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
            with mock.patch.object(custody, '_write_new', side_effect=interrupted_write):
                with self.assertRaises(KeyboardInterrupt):
                    custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                      self.root / 'fake-ledger', **options)
            marker = self.root / ('injection-heldout-' + self.freeze.sha256 + '.preparation-started.json')
            self.assertEqual(marker.stat().st_size, 13)
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                  self.root / 'fake-ledger', **options)
            draw.assert_not_called()

    def test_failed_marker_open_cannot_be_retried_in_the_same_process(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        options = self.fixed_options()
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
            with mock.patch.object(custody, '_write_new', side_effect=OSError('fabricated open failure')):
                with self.assertRaises(OSError):
                    custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                      self.root / 'fake-ledger', **options)
            marker = self.root / ('injection-heldout-' + self.freeze.sha256 + '.preparation-started.json')
            self.assertFalse(marker.exists())
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                  self.root / 'fake-ledger', **options)
            self.assertFalse(marker.exists())
            draw.assert_not_called()

    def test_mutated_or_hardlinked_handoff_cannot_authorize_preparation(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
            for modification in ('oversize', 'hardlink'):
                with self.subTest(modification=modification), tempfile.TemporaryDirectory(dir=self.root) as temp:
                    parent = Path(temp)
                    marker = parent / ('injection-heldout-' + self.freeze.sha256 + '.preparation-started.json')
                    def reconcile(*args, **kwargs):
                        if modification == 'oversize':
                            marker.write_bytes(b'x' * 8193)
                        else:
                            os.link(marker, parent / 'extra-link')
                        return True
                    options = self.fixed_options()
                    options['reconcile_phase'] = reconcile
                    with self.assertRaises(custody.CustodyError):
                        custody.prepare_study_preclaimed(self.v3_public(), parent,
                                                          self.root / 'fake-ledger', **options)
                    self.assertTrue(marker.exists())
            draw.assert_not_called()

    def test_runtime_drift_after_gate_stops_before_the_single_draw(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        options = self.fixed_options()
        checks = []
        options['runtime_check'] = lambda *args: checks.append('runtime') or len(checks) == 1
        options['gate_allocation'] = lambda freeze: checks.append('gate') or True
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                  self.root / 'fake-ledger', **options)
            phase = self.root / ('injection-heldout-' + self.freeze.sha256)
            self.assertEqual(checks, ['runtime', 'gate', 'runtime'])
            self.assertTrue((phase / 'preparation-failure.json').is_file())
            self.assertFalse((phase / 'nonce.bin').exists())
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                  self.root / 'fake-ledger', **options)
            draw.assert_not_called()

    def test_changed_handoff_or_ledger_after_gate_stops_before_draw(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
            for changed in ('marker', 'ledger', 'side_evidence', 'claim'):
                with self.subTest(changed=changed), tempfile.TemporaryDirectory(dir=self.root) as temp:
                    parent = Path(temp)
                    phase = parent / ('injection-heldout-' + self.freeze.sha256)
                    marker = parent / (phase.name + '.preparation-started.json')
                    state = {'after_gate': False}
                    options = self.fixed_options()
                    def gate(freeze):
                        state['after_gate'] = True
                        if changed == 'marker':
                            marker.write_bytes(marker.read_bytes() + b' ')
                        if changed == 'claim':
                            (phase / 'claim.json').write_bytes(b'fabricated-invalid')
                        return True
                    options['gate_allocation'] = gate
                    options['verify_preclaim'] = lambda *args, **kw: not (changed == 'ledger' and state['after_gate'])
                    options['reconcile_phase'] = lambda *args, **kw: not (changed == 'side_evidence' and state['after_gate'])
                    with self.assertRaises(custody.CustodyError):
                        custody.prepare_study_preclaimed(self.v3_public(), parent,
                                                          self.root / 'fake-ledger', **options)
                    self.assertTrue((phase / 'preparation-failure.json').is_file())
                    self.assertFalse((phase / 'nonce.bin').exists())
            draw.assert_not_called()

    def test_changed_claim_during_entropy_retains_nonce_and_stops(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        phase = self.root / ('injection-heldout-' + self.freeze.sha256)
        def draw(size):
            (phase / 'claim.json').write_bytes(b'fabricated-invalid')
            return SPLIT_NONCE
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', side_effect=draw) as entropy:
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root, self.root / 'fake-ledger',
                                                  **self.fixed_options())
            self.assertEqual((phase / 'nonce.bin').read_bytes(), SPLIT_NONCE)
            self.assertTrue((phase / 'preparation-failure.json').is_file())
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root, self.root / 'fake-ledger',
                                                  **self.fixed_options())
            self.assertEqual(entropy.call_count, 1)

    def test_nonce_is_drawn_only_while_preclaim_protection_is_held(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        state = {'held': False}
        @contextmanager
        def guard(*args, **kwargs):
            state['held'] = True
            try:
                yield True
            finally:
                state['held'] = False
        def draw(size):
            self.assertTrue(state['held'])
            return SPLIT_NONCE
        options = self.fixed_options()
        options['protect_preclaim'] = guard
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', side_effect=draw):
            result = custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                       self.root / 'fake-ledger', **options)
        self.assertFalse(state['held'])
        self.assertTrue((result.phase_root / 'preparation-complete.json').is_file())

    def test_uncertain_guard_release_retains_nonce_without_redrawing(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        @contextmanager
        def failed_release(*args, **kwargs):
            yield True
            raise OSError('fabricated uncertain transaction release')
        options = self.fixed_options()
        options['protect_preclaim'] = failed_release
        with mock.patch.multiple(custody, **pins), mock.patch.object(
                custody.secrets, 'token_bytes', return_value=SPLIT_NONCE) as draw:
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                  self.root / 'fake-ledger', **options)
            phase = self.root / ('injection-heldout-' + self.freeze.sha256)
            self.assertTrue((phase / 'preparation-failure.json').is_file())
            self.assertEqual((phase / 'nonce.bin').read_bytes(), SPLIT_NONCE)
            with self.assertRaises(custody.CustodyError):
                custody.prepare_study_preclaimed(self.v3_public(), self.root,
                                                  self.root / 'fake-ledger', **options)
            self.assertEqual(draw.call_count, 1)

    def test_preclaim_marker_binds_ledger_and_phase_path(self):
        public = self.v3_public()
        pins = {'_STUDY_FREEZE': self.freeze.sha256}
        with mock.patch.multiple(custody, **pins):
            record = custody._preclaim_record(public, self.root / 'ledger', self.root / 'phase', '1' * 64)
            changed_ledger = custody._preclaim_record(public, self.root / 'other-ledger', self.root / 'phase', '1' * 64)
            changed_phase = custody._preclaim_record(public, self.root / 'ledger', self.root / 'other-phase', '1' * 64)
        self.assertNotEqual(record['ledger_path_sha256'], changed_ledger['ledger_path_sha256'])
        self.assertNotEqual(record['phase_path_sha256'], changed_phase['phase_path_sha256'])

    def test_fixed_entry_accepts_only_the_independently_pinned_inputs(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', return_value=SPLIT_NONCE):
            options = self.fixed_options()
            options['gate_allocation'] = lambda freeze: self.claim.events.append('gate') or True
            prepared = custody.prepare_study_once(self.v3_public(), self.root, self.root / 'fake-ledger', **options)
        self.assertTrue((prepared.phase_root / 'preparation-complete.json').is_file())
        self.assertEqual(self.claim.events, ['gate'])

    def test_bad_continuation_aborts_without_sealing(self):
        class BadTokenizer(FakeTokenizer):
            def encode(self, text, *, add_special_tokens):
                ids = super().encode(text, add_special_tokens=add_special_tokens)
                return ids[:-1] + [42] if text.endswith(' were') else ids
        with self.assertRaises(custody.CustodyError):
            self.prepare(tokenizer=BadTokenizer())
        phase = self.root / ('injection-heldout-' + self.freeze.sha256)
        self.assertTrue((phase / 'preparation-failure.json').is_file())
        self.assertFalse((phase / 'commitment.json').exists())

    def test_known_full_row_prefixes_rejected_in_both_directions(self):
        cases = [[{'input_ids': [10, 100 + 4 * family + item, 200]}
                  for item in range(4)] for family in range(4)]
        for known in ([10, 100], [10, 100, 200, 300]):
            with self.subTest(known=known), self.assertRaises(custody.CustodyError):
                custody.reject_known_row_prefixes(cases, canonical({'records': [{'token_ids': known}]}))
        custody.reject_known_row_prefixes(cases, canonical({'records': [{'token_ids': [10, 999]}]}))

    def test_write_cancellation_survives_an_ordinary_close_failure(self):
        class FailingStream:
            def close(self):
                raise OSError('fabricated close failure')
            def write(self, data):
                raise KeyboardInterrupt('fabricated cancellation')
        with mock.patch.object(Path, 'open', return_value=FailingStream()):
            with self.assertRaises(KeyboardInterrupt):
                custody._write_new(self.root / 'fault', b'x')

    def test_postclaim_inventory_change_consumes_claim_without_a_draw(self):
        observed = iter([self.inventory_sha, '0' * 64])
        with self.assertRaises(custody.CustodyError):
            self.prepare(observe=lambda: next(observed))
        self.assertEqual(self.claim.events, ['claim'])
        phase = self.root / ('injection-heldout-' + self.freeze.sha256)
        self.assertTrue((phase / 'preparation-failure.json').is_file())
        self.assertFalse((phase / 'nonce.bin').exists())

    def test_postgate_inventory_change_stops_before_the_nonce(self):
        observed = iter([self.inventory_sha, self.inventory_sha, '0' * 64])
        with self.assertRaises(custody.CustodyError):
            self.prepare(observe=lambda: next(observed))
        self.assertEqual(self.claim.events, ['claim', 'gate'])
        phase = self.root / ('injection-heldout-' + self.freeze.sha256)
        self.assertTrue((phase / 'preparation-failure.json').is_file())
        self.assertFalse((phase / 'nonce.bin').exists())

    def test_private_tokenizer_failure_is_sanitized(self):
        class SecretErrorTokenizer(FakeTokenizer):
            def encode(self, text, *, add_special_tokens):
                raise ValueError('FABRICATED_PRIVATE_TEXT')
        with self.assertRaises(custody.CustodyError) as raised:
            self.prepare(tokenizer=SecretErrorTokenizer())
        self.assertNotIn('FABRICATED_PRIVATE_TEXT', str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)

    def test_partial_record_write_is_not_accepted(self):
        class ShortStream:
            def write(self, data):
                return 0
            def flush(self):
                pass
            def fileno(self):
                return 3
            def close(self):
                pass
        with mock.patch.object(Path, 'open', return_value=ShortStream()), mock.patch.object(custody.os, 'fsync'):
            with self.assertRaises(OSError):
                custody._write_new(self.root / 'fault', b'abc')

    def test_fatal_cancellation_during_failure_record_is_not_swallowed(self):
        class BadTokenizer(FakeTokenizer):
            def encode(self, text, *, add_special_tokens):
                raise ValueError('fabricated ordinary failure')
        real_write = custody._write_new
        def interrupted(path, data):
            if path.name == 'preparation-failure.json':
                raise KeyboardInterrupt('fabricated cancellation')
            return real_write(path, data)
        with mock.patch.object(custody, '_write_new', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare(tokenizer=BadTokenizer())
        self.assertTrue(self.claim.claimed)

    def _preparation_wiring(self, *, claimant_exit=True, replay_ok=True, change_after_gate=None,
                            wrong_claim_row=False):
        operation = importlib.import_module('tools.injection_preparation_operation')
        from looking_glass import injection_preclaim_guard as real_guard
        from looking_glass import injection_phase_accounting as real_accounting
        public = self.v3_public()
        phase = self.root / ('injection-heldout-' + self.freeze.sha256)
        ledger = self.root / 'fake-ledger'
        pilot = self.root / 'pilot'
        pilot.mkdir()
        rehearsal = tempfile.TemporaryDirectory()
        self.addCleanup(rehearsal.cleanup)
        rehearsal_path = Path(rehearsal.name)
        parent_info, pilot_info, rehearsal_info = self.root.stat(), pilot.stat(), rehearsal_path.stat()
        roots = real_accounting.PhaseRoots(str(self.root), parent_info.st_dev, parent_info.st_ino,
            str(phase), str(pilot), pilot_info.st_dev, pilot_info.st_ino,
            str(rehearsal_path), rehearsal_info.st_dev, rehearsal_info.st_ino)
        anchor = real_guard.ClaimAnchor(str(ledger), 1, 2, str(phase), '1' * 64, '2' * 64,
                                        '3' * 64, '4' * 64, '5' * 64)
        handoff = operation.ClaimHandoff(str(self.root), b'{}', digest(public.runtime_replay_plan_bytes),
            '6' * 64, (parent_info.st_dev, parent_info.st_ino), (1, 2))
        state = {'locked': False, 'valid': True, 'gate': False, 'replay_valid': True, 'config_valid': True}

        class Guard:
            ClaimAnchor = real_guard.ClaimAnchor

            @staticmethod
            def verify_existing_claim(path, name, *, output_path, anchor):
                return state['valid'] and path == ledger and name == 'injection_heldout' and output_path == phase

            @staticmethod
            @contextmanager
            def hold_existing_claim(path, name, *, output_path, anchor):
                if not state['valid']:
                    raise ValueError('fabricated row changed')
                state['locked'] = True
                try:
                    yield True
                finally:
                    state['locked'] = False

        class Accounting:
            PhaseRoots = real_accounting.PhaseRoots

            @staticmethod
            def reconcile_phase_files(actual_roots, *, stage, replay_tensor_bytes):
                marker = self.root / (phase.name + '.preparation-started.json')
                if (actual_roots != roots or replay_tensor_bytes != 1024 or not marker.exists()
                        or (stage == 'before_phase_creation' and phase.exists())
                        or (stage == 'before_nonce' and not (phase / 'claim.json').exists())):
                    raise ValueError('fabricated accounting differs')
                return {'status': 'bounded', 'stage': stage}

        class ReplayOwner:
            @staticmethod
            def verify_preclaim_handoff(**kwargs):
                return {'status': 'row_and_receipt_consistent_awaiting_observed_exit',
                        'receipt_sha256': handoff.receipt_sha256, 'intent_sha256': '7' * 64,
                        'claim_row_sha256': '8' * 64 if wrong_claim_row else anchor.heldout_claim_sha256}

        def gate(freeze):
            state['gate'] = True
            if change_after_gate == 'claim_after_gate':
                state['valid'] = False
            elif change_after_gate == 'replay_after_gate':
                state['replay_valid'] = False
            elif change_after_gate == 'config_after_gate':
                state['config_valid'] = False
            return freeze == self.freeze.sha256

        options = dict(custody=custody, guard=Guard, accounting=Accounting, replay_owner=ReplayOwner,
            replay_child=object(), replay_claim=object(), public=public, private_parent=self.root,
            ledger_path=ledger, anchor=anchor, roots=roots, handoff=handoff, replay_tensor_bytes=1024,
            config_bytes=b'fabricated-config', site_nonce=SITE_NONCE, site_record_bytes=self.site.record_bytes,
            tokenizer=FakeTokenizer(), observe_claimant_exit=lambda receipt: claimant_exit,
            observe_replay_config=lambda: b'{}' if state['config_valid'] else b'{"changed":true}',
            verify_replay_evidence=lambda plan, result, size: replay_ok and state['replay_valid'],
            publication_check=lambda name, raw: True, source_check=lambda binding: True,
            prediction_check=lambda raw, sha: True, gate_allocation=gate,
            observe_inventory=lambda: self.inventory_sha,
            runtime_check=lambda runtime, deviation, plan, replay, binding: True)
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': digest(self.runtime), '_STUDY_PREDICTION': digest(self.prediction),
                '_STUDY_PLAN': digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        return operation, options, state, pins, phase

    def test_wired_preclaim_draws_only_inside_the_guard_after_the_gate(self):
        operation, options, state, pins, phase = self._preparation_wiring()
        def draw(size):
            self.assertTrue(state['locked'])
            self.assertTrue(state['gate'])
            self.assertEqual(size, 32)
            return SPLIT_NONCE
        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', side_effect=draw):
            result = operation.prepare_from_handoff(**options)
        self.assertEqual(result.phase_root, phase)
        self.assertEqual((phase / 'nonce.bin').read_bytes(), SPLIT_NONCE)
        self.assertTrue((phase / 'preparation-complete.json').exists())
        self.assertFalse(state['locked'])

    def test_wired_preparation_stops_before_draw_on_missing_evidence_or_changed_claim(self):
        for failure in ('claimant_exit', 'replay', 'wrong_claim_row', 'claim_after_gate',
                        'replay_after_gate', 'config_after_gate'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=self.root) as temp:
                original = self.root
                self.root = Path(temp)
                try:
                    operation, options, _, pins, phase = self._preparation_wiring(
                        claimant_exit=failure != 'claimant_exit', replay_ok=failure != 'replay',
                        change_after_gate=failure, wrong_claim_row=failure == 'wrong_claim_row')
                    with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes') as draw:
                        with self.assertRaises(ValueError):
                            operation.prepare_from_handoff(**options)
                        draw.assert_not_called()
                    self.assertFalse((phase / 'nonce.bin').exists())
                    marker = self.root / (phase.name + '.preparation-started.json')
                    self.assertEqual(marker.exists(), failure.endswith('_after_gate'))
                finally:
                    self.root = original

    def test_wiring_uses_real_sqlite_guard_and_phase_accounting(self):
        import sqlite3
        from contextlib import closing
        from looking_glass import injection_preclaim_guard as guard
        from looking_glass import injection_phase_accounting as accounting
        operation, options, _, pins, phase = self._preparation_wiring()
        ledger = options['ledger_path']
        roots = options['roots']
        (Path(roots.pilot_path) / 'replay.bin').write_bytes(b'fabricated replay')
        (Path(roots.rehearsal_path) / 'diagnostic.bin').write_bytes(b'fabricated diagnostic')
        original_rows = [
            ('natural_discovery', str(self.root / 'natural-discovery'), 128 * 1024**2, 4096 * 1024**2, 0),
            ('injection_discovery', str(self.root / 'injection-discovery'), 48 * 1024**2, 128 * 1024**2, 13 * 1024**3),
        ]
        row = ('injection_heldout', str(phase), 72 * 1024**2, 192 * 1024**2, 13 * 1024**3)
        history = digest(b'reconciliations:present\nretired_evidence:present\n')
        scope = '1' * 64
        meta = canonical({'schema': 'mlg-agreement-phase-ledger-v6',
                          'current_history_sha256': history,
                          'evidence_scope_amendment_sha256': scope}).decode()
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
        identity = ledger.stat()
        anchor = guard.ClaimAnchor(str(ledger), identity.st_dev, identity.st_ino, str(phase),
            state.hexdigest(), history, scope,
            digest(json.dumps(sorted(original_rows), ensure_ascii=True, separators=(',', ':')).encode()),
            digest(json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')))
        handoff = replace(options['handoff'], ledger_identity=(identity.st_dev, identity.st_ino))

        class ReplayOwner:
            @staticmethod
            def verify_preclaim_handoff(**kwargs):
                self.assertTrue(guard.verify_existing_claim(ledger, 'injection_heldout',
                                                           output_path=phase, anchor=anchor))
                return {'status': 'row_and_receipt_consistent_awaiting_observed_exit',
                        'receipt_sha256': handoff.receipt_sha256, 'intent_sha256': '7' * 64,
                        'claim_row_sha256': anchor.heldout_claim_sha256}

        options.update(guard=guard, accounting=accounting, replay_owner=ReplayOwner,
                       anchor=anchor, handoff=handoff)

        def draw(size):
            with closing(sqlite3.connect(ledger, timeout=0, isolation_level=None)) as other:
                with self.assertRaises(sqlite3.OperationalError):
                    other.execute('BEGIN IMMEDIATE')
            return SPLIT_NONCE

        with mock.patch.multiple(custody, **pins), mock.patch.object(custody.secrets, 'token_bytes', side_effect=draw):
            result = operation.prepare_from_handoff(**options)
        self.assertTrue((result.phase_root / 'preparation-complete.json').is_file())
        with closing(sqlite3.connect(ledger, timeout=0, isolation_level=None)) as other:
            other.execute('BEGIN IMMEDIATE')
            other.rollback()

    def natural_split(self, nonce):
        freeze = commitments.make_prediction_freeze({'natural': {
            'source_commit': 'a' * 40, 'runtime_sha256': 'b' * 64, 'prediction_sha256': 'c' * 64}})
        snapshot = commitments.load_inventory_snapshot(self.inventory, self.inventory_sha)
        allocation = commitments.sample_families(freeze, nonce, inventory_snapshot=snapshot,
                                                current_inventory_sha256=snapshot.sha256)
        cases = custody.tokenize_assignment(allocation, FakeTokenizer())
        sealed = commitments.seal_heldout(freeze, nonce, cases, allocation=allocation,
                                           inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256)
        public = canonical({'schema': 'mlg-natural-heldout-public-commitment-v1',
                            'prediction_freeze_sha256': freeze.sha256,
                            'exposure_inventory_sha256': snapshot.sha256,
                            'heldout_commitment': sealed.commitment_sha256,
                            'wrapper_sha256': '6' * 64, 'deviation_sha256': '7' * 64,
                            'operational_binding_sha256': '8' * 64})
        return custody.CommittedSplit('natural', freeze.record_bytes, freeze.sha256, nonce,
            sealed.record_bytes, public, self.inventory, self.inventory_sha)

    def injection_split(self, prepared):
        phase = prepared.phase_root
        return custody.CommittedSplit('injection', self.freeze.record_bytes, self.freeze.sha256,
            (phase / 'nonce.bin').read_bytes(), (phase / 'heldout-record.json').read_bytes(),
            (phase / 'commitment.json').read_bytes(), self.inventory, self.inventory_sha,
            SITE_NONCE, self.site.record_bytes, self.site.commitment_sha256)

    def test_committed_split_comparator_reports_only_clear_or_collision(self):
        self.assertTrue(hasattr(custody, 'CommittedSplit'), 'committed split comparison is not implemented')
        injection = self.injection_split(self.prepare())
        identical = self.natural_split(SPLIT_NONCE)
        collision = custody.compare_committed_splits(identical, injection,
            comparator_source_sha256=digest(Path(custody.__file__).read_bytes()),
            publication_check=lambda name, raw: True)
        self.assertEqual(json.loads(collision)['status'], 'collision')
        self.assertNotIn(b'input_ids', collision)
        self.assertNotIn(b'nonce', collision)
        for counter in range(1, 256):
            natural = self.natural_split(bytes([counter]) * 32)
            clear = custody.compare_committed_splits(natural, injection,
                comparator_source_sha256=digest(Path(custody.__file__).read_bytes()),
                publication_check=lambda name, raw: True)
            if json.loads(clear)['status'] == 'clear':
                break
        else:
            self.fail('fabricated pools did not produce disjoint rows')
        self.assertEqual(json.loads(clear)['natural_case_count'], 16)
        self.assertEqual(json.loads(clear)['injection_case_count'], 16)
        self.assertTrue(hasattr(custody, 'verify_clear_receipt'), 'clear receipt verification is not implemented')
        expected = {'natural_freeze_sha256': natural.freeze_sha256,
                    'injection_freeze_sha256': injection.freeze_sha256,
                    'natural_inventory_sha256': natural.inventory_sha256,
                    'injection_inventory_sha256': injection.inventory_sha256,
                    'natural_commitment_bytes': natural.commitment_bytes,
                    'injection_commitment_bytes': injection.commitment_bytes,
                    'comparator_source_sha256': digest(Path(custody.__file__).read_bytes())}
        self.assertTrue(custody.verify_clear_receipt(clear, **expected))
        for altered in (collision, clear.replace(b'"natural_case_count":16', b'"natural_case_count":true')):
            with self.assertRaises(custody.CustodyError):
                custody.verify_clear_receipt(altered, **expected)
        with self.assertRaises(custody.CustodyError):
            custody.compare_committed_splits(natural, injection,
                comparator_source_sha256='0' * 64, publication_check=lambda name, raw: True)

    def test_extra_private_field_is_rejected_before_commitment_readback(self):
        injection = self.injection_split(self.prepare())
        natural = self.natural_split(SPLIT_NONCE)
        for target in ('natural', 'injection'):
            original = natural if target == 'natural' else injection
            modified = json.loads(original.commitment_bytes)
            modified['site'] = 'FABRICATED_PRIVATE_VALUE'
            altered = replace(original, commitment_bytes=canonical(modified))
            callbacks = []
            with self.subTest(target=target):
                with self.assertRaises(custody.CustodyError):
                    custody.compare_committed_splits(altered if target == 'natural' else natural,
                        altered if target == 'injection' else injection,
                        comparator_source_sha256=digest(Path(custody.__file__).read_bytes()),
                        publication_check=lambda name, raw: callbacks.append(name) or True)
                self.assertNotIn(target + '_commitment', callbacks)


if __name__ == '__main__':
    unittest.main()
