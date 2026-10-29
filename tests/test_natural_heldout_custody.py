"""Fabricated pools/nonces only: no real study cases or model forwards."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest import mock

from looking_glass import pretrained_exposures_v7 as exposures
from looking_glass import pretrained_number_commitments as commitments
from tools import natural_heldout_custody as custody


NOUNS = ('alpha/alphas', 'beta/betas', 'gamma/gammas', 'delta/deltas',
         'epsilon/epsilons', 'zeta/zetas', 'eta/etas', 'theta/thetas')
PLURAL = ('near boxes', 'behind cars', 'above lamps', 'below desks')
SINGULAR = ('near box', 'behind car', 'above lamp', 'below desk')
NONCE = bytes(range(32))


def outer_command(inner):
    return inner[:7] + [custody.OUTER_BOOTSTRAP] + inner[8:9] + ['supervise-prepare'] + inner[10:]


class FakeTokenizer:
    def __init__(self):
        self.subjects = {subject: 20000 + index for index, subject in enumerate(
            part for noun in NOUNS for part in noun.split('/'))}
        self.contexts = {context: 30000 + index for index, context in enumerate(PLURAL + SINGULAR)}

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError('unexpected special tokens')
        if text == 'The':
            return [10]
        if text.startswith(' ') and text[1:] in self.contexts:
            return [self.contexts[text[1:]]]
        if text.startswith('The '):
            body = text[4:]
            subject, _, rest = body.partition(' ')
            if subject not in self.subjects:
                raise ValueError('unexpected subject')
            result = [10, self.subjects[subject]]
            if not rest:
                return result
            for continuation, token in custody.TARGETS.items():
                if rest.endswith(continuation):
                    return result + [self.contexts[rest[:-len(continuation)]]] + [token]
            return result + [self.contexts[rest]]
        raise ValueError('unexpected text')


class FakeLedger:
    def __init__(self, calls):
        self.calls = calls
        self.claimed = False

    def __call__(self, phase, *, output_path):
        self.calls.append('claim')
        if self.claimed:
            raise RuntimeError('phase already claimed')
        self.claimed = True
        return SimpleNamespace(phase=phase, output_path=output_path,
                               logical_limit_bytes=custody.LOGICAL_LIMIT,
                               disk_limit_bytes=custody.DISK_LIMIT)


class CustodyTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.multiple(commitments, NOUNS=NOUNS, PLURAL_CONTEXTS=PLURAL,
                                    SINGULAR_CONTEXTS=SINGULAR)
        patch.start()
        self.addCleanup(patch.stop)
        self.calls = []
        self.ledger = FakeLedger(self.calls)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config_path = self.root / 'private-config.json'
        self.config_bytes = b'{"schema":"fabricated"}'
        self.config_path.write_bytes(self.config_bytes)
        self.wrapper = Path(custody.__file__).read_bytes()
        self.deviation, self.prediction = b'fake deviation', b'fake prediction'
        self.inventory_source = b'fabricated inventory source'
        self.binding = json.dumps({'source_commit': custody.SOURCE_COMMIT,
                                   'checkpoint_revision': custody.CHECKPOINT}).encode()
        self.sha_patch = mock.patch.multiple(custody,
            BINDING_SHA256=hashlib.sha256(self.binding).hexdigest(),
            PREDICTION_SHA256=hashlib.sha256(self.prediction).hexdigest(),
            CURRENT_INVENTORY_VERSION=8, CURRENT_INVENTORY_SHA256=exposures.inventory_digest(),
            CURRENT_INVENTORY_SOURCE_SHA256=hashlib.sha256(self.inventory_source).hexdigest())
        self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)
        self.freeze = commitments.make_prediction_freeze({'natural': {
            'source_commit': custody.SOURCE_COMMIT, 'runtime_sha256': custody.BINDING_SHA256,
            'prediction_sha256': custody.PREDICTION_SHA256}})
        self.aggregate_patch = mock.patch.object(custody, 'AGGREGATE_SHA256', self.freeze.sha256)
        self.aggregate_patch.start()
        self.addCleanup(self.aggregate_patch.stop)
        base = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'a' * 40 + '/'
        source_data = {'custody': self.wrapper, 'deviation': self.deviation,
                       'child': b'fake child', 'owner': b'fake owner', 'supervisor': b'fake supervisor',
                       'ledger': b'fake ledger', 'auditor': b'fake auditor'}
        sources = {name: {'url': base + suffix, 'sha256': hashlib.sha256(source_data[name]).hexdigest()}
                   for name, suffix in custody.SOURCE_SUFFIXES.items()}
        operation = {'schema': 'mlg-natural-heldout-operational-binding-v1',
                     'core_source_commit': custody.SOURCE_COMMIT,
                     'binding_sha256': custody.BINDING_SHA256,
                     'prediction_sha256': custody.PREDICTION_SHA256,
                     'aggregate_sha256': custody.AGGREGATE_SHA256,
                     'inventory_version': custody.CURRENT_INVENTORY_VERSION,
                     'inventory_sha256': custody.CURRENT_INVENTORY_SHA256,
                     'inventory_source_sha256': custody.CURRENT_INVENTORY_SOURCE_SHA256,
                     'runtime_overrides': {'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None},
                     'limits': {'tensor_bytes': custody.LOGICAL_LIMIT, 'event_bytes': custody.EVENT_LIMIT,
                                'side_bytes': custody.SIDE_LIMIT}, 'sources': sources}
        self.operation = custody._json_bytes(operation)
        self.public = custody.PublicInputs(
            binding=self.binding, prediction=self.prediction, aggregate=self.freeze.record_bytes,
            inventory=exposures.canonical_inventory_bytes(), v7_inventory=exposures.canonical_inventory_bytes(),
            inventory_source=self.inventory_source, operational_binding=self.operation,
            binding_url=base + 'results/distilgpt2-natural-agreement-v1-20261028/binding-v2.json',
            prediction_url=base + 'results/distilgpt2-natural-agreement-v1-20261028/natural-prediction-v2.json',
            aggregate_url=base + 'results/distilgpt2-natural-agreement-v1-20261028/natural-prediction-freeze.json',
            inventory_sha256=exposures.inventory_digest(),
            inventory_source_url=base + 'looking_glass/pretrained_exposures_v8.py',
            operational_binding_url=base + 'results/distilgpt2-natural-agreement-v1-20261028/heldout-operational-binding-v1.json',
            wrapper_url=base + 'tools/natural_heldout_custody.py',
            deviation_url=base + 'docs/NATURAL_AGREEMENT_HELDOUT_DEVIATION.md',
            deviation_bytes=self.deviation)
        self.remote = {self.public.binding_url: self.binding, self.public.prediction_url: self.prediction,
                       self.public.aggregate_url: self.freeze.record_bytes,
                       self.public.inventory_source_url: self.inventory_source,
                       self.public.operational_binding_url: self.operation,
                       self.public.wrapper_url: self.wrapper, self.public.deviation_url: self.deviation}
        self.remote.update({sources[name]['url']: data for name, data in source_data.items()})
        self.heldout = SimpleNamespace(load_frozen_natural_predictions=lambda data, freeze: None)

    def reader(self, url, suffix):
        self.calls.append('published')
        return self.remote[url]

    def nonce(self, count):
        self.calls.append('nonce')
        self.assertTrue(self.ledger.claimed)
        self.assertEqual(count, 32)
        return NONCE

    def observe_inventory(self):
        self.calls.append('observe')
        return {'inventory_sha256': exposures.inventory_digest(),
                'source_sha256': hashlib.sha256(self.inventory_source).hexdigest(),
                'head': 'a' * 40}

    def prepare(self, **changes):
        options = {'commitments': commitments, 'heldout': self.heldout, 'tokenizer': FakeTokenizer(),
                   'wrapper_bytes': self.wrapper, 'claim_phase': self.ledger, 'reader': self.reader,
                   'nonce_source': self.nonce, 'observe_inventory': self.observe_inventory,
                   'config_bytes': self.config_bytes, 'config_path': self.config_path}
        options.update(changes)
        return custody.prepare_once(self.public, self.root, **options)

    def supervised(self, attempt):
        directory = self.root / 'preparation-supervisor'
        directory.mkdir()
        (directory / 'stdout.txt').write_bytes(b'completed\n')
        (directory / 'stderr.txt').write_bytes(b'')
        receipt = {'status': 'evidence_complete', 'exit_code': 0, 'child_reaped': True,
                   'child_pid': 12345, 'logs_complete': True, 'retry_safe': False,
                   'cancellation_requested': False, 'quarantined': False,
                   'command': [sys.executable, '-I', '-S', '-B', '-X',
                               'pycache_prefix=' + str(self.root / 'unused-pycache'),
                               '-c', custody.BOOTSTRAP, str(Path(custody.__file__)), 'prepare',
                               '--config', str(self.config_path), '--config-sha',
                               hashlib.sha256(self.config_bytes).hexdigest()],
                   'job_drained': True, 'job_closed': True, 'cleanup_verified': True,
                   'supervisor_error': None, 'log_overflow': False,
                   'audit': custody.audit_preparation(attempt, self.public, self.wrapper),
                   'retained_tensor_files': {'accounting_complete': True, 'total_bytes': 0},
                   'stdout_sha256': hashlib.sha256(b'completed\n').hexdigest(),
                   'stderr_sha256': hashlib.sha256(b'').hexdigest()}
        raw = custody._json_bytes(receipt)
        (directory / 'supervisor.json').write_bytes(raw)
        observation = {'schema': 'mlg-natural-heldout-preparation-outer-observation-v1',
                       'status': 'completed', 'exit_code': 0, 'supervisor_reaped': True,
                       'outer_job_drained': True, 'outer_job_closed': True,
                       'supervisor_pid': 54321, 'supervisor_birth_100ns': 123456789,
                       'command': outer_command(receipt['command']),
                       'supervisor_record_sha256': hashlib.sha256(raw).hexdigest(),
                       'preparation_config_sha256': hashlib.sha256(self.config_bytes).hexdigest(),
                       'operational_binding_sha256': hashlib.sha256(self.operation).hexdigest()}
        (directory / 'supervisor-observation.json').write_bytes(custody._json_bytes(observation))
        return directory

    def test_publication_precedes_one_durable_claim_and_nonce(self):
        attempt, public = self.prepare()
        self.assertEqual(self.calls, ['published'] * 14 + ['observe', 'claim', 'observe', 'nonce'])
        self.assertEqual((attempt / 'nonce.bin').read_bytes(), NONCE)
        self.assertEqual(set(public), {'schema', 'prediction_freeze_sha256', 'exposure_inventory_sha256',
                                       'heldout_commitment', 'wrapper_sha256', 'deviation_sha256',
                                       'operational_binding_sha256'})
        self.assertNotIn('alpha', json.dumps(public))
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()
        self.assertEqual(self.calls.count('nonce'), 1)

    def test_prepared_cases_require_exact_public_commitment_and_retokenization(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'c' * 40 +
               '/results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json')
        raw = (attempt / 'commitment.json').read_bytes()
        self.remote[url] = raw
        args = {'commitments': commitments, 'heldout': self.heldout, 'tokenizer': FakeTokenizer(),
                'wrapper_bytes': self.wrapper, 'observe_inventory': self.observe_inventory, 'reader': self.reader,
                'preparation_supervisor': supervisor}
        freeze, snapshot, nonce, _, commitment = custody.verify_prepared(attempt, self.public, url, **args)
        self.assertEqual((freeze.sha256, snapshot.sha256, nonce),
                         (self.freeze.sha256, exposures.inventory_digest(), NONCE))
        self.assertEqual(commitment['prediction_freeze_sha256'], self.freeze.sha256)
        custody.start_owner_once(attempt, commitment)
        with self.assertRaises(FileExistsError):
            custody.start_owner_once(attempt, commitment)
        self.remote[url] = b'changed'
        with self.assertRaises(custody.CustodyError):
            custody.verify_prepared(attempt, self.public, url, **args)
        self.remote[url] = raw
        (attempt / 'heldout-record.json').write_bytes(b'{}')
        with self.assertRaises(custody.CustodyError):
            custody.verify_prepared(attempt, self.public, url, **args)

    def test_publication_mismatch_fails_before_claim_or_nonce(self):
        self.remote[self.public.aggregate_url] = b'different'
        with self.assertRaises(custody.CustodyError):
            self.prepare()
        self.assertNotIn('claim', self.calls)
        self.assertNotIn('nonce', self.calls)

    def test_operational_source_mismatch_fails_before_claim_or_nonce(self):
        url = next(url for url in self.remote if url.endswith('tools/natural_agreement_heldout_child.py'))
        self.remote[url] = b'different child'
        with self.assertRaises(custody.CustodyError):
            self.prepare()
        self.assertNotIn('claim', self.calls)
        self.assertNotIn('nonce', self.calls)

    def test_interruption_after_nonce_has_no_second_draw(self):
        original = custody._write_new
        def interrupt(path, data):
            if Path(path).name == 'nonce.bin':
                raise KeyboardInterrupt()
            return original(path, data)
        with mock.patch.object(custody, '_write_new', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertEqual(self.calls.count('nonce'), 1)
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()
        self.assertEqual(self.calls.count('nonce'), 1)

    def test_interruption_after_grant_before_attempt_has_no_second_draw(self):
        original = Path.mkdir
        def interrupt(path, *args, **kwargs):
            if path.name == 'natural-heldout-' + self.freeze.sha256:
                raise KeyboardInterrupt()
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, 'mkdir', interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertTrue(self.ledger.claimed)
        self.assertNotIn('nonce', self.calls)
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()

    def test_interruption_after_commitment_write_keeps_claim_consumed(self):
        original = custody._write_new
        def interrupt(path, data):
            original(path, data)
            if Path(path).name == 'commitment.json':
                raise KeyboardInterrupt()
        with mock.patch.object(custody, '_write_new', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        attempt = self.root / ('natural-heldout-' + self.freeze.sha256)
        self.assertTrue((attempt / 'commitment.json').is_file())
        supervisor = self.root / 'unsuccessful-supervisor'
        supervisor.mkdir()
        url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'c' * 40 +
               '/results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json')
        self.remote[url] = (attempt / 'commitment.json').read_bytes()
        with self.assertRaises(custody.CustodyError):
            custody.verify_prepared(attempt, self.public, url, commitments=commitments, heldout=self.heldout,
                                    tokenizer=FakeTokenizer(), wrapper_bytes=self.wrapper,
                                    observe_inventory=self.observe_inventory, reader=self.reader,
                                    preparation_supervisor=supervisor)
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()
        self.assertEqual(self.calls.count('nonce'), 1)

    def test_double_fault_cannot_replace_a_successful_supervisor_receipt(self):
        original = custody._write_new
        def interrupt(path, data):
            if Path(path).name == 'preparation-failure.json':
                raise OSError('synthetic failure marker fault')
            original(path, data)
            if Path(path).name == 'preparation-complete.json':
                raise KeyboardInterrupt()
        with mock.patch.object(custody, '_write_new', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        attempt = self.root / ('natural-heldout-' + self.freeze.sha256)
        self.assertTrue((attempt / 'preparation-complete.json').is_file())
        self.assertFalse((attempt / 'preparation-failure.json').exists())
        supervisor = self.root / 'unsuccessful-supervisor'
        supervisor.mkdir()
        url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'c' * 40 +
               '/results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json')
        self.remote[url] = (attempt / 'commitment.json').read_bytes()
        with self.assertRaises((OSError, custody.CustodyError)):
            custody.verify_prepared(attempt, self.public, url, commitments=commitments, heldout=self.heldout,
                                    tokenizer=FakeTokenizer(), wrapper_bytes=self.wrapper,
                                    observe_inventory=self.observe_inventory, reader=self.reader,
                                    preparation_supervisor=supervisor)
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()
        self.assertEqual(self.calls.count('nonce'), 1)

    def test_supervisor_receipt_binds_command_identity_and_outer_exit(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        receipt_path = supervisor / 'supervisor.json'
        observed_path = supervisor / 'supervisor-observation.json'
        raw_receipt = receipt_path.read_bytes()
        raw_observed = observed_path.read_bytes()
        custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        observed_path.unlink()
        with self.assertRaises(custody.CustodyError):
            custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        # The process itself can inspect its finished inner receipt, but that
        # provisional check cannot authorize publication or an owner.
        custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper,
                                              require_outer_observation=False)
        observed_path.write_bytes(raw_observed)
        for field, value in (('child_pid', None), ('child_pid', True), ('quarantined', True),
                             ('cancellation_requested', True), ('logs_complete', False),
                             ('retry_safe', True)):
            with self.subTest(field=field, value=value):
                receipt = json.loads(raw_receipt)
                receipt[field] = value
                altered = custody._json_bytes(receipt)
                receipt_path.write_bytes(altered)
                observation = json.loads(raw_observed)
                observation['supervisor_record_sha256'] = hashlib.sha256(altered).hexdigest()
                observed_path.write_bytes(custody._json_bytes(observation))
                with self.assertRaises(custody.CustodyError):
                    custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        for index, value in ((8, str(self.root / 'unrelated.py')), (9, 'validate'),
                             (11, str(self.root / 'other-config.json')), (13, 'f' * 64)):
            with self.subTest(command_index=index):
                receipt = json.loads(raw_receipt)
                receipt['command'][index] = value
                altered = custody._json_bytes(receipt)
                receipt_path.write_bytes(altered)
                observation = json.loads(raw_observed)
                observation['supervisor_record_sha256'] = hashlib.sha256(altered).hexdigest()
                observed_path.write_bytes(custody._json_bytes(observation))
                with self.assertRaises(custody.CustodyError):
                    custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        receipt_path.write_bytes(raw_receipt)
        observed_path.write_bytes(raw_observed)
        self.config_path.write_bytes(b'changed after preparation')
        with self.assertRaises(custody.CustodyError):
            custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        self.config_path.write_bytes(self.config_bytes)
        for name in ('claim.json', 'preparation-config.json', 'exposure-observations.json', 'nonce.bin',
                     'heldout-record.json', 'inventory.json', 'commitment.json', 'preparation-complete.json'):
            with self.subTest(file=name):
                path = attempt / name
                original = path.read_bytes()
                path.write_bytes(original + b'!')
                with self.assertRaises(custody.CustodyError):
                    custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
                path.write_bytes(original)

    def test_outer_observer_gates_reaps_and_binds_real_process_identity(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        observation_path = supervisor / 'supervisor-observation.json'
        observation_path.unlink()
        inner = json.loads((supervisor / 'supervisor.json').read_bytes())['command']
        command = outer_command(inner)
        calls = []
        class Input(io.BytesIO):
            def write(self, data):
                calls.append('gate')
                return super().write(data)
        class Process:
            pid = 54321
            stdin = Input()
            def wait(self, timeout):
                calls.append('wait')
                self_test.assertIn('gate', calls)
                return 0
        class Job:
            def assign(self, process): calls.append('assign')
            def terminate_and_wait(self): calls.append('drain')
            def close(self): calls.append('close')
        self_test = self
        def create(*args, **kwargs):
            calls.append('spawn')
            return Process()
        def birth(process):
            calls.append('birth')
            return 123456789
        with mock.patch.object(custody.subprocess, 'Popen', side_effect=create), \
             mock.patch.object(custody, '_process_birth_100ns', side_effect=birth):
            result = custody.observe_supervisor_process(command, attempt, supervisor, self.public,
                                                        self.wrapper, config_path=self.config_path,
                                                        config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                                                        job_factory=Job)
        self.assertEqual(calls, ['spawn', 'assign', 'birth', 'gate', 'wait', 'drain', 'close'])
        self.assertEqual(result['status'], 'observed_preparation_exit')
        custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        observation = json.loads(observation_path.read_bytes())
        for field, value in (('supervisor_pid', True), ('supervisor_pid', 12345),
                             ('supervisor_birth_100ns', 0), ('outer_job_drained', False),
                             ('outer_job_closed', False)):
            with self.subTest(field=field):
                changed = dict(observation)
                changed[field] = value
                observation_path.write_bytes(custody._json_bytes(changed))
                with self.assertRaises(custody.CustodyError):
                    custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        observation_path.write_bytes(custody._json_bytes(observation))

    @unittest.skipUnless(os.name == 'nt', 'requires a real Windows Job')
    def test_real_windows_gated_observer_with_fabricated_receipt(self):
        from scripts import v2_source_guards as guards
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        observation = supervisor / 'supervisor-observation.json'
        observation.unlink()
        fake_supervisor = self.root / 'gated-supervisor.py'
        repo = str(Path(custody.__file__).resolve().parents[1])
        fake_supervisor.write_text(
            'import sys\n'
            'sys.path.insert(0, ' + repr(repo) + ')\n'
            'from looking_glass.natural_agreement_controller import await_start\n'
            'await_start()\n',
            encoding='utf-8')
        receipt_path = supervisor / 'supervisor.json'
        receipt = json.loads(receipt_path.read_bytes())
        receipt['command'][8] = str(fake_supervisor)
        receipt_path.write_bytes(custody._json_bytes(receipt))
        command = outer_command(receipt['command'])
        with mock.patch.object(custody, '__file__', str(fake_supervisor)), \
             mock.patch.dict(os.environ, {'MLG_NATURAL_SITE': str(self.root)}):
            result = custody.observe_supervisor_process(command, attempt, supervisor, self.public, self.wrapper,
                config_path=self.config_path, config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                job_factory=lambda: custody._outer_supervisor_job(guards))
            self.assertEqual(result['status'], 'observed_preparation_exit')
            custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        saved = json.loads(observation.read_bytes())
        self.assertTrue(saved['outer_job_drained'] and saved['outer_job_closed'])
        self.assertGreater(saved['supervisor_birth_100ns'], 0)

    def test_outer_observer_nonzero_or_interrupt_never_grants_success(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        observation_path = supervisor / 'supervisor-observation.json'
        observation_path.unlink()
        inner = json.loads((supervisor / 'supervisor.json').read_bytes())['command']
        command = outer_command(inner)
        for outcome in ('nonzero', 'before_gate', 'after_gate'):
            calls = []
            class Process:
                pid = 54321
                stdin = io.BytesIO()
                def wait(self, timeout):
                    calls.append('wait')
                    if outcome == 'after_gate' and calls.count('wait') == 1:
                        raise KeyboardInterrupt()
                    return 7 if outcome == 'nonzero' else 0
                def poll(self): return None
                def kill(self): calls.append('kill')
            class Job:
                def assign(self, process): calls.append('assign')
                def terminate_and_wait(self): calls.append('drain')
                def close(self): calls.append('close')
            def birth(process):
                if outcome == 'before_gate':
                    raise KeyboardInterrupt()
                return 123456789
            with self.subTest(outcome=outcome), \
                 mock.patch.object(custody.subprocess, 'Popen', return_value=Process()), \
                 mock.patch.object(custody, '_process_birth_100ns', side_effect=birth):
                with self.assertRaises((custody.CustodyError, KeyboardInterrupt)):
                    custody.observe_supervisor_process(command, attempt, supervisor, self.public,
                                                       self.wrapper, config_path=self.config_path,
                                                       config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                                                       job_factory=Job)
                self.assertIn('drain', calls)
                self.assertIn('close', calls)
                self.assertFalse(observation_path.exists())
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()

    def test_lost_popen_return_cannot_open_the_outer_gate(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        observation = supervisor / 'supervisor-observation.json'
        observation.unlink()
        inner = json.loads((supervisor / 'supervisor.json').read_bytes())['command']
        command = outer_command(inner)
        marker = self.root / 'gate-opened'
        children = []
        real_popen = custody.subprocess.Popen
        code = ('import pathlib,sys; token=sys.stdin.buffer.readline(); '
                'pathlib.Path(sys.argv[1]).write_text("opened") '
                'if token==b"MLG-natural-discovery-start-v1\\n" else None')
        class Job:
            def assign(self, process): self.fail('a lost handle cannot be assigned')
            def terminate_and_wait(self): pass
            def close(self): pass
        def lost(*args, **kwargs):
            child = real_popen([sys.executable, '-I', '-S', '-B', '-c', code, str(marker)],
                               stdin=custody.subprocess.PIPE, stdout=custody.subprocess.DEVNULL,
                               stderr=custody.subprocess.DEVNULL, creationflags=custody.subprocess.DETACHED_PROCESS)
            children.append(child)
            raise OSError('synthetic lost Popen return')
        with mock.patch.object(custody.subprocess, 'Popen', side_effect=lost):
            with self.assertRaises(custody.CustodyError):
                custody.observe_supervisor_process(command, attempt, supervisor, self.public, self.wrapper,
                    config_path=self.config_path, config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                    job_factory=Job)
        children[0].stdin.close()
        self.assertEqual(children[0].wait(timeout=10), 0)
        self.assertFalse(marker.exists())
        self.assertFalse(observation.exists())

    @unittest.skipUnless(os.name == 'nt', 'requires the Windows gate bootstrap')
    def test_outer_bootstrap_will_not_load_source_without_a_gate(self):
        marker = self.root / 'source-ran'
        source = self.root / 'not-started.py'
        source.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("ran")\n',
                          encoding='utf-8')
        process = custody.subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', custody.OUTER_BOOTSTRAP,
                                           str(source)], stdin=custody.subprocess.PIPE,
                                          stdout=custody.subprocess.DEVNULL,
                                          stderr=custody.subprocess.DEVNULL,
                                          creationflags=custody.subprocess.DETACHED_PROCESS)
        process.stdin.close()
        self.assertNotEqual(process.wait(timeout=15), 0)
        self.assertFalse(marker.exists())

    def test_observer_entry_bootstrap_hashes_before_execution(self):
        marker = self.root / 'observer-code-ran'
        source = self.root / 'observer-entry.py'
        raw = ('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("ran")\n').encode()
        source.write_bytes(raw)
        for digest, expected in (('f' * 64, False), (hashlib.sha256(raw).hexdigest(), True)):
            with self.subTest(valid=expected):
                result = custody.subprocess.run([sys.executable, '-I', '-S', '-B', '-c',
                                                 custody.OBSERVER_ENTRY_BOOTSTRAP,
                                                 str(source), digest], capture_output=True)
                self.assertEqual(result.returncode == 0, expected)
                self.assertEqual(marker.exists(), expected)
                if marker.exists():
                    marker.unlink()

    def test_observer_preserves_cancellation_when_cleanup_also_fails(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        (supervisor / 'supervisor-observation.json').unlink()
        inner = json.loads((supervisor / 'supervisor.json').read_bytes())['command']
        command = outer_command(inner)
        class Process:
            pid = 54321
            stdin = io.BytesIO()
            def wait(self, timeout): return 0
            def poll(self): return None
            def kill(self): pass
        class Job:
            def assign(self, process): pass
            def terminate_and_wait(self): raise OSError('synthetic cleanup failure')
            def close(self): pass
        with mock.patch.object(custody.subprocess, 'Popen', return_value=Process()), \
             mock.patch.object(custody, '_process_birth_100ns', side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt) as caught:
                custody.observe_supervisor_process(command, attempt, supervisor, self.public, self.wrapper,
                    config_path=self.config_path, config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                    job_factory=Job)
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_uncertain_cleanup_after_gate_roots_quarantine_handles(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        (supervisor / 'supervisor-observation.json').unlink()
        inner = json.loads((supervisor / 'supervisor.json').read_bytes())['command']
        command = outer_command(inner)
        class Process:
            pid = 54321
            stdin = io.BytesIO()
            calls = 0
            def wait(self, timeout):
                self.calls += 1
                if self.calls == 1: raise KeyboardInterrupt()
                return 0
            def poll(self): return None
            def kill(self): pass
        class Job:
            def assign(self, process): pass
            def terminate_and_wait(self): raise OSError('synthetic cleanup failure')
            def close(self): pass
        process, job = Process(), Job()
        with mock.patch.object(custody.subprocess, 'Popen', return_value=process), \
             mock.patch.object(custody, '_process_birth_100ns', return_value=123456789):
            with self.assertRaises(KeyboardInterrupt) as caught:
                custody.observe_supervisor_process(command, attempt, supervisor, self.public, self.wrapper,
                    config_path=self.config_path, config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                    job_factory=lambda: job)
        retained = caught.exception._mlg_quarantined_process
        self.assertIs(retained['process'], process)
        self.assertIs(retained['job'], job)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        with mock.patch.object(custody.time, 'sleep', side_effect=KeyboardInterrupt()), \
             mock.patch.object(sys, 'stderr', new_callable=io.StringIO) as captured:
            with self.assertRaises(KeyboardInterrupt):
                custody._hold_unverified_process(caught.exception)
            self.assertEqual(json.loads(captured.getvalue())['status'], 'quarantined')

    def test_observer_rejects_wrong_command_before_spawn(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        inner = json.loads((supervisor / 'supervisor.json').read_bytes())['command']
        command = outer_command(inner)
        command[9] = 'prepare'
        with mock.patch.object(custody.subprocess, 'Popen') as spawn:
            with self.assertRaises(custody.CustodyError):
                custody.observe_supervisor_process(command, attempt, supervisor, self.public, self.wrapper,
                    config_path=self.config_path, config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                    job_factory=lambda: None)
            spawn.assert_not_called()

    def test_complete_observation_after_write_error_is_recoverable_without_redraw(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        observation_path = supervisor / 'supervisor-observation.json'
        observation_path.unlink()
        inner = json.loads((supervisor / 'supervisor.json').read_bytes())['command']
        command = outer_command(inner)
        class Process:
            pid = 54321
            stdin = io.BytesIO()
            def wait(self, timeout): return 0
        class Job:
            def assign(self, process): pass
            def terminate_and_wait(self): pass
            def close(self): pass
        original = custody._write_new
        def interrupted(path, data):
            original(path, data)
            if Path(path) == observation_path:
                raise OSError('synthetic post-write interruption')
        with mock.patch.object(custody.subprocess, 'Popen', return_value=Process()), \
             mock.patch.object(custody, '_process_birth_100ns', return_value=123456789), \
             mock.patch.object(custody, '_write_new', side_effect=interrupted):
            with self.assertRaises(OSError):
                custody.observe_supervisor_process(command, attempt, supervisor, self.public,
                                                   self.wrapper, config_path=self.config_path,
                                                   config_sha256=hashlib.sha256(self.config_bytes).hexdigest(),
                                                   job_factory=Job)
        custody.verify_preparation_supervisor(attempt, supervisor, self.public, self.wrapper)
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()
        self.assertEqual(self.calls.count('nonce'), 1)

    def test_changed_preparation_config_fails_before_claim(self):
        self.config_path.write_bytes(b'changed')
        with self.assertRaises(custody.CustodyError):
            self.prepare()
        self.assertNotIn('claim', self.calls)
        self.assertNotIn('nonce', self.calls)

    def test_bad_token_or_continuation_aborts_and_retains_nonce(self):
        class BadTokenizer(FakeTokenizer):
            def encode(self, text, *, add_special_tokens):
                result = super().encode(text, add_special_tokens=add_special_tokens)
                return result[:-1] + [True] if text.endswith(' is') else result
        with self.assertRaises(custody.CustodyError):
            self.prepare(tokenizer=BadTokenizer())
        attempt = self.root / ('natural-heldout-' + self.freeze.sha256)
        self.assertEqual((attempt / 'nonce.bin').read_bytes(), NONCE)
        self.assertEqual(json.loads((attempt / 'preparation-failure.json').read_bytes())['status'], 'incomplete')
        self.assertFalse((attempt / 'commitment.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()

    def test_known_complete_rows_are_excluded_in_both_prefix_directions(self):
        cases = [[{'input_ids': [10, 20000 + family * 4 + item, 30000]}
                  for item in range(4)] for family in range(4)]
        for known in ([10, 20000], [10, 20000, 30000, 7], [10, 20000, 30000]):
            inventory = custody._json_bytes({'records': [{'token_ids': known}]})
            with self.subTest(known=known), self.assertRaises(custody.CustodyError):
                custody.reject_known_row_prefixes(cases, inventory)
        custody.reject_known_row_prefixes(cases, custody._json_bytes({'records': [{'token_ids': [11, 1]}]}))

    def test_collision_after_one_draw_fails_before_seal_and_cannot_retry(self):
        original = custody.tokenize_assignment
        known = json.loads(self.public.inventory)['records'][0]['token_ids']
        def collision(allocation, tokenizer):
            cases = original(allocation, tokenizer)
            cases[0][0]['input_ids'] = list(known) + [7]
            return cases
        with mock.patch.object(custody, 'tokenize_assignment', side_effect=collision), \
             mock.patch.object(commitments, 'seal_heldout', wraps=commitments.seal_heldout) as seal:
            with self.assertRaises(custody.CustodyError):
                self.prepare()
            seal.assert_not_called()
        attempt = self.root / ('natural-heldout-' + self.freeze.sha256)
        self.assertTrue(self.ledger.claimed)
        self.assertEqual((attempt / 'nonce.bin').read_bytes(), NONCE)
        self.assertEqual(json.loads((attempt / 'preparation-failure.json').read_bytes())['status'], 'incomplete')
        self.assertFalse((attempt / 'heldout-record.json').exists())
        self.assertFalse((attempt / 'commitment.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            self.prepare()
        self.assertEqual(self.calls.count('nonce'), 1)

    def test_recheck_rejects_known_row_extension_before_owner_or_reseal(self):
        attempt, _ = self.prepare()
        supervisor = self.supervised(attempt)
        url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'c' * 40 +
               '/results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json')
        self.remote[url] = (attempt / 'commitment.json').read_bytes()
        original = custody.tokenize_assignment
        known = json.loads(self.public.inventory)['records'][0]['token_ids']
        def collision(allocation, tokenizer):
            cases = original(allocation, tokenizer)
            cases[0][0]['input_ids'] = list(known) + [7]
            return cases
        with mock.patch.object(custody, 'tokenize_assignment', side_effect=collision), \
             mock.patch.object(commitments, 'seal_heldout', wraps=commitments.seal_heldout) as seal:
            with self.assertRaises(custody.CustodyError):
                custody.verify_prepared(attempt, self.public, url, commitments=commitments, heldout=self.heldout,
                                        tokenizer=FakeTokenizer(), wrapper_bytes=self.wrapper,
                                        observe_inventory=self.observe_inventory, reader=self.reader,
                                        preparation_supervisor=supervisor)
            seal.assert_not_called()
        self.assertFalse((attempt / 'owner-started.json').exists())

    def test_one_existing_trusted_dependency_site_is_accepted(self):
        site = self.root / 'site'
        site.mkdir()
        for initial in ([], [str(site)]):
            with self.subTest(initial=initial), mock.patch.object(sys, 'path', list(initial)):
                custody._install_dependency_site(site)
                self.assertEqual(sys.path.count(str(site)), 1)
        with mock.patch.object(sys, 'path', [str(site), str(site)]):
            with self.assertRaises(custody.CustodyError):
                custody._install_dependency_site(site)

    def test_current_inventory_change_after_claim_stops_before_nonce(self):
        def changed():
            self.calls.append('observe')
            return {'inventory_sha256': exposures.inventory_digest() if not self.ledger.claimed else 'f' * 64,
                    'source_sha256': hashlib.sha256(self.inventory_source).hexdigest()}
        with self.assertRaises(custody.CustodyError):
            self.prepare(observe_inventory=changed)
        self.assertTrue(self.ledger.claimed)
        self.assertNotIn('nonce', self.calls)
        attempt = self.root / ('natural-heldout-' + self.freeze.sha256)
        self.assertEqual(json.loads((attempt / 'preparation-failure.json').read_bytes())['status'], 'incomplete')

    def test_stale_inventory_is_rejected_before_claim(self):
        with mock.patch.object(custody, 'CURRENT_INVENTORY_SHA256', 'f' * 64):
            with self.assertRaises(custody.CustodyError):
                self.prepare()
        self.assertFalse(self.ledger.claimed)
        self.assertNotIn('nonce', self.calls)

    def test_phase_accounting_includes_unrecognized_and_partial_files(self):
        attempt = self.root / 'accounting'
        tensors = attempt / 'owner' / 'attempt' / 'tensors'
        tensors.mkdir(parents=True)
        (tensors / 'partial').write_bytes(b'12345')
        (tensors.parent / 'events.jsonl').write_bytes(b'123')
        (attempt / 'orphan.tmp').write_bytes(b'1234567')
        self.assertEqual(custody.account_phase(attempt),
                         {'tensor_bytes': 5, 'event_bytes': 3, 'side_bytes': 7})
        with mock.patch.object(custody, 'SIDE_LIMIT', 6):
            with self.assertRaises(custody.CustodyError):
                custody.account_phase(attempt)

    def test_phase_accounting_rejects_failed_directory_enumeration(self):
        attempt = self.root / 'accounting'
        hidden = attempt / 'unreadable'
        hidden.mkdir(parents=True)
        (hidden / 'partial').write_bytes(b'not counted if enumeration is skipped')
        original = os.scandir
        def failing(path):
            if Path(path) == hidden:
                raise PermissionError('synthetic failure')
            return original(path)
        with mock.patch.object(os, 'scandir', side_effect=failing):
            with self.assertRaises(custody.CustodyError):
                custody.account_phase(attempt)

    def test_fresh_runtime_rejects_preimported_numerical_dependencies(self):
        with mock.patch.dict(sys.modules, {'torch.fake': types.ModuleType('torch.fake')}):
            with self.assertRaises(custody.CustodyError):
                custody._require_fresh_dependencies()

    def test_cache_staging_is_counted_and_free_space_fails_closed(self):
        cache = self.root / 'cache'
        stage = cache / 'looking-glass-stage-fake'
        stage.mkdir(parents=True)
        (stage / 'partial').write_bytes(b'123456')
        usage = SimpleNamespace(free=custody.REQUIRED_FREE_BYTES + 1)
        with mock.patch.object(custody.shutil, 'disk_usage', return_value=usage):
            result = custody.inspect_cache_staging(cache)
        self.assertEqual(result['stages'], [{'name': stage.name, 'files': 1, 'apparent_bytes': 6}])
        with mock.patch.object(custody.shutil, 'disk_usage',
                               return_value=SimpleNamespace(free=custody.REQUIRED_FREE_BYTES - 1)):
            with self.assertRaises(custody.CustodyError):
                custody.inspect_cache_staging(cache)

    def test_cache_and_evidence_volumes_are_checked_separately(self):
        cache = self.root / 'cache'
        evidence = self.root / 'evidence'
        cache.mkdir()
        evidence.mkdir()
        def usage(path):
            free = custody.REQUIRED_FREE_BYTES - 1 if Path(path) == evidence else custody.REQUIRED_FREE_BYTES + 1
            return SimpleNamespace(free=free)
        with mock.patch.object(custody.shutil, 'disk_usage', side_effect=usage):
            custody.inspect_cache_staging(cache)
            with self.assertRaises(custody.CustodyError):
                custody.inspect_evidence_destination(evidence)

    def test_root_and_private_file_reparse_points_are_rejected(self):
        root = self.root
        file = root / 'evidence'
        file.write_bytes(b'x')
        original = Path.lstat
        def fake_stat(path):
            info = original(path)
            if path == root:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info
        with mock.patch.object(Path, 'lstat', fake_stat):
            with self.assertRaises(custody.CustodyError):
                custody.account_phase(root)
            with self.assertRaises(custody.CustodyError):
                custody._read_private(file, 100)

    def test_modified_frozen_bootstrap_is_rejected_before_execution(self):
        source = self.root / 'controller.py'
        marker = self.root / 'executed'
        source.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("bad")\n',
                          encoding='utf-8')
        loader = custody._FrozenSourceLoader(source, hashlib.sha256(b'original source').hexdigest())
        with self.assertRaises(custody.CustodyError):
            loader.exec_module(types.ModuleType('untrusted_controller'))
        self.assertFalse(marker.exists())

    def test_live_main_observation_rejects_a_newer_exposure_or_changed_source(self):
        head = 'b' * 40
        listing = [{'name': 'pretrained_exposures_v7.py'}, {'name': 'pretrained_exposures_v8.py'}]
        def api(url):
            return {'sha': head} if url.endswith('/commits/main') else listing
        def source(url, suffix):
            self.assertIn('/' + head + '/', url)
            return self.inventory_source
        observation = custody.observe_current_inventory(api_reader=api, source_reader=source)
        self.assertEqual(observation['head'], head)
        listing.append({'name': 'pretrained_exposures_v9.py'})
        with self.assertRaises(custody.CustodyError):
            custody.observe_current_inventory(api_reader=api, source_reader=source)
        listing.pop()
        with self.assertRaises(custody.CustodyError):
            custody.observe_current_inventory(api_reader=api, source_reader=lambda *_: b'changed')
        listing.extend({'name': f'unrelated-{index}.py'} for index in range(998))
        with self.assertRaises(custody.CustodyError):
            custody.observe_current_inventory(api_reader=api, source_reader=source)

    def test_rejects_nonimmutable_public_urls(self):
        for url in ('http://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'a' * 40 + '/x',
                    'https://raw.githubusercontent.com/EastonSell/model-looking-glass/main/x',
                    'https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'a' * 40 + '/x?y=1'):
            with self.subTest(url=url), self.assertRaises(custody.CustodyError):
                custody._official_url(url, 'x')


if __name__ == '__main__':
    unittest.main()
