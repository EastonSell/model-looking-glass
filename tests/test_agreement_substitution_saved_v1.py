"""Challenge an independent reader against a saved, fabricated toy run."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from looking_glass import agreement_substitution_science_v1 as reader
from looking_glass.agreement_journal import AgreementJournal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_substitution_fake_inputs',
    Path(__file__).with_name('test_agreement_substitution_science_v1.py'))
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)


def create_fabricated_journal(folder, science):
    cases, directions, binding = fake.fabricated_cases(), {
        'u': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        'v1': (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        'v2': (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}, {
        'prelaunch_sha256': '1' * 64, 'reservation_sha256': '2' * 64,
        'latest_exposure_sha256': '3' * 64}
    revision, scope, attempt = 'f' * 40, 'fabricated substitution', folder / 'attempt'
    schedule = reader.expected_schedule(cases)
    schedule_raw = json.dumps(schedule, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=True, allow_nan=False).encode('ascii')
    schedule_sha = hashlib.sha256(schedule_raw).hexdigest()
    placement = {axis + '_' + sign: [sum(x['action'] == axis + '_' + sign and x['slot'] == slot
                                       for x in schedule) for slot in range(6)]
                 for axis in reader.AXES for sign in ('plus', 'minus')}
    width, vocabulary = 8, 600
    success = 486 * vocabulary * 4 + 378 * (2 * 12 * width * 4 + 2 * width * 4)
    failure = 4 * (12 * vocabulary + 2 * 12 * width + 2 * width)
    budget = {'successful_logical_bytes_upper_bound': success, 'failed_return_bytes_reserve': failure,
              'invalid_candidate_bytes_reserve': 2 * width * 4, 'retained_tensor_bytes_limit': 512 * 1024**2,
              'per_forward_bytes_limit': 1024**2, 'event_log_bytes_limit': 32 * 1024**2}
    journal = AgreementJournal(attempt, {'scope': scope, 'external_binding': binding,
        'source_revision': revision}, max_forward_bytes=budget['per_forward_bytes_limit'],
        max_total_bytes=budget['retained_tensor_bytes_limit'], max_event_log_bytes=budget['event_log_bytes_limit'],
        max_events=488)
    state_hash = 'a' * 64
    journal.record({'kind': 'run_contract', 'expected_forwards': 486, 'seed': 20261030,
        'environment': {}, 'execution_configuration': [], 'model_state_sha256': state_hash,
        'external_binding': binding, 'schedule_sha256': schedule_sha, 'arm_placement_counts': placement,
        'evidence_budget': budget}, {})
    measurements = fake.fake_margins(cases)
    by_id = {case['case_id']: case for case in cases}
    base = struct.pack('<8f', *([0.0] * 8))
    margin_ids = {'present': (318, 389), 'past': (373, 547), 'the_vs_a': (262, 257), 'in_vs_on': (287, 319)}
    for item in schedule:
        case, action = by_id[item['case_id']], item['action']
        length = len(case['token_ids'])
        values = [0.0] * vocabulary
        for metric, (positive, negative) in margin_ids.items():
            values[positive], values[negative] = measurements[case['case_id']][action][metric], 0.0
        logits = struct.pack('<600f', *values)
        event = {'kind': 'forward', **item, 'input_ids': case['token_ids'],
                 'attention_mask': [1] * length, 'all_sequence_logits_finite': True,
                 'plan_metadata': {'steps': []}, 'dose': None}
        tensors = {'last_logits': ((vocabulary,), logits)}
        if action not in ('clean', 'clean_after'):
            site, plan = science.expected_hook_plan(action, sequence_length=length, width=width)
            after = base
            if action != 'noop_l3_final':
                axis, sign = action.rsplit('_', 1)
                amount = 2.0 if sign == 'plus' else -2.0
                after, dose = science.direction_delivery(base, directions[axis], amount)
                original = science.decode_float32(base, expected_count=width)
                delivered = science.decode_float32(after, expected_count=width)
                projection = math.fsum((a-b)*d for a, b, d in zip(delivered, original, directions[axis]))
                relative = abs(abs(projection) - 2.0) / 2.0
                dose['signed_projected_displacement'] = projection
                dose['projected_relative_error'] = relative
                dose['dose_valid'] = dose['dose_valid'] and projection * amount > 0 and relative <= 1e-5
                event['dose'] = dose
            event.update({'site': site, 'plan_metadata': plan, 'donor': None})
            whole_before = base * length
            whole_after = base * (length - 1) + after
            tensors.update({'before': ((width,), base), 'after': ((width,), after),
                            'whole_before': ((1, length, width), whole_before),
                            'whole_after': ((1, length, width), whole_after)})
        journal.record(event, tensors)
    journal.finish('complete', {'status': 'complete_candidate', 'forwards': 486,
        'model_state_sha256_before_after': state_hash, 'retained_tensor_bytes': journal.retained_bytes,
        'schedule_sha256': schedule_sha})
    return {'scope': scope, 'source_revision': revision, 'events_sha256': hashlib.sha256(
        (attempt / 'events.jsonl').read_bytes()).hexdigest(), 'binding': binding,
        'cases': cases, 'directions': directions}


class SavedReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='substitution-saved-test-')
        cls.folder = Path(cls.temp.name)
        cls.science = reader.load_numeric(ROOT / 'looking_glass/agreement_science_v1.py')
        cls.fixture = create_fabricated_journal(cls.folder, cls.science)
        cls.records = [json.loads(line) for line in (cls.folder / 'attempt/events.jsonl').read_bytes().splitlines()]
        cls.report = reader.audit_attempt(cls.folder / 'attempt', cls.fixture['cases'], cls.fixture['directions'],
            cls.science, expected_binding=cls.fixture['binding'],
            expected_source_revision=cls.fixture['source_revision'],
            expected_events_sha256=cls.fixture['events_sha256'], expected_scope=cls.fixture['scope'],
            width=8, vocabulary=600, allow_synthetic=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_journal_and_tensor_inventory(self):
        self.assertEqual(self.report['journal']['events'], 487)
        self.assertEqual(len(self.report['cases']), 54)
        self.assertEqual(len(self.report['cells']), 12)
        self.assertEqual(len(self.report['fragments']), 6)
        self.assertEqual(self.report['baseline']['overall']['total'], 48)
        self.assertTrue(all(len(row['vocabulary']) == 9 and len(row['deliveries']) == 6
                            for row in self.report['cases']))

    def test_independent_scalar_oracle_all_cases_and_cells(self):
        pair_ids = {'present': (318, 389), 'past': (373, 547),
                    'the_vs_a': (262, 257), 'in_vs_on': (287, 319)}
        saved = {}
        for record in self.records[1:]:
            event, ref = record['event'], record['tensors']['last_logits']
            raw = (self.folder / 'attempt/tensors' / (ref['sha256'] + '.f32le')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), ref['sha256'])
            values = {name: struct.unpack_from('<f', raw, 4 * a)[0] - struct.unpack_from('<f', raw, 4 * b)[0]
                      for name, (a, b) in pair_ids.items()}
            saved[event['case_id'], event['action']] = values
        odd, actual = {}, {row['case_id']: row for row in self.report['cases']}
        for case in self.fixture['cases']:
            cid = case['case_id']
            for axis in ('u', 'v1', 'v2'):
                for metric in pair_ids:
                    plus, minus, clean = (saved[cid, axis + '_plus'][metric],
                                          saved[cid, axis + '_minus'][metric], saved[cid, 'clean'][metric])
                    o = math.fsum((plus, -minus)) / 2
                    e = math.fsum((plus, minus)) / 2 - clean
                    self.assertEqual(actual[cid]['odd'][axis][metric], o)
                    self.assertEqual(actual[cid]['even'][axis][metric], e)
                    odd[cid, axis, metric] = o
        for cell in self.report['cells']:
            a_s, a_p, b_s, b_p = cell['case_ids']
            for metric in pair_ids:
                for axis in ('u', 'v1', 'v2'):
                    a, b, c, d = (odd[cid, axis, metric] for cid in (a_s, a_p, b_s, b_p))
                    opposite = math.fsum((abs(a-d), abs(c-b))) / 2
                    same = math.fsum((abs(a-c), abs(b-d))) / 2
                    measured = cell['contrasts'][metric][axis]
                    self.assertEqual(measured['opposite'], opposite)
                    self.assertEqual(measured['same'], same)
                    self.assertEqual(measured['delta'], opposite - same)
                    self.assertEqual(measured['number'], math.fsum((a-b, c-d)) / 2)
                    self.assertEqual(measured['lemma'], math.fsum((a-c, b-d)) / 2)
                    self.assertEqual(measured['interaction'], (a-b) - (c-d))

    def test_changed_order_input_finiteness_dose_and_landing_reject(self):
        samples = []
        altered = copy.deepcopy(self.records)
        altered[1]['event']['index'] += 1
        samples.append(altered)
        altered = copy.deepcopy(self.records)
        altered[1]['event']['input_ids'][0] += 1
        samples.append(altered)
        altered = copy.deepcopy(self.records)
        altered[1]['event']['all_sequence_logits_finite'] = False
        samples.append(altered)
        altered = copy.deepcopy(self.records)
        altered[3]['event']['dose']['signed_projected_displacement'] += 1.0
        samples.append(altered)
        altered = copy.deepcopy(self.records)
        altered[3]['tensors']['after'] = altered[3]['tensors']['before']
        samples.append(altered)
        for index, records in enumerate(samples):
            with self.subTest(probe=index):
                with self.assertRaises((ValueError, OSError)):
                    reader.audit_records(records, self.fixture['cases'], self.folder / 'attempt/tensors',
                        self.fixture['directions'], self.science, expected_binding=self.fixture['binding'],
                        width=8, vocabulary=600, allow_synthetic=True)

    def test_missing_record_and_wrong_external_binding_reject(self):
        with self.assertRaises(ValueError):
            reader.audit_records(self.records[:-1], self.fixture['cases'], self.folder / 'attempt/tensors',
                self.fixture['directions'], self.science, expected_binding=self.fixture['binding'],
                width=8, vocabulary=600, allow_synthetic=True)
        with self.assertRaises(ValueError):
            reader.audit_records(self.records, self.fixture['cases'], self.folder / 'attempt/tensors',
                self.fixture['directions'], self.science,
                expected_binding={'prelaunch_sha256': 'e' * 64, 'reservation_sha256': '2' * 64,
                                  'latest_exposure_sha256': '3' * 64},
                width=8, vocabulary=600, allow_synthetic=True)

    def test_failed_or_contradictory_terminal_and_wrong_log_pin_reject(self):
        args = {'expected_binding': self.fixture['binding'],
                'expected_source_revision': self.fixture['source_revision'],
                'expected_events_sha256': self.fixture['events_sha256'],
                'expected_scope': self.fixture['scope'], 'width': 8, 'vocabulary': 600,
                'allow_synthetic': True}
        wrong = dict(args, expected_events_sha256='0' * 64)
        with self.assertRaises(ValueError):
            reader.audit_attempt(self.folder / 'attempt', self.fixture['cases'], self.fixture['directions'],
                                 self.science, **wrong)
        terminal_path = self.folder / 'attempt/terminal.json'
        terminal = json.loads(terminal_path.read_bytes())
        original_read = reader.bounded_file
        for field, value in (('status', 'failed'), ('events', True)):
            corrupted = copy.deepcopy(terminal)
            corrupted[field] = value
            def changed(path, limit):
                if Path(path) == terminal_path:
                    return json.dumps(corrupted, sort_keys=True).encode('ascii')
                return original_read(path, limit)
            with patch.object(reader, 'bounded_file', side_effect=changed):
                with self.assertRaises(ValueError):
                    reader.audit_attempt(self.folder / 'attempt', self.fixture['cases'], self.fixture['directions'],
                                         self.science, **args)


if __name__ == '__main__':
    unittest.main()
