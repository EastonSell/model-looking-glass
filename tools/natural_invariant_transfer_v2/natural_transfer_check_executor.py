import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

OWNER = Path(os.environ['INVARIANT_TEST_OWNER_SOURCE'])
RUNTIME = Path(os.environ.get('NATURAL_RUNTIME_SOURCE', str(OWNER)))
sys.path.insert(0, str(RUNTIME))

import torch
from looking_glass.agreement_journal import AgreementJournal
import natural_transfer_executor as e
import natural_transfer_prior as p

spec = importlib.util.spec_from_file_location('_prior_toy_tests', OWNER / 'tests/test_agreement_invariant_executor_v1.py')
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepared():
    root = Path(os.environ['NATURAL_TOY_JOURNAL'])
    events = [json.loads(x) for x in (root / 'events.jsonl').read_bytes().splitlines()]
    cases = json.loads(Path(os.environ['NATURAL_CASES']).read_bytes())['cases']
    mapping = json.loads(Path(os.environ['NATURAL_MAPPING']).read_bytes())
    for i, case in enumerate(cases):
        ids = events[1 + 9 * i]['event']['input_ids']
        case['token_ids'], case['final_index'] = ids, len(ids) - 1
    by_id = {row['case_id']: row for row in cases}
    for row in mapping['receivers']:
        case = by_id[row['receiver_case_id']]
        row['receiver_token_ids'], row['final_index'] = case['token_ids'], case['final_index']
    prior = p.load_prior(root, cases, mapping, events_sha256=digest(root/'events.jsonl'),
                         started_sha256=digest(root/'started.json'), terminal_sha256=digest(root/'terminal.json'),
                         expected_schedule_sha256=os.environ.get('NATURAL_TOY_SCHEDULE_SHA', p.EXPECTED_SCHEDULE_SHA256), width=8, vocabulary=1500)
    return mapping, prior, events[0]['event']['model_state_sha256']


class TransferExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping, cls.prior, cls.model_sha = prepared()

    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ['NATURAL_TEST_TEMP'])
        self.model = toy.model()

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.tmp.cleanup()

    def journal(self):
        return AgreementJournal(Path(self.tmp.name)/'attempt', {'scope': 'fabricated'},
                                max_total_bytes=e.RAW_LIMIT, max_forward_bytes=4*1024*1024,
                                max_events=200, max_event_log_bytes=e.METADATA_LIMIT)

    def run_phase(self, journal, prior=None):
        return e.run_phase(self.model, journal, self.mapping, prior or self.prior, 'discovery',
                           prior_model_sha256=self.model_sha,
                           binding={x: '1'*64 for x in ('pre_run_sha256','mapping_sha256',
                                      'prior_receipt_sha256','latest_exposure_sha256')},
                           custody_check=lambda: None)

    def test_fabricated_discovery_schedule_and_landings(self):
        journal = self.journal()
        result = self.run_phase(journal)
        self.assertEqual(result['forwards'], 96)
        self.assertEqual(self.model.forward_calls, 96)
        entries = [json.loads(x) for x in (Path(self.tmp.name)/'attempt/events.jsonl').read_bytes().splitlines()]
        actions = [x for x in entries if x['event']['kind'] == 'forward']
        self.assertEqual(len(actions), 96)
        receivers = [row for row in self.mapping['receivers'] if row['split']=='discovery']
        plans = {}
        attempt = Path(self.tmp.name)/'attempt'
        def payload(entry, name):
            ref = entry['tensors'][name]
            data = (attempt/'tensors'/(ref['sha256']+'.f32le')).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), ref['sha256'])
            return data
        for i, entry in enumerate(actions):
            event = entry['event']
            row, key = receivers[i//4], ('Q1','Q2','Q3','Q4')[i%4]
            case_id = row['receiver_case_id']
            if case_id not in plans:
                c = row['candidates']
                plans[case_id] = e.plan_candidates(self.prior[case_id]['vector_f32'],
                    self.prior[c['Q2']['donor_case_id']]['vector_f32'],
                    self.prior[c['Q1']['donor_case_id']]['vector_f32'],
                    self.prior[c['Q4']['donor_case_id']]['vector_f32'])
            expected = plans[case_id]['candidates'][key]
            self.assertEqual((event['case_id'], event['candidate']), (case_id, key))
            self.assertEqual(event['dose']['certificate'], expected['certificate'])
            self.assertTrue(event['dose']['certificate']['pass'])
            before, after = payload(entry, 'before'), payload(entry, 'after')
            whole_before, whole_after = payload(entry, 'whole_before'), payload(entry, 'whole_after')
            self.assertEqual(before, self.prior[case_id]['vector_f32'])
            self.assertEqual(after, expected['replacement_f32'])
            self.assertEqual(whole_before[-len(before):], before)
            self.assertEqual(whole_after[-len(after):], after)
            self.assertEqual(whole_before[:-len(before)], whole_after[:-len(after)])
            displacement = (attempt/event['dose']['displacement_file']).read_bytes()
            self.assertEqual(displacement, expected['displacement_f64'])
            self.assertEqual(hashlib.sha256(displacement).hexdigest(), event['dose']['displacement_sha256'])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_invalid_certificate_stops_before_first_forward(self):
        original = e.plan_candidates
        def bad(*args):
            result = original(*args)
            result['candidates']['Q1']['certificate']['pass'] = False
            return result
        journal = self.journal()
        with mock.patch.object(e, 'plan_candidates', bad):
            with self.assertRaises(Exception):
                self.run_phase(journal)
        self.assertEqual(self.model.forward_calls, 0)
        events = [json.loads(x)['event'] for x in (Path(self.tmp.name)/'attempt/events.jsonl').read_bytes().splitlines()]
        self.assertIn('invalid_delivery', [x['kind'] for x in events])
        self.assertNotEqual(json.loads((Path(self.tmp.name)/'attempt/terminal.json').read_bytes())['status'], 'complete_candidate')

    def test_receiver_mismatch_stops_before_patch_and_retains_bits(self):
        prior = copy.deepcopy(self.prior)
        first = next(x['receiver_case_id'] for x in self.mapping['receivers'] if x['split']=='discovery')
        raw = prior[first]['vector_f32']
        changed = bytes([raw[0] ^ 1]) + raw[1:]
        prior[first]['vector_f32'], prior[first]['vector_sha256'] = changed, hashlib.sha256(changed).hexdigest()
        journal = self.journal()
        with self.assertRaises(Exception):
            self.run_phase(journal, prior)
        self.assertEqual(self.model.forward_calls, 1)
        events = [json.loads(x) for x in (Path(self.tmp.name)/'attempt/events.jsonl').read_bytes().splitlines()]
        mismatch = [x for x in events if x['event']['kind']=='prior_mismatch']
        self.assertEqual(len(mismatch), 1)
        self.assertNotEqual(mismatch[0]['tensors']['expected']['sha256'], mismatch[0]['tensors']['actual']['sha256'])
        self.assertNotEqual(json.loads((Path(self.tmp.name)/'attempt/terminal.json').read_bytes())['status'], 'complete_candidate')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_partial_sidecar_write_is_counted_after_failure(self):
        def short_write(path, data):
            with open(path, 'xb') as out:
                out.write(data[:13])
            raise OSError('fabricated interrupted write')
        journal = self.journal()
        with mock.patch.object(e, '_write_exclusive', short_write):
            with self.assertRaises(OSError):
                self.run_phase(journal)
        terminal = json.loads((Path(self.tmp.name)/'attempt/terminal.json').read_bytes())
        inventory = terminal['detail']['displacement_inventory']
        self.assertEqual(self.model.forward_calls, 0)
        self.assertEqual(terminal['detail']['displacement_bytes_successfully_written'], 0)
        self.assertTrue(inventory['complete'])
        self.assertEqual(inventory['bytes'], 13)
        self.assertEqual(inventory['files'][0]['bytes'], 13)

    def test_evaluation_requires_frozen_prediction_binding(self):
        journal = self.journal()
        with self.assertRaises(ValueError):
            e.run_phase(self.model, journal, self.mapping, self.prior, 'evaluation',
                        prior_model_sha256=self.model_sha,
                        binding={x: '1'*64 for x in ('pre_run_sha256','mapping_sha256',
                                   'prior_receipt_sha256','latest_exposure_sha256')},
                        custody_check=lambda: None)
        self.assertEqual(self.model.forward_calls, 0)

    def test_insufficient_reserve_never_enters_model(self):
        with mock.patch.object(e, 'RAW_LIMIT', 1000):
            journal = self.journal()
            with self.assertRaises(Exception):
                self.run_phase(journal)
        self.assertEqual(self.model.forward_calls, 0)


if __name__ == '__main__':
    unittest.main()
