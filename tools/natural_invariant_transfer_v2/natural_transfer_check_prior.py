import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import natural_transfer_prior as p


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PriorEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ['NATURAL_TOY_JOURNAL'])
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ['NATURAL_TEST_TEMP'])
        cls.root = Path(cls.tmp.name) / 'attempt'
        shutil.copytree(source, cls.root)
        cls.original_events = (cls.root / 'events.jsonl').read_bytes()
        cls.events = [json.loads(x) for x in cls.original_events.splitlines()]
        cls.cases = json.loads(Path(os.environ['NATURAL_CASES']).read_bytes())['cases']
        cls.mapping = json.loads(Path(os.environ['NATURAL_MAPPING']).read_bytes())
        for i, case in enumerate(cls.cases):
            tokens = cls.events[1 + 9 * i]['event']['input_ids']
            case['token_ids'], case['final_index'] = tokens, len(tokens) - 1
        by_id = {x['case_id']: x for x in cls.cases}
        for row in cls.mapping['receivers']:
            case = by_id[row['receiver_case_id']]
            row['receiver_token_ids'], row['final_index'] = case['token_ids'], case['final_index']
        noop = cls.events[2]['tensors']
        cls.width = noop['before']['shape'][0]
        cls.vocab = noop['last_logits']['shape'][0]
        cls.started = sha((cls.root / 'started.json').read_bytes())
        cls.terminal = sha((cls.root / 'terminal.json').read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        (self.root / 'events.jsonl').write_bytes(self.original_events)

    def load(self, event_hash=None, mapping=None):
        return p.load_prior(self.root, self.cases, mapping or self.mapping,
                            events_sha256=event_hash or sha(self.original_events),
                            started_sha256=self.started, terminal_sha256=self.terminal,
                            expected_schedule_sha256=os.environ.get('NATURAL_TOY_SCHEDULE_SHA', p.EXPECTED_SCHEDULE_SHA256),
                            width=self.width, vocabulary=self.vocab)

    def write_events(self, events):
        raw = b''.join((json.dumps(x, sort_keys=True, separators=(',', ':')) + '\n').encode() for x in events)
        (self.root / 'events.jsonl').write_bytes(raw)
        return sha(raw)

    def test_toy_closed_journal_reuses_all_48_baselines(self):
        rows = self.load()
        self.assertEqual(set(rows), {r['receiver_case_id'] for r in self.mapping['receivers']})
        self.assertTrue(all(len(x['vector_f32']) == self.width * 4 for x in rows.values()))
        self.assertTrue(all(len(x['last_logits_f32']) == self.vocab * 4 for x in rows.values()))

    def test_rehashed_wrong_site_or_input_fails(self):
        for field in ('step', 'input'):
            events = copy.deepcopy(self.events)
            if field == 'step':
                events[2]['event']['plan_metadata']['steps'][1]['token_index'] -= 1
            else:
                events[2]['event']['input_ids'][0] += 1
            with self.assertRaises(p.PriorEvidenceError):
                self.load(event_hash=self.write_events(events))

    def test_rehashed_wrong_action_order_fails(self):
        events = copy.deepcopy(self.events)
        events[2]['event']['action'] = 'clean'
        with self.assertRaises(p.PriorEvidenceError):
            self.load(event_hash=self.write_events(events))

    def test_bool_in_integer_metadata_fails_even_when_rehashed(self):
        paths = [
            lambda e: e[0].__setitem__('index', False),
            lambda e: e[1]['event'].__setitem__('index', False),
            lambda e: e[1]['event'].__setitem__('case_index', False),
            lambda e: e[2]['event']['site'].__setitem__('output_selector', [False]),
            lambda e: e[2]['event']['plan_metadata']['steps'][0].__setitem__('order', False),
            lambda e: e[2]['event']['plan_metadata']['steps'][1].__setitem__('batch_index', False),
            lambda e: e[2]['event']['plan_metadata']['steps'][1].__setitem__('invocation', False),
            lambda e: e[2]['tensors']['whole_before']['shape'].__setitem__(0, True),
        ]
        for mutate in paths:
            events = copy.deepcopy(self.events)
            mutate(events)
            with self.assertRaises(p.PriorEvidenceError):
                self.load(event_hash=self.write_events(events))

    def test_rehashed_bad_noop_execution_metadata_fails(self):
        changes = [('dtype','torch.float64'),('device','cuda:0'),('forward_index',1),
                   ('preceding_same_site_actions',[]),('output_shape',[1,1,8]),
                   ('selected_shape',[1,8]),('source_capture','before')]
        for key,value in changes:
            events = copy.deepcopy(self.events)
            events[2]['event']['plan_metadata']['steps'][2][key] = value
            with self.subTest(field=key):
                with self.assertRaises(p.PriorEvidenceError):
                    self.load(event_hash=self.write_events(events))

    def test_changed_tensor_bytes_fail_even_if_journal_is_unchanged(self):
        ref = self.events[2]['tensors']['before']['sha256']
        target = self.root / 'tensors' / (ref + '.f32le')
        original = target.read_bytes()
        try:
            target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaises(p.PriorEvidenceError):
                self.load()
        finally:
            target.write_bytes(original)

    def test_mismatched_receiver_mapping_fails(self):
        mapping = copy.deepcopy(self.mapping)
        mapping['receivers'][0]['receiver_token_ids'][0] += 1
        with self.assertRaises(p.PriorEvidenceError):
            self.load(mapping=mapping)


if __name__ == '__main__':
    unittest.main()
