"""Fabricated-only tamper tests for the independent saved-journal reader."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PUBLISHED_READER = HERE / 'flan_sentiment_intervention_independent_audit.py'
reader = module('flan_independent_reader', PUBLISHED_READER if PUBLISHED_READER.is_file()
                else HERE / 'audit_flan_sentiment_intervention.py')
fake = module('flan_independent_cases', HERE / 'review_flan_sentiment_linear_toy.py')


def fake_committed_selection():
    screens = []
    source_groups = fake.groups()
    lookup = {}
    subjects = ('apple', 'pear', 'plum', 'mango')
    adjectives = (('bright', 'dim', 'warm', 'cold'), ('nice', 'vile', 'sweet', 'bitter'),
                  ('lively', 'boring', 'pleasant', 'ugly'), ('happy', 'sad', 'fine', 'awful'))
    for n in range(12):
        eligible = 1 <= n <= 4
        instructions = []
        for i in range(2):
            source = source_groups[2 * ((n - 1) % 4) + i]
            cases = []
            for position, side in enumerate(reader.SIDES):
                case = copy.deepcopy(source['cases'][side])
                j = (n - 1) % 4
                prefix = reader.INSTRUCTIONS[i] + ' The '
                subject, adjective = subjects[j], adjectives[j][position]
                case['text'] = f'{prefix}{subject} was {adjective}.'
                subject_start = len(prefix)
                adjective_start = subject_start + len(subject) + len(' was ')
                case['offset_mapping'] = [[0, 1], [subject_start - 1, subject_start + len(subject)],
                                          [adjective_start - 1, adjective_start + len(adjective)], [0, 0]]
                lookup[case['text']] = {k: copy.deepcopy(case[k]) for k in
                                        ('input_ids', 'attention_mask', 'special_tokens_mask', 'offset_mapping')}
                cases.append(case)
            instructions.append({'instruction_index': i, 'eligible': eligible,
                                 'failure_reasons': [] if eligible else ['fabricated-ineligible'],
                                 'cases': cases})
        screens.append({'unit_id': f'u{n:02d}', 'semantic_ok': eligible,
                        'lexical_exclusion_ok': eligible, 'instructions': instructions})
    chosen = [f'u{x:02d}' for x in range(1, 5)]
    selection = {'schema': 'flan-sentiment-private-selection-v1', 'pool_commitment': 'a' * 64,
                 'protocol_sha256': '83d666bc67af0951d21eceaa8efb8e7cb16f53a40aefdb72eff03f8b57bd7813',
                 'runner_revision': 'b' * 40,
                 'asset_manifest_sha256': 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083',
                 'exposure_sources': [], 'unit_screens': screens, 'selected_units': chosen,
                 'donor_mapping': [{'unit_id': ident, 'opposite': dict(reader.OPP), 'same': dict(reader.SAME)}
                                   for ident in chosen]}
    salt = b'x' * 32
    def bind(value):
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                               allow_nan=False).encode('utf8')
        return {'pool': 'a' * 64, 'runner_revision': 'b' * 40,
                'selection': hashlib.sha256(b'FLAN-SENTIMENT-SELECTION-V1\0' + salt + b'\0' + canonical).hexdigest()}
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs == {'return_offsets_mapping': True, 'return_special_tokens_mask': True}
            return lookup[text]
    return selection, salt, bind, FakeTokenizer()


class SyntheticJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader.torch.set_num_threads(1)
        cls.temp = Path(tempfile.mkdtemp(prefix='flan-reader-tamper-', dir=HERE)).resolve()
        if not cls.temp.is_relative_to(HERE.resolve()):
            raise ValueError('temporary fixture escaped the review workspace')
        cls.path = cls.temp / 'attempt'
        fixture = Path(os.environ.get('FLAN_SENTIMENT_FAKE_ATTEMPT',
                                      str(HERE / 'flan-sentiment-linear-fake-v1')))
        shutil.copytree(fixture, cls.path)
        cls.events_bytes = (cls.path / 'events.jsonl').read_bytes()
        cls.terminal_bytes = (cls.path / 'terminal.json').read_bytes()
        cls.base_events = [json.loads(line) for line in cls.events_bytes.splitlines()]
        cls.base_terminal = json.loads(cls.terminal_bytes)

    @classmethod
    def tearDownClass(cls):
        if not cls.temp.is_relative_to(HERE.resolve()) or not cls.temp.name.startswith('flan-reader-tamper-'):
            raise ValueError('refusing to remove an unverified fixture directory')
        shutil.rmtree(cls.temp)

    def setUp(self):
        self.events = copy.deepcopy(self.base_events)
        self.terminal = copy.deepcopy(self.base_terminal)
        self.changed_files = {}

    def tearDown(self):
        (self.path / 'events.jsonl').write_bytes(self.events_bytes)
        (self.path / 'terminal.json').write_bytes(self.terminal_bytes)
        for name, data in self.changed_files.items():
            (self.path / name).write_bytes(data)

    def save(self):
        (self.path / 'events.jsonl').write_text(''.join(json.dumps(x, separators=(',', ':'),
                                                                  allow_nan=False) + '\n' for x in self.events),
                                               encoding='utf8', newline='\n')
        (self.path / 'terminal.json').write_text(json.dumps(self.terminal, separators=(',', ':'),
                                                           allow_nan=False) + '\n', encoding='utf8', newline='\n')

    def audit(self):
        return reader.audit(self.path, fake.groups(), fake.Tokenizer(), 2, 8, synthetic=True)

    def rejected(self):
        self.save()
        with self.assertRaises((ValueError, KeyError, TypeError, IndexError)):
            self.audit()

    def first(self, event, predicate=lambda _: True):
        return next(e['detail'] for e in self.events if e['event'] == event and predicate(e['detail']))

    def replace_tensor(self, ref, replacement):
        name = ref['path']
        target = self.path / name
        original = target.read_bytes()
        self.changed_files[name] = original
        self.assertEqual(len(replacement), len(original))
        target.write_bytes(replacement)
        new_sha = hashlib.sha256(replacement).hexdigest()

        def change(value):
            if type(value) is dict:
                if value.get('path') == name and 'sha256' in value:
                    value['sha256'] = new_sha
                for item in value.values():
                    change(item)
            elif type(value) is list:
                for item in value:
                    change(item)
        change(self.events)
        change(self.terminal)

    def test_positive_complete_journal(self):
        result = self.audit()
        self.assertEqual((result['status'], result['evaluations'], result['primary']['passed_units']),
                         ('saved-records-consistent', 484, 4))
        self.assertEqual(len(result['cases']), 32)
        self.assertEqual(sum(len(case['matched']) for case in result['cases']), 256)
        self.assertTrue(all(case['natural_reference'] is not None for case in result['cases']))
        self.assertTrue(all(len(case['natural_distances']) == 4 for case in result['cases']))
        self.assertTrue(all(item['relative_dose_error'] == 0.0
                            for case in result['cases'] for item in case['matched']))

    def test_boolean_selected_attention_mask_is_not_integer(self):
        self.first('evaluation_start')['attention_mask'][0] = True
        self.rejected()

    def test_boolean_decoder_mask_is_not_integer(self):
        self.first('component_start', lambda x: x['kind'] == 'decoder')['decoder_mask'][0] = True
        self.rejected()

    def test_rehashed_rounded_source_with_same_dose_wrong_direction(self):
        geometry = self.first('group_geometry')
        ref = geometry['landings']['P1:opposite_adjective:high']['value_ref']
        self.assertEqual(struct.unpack('<2f', (self.path / ref['path']).read_bytes()), (1., 1.))
        self.assertEqual(abs(5. - 3.), abs(1. - 3.))
        self.replace_tensor(ref, struct.pack('<2f', 5., 1.))
        self.rejected()

    def test_rehashed_untouched_position_change(self):
        row = self.first('evaluation_return', lambda x: x['action']['kind'] == 'matched')
        ref = row['after']
        old = list(struct.unpack('<8f', (self.path / ref['path']).read_bytes()))
        # Perturb an element outside the action's selected position.
        position = row['action']['position']
        index = 0 if position != 0 else 2
        old[index] += 1.
        self.replace_tensor(ref, struct.pack('<8f', *old))
        self.rejected()

    def test_rehashed_nonfinite_untouched_position_is_rejected_as_nonfinite(self):
        row = self.first('evaluation_return', lambda x: x['action']['kind'] == 'clean')
        ref = row['after']
        raw = bytearray((self.path / ref['path']).read_bytes())
        # The final EOS position is not any of the three intervention sites.
        raw[-4:] = struct.pack('<I', 0x7fc00000)
        self.replace_tensor(ref, bytes(raw))
        self.save()
        with self.assertRaisesRegex(ValueError, 'nonfinite saved tensor'):
            self.audit()

    def test_teacher_score_must_match_retained_logits(self):
        self.first('evaluation_return')['teacher']['positive']['score'] += 0.5
        self.rejected()

    def test_total_variation_must_match_full_logits(self):
        self.first('output_comparison')['tv'] = 0.5
        self.rejected()

    def test_outcome_and_terminal_cannot_override_recomputed_decision(self):
        group = self.first('group_outcome', lambda x: x['unit'] == 0 and x['instruction'] == 0)
        group['passes'] = False
        self.terminal['detail']['groups'][0][0]['passes'] = False
        self.terminal['detail']['primary']['passed_units'] = 3
        self.rejected()

    def test_radius_and_action_order_must_match_source(self):
        self.first('group_geometry')['q'] = 3.0
        self.rejected()

    def test_incomplete_has_no_primary_claim(self):
        self.terminal['status'] = 'incomplete'
        self.save()
        report = self.audit()
        self.assertEqual(report['status'], 'incomplete')
        self.assertIsNone(report['primary'])

    def test_fabricated_records_cannot_be_accepted_as_production(self):
        with self.assertRaises(ValueError):
            reader.audit(self.path, fake.groups(), fake.Tokenizer(), 2, 8, synthetic=False)

    def test_inconsistent_synthetic_mode_is_rejected(self):
        self.terminal['detail']['testing_only'] = False
        self.rejected()

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e400}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                reader.strict_json(payload)

    def test_owner_record_and_reader_pin_must_match_expected_commitments(self):
        oracle = reader.load_oracle()
        pool, selected, revision, start_sha = 'a' * 64, 'b' * 64, 'c' * 40, 'd' * 64
        manifest_sha = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
        bindings = {'pool': pool, 'selection': selected, 'runner_revision': revision,
                    'started_sha256': start_sha}
        started = {'schema': 'flan-t5-sentiment-intervention-owner-v1',
                   'source': {'revision': revision, 'sha256': {}}, 'pool_commitment': pool,
                   'selection_commitment': selected, 'asset_manifest_sha256': manifest_sha, 'scope': 'fake'}
        paths = {'scripts/flan_sentiment_intervention_independent_audit.py': Path(reader.__file__),
                 'scripts/flan_sentiment_dose_oracle.py': HERE / 'flan_sentiment_dose_oracle.py',
                 'scripts/bert_taste_matched_independent.py': oracle.helper_path}
        pin = {'schema': 'flan-t5-sentiment-scientific-reader-pin-v1', 'revision': 'e' * 40,
               'sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}}
        review = {'status': 'consistent', 'pool_commitment': pool, 'selection_commitment': selected,
                  'runner_revision': revision, 'asset_manifest_sha256': manifest_sha, 'unit_count': 12,
                  'token_rows': 96, 'selection_count': 4, 'enough_selected': True}
        contract = {'threads': 1, 'interop_threads': 1, 'deterministic': True,
                    'deterministic_warn_only': False, 'float32_matmul_precision': 'highest',
                    'byte_order': 'little', 'python_float_rounds': 1, 'seed': 20261028,
                    'device': 'cpu', 'dtype': 'float32', 'attention': 'eager', 'decoder_cache': False,
                    'screen_replay': review, 'scientific_reader': pin, 'torch': 'fake', 'python': 'fake'}
        settings = {'device': 'cpu', 'num_threads': 1, 'num_interop_threads': 1,
                    'deterministic': True, 'deterministic_warn_only': False, 'default_dtype': 'torch.float32',
                    'float32_matmul_precision': 'highest', 'cpu_autocast': False,
                    'attention_implementation': 'eager', 'config_use_cache': False,
                    'hidden_width': 512, 'vocabulary': 32128, 'decoder_start_id': 0, 'eos_id': 1,
                    'torch_version': 'fake', 'python_version': 'fake'}
        records = {'owner_contract': [contract], 'runtime_observation': [{'settings': settings}]}
        self.assertEqual(reader.audit_contract(started, records, bindings, oracle, start_sha)['pool_commitment'], pool)
        review['selection_commitment'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'selection replay'):
            reader.audit_contract(started, records, bindings, oracle, start_sha)

    def test_salted_selection_derivation_and_tokenizer_checks(self):
        selection, salt, bind, tokenizer = fake_committed_selection()
        groups = reader.selected_groups(selection, salt, bind(selection), tokenizer)
        self.assertEqual(len(groups), 8)
        self.assertIn('apple was bright', groups[0]['cases']['P1']['text'])
        changed = copy.deepcopy(selection)
        changed['selected_units'] = ['u02', 'u01', 'u03', 'u04']
        with self.assertRaisesRegex(ValueError, 'first-four'):
            reader.selected_groups(changed, salt, bind(changed), tokenizer)
        changed = copy.deepcopy(selection)
        changed['donor_mapping'][0]['opposite']['P1'] = 'N2'
        with self.assertRaisesRegex(ValueError, 'donor mapping'):
            reader.selected_groups(changed, salt, bind(changed), tokenizer)
        changed = copy.deepcopy(selection)
        changed['unit_screens'][1]['instructions'][0]['cases'][0]['input_ids'][0] = 4
        with self.assertRaisesRegex(ValueError, 'tokenizer mismatch'):
            reader.selected_groups(changed, salt, bind(changed), tokenizer)
        changed = copy.deepcopy(selection)
        changed['unit_screens'][1]['instructions'][0]['cases'][0]['subject_index'] = 0
        with self.assertRaisesRegex(ValueError, 'subject_index'):
            reader.selected_groups(changed, salt, bind(changed), tokenizer)
        changed = copy.deepcopy(selection)
        changed['unit_screens'][1]['instructions'][0]['cases'][0]['adjective_index'] = 1
        with self.assertRaisesRegex(ValueError, 'adjective_index'):
            reader.selected_groups(changed, salt, bind(changed), tokenizer)
        with self.assertRaisesRegex(ValueError, 'commitment'):
            reader.selected_groups(changed, salt, bind(selection), tokenizer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
