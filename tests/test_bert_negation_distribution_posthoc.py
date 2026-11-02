"""Checks for the model-free, post-hoc negation-cue distribution replay."""
import base64
import importlib.util
import io
import json
import math
import os
import struct
import subprocess
import sys
import unittest

from looking_glass import bert_negation_distribution_posthoc as subject


class DistributionTests(unittest.TestCase):
    def test_binary64_subnormal_guard_rejects_flush_mode(self):
        subject._require_binary64_subnormals()
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch is unavailable for an isolated flush-mode probe')
        code = '''\
import torch
from looking_glass.bert_negation_distribution_posthoc import _require_binary64_subnormals
if not torch.set_flush_denormal(True):
    raise SystemExit(77)
try:
    _require_binary64_subnormals()
except ValueError:
    pass
else:
    raise SystemExit("flush mode was accepted")
torch.set_flush_denormal(False)
_require_binary64_subnormals()
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
        process = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                                 capture_output=True, text=True, timeout=45)
        if process.returncode == 77:
            self.skipTest('this platform cannot set flush mode')
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_softmax_allows_ordinary_exponential_underflow(self):
        values = [0.0, -1000.0, -2000.0]
        result, _, _ = subject.compare_logits(values, values, any_id=0, some_id=1,
                                               vocabulary=('any', 'some', 'other'))
        self.assertEqual(result['total_variation'], 0.0)

    def test_report_comparison_does_not_hide_small_nonzero_errors(self):
        self.assertFalse(subject._same_nested(1e-100, 1e-16))
        self.assertFalse(subject._same_nested(0.0, 1e-15))
        self.assertFalse(subject._same_nested(1, 1.0))
        self.assertTrue(subject._same_nested(1.0, 1.0 + 5e-13))

    def test_conditional_other_tv_ignores_uniform_rescaling(self):
        clean = [math.log(x) for x in (0.4, 0.3, 0.2, 0.1)]
        changed = [math.log(x) for x in (0.2, 0.15, 0.43333333333333335, 0.21666666666666667)]
        result, _, _ = subject.compare_logits(clean, changed, any_id=0, some_id=1,
                                               vocabulary=('any', 'some', 'alpha', 'beta'))
        self.assertAlmostEqual(result['total_variation'], 0.35)
        self.assertAlmostEqual(result['other_conditional_total_variation'], 0.0)
        self.assertAlmostEqual(result['any_some_absolute_movement_share'], 0.5)

    def test_distribution_metrics_detect_changes_among_other_tokens(self):
        clean = [math.log(x) for x in (0.4, 0.3, 0.2, 0.1)]
        changed = [math.log(x) for x in (0.4, 0.3, 0.1, 0.2)]
        result, _, _ = subject.compare_logits(clean, changed, any_id=0, some_id=1,
                                               vocabulary=('any', 'some', 'alpha', 'beta'))
        self.assertAlmostEqual(result['total_variation'], 0.1)
        self.assertAlmostEqual(result['other_conditional_total_variation'], 1/3)
        self.assertAlmostEqual(result['any_some_margin_change'], 0.0)
        self.assertAlmostEqual(result['any_some_absolute_movement_share'], 0.0)

    def test_rank_ties_choose_lower_token_id(self):
        result, _, _ = subject.compare_logits([2.0, 2.0, 1.0], [1.0, 2.0, 2.0],
                                               any_id=0, some_id=1,
                                               vocabulary=('any', 'some', 'other'))
        self.assertEqual(result['clean']['top'][0]['id'], 0)
        self.assertEqual(result['patched']['top'][0]['id'], 1)
        self.assertEqual(result['clean']['any_rank'], 1)
        self.assertEqual(result['patched']['any_rank'], 3)

    def test_zero_and_opposite_displacements(self):
        self.assertIsNone(subject.displacement_cosine([0.5, 0.5], [0.5, 0.5], [0.7, 0.3]))
        self.assertAlmostEqual(subject.displacement_cosine([0.5, 0.5], [0.7, 0.3], [0.3, 0.7]), -1)
        self.assertAlmostEqual(subject.displacement_cosine([1, 1e-200], [1, 2e-200], [1, 3e-200]), 1)
        result, _, _ = subject.compare_logits([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                               any_id=0, some_id=1, vocabulary=('a', 'b', 'c'))
        self.assertEqual(result['top_other_probability_movers'], [])
        self.assertIsNone(result['any_some_absolute_movement_share'])

    def test_binary32_decoding_keeps_subnormal_and_rejects_nonfinite(self):
        raw = struct.pack('<3I', 1, 0x80000001, 0x3f800000)
        self.assertEqual(subject.decode_f32(raw, 3), (2**-149, -(2**-149), 1.0))
        with self.assertRaises(ValueError):
            subject.decode_f32(struct.pack('<3I', 0x7f800000, 0, 0), 3)

    def test_invalid_tokens_and_nonfinite_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            subject.compare_logits([0., 1., float('nan')], [0., 1., 2.],
                                   any_id=0, some_id=1, vocabulary=('a', 'b', 'c'))
        with self.assertRaises(ValueError):
            subject.compare_logits([0., 1.], [0., 1.], any_id=0, some_id=0, vocabulary=('a', 'b'))


class ScheduleTests(unittest.TestCase):
    def make_records(self):
        cues = {'not': 10, 'never': 11, 'just': 12, 'also': 13}
        case = {'id': 'fake-01', 'split': 'evaluation', 'just_ids': [101, 2, 3, 4, 12, 5]}
        manifest = {'cases': [case], 'cues': cues, 'scored_tokens': {'any': 0, 'some': 1},
                    'indices': {'cue': 4}}
        clean = {'not': [3., 0., 1., 0.], 'never': [2., 0., 1., 0.],
                 'just': [0., 1., 2., 0.], 'also': [0., 2., 1., 0.]}
        enc = lambda x: base64.b64encode(struct.pack('<4f', *x)).decode('ascii')
        row = lambda cue: [101, 2, 3, 4, cues[cue], 5]
        events = []
        for cue in subject.CUES:
            events.append({'kind': 'clean', 'case_id': case['id'], 'cue': cue,
                           'input_ids': row(cue), 'mask_logits_f32le': enc(clean[cue])})
        events.append({'kind': 'radius', 'case_id': case['id'], 'radius': 1.0})
        effects = {}
        for site in subject.SITES:
            for cue in subject.CUES:
                events.append({'kind': 'self', 'case_id': case['id'], 'site': site, 'target': cue,
                               'input_ids': row(cue), 'mask_logits_f32le': enc(clean[cue])})
            entries = []
            for donor, target in subject.TRANSFERS:
                patched = clean[target][:]
                if site != 'cue5':
                    patched[0] += .25
                events.append({'kind': 'transfer', 'case_id': case['id'], 'site': site,
                               'donor': donor, 'target': target, 'input_ids': row(target),
                               'requested_radius': 1.0, 'actual_norm': 1.0, 'relative_error': 0.0,
                               'mask_logits_f32le': enc(patched)})
                entries.append({'donor': donor, 'target': target, 'delta': patched[0] - clean[target][0]})
            effects[site] = entries
        for cue in subject.CUES:
            events.append({'kind': 'repeat', 'case_id': case['id'], 'cue': cue,
                           'input_ids': row(cue), 'mask_logits_f32le': enc(clean[cue])})
        saved = {'case_id': case['id'], 'split': 'evaluation', 'forwards': 56, 'radius': 1.,
                 'clean_margins': {k: v[0] - v[1] for k, v in clean.items()}, 'effects': effects}
        return manifest, saved, events

    def test_schedule_rejects_wrong_identity_and_false_self_control(self):
        manifest, saved, events = self.make_records()
        data = '\n'.join(json.dumps(event) for event in events) + '\n'
        results, controls = subject.scan_case_records(iter(data.splitlines()), manifest['cases'][0], saved,
                                                      manifest['cues'], 0, 1, 4)
        self.assertEqual(len(results), 36)
        self.assertEqual(controls, {'self': 16, 'repeat': 4, 'final_cue': 8})
        broken = list(events)
        broken[6] = {**broken[6], 'target': 'wrong'}
        with self.assertRaises(ValueError):
            subject.scan_case_records(iter(json.dumps(event) for event in broken), manifest['cases'][0], saved,
                                      manifest['cues'], 0, 1, 4)
        broken = list(events)
        broken[5] = {**broken[5], 'mask_logits_f32le': base64.b64encode(struct.pack('<4f', 0, 0, 0, 0)).decode()}
        with self.assertRaises(ValueError):
            subject.scan_case_records(iter(json.dumps(event) for event in broken), manifest['cases'][0], saved,
                                      manifest['cues'], 0, 1, 4)

    def test_selected_subset_roundtrips_and_rejects_corrupted_bytes(self):
        manifest, saved, events = self.make_records()
        records, _ = subject.scan_case_records(iter(json.dumps(x) for x in events), manifest['cases'][0], saved,
                                                manifest['cues'], 0, 1, 4)
        stream = io.BytesIO()
        entries, end = subject.write_subset_case(stream, records, 0, 0, 4)
        self.assertEqual(len(entries), 36)
        self.assertLess(end, 36 * 16)
        restored, position = subject.read_subset_case(io.BytesIO(stream.getvalue()), entries, 0, 0, 4)
        self.assertEqual(position, end)
        self.assertEqual(restored, records)
        bad = bytearray(stream.getvalue())
        bad[0] ^= 1
        with self.assertRaises(ValueError):
            subject.read_subset_case(io.BytesIO(bad), entries, 0, 0, 4)
        altered = [dict(entry) for entry in entries]
        altered[1]['source_event_index'] += 1
        with self.assertRaises(ValueError):
            subject.read_subset_case(io.BytesIO(stream.getvalue()), altered, 0, 0, 4)

    def test_bounded_journal_rejects_extra_or_incomplete_lines(self):
        self.assertEqual(len(list(subject._bounded_lines(io.BytesIO(b'{}\n' * 456)))), 456)
        for data in (b'{}\n' * 456 + b'x', b'{}\n' * 455 + b'{}'):
            with self.subTest(tail=data[-5:]), self.assertRaises(ValueError):
                list(subject._bounded_lines(io.BytesIO(data)))

    def test_analysis_pairs_controls_with_the_same_recipient(self):
        manifest, saved, events = self.make_records()
        records, _ = subject.scan_case_records(iter(json.dumps(x) for x in events), manifest['cases'][0], saved,
                                                manifest['cues'], 0, 1, 4)
        row = subject.analyze_case(records, manifest['cases'][0], saved, 0, 1, ('any', 'some', 'x', 'y'))
        comparisons = row['sites']['cue2']['paired']
        self.assertEqual(len(comparisons), 4)
        for entry in comparisons:
            self.assertEqual(entry['primary']['target'], entry['also_control']['target'])
            self.assertEqual(entry['also_control']['donor'], 'also')
        forward = [entry for entry in comparisons if entry['direction'] == 'negative_to_just']
        self.assertEqual(len(forward), 2)
        self.assertEqual(forward[0]['also_control'], forward[1]['also_control'])
        summary = subject.summarize_cases([row])
        self.assertEqual(summary['evaluation']['cue2']['negative_to_just']['comparisons'], 2)
        self.assertEqual(summary['evaluation']['cue2']['negative_to_just']['contexts'], 1)
        actions = row['sites']['cue2']['actions']
        actions['not->just']['any_some_margin_change'] = .3
        actions['never->just']['any_some_margin_change'] = .1
        actions['also->just']['any_some_margin_change'] = -.2
        scores = subject.summarize_cases([row])['evaluation']['cue2']['negative_to_just']['any_some_margin_change']
        self.assertEqual(scores['primary_greater_than_control'], 2)
        self.assertEqual(scores['primary_positive_and_exceeds_absolute_control'], 1)
        self.assertEqual(scores['control_absolute_at_least_primary'], 1)

    def test_subset_replay_rejects_shifted_clean_even_if_effects_match(self):
        manifest, saved, events = self.make_records()
        records, _ = subject.scan_case_records(iter(json.dumps(x) for x in events), manifest['cases'][0], saved,
                                                manifest['cues'], 0, 1, 4)
        for key, raw in list(records.items()):
            if key == ('clean', 'not') or (key[0] == 'transfer' and key[-1] == 'not'):
                values = list(struct.unpack('<4f', raw))
                values[0] += 1.0
                records[key] = struct.pack('<4f', *values)
        with self.assertRaises(ValueError):
            subject.analyze_case(records, manifest['cases'][0], saved, 0, 1, ('any', 'some', 'x', 'y'))


if __name__ == '__main__':
    unittest.main()
