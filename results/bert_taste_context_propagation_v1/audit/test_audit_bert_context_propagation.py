"""Synthetic-only adversarial tests for the independent propagation reader."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('propagation_review', HERE / 'audit_bert_context_propagation.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class SavedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((HERE / 'bert-propagation-toy-fixture-v1.json').read_bytes())

    def audit(self, item):
        return review.audit(item['rows'], {'toy': item['events']}, item['result'], 16, 64, 1, 2)

    def changed(self):
        return copy.deepcopy(self.fixture)

    def reject(self, item, text):
        with self.assertRaisesRegex(ValueError, text):
            self.audit(item)

    def test_complete_synthetic_record(self):
        report = self.audit(self.fixture)
        self.assertEqual((report['forwards'], report['landings'], report['self_controls'],
                          report['retained_full_logit_vectors']), (160, 128, 16, 132))
        self.assertEqual(len(self.fixture['result']['comparisons']), 16)
        self.assertEqual(len(self.fixture['result']['propagation']), 16)

    def test_model_entry_and_donor_identity(self):
        item = self.changed()
        item['events'][1]['model_entry']['input_ids'][2] += 1
        self.reject(item, 'actual recorded model entry')
        item = self.changed()
        index = next(i for i, e in enumerate(item['events']) if e.get('event') == 'forward_start'
                     and e.get('mode') == 'lex_plus')
        for event in item['events'][index:index + 2]:
            event['donor'] = 'toy-with-sugar'
        self.reject(item, 'forward identity donor')

    def test_landings_and_common_radius(self):
        item = self.changed()
        record = next(e for e in item['events'] if e.get('event') == 'forward_complete' and e.get('mode') == 'lex_plus')
        raw = bytearray(review.base64.b64decode(record['source_float32_le_base64']))
        raw[0] ^= 1
        encoded = review.base64.b64encode(raw).decode('ascii')
        record['source_float32_le_base64'] = record['after_float32_le_base64'] = encoded
        self.reject(item, 'source or landing bits differ')
        item = self.changed()
        plan = next(e for e in item['events'] if e.get('event') == 'dose_plan')
        plan['radius'] *= 2
        self.reject(item, 'plan radius')

    def test_full_logits_and_controls(self):
        item = self.changed()
        record = next(e for e in item['events'] if e.get('event') == 'forward_complete' and e.get('mode') == 'lex_plus')
        raw = bytearray(review.base64.b64decode(record['mask_logits_float32_le_base64']))
        raw[0] ^= 1
        record['mask_logits_float32_le_base64'] = review.base64.b64encode(raw).decode('ascii')
        self.reject(item, 'baseline full logits differ')
        item = self.changed()
        record = next(e for e in item['events'] if e.get('event') == 'forward_complete' and e.get('mode') == 'self')
        record['summary']['sweet_minus_sour'] += 1
        self.reject(item, 'score')
        item = self.changed()
        record = next(e for e in item['events'] if e.get('event') == 'forward_complete' and e.get('mode') == 'self')
        record['later_float32_le_base64']['late_the'] = self.fixture['events'][1]['before_float32_le_base64']
        self.reject(item, 'self later capture bits')

    def test_saved_projection_tv_and_later_vectors(self):
        item = self.changed()
        axis = item['result']['comparisons'][0]['axes']['lex']
        axis['later']['late_the']['projection']['coefficient'] += 1
        self.reject(item, 'saved result comparisons')
        item = self.changed()
        item['result']['comparisons'][0]['axes']['orth1']['tv_minus'] += .1
        self.reject(item, 'saved result comparisons')
        item = self.changed()
        item['result']['propagation'][0]['propagation_specificity'] += 1
        self.reject(item, 'saved result propagation')

    def test_eligibility_and_aggregate_are_not_selectively_changed(self):
        item = self.changed()
        item['result']['eligibility_descriptive']['toy-after'] = not item['result']['eligibility_descriptive']['toy-after']
        self.reject(item, 'saved result eligibility_descriptive')
        item = self.changed()
        mean = item['result']['taste_aggregates']['early_ingredient']['taste_specificity']['means']['difference']
        mean['value'] += 1
        self.reject(item, 'saved result taste_aggregates')

    def test_both_denormal_and_order_guards(self):
        item = self.changed()
        item['events'][1]['binary64_after_scoring'] = 'flushed'
        self.reject(item, 'binary64_after_scoring')
        item = self.changed()
        item['events'][-1], item['events'][-3] = item['events'][-3], item['events'][-1]
        self.reject(item, 'forward identity')


class MathTests(unittest.TestCase):
    def test_json_is_not_parsed_before_denormal_guard(self):
        import sys
        with patch.object(sys, 'argv', ['reader', '--fixture', str(HERE / 'bert-propagation-toy-fixture-v1.json'),
                                       '--output', str(HERE / 'unused-output.json')]), \
             patch.object(review.prior, 'require_binary64_subnormals', side_effect=ValueError('flushed')), \
             patch.object(review.json, 'loads', side_effect=AssertionError('parsed before arithmetic guard')):
            with self.assertRaisesRegex(ValueError, 'flushed'):
                review.main()

    def test_odd_even_sign_center_and_projection(self):
        odd, even = review.odd_even((3., 5.), (1., 1.), (2., 2.))
        self.assertEqual(odd, (1., 2.))
        self.assertEqual(even, (0., 1.))
        self.assertEqual(review.centered((11., 13., 15.)), (-2., 0., 2.))
        proj = review.projection((2., -1.), (4., -2.))
        self.assertEqual((proj['coefficient'], proj['residual_norm']), (.5, 0.))
        self.assertAlmostEqual(proj['cosine'], 1.)
        self.assertEqual(review.specificity(-2., 1., -3.), -4.)
        self.assertEqual(review.specificity(2., 1., -3.), 0.)

    def test_softmax_tv_and_null_aggregation(self):
        self.assertAlmostEqual(review.tv((0., 0.), (math.log(3.), 0.)), .25)
        self.assertIsNone(review.projection((1., 0.), (0., 0.))['coefficient'])
        self.assertIsNone(review.projection((0., 0.), (1., 0.))['cosine'])
        self.assertEqual(review.required_mean([1., None]),
                         {'value': None, 'missing_count': 1, 'reason': 'required value unavailable'})
        cells = [{'group': 'g', 'stem': 's', 'template': template, 'x': value}
                 for template, values in [('after', (1., None)), ('with', (2., 4.))] for value in values]
        summary = review.aggregate(cells, ['s'], ['g'], ['x'])['g']['x']
        self.assertIsNone(summary['stems']['s']['difference'])
        self.assertEqual(summary['means']['difference']['missing_count'], 1)
        self.assertEqual(summary['means']['with']['value'], 3.)

    def test_six_stem_paired_aggregation_with_negative_and_null(self):
        stems = [f's{i}' for i in range(6)]
        cells = []
        for i, stem in enumerate(stems):
            for template, values in [('after', (float(i), float(i + 2))),
                                     ('with', (float(i - 3), float(i - 1)))]:
                for value in values:
                    cells.append({'group': 'position', 'stem': stem, 'template': template, 'x': value})
        summary = review.aggregate(cells, stems, ['position'], ['x'])['position']['x']
        self.assertEqual(summary['means']['after']['value'], 3.5)
        self.assertEqual(summary['means']['with']['value'], .5)
        self.assertEqual(summary['means']['difference']['value'], 3.)
        self.assertEqual(summary['stems']['s0']['with']['value'], -2.)
        cells[-1]['x'] = None
        summary = review.aggregate(cells, stems, ['position'], ['x'])['position']['x']
        self.assertIsNone(summary['means']['difference']['value'])
        self.assertEqual(summary['means']['difference']['missing_count'], 1)


if __name__ == '__main__':
    unittest.main()
