"""Fabricated-only checks for the independent crossed-template record reader."""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import unittest

import audit_bert_template_direction as audit


FIXTURE = Path(__file__).with_name('bert-template-toy-fixture-v1.json')


def encoded(events):
    return b''.join((json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                    for x in events)


class IndependentAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_bytes())

    def check_fixture(self, fixture):
        return audit.audit({'rows': fixture['rows'], 'selected_stems': ['toy']}, fixture['result'],
                           {'toy': encoded(fixture['events'])}, width=16, vocab=64, sweet_id=1, sour_id=2)

    def test_actual_hook_fixture(self):
        result = self.check_fixture(copy.deepcopy(self.fixture))
        self.assertEqual((result['forwards'], result['matched_actions'], result['self_controls'],
                          result['final_block_controls']), (56, 24, 8, 4))

    def test_hand_calculated_directions_and_interaction(self):
        data = {
            ('after', 'sugar_to_lemon'): (4., -2., 1., -1., 1., -1.),
            ('after', 'lemon_to_sugar'): (-6., 2., -2., 2., -2., 2.),
            ('with', 'sugar_to_lemon'): (3., -1., 2., 0., 1., -1.),
            ('with', 'lemon_to_sugar'): (-2., 2., -1., 1., .5, -.5),
        }
        modes = ('ingredient_toward', 'ingredient_away', 'the_toward', 'the_away', 'orth_plus', 'orth_minus')
        effects = [{'stem': 'toy', 'template': t, 'direction': d, 'mode': mode, 'effect': value}
                   for (t, d), values in data.items() for mode, value in zip(modes, values)]
        comparisons, summaries, means = audit.summarize_effects(['toy'], effects,
                                                                {'toy-after': False, 'toy-with': True})
        self.assertEqual([c['signed_matched_position_contrast'] for c in comparisons], [3., 4., 1., 1.])
        self.assertEqual([c['donor_odd_minus_abs_orth'] for c in comparisons], [2., 2., 1., 1.5])
        self.assertEqual([c['eligible_descriptive'] for c in comparisons], [False, False, True, True])
        s = summaries[0]
        self.assertEqual((s['after'], s['with'], s['template_difference'], s['direction_main'],
                          s['template_direction_interaction']), (3.5, 1., 2.5, -.5, -1.))
        self.assertEqual((s['direction_specificity_after'], s['direction_specificity_with'],
                          s['direction_specificity_template_difference']), (2., 1.25, .75))
        self.assertEqual(means['after'], 3.5)

    def test_probability_order_even_at_underflow(self):
        summary = {'sweet_bits': 0x40000000, 'sour_bits': 0x3f800000,
                   'sweet_logit': 2., 'sour_logit': 1., 'sweet_minus_sour': 1.,
                   'sweet_probability': 0., 'sour_probability': 1e-310,
                   'top_five_ids': [0, 1, 2, 3, 4],
                   'top_five_tokens': ['0', '1', '2', '3', '4'],
                   'mask_logits_sha256': '0' * 64}
        with self.assertRaises(ValueError):
            audit.verify_summary(summary, 1, 2, 64)
        summary['sour_probability'] = 0.
        audit.verify_summary(summary, 1, 2, 64)

    def test_full_baseline_subnormal_probability_is_not_erased(self):
        values = [0.] * 64
        values[1], values[2] = -710., -711.
        raw = b''.join(struct.pack('<f', value) for value in values)
        denominator = math.fsum(math.exp(value) for value in values)
        p, q = math.exp(-710.) / denominator, math.exp(-711.) / denominator
        self.assertGreater(p, 0.)
        summary = {'sweet_bits': 0xc4318000, 'sour_bits': 0xc431c000,
                   'sweet_logit': -710., 'sour_logit': -711., 'sweet_minus_sour': 1.,
                   'sweet_probability': 0., 'sour_probability': 0.,
                   'top_five_ids': [0, 3, 4, 5, 6],
                   'top_five_tokens': ['0', '3', '4', '5', '6'],
                   'mask_logits_sha256': hashlib.sha256(raw).hexdigest()}
        audit.verify_summary(summary, 1, 2, 64)
        with self.assertRaises(ValueError):
            audit.verify_baseline(raw, summary, 64, 1, 2)
        summary['sweet_probability'], summary['sour_probability'] = p, q
        audit.verify_baseline(raw, summary, 64, 1, 2)

    def test_single_record_corruptions_fail(self):
        def first(fixture, mode):
            return next(e for e in fixture['events'] if e['event'] == 'forward_complete' and e['mode'] == mode)

        def bad_input(fixture):
            first(fixture, 'capture_ingredient')['model_entry']['input_ids'][0] += 1

        def boolean_manifest_mask(fixture):
            fixture['rows'][0]['attention_mask'][0] = True

        def bad_start(fixture):
            fixture['events'][0]['position'] = 1

        def bad_donor(fixture):
            e = first(fixture, 'ingredient_donor')
            data = bytearray(base64.b64decode(e['donor_clean_float32_le_base64']))
            data[0] ^= 1
            e['donor_clean_float32_le_base64'] = base64.b64encode(data).decode()

        def wrong_landing(fixture):
            e = first(fixture, 'ingredient_toward')
            data = bytearray(base64.b64decode(e['source_float32_le_base64']))
            data[0:4], data[4:8] = data[4:8], data[0:4]
            e['source_float32_le_base64'] = e['after_float32_le_base64'] = base64.b64encode(data).decode()

        def wrong_dose(fixture):
            first(fixture, 'the_toward')['dose']['delivered_norm'] += .01

        def wrong_meta(fixture):
            first(fixture, 'ingredient_away')['events'][2]['token_index'] = 1

        def failed_control(fixture):
            first(fixture, 'ingredient_self')['untouched_positions_equal'] = False

        def wrong_denormal(fixture):
            first(fixture, 'capture_ingredient')['denormal_after_scoring']['normal_half_bits'] = 0

        def wrong_score(fixture):
            first(fixture, 'orth_plus')['summary']['sweet_bits'] ^= 1

        def out_of_vocab_top_id(fixture):
            first(fixture, 'ingredient_toward')['summary']['top_five_ids'][0] = 999999

        def wrong_zero_sign(fixture):
            # The generic comparator must distinguish binary64 signed zero.
            audit.equal(0., -0., 'signed zero')

        def wrong_same_hash_probability(fixture):
            summary = first(fixture, 'ingredient_self')['summary']
            summary['sweet_probability'] *= .9
            summary['sour_probability'] *= .9

        def wrong_result(fixture):
            fixture['result']['stem_summaries'][0]['template_direction_interaction'] += 1.

        def wrong_comparison_type(fixture):
            fixture['result']['comparisons'][0]['eligible_descriptive'] = 0

        def truncate(fixture):
            fixture['events'].pop()

        for mutate in (bad_input, boolean_manifest_mask, bad_start, bad_donor, wrong_landing, wrong_dose, wrong_meta,
                       failed_control, wrong_denormal, wrong_score, out_of_vocab_top_id, wrong_same_hash_probability,
                       wrong_result, wrong_comparison_type, truncate):
            with self.subTest(case=mutate.__name__):
                sample = copy.deepcopy(self.fixture)
                mutate(sample)
                with self.assertRaises((ValueError, KeyError, TypeError, IndexError)):
                    self.check_fixture(sample)
        with self.assertRaises(ValueError):
            wrong_zero_sign(None)


if __name__ == '__main__':
    unittest.main()
