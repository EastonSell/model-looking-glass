"""Fabricated CPU checks for the prospective pairwise dose protocol."""
import copy
from fractions import Fraction
import math
import struct
import sys
import unittest

import torch

from looking_glass.copy_head_pairs_dose_v1 import (
    ARMS, HEADS, PAIRS, PairProjectionWitness, action_order, joint_achieved,
    _norm, plan_family, summarize,
)
from looking_glass.copy_head_factorial import head_slice


class PairDoseTests(unittest.TestCase):
    def test_literal_power_midpoint_and_norm_order(self):
        x = float.fromhex('-0x1.8198ccc000000p+0')
        lower = float.fromhex('0x1.22667986a3e14p+1')
        upper = float.fromhex('0x1.22667986a3e15p+1')
        self.assertEqual(Fraction.from_float(x) ** 2,
                         (Fraction.from_float(lower) + Fraction.from_float(upper)) / 2)
        self.assertEqual(x * x, lower)
        if sys.platform == 'win32' and sys.version_info[:3] == (3, 11, 15):
            self.assertEqual(x ** 2, upper)
        y = float.fromhex('0x1.22bc2caef4053p+0')
        self.assertEqual(_norm((x, y)), math.sqrt(math.fsum((x ** 2, y ** 2))))
        if sys.platform == 'win32' and sys.version_info[:3] == (3, 11, 15):
            self.assertNotEqual(_norm((x, y)), math.sqrt(math.fsum((x * x, y * y))))

    @staticmethod
    def inputs():
        rows = []
        for variant in range(4):
            source = bytearray(11 * 768 * 4)
            for head in HEADS:
                for index in range(64):
                    value = (variant + 1) * (head + 1) * (index + 1) / 1000
                    struct.pack_into('<f', source, (10 * 768 + head * 64 + index) * 4, value)
            rows.append(bytes(source))
        return tuple(rows)

    def test_exact_plan_and_low_pair_total_match(self):
        original = self.inputs()
        plan = plan_family(original)
        self.assertEqual(plan.requested['low'], 0.5 * plan.m)
        self.assertEqual(plan.requested['high'], plan.m)
        self.assertEqual(plan.requested['match'], math.sqrt(2.0) * plan.requested['low'])
        for variant, row in enumerate(plan.variants):
            for name in ('low', 'match', 'high'):
                for head in HEADS:
                    dose = row[name][head]
                    before = struct.unpack('<64f', head_slice(original[variant], head))
                    source_norm = math.sqrt(math.fsum(x * x for x in before))
                    fraction = plan.requested[name] / source_norm
                    expected = b''.join(struct.pack('<f', (1.0 - fraction) * x) for x in before)
                    self.assertEqual(dose.delivered, expected)
                    before_bits = struct.unpack('<64I', dose.before)
                    after_bits = struct.unpack('<64I', dose.delivered)
                    self.assertEqual(dose.changed_elements, sum(a != b for a, b in zip(before_bits, after_bits)))
                    self.assertEqual(dose.changed_bits, sum((a ^ b).bit_count() for a, b in zip(before_bits, after_bits)))
                    self.assertLessEqual(abs(dose.achieved - plan.requested[name]),
                                         1e-5 * plan.requested[name])
            for pair in PAIRS:
                low_total = joint_achieved(row['low'], pair)
                self.assertLessEqual(abs(low_total - plan.requested['match']),
                                     1e-5 * plan.requested['match'])

    def test_nonfinite_and_small_sources_fail_closed(self):
        with self.assertRaises(ValueError):
            plan_family((bytes(11 * 768 * 4),) * 4)
        source = list(self.inputs())
        changed = bytearray(source[0])
        struct.pack_into('<I', changed, (10 * 768 + 2 * 64) * 4, 0x7fc00001)
        source[0] = bytes(changed)
        with self.assertRaises(ValueError):
            plan_family(tuple(source))

    def test_joint_delivery_matches_singles_and_preserves_other_bits(self):
        original = self.inputs()[0]
        plan = plan_family(self.inputs())
        tensor = torch.frombuffer(bytearray(original), dtype=torch.float32).reshape(1, 11, 768)
        for name in ('low', 'high'):
            for pair in PAIRS:
                deliveries = {h: plan.variants[0][name][h].delivered for h in pair}
                layer = torch.nn.Identity()
                with PairProjectionWitness(layer, expected_before=original, deliveries=deliveries) as witness:
                    layer(tensor)
                after = witness.after.numpy().tobytes()
                for head in pair:
                    self.assertEqual(head_slice(after, head), deliveries[head])
                before_bits = struct.unpack('<8448I', original)
                after_bits = struct.unpack('<8448I', after)
                selected = {10 * 768 + h * 64 + i for h in pair for i in range(64)}
                self.assertTrue(all(before_bits[i] == after_bits[i] for i in range(8448) if i not in selected))
                self.assertFalse(layer._forward_pre_hooks or layer._forward_hooks)
        with self.assertRaises(ValueError):
            PairProjectionWitness(torch.nn.Identity(), expected_before=original, deliveries={2.0: bytes(256)})

    def test_order_has_eight_rotations_per_variant_and_balanced_slots(self):
        counts = {arm: [0] * 15 for arm in ARMS}
        for variant in range(4):
            orders = [action_order(family, variant) for family in range(8)]
            self.assertEqual(len(set(orders)), 8)
            for order in orders:
                self.assertEqual(set(order), set(ARMS))
                for index, arm in enumerate(order):
                    counts[arm][index] += 1
        self.assertTrue(all(2 <= count <= 3 for values in counts.values() for count in values))
        with self.assertRaises(ValueError):
            action_order(True, 0)

    def test_summary_keeps_pair_dose_and_target_logit_interactions_distinct(self):
        signals = {
            'h2_low': -1.0, 'h7_low': -2.0, 'h10_low': -3.0,
            'h2_h7_low': -3.25, 'h2_h10_low': -4.5, 'h7_h10_low': -5.75,
            'h2_match': -1.25, 'h7_match': -2.25, 'h10_match': -3.25,
            'h2_high': -2.0, 'h7_high': -4.0, 'h10_high': -6.0,
            'h2_h7_high': -7.0, 'h2_h10_high': -9.5, 'h7_h10_high': -11.0,
        }
        records = []
        for family in range(8):
            order = [(v, a) for v in range(4) for a in ('clean', 'noop')]
            order += [(v, a) for v in range(4) for a in (*action_order(family, v), 'clean_after')]
            for variant, action in order:
                value = signals.get(action, 0.0) if variant == 0 else 0.0
                logit = value * 2.0 + (1.0 if variant == 0 and action == 'h2_h10_low' else 0.0)
                records.append({'group': family + 1, 'variant': variant, 'action': action,
                                'target_id': 3, 'score': {'target_log_probability': value,
                                'target_logit': logit, 'target_rank': 1, 'top_token_id': 3}})
        summary = summarize(records)
        family = summary['families'][0]
        self.assertEqual(family['interactions']['low']['2_10']['target_log_probability'], -0.5)
        self.assertEqual(family['interactions']['low']['2_10']['target_logit'], 0.0)
        self.assertEqual(family['interactions']['high']['2_10']['target_log_probability'], -1.5)
        self.assertEqual(family['total_match_comparisons']['2_10']['h2'], -3.25)
        self.assertEqual(summary['mean_interactions']['low']['2_10']['target_log_probability'], -0.5)
        self.assertEqual(summary['mean_pair_comparisons']['low']['target_log_probability']['2_10_minus_2_7'], -0.25)
        self.assertEqual(summary['mean_dose_comparisons']['2_10']['target_log_probability'], -1.0)
        self.assertEqual(summary['mean_total_match_comparisons']['target_log_probability']['2_10']['h2'], -3.25)
        bad = list(records); bad[8], bad[9] = bad[9], bad[8]
        with self.assertRaises(ValueError):
            summarize(bad)
        bad = copy.deepcopy(records); bad[9]['score']['target_logit'] = float('nan')
        with self.assertRaises(ValueError):
            summarize(bad)
        bad = copy.deepcopy(records); bad[9]['target_id'] += 1
        with self.assertRaises(ValueError):
            summarize(bad)


if __name__ == '__main__':
    unittest.main()
