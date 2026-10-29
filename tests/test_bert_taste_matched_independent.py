"""Fabricated-only tests for the independent binary32 and dose primitives."""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest

from scripts import bert_taste_matched_independent as audit


def fabricated_journal(sugar_baseline=3.):
    """One eligible fabricated pair, two layers, unequal signed contrasts."""
    sugar = {'id': 'toy-sugar', 'input_ids': [0, 1, 2, 3, 4, 5, 6, 3, 0, 0], 'attention_mask': [1] * 10}
    lemon = {'id': 'toy-lemon', 'input_ids': [0, 1, 2, 3, 4, 5, 6, 4, 0, 0], 'attention_mask': [1] * 10}
    manifest = {'rows': [sugar, lemon], 'tokens': {'sweet': 1, 'sour': 2}}
    captures = {'toy-sugar': {'ingredient': (0., 0.), 'control': (0., 0.)},
                'toy-lemon': {'ingredient': (4., 0.), 'control': (0., 1.)}}
    rows = [sugar, lemon]
    events = []

    def enc(values):
        return base64.b64encode(struct.pack('<' + str(len(values)) + 'f', *values)).decode('ascii')

    def logits(margin):
        values = [0.] * 8
        values[1], values[2] = max(margin, 0.), max(-margin, 0.)
        raw = struct.pack('<8f', *values)
        maximum = max(values)
        denominator = math.fsum(math.exp(value - maximum) for value in values)
        summary = {'sweet_bits': int.from_bytes(raw[4:8], 'little'),
                   'sour_bits': int.from_bytes(raw[8:12], 'little'),
                   'sweet_logit': values[1], 'sour_logit': values[2],
                   'sweet_minus_sour': math.fsum((values[1], -values[2])),
                   'sweet_probability': math.exp(values[1]-maximum)/denominator,
                   'sour_probability': math.exp(values[2]-maximum)/denominator,
                   'top_five_ids': sorted(range(8), key=lambda i: (-values[i], i))[:5],
                   'mask_logits_sha256': hashlib.sha256(raw).hexdigest()}
        return summary, base64.b64encode(raw).decode('ascii')

    def forward(row, layer, site, mode, donor=None):
        position = 7 if site == 'ingredient' else 1
        self_patch = mode.endswith('_self')
        matched = mode.endswith('_matched')
        patched = mode not in ('capture_ingredient', 'capture_control', 'final_clean')
        identity = {'row_id': row['id'], 'layer': layer, 'position': position, 'mode': mode, 'donor': donor}
        events.append({'event': 'forward_start', **identity})
        names = ['full_before', 'before'] + (['patch', 'after', 'full_after'] if patched else [])
        steps = []
        for order, name in enumerate(names):
            whole = name in ('full_before', 'full_after')
            steps.append({'name': name, 'operation': 'patch' if name == 'patch' else 'capture', 'order': order,
                'module_path': f'distilbert.transformer.layer.{layer}',
                'module_aliases': [f'distilbert.transformer.layer.{layer}'], 'forward_index': 0, 'invocation': 0,
                'batch_index': None if whole else 0, 'token_index': None if whole else position,
                'output_selector': [0], 'output_shape': [1, 10, 2],
                'selected_shape': [1, 10, 2] if whole else [2], 'dtype': 'torch.float32', 'device': 'cpu',
                'preceding_same_site_actions': names[:order],
                'source_capture': 'before' if self_patch and name == 'patch' else None})
        base = captures[row['id']][site]
        other = captures[donor][site] if donor and not self_patch else None
        source = None
        dose = actual = None
        if patched and not self_patch:
            if matched:
                fraction = 0.25 if site == 'ingredient' else 1.0
                source = tuple(audit.f32(bits[0]) for bits in struct.iter_unpack('<I', audit.interpolate(
                    base, audit.displacement(other, base), fraction)))
                actual = audit.l2(audit.displacement(source, base))
                base_norm = audit.l2(base)
                dose = {'fraction': fraction, 'requested_norm': 1.0, 'delivered_norm': actual,
                        'relative_error': abs(actual - 1.0), 'recipient_norm': base_norm,
                        'recipient_norm_status': 'zero' if base_norm == 0 else 'positive',
                        'dose_to_recipient_norm': None if base_norm == 0 else actual / base_norm}
            else:
                source = other
        margin = sugar_baseline if row['id'].endswith('sugar') else -3.
        if layer == 0 and matched:
            margin = (1. if site == 'ingredient' else 2.) * (1 if row['id'].endswith('sugar') else -1)
        summary, full_logits = logits(margin)
        full_before = ('a' if row['id'].endswith('sugar') else 'b') * 64
        probe = {'smallest_times_one_bits': 1, 'normal_half_bits': 0x00400000}
        events.append({'event': 'forward_complete', **identity,
            'model_entry': {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']},
            'summary': summary, 'denormal_before': probe, 'denormal_after': probe, 'denormal_after_scoring': probe,
            'baseline_mask_logits_float32_le_base64': full_logits if layer == 0 and mode == 'capture_ingredient' else None,
            'before_float32_le_base64': enc(base),
            'after_float32_le_base64': enc(base if self_patch else source) if patched else None,
            'source_float32_le_base64': None if source is None else enc(source),
            'donor_clean_float32_le_base64': None if other is None else enc(other),
            'full_before_sha256': full_before, 'full_after_sha256': full_before if self_patch else 'c' * 64 if patched else None,
            'untouched_positions_equal': True if patched else None,
            'actual_pre_post_norm': actual, 'dose': dose, 'events': steps})

    for layer in range(2):
        for row in rows:
            for site in ('ingredient', 'control'):
                forward(row, layer, site, 'capture_' + site)
        events.append({'event': 'dose_plan', 'row_id': 'toy-sugar', 'frame': 'toy', 'layer': layer,
                       'radius': 1., 'distances': {'ingredient': 4., 'control': 1.},
                       'fractions': {'ingredient': .25, 'control': 1.}})
        for i, row in enumerate(rows):
            donor = rows[1-i]['id']
            for mode, site, source in (('ingredient_donor', 'ingredient', donor),
                    ('ingredient_self', 'ingredient', row['id']), ('control_self', 'control', row['id']),
                    ('ingredient_matched', 'ingredient', donor), ('control_matched', 'control', donor)):
                forward(row, layer, site, mode, source)
    for row in rows:
        forward(row, 0, 'ingredient', 'final_clean')
    return manifest, events


class IndependentPrimitives(unittest.TestCase):
    def test_signed_zero_subnormals_ties_and_overflow(self):
        for bits in (0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                     0x80800000, 0x3f800000, 0x7f7fffff):
            self.assertEqual(audit.f32_round(audit.f32(bits)), bits)
        self.assertEqual(audit.f32_round(math.ldexp(1., -150)), 0)
        self.assertEqual(audit.f32_round(-math.ldexp(1., -150)), 0x80000000)
        self.assertEqual(audit.f32_round(1. + math.ldexp(1., -24)), 0x3f800000)
        self.assertEqual(audit.f32_round(1. + 3 * math.ldexp(1., -24)), 0x3f800002)
        for invalid in (float('nan'), float('inf'), -float('inf'), math.ldexp(1., 128)):
            with self.assertRaises(ValueError):
                audit.f32_round(invalid)

    def test_random_finite_rounding_against_native_binary32(self):
        rng = random.Random(20261029)
        for _ in range(20000):
            bits = rng.getrandbits(64)
            value = struct.unpack('<d', bits.to_bytes(8, 'little'))[0]
            if not math.isfinite(value):
                continue
            try:
                raw = struct.pack('<f', value)
            except OverflowError:
                with self.assertRaises(ValueError):
                    audit.f32_round(value)
            else:
                self.assertEqual(audit.f32_round(value), int.from_bytes(raw, 'little'))

    def test_fabricated_matched_direction_and_canonical_encoding(self):
        raw = struct.pack('<4I', 0x3f800000, 0, 0x80000000, 1)
        enc = base64.b64encode(raw).decode('ascii')
        self.assertEqual(audit.vector(enc, 4)[0], raw)
        base = (1., 0., 0., 0.)
        direction = audit.displacement((3., 2., 0., 0.), base)
        self.assertEqual(audit.l2(direction), math.sqrt(8.))
        submitted = audit.interpolate(base, direction, 0.5)
        self.assertEqual(submitted, struct.pack('<4f', 2., 1., 0., 0.))
        for bad in (enc + '\n', enc[:-1], base64.b64encode(raw + b'x').decode('ascii')):
            with self.assertRaises(ValueError):
                audit.vector(bad, 4)

    def test_literal_power_squared_has_concrete_one_ulp_difference(self):
        values = tuple(float.fromhex(item) for item in (
            '0x1.8e3cfdp+2', '-0x1.0e5592p+3', '0x1.441b9p+0', '-0x1.0a778bcp+3',
            '0x1.96c3c6ap+1', '-0x1.500d4ap+3', '-0x1.99bf2p+0', '0x1.05299ap+2',
            '0x1.ecd633p+2', '0x1.714a7ep+0', '-0x1.b5a94a4p+2', '0x1.bcffe58p+1',
            '-0x1.fd2e7ep+2', '-0x1.87b5aa6p+2', '-0x1.2e159edcp+2', '0x1.a156f8p-1'))
        powered = math.sqrt(math.fsum(value ** 2 for value in values))
        multiplied = math.sqrt(math.fsum(value * value for value in values))
        self.assertEqual(audit.l2(values), powered)
        self.assertEqual(powered.hex(), '0x1.7c2abfdd72014p+4')
        self.assertEqual(multiplied.hex(), '0x1.7c2abfdd72013p+4')
        self.assertEqual(powered - multiplied, math.ulp(multiplied))

    def test_full_saved_journal_and_corruptions(self):
        manifest, events = fabricated_journal()
        result = audit.audit_records(manifest, events, hidden=2, vocab=8, layer_count=2)
        self.assertEqual(result['completed_forwards'], 30)
        self.assertEqual(result['eligibility'], {'toy': True})
        self.assertEqual(result['layer_summary'][0]['sugar_to_lemon'], {'n': 1, 'mean': 1.0})
        self.assertEqual(result['layer_summary'][0]['lemon_to_sugar'], {'n': 1, 'mean': 1.0})
        self.assertEqual(result['layer_summary'][1]['sugar_to_lemon'], {'n': 1, 'mean': 0.0})
        for mutation in ('source', 'entry', 'dose', 'landing', 'summary', 'control', 'ordering',
                         'post_score_probe', 'recipient_norm_status', 'boolean_top_id', 'self_probability'):
            altered = copy.deepcopy(events)
            action = next(row for row in altered if row.get('event') == 'forward_complete'
                          and row.get('mode') == 'ingredient_matched' and row.get('layer') == 0)
            if mutation == 'source':
                action['donor_clean_float32_le_base64'] = action['before_float32_le_base64']
            elif mutation == 'entry':
                action['model_entry']['input_ids'][7] += 1
            elif mutation == 'dose':
                action['dose']['delivered_norm'] += 1
            elif mutation == 'landing':
                action['after_float32_le_base64'] = action['before_float32_le_base64']
            elif mutation == 'summary':
                action['summary']['sweet_bits'] ^= 1
            elif mutation == 'control':
                row = next(row for row in altered if row.get('mode') == 'control_self'
                           and row.get('event') == 'forward_complete')
                row['after_float32_le_base64'] = action['after_float32_le_base64']
            elif mutation == 'post_score_probe':
                action['denormal_after_scoring']['smallest_times_one_bits'] = 0
            elif mutation == 'recipient_norm_status':
                action['dose']['recipient_norm_status'] = 'positive' if action['dose']['recipient_norm_status'] == 'zero' else 'zero'
            elif mutation == 'boolean_top_id':
                baseline = next(row for row in altered if row.get('mode') == 'capture_ingredient'
                                and row.get('event') == 'forward_complete' and row.get('layer') == 0)
                baseline['summary']['top_five_ids'][0] = True
            elif mutation == 'self_probability':
                row = next(row for row in altered if row.get('mode') == 'ingredient_self'
                           and row.get('event') == 'forward_complete')
                row['summary']['sweet_probability'] = 0.0
            else:
                altered[0], altered[1] = altered[1], altered[0]
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                audit.audit_records(manifest, altered, hidden=2, vocab=8, layer_count=2)

    def test_ineligible_and_tied_baselines_are_retained(self):
        for baseline in (-3., 0.):
            manifest, events = fabricated_journal(sugar_baseline=baseline)
            result = audit.audit_records(manifest, events, hidden=2, vocab=8, layer_count=2)
            self.assertEqual(result['completed_forwards'], 30)
            self.assertEqual(len(result['effects']), 20)
            self.assertEqual(result['eligibility'], {'toy': False})
            self.assertEqual(result['layer_summary'][0]['sugar_to_lemon'], {'n': 0, 'mean': None})

    def test_repeated_nonbaseline_logits_require_identical_summaries(self):
        manifest, events = fabricated_journal()
        ingredient = next(row for row in events if row.get('event') == 'forward_complete'
                          and row.get('mode') == 'ingredient_matched' and row.get('row_id') == 'toy-sugar'
                          and row.get('layer') == 0)
        control = next(row for row in events if row.get('event') == 'forward_complete'
                       and row.get('mode') == 'control_matched' and row.get('row_id') == 'toy-sugar'
                       and row.get('layer') == 0)
        control['summary'] = copy.deepcopy(ingredient['summary'])
        audit.audit_records(manifest, events, hidden=2, vocab=8, layer_count=2)
        control['summary']['sweet_probability'] *= 0.5
        control['summary']['sour_probability'] *= 0.5
        with self.assertRaisesRegex(ValueError, 'identical mask-logit hash'):
            audit.audit_records(manifest, events, hidden=2, vocab=8, layer_count=2)

    def test_probability_sanity_for_unsaved_patched_logits(self):
        _, events = fabricated_journal()
        event = next(row for row in events if row.get('event') == 'forward_complete'
                     and row.get('mode') == 'ingredient_matched' and row.get('row_id') == 'toy-sugar'
                     and row.get('layer') == 0)
        for ps, po in ((0.8, 0.4), (0.1, 0.8), (-0.1, 0.0), (float('nan'), 0.0)):
            changed = copy.deepcopy(event)
            changed['summary'].update(sweet_probability=ps, sour_probability=po)
            with self.subTest(ps=ps, po=po), self.assertRaisesRegex(ValueError, 'target probabilities'):
                audit._summary(changed)
        mismatch = copy.deepcopy(event)
        mismatch['summary'].update(sweet_probability=0.4, sour_probability=0.1)
        with self.assertRaisesRegex(ValueError, 'log odds'):
            audit._summary(mismatch)
        for ps, po in ((0.0, 0.0), (float.fromhex('0x0.0000000000001p-1022'), 0.0)):
            changed = copy.deepcopy(event)
            changed['summary'].update(sweet_probability=ps, sour_probability=po)
            audit._summary(changed)

    def test_closed_files_reject_tampering_and_incomplete_marker(self):
        manifest, events = fabricated_journal()
        report = audit.audit_records(manifest, events, hidden=2, vocab=8, layer_count=2)
        manifest_raw = json.dumps(manifest, sort_keys=True).encode()
        journal_raw = b''.join((json.dumps(event, sort_keys=True, separators=(',', ':')) + '\n').encode()
                               for event in events)
        owner_effects = []
        for item in report['effects']:
            owner = {k: v for k, v in item.items() if k not in ('sweet_bits', 'sour_bits', 'saved_summary')}
            owner['summary'] = item['saved_summary']
            owner_effects.append(owner)
        result = {k: v for k, v in report.items() if k not in ('scope', 'effects')}
        result.update(status='complete', effects=owner_effects,
                      journals={'toy.jsonl': {'size': len(journal_raw), 'sha256': hashlib.sha256(journal_raw).hexdigest()}})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            root = Path(name)
            (root / 'manifest.json').write_bytes(manifest_raw)
            (root / 'toy.jsonl').write_bytes(journal_raw)
            (root / 'result.json').write_text(json.dumps(result), encoding='utf-8')
            kwargs = {'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(), 'hidden': 2, 'vocab': 8, 'layer_count': 2}
            self.assertEqual(audit.audit_files(root / 'manifest.json', root / 'result.json', root, **kwargs)['completed_forwards'], 30)
            (root / 'incomplete.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                audit.audit_files(root / 'manifest.json', root / 'result.json', root, **kwargs)
            (root / 'incomplete.json').unlink()
            # A changed reported effect must fail even with untouched journal hashes.
            result['effects'][0]['effect'] += 1.0
            (root / 'result.json').write_text(json.dumps(result), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit.audit_files(root / 'manifest.json', root / 'result.json', root, **kwargs)
            result['effects'][0]['effect'] -= 1.0
            result['comparisons'][0]['eligible'] = 1
            (root / 'result.json').write_text(json.dumps(result), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit.audit_files(root / 'manifest.json', root / 'result.json', root, **kwargs)


if __name__ == '__main__':
    unittest.main()
