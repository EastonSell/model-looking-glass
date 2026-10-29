"""Bit-defined matched per-head and pair doses for public copying research."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from looking_glass.copy_head_factorial import (
    EvidenceProjectionWitness, MatchedProjectionWitness, SHAPE_BYTES,
    _bits, _norm, _values, f64_to_f32_bits, head_slice,
)


HEADS = (2, 7, 10)
PAIRS = ((2, 7), (2, 10), (7, 10))
ARMS = (
    'h2_low', 'h7_low', 'h10_low', 'h2_h7_low', 'h2_h10_low', 'h7_h10_low',
    'h2_match', 'h7_match', 'h10_match',
    'h2_high', 'h7_high', 'h10_high', 'h2_h7_high', 'h2_h10_high', 'h7_h10_high',
)
ACTION_HEADS = {
    **{f'h{head}_{dose}': (head,) for dose in ('low', 'match', 'high') for head in HEADS},
    **{f'h{a}_h{b}_{dose}': (a, b) for dose in ('low', 'high') for a, b in PAIRS},
}


@dataclass(frozen=True)
class SliceDose:
    head: int
    norm: float
    fraction: float
    achieved: float
    changed_elements: int
    changed_bits: int
    before: bytes
    delivered: bytes


@dataclass(frozen=True)
class FamilyPlan:
    m: float
    requested: dict[str, float]
    variants: tuple[dict[str, dict[int, SliceDose]], ...]


def joint_achieved(row: dict[int, SliceDose], pair: tuple[int, int]) -> float:
    """Calculate actual concatenated input L2 from the delivered binary32 bits."""
    if type(row) is not dict or set(row) != set(HEADS) or pair not in PAIRS:
        raise ValueError('unexpected pair or variant dose')
    return _norm(after - before for head in pair
                 for after, before in zip(_values(row[head].delivered), _values(row[head].before)))


def plan_family(noop_inputs: tuple[bytes, ...]) -> FamilyPlan:
    """Plan three single-head doses from four original no-op input witnesses."""
    if type(noop_inputs) is not tuple or len(noop_inputs) != 4:
        raise ValueError('expected four no-op projection inputs')
    originals = [{head: head_slice(raw, head) for head in HEADS} for raw in noop_inputs]
    norms = [{head: _norm(_values(row[head])) for head in HEADS} for row in originals]
    m = min(row[head] for row in norms for head in HEADS)
    if not math.isfinite(m) or m <= 2.0 ** -19:
        raise ValueError('family dose is nonfinite or too small')
    low = 0.5 * m
    requested = {'low': low, 'match': math.sqrt(2.0) * low, 'high': m}
    variants = []
    for source, row_norms in zip(originals, norms):
        planned = {}
        for name in ('low', 'match', 'high'):
            dose = requested[name]
            heads = {}
            for head in HEADS:
                before = source[head]
                old = _values(before)
                fraction = dose / row_norms[head]
                scale = 1.0 - fraction
                after_bits = tuple(f64_to_f32_bits(scale * value) for value in old)
                delivered = struct.pack('<64I', *after_bits)
                after = _values(delivered)
                achieved = _norm(new - previous for new, previous in zip(after, old))
                if abs(achieved - dose) > 1e-5 * dose:
                    raise ValueError('realized slice dose does not match the request')
                before_bits = _bits(before)
                heads[head] = SliceDose(head, row_norms[head], fraction, achieved,
                    sum(left != right for left, right in zip(before_bits, after_bits)),
                    sum((left ^ right).bit_count() for left, right in zip(before_bits, after_bits)),
                    before, delivered)
            planned[name] = heads
        for pair in PAIRS:
            if abs(joint_achieved(planned['low'], pair) - requested['match']) > 1e-5 * requested['match']:
                raise ValueError('low pair total dose does not match requested total')
        variants.append(planned)
    return FamilyPlan(m, requested, tuple(variants))


class PairProjectionWitness(MatchedProjectionWitness):
    """Reuse the reviewed projection hooks for any one head or fixed head pair."""
    def __init__(self, projection, *, expected_before: bytes, deliveries: dict[int, bytes]):
        EvidenceProjectionWitness.__init__(self, projection, sequence_length=11, width=768, heads=12, mode='noop')
        allowed = ({2}, {7}, {10}, {2, 7}, {2, 10}, {7, 10})
        if type(expected_before) is not bytes or len(expected_before) != SHAPE_BYTES:
            raise ValueError('unexpected reference projection input')
        if (type(deliveries) is not dict or set(deliveries) not in allowed
                or any(type(head) is not int for head in deliveries)
                or any(type(raw) is not bytes or len(raw) != 64 * 4 for raw in deliveries.values())):
            raise ValueError('unexpected head deliveries')
        self.expected_before = expected_before
        self.deliveries = dict(deliveries)


def action_order(family_index: int, variant: int) -> tuple[str, ...]:
    if (type(family_index) is not int or not 0 <= family_index < 8
            or type(variant) is not int or not 0 <= variant < 4):
        raise ValueError('family and variant are outside the fixed range')
    shift = (4 * family_index + variant) % 15
    return ARMS[shift:] + ARMS[:shift]


def projection_diagnostics(outputs: dict[str, bytes]) -> dict:
    """Report final projection output changes and pair-minus-singles residuals."""
    if type(outputs) is not dict or set(outputs) != {'noop', *ARMS}:
        raise ValueError('expected every projection output')
    vectors = {}
    for name, raw in outputs.items():
        if type(raw) is not bytes or len(raw) != SHAPE_BYTES:
            raise ValueError('unexpected projection output')
        vectors[name] = _values(raw[10 * 768 * 4:11 * 768 * 4])
    reference = vectors['noop']
    changes = {name: _norm(after - before for after, before in zip(vectors[name], reference))
               for name in ARMS}
    residuals = {}
    for dose in ('low', 'high'):
        residuals[dose] = {}
        for a, b in PAIRS:
            pair = vectors[f'h{a}_h{b}_{dose}']
            first, second = vectors[f'h{a}_{dose}'], vectors[f'h{b}_{dose}']
            residuals[dose][f'{a}_{b}'] = _norm(((p - x) - y) + clean
                for p, x, y, clean in zip(pair, first, second, reference))
    return {'final_output_delta_l2': changes, 'pair_minus_singles_output_residual_l2': residuals}


def summarize(records: list[dict], expected_cases: list[dict] | None = None) -> dict:
    """Compute fixed per-prompt, family and mean outcomes in binary64 order."""
    if type(records) is not list or len(records) != 576:
        raise ValueError('expected 576 records')
    if expected_cases is not None and (type(expected_cases) is not list or len(expected_cases) != 32):
        raise ValueError('expected 32 frozen cases')
    families = []
    for family in range(8):
        segment = records[72 * family:72 * (family + 1)]
        expected = [(v, a) for v in range(4) for a in ('clean', 'noop')]
        expected += [(v, a) for v in range(4) for a in (*action_order(family, v), 'clean_after')]
        if ([(r['variant'], r['action']) for r in segment] != expected
                or any(r['group'] != family + 1 for r in segment)):
            raise ValueError('records do not follow the fixed family schedule')
        rows = [{r['action']: r for r in segment if r['variant'] == v} for v in range(4)]
        for variant, row in enumerate(rows):
            target = row['clean'].get('target_id')
            if type(target) is not int or not 0 <= target < 50_257:
                raise ValueError('invalid case target')
            case_id = row['clean'].get('case_id')
            for record in row.values():
                if record.get('target_id') != target or record.get('case_id') != case_id:
                    raise ValueError('action identity differs within a case')
                score = record.get('score')
                if (type(score) is not dict
                        or any(type(score.get(k)) not in (int, float) or not math.isfinite(score[k])
                               for k in ('target_log_probability', 'target_logit'))
                        or type(score.get('target_rank')) is not int or not 1 <= score['target_rank'] <= 50_257
                        or type(score.get('top_token_id')) is not int or not 0 <= score['top_token_id'] < 50_257):
                    raise ValueError('invalid or nonfinite score')
            if expected_cases is not None:
                case = expected_cases[4 * family + variant]
                if (type(case) is not dict or case.get('case_id') != case_id or case.get('target_id') != target
                        or case.get('group') != family + 1 or case.get('variant') != variant):
                    raise ValueError('records differ from the frozen case')
        result = {'group': family + 1, 'arms': {}, 'interactions': {},
                  'total_match_comparisons': {}, 'total_match_logit_comparisons': {},
                  'pair_comparisons': {}, 'dose_comparisons': {},
                  'clean_target_ranks': [r['clean']['score']['target_rank'] for r in rows],
                  'clean_target_top1': [r['clean']['score']['top_token_id'] == r['clean']['target_id'] for r in rows]}
        for name in ARMS:
            arm = {'target_ranks': [r[name]['score']['target_rank'] for r in rows],
                   'target_top1': [r[name]['score']['top_token_id'] == r[name]['target_id'] for r in rows]}
            for metric in ('target_log_probability', 'target_logit'):
                deltas = [r[name]['score'][metric] - r['clean']['score'][metric] for r in rows]
                shuffle = math.fsum(deltas[1:]) / 3
                arm[metric] = {'case_deltas': deltas, 'repeat_delta': deltas[0],
                               'shuffle_mean_delta': shuffle, 'family_contrast': deltas[0] - shuffle}
            result['arms'][name] = arm
        for dose in ('low', 'high'):
            result['interactions'][dose] = {}
            for a, b in PAIRS:
                key = f'{a}_{b}'
                pair, first, second = (result['arms'][name] for name in
                    (f'h{a}_h{b}_{dose}', f'h{a}_{dose}', f'h{b}_{dose}'))
                item = {}
                for metric in ('target_log_probability', 'target_logit'):
                    d_pair, d_first, d_second = (arm[metric]['family_contrast'] for arm in (pair, first, second))
                    item[metric] = (d_pair - d_first) - d_second
                    item[f'{metric}_case_interactions'] = [
                        (p - x) - y for p, x, y in zip(pair[metric]['case_deltas'],
                            first[metric]['case_deltas'], second[metric]['case_deltas'])]
                result['interactions'][dose][key] = item
            result['pair_comparisons'][dose] = {
                metric: {f'2_10_minus_{key}': result['interactions'][dose]['2_10'][metric]
                         - result['interactions'][dose][key][metric] for key in ('2_7', '7_10')}
                for metric in ('target_log_probability', 'target_logit')}
        for a, b in PAIRS:
            key = f'{a}_{b}'
            result['dose_comparisons'][key] = {
                metric: result['interactions']['high'][key][metric] - result['interactions']['low'][key][metric]
                for metric in ('target_log_probability', 'target_logit')}
            pair = result['arms'][f'h{a}_h{b}_low']
            for metric, output in (('target_log_probability', 'total_match_comparisons'),
                                   ('target_logit', 'total_match_logit_comparisons')):
                result[output][key] = {f'h{head}': pair[metric]['family_contrast']
                    - result['arms'][f'h{head}_match'][metric]['family_contrast'] for head in (a, b)}
        families.append(result)
    metrics = ('target_log_probability', 'target_logit')
    return {
        'families': families,
        'mean_contrast_by_arm': {metric: {name: math.fsum(f['arms'][name][metric]['family_contrast']
            for f in families) / 8 for name in ARMS} for metric in metrics},
        'mean_repeat_delta_by_arm': {metric: {name: math.fsum(f['arms'][name][metric]['repeat_delta']
            for f in families) / 8 for name in ARMS} for metric in metrics},
        'mean_shuffle_delta_by_arm': {metric: {name: math.fsum(f['arms'][name][metric]['shuffle_mean_delta']
            for f in families) / 8 for name in ARMS} for metric in metrics},
        'mean_interactions': {dose: {f'{a}_{b}': {metric: math.fsum(
            f['interactions'][dose][f'{a}_{b}'][metric] for f in families) / 8 for metric in metrics}
            for a, b in PAIRS} for dose in ('low', 'high')},
        'mean_pair_comparisons': {dose: {metric: {key: math.fsum(
            f['pair_comparisons'][dose][metric][key] for f in families) / 8
            for key in ('2_10_minus_2_7', '2_10_minus_7_10')} for metric in metrics}
            for dose in ('low', 'high')},
        'mean_dose_comparisons': {f'{a}_{b}': {metric: math.fsum(
            f['dose_comparisons'][f'{a}_{b}'][metric] for f in families) / 8 for metric in metrics}
            for a, b in PAIRS},
        'mean_total_match_comparisons': {metric: {f'{a}_{b}': {f'h{head}': math.fsum(
            f['total_match_comparisons' if metric == 'target_log_probability'
              else 'total_match_logit_comparisons'][f'{a}_{b}'][f'h{head}'] for f in families) / 8
            for head in (a, b)} for a, b in PAIRS} for metric in metrics},
        'top1_clean': {'repeated': sum(f['clean_target_top1'][0] for f in families),
                       'shuffled': sum(sum(f['clean_target_top1'][1:]) for f in families)},
        'top1_by_arm': {name: {'repeated': sum(f['arms'][name]['target_top1'][0] for f in families),
            'shuffled': sum(sum(f['arms'][name]['target_top1'][1:]) for f in families)} for name in ARMS},
    }
