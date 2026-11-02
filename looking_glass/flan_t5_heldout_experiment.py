"""Fixed held-out T5 response rules; no private case data is embedded here."""
import math

from .flan_t5_dose_response import LEVELS, SITES, match_response_levels, response_decision, save_response_dose
from .flan_t5_equal_dose import equal_bits, validate_landing


def baseline_gate(margins, context):
    if len(margins) != 2 or not all(math.isfinite(v) for v in [*margins, context]):
        raise ValueError('nonfinite or malformed baseline')
    first, second = margins
    delta = first - second
    raw_reasons = []
    if first <= 0:
        raw_reasons.append('first-margin-not-positive')
    if second >= 0:
        raw_reasons.append('second-margin-not-negative')
    if delta < 1.0:
        raw_reasons.append('gap-below-one')
    reference_reasons = []
    if first - context <= 0:
        reference_reasons.append('first-not-above-context')
    if second - context >= 0:
        reference_reasons.append('second-not-below-context')
    return {'delta': delta, 'context_relative_margins': [first - context, second - context],
            'gap_supported': delta >= 1.0, 'raw_supported': not raw_reasons,
            'strict_supported': not raw_reasons and not reference_reasons,
            'raw_reasons': raw_reasons, 'context_reasons': reference_reasons}


def track_decisions(cells, key):
    allowed = ('positive', 'negative', 'semantic-ineligible', 'alignment-ineligible',
               'baseline-unsupported', 'gap-ineligible', 'dose-ineligible', 'dose-unmatchable')
    expected = [(i, t) for i in range(6) for t in ('template0', 'template1')]
    if (len(cells) != 12 or [(c.get('pair_id'), c.get('template_id')) for c in cells] != expected
            or any(c.get(key) not in allowed for c in cells)):
        return {'status': 'incomplete', 'pairs': []}
    pairs = []
    for index in range(6):
        statuses = [cell[key] for cell in cells[2*index:2*index+2]]
        if any(status not in ('positive', 'negative') for status in statuses):
            pairs.append('unevaluable')
        elif all(status == 'positive' for status in statuses):
            pairs.append('positive')
        else:
            pairs.append('negative')
    positive, negative = pairs.count('positive'), pairs.count('negative')
    status = ('positive' if positive >= 3 and positive > negative else
              'negative' if negative >= 3 and negative > positive else 'inconclusive')
    return {'status': status, 'pairs': pairs}


def all_track_decisions(cells):
    return {'primary': track_decisions(cells, 'status'), 'raw_gate': track_decisions(cells, 'raw_status'),
            'gap_descriptive': track_decisions(cells, 'descriptive_status')}

def run_cases(manifest, eligibility, evidence, records, active, forward):
    import torch
    summaries = []
    for index, case in enumerate(manifest['cases']):
        active.update(case_id=case['case_id'], forward=None, phase='case')
        summary = {'case_id': case['case_id'], 'pair_id': case['pair_id'], 'template_id': case['template_id']}
        if not case['semantic_eligible']:
            summary.update(status='semantic-ineligible', raw_status='semantic-ineligible',
                           descriptive_status='semantic-ineligible', reasons=case['semantic_reasons'])
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        if eligibility[index]:
            summary.update(status='alignment-ineligible', raw_status='alignment-ineligible',
                           descriptive_status='alignment-ineligible', reasons=eligibility[index])
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        targets = case['targets']
        context = [forward(case['neutral'], target, f'{case["case_id"]}-context-{j}')[0]
                   for j, target in enumerate(targets)]
        bases, states, scores = {}, {}, {}
        for side in range(2):
            for target in range(2):
                score, logits, state, _, _ = forward(case['source'][side], targets[target],
                                                      f'{case["case_id"]}-base-{side}-{target}')
                bases[side, target] = logits
                scores['base', side, target] = score
                if side in states and not equal_bits(states[side], state):
                    raise ValueError('encoder state differs between target forwards')
                states[side] = state
        margins = [scores['base', side, 0] - scores['base', side, 1] for side in range(2)]
        gate = baseline_gate(margins, context[0] - context[1])
        delta = gate['delta']
        summary.update(baseline_margins=margins, delta=delta, context_scores=context,
                       context_margin=context[0] - context[1], baseline_gate=gate)
        if not gate['gap_supported']:
            summary.update(status='baseline-unsupported', raw_status='baseline-unsupported',
                           descriptive_status='gap-ineligible')
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        positions = {'word': case['word_position'], 'instruction': case['instruction_position'],
                     'punctuation': case['context_position']}
        dose, deliveries = match_response_levels(states, positions)
        summary['dose'] = dose
        save_response_dose(evidence, index, case['case_id'], dose, states, positions, deliveries)
        if dose['status'] != 'matchable':
            summary.update(status=dose['status'] if gate['strict_supported'] else 'baseline-unsupported',
                           raw_status=dose['status'] if gate['raw_supported'] else 'baseline-unsupported',
                           descriptive_status=dose['status'])
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue

        def execute(condition, side, action, pos, value, level):
            own, donor = states[side], states[1-side]
            expected = None if pos is None else (torch.zeros_like(own[0, pos]) if action == 'zero' else
                                                  own[0, pos] if action == 'null' else value)
            for target in range(2):
                score, logits, state, before, after = forward(
                    case['source'][side], targets[target], f'{case["case_id"]}-{condition}-{side}-{target}',
                    action, value, pos)
                scores[condition, side, target] = score
                validate_landing(condition, action, pos, own, donor, expected, value,
                                 state, before, after, level)
                if condition in ('repeat', 'null_word', 'self_word', 'self_instruction',
                                 'self_punctuation', 'full_self') and not equal_bits(logits, bases[side, target]):
                    raise ValueError(f'identity logits control failed: {condition}')
                if condition == 'full_swap' and not equal_bits(logits, bases[1-side, target]):
                    raise ValueError('full-swap logits control failed')

        anchors = ('null_word', 'self_word', 'self_instruction', 'self_punctuation',
                   'full_self', 'full_swap', 'full_word_swap', 'zero_word')
        for condition in anchors:
            for side in range(2):
                own, donor = states[side], states[1-side]
                if condition in ('full_self', 'full_swap'):
                    action, pos, value = 'patch', None, own if condition == 'full_self' else donor
                else:
                    site = ('instruction' if condition.endswith('instruction') else
                            'punctuation' if condition.endswith('punctuation') else 'word')
                    pos = positions[site]
                    action = 'null' if condition == 'null_word' else 'zero' if condition == 'zero_word' else 'patch'
                    value = (None if action != 'patch' else donor[0, pos] if condition == 'full_word_swap'
                             else own[0, pos])
                execute(condition, side, action, pos, value, None)

        for level in dose['levels']:
            a, b = level['numerator'], level['denominator']
            for site in SITES:
                condition = f'matched_{site}_{a}of{b}'
                for side in range(2):
                    execute(condition, side, 'patch', positions[site], deliveries[a, b, site, side], level)
        # The final capture repeat follows all interventions to detect state drift.
        for side in range(2):
            execute('repeat', side, 'capture', None, None, None)

        all_conditions = [*anchors, 'repeat'] + [f'matched_{site}_{a}of{b}' for a, b in LEVELS for site in SITES]
        condition_margins = {condition: [scores[condition, side, 0] - scores[condition, side, 1]
                                          for side in range(2)] for condition in all_conditions}
        moves = {condition: [margins[0] - v[0], v[1] - margins[1]]
                 for condition, v in condition_margins.items()}
        curves = {site: [moves[f'matched_{site}_{a}of{b}'] for a, b in LEVELS] for site in SITES}
        status, reasons = response_decision(delta, curves)
        full_curves = {site: [[0.0, 0.0], *curve] for site, curve in curves.items()}
        summary.update(status=status if gate['strict_supported'] else 'baseline-unsupported',
                       raw_status=status if gate['raw_supported'] else 'baseline-unsupported',
                       descriptive_status=status, reasons=reasons, positions=positions, condition_margins=condition_margins,
                       signed_movements=moves,
                       absolute_movements={k: [abs(x) for x in v] for k, v in moves.items()},
                       response_curves=full_curves,
                       absolute_response_curves={site: [[abs(x) for x in row] for row in curve]
                                                 for site, curve in full_curves.items()})
        summaries.append(summary)
        evidence.json(f'case-{index:02d}.json', summary)
    return summaries
