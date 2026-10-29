"""Reconstruct the fixed antonym screen from bound saved full-f32 rows."""
from fractions import Fraction
import math

from . import distilbert_color_edit_metrics_v1 as metrics
from . import distilbert_color_edit_records_v1 as records

ORDER = ['box', 'rope', 'drink', 'room']
ROLES = ('discovery', 'calibration-01', 'calibration-02')


def screen_rows(manifest):
    if manifest.get('candidate_order') != ORDER:
        raise ValueError('fixed candidate order differs')
    rows = []
    for obj in ORDER:
        for role in ROLES:
            matching = [row for row in manifest['cases'] if row['object'] == obj and row['role'] == role]
            if len(matching) != 1:
                raise ValueError('missing or duplicate fixed screen case')
            rows.append(matching[0])
    return rows


def run_screen(engine, journal, manifest):
    for row in screen_rows(manifest):
        engine.run(journal, row, phase='screen', condition='original')


def rebuild_screen(manifest, events, *, vocabulary=30522):
    rows = screen_rows(manifest)
    if len(events) != 24:
        raise ValueError('exactly twelve completed screen actions are required')
    decisions, candidates = [], {obj: True for obj in ORDER}
    for index, row in enumerate(rows):
        start, done = events[2 * index:2 * index + 2]
        binding = {'action': index, 'phase': 'screen', 'object': row['object'],
                   'case_id': row['id'], 'condition': 'original'}
        if (type(start.get('seq')) is not int or type(done.get('seq')) is not int or
                type(start.get('action')) is not int or type(done.get('action')) is not int or
                any(type(event.get(key)) is not list for event in (start, done)
                    for key in ('input_ids', 'attention_mask')) or
                any(type(item) is not int for event in (start, done)
                    for key in ('input_ids', 'attention_mask') for item in event[key]) or
                any(type(event.get('mask_index')) is not int for event in (start, done)) or
                start.get('seq') != 2 * index or done.get('seq') != 2 * index + 1 or
                start.get('kind') != 'start' or done.get('kind') != 'complete' or
                not {'parameter', 'parameter_sha256', 'realized_norm'}.issubset(done) or
                done.get('parameter') is not None or done.get('parameter_sha256') is not None or
                type(done.get('realized_norm')) is not float or done['realized_norm'] != 0.0 or
                any(start.get(k) != v or done.get(k) != v for k, v in binding.items()) or
                any(event.get(k) != row[k] for event in (start, done)
                    for k in ('input_ids', 'attention_mask', 'mask_index'))):
            raise ValueError('screen action or input binding differs')
        values = records.decode_raw(done.get('logits'), vocabulary)
        accepted = [row['word_ids'][word] for word in row['accepted']]
        factual = row['word_ids'][row['factual']]
        cue = row['word_ids'][row['counterfactual']]
        wrong = row['word_ids'][row['wrong_target']]
        if not accepted or accepted[0] != factual:
            raise ValueError('accepted factual IDs differ')
        info = metrics.describe(values, accepted + [cue, wrong], accepted_ids=accepted)
        top_logp = metrics.log_probabilities(values)[info['top_representative']]
        margin = Fraction.from_float(values[factual]) - Fraction.from_float(values[cue])
        margin_pass = margin > Fraction(1, 10)
        top_pass = row['role'] != 'discovery' or info['unique_top'] == factual
        eligible = margin_pass and top_pass
        candidates[row['object']] &= eligible
        high, low = sorted((info['log_probabilities'][str(factual)],
                            info['log_probabilities'][str(cue)]), reverse=True)
        pair_logp = high + math.log1p(math.exp(low - high))
        decisions.append({'case_id': row['id'], 'object': row['object'], 'role': row['role'],
                          'eligible': eligible, 'margin_pass': margin_pass,
                          'discovery_top_pass': top_pass, 'factual_minus_cue': float(margin),
                          'margin_exact': {'numerator': margin.numerator, 'denominator': margin.denominator},
                          'unique_top': info['unique_top'], 'top_tie_count': len(info['top_ids']),
                          'top_representative': info['top_representative'], 'top_gap': info['top_gap'],
                          'top_logit': values[info['top_representative']],
                          'top_log_probability': top_logp, 'top_probability': math.exp(top_logp),
                          'top_is_accepted': info['top_is_accepted'], 'factual_rank': info['ranks'][str(factual)],
                          'accepted_nll': info['accepted_nll'], 'ranks': info['ranks'], 'logits': info['logits'],
                          'log_probabilities': info['log_probabilities'], 'probabilities': info['probabilities'],
                          'factual_plus_cue_log_probability': pair_logp,
                          'factual_plus_cue_probability': math.exp(pair_logp)})
    selected = next((obj for obj in ORDER if candidates[obj]), None)
    return {'status': 'eligible' if selected is not None else 'baseline-ineligible',
            'selected_object': selected, 'candidate_eligible': candidates, 'rows': decisions,
            'threshold_nats': 0.1, 'threshold_comparison': 'strictly greater than exact 1/10'}
