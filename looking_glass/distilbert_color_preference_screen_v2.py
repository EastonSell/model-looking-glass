"""Rebuild the outcome-informed color-preference screen from saved logits."""
from fractions import Fraction
import math

from . import distilbert_color_edit_records_v1 as records


def rebuild_screen(manifest, events, *, vocabulary=30522):
    """Apply the strict pairwise screen after the reviewed v1 record checks."""
    if manifest.get('candidate_order') != ['banana', 'strawberry', 'lemon']:
        raise ValueError('fixed candidate order differs')
    report = records.rebuild_screen(manifest, events, vocabulary=vocabulary)
    cases = {row['id']: row for row in manifest['cases']}
    if len(cases) != len(manifest['cases']):
        raise ValueError('case IDs must be unique')
    candidates = {obj: True for obj in manifest['candidate_order']}
    for result in report['rows']:
        row = cases[result['case_id']]
        factual = str(row['word_ids'][row['factual']])
        purple = str(row['word_ids'][row['counterfactual']])
        logits = result['logits']
        # Stored f32 logits are exact binary rationals. Compare to exactly 1/10
        # rather than silently moving the strict boundary through float rounding.
        margin = Fraction.from_float(logits[factual]) - Fraction.from_float(logits[purple])
        eligible = margin > Fraction(1, 10)
        result['eligible'] = eligible
        result['factual_minus_purple'] = float(margin)
        result['margin_exact'] = {'numerator': margin.numerator, 'denominator': margin.denominator}
        logps = result['log_probabilities']
        high, low = sorted((logps[factual], logps[purple]), reverse=True)
        pair_logp = high + math.log1p(math.exp(low - high))
        result['factual_plus_purple_log_probability'] = pair_logp
        result['factual_plus_purple_probability'] = math.exp(pair_logp)
        candidates[row['object']] &= eligible
    selected = next((obj for obj in manifest['candidate_order'] if candidates[obj]), None)
    report['candidate_eligible'] = candidates
    report['selected_object'] = selected
    report['status'] = 'eligible' if selected is not None else 'baseline-ineligible'
    report['threshold_nats'] = 0.1
    report['threshold_comparison'] = 'strictly greater than exact 1/10'
    return report
