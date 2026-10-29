"""Model-free, full-logit replay for the planned pairwise color screen.

The v1 helper validates the identical start/completion and f32le envelope
schema. This module supplies the different v2 selection rule. It checks saved
bytes and reported effects, not execution, loaded source or process identity.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_distilbert_color_screen as v1
from review_distilbert_target_metrics import require_denormal_support

PUBLIC_CASE_SHA = 'f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10'
THRESHOLD = Fraction(1, 10)


def factual_preference(factual, alternative):
    if (type(factual) is not float or type(alternative) is not float
            or not math.isfinite(factual) or not math.isfinite(alternative)):
        raise ValueError('nonfinite or invalid selected logits')
    preference = factual - alternative
    if not math.isfinite(preference):
        raise ValueError('nonfinite factual preference')
    # The decimal threshold is exact; use the actual saved binary32 values,
    # rather than a rounded binary64 subtraction or binary64 approximation to
    # 0.10. The rounded value is displayed only as a convenience.
    exact = Fraction.from_float(factual) - Fraction.from_float(alternative)
    return preference, exact > THRESHOLD


def pair_probability_mass(first_logp, second_logp):
    if (type(first_logp) is not float or type(second_logp) is not float
            or not math.isfinite(first_logp) or not math.isfinite(second_logp)):
        raise ValueError('invalid selected log probabilities')
    maximum = max(first_logp, second_logp)
    minimum = min(first_logp, second_logp)
    # Summing individually rounded probabilities can lose a representable
    # subnormal total, or round that total twice. Combine before exponentiation.
    return math.exp(math.fsum((maximum, math.log1p(math.exp(minimum - maximum)))))


def stable_log_probabilities(values, indices):
    """Retain small tail corrections and combine underflowing tails first."""
    maximum = max(values)
    maxima = sum(value == maximum for value in values)
    tails = [value for value in values if value < maximum]
    base = math.log(maxima)
    if tails:
        tail_peak = max(tails)
        tail_log_sum = math.fsum((tail_peak, -maximum,
                                  math.log(math.fsum(math.exp(value - tail_peak) for value in tails))))
        correction = math.log1p(math.exp(tail_log_sum) / maxima)
    else:
        correction = 0.0
    return {index: math.fsum((values[index], -maximum, -base, -correction)) for index in indices}


def audit_bytes(manifest_raw, journal_raw, *, expected_manifest_sha, vocabulary):
    require_denormal_support()
    # This helper's unique-top decision is discarded: v2 eligibility uses the
    # independent pairwise calculation below, even if another token is top.
    result = v1.audit_bytes(manifest_raw, journal_raw,
                            expected_manifest_sha=expected_manifest_sha, vocabulary=vocabulary)
    manifest = v1._json(manifest_raw)
    candidates = {name: True for name in manifest['candidate_order']}
    events = [v1._json(line) for line in journal_raw.splitlines()]
    for index, row in enumerate(result['rows']):
        tokens = row['tokens']
        values = v1._f32(events[2 * index + 1]['logits'], vocabulary)
        indices = {item['id'] for item in tokens.values()} | {row['top_representative']}
        logps = stable_log_probabilities(values, indices)
        for item in tokens.values():
            item['log_probability'] = logps[item['id']]
            item['probability'] = math.exp(item['log_probability'])
        maximum = values[row['top_representative']]
        below = max((value for value in values if value < maximum), default=None)
        row['top_logit'] = maximum
        row['top_gap'] = (0.0 if len(row['top_ids']) > 1 else
                          None if below is None else maximum - below)
        row['top_log_probability'] = logps[row['top_representative']]
        row['top_probability'] = math.exp(row['top_log_probability'])
        preference, eligible = factual_preference(tokens['factual']['logit'],
                                                  tokens['counterfactual']['logit'])
        row['eligible'] = eligible
        row['factual_minus_purple'] = preference
        exact = (Fraction.from_float(tokens['factual']['logit'])
                 - Fraction.from_float(tokens['counterfactual']['logit']))
        row['factual_minus_purple_exact'] = {'numerator': exact.numerator, 'denominator': exact.denominator}
        row['pair_probability_mass'] = pair_probability_mass(tokens['factual']['log_probability'],
                                                             tokens['counterfactual']['log_probability'])
        candidates[row['object']] &= eligible
    selected = next((name for name in manifest['candidate_order'] if candidates[name]), None)
    result['status'] = 'eligible' if selected is not None else 'baseline-ineligible'
    result['selected_object'] = selected
    result['candidate_eligible'] = candidates
    result['selection_rule'] = 'all three exact saved-f32 factual-minus-purple margins strictly exceed 1/10'
    result['scope'] = ('Saved full-logit and input-record consistency only; execution, source, '
                       'process identity and uncited history require separate evidence.')
    require_denormal_support()
    return result


def main():
    require_denormal_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit_bytes(args.manifest.read_bytes(), args.journal.read_bytes(),
                         expected_manifest_sha=PUBLIC_CASE_SHA, vocabulary=30522)
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as output:
        if output.write(payload) != len(payload):
            raise OSError('short report write')
    print('independent screen report SHA-256', hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    main()
