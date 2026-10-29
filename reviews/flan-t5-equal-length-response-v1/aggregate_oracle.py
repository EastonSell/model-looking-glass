"""Exhaustive fabricated check of the six-pair aggregate; no model or real results."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkout', type=Path, required=True)
parser.add_argument('--commit', required=True)
parser.add_argument('--source-sha256', required=True)
args = parser.parse_args()
checkout = args.checkout.resolve()
source = checkout / 'looking_glass/flan_t5_equal_length_response.py'
relative = source.relative_to(checkout).as_posix()


def verify_source():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    if head != args.commit:
        raise ValueError('checkout commit differs')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.source_sha256:
        raise ValueError('source digest differs')
    if raw != subprocess.check_output(['git', 'show', f'{head}:{relative}'], cwd=checkout):
        raise ValueError('working source differs from its Git blob')


verify_source()
sys.path.insert(0, str(checkout))
from looking_glass.flan_t5_equal_length_response import aggregate_decision


def rows(pairs):
    return [{'pair_id': i, 'template_id': template, 'status': status}
            for i, (a, b) in enumerate(pairs)
            for template, status in (('translate', a), ('question', b))]


def expected(pairs):
    positive = sum(a == b == 'positive' for a, b in pairs)
    negative = sum(a in ('positive', 'negative') and b in ('positive', 'negative')
                   and (a == 'negative' or b == 'negative') for a, b in pairs)
    if positive >= 3 and positive > negative:
        return 'positive'
    if negative >= 3 and negative > positive:
        return 'negative'
    return 'inconclusive'


representatives = (('positive', 'positive'), ('negative', 'positive'),
                   ('baseline-unsupported', 'positive'))
checks = 0
for pairs in itertools.product(representatives, repeat=6):
    actual = aggregate_decision(rows(pairs))
    want = expected(pairs)
    if actual != want:
        raise AssertionError((pairs, actual, want))
    checks += 1

# An unevaluable template dominates the partner's positive or negative status;
# either template can supply a negative for a fully evaluable pair.
variants = (('positive', 'negative'), ('negative', 'negative'),
            ('positive', 'baseline-unsupported'), ('negative', 'baseline-unsupported'),
            ('baseline-unsupported', 'negative'), ('dose-ineligible', 'positive'),
            ('positive', 'dose-unmatchable'), ('alignment-ineligible', 'negative'))
base = [('positive', 'positive'), ('positive', 'positive'), ('positive', 'positive'),
        ('negative', 'positive'), ('negative', 'positive'), ('baseline-unsupported', 'positive')]
for position in range(6):
    for replacement in variants:
        pairs = base.copy()
        pairs[position] = replacement
        actual = aggregate_decision(rows(pairs))
        want = expected(pairs)
        if actual != want:
            raise AssertionError((pairs, actual, want))
        checks += 1

verify_source()
print(json.dumps({'status': 'passed', 'checks': checks, 'all_pair_category_combinations': 729,
                  'source_sha256': args.source_sha256, 'implementation_commit': args.commit,
                  'scope': 'fabricated summaries only; no model or actual study evidence'}))
