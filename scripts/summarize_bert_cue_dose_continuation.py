"""Summarize a stopped cue study and a separately audited continuation.

The original attempt remains incomplete. Any combined outcome is a post-stop,
cross-process descriptive calculation and requires a successful direct-exit
observation plus the exact farmer-prefix comparator. This script does not load
a model or authenticate execution; its inputs are independently audited
reports and an externally observed comparator/receipt.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

PREFIX_SHA = '510a79781a37bd6df254d64845911cd63d950bf3b237abaf7d5c447fdd7b6d33'
ORIGINAL_EVENTS_SHA = 'ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05'
MANIFEST_SHA = '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a'
SOURCE = 'cdc9630e24542fed84b0c9556bb7f37c60d722d1'
GROUPS = ('prior', 'new')
TRIPLES = ('fresh-01', 'fresh-02', 'fresh-03')
DOSES = ('low', 'high')
PREFIX_IDS = ('exposed-calibration-prior', 'exposed-calibration-new', 'fresh-01-prior', 'fresh-01-new')
CONTINUED_IDS = ('fresh-02-prior', 'fresh-02-new', 'fresh-03-prior', 'fresh-03-new')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_json(data):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj
    def invalid(value):
        raise ValueError('invalid JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def normalize(row, ident, origin):
    if type(row.get('baseline_eligible')) is not bool or type(row.get('doses')) is not dict:
        raise ValueError('invalid baseline or dose data')
    if set(row['doses']) != set(DOSES):
        raise ValueError('unexpected dose labels')
    doses = {}
    for label in DOSES:
        dose = row['doses'][label]
        if type(dose.get('eligible')) is not bool:
            raise ValueError('dose eligibility must be boolean')
        eligible, value, passed = dose['eligible'], dose.get('cue_min_margin'), dose.get('specificity_passed')
        if eligible:
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    type(passed) is not bool or passed != (value > .05)):
                raise ValueError('invalid saved specificity result')
        elif value is not None or passed is not None:
            raise ValueError('ineligible dose has an outcome')
        doses[label] = {'eligible': eligible, 'cue_min_margin': value, 'specificity_passed': passed}
    if doses['high']['eligible'] and not doses['low']['eligible']:
        raise ValueError('high dose cannot be eligible without low dose')
    return {'case_id': ident, 'origin': origin, 'baseline_eligible': row['baseline_eligible'], 'doses': doses}


def summarize(rows, group, triples):
    cells = {}
    for label in DOSES:
        eligible = [rows[t, group] for t in triples if rows[t, group]['baseline_eligible'] and
                    rows[t, group]['doses'][label]['eligible']]
        passing = sum(x['doses'][label]['specificity_passed'] is True for x in eligible)
        cells[label] = {'total': len(triples), 'eligible': len(eligible), 'passing': passing,
                        'status': 'unevaluable' if len(eligible) < 2 else 'passed' if passing >= 2 else 'failed'}
    values = []
    transitions = {'fail_to_pass': 0, 'pass_to_fail': 0, 'pass_to_pass': 0, 'fail_to_fail': 0}
    for triple in triples:
        row = rows[triple, group]
        low, high = row['doses']['low'], row['doses']['high']
        if row['baseline_eligible'] and low['eligible'] and high['eligible']:
            transition = ('pass' if low['specificity_passed'] else 'fail') + '_to_' + (
                'pass' if high['specificity_passed'] else 'fail')
            transitions[transition] += 1
            values.append({'triple_id': triple, 'low_margin': low['cue_min_margin'],
                           'high_margin': high['cue_min_margin'],
                           'difference': math.fsum((high['cue_min_margin'], -low['cue_min_margin'])),
                           'transition': transition})
    n = len(values)
    paired = {'eligible': n, 'transitions': transitions, 'values': values,
              'status': ('unevaluable' if n < 2 else 'changed' if
                         transitions['fail_to_pass'] or transitions['pass_to_fail'] else 'unchanged'),
              'mean': math.fsum(x['difference'] for x in values) / n if n >= 2 else None}
    return {'cells': cells, 'paired': paired}


def calculate(prefix, continuation=None, comparison=None, receipt=None):
    if (prefix.get('status') != 'incomplete_prefix_validated' or prefix.get('study_outcome') != 'incomplete' or
            prefix.get('events') != 509 or prefix.get('saved_forwards') != 499 or
            prefix.get('partial_case_id') != 'fresh-02-prior' or
            tuple(x.get('case_id') for x in prefix.get('completed_cases', [])) != PREFIX_IDS):
        raise ValueError('original prefix report is not the fixed stopped attempt')
    cases = {x['case_id']: normalize(x, x['case_id'], 'original-completed')
             for x in prefix['completed_cases']}
    reasons = []
    if continuation is None:
        reasons.append('continued full-witness report unavailable')
    else:
        if (continuation.get('status') != 'passed' or continuation.get('synthetic') is not False or
                continuation.get('source_commit') != SOURCE or continuation.get('manifest_sha256') != MANIFEST_SHA or
                tuple(x.get('id') for x in continuation.get('cases', [])) != CONTINUED_IDS):
            raise ValueError('continuation report is not the fixed audited four-cell attempt')
        for row in continuation['cases']:
            ident = row['id']
            cases[ident] = normalize(row, ident, 'continuation-partially-exposed' if ident == CONTINUED_IDS[0]
                                      else 'continuation-not-recorded-in-original-prefix')
    if comparison is None:
        reasons.append('farmer-prefix comparison unavailable')
    elif (comparison.get('matched') is not True or comparison.get('reasons') != [] or
          any(type(comparison.get(key)) is not int for key in
              ('prefix_records_expected', 'original_suffix_records_capped_at_22',
               'continuation_prefix_records_available')) or
          comparison.get('prefix_records_expected') != 21 or
          comparison.get('original_suffix_records_capped_at_22') != 21 or
          comparison.get('continuation_prefix_records_available') != 21 or
          comparison.get('original_sha256') != ORIGINAL_EVENTS_SHA or continuation is None or
          comparison.get('continuation_sha256') != continuation.get('events_sha256')):
        reasons.append('farmer-prefix comparison did not pass or bind both journals')
    if receipt is None:
        reasons.append('direct-process exit observation unavailable')
    elif (receipt.get('source_commit') != SOURCE or receipt.get('process_waited') is not True or
          type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0 or
          ('timed_out' in receipt and receipt['timed_out'] is not False)):
        reasons.append('direct-process exit observation was not a successful wait')
    result = {'original_attempt': 'incomplete', 'pooling': 'unavailable' if reasons else 'available',
              'no_pool_reasons': reasons, 'cases': cases, 'pooled': None, 'sensitivity_excluding_farmer_prior': None,
              'scope': 'post-stop cross-process descriptive calculation; not the original one-shot result, '
                       'blind confirmation, population inference or execution attestation'}
    if not reasons:
        rows = {(t, g): cases[t + '-' + g] for t in TRIPLES for g in GROUPS}
        result['pooled'] = {g: summarize(rows, g, TRIPLES) for g in GROUPS}
        result['sensitivity_excluding_farmer_prior'] = summarize(rows, 'prior', ('fresh-01', 'fresh-03'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix-report', type=Path, required=True)
    for name in ('continuation-report', 'comparison', 'receipt'):
        p.add_argument('--' + name, type=Path)
        p.add_argument('--' + name + '-sha256')
    p.add_argument('--continuation-events-sha256')
    p.add_argument('--continuation-result-sha256')
    args = p.parse_args()
    paths = {'prefix_report': args.prefix_report}
    data = {'prefix_report': strict_json(args.prefix_report.read_bytes())}
    before = {'prefix_report': sha(args.prefix_report)}
    if before['prefix_report'] != PREFIX_SHA:
        raise ValueError('original audited prefix report hash differs')
    for name in ('continuation_report', 'comparison', 'receipt'):
        path, expected = getattr(args, name), getattr(args, name + '_sha256')
        if path is None and expected is None:
            data[name] = None
            continue
        if path is None or not re.fullmatch('[0-9a-f]{64}', expected or ''):
            raise ValueError('each supplied input requires its observed SHA-256')
        paths[name] = path
        before[name] = sha(path)
        if before[name] != expected:
            raise ValueError('input hash differs: ' + name)
        data[name] = strict_json(path.read_bytes())
    if data.get('continuation_report') is not None:
        for field, expected in (('events_sha256', args.continuation_events_sha256),
                                ('result_sha256', args.continuation_result_sha256)):
            if (not re.fullmatch('[0-9a-f]{64}', expected or '') or
                    data['continuation_report'].get(field) != expected):
                raise ValueError('continuation audited record identity differs: ' + field)
    elif args.continuation_events_sha256 is not None or args.continuation_result_sha256 is not None:
        raise ValueError('continuation digests require an audited continuation report')
    result = calculate(data['prefix_report'], data.get('continuation_report'), data.get('comparison'), data.get('receipt'))
    if {name: sha(path) for name, path in paths.items()} != before:
        raise ValueError('an input changed during summary')
    result['input_sha256'] = before
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
