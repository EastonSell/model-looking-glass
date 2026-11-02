"""Replay a post-outcome target-score projection of four continued cells.

This stdlib-only reader verifies the projected schedule and selected-score
arithmetic. The subset cannot authenticate its derivation, full logits, dose,
landing witnesses, execution, or the earlier interrupted attempt.
"""
import argparse
import base64
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct

SOURCE = 'cdc9630e24542fed84b0c9556bb7f37c60d722d1'
ORIGINAL_SOURCE = 'b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4'
MANIFEST_SHA = '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a'
ORIGINAL_JOURNAL_SHA = 'ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05'
JOURNAL_SHA = '67c022d36d7427840334390602706e286aab60168c50c42dad72b7e711b2c87f'
START_SHA = 'c7d35e9c1113eef722a7a6654965cfcda0fcde801314513b22270a5011e22e2c'
RESULT_SHA = 'cce115669ca1a957428d000950aa36e04c8087a0a96a87dc78492cdff59662d5'
EXIT_SHA = 'a932efaf2be7b8a8cf72456c73e9bbcb8dd4623e1fa370ef6fe56384dd38f042'
CASE_IDS = ['fresh-02-prior', 'fresh-02-new', 'fresh-03-prior', 'fresh-03-new']
SETS = {'prior': (('never', 'rarely'), ('always', 'often')),
        'new': (('not', 'hardly'), ('really', 'certainly'))}
TRIPLES = ('exposed-calibration', 'fresh-01', 'fresh-02', 'fresh-03')
SITES = ('cue2', 'object2', 'mask2', 'cue5')
DOSES = {'low': .5, 'high': .7}
HEADER = {'kind': 'continuation_header', 'schema': 'bert-cue-dose-continuation-score-subset-v1',
          'source_commit': SOURCE, 'original_source_commit': ORIGINAL_SOURCE,
          'manifest_sha256': MANIFEST_SHA, 'original_journal_sha256': ORIGINAL_JOURNAL_SHA,
          'continuation_journal_sha256': JOURNAL_SHA, 'started_sha256': START_SHA,
          'result_sha256': RESULT_SHA, 'exit_receipt_sha256': EXIT_SHA,
          'case_ids': CASE_IDS, 'farmer_prior_partially_exposed': True}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def selected_pair(event):
    pair = base64.b64decode(event['mask_logits_pair_f32le'], validate=True)
    if len(pair) != 8:
        raise ValueError('invalid selected-logit pair')
    a, b = struct.unpack('<ff', pair)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('nonfinite selected logit')
    return pair


def margin(event):
    a, b = struct.unpack('<ff', selected_pair(event))
    return math.fsum((a, -b))


class PrefixEnd(Exception):
    pass


def replay(manifest, stream):
    cases = manifest['cases']
    expected = [(t, g, 'calibration' if t == TRIPLES[0] else 'evaluation', 'main') for t in TRIPLES for g in SETS]
    if (len(cases) != 8 or [(c['triple_id'], c['cue_set'], c['split'], c['frame']) for c in cases] != expected
            or len({c['id'] for c in cases}) != 8 or manifest['scored_tokens'] != {'any': 2151, 'some': 2070}):
        raise ValueError('fixed case identities differ')
    cases = cases[4:]
    if [c['id'] for c in cases] != CASE_IDS:
        raise ValueError('fixed continuation order differs')
    line = stream.readline(2 * 1024 * 1024 + 1)
    if not line or len(line) > 2 * 1024 * 1024 or not line.endswith(b'\n') or strict_json(line) != HEADER:
        raise ValueError('subset header differs')
    count = forwards = 0
    last_expected = None
    completed = []
    partial = None

    def take(kind, case_id, **identity):
        nonlocal count, forwards, last_expected
        last_expected = {'kind': kind, 'case_id': case_id, **identity}
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line:
            raise PrefixEnd()
        if len(line) > 2 * 1024 * 1024 or not line.endswith(b'\n'):
            raise ValueError('oversized or unterminated event')
        event = strict_json(line)
        if (event.get('kind') != kind or event.get('case_id') != case_id
                or any(event.get(key) != value for key, value in identity.items())):
            raise ValueError('event order or input differs')
        count += 1
        forwards += kind in ('clean', 'self', 'transfer', 'repeat')
        return event

    for case in cases:
        ident, group = case['id'], case['cue_set']
        neg, pos = SETS[group]
        cues = neg + pos
        transfers = [(donor, target) for target in cues for donor in cues if donor != target]
        progress = {'case_id': ident, 'triple_id': case['triple_id'], 'split': case['split'], 'cue_set': group,
                    'exposure': 'partially-observed-in-original-prefix' if ident == CASE_IDS[0]
                                else 'not-observed-in-original-saved-prefix',
                    'clean_margins': {}, 'baseline_eligible': None, 'baseline_gaps': None,
                    'minimum_reported_distance': None, 'dose_eligibility': None, 'recorded_effects': {}}
        try:
            clean = progress['clean_margins']
            clean_pairs = {}
            for cue in cues:
                event = take('clean', ident, cue=cue, input_ids=case['rows'][cue]['input_ids'])
                clean_pairs[cue] = selected_pair(event)
                clean[cue] = margin(event)
            gaps = [{'negative': n, 'affirmative': p, 'gap': math.fsum((clean[n], -clean[p]))}
                    for n in neg for p in pos]
            baseline = all(row['gap'] > .5 for row in gaps)
            event = take('baseline', ident)
            if type(event['eligible']) is not bool or event['eligible'] is not baseline or event['gaps'] != gaps:
                raise ValueError('baseline screen differs')
            progress['baseline_eligible'], progress['baseline_gaps'] = baseline, gaps
            event = take('radius', ident)
            distances = event['distances']
            keys = [(a, b, site) for a, b in itertools.combinations(cues, 2) for site in SITES]
            if (len(distances) != 24 or [(x['left'], x['right'], x['site']) for x in distances] != keys
                    or any(type(x['distance']) not in (int, float) or not math.isfinite(x['distance'])
                           or x['distance'] < 0 for x in distances)):
                raise ValueError('reported distance order or values differ')
            minimum = min(row['distance'] for row in distances)
            eligible = {label: minimum >= amount for label, amount in DOSES.items()}
            if (event['minimum_distance'] != minimum
                    or event['binding_edges'] != [row for row in distances if row['distance'] == minimum]
                    or type(event['doses']) is not dict or set(event['doses']) != set(DOSES)
                    or any(type(event['doses'][label]['eligible']) is not bool or
                           event['doses'][label] != {'radius': amount, 'eligible': eligible[label]}
                           for label, amount in DOSES.items())):
                raise ValueError('reported minimum or eligibility differs')
            progress['minimum_reported_distance'], progress['dose_eligibility'] = minimum, eligible
            for label, amount in DOSES.items():
                if not eligible[label]:
                    take('dose_ineligible', ident, dose=label, radius=amount, minimum_distance=minimum)
            effects = {label: {} for label in DOSES if eligible[label]}
            progress['recorded_effects'] = effects
            if eligible['low']:
                for site in SITES:
                    for cue in cues:
                        event = take('self', ident, site=site, target=cue, input_ids=case['rows'][cue]['input_ids'])
                        if selected_pair(event) != clean_pairs[cue]:
                            raise ValueError('selected self logits differ')
                    for label, amount in DOSES.items():
                        if not eligible[label]:
                            continue
                        rows = effects[label][site] = []
                        for donor, target in transfers:
                            event = take('transfer', ident, site=site, dose=label, donor=donor, target=target,
                                         input_ids=case['rows'][target]['input_ids'], requested_radius=amount)
                            value = margin(event)
                            if site == 'cue5' and selected_pair(event) != clean_pairs[target]:
                                raise ValueError('selected final-cue logits differ')
                            rows.append({'donor': donor, 'target': target,
                                         'delta': math.fsum((value, -clean[target]))})
            for cue in cues:
                event = take('repeat', ident, cue=cue, input_ids=case['rows'][cue]['input_ids'])
                if selected_pair(event) != clean_pairs[cue]:
                    raise ValueError('selected repeat logits differ')
            results = {}
            for label, amount in DOSES.items():
                if not eligible[label]:
                    results[label] = {'eligible': False, 'cue_min_margin': None, 'targets': {},
                                      'limiting_targets': [], 'specificity_passed': None, 'case_pass': None}
                    continue
                values = {(x['donor'], x['target']): x['delta'] for x in effects[label]['cue2']}
                targets = {}
                for target in cues:
                    own, opposite, sign = (neg, pos, -1) if target in neg else (pos, neg, 1)
                    same = next(c for c in own if c != target)
                    cross = {donor: sign * values[donor, target] for donor in opposite}
                    same_delta = values[same, target]
                    value = math.fsum((min(cross.values()), -abs(same_delta)))
                    targets[target] = {'cross_signed': cross, 'same_donor': same, 'same_delta': same_delta,
                                       'same_abs': abs(same_delta), 'margin': value, 'passed': value > .05}
                minimum_margin = min(targets[c]['margin'] for c in cues)
                results[label] = {'eligible': True, 'cue_min_margin': minimum_margin, 'targets': targets,
                                  'limiting_targets': [c for c in cues if targets[c]['margin'] == minimum_margin],
                                  'specificity_passed': minimum_margin > .05,
                                  'case_pass': minimum_margin > .05 if baseline else None}
            progress['doses'] = results
            completed.append(progress)
        except PrefixEnd:
            partial = {'case': progress, 'next_expected_event': last_expected, 'case_outcome': None}
            break
    else:
        if stream.readline(1):
            raise ValueError('extra event after the fixed four cases')
    if (count, forwards) != (488, 480) or len(completed) != 4 or partial is not None:
        raise ValueError('closed continuation boundary differs')
    return {'status': 'continuation_scores_replayed', 'events': count, 'saved_forwards': forwards,
            'completed_cases': completed, 'original_attempt': 'incomplete', 'overall_decision': None,
            'scope': 'post-outcome operator-derived selected-logit projection and recorded-distance labels; '
                     'not proof of derivation, full logits, doses, landing, execution or other exposure'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('subset', type=Path)
    parser.add_argument('--expected-subset-sha256', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.expected_subset_sha256):
        raise ValueError('invalid subset identity')
    before = (sha(args.manifest), sha(args.subset))
    if before != (MANIFEST_SHA, args.expected_subset_sha256):
        raise ValueError('manifest or subset hash differs')
    manifest = strict_json(args.manifest.read_bytes())
    with gzip.open(args.subset, 'rb') as stream:
        report = replay(manifest, stream)
    if (sha(args.manifest), sha(args.subset)) != before:
        raise ValueError('manifest or subset changed during replay')
    report.update({'source_commit': SOURCE, 'original_source_commit': ORIGINAL_SOURCE,
                   'manifest_sha256': MANIFEST_SHA, 'subset_sha256': before[1],
                   'original_journal_sha256_declared': ORIGINAL_JOURNAL_SHA,
                   'continuation_journal_sha256_declared': JOURNAL_SHA,
                   'started_sha256_declared': START_SHA, 'result_sha256_declared': RESULT_SHA,
                   'exit_receipt_sha256_declared': EXIT_SHA})
    data = (json.dumps(report, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    if args.output:
        with args.output.open('xb') as stream:
            stream.write(data)
    else:
        print(data.decode('utf-8'), end='')


if __name__ == '__main__':
    main()
