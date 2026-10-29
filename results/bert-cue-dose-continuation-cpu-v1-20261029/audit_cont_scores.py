"""Replay selected logits from the separate four-cell dose continuation.

No model is loaded.  This is a saved-score check, not full-witness, dose,
execution or cross-process-combination verification.
"""
import argparse
import base64
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DEPENDENCY = HERE / 'review_bert_cue_dose_saved_scores.py'
DEPENDENCY_SHA = '3817034b95f5cf30abc9e196c71998b8082ce1304d921a1192956c80e1837d25'
if hashlib.sha256(DEPENDENCY.read_bytes()).hexdigest() != DEPENDENCY_SHA:
    raise ValueError('prior scalar reader bytes differ')
spec = importlib.util.spec_from_file_location('fixed_dose_scalar', DEPENDENCY)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ORIGINAL_SOURCE = 'b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4'
ORIGINAL_HASHES = {'events': 'ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05',
                   'started': '19baa22fc6b6fe3da139bf22b125a4f3122a39e6e82dc9ec38604389127bd9e7',
                   'failure': '12b861407be2e4050b7f5e7b792847867f7b7c588cf75857783fc36d1bb8317a',
                   'receipt': 'f5499d680cbd2ea1d2ebdfdb7268ce30598ad8aa7d6af9ad8ef0ff806ed54f34'}
CASE_IDS = ('fresh-02-prior', 'fresh-02-new', 'fresh-03-prior', 'fresh-03-new')


def selected_bits(event, any_id, some_id, vocab):
    raw = base64.b64decode(event['mask_logits_f32le'], validate=True)
    if (len(raw) % 4 or len(raw) < 4 * (1 + max(any_id, some_id))
            or (vocab and len(raw) != 4 * vocab)):
        raise ValueError('malformed selected logits')
    return raw[4 * any_id:4 * (any_id + 1)] + raw[4 * some_id:4 * (some_id + 1)]


def replay(manifest, result, stream, synthetic=False):
    cases = manifest['cases'][4:]
    if (len(manifest['cases']) != 8 or tuple(c['id'] for c in cases) != CASE_IDS
            or any(c['split'] != 'evaluation' or c['frame'] != 'main' for c in cases)
            or result.get('status') != 'complete' or len(result.get('cases', [])) != 4):
        raise ValueError('fixed continuation case schedule differs')
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    if type(any_id) is not int or type(some_id) is not int or any_id == some_id or min(any_id, some_id) < 0:
        raise ValueError('invalid selected-token IDs')
    vocab = None if synthetic else 30522
    events = 0
    summaries = []

    def take(kind, case_id, **identity):
        nonlocal events
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024 or not line.endswith(b'\n'):
            raise ValueError('missing, oversized or unterminated continuation event')
        event = base.strict_json(line)
        if (event.get('kind') != kind or event.get('case_id') != case_id
                or any(event.get(k) != v for k, v in identity.items())):
            raise ValueError('event order, case or input differs')
        events += 1
        return event

    for case, saved in zip(cases, result['cases']):
        ident, group = case['id'], case['cue_set']
        neg, pos = base.SETS[group]
        cues = neg + pos
        transfers = [(donor, target) for target in cues for donor in cues if donor != target]
        if any(saved.get(k) != case[k] for k in ('id', 'split', 'triple_id', 'frame', 'cue_set')):
            raise ValueError('result case identity differs')
        ids = lambda cue: case['rows'][cue]['input_ids']
        clean, clean_pairs = {}, {}
        for cue in cues:
            event = take('clean', ident, cue=cue, input_ids=ids(cue))
            clean[cue] = base.margin(event, any_id, some_id, vocab)
            clean_pairs[cue] = selected_bits(event, any_id, some_id, vocab)
        gaps = [{'negative': n, 'affirmative': p, 'gap': math.fsum((clean[n], -clean[p]))}
                for n in neg for p in pos]
        baseline = all(row['gap'] > .5 for row in gaps)
        event = take('baseline', ident)
        if (event['eligible'] is not baseline or event['gaps'] != gaps or
                saved['baseline_eligible'] is not baseline or saved['baseline_gaps'] != gaps or
                saved['clean_margins'] != clean):
            raise ValueError('baseline or clean margins differ')
        event = take('radius', ident)
        distances = event['distances']
        keys = [(a, b, site) for a, b in itertools.combinations(cues, 2) for site in base.SITES]
        if (len(distances) != 24 or [(x['left'], x['right'], x['site']) for x in distances] != keys
                or any(type(x['distance']) not in (int, float) or not math.isfinite(x['distance'])
                       or x['distance'] < 0 for x in distances) or saved['distances'] != distances):
            raise ValueError('recorded distances differ')
        minimum = min(x['distance'] for x in distances)
        if (event['minimum_distance'] != minimum or saved['minimum_distance'] != minimum
                or event['binding_edges'] != [x for x in distances if x['distance'] == minimum]
                or type(event['doses']) is not dict or set(event['doses']) != set(base.DOSES)
                or type(saved['doses']) is not dict or set(saved['doses']) != set(base.DOSES)):
            raise ValueError('minimum or fixed dose labels differ')
        eligible = {label: minimum >= amount for label, amount in base.DOSES.items()}
        for label, amount in base.DOSES.items():
            if (type(event['doses'][label]['eligible']) is not bool
                    or event['doses'][label] != {'radius': amount, 'eligible': eligible[label]}
                    or saved['doses'][label]['radius'] != amount
                    or saved['doses'][label]['eligible'] is not eligible[label]):
                raise ValueError('recorded dose eligibility differs')
            if not eligible[label]:
                take('dose_ineligible', ident, dose=label, radius=amount, minimum_distance=minimum)
        effects = {label: {} for label in base.DOSES if eligible[label]}
        if eligible['low']:
            for site in base.SITES:
                for cue in cues:
                    event = take('self', ident, site=site, target=cue, input_ids=ids(cue))
                    if selected_bits(event, any_id, some_id, vocab) != clean_pairs[cue]:
                        raise ValueError('selected self logits differ')
                for label, amount in base.DOSES.items():
                    if not eligible[label]:
                        continue
                    effects[label][site] = {}
                    for donor, target in transfers:
                        event = take('transfer', ident, site=site, dose=label, donor=donor, target=target,
                                     input_ids=ids(target), requested_radius=amount)
                        value = base.margin(event, any_id, some_id, vocab)
                        if site == 'cue5' and selected_bits(event, any_id, some_id, vocab) != clean_pairs[target]:
                            raise ValueError('selected final-cue logits differ')
                        effects[label][site][donor, target] = math.fsum((value, -clean[target]))
        for cue in cues:
            event = take('repeat', ident, cue=cue, input_ids=ids(cue))
            if selected_bits(event, any_id, some_id, vocab) != clean_pairs[cue]:
                raise ValueError('selected repeat logits differ')
        doses = {}
        for label, amount in base.DOSES.items():
            item = saved['doses'][label]
            sites = {}
            if eligible[label]:
                if set(item['effects']) != set(base.SITES) or set(item['specificity']) != set(base.SITES):
                    raise ValueError('missing comparator site')
                for site in base.SITES:
                    values = effects[label][site]
                    if item['effects'][site] != [{'donor': d, 'target': t, 'delta': values[d, t]}
                                                  for d, t in transfers]:
                        raise ValueError('saved donor effects differ')
                    targets = {}
                    for target in cues:
                        own, opposite, sign = (neg, pos, -1) if target in neg else (pos, neg, 1)
                        same = next(x for x in own if x != target)
                        cross = {x: sign * values[x, target] for x in opposite}
                        cross_min, same_delta = min(cross.values()), values[same, target]
                        value = math.fsum((cross_min, -abs(same_delta)))
                        targets[target] = {'cross_signed': cross, 'cross_min': cross_min, 'same_donor': same,
                                           'same_delta': same_delta, 'same_abs': abs(same_delta),
                                           'margin': value, 'passed': value > .05}
                    sites[site] = {'targets': targets, 'passed': all(x['passed'] for x in targets.values())}
                    if item['specificity'][site] != sites[site]:
                        raise ValueError('saved target specificity differs')
                minimum_margin = min(x['margin'] for x in sites['cue2']['targets'].values())
                limiting = [cue for cue in cues if sites['cue2']['targets'][cue]['margin'] == minimum_margin]
                passed = sites['cue2']['passed']
            else:
                if item['effects'] or item['specificity']:
                    raise ValueError('ineligible dose reports interventions')
                minimum_margin, limiting, passed = None, [], None
            if (item['cue_min_margin'] != minimum_margin or item['limiting_targets'] != limiting
                    or item['specificity_passed'] is not passed):
                raise ValueError('minimum, limiting target or outcome differs')
            doses[label] = {'eligible': eligible[label], 'cue_min_margin': minimum_margin,
                            'limiting_targets': limiting, 'specificity_passed': passed, 'site_specificity': sites}
        forwards = 8 if not eligible['low'] else 24 + 48 * sum(eligible.values())
        if saved['forwards'] != forwards:
            raise ValueError('case forward count differs')
        paired = None
        if baseline and eligible['low'] and eligible['high']:
            low, high = doses['low'], doses['high']
            paired = {'difference': math.fsum((high['cue_min_margin'], -low['cue_min_margin'])),
                      'transition': ('pass' if low['specificity_passed'] else 'fail') + '_to_' +
                                    ('pass' if high['specificity_passed'] else 'fail')}
        summaries.append({'case_id': ident, 'cue_set': group, 'triple_id': case['triple_id'],
                          'baseline_eligible': baseline, 'baseline_gaps': gaps, 'clean_margins': clean,
                          'minimum_recorded_distance': minimum, 'doses': doses, 'paired': paired,
                          'forwards': forwards})
    if stream.readline(1) or result['forwards'] != sum(c['forwards'] for c in summaries):
        raise ValueError('extra events or total forwards differ')
    return {'status': 'passed', 'events': events, 'forwards': result['forwards'], 'cases': summaries,
            'scope': 'selected logits and recorded distance labels in the four continued cases; '
                     'not full logits, dose, landing, execution, farmer comparison or pooled decisions'}


def main():
    import gzip
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('events', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('--started', type=Path)
    p.add_argument('--expected-source')
    p.add_argument('--expected-result-sha256')
    p.add_argument('--expected-started-sha256')
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    paths = (a.manifest, a.events, a.result)
    before = tuple(base.sha(x) for x in paths)
    manifest, result = base.strict_json(a.manifest.read_bytes()), base.strict_json(a.result.read_bytes())
    if not a.synthetic:
        if (not re.fullmatch('[0-9a-f]{40}', a.expected_source or '')
                or not re.fullmatch('[0-9a-f]{64}', a.expected_result_sha256 or '')
                or not re.fullmatch('[0-9a-f]{64}', a.expected_started_sha256 or '') or not a.started
                or before[0] != base.MANIFEST_SHA or before[2] != a.expected_result_sha256
                or result.get('source_commit') != a.expected_source
                or result.get('source_sha256', {}).get(base.MANIFEST_PATH) != before[0]
                or result.get('events_sha256') != before[1] or a.events.suffix != '.gz'
                or result.get('original_source_commit') != ORIGINAL_SOURCE
                or result.get('original_sha256') != ORIGINAL_HASHES
                or base.sha(a.started) != a.expected_started_sha256):
            raise ValueError('continuation source or saved-byte binding differs')
        started = base.strict_json(a.started.read_bytes())
        if (started.get('case_ids') != list(CASE_IDS) or started.get('maximum_forwards') != 480
                or any(started.get(k) != result.get(k) for k in
                       ('source_commit', 'source_sha256', 'original_source_commit', 'original_sha256',
                        'assets', 'versions', 'seed', 'cpu_environment'))):
            raise ValueError('started/result binding differs')
    opener = gzip.open if a.events.suffix == '.gz' else open
    with opener(a.events, 'rb') as stream:
        report = replay(manifest, result, stream, synthetic=a.synthetic)
    if (tuple(base.sha(x) for x in paths) != before or base.sha(DEPENDENCY) != DEPENDENCY_SHA
            or (not a.synthetic and base.sha(a.started) != a.expected_started_sha256)):
        raise ValueError('an input changed during replay')
    report.update({'synthetic': a.synthetic, 'manifest_sha256': before[0], 'events_sha256': before[1],
                   'result_sha256': before[2], 'source_commit': result.get('source_commit'),
                   'dependency_sha256': DEPENDENCY_SHA,
                   'identity_scope': 'operator-supplied expected source and started/result consistency; '
                                     'not independent source/asset pin or execution attestation'})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
