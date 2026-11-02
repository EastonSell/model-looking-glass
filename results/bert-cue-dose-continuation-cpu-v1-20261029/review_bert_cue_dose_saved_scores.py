"""Replay the fixed cue-dose study's saved target logits and decisions.

Full logits, independently reconstructed distances/doses, landing witnesses,
execution, and chronology need separate checks; no model is loaded here.
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

SETS = {'prior': (('never', 'rarely'), ('always', 'often')),
        'new': (('not', 'hardly'), ('really', 'certainly'))}
TRIPLES = ('exposed-calibration', 'fresh-01', 'fresh-02', 'fresh-03')
SITES = ('cue2', 'object2', 'mask2', 'cue5')
DOSES = {'low': .5, 'high': .7}
MANIFEST_SHA = '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a'
MANIFEST_PATH = 'experiments/bert_cue_dose_response_cases_v1.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(data):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError('duplicate JSON key')
            output[key] = value
        return output
    def bad_constant(_):
        raise ValueError('nonfinite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def margin(event, any_id, some_id, vocab):
    raw = base64.b64decode(event['mask_logits_f32le'], validate=True)
    if len(raw) % 4 or len(raw) < 4 * (1 + max(any_id, some_id)) or (vocab and len(raw) != 4 * vocab):
        raise ValueError('malformed saved mask logits')
    a, b = (struct.unpack_from('<f', raw, 4 * index)[0] for index in (any_id, some_id))
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('nonfinite target logit')
    return math.fsum((a, -b))


def replay(manifest, result, stream, synthetic=False):
    cases = manifest['cases']
    wanted = [(t, s) for t in TRIPLES for s in SETS]
    if (len(cases) != 8 or [(c['triple_id'], c['cue_set']) for c in cases] != wanted
            or any(c['frame'] != 'main' or c['split'] != ('calibration' if c['triple_id'] == TRIPLES[0]
                                                           else 'evaluation') for c in cases)
            or len({c['id'] for c in cases}) != 8 or result['status'] != 'complete'
            or len(result['cases']) != 8):
        raise ValueError('incomplete or mismatched case schedule')
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    if type(any_id) is not int or type(some_id) is not int or any_id == some_id or min(any_id, some_id) < 0:
        raise ValueError('invalid target IDs')
    vocab = None if synthetic else 30522
    count, rows = 0, []

    def take(kind, ident, **keys):
        nonlocal count
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024:
            raise ValueError('missing or oversized event')
        event = strict_json(line)
        count += 1
        if (event.get('kind') != kind or event.get('case_id') != ident
                or any(event.get(k) != v for k, v in keys.items())):
            raise ValueError('event order, identity or input differs')
        return event

    for case, saved in zip(cases, result['cases']):
        ident, group = case['id'], case['cue_set']
        neg, pos = SETS[group]
        cues = neg + pos
        transfers = [(donor, target) for target in cues for donor in cues if donor != target]
        if any(saved.get(k) != case[k] for k in ('id', 'split', 'triple_id', 'frame', 'cue_set')):
            raise ValueError('case identity differs')
        ids = lambda cue: case['rows'][cue]['input_ids']
        clean = {cue: margin(take('clean', ident, cue=cue, input_ids=ids(cue)), any_id, some_id, vocab)
                 for cue in cues}
        gaps = [{'negative': n, 'affirmative': p, 'gap': math.fsum((clean[n], -clean[p]))}
                for n in neg for p in pos]
        baseline = all(g['gap'] > .5 for g in gaps)
        be = take('baseline', ident)
        if (be['eligible'] is not baseline or be['gaps'] != gaps or
                saved['baseline_eligible'] is not baseline or saved['baseline_gaps'] != gaps or
                saved['clean_margins'] != clean):
            raise ValueError('baseline values or eligibility differ')
        radius = take('radius', ident)
        distances = radius['distances']
        keys = [(a, b, site) for a, b in itertools.combinations(cues, 2) for site in SITES]
        if (len(distances) != 24 or [(x['left'], x['right'], x['site']) for x in distances] != keys
                or any(type(x['distance']) not in (int, float) or not math.isfinite(x['distance'])
                       or x['distance'] < 0 for x in distances) or saved['distances'] != distances):
            raise ValueError('reported distance list or order differs')
        minimum = min(x['distance'] for x in distances)
        if (radius['minimum_distance'] != minimum or saved['minimum_distance'] != minimum
                or radius['binding_edges'] != [x for x in distances if x['distance'] == minimum]):
            raise ValueError('reported minimum or binding distances differ')
        eligible = {label: minimum >= amount for label, amount in DOSES.items()}
        if (type(radius['doses']) is not dict or set(radius['doses']) != set(DOSES)
                or type(saved['doses']) is not dict or set(saved['doses']) != set(DOSES)):
            raise ValueError('dose keys differ from the fixed design')
        for label, amount in DOSES.items():
            if (type(radius['doses'][label]) is not dict
                    or type(radius['doses'][label].get('eligible')) is not bool
                    or radius['doses'][label] != {'radius': amount, 'eligible': eligible[label]}
                    or saved['doses'][label]['radius'] != amount
                    or saved['doses'][label]['eligible'] is not eligible[label]):
                raise ValueError('fixed dose or reported eligibility differs')
            if not eligible[label]:
                take('dose_ineligible', ident, dose=label, radius=amount, minimum_distance=minimum)
        effects = {label: {} for label in DOSES if eligible[label]}
        if eligible['low']:
            for site in SITES:
                for cue in cues:
                    event = take('self', ident, site=site, target=cue, input_ids=ids(cue))
                    if margin(event, any_id, some_id, vocab) != clean[cue]:
                        raise ValueError('self target margin differs')
                for label, amount in DOSES.items():
                    if not eligible[label]:
                        continue
                    effects[label][site] = {}
                    for donor, target in transfers:
                        event = take('transfer', ident, site=site, dose=label, donor=donor, target=target,
                                     input_ids=ids(target), requested_radius=amount)
                        value = margin(event, any_id, some_id, vocab)
                        if site == 'cue5' and value != clean[target]:
                            raise ValueError('final-cue target margin differs')
                        effects[label][site][donor, target] = math.fsum((value, -clean[target]))
        for cue in cues:
            event = take('repeat', ident, cue=cue, input_ids=ids(cue))
            if margin(event, any_id, some_id, vocab) != clean[cue]:
                raise ValueError('repeat target margin differs')

        results = {}
        for label, amount in DOSES.items():
            item = saved['doses'][label]
            summaries = {}
            if eligible[label]:
                if set(item['effects']) != set(SITES) or set(item['specificity']) != set(SITES):
                    raise ValueError('missing active comparator site')
                for site in SITES:
                    site_effects = effects[label][site]
                    if item['effects'][site] != [{'donor': d, 'target': t, 'delta': site_effects[d, t]}
                                                  for d, t in transfers]:
                        raise ValueError('saved donor effects differ')
                    targets = {}
                    for target in cues:
                        own, opposite, sign = (neg, pos, -1) if target in neg else (pos, neg, 1)
                        same = next(x for x in own if x != target)
                        cross = {x: sign * site_effects[x, target] for x in opposite}
                        cross_min, same_delta = min(cross.values()), site_effects[same, target]
                        value = math.fsum((cross_min, -abs(same_delta)))
                        targets[target] = {'cross_signed': cross, 'cross_min': cross_min,
                                           'same_donor': same, 'same_delta': same_delta,
                                           'same_abs': abs(same_delta), 'margin': value, 'passed': value > .05}
                    detail = {'targets': targets, 'passed': all(x['passed'] for x in targets.values())}
                    if item['specificity'][site] != detail:
                        raise ValueError('specificity or control differs')
                    summaries[site] = detail
                cue_min = min(t['margin'] for t in summaries['cue2']['targets'].values())
                limiting = [cue for cue in cues if summaries['cue2']['targets'][cue]['margin'] == cue_min]
                passed = summaries['cue2']['passed']
            else:
                if item['effects'] or item['specificity']:
                    raise ValueError('ineligible dose reports interventions')
                cue_min, limiting, passed = None, [], None
            if (item['cue_min_margin'] != cue_min or item['limiting_targets'] != limiting
                    or item['specificity_passed'] is not passed):
                raise ValueError('cue minimum, limiting target or decision differs')
            results[label] = {'radius': amount, 'eligible': eligible[label], 'cue_min_margin': cue_min,
                              'limiting_targets': limiting, 'specificity_passed': passed,
                              'site_specificity': summaries}
        forwards = 8 if not eligible['low'] else 24 + 48 * sum(eligible.values())
        if saved['forwards'] != forwards:
            raise ValueError('case forward count differs')
        rows.append({'id': ident, 'triple_id': case['triple_id'], 'frame': case['frame'], 'split': case['split'],
                     'cue_set': group, 'baseline_eligible': baseline, 'minimum_distance': minimum,
                     'clean_margins': clean, 'baseline_gaps': gaps, 'doses': results, 'forwards': forwards})
    if stream.readline(1) or result['forwards'] != sum(r['forwards'] for r in rows):
        raise ValueError('extra event or forward count mismatch')

    lookup = {(r['triple_id'], r['cue_set']): r for r in rows}
    cells, paired = {}, {}
    for group in SETS:
        cells[group] = {}
        for label in DOSES:
            candidates = [lookup[t, group] for t in TRIPLES[1:]]
            eligible = [r for r in candidates if r['baseline_eligible'] and r['doses'][label]['eligible']]
            passing = sum(r['doses'][label]['specificity_passed'] for r in eligible)
            status = 'unevaluable' if len(eligible) < 2 else ('passed' if passing >= 2 else 'failed')
            cells[group][label] = {'total': 3, 'eligible': len(eligible), 'passing': passing, 'status': status}
        counts = {'fail_to_pass': 0, 'pass_to_fail': 0, 'pass_to_pass': 0, 'fail_to_fail': 0}
        values = []
        for triple in TRIPLES[1:]:
            row = lookup[triple, group]
            low, high = row['doses']['low'], row['doses']['high']
            if row['baseline_eligible'] and low['eligible'] and high['eligible']:
                transition = ('pass' if low['specificity_passed'] else 'fail') + '_to_' + ('pass' if high['specificity_passed'] else 'fail')
                counts[transition] += 1
                values.append({'triple_id': triple, 'low_margin': low['cue_min_margin'],
                               'high_margin': high['cue_min_margin'],
                               'difference': math.fsum((high['cue_min_margin'], -low['cue_min_margin'])),
                               'transition': transition})
        n = len(values)
        status = 'unevaluable' if n < 2 else ('changed' if counts['fail_to_pass'] or counts['pass_to_fail'] else 'unchanged')
        paired[group] = {'eligible': n, 'status': status, 'transitions': counts, 'values': values,
                         'mean': math.fsum(x['difference'] for x in values) / n if n >= 2 else None}
    evaluation = {'cells': cells, 'paired': paired}
    if result['evaluation'] != evaluation:
        raise ValueError('dose decision or paired contrast differs')
    return {'status': 'passed', 'forwards': result['forwards'], 'events': count, 'cases': rows,
            'evaluation': evaluation, 'scope': 'saved selected logits, event order, recorded minimum, screens and '
            'paired contrasts; not independent full logits, dose vectors, landing, execution or chronology'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('events', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('--expected-source')
    p.add_argument('--expected-result-sha256')
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    manifest_hash, events_hash, result_hash = sha(a.manifest), sha(a.events), sha(a.result)
    manifest, result = strict_json(a.manifest.read_bytes()), strict_json(a.result.read_bytes())
    if not a.synthetic and (not re.fullmatch('[0-9a-f]{40}', a.expected_source or '')
            or not re.fullmatch('[0-9a-f]{64}', a.expected_result_sha256 or '')
            or manifest_hash != MANIFEST_SHA or result_hash != a.expected_result_sha256
            or result.get('source_commit') != a.expected_source
            or result.get('source_sha256', {}).get(MANIFEST_PATH) != manifest_hash
            or result.get('events_sha256') != events_hash or a.events.suffix != '.gz'):
        raise ValueError('source, manifest, result or journal binding differs')
    opener = gzip.open if a.events.suffix == '.gz' else open
    with opener(a.events, 'rb') as stream:
        report = replay(manifest, result, stream, synthetic=a.synthetic)
    if (sha(a.manifest), sha(a.events), sha(a.result)) != (manifest_hash, events_hash, result_hash):
        raise ValueError('manifest, result or journal changed during replay')
    report.update({'synthetic': a.synthetic, 'manifest_sha256': manifest_hash, 'events_sha256': events_hash,
                   'result_sha256': result_hash, 'source_commit': result.get('source_commit')})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
