"""Replay saved cue-factorial target logits and decisions without a model.

This checks fixed order, input identity, selected mask logits, scalar screens,
and the four-cell contrasts. Full vocabulary controls, dose vectors, landing,
actual execution and chronology require separate evidence.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct

SETS = {'prior': (('never', 'rarely'), ('always', 'often')),
        'new': (('not', 'hardly'), ('really', 'certainly'))}
FRAMES = ('main', 'subject_relative')
TRIPLES = ('triple-01', 'triple-02', 'triple-03')
SITES = ('cue2', 'object2', 'mask2', 'cue5')
MANIFEST_SHA = 'b91eba1e0a52f6a183a59b0bfbb365dd54d67341bd74a4fb56325e72a7f45d4f'
MANIFEST_PATH = 'experiments/bert_cue_factorial_cases_v1.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def margin(event, any_id, some_id, vocab):
    raw = base64.b64decode(event['mask_logits_f32le'], validate=True)
    if len(raw) % 4 or len(raw) < 4 * (1 + max(any_id, some_id)) or (vocab and len(raw) != 4 * vocab):
        raise ValueError('malformed saved mask logits')
    a, b = (struct.unpack_from('<f', raw, 4 * idx)[0] for idx in (any_id, some_id))
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('nonfinite target logit')
    return math.fsum((a, -b))


def replay(manifest, result, stream, synthetic=False):
    cases = manifest['cases']
    wanted = [(t, f, s) for t in TRIPLES for f in FRAMES for s in SETS]
    if (len(cases) != 12 or [(c['triple_id'], c['frame'], c['cue_set']) for c in cases] != wanted
            or any(c['split'] != 'evaluation' for c in cases) or len({c['id'] for c in cases}) != 12
            or result['status'] != 'complete' or len(result['cases']) != 12):
        raise ValueError('incomplete or mismatched case schedule')
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    vocab = None if synthetic else 30522
    if type(any_id) is not int or type(some_id) is not int or any_id == some_id or min(any_id, some_id) < 0:
        raise ValueError('invalid target IDs')
    count, rows = 0, []

    def take(kind, ident, **keys):
        nonlocal count
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024:
            raise ValueError('missing or oversized event')
        event = json.loads(line)
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
        minimum = radius['minimum_distance']
        if (type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum < 0 or
                radius['radius'] != .5 or saved['radius'] != .5 or saved['minimum_distance'] != minimum):
            raise ValueError('fixed dose or reported minimum differs')
        dose = minimum >= .5
        if radius['dose_eligible'] is not dose or saved['dose_eligible'] is not dose:
            raise ValueError('dose eligibility differs')
        summaries = {}
        if not dose:
            take('dose_ineligible', ident, radius=.5, minimum_distance=minimum)
            if saved['effects'] or saved['specificity'] or saved['specificity_passed'] is not None:
                raise ValueError('ineligible case reports interventions')
            cue_min = None
            limiting = []
        else:
            for site in SITES:
                for cue in cues:
                    e = take('self', ident, site=site, target=cue, input_ids=ids(cue))
                    if margin(e, any_id, some_id, vocab) != clean[cue]:
                        raise ValueError('self target margin differs')
                effects = {}
                for donor, target in transfers:
                    e = take('transfer', ident, site=site, donor=donor, target=target, input_ids=ids(target))
                    effects[donor, target] = math.fsum((margin(e, any_id, some_id, vocab), -clean[target]))
                if saved['effects'][site] != [{'donor': d, 'target': t, 'delta': effects[d, t]}
                                              for d, t in transfers]:
                    raise ValueError('saved effects differ')
                targets = {}
                for target in cues:
                    own, opposite, sign = (neg, pos, -1) if target in neg else (pos, neg, 1)
                    same = next(x for x in own if x != target)
                    cross = {x: sign * effects[x, target] for x in opposite}
                    minimum_cross, same_delta = min(cross.values()), effects[same, target]
                    value = math.fsum((minimum_cross, -abs(same_delta)))
                    targets[target] = {'cross_signed': cross, 'cross_min': minimum_cross,
                                       'same_donor': same, 'same_delta': same_delta,
                                       'same_abs': abs(same_delta), 'margin': value, 'passed': value > .05}
                detail = {'targets': targets, 'passed': all(x['passed'] for x in targets.values())}
                if saved['specificity'][site] != detail:
                    raise ValueError('specificity values differ')
                summaries[site] = detail
            cue_min = min(t['margin'] for t in summaries['cue2']['targets'].values())
            limiting = [c for c in cues if summaries['cue2']['targets'][c]['margin'] == cue_min]
            if saved['specificity_passed'] is not summaries['cue2']['passed']:
                raise ValueError('specificity decision differs')
        if saved['cue_min_margin'] != cue_min:
            raise ValueError('cue minimum differs')
        if ('limiting_targets' in saved or not synthetic) and saved.get('limiting_targets') != limiting:
            raise ValueError('limiting target identities differ')
        for cue in cues:
            if margin(take('repeat', ident, cue=cue, input_ids=ids(cue)), any_id, some_id, vocab) != clean[cue]:
                raise ValueError('repeat target margin differs')
        forwards = 72 if dose else 8
        if saved['forwards'] != forwards:
            raise ValueError('case forward count differs')
        rows.append({'id': ident, 'triple_id': case['triple_id'], 'frame': case['frame'], 'cue_set': group,
                     'baseline_eligible': baseline, 'dose_eligible': dose, 'cue_min_margin': cue_min,
                     'limiting_targets': limiting,
                     'specificity_passed': summaries['cue2']['passed'] if dose else None,
                     'clean_margins': clean, 'baseline_gaps': gaps, 'site_specificity': summaries,
                     'forwards': forwards})
    if stream.readline(1) or result['forwards'] != sum(r['forwards'] for r in rows):
        raise ValueError('extra event or forward count mismatch')

    lookup = {(r['triple_id'], r['frame'], r['cue_set']): r for r in rows}
    eligible = lambda row: row['baseline_eligible'] and row['dose_eligible']
    cells = {}
    for group in SETS:
        for frame in FRAMES:
            members = [lookup[t, frame, group] for t in TRIPLES if eligible(lookup[t, frame, group])]
            passing = sum(r['specificity_passed'] for r in members)
            cells[group + '/' + frame] = {'total': 3, 'eligible': len(members), 'passing': passing,
                'status': 'unevaluable' if len(members) < 2 else ('passed' if passing >= 2 else 'failed')}

    def describe(values):
        return {'status': 'evaluable' if len(values) >= 2 else 'unevaluable', 'eligible': len(values),
                'values': [{'triple_id': t, 'value': v} for t, v in values],
                'mean': math.fsum(v for _, v in values) / len(values) if len(values) >= 2 else None}

    cue, frame_contrasts = {}, {}
    for f in FRAMES:
        cue[f] = describe([(t, math.fsum((lookup[t, f, 'new']['cue_min_margin'],
                                         -lookup[t, f, 'prior']['cue_min_margin']))) for t in TRIPLES
                           if eligible(lookup[t, f, 'new']) and eligible(lookup[t, f, 'prior'])])
    for s in SETS:
        frame_contrasts[s] = describe([(t, math.fsum((lookup[t, 'subject_relative', s]['cue_min_margin'],
                                                       -lookup[t, 'main', s]['cue_min_margin']))) for t in TRIPLES
                                       if eligible(lookup[t, 'subject_relative', s]) and eligible(lookup[t, 'main', s])])
    interaction = describe([(t, math.fsum((lookup[t, 'subject_relative', 'new']['cue_min_margin'],
                                             -lookup[t, 'subject_relative', 'prior']['cue_min_margin'],
                                             -lookup[t, 'main', 'new']['cue_min_margin'],
                                             lookup[t, 'main', 'prior']['cue_min_margin']))) for t in TRIPLES
                            if all(eligible(lookup[t, f, s]) for f in FRAMES for s in SETS)])
    evaluation = {'cells': cells, 'cue_contrasts': cue, 'frame_contrasts': frame_contrasts,
                  'interaction': interaction}
    if result['evaluation'] != evaluation:
        raise ValueError('factorial decision or contrast differs')
    return {'status': 'passed', 'forwards': result['forwards'], 'events': count, 'cases': rows,
            'evaluation': evaluation, 'scope': 'saved selected logits, order, screens and contrasts; not full logits, dose, landing or execution'}


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
    manifest, result = json.loads(a.manifest.read_bytes()), json.loads(a.result.read_bytes())
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
