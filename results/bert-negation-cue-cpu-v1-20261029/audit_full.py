"""Replay saved DistilBERT cue-transfer vectors and controls; never load a model.

The real entry point binds the public case manifest, saved journal and specified
source revision.  --synthetic is a deliberately relaxed fabricated-data mode.
This checks consistency of retained bytes; it does not attest execution.
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

if not __debug__:
    raise RuntimeError('optimized mode is not supported')

CUES = ('not', 'never', 'just', 'also')
SITES = (('cue2', 4), ('object2', 9), ('mask2', 6), ('cue5', 4))
PAIRS = (('not', 'just'), ('never', 'just'), ('also', 'just'),
         ('also', 'not'), ('also', 'never'))
TRANSFERS = (('not', 'just'), ('just', 'not'), ('never', 'just'), ('just', 'never'),
             ('also', 'just'), ('just', 'also'), ('also', 'not'), ('also', 'never'))
CASE_HASH = '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def raw(encoded, count):
    data = base64.b64decode(encoded, validate=True)
    if len(data) != count * 4:
        raise ValueError('wrong float32 vector length')
    if any(not math.isfinite(item[0]) for item in struct.iter_unpack('<f', data)):
        raise ValueError('nonfinite saved float32 value')
    return data


def values(data):
    return tuple(x[0] for x in struct.iter_unpack('<f', data))


def norm(left, right):
    a, b = values(left), values(right)
    if len(a) != len(b):
        raise ValueError('incompatible vector lengths')
    answer = math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    if not math.isfinite(answer):
        raise ValueError('nonfinite norm')
    return answer


def patch(target, donor, fraction):
    return b''.join(struct.pack('<f', x + fraction * (y - x))
                    for x, y in zip(values(target), values(donor)))


def margin(vector, any_id, some_id):
    return math.fsum((struct.unpack_from('<f', vector, 4 * any_id)[0],
                      -struct.unpack_from('<f', vector, 4 * some_id)[0]))


def baseline(vector, any_id, some_id):
    v = values(vector)
    peak = max(v)
    denom = math.fsum(math.exp(x - peak) for x in v)
    top = sorted(range(len(v)), key=lambda i: (-v[i], i))[:5]
    return {'any_minus_some': margin(vector, any_id, some_id),
            'any_probability': math.exp(v[any_id] - peak) / denom,
            'some_probability': math.exp(v[some_id] - peak) / denom,
            'top_five': [{'token_id': i, 'logit': v[i]} for i in top]}


def run(manifest, result, stream, width, vocab):
    cases = manifest['cases']
    if len(cases) != 8 or [c['split'] for c in cases] != ['development'] * 2 + ['evaluation'] * 6:
        raise ValueError('wrong case split')
    if result['status'] != 'complete' or result['forwards'] != 448 or len(result['cases']) != 8:
        raise ValueError('incomplete result')
    cues, any_id, some_id = manifest['cues'], manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    if set(cues) != set(CUES) or not 0 <= any_id < vocab or not 0 <= some_id < vocab or any_id == some_id:
        raise ValueError('invalid token identities')
    count, summaries = 0, []

    def take(kind, case_id, **identity):
        nonlocal count
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024:
            raise ValueError('missing or oversized event')
        event = json.loads(line)
        count += 1
        if event.get('kind') != kind or event.get('case_id') != case_id:
            raise ValueError('wrong event order or case')
        if any(event.get(k) != v for k, v in identity.items()):
            raise ValueError('wrong event identity')
        return event

    def row(case, cue):
        ids = case['just_ids'].copy()
        ids[4] = cues[cue]
        return ids

    def witness(event, token):
        before = raw(event['before_f32le'], width)
        after = raw(event['after_f32le'], width)
        full_before = raw(event['full_before_f32le'], 12 * width)
        full_after = raw(event['full_after_f32le'], 12 * width)
        begin, end = token * width * 4, (token + 1) * width * 4
        if full_before[begin:end] != before or full_after[begin:end] != after:
            raise ValueError('selected vectors disagree with full witnesses')
        if full_before[:begin] != full_after[:begin] or full_before[end:] != full_after[end:]:
            raise ValueError('unselected coordinates changed')
        return before, after

    for case, saved in zip(cases, result['cases']):
        ident = case['id']
        if (saved['case_id'], saved['split'], saved['forwards']) != (ident, case['split'], 56):
            raise ValueError('result case identity mismatch')
        clean, captures, margins, baseline_records = {}, {}, {}, {}
        for cue in CUES:
            event = take('clean', ident, cue=cue, input_ids=row(case, cue))
            clean[cue] = raw(event['mask_logits_f32le'], vocab)
            if set(event['captures_f32le']) != {site for site, _ in SITES}:
                raise ValueError('clean capture sites differ')
            captures[cue] = {site: raw(event['captures_f32le'][site], width) for site, _ in SITES}
            margins[cue] = margin(clean[cue], any_id, some_id)
            baseline_records[cue] = baseline(clean[cue], any_id, some_id)
        if saved['clean_margins'] != margins or saved['baselines'] != baseline_records:
            raise ValueError('saved baseline summary differs')
        distances = [{'left': left, 'right': right, 'site': site,
                      'distance': norm(captures[left][site], captures[right][site])}
                     for left, right in PAIRS for site, _ in SITES]
        r = min(d['distance'] for d in distances)
        if r < 0.001:
            raise ValueError('common radius ineligible')
        radius = take('radius', ident)
        if (radius['radius'] != r or radius['distances'] != distances or
                radius['binding_edges'] != [d for d in distances if d['distance'] == r] or
                saved['radius'] != r or saved['distances'] != distances):
            raise ValueError('common-radius record mismatch')
        effects = {}
        for site, token in SITES:
            for cue in CUES:
                event = take('self', ident, site=site, target=cue, input_ids=row(case, cue))
                before, after = witness(event, token)
                if before != captures[cue][site] or after != before or raw(event['mask_logits_f32le'], vocab) != clean[cue]:
                    raise ValueError('self patch control mismatch')
            effects[site] = {}
            for donor, target in TRANSFERS:
                event = take('transfer', ident, site=site, donor=donor, target=target,
                             input_ids=row(case, target))
                before, after = witness(event, token)
                source = captures[donor][site]
                target_clean = captures[target][site]
                if before != target_clean:
                    raise ValueError('same-forward pre-state differs from clean')
                dist = norm(target_clean, source)
                if dist <= 0 or r > dist:
                    raise ValueError('invalid or amplifying dose')
                fraction = r / dist
                submitted = raw(event['submitted_f32le'], width)
                if submitted != patch(target_clean, source, fraction) or after != submitted:
                    raise ValueError('planned/landed patch bits differ')
                planned = norm(target_clean, submitted)
                actual = norm(before, after)
                error = abs(actual - r) / r
                if (event['requested_radius'] != r or event['requested_fraction'] != fraction or
                        event['planned_norm'] != planned or event['actual_norm'] != actual or
                        event['relative_error'] != error or error > 0.001):
                    raise ValueError('actual-dose record differs')
                logits = raw(event['mask_logits_f32le'], vocab)
                if site == 'cue5' and logits != clean[target]:
                    raise ValueError('final-layer cue changed mask logits')
                effects[site][(donor, target)] = math.fsum((margin(logits, any_id, some_id), -margins[target]))
            if saved['effects'][site] != [{'donor': d, 'target': t, 'delta': effects[site][(d, t)]}
                                         for d, t in TRANSFERS]:
                raise ValueError('saved score effects differ')
        for cue in CUES:
            event = take('repeat', ident, cue=cue, input_ids=row(case, cue))
            if raw(event['mask_logits_f32le'], vocab) != clean[cue]:
                raise ValueError('repeat full logits differ')
            if set(event['captures_f32le']) != {site for site, _ in SITES}:
                raise ValueError('repeat capture sites differ')
            if any(raw(event['captures_f32le'][site], width) != captures[cue][site] for site, _ in SITES):
                raise ValueError('repeat captured states differ')
        by_cue = {cue: math.fsum((effects['cue2'][(cue, 'just')],
                                   -effects['cue2'][('just', cue)])) / 2 for cue in ('not', 'never')}
        score = math.fsum(by_cue.values()) / 2
        if saved['directional_score'] != score:
            raise ValueError('directional score mismatch')
        summaries.append({'case_id': ident, 'split': case['split'], 'radius': r,
                          'clean_gaps': {cue: math.fsum((margins[cue], -margins['just']))
                                         for cue in ('not', 'never')},
                          'directional_by_cue': by_cue, 'directional_score': score})
    if stream.readline(1) or count != 456:
        raise ValueError('extra saved events')
    if result['evaluation_cases'] != [c['id'] for c in cases[2:]]:
        raise ValueError('evaluation identity/order mismatch')
    scores = [s['directional_score'] for s in summaries[2:]]
    mean, positive = math.fsum(scores) / 6, sum(x > 0 for x in scores)
    decision = {'mean': mean, 'positive_cases': positive, 'passed': mean > 0.02 and positive >= 4}
    if result['directional_prediction'] != decision:
        raise ValueError('aggregate threshold mismatch')
    return {'status': 'passed', 'forwards': 448, 'events': count, 'cases': summaries,
            'directional_prediction': decision,
            'scope': 'saved bytes, full logits, controls, dose and landing; not execution or exposure proof'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('events', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('--expected-source')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--vocab', type=int, default=30522)
    args = p.parse_args()
    manifest_sha, event_sha, result_sha = sha(args.manifest), sha(args.events), sha(args.result)
    manifest, result = json.loads(args.manifest.read_bytes()), json.loads(args.result.read_bytes())
    if not args.synthetic:
        if args.width != 768 or args.vocab != 30522 or not re.fullmatch('[0-9a-f]{40}', args.expected_source or ''):
            raise ValueError('real audit requires fixed shapes and source revision')
        if (manifest_sha != CASE_HASH or result.get('source_commit') != args.expected_source or
                result.get('source_sha256', {}).get('experiments/bert_negation_public_cases_v1.json') != manifest_sha or
                result.get('events_sha256') != event_sha or args.events.suffix != '.gz'):
            raise ValueError('real source, case, or journal identity mismatch')
        if (result.get('state_sha256_before') != result.get('state_sha256_after') or
                not re.fullmatch('[0-9a-f]{64}', result.get('state_sha256_before', '')) or
                result.get('cpu_threads') != 1 or result.get('cpu_interop_threads') != 1 or
                result.get('attention') != 'eager' or result.get('cuda_initialized') is not False):
            raise ValueError('saved runtime/state metadata mismatch')
    opener = gzip.open if args.events.suffix == '.gz' else open
    with opener(args.events, 'rb') as stream:
        report = run(manifest, result, stream, args.width, args.vocab)
    report.update({'synthetic': args.synthetic, 'manifest_sha256': manifest_sha,
                   'events_sha256': event_sha, 'result_sha256': result_sha,
                   'source_commit': result.get('source_commit')})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
