"""Recompute fixed negation-cue decisions from saved mask-logit bytes only.

This separate stdlib reader validates the action order, input token rows,
target logits, baseline gaps, directional effects and decision. It does not
establish model execution, runtime identity, patch landing, or dose validity.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

CUES = ('not', 'never', 'just', 'also')
SITES = ('cue2', 'object2', 'mask2', 'cue5')
TRANSFERS = (('not', 'just'), ('just', 'not'), ('never', 'just'), ('just', 'never'),
             ('also', 'just'), ('just', 'also'), ('also', 'not'), ('also', 'never'))

def margin(event, any_id, some_id, expected_vocab):
    raw = base64.b64decode(event['mask_logits_f32le'], validate=True)
    if (len(raw) % 4 or len(raw) < 4 * (1 + max(any_id, some_id))
            or (expected_vocab is not None and len(raw) != 4 * expected_vocab)):
        raise ValueError('malformed mask logits')
    a = struct.unpack_from('<f', raw, 4 * any_id)[0]
    b = struct.unpack_from('<f', raw, 4 * some_id)[0]
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('nonfinite scored logits')
    return math.fsum((a, -b))

def audit(events, result, manifest, *, synthetic=False):
    cases = manifest['cases']
    if len(cases) != 8 or [c['split'] for c in cases] != ['development'] * 2 + ['evaluation'] * 6:
        raise ValueError('unexpected fixed split')
    if len(events) != 456 or result['status'] != 'complete' or result['forwards'] != 448:
        raise ValueError('incomplete event schedule')
    if len(result['cases']) != 8:
        raise ValueError('unexpected result count')
    cue_ids = manifest['cues']
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    expected_vocab = None if synthetic else 30522
    pos = 0
    calculated = []
    def take(kind, case_id, **expected):
        nonlocal pos
        e = events[pos]
        pos += 1
        if e.get('kind') != kind or e.get('case_id') != case_id or any(e.get(k) != v for k, v in expected.items()):
            raise ValueError('unexpected event identity or order')
        return e
    for case, saved in zip(cases, result['cases']):
        case_id = case['id']
        if saved['case_id'] != case_id or saved['split'] != case['split'] or saved['forwards'] != 56:
            raise ValueError('result case identity mismatch')
        def input_row(cue):
            ids = case['just_ids'].copy()
            ids[4] = cue_ids[cue]
            return ids
        clean = {}
        for cue in CUES:
            event = take('clean', case_id, cue=cue, input_ids=input_row(cue))
            clean[cue] = margin(event, any_id, some_id, expected_vocab)
        take('radius', case_id)
        if saved['clean_margins'] != clean:
            raise ValueError('clean target margins differ')
        effects = {}
        for site in SITES:
            for cue in CUES:
                event = take('self', case_id, site=site, target=cue, input_ids=input_row(cue))
                if margin(event, any_id, some_id, expected_vocab) != clean[cue]:
                    raise ValueError('self target margin differs')
            effects[site] = {}
            for donor, target in TRANSFERS:
                event = take('transfer', case_id, site=site, donor=donor, target=target, input_ids=input_row(target))
                effects[site][(donor, target)] = math.fsum((margin(event, any_id, some_id, expected_vocab), -clean[target]))
            expected = [{'donor': d, 'target': t, 'delta': effects[site][(d, t)]} for d, t in TRANSFERS]
            if saved['effects'][site] != expected:
                raise ValueError('saved effects differ from raw target logits')
        for cue in CUES:
            event = take('repeat', case_id, cue=cue, input_ids=input_row(cue))
            if margin(event, any_id, some_id, expected_vocab) != clean[cue]:
                raise ValueError('repeat target margin differs')
        by_site = {site: {cue: math.fsum((effects[site][(cue, 'just')],
                                          -effects[site][('just', cue)])) / 2
                          for cue in ('not', 'never')} for site in SITES}
        by_cue = [by_site['cue2'][cue] for cue in ('not', 'never')]
        score = math.fsum(by_cue) / 2
        if saved['directional_score'] != score:
            raise ValueError('case directional score mismatch')
        calculated.append({'case_id': case_id, 'split': case['split'], 'directional_score': score,
                           'clean_gaps': {cue: math.fsum((clean[cue], -clean['just'])) for cue in ('not', 'never')},
                           'bidirectional_by_site': by_site,
                           'cue_to_just': {cue: effects['cue2'][(cue, 'just')]
                                           for cue in ('not', 'never', 'also')},
                           'cue_vs_also_same_recipient': {
                               cue: math.fsum((effects['cue2'][(cue, 'just')],
                                               -effects['cue2'][('also', 'just')]))
                               for cue in ('not', 'never')}})
    if pos != len(events) or result['evaluation_cases'] != [x['id'] for x in cases[2:]]:
        raise ValueError('extra event or wrong evaluation order')
    evaluation = [x['directional_score'] for x in calculated[2:]]
    mean = math.fsum(evaluation) / 6
    count = sum(x > 0 for x in evaluation)
    decision = {'mean': mean, 'positive_cases': count, 'passed': mean > 0.02 and count >= 4}
    if result['directional_prediction'] != decision:
        raise ValueError('fixed aggregate decision mismatch')
    return {'status': 'passed', 'scope': 'saved target logits, fixed schedule, input rows and decision; not execution or landing',
            'forwards': 448, 'events': pos, 'cases': calculated, 'directional_prediction': decision}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('events', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('--synthetic', action='store_true', help='allow small fabricated fixtures and raw JSONL')
    args = parser.parse_args()
    raw_manifest, raw_result = args.manifest.read_bytes(), args.result.read_bytes()
    result = json.loads(raw_result)
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    if not args.synthetic and manifest_sha != '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff':
        raise ValueError('manifest differs from the frozen public cases')
    if not args.synthetic and result.get('source_sha256', {}).get('experiments/bert_negation_public_cases_v1.json') != manifest_sha:
        raise ValueError('result does not bind the frozen case manifest')
    hasher = hashlib.sha256()
    with args.events.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    event_sha = hasher.hexdigest()
    if not args.synthetic and result.get('events_sha256') != event_sha:
        raise ValueError('result does not bind the compressed event journal')
    if not args.synthetic and args.events.suffix != '.gz':
        raise ValueError('the real result requires the compressed journal')
    events = []
    opener = gzip.open if args.events.suffix == '.gz' else open
    with opener(args.events, 'rb') as stream:
        while line := stream.readline(2 * 1024 * 1024 + 1):
            if len(line) > 2 * 1024 * 1024 or len(events) >= 456:
                raise ValueError('oversized or excessive event records')
            events.append(json.loads(line))
    report = audit(events, result, json.loads(raw_manifest), synthetic=args.synthetic)
    report.update({'manifest_sha256': manifest_sha, 'events_sha256': event_sha,
                   'result_sha256': hashlib.sha256(raw_result).hexdigest(),
                   'synthetic': args.synthetic})
    print(json.dumps(report, sort_keys=True, allow_nan=False))

if __name__ == '__main__':
    main()
