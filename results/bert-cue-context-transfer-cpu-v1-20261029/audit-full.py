"""Replay saved cue-context geometry and full-vector controls without a model.

This adapts the earlier full-witness reader to the fixed two-context schedule.
It validates saved state/score consistency, not execution or generalization.
The separate scalar reader owns full-vocabulary TV and probability replay.
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

CUES = ('never', 'rarely', 'always', 'often')
SITES = ('cue2', 'object2', 'mask2', 'cue5')
PAIR_IDS = ('discovery-01', 'evaluation-01', 'evaluation-02')
MANIFEST_HASH = '4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7'
# Independently read back against the reviewed public execution tree.
SOURCE_COMMIT = '40752bde00f11b69832066c2d16ce52c49f96810'
SOURCE_HASHES = {
    'docs/BERT_CUE_CONTEXT_TRANSFER_V1.md': '2404699a8c626572c11567a56169d5ae95267bbf7b30b8e617368520653f217e',
    'docs/BERT_NEGATION_CUE_V1.md': '624210b55973b79b421256481da231b3c7ae5268ef43c9fd8d30f6b6caf5d96d',
    'experiments/bert_cue_context_public_exposure_check_v1.json': '1c04ca6e268b0b84337ed47cbaf037ef4790766c2861dc773820abd6c70100b9',
    'experiments/bert_cue_context_public_exposure_independent_v1.json': 'afcbccb6360ade55bc959818ddbcea1768c0e5a4a74c9df694ac307baeab3a1c',
    'experiments/bert_cue_context_transfer_cases_v1.json': '4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7',
    'experiments/bert_negation_public_cases_v1.json': '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff',
    'looking_glass/__init__.py': '9e937f02ec3e4ce0908849aa1741d8afafe378e64212362dbf67af73e11a4fe0',
    'looking_glass/bert_cue_class.py': 'bda7e8e60bdbe66966cd5f6b66be8c965a9a6e70eff14ba2d32a8fc14d3d0513',
    'looking_glass/bert_cue_context_transfer.py': '58f541ea544140122a195dfff7f4d9061a9b11436185f96254482e4f1f2a299a',
    'looking_glass/bert_cue_factorial.py': '8fde76425176c7c09214a4b73e8d6186087b3545c7882ced0098afb0d1704955',
    'looking_glass/bert_negation_cue.py': '2757bd3d7d694dec5a62e4c8e9fc03aae53150d522e5132b77f636ccbda8a6e4',
    'looking_glass/intervention_plan.py': '6470676938e73e48234128c30ba1d10a49104ee8513434871bc6e973c55d6120',
    'looking_glass/interventions.py': '71cbcf4776660d2dd883b82815f710b0c909228fd1762504f9e3f984f7f6c60f',
    'looking_glass/model_ownership.py': '99fee108f56ff8b7b7361e39de982c242cff590cc06c5d7a640cc12123cfe434',
    'scripts/bert_cue_context_transfer_v1.py': '46f8a2ef6fe7ea14496a55093588a47ca043a9e3322bb9a7feaf332497c1a161',
    'scripts/bert_negation_cue_v1.py': '7580c4041901a567f2693c8609b939984ed5632afa30ac297af0f463c4c5e204',
    'scripts/build_bert_cue_context_cases.py': '0200e6ea541c71733e1bc70249bcf9ebcc50cd2351adfc04e5dc6419f8b4c9ba',
}
ASSETS = {
    'config.json': {'bytes': 483, 'sha256': '69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83'},
    'model.safetensors': {'bytes': 267954768, 'sha256': '5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'},
    'tokenizer.json': {'bytes': 466062, 'sha256': 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'},
    'tokenizer_config.json': {'bytes': 48, 'sha256': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'},
    'vocab.txt': {'bytes': 231508, 'sha256': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'},
}
VERSIONS = {'torch': '2.8.0+cu128', 'transformers': '4.57.3', 'tokenizers': '0.22.2',
            'safetensors': '0.8.0', 'numpy': '2.4.6'}


def numeric_mode():
    tiny64 = struct.unpack('<d', bytes.fromhex('0100000000000000'))[0]
    neg64 = struct.unpack('<d', bytes.fromhex('0100000000000080'))[0]
    tiny32 = struct.unpack('<f', bytes.fromhex('01000000'))[0]
    neg32 = struct.unpack('<f', bytes.fromhex('01000080'))[0]
    if (struct.pack('<d', tiny64 * 2) != bytes.fromhex('0200000000000000') or
            struct.pack('<d', neg64 * 2) != bytes.fromhex('0200000000000080') or
            struct.pack('<d', tiny32) != struct.pack('<d', math.ldexp(1.0, -149)) or
            struct.pack('<d', neg32) != struct.pack('<d', -math.ldexp(1.0, -149)) or
            struct.pack('<f', 1 + 2 ** -24) != bytes.fromhex('0000803f') or
            struct.pack('<f', 1 + 3 * 2 ** -24) != bytes.fromhex('0200803f')):
        raise ValueError('floating-point mode differs from the required mode')


def strict_json(data):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def exact(a, b):
    """Compare JSON values without treating booleans as integers."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def raw(encoded, count):
    data = base64.b64decode(encoded, validate=True)
    if len(data) != count * 4 or any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', data)):
        raise ValueError('wrong or nonfinite f32 witness')
    return data


def floats(data):
    return tuple(v[0] for v in struct.iter_unpack('<f', data))


def norm(a, b):
    x, y = floats(a), floats(b)
    if len(x) != len(y):
        raise ValueError('vector widths differ')
    answer = math.sqrt(math.fsum((float(i) - float(j)) ** 2 for i, j in zip(x, y)))
    if not math.isfinite(answer):
        raise ValueError('nonfinite distance')
    return answer


def patch(a, b, fraction):
    return b''.join(struct.pack('<f', x + fraction * (y - x)) for x, y in zip(floats(a), floats(b)))


def margin(data, any_id, some_id):
    return math.fsum((struct.unpack_from('<f', data, any_id * 4)[0],
                      -struct.unpack_from('<f', data, some_id * 4)[0]))


def donors(context, cue):
    return [(1 - context, c) for c in CUES] + [(context, c) for c in CUES if c != cue]


def other_classmate(cue):
    group = CUES[:2] if cue in CUES[:2] else CUES[2:]
    return next(x for x in group if x != cue)


def pair_decision(effects, baseline, eligible):
    if not eligible:
        return {'status': 'dose_ineligible', 'informative': False, 'recipients': [],
                'same_cue_mean': None, 'same_class_mean': None, 'half_difference': None}
    rows = []
    for ci in range(2):
        for cue in CUES:
            rows.append({'target_context': ci, 'target_cue': cue,
                         'same_cue_delta': effects['cue2', ci, cue, 1-ci, cue],
                         'same_class_delta': effects['cue2', ci, cue, 1-ci, other_classmate(cue)]})
    a = math.fsum(abs(r['same_cue_delta']) for r in rows) / 8
    b = math.fsum(abs(r['same_class_delta']) for r in rows) / 8
    h = math.fsum((a, -b / 2))
    status = ('baseline_ineligible' if not baseline else 'uninformative' if b < 0.01
              else 'at_least_half' if h >= 0 else 'below_half')
    return {'status': status, 'informative': baseline and b >= 0.01, 'recipients': rows,
            'same_cue_mean': a, 'same_class_mean': b, 'half_difference': h}


def aggregate(pairs):
    ev = pairs[1:]
    informative = [p for p in ev if p['comparison']['informative']]
    if len(informative) != 2:
        return {'status': 'unevaluable', 'informative_pairs': len(informative),
                'recipients': 8 * len(informative), 'same_cue_mean': None,
                'same_class_mean': None, 'half_difference': None}
    rows = [r for p in ev for r in p['comparison']['recipients']]
    if len(rows) != 16:
        raise ValueError('evaluation recipient count differs')
    a = math.fsum(abs(r['same_cue_delta']) for r in rows) / 16
    b = math.fsum(abs(r['same_class_delta']) for r in rows) / 16
    h = math.fsum((a, -b / 2))
    return {'status': 'at_least_half' if h >= 0 else 'below_half', 'informative_pairs': 2,
            'recipients': 16, 'same_cue_mean': a, 'same_class_mean': b, 'half_difference': h}


def run(manifest, result, stream, width, vocab):
    pairs = manifest['pairs']
    if (tuple(p['id'] for p in pairs) != PAIR_IDS or
            [p['split'] for p in pairs] != ['discovery', 'evaluation', 'evaluation'] or
            result['status'] != 'complete' or len(result['pairs']) != 3):
        raise ValueError('fixed pair order or completed result differs')
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    if (type(any_id) is not int or type(some_id) is not int or any_id == some_id or
            min(any_id, some_id) < 0 or max(any_id, some_id) >= vocab):
        raise ValueError('scored token identities differ')
    events = total = 0
    summaries = []

    def take(kind, pair, **expected):
        nonlocal events
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024 or not line.endswith(b'\n'):
            raise ValueError('missing, oversized or incomplete event')
        event = strict_json(line)
        if (event.get('kind') != kind or event.get('pair_id') != pair or
                any(k not in event or not exact(event[k], v) for k, v in expected.items())):
            raise ValueError('event identity or schedule differs')
        events += 1
        return event

    def witness(event, token, length):
        before, after = raw(event['before_f32le'], width), raw(event['after_f32le'], width)
        full_before = raw(event['full_before_f32le'], length * width)
        full_after = raw(event['full_after_f32le'], length * width)
        a, b = token * width * 4, (token + 1) * width * 4
        if (full_before[a:b] != before or full_after[a:b] != after or
                full_before[:a] != full_after[:a] or full_before[b:] != full_after[b:]):
            raise ValueError('selected or unselected full-state bits differ')
        return before, after

    for pair, saved in zip(pairs, result['pairs'], strict=True):
        pid, contexts = pair['id'], pair['contexts']
        if len(contexts) != 2 or saved['id'] != pid or saved['split'] != pair['split']:
            raise ValueError('pair or context identity differs')
        ids, positions = {}, {}
        for ci, context in enumerate(contexts):
            ix = context['indices']
            if (set(ix) != {'cue', 'mask', 'object'} or len(set(ix.values())) != 3 or
                    set(context['rows']) != set(CUES)):
                raise ValueError('context token layout differs')
            length = len(context['rows'][CUES[0]]['input_ids'])
            if not 1 <= length <= 64 or any(type(x) is not int or not 0 <= x < length for x in ix.values()):
                raise ValueError('invalid context shape')
            positions[ci] = {'cue2': ix['cue'], 'object2': ix['object'],
                             'mask2': ix['mask'], 'cue5': ix['cue']}
            for cue in CUES:
                row = context['rows'][cue]
                ids[ci, cue] = row['input_ids']
                if (len(ids[ci, cue]) != length or row['attention_mask'] != [1] * length or
                        any(type(x) is not int for x in ids[ci, cue])):
                    raise ValueError('invalid input row or mask')
        clean, captures, margins = {}, {}, {}
        for ci in range(2):
            for cue in CUES:
                event = take('clean', pid, context=ci, cue=cue, input_ids=ids[ci, cue])
                clean[ci, cue] = raw(event['mask_logits_f32le'], vocab)
                if set(event['captures_f32le']) != set(SITES):
                    raise ValueError('clean capture sites differ')
                captures[ci, cue] = {s: raw(event['captures_f32le'][s], width) for s in SITES}
                margins[ci, cue] = margin(clean[ci, cue], any_id, some_id)
        screens = []
        for ci in range(2):
            gaps = [{'negative': n, 'affirmative': a, 'gap': math.fsum((margins[ci, n], -margins[ci, a]))}
                    for n in CUES[:2] for a in CUES[2:]]
            screens.append({'eligible': all(g['gap'] > 0.5 for g in gaps), 'gaps': gaps})
        baseline = all(s['eligible'] for s in screens)
        event = take('baseline', pid)
        if (type(event['eligible']) is not bool or event['eligible'] != baseline or
                not exact(event['contexts'], screens) or not exact(saved['baseline_screens'], screens) or
                type(saved['baseline_eligible']) is not bool or saved['baseline_eligible'] != baseline):
            raise ValueError('baseline screen differs')
        distances = []
        for site in SITES:
            for ci in range(2):
                for cue in CUES:
                    for di, donor in donors(ci, cue):
                        distances.append({'site': site, 'target_context': ci, 'target_cue': cue,
                                          'donor_context': di, 'donor_cue': donor,
                                          'distance': norm(captures[ci, cue][site], captures[di, donor][site])})
        minimum = min(x['distance'] for x in distances)
        q = min(0.5, minimum)
        eligible = q >= 0.05
        event = take('radius', pid)
        if (not exact(event['distances'], distances) or not exact(event['minimum_distance'], minimum) or
                not exact(event['q'], q) or
                not exact(event['binding_edges'], [r for r in distances if r['distance'] == minimum]) or
                type(event['dose_eligible']) is not bool or event['dose_eligible'] != eligible or
                not exact(saved['distances'], distances) or not exact(saved['minimum_distance'], minimum) or
                not exact(saved['q'], q) or
                type(saved['dose_eligible']) is not bool or saved['dose_eligible'] != eligible):
            raise ValueError('natural distance, radius or eligibility differs')
        if not eligible:
            take('dose_ineligible', pid, q=q)
        effects = {}
        if eligible:
            for site in SITES:
                for ci in range(2):
                    for cue in CUES:
                        event = take('self', pid, site=site, target_context=ci, target_cue=cue,
                                     input_ids=ids[ci, cue])
                        before, after = witness(event, positions[ci][site], len(ids[ci, cue]))
                        if (before != captures[ci, cue][site] or after != before or
                                raw(event['mask_logits_f32le'], vocab) != clean[ci, cue]):
                            raise ValueError('self control differs')
                for ci in range(2):
                    for cue in CUES:
                        for di, donor in donors(ci, cue):
                            event = take('transfer', pid, site=site, target_context=ci, target_cue=cue,
                                         donor_context=di, donor_cue=donor, input_ids=ids[ci, cue])
                            before, after = witness(event, positions[ci][site], len(ids[ci, cue]))
                            target, source = captures[ci, cue][site], captures[di, donor][site]
                            distance = norm(target, source)
                            if before != target or distance <= 0 or q > distance:
                                raise ValueError('pre-state, natural direction or dose differs')
                            fraction = q / distance
                            submitted = raw(event['submitted_f32le'], width)
                            if submitted != patch(target, source, fraction) or after != submitted:
                                raise ValueError('submitted or landed bits differ')
                            planned, actual = norm(target, submitted), norm(before, after)
                            error = abs(actual - q) / q
                            if (not exact(event['requested_radius'], q) or not exact(event['requested_fraction'], fraction) or
                                    not exact(event['planned_norm'], planned) or not exact(event['actual_norm'], actual) or
                                    not exact(event['relative_error'], error) or actual <= 0 or error > 0.001):
                                raise ValueError('achieved dose differs')
                            logits = raw(event['mask_logits_f32le'], vocab)
                            if site == 'cue5' and logits != clean[ci, cue]:
                                raise ValueError('final cue altered full mask logits')
                            effects[site, ci, cue, di, donor] = math.fsum(
                                (margin(logits, any_id, some_id), -margins[ci, cue]))
        expected_effects = [{'site': site, 'target_context': ci, 'target_cue': cue,
                             'donor_context': di, 'donor_cue': donor, 'delta': value}
                            for (site, ci, cue, di, donor), value in effects.items()]
        if len(saved['effects']) != len(expected_effects) or any(
                any(k not in row or not exact(row[k], value) for k, value in expected.items())
                for row, expected in zip(saved['effects'], expected_effects, strict=True)):
            raise ValueError('signed transfer effects or their order differ')
        for ci in range(2):
            for cue in CUES:
                event = take('repeat', pid, context=ci, cue=cue, input_ids=ids[ci, cue])
                if raw(event['mask_logits_f32le'], vocab) != clean[ci, cue] or set(event['captures_f32le']) != set(SITES):
                    raise ValueError('repeat logits or capture sites differ')
                if any(raw(event['captures_f32le'][site], width) != captures[ci, cue][site] for site in SITES):
                    raise ValueError('repeat capture differs')
        count = 272 if eligible else 16
        decision = pair_decision(effects, baseline, eligible)
        if not exact(saved['forwards'], count) or not exact(saved['comparison'], decision):
            raise ValueError('pair count or comparison differs')
        total += count
        summaries.append({'id': pid, 'split': pair['split'], 'forwards': count,
                          'baseline_eligible': baseline, 'dose_eligible': eligible,
                          'minimum_distance': minimum, 'q': q, 'comparison': decision})
    if stream.readline(1) or not exact(result['forwards'], total) or not exact(result['evaluation'], aggregate(summaries)):
        raise ValueError('total schedule or aggregate differs')
    return {'status': 'passed', 'forwards': total, 'events': events, 'pairs': summaries,
            'evaluation': aggregate(summaries),
            'scope': 'saved full-logit controls, geometry, landing and selected effects; '
                     'TV/probabilities require the separate scalar audit; not execution proof'}


def valid_cpu_environment(env):
    if type(env) is not dict or set(env) != {'MKL_CBWR', 'ATEN_CPU_CAPABILITY'}:
        return False
    for value in env.values():
        if (type(value) is not dict or set(value) != {'present', 'value'} or
                type(value['present']) is not bool or
                (type(value['value']) is not str if value['present'] else value['value'] is not None)):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('events', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('--started', type=Path)
    p.add_argument('--expected-started-sha256')
    p.add_argument('--expected-result-sha256')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--vocab', type=int, default=30522)
    args = p.parse_args()
    numeric_mode()
    paths = (args.manifest, args.events, args.result)
    hashes = tuple(sha(path) for path in paths)
    manifest, result = strict_json(args.manifest.read_bytes()), strict_json(args.result.read_bytes())
    if not args.synthetic:
        if SOURCE_COMMIT is None or SOURCE_HASHES is None:
            raise ValueError('real source pins have not yet been frozen')
        if (args.width, args.vocab) != (768, 30522) or args.started is None:
            raise ValueError('real inputs require the fixed shape and started record')
        if (not re.fullmatch('[0-9a-f]{64}', args.expected_started_sha256 or '') or
                not re.fullmatch('[0-9a-f]{64}', args.expected_result_sha256 or '') or
                hashes[0] != MANIFEST_HASH or hashes[2] != args.expected_result_sha256 or
                sha(args.started) != args.expected_started_sha256 or result.get('events_sha256') != hashes[1]):
            raise ValueError('manifest, started, result or journal hash differs')
        started = strict_json(args.started.read_bytes())
        tokenization = {'pairs': 3, 'contexts': 6, 'rows': 24, 'any_id': 2151, 'some_id': 2070}
        if (started.get('source_commit') != SOURCE_COMMIT or result.get('source_commit') != SOURCE_COMMIT or
                not exact(started.get('source_sha256'), SOURCE_HASHES) or not exact(result.get('source_sha256'), SOURCE_HASHES) or
                not exact(started.get('assets'), ASSETS) or not exact(result.get('assets'), ASSETS) or
                not exact(started.get('versions'), VERSIONS) or not exact(result.get('versions'), VERSIONS) or
                not exact(started.get('seed'), 2026102906) or not exact(result.get('seed'), 2026102906) or
                not exact(started.get('tokenizer'), tokenization) or not exact(result.get('tokenizer'), tokenization) or
                not exact(started.get('maximum_forwards'), 816) or not exact(started.get('event_max'), 2 * 1024**2) or
                not exact(started.get('journal_max'), 512 * 1024**2) or args.events.stat().st_size > 512 * 1024**2 or
                not valid_cpu_environment(started.get('cpu_environment')) or
                not exact(result.get('cpu_environment'), started.get('cpu_environment')) or
                not exact(result.get('cpu_threads'), 1) or not exact(result.get('cpu_interop_threads'), 1) or
                result.get('attention') != 'eager' or result.get('cuda_initialized') is not False or
                not re.fullmatch('[0-9a-f]{64}', result.get('state_sha256_before', '')) or
                result.get('state_sha256_before') != result.get('state_sha256_after')):
            raise ValueError('source, runtime, state or asset metadata differs')
    with gzip.open(args.events, 'rb') as stream:
        report = run(manifest, result, stream, args.width, args.vocab)
    if tuple(sha(path) for path in paths) != hashes:
        raise ValueError('an input changed during replay')
    if not args.synthetic and sha(args.started) != args.expected_started_sha256:
        raise ValueError('started record changed during replay')
    report.update({'manifest_sha256': hashes[0], 'events_sha256': hashes[1], 'result_sha256': hashes[2],
                   'source_commit': None if args.synthetic else SOURCE_COMMIT, 'synthetic': args.synthetic,
                   'identity_scope': 'saved-byte and reported-metadata consistency; not execution attestation'})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
