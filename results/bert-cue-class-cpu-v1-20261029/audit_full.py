"""Replay saved cue-class logits and patch witnesses without loading a model.

This reader independently reconstructs schedules, full-vector controls, common
doses, submitted/landed float32 bits and the fixed three-frame decision.  Its
identity checks bind reported metadata and saved bytes; they do not establish
execution, external exposure, or causal generalization.  ``--synthetic`` is an
explicit relaxed mode for fabricated fixtures and must not be used for results.
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

if not __debug__:
    raise RuntimeError('optimized mode is not supported')

CUES = ('never', 'rarely', 'always', 'often')
NEG, AFF = CUES[:2], CUES[2:]
SITES = ('cue2', 'object2', 'mask2', 'cue5')
FRAMES = ('main', 'embedded', 'relative')
PAIRS = tuple(itertools.combinations(CUES, 2))
TRANSFERS = tuple((d, t) for t in CUES for d in CUES if d != t)
MANIFEST_PATH = 'experiments/bert_cue_class_public_cases_v1.json'
MANIFEST_HASH = '57a55d89097ab1b955eb4e6bab38d48a4108aa24608b96827f3d27830fc7bf8c'
SOURCE_COMMIT = '84f15048a22c094794a86a6c41946e276df876a4'
SOURCE_HASHES = {
    'docs/BERT_NEGATION_CUE_V1.md': '624210b55973b79b421256481da231b3c7ae5268ef43c9fd8d30f6b6caf5d96d',
    'experiments/bert_negation_public_cases_v1.json': '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff',
    'looking_glass/__init__.py': '9e937f02ec3e4ce0908849aa1741d8afafe378e64212362dbf67af73e11a4fe0',
    'looking_glass/bert_negation_cue.py': '2757bd3d7d694dec5a62e4c8e9fc03aae53150d522e5132b77f636ccbda8a6e4',
    'looking_glass/intervention_plan.py': '6470676938e73e48234128c30ba1d10a49104ee8513434871bc6e973c55d6120',
    'looking_glass/interventions.py': '71cbcf4776660d2dd883b82815f710b0c909228fd1762504f9e3f984f7f6c60f',
    'looking_glass/model_ownership.py': '99fee108f56ff8b7b7361e39de982c242cff590cc06c5d7a640cc12123cfe434',
    'scripts/bert_negation_cue_v1.py': '7580c4041901a567f2693c8609b939984ed5632afa30ac297af0f463c4c5e204',
    'docs/BERT_CUE_CLASS_SPECIFICITY_V1.md': 'dd7a7541eb2d1fe1056d187bb639caec875b84d10f0de33d31a036963053e686',
    MANIFEST_PATH: MANIFEST_HASH,
    'experiments/bert_cue_class_public_exposure_check_v1.json': 'b05f8ef22eb7627b00f1b75c00c0bbeaf7373b0a489885d3384229cafefde025',
    'looking_glass/bert_cue_class.py': '41c596d3a22434592279c1a1075c0bce36fa84e14adf97777d6197b895cd676e',
    'scripts/bert_cue_class_v1.py': '9e5a71b2e37bf57829732787951828d79b148ae73b121565b9385767f9266d2b',
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


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def raw(encoded, count):
    data = base64.b64decode(encoded, validate=True)
    if len(data) != count * 4:
        raise ValueError('wrong float32 byte count')
    if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f', data)):
        raise ValueError('nonfinite float32 evidence')
    return data


def values(data):
    return tuple(x[0] for x in struct.iter_unpack('<f', data))


def norm(a, b):
    aa, bb = values(a), values(b)
    if len(aa) != len(bb):
        raise ValueError('incompatible vector lengths')
    answer = math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(aa, bb)))
    if not math.isfinite(answer):
        raise ValueError('nonfinite dose')
    return answer


def patch(target, donor, fraction):
    return b''.join(struct.pack('<f', x + fraction * (y - x))
                    for x, y in zip(values(target), values(donor)))


def margin(data, any_id, some_id):
    return math.fsum((struct.unpack_from('<f', data, 4 * any_id)[0],
                      -struct.unpack_from('<f', data, 4 * some_id)[0]))


def baseline(data, any_id, some_id):
    vector = values(data)
    peak = max(vector)
    denom = math.fsum(math.exp(x - peak) for x in vector)
    top = sorted(range(len(vector)), key=lambda i: (-vector[i], i))[:5]
    return {'any_minus_some': margin(data, any_id, some_id),
            'any_probability': math.exp(vector[any_id] - peak) / denom,
            'some_probability': math.exp(vector[some_id] - peak) / denom,
            'top_five': [{'token_id': i, 'logit': vector[i]} for i in top]}


def specificity(effects):
    targets = {}
    for target in CUES:
        own, opposite, sign = (NEG, AFF, -1) if target in NEG else (AFF, NEG, 1)
        same = next(c for c in own if c != target)
        cross = {c: sign * effects[c, target] for c in opposite}
        minimum, same_delta = min(cross.values()), effects[same, target]
        value = math.fsum((minimum, -abs(same_delta)))
        targets[target] = {'cross_signed': cross, 'cross_min': minimum,
                           'same_donor': same, 'same_delta': same_delta,
                           'same_abs': abs(same_delta), 'margin': value,
                           'passed': value > 0.05}
    return {'targets': targets, 'passed': all(v['passed'] for v in targets.values())}


def run(manifest, result, stream, width, vocab):
    cases = manifest['cases']
    if (len(cases) != 12 or len({c['id'] for c in cases}) != 12 or
            [c['split'] for c in cases] != ['development'] * 3 + ['evaluation'] * 9 or
            result['status'] != 'complete' or len(result['cases']) != 12):
        raise ValueError('incomplete case schedule')
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    if any_id == some_id or min(any_id, some_id) < 0 or max(any_id, some_id) >= vocab:
        raise ValueError('invalid scored-token identities')
    count, total, summaries = 0, 0, []

    def take(kind, ident, **identity):
        nonlocal count
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024:
            raise ValueError('missing or oversized event')
        event = json.loads(line)
        count += 1
        if (event.get('kind') != kind or event.get('case_id') != ident or
                any(event.get(k) != v for k, v in identity.items())):
            raise ValueError('wrong event identity or order')
        return event

    def witness(event, token, length):
        before, after = raw(event['before_f32le'], width), raw(event['after_f32le'], width)
        full_before = raw(event['full_before_f32le'], length * width)
        full_after = raw(event['full_after_f32le'], length * width)
        first, last = token * width * 4, (token + 1) * width * 4
        if full_before[first:last] != before or full_after[first:last] != after:
            raise ValueError('selected slice disagrees with full witness')
        if full_before[:first] != full_after[:first] or full_before[last:] != full_after[last:]:
            raise ValueError('unselected token bits changed')
        return before, after

    for case, saved in zip(cases, result['cases']):
        ident, indices = case['id'], case['indices']
        if (saved['id'], saved['split'], saved['frame']) != (ident, case['split'], case['frame']):
            raise ValueError('result case identity differs')
        if case['frame'] not in FRAMES or set(indices) != {'cue', 'mask', 'object'} or set(case['rows']) != set(CUES):
            raise ValueError('invalid case layout')
        ids = {c: case['rows'][c]['input_ids'] for c in CUES}
        length = len(ids[CUES[0]])
        if (not 1 <= length <= 64 or any(len(row) != length for row in ids.values()) or
                len(set(indices.values())) != 3 or any(type(i) is not int or not 0 <= i < length for i in indices.values())):
            raise ValueError('invalid sequence length or token positions')
        cue_pos = indices['cue']
        if (len({ids[c][cue_pos] for c in CUES}) != 4 or
                any(ids[c][:cue_pos] + ids[c][cue_pos + 1:] !=
                    ids[CUES[0]][:cue_pos] + ids[CUES[0]][cue_pos + 1:] for c in CUES) or
                any(case['rows'][c]['attention_mask'] != [1] * length for c in CUES)):
            raise ValueError('cue rows are not aligned')
        if 'cue_ids' in manifest and any(ids[c][cue_pos] != manifest['cue_ids'][c] for c in CUES):
            raise ValueError('cue identities differ')
        positions = {'cue2': indices['cue'], 'object2': indices['object'],
                     'mask2': indices['mask'], 'cue5': indices['cue']}
        clean, captures, margins, baselines = {}, {}, {}, {}
        for cue in CUES:
            event = take('clean', ident, cue=cue, input_ids=ids[cue])
            clean[cue] = raw(event['mask_logits_f32le'], vocab)
            if set(event['captures_f32le']) != set(SITES):
                raise ValueError('clean capture sites differ')
            captures[cue] = {site: raw(event['captures_f32le'][site], width) for site in SITES}
            margins[cue] = margin(clean[cue], any_id, some_id)
            baselines[cue] = baseline(clean[cue], any_id, some_id)
        gaps = [{'negative': n, 'affirmative': a, 'gap': math.fsum((margins[n], -margins[a]))}
                for n in NEG for a in AFF]
        baseline_eligible = all(g['gap'] > 0.5 for g in gaps)
        event = take('baseline', ident)
        if (event['eligible'] != baseline_eligible or event['gaps'] != gaps or
                saved['baseline_eligible'] != baseline_eligible or saved['baseline_gaps'] != gaps or
                saved['clean_margins'] != margins or saved['baselines'] != baselines):
            raise ValueError('saved baseline or screen differs')
        distances = [{'left': a, 'right': b, 'site': site,
                      'distance': norm(captures[a][site], captures[b][site])}
                     for a, b in PAIRS for site in SITES]
        radius = min(d['distance'] for d in distances)
        dose_eligible = radius >= 0.001
        event = take('radius', ident)
        if (event['radius'] != radius or event['distances'] != distances or
                event['binding_edges'] != [d for d in distances if d['distance'] == radius] or
                event['dose_eligible'] != dose_eligible or saved['radius'] != radius or
                saved['distances'] != distances or saved['dose_eligible'] != dose_eligible):
            raise ValueError('common-radius evidence differs')
        details = {}
        if not dose_eligible:
            take('dose_ineligible', ident, radius=radius)
            if saved['effects'] or saved['specificity'] or saved['specificity_passed'] is not None:
                raise ValueError('ineligible case reports interventions')
        else:
            for site in SITES:
                for cue in CUES:
                    event = take('self', ident, site=site, target=cue, input_ids=ids[cue])
                    before, after = witness(event, positions[site], length)
                    if (before != captures[cue][site] or after != before or
                            raw(event['mask_logits_f32le'], vocab) != clean[cue]):
                        raise ValueError('self control differs')
                effects = {}
                for donor, target in TRANSFERS:
                    event = take('transfer', ident, site=site, donor=donor, target=target,
                                 input_ids=ids[target])
                    before, after = witness(event, positions[site], length)
                    target_clean, source = captures[target][site], captures[donor][site]
                    if before != target_clean:
                        raise ValueError('pre-action state differs from clean capture')
                    distance = norm(target_clean, source)
                    if distance <= 0 or radius > distance:
                        raise ValueError('missing direction or amplifying dose')
                    fraction = radius / distance
                    submitted = raw(event['submitted_f32le'], width)
                    if submitted != patch(target_clean, source, fraction) or after != submitted:
                        raise ValueError('submitted or landed patch bits differ')
                    planned, actual = norm(target_clean, submitted), norm(before, after)
                    error = abs(actual - radius) / radius
                    if (event['requested_radius'] != radius or event['requested_fraction'] != fraction or
                            event['planned_norm'] != planned or event['actual_norm'] != actual or
                            event['relative_error'] != error or error > 0.001):
                        raise ValueError('dose record differs')
                    logits = raw(event['mask_logits_f32le'], vocab)
                    if site == 'cue5' and logits != clean[target]:
                        raise ValueError('final-cue control changed mask logits')
                    effects[donor, target] = math.fsum((margin(logits, any_id, some_id), -margins[target]))
                if saved['effects'][site] != [{'donor': d, 'target': t, 'delta': effects[d, t]}
                                               for d, t in TRANSFERS]:
                    raise ValueError('stored effects differ')
                details[site] = specificity(effects)
                if saved['specificity'][site] != details[site]:
                    raise ValueError('stored target decision differs')
            if saved['specificity_passed'] != details['cue2']['passed']:
                raise ValueError('cue specificity decision differs')
        for cue in CUES:
            event = take('repeat', ident, cue=cue, input_ids=ids[cue])
            if raw(event['mask_logits_f32le'], vocab) != clean[cue]:
                raise ValueError('repeat full logits differ')
            if set(event['captures_f32le']) != set(SITES) or any(
                    raw(event['captures_f32le'][site], width) != captures[cue][site] for site in SITES):
                raise ValueError('repeat clean captures differ')
        expected = 72 if dose_eligible else 8
        if saved['forwards'] != expected:
            raise ValueError('case forward count differs')
        total += expected
        summaries.append({'id': ident, 'split': case['split'], 'frame': case['frame'],
                          'forwards': expected, 'radius': radius, 'baseline_eligible': baseline_eligible,
                          'baseline_gaps': gaps, 'dose_eligible': dose_eligible,
                          'clean_margins': margins, 'specificity': details,
                          'specificity_passed': details['cue2']['passed'] if dose_eligible else None})
    if stream.readline(1) or result['forwards'] != total or count != total + 2 * 12 + sum(
            not row['dose_eligible'] for row in summaries):
        raise ValueError('extra event or total forward count differs')
    frames = {}
    for frame in FRAMES:
        members = [r for r in summaries if r['split'] == 'evaluation' and r['frame'] == frame]
        if len(members) != 3:
            raise ValueError('evaluation frame count differs')
        eligible = [r for r in members if r['baseline_eligible'] and r['dose_eligible']]
        frames[frame] = {'total': 3, 'eligible': len(eligible),
                         'passing': sum(r['specificity_passed'] is True for r in eligible)}
    status = ('unevaluable' if any(f['eligible'] < 2 for f in frames.values()) else
              'passed' if all(f['passing'] >= 2 for f in frames.values()) else 'failed')
    evaluation = {'status': status, 'frames': frames}
    if result['evaluation'] != evaluation:
        raise ValueError('frame-level decision differs')
    return {'status': 'passed', 'forwards': total, 'events': count, 'cases': summaries,
            'evaluation': evaluation,
            'scope': 'saved logits, control witnesses, dose and landing; not execution or exposure proof'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('events', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('--started', type=Path)
    p.add_argument('--expected-started-sha256')
    p.add_argument('--expected-result-sha256')
    p.add_argument('--expected-source')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--vocab', type=int, default=30522)
    a = p.parse_args()
    if a.width < 1 or a.vocab < 5:
        raise ValueError('invalid dimensions')
    manifest_sha, events_sha, result_sha = sha(a.manifest), sha(a.events), sha(a.result)
    manifest, result = json.loads(a.manifest.read_bytes()), json.loads(a.result.read_bytes())
    if not a.synthetic:
        if (a.width != 768 or a.vocab != 30522 or a.events.suffix != '.gz' or
                a.expected_source != SOURCE_COMMIT or
                not re.fullmatch('[0-9a-f]{64}', a.expected_result_sha256 or '') or
                result_sha != a.expected_result_sha256 or manifest_sha != MANIFEST_HASH or
                result.get('source_commit') != SOURCE_COMMIT or
                result.get('source_sha256') != SOURCE_HASHES or result.get('assets') != ASSETS or
                result.get('versions') != VERSIONS or
                result.get('events_sha256') != events_sha):
            raise ValueError('source, manifest, result or journal binding differs')
        if (not a.started or not re.fullmatch('[0-9a-f]{64}', a.expected_started_sha256 or '') or
                sha(a.started) != a.expected_started_sha256):
            raise ValueError('started-record binding differs')
        started = json.loads(a.started.read_bytes())
        if any(started.get(k) != result.get(k) for k in
               ('source_commit', 'source_sha256', 'assets', 'versions', 'seed', 'cpu_environment')):
            raise ValueError('started/result metadata differs')
        if (started.get('maximum_forwards') != 864 or result.get('seed') != 2026102902 or
                result.get('tokenizer') != {'rows': 48, 'any_id': 2151, 'some_id': 2070} or
                result.get('state_sha256_before') != result.get('state_sha256_after') or
                not re.fullmatch('[0-9a-f]{64}', result.get('state_sha256_before', '')) or
                result.get('cpu_threads') != 1 or result.get('cpu_interop_threads') != 1 or
                result.get('attention') != 'eager' or result.get('cuda_initialized') is not False):
            raise ValueError('saved runtime/state metadata differs')
    opener = gzip.open if a.events.suffix == '.gz' else open
    with opener(a.events, 'rb') as stream:
        report = run(manifest, result, stream, a.width, a.vocab)
    report.update({'synthetic': a.synthetic, 'source_commit': result.get('source_commit'),
                   'manifest_sha256': manifest_sha, 'events_sha256': events_sha,
                   'result_sha256': result_sha,
                   'identity_scope': ('Recorded source/asset/version pins and start/result consistency; '
                                      'not an attestation that these bytes executed or were loaded.')})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
