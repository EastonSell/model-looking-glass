"""Replay the four-cell two-dose continuation without loading a model.

This reader adapts the earlier full-witness audit to the four-cell schedule.
It reconstructs full-vector controls, doses, submitted/landed float32 bits and
per-cell decisions; any cross-process pooling requires a separate audit. Its
identity checks bind reported metadata and saved bytes; they do not establish
execution, external exposure, or causal generalization. ``--synthetic`` is an
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

GROUPS = {'prior': ('never', 'rarely', 'always', 'often'),
          'new': ('not', 'hardly', 'really', 'certainly')}
SITES = ('cue2', 'object2', 'mask2', 'cue5')
TRIPLES = ('fresh-01', 'fresh-02', 'fresh-03')
CALIBRATION = 'exposed-calibration'
DOSES = {'low': 0.5, 'high': 0.7}
MANIFEST_PATH = 'experiments/bert_cue_dose_response_cases_v1.json'
MANIFEST_HASH = '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a'
# Real mode binds the public runner and every recorded source path.
# Synthetic fixtures use --synthetic.
SOURCE_COMMIT = 'cdc9630e24542fed84b0c9556bb7f37c60d722d1'
SOURCE_HASHES = {
    'docs/BERT_NEGATION_CUE_V1.md': '624210b55973b79b421256481da231b3c7ae5268ef43c9fd8d30f6b6caf5d96d',
    'experiments/bert_negation_public_cases_v1.json': '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff',
    'looking_glass/__init__.py': '9e937f02ec3e4ce0908849aa1741d8afafe378e64212362dbf67af73e11a4fe0',
    'looking_glass/bert_negation_cue.py': '2757bd3d7d694dec5a62e4c8e9fc03aae53150d522e5132b77f636ccbda8a6e4',
    'looking_glass/intervention_plan.py': '6470676938e73e48234128c30ba1d10a49104ee8513434871bc6e973c55d6120',
    'looking_glass/interventions.py': '71cbcf4776660d2dd883b82815f710b0c909228fd1762504f9e3f984f7f6c60f',
    'looking_glass/model_ownership.py': '99fee108f56ff8b7b7361e39de982c242cff590cc06c5d7a640cc12123cfe434',
    'scripts/bert_negation_cue_v1.py': '7580c4041901a567f2693c8609b939984ed5632afa30ac297af0f463c4c5e204',
    'looking_glass/bert_cue_class.py': '41c596d3a22434592279c1a1075c0bce36fa84e14adf97777d6197b895cd676e',
    'looking_glass/bert_cue_factorial.py': '8fde76425176c7c09214a4b73e8d6186087b3545c7882ced0098afb0d1704955',
    'experiments/bert_cue_factorial_cases_v1.json': 'b91eba1e0a52f6a183a59b0bfbb365dd54d67341bd74a4fb56325e72a7f45d4f',
    'docs/BERT_CUE_DOSE_RESPONSE_V1.md': 'a4af23e2c3fd678dbd0af1329e305edcf68b11a29a36d362f1652aaf0dec22e0',
    'experiments/bert_cue_dose_response_cases_v1.json': '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a',
    'experiments/bert_cue_dose_response_public_exposure_check_v1.json': '88033d47752b7803bdb13fcac1fa0c820d9e39c7f59548d25878caf04df346cb',
    'scripts/build_bert_cue_dose_manifest.py': 'f93be6eb9c14db0f1ac9d8cc052c5734cf61e446f18caf912fd4c9551c0370dc',
    'looking_glass/bert_cue_dose_response.py': 'fea564457fc403f21745a1762df8301559a754b15771dae27b6f2d780210d247',
    'scripts/bert_cue_dose_response_v1.py': 'f2e1443c0d3e169b67b95e54cb47febd9e107120100c72b2ccfc478afea7e307',
    'docs/BERT_CUE_DOSE_CONTINUATION_V1.md': 'cb17169645a5830c96dbd4988c150f60c7b145182a10c8aeb037c043ee0d5d21',
    'scripts/bert_cue_dose_continuation_v1.py': '38941345de356ed0d25a8b1687ee8675e3c8a20fb918983e490056deb236b265',
    'scripts/compare_bert_cue_dose_continuation_v1.py': '9fa19c1481aab9d234fb146798d142d71241ee26d48c288286a9ea98f57163e8',
}
ORIGINAL_SOURCE = 'b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4'
ORIGINAL_HASHES = {
    'events': 'ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05',
    'started': '19baa22fc6b6fe3da139bf22b125a4f3122a39e6e82dc9ec38604389127bd9e7',
    'failure': '12b861407be2e4050b7f5e7b792847867f7b7c588cf75857783fc36d1bb8317a',
    'receipt': 'f5499d680cbd2ea1d2ebdfdb7268ce30598ad8aa7d6af9ad8ef0ff806ed54f34',
}
ORIGINAL_CPU_ENVIRONMENT = {
    'MKL_CBWR': {'present': False, 'value': None},
    'ATEN_CPU_CAPABILITY': {'present': False, 'value': None},
}
CASE_IDS = ('fresh-02-prior', 'fresh-02-new', 'fresh-03-prior', 'fresh-03-new')
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
    """Reject flushed subnormals or a non-nearest-even binary32 conversion."""
    tiny64 = struct.unpack('<d', bytes.fromhex('0100000000000000'))[0]
    neg64 = struct.unpack('<d', bytes.fromhex('0100000000000080'))[0]
    tiny32 = struct.unpack('<f', bytes.fromhex('01000000'))[0]
    neg32 = struct.unpack('<f', bytes.fromhex('01000080'))[0]
    if (struct.pack('<d', tiny64 * 2.0) != bytes.fromhex('0200000000000000') or
            struct.pack('<d', neg64 * 2.0) != bytes.fromhex('0200000000000080') or
            struct.pack('<d', tiny32) != struct.pack('<d', math.ldexp(1.0, -149)) or
            struct.pack('<d', neg32) != struct.pack('<d', -math.ldexp(1.0, -149)) or
            struct.pack('<f', 1.0 + 2.0 ** -24) != bytes.fromhex('0000803f') or
            struct.pack('<f', 1.0 + 3.0 * 2.0 ** -24) != bytes.fromhex('0200803f')):
        raise RuntimeError('numeric mode must preserve subnormals and round f32 ties to even')


def valid_cpu_environment(environment):
    if type(environment) is not dict or set(environment) != {'MKL_CBWR', 'ATEN_CPU_CAPABILITY'}:
        return False
    for record in environment.values():
        if (type(record) is not dict or set(record) != {'present', 'value'} or
                type(record['present']) is not bool or
                (type(record['value']) is not str if record['present'] else record['value'] is not None)):
            return False
    return True


def strict_json(data):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('invalid JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


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


def specificity(effects, cues):
    negative, affirmative = cues[:2], cues[2:]
    targets = {}
    for target in cues:
        own, opposite, sign = ((negative, affirmative, -1) if target in negative
                               else (affirmative, negative, 1))
        same = next(c for c in own if c != target)
        cross = {c: sign * effects[c, target] for c in opposite}
        minimum, same_delta = min(cross.values()), effects[same, target]
        value = math.fsum((minimum, -abs(same_delta)))
        targets[target] = {'cross_signed': cross, 'cross_min': minimum,
                           'same_donor': same, 'same_delta': same_delta,
                           'same_abs': abs(same_delta), 'margin': value,
                           'passed': value > 0.05}
    return {'targets': targets, 'passed': all(v['passed'] for v in targets.values())}


def dose_decisions(rows):
    """Reconstruct fresh-only screened cells and paired dose transitions."""
    lookup = {(r['triple_id'], r['cue_set']): r for r in rows}
    if len(lookup) != 8:
        raise ValueError('duplicate or absent two-dose cells')
    cells, paired = {}, {}
    for group in GROUPS:
        cells[group] = {}
        for label in DOSES:
            eligible = [lookup[t, group] for t in TRIPLES
                        if lookup[t, group]['baseline_eligible'] and lookup[t, group]['doses'][label]['eligible']]
            passing = sum(row['doses'][label]['specificity_passed'] is True for row in eligible)
            status = 'unevaluable' if len(eligible) < 2 else 'passed' if passing >= 2 else 'failed'
            cells[group][label] = {'total': 3, 'eligible': len(eligible), 'passing': passing, 'status': status}
        values = []
        transitions = {'fail_to_pass': 0, 'pass_to_fail': 0, 'pass_to_pass': 0, 'fail_to_fail': 0}
        for triple in TRIPLES:
            row = lookup[triple, group]
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
        status = ('unevaluable' if n < 2 else 'changed' if transitions['fail_to_pass'] or
                  transitions['pass_to_fail'] else 'unchanged')
        paired[group] = {'eligible': n, 'status': status, 'transitions': transitions, 'values': values,
                         'mean': math.fsum(v['difference'] for v in values) / n if n >= 2 else None}
    return {'cells': cells, 'paired': paired}


def run(manifest, result, stream, width, vocab):
    original_cases = manifest['cases']
    expected_ids = ('exposed-calibration-prior', 'exposed-calibration-new',
                    'fresh-01-prior', 'fresh-01-new', *CASE_IDS)
    if tuple(c['id'] for c in original_cases) != expected_ids:
        raise ValueError('fixed original case identities differ')
    cases = original_cases[4:]
    identities = [(t, g, 'evaluation', 'main')
                  for t in ('fresh-02', 'fresh-03') for g in GROUPS]
    if (len(cases) != 4 or len({c['id'] for c in cases}) != 4 or
            [(c['triple_id'], c['cue_set'], c['split'], c['frame']) for c in cases] != identities or
            any(manifest['cue_sets'][g] != {'negative': list(GROUPS[g][:2]),
                                             'affirmative': list(GROUPS[g][2:])} for g in GROUPS) or
            set(manifest['cue_sets']) != set(GROUPS) or
            result['status'] != 'complete' or len(result['cases']) != 4):
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
        event = strict_json(line)
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
        if (saved['id'], saved['split'], saved['frame'], saved['cue_set'], saved['triple_id']) != (
                ident, case['split'], case['frame'], case['cue_set'], case['triple_id']):
            raise ValueError('result case identity differs')
        CUES = GROUPS[case['cue_set']]
        NEG, AFF = CUES[:2], CUES[2:]
        PAIRS = tuple(itertools.combinations(CUES, 2))
        TRANSFERS = tuple((d, t) for t in CUES for d in CUES if d != t)
        if case['frame'] != 'main' or set(indices) != {'cue', 'mask', 'object'} or set(case['rows']) != set(CUES):
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
        if (type(event['eligible']) is not bool or type(saved['baseline_eligible']) is not bool or
                event['eligible'] != baseline_eligible or event['gaps'] != gaps or
                saved['baseline_eligible'] != baseline_eligible or saved['baseline_gaps'] != gaps or
                saved['clean_margins'] != margins or saved['baselines'] != baselines):
            raise ValueError('saved baseline or screen differs')
        distances = [{'left': a, 'right': b, 'site': site,
                      'distance': norm(captures[a][site], captures[b][site])}
                     for a, b in PAIRS for site in SITES]
        minimum_distance = min(d['distance'] for d in distances)
        eligible = {label: minimum_distance >= radius for label, radius in DOSES.items()}
        event = take('radius', ident)
        if (event['minimum_distance'] != minimum_distance or
                event['distances'] != distances or
                event['binding_edges'] != [d for d in distances if d['distance'] == minimum_distance] or
                event['doses'] != {label: {'radius': radius, 'eligible': eligible[label]}
                                   for label, radius in DOSES.items()} or
                saved['minimum_distance'] != minimum_distance or saved['distances'] != distances or
                set(saved['doses']) != set(DOSES)):
            raise ValueError('fixed-dose or minimum-distance evidence differs')
        for label, radius in DOSES.items():
            dose = saved['doses'][label]
            if (type(dose['eligible']) is not bool or dose['eligible'] != eligible[label] or
                    dose['radius'] != radius or type(event['doses'][label]['eligible']) is not bool):
                raise ValueError('dose eligibility or radius differs')
            if not eligible[label]:
                take('dose_ineligible', ident, dose=label, radius=radius, minimum_distance=minimum_distance)
                if (dose['effects'] or dose['specificity'] or dose['specificity_passed'] is not None or
                        dose['cue_min_margin'] is not None or dose['limiting_targets'] != []):
                    raise ValueError('ineligible dose reports interventions')
        details = {label: {} for label in DOSES if eligible[label]}
        if eligible['low']:
            for site in SITES:
                for cue in CUES:
                    event = take('self', ident, site=site, target=cue, input_ids=ids[cue])
                    before, after = witness(event, positions[site], length)
                    if (before != captures[cue][site] or after != before or
                            raw(event['mask_logits_f32le'], vocab) != clean[cue]):
                        raise ValueError('self control differs')
                for label, radius in DOSES.items():
                    if not eligible[label]:
                        continue
                    effects = {}
                    for donor, target in TRANSFERS:
                        event = take('transfer', ident, site=site, dose=label, donor=donor, target=target,
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
                    if saved['doses'][label]['effects'][site] != [
                            {'donor': d, 'target': t, 'delta': effects[d, t]} for d, t in TRANSFERS]:
                        raise ValueError('stored effects differ')
                    details[label][site] = specificity(effects, CUES)
                    if saved['doses'][label]['specificity'][site] != details[label][site]:
                        raise ValueError('stored target decision differs')
        dose_summary = {}
        for label, radius in DOSES.items():
            data = saved['doses'][label]
            if eligible[label]:
                if set(data['effects']) != set(SITES) or set(data['specificity']) != set(SITES):
                    raise ValueError('dose has missing or unexpected comparator sites')
                cue_min = min(x['margin'] for x in details[label]['cue2']['targets'].values())
                limiting = [c for c in CUES if details[label]['cue2']['targets'][c]['margin'] == cue_min]
                passed = details[label]['cue2']['passed']
                if (type(data['specificity_passed']) is not bool or data['specificity_passed'] != passed or
                        data['cue_min_margin'] != cue_min or data['limiting_targets'] != limiting):
                    raise ValueError('cue specificity or limiting targets differ')
            else:
                cue_min, limiting, passed = None, [], None
            dose_summary[label] = {'radius': radius, 'eligible': eligible[label],
                                   'specificity': details.get(label, {}), 'cue_min_margin': cue_min,
                                   'limiting_targets': limiting, 'specificity_passed': passed}
        for cue in CUES:
            event = take('repeat', ident, cue=cue, input_ids=ids[cue])
            if raw(event['mask_logits_f32le'], vocab) != clean[cue]:
                raise ValueError('repeat full logits differ')
            if set(event['captures_f32le']) != set(SITES) or any(
                    raw(event['captures_f32le'][site], width) != captures[cue][site] for site in SITES):
                raise ValueError('repeat clean captures differ')
        expected = 8 if not eligible['low'] else 24 + 48 * sum(eligible.values())
        if saved['forwards'] != expected:
            raise ValueError('case forward count differs')
        total += expected
        summaries.append({'id': ident, 'split': case['split'], 'frame': case['frame'],
                          'triple_id': case['triple_id'], 'cue_set': case['cue_set'],
                          'forwards': expected, 'minimum_distance': minimum_distance,
                          'baseline_eligible': baseline_eligible,
                          'baseline_gaps': gaps, 'clean_margins': margins, 'doses': dose_summary})
    if stream.readline(1) or result['forwards'] != total or count != total + 2 * 4 + sum(
            not row['doses'][label]['eligible'] for row in summaries for label in DOSES):
        raise ValueError('extra event or total forward count differs')
    return {'status': 'passed', 'forwards': total, 'events': count, 'cases': summaries,
            'scope': 'four continued cells only; original attempt remains incomplete; '
                     'saved logits, controls, dose and landing are not execution proof'}


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
    numeric_mode()
    if a.width < 1 or a.vocab < 5:
        raise ValueError('invalid dimensions')
    manifest_sha, events_sha, result_sha = sha(a.manifest), sha(a.events), sha(a.result)
    manifest, result = strict_json(a.manifest.read_bytes()), strict_json(a.result.read_bytes())
    if not a.synthetic:
        if (not SOURCE_COMMIT or not SOURCE_HASHES or
                a.width != 768 or a.vocab != 30522 or a.events.suffix != '.gz' or
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
        started = strict_json(a.started.read_bytes())
        if (not valid_cpu_environment(started.get('cpu_environment')) or
                started.get('cpu_environment') != ORIGINAL_CPU_ENVIRONMENT):
            raise ValueError('recorded CPU environment differs from the original attempt')
        if any(started.get(k) != result.get(k) for k in
               ('source_commit', 'source_sha256', 'original_source_commit', 'original_sha256',
                'assets', 'versions', 'seed', 'cpu_environment')):
            raise ValueError('started/result metadata differs')
        if (started.get('original_source_commit') != ORIGINAL_SOURCE or
                started.get('original_sha256') != ORIGINAL_HASHES or
                started.get('case_ids') != list(CASE_IDS) or
                result.get('analysis_scope') != 'four continued cells only; original attempt remains incomplete' or
                started.get('maximum_forwards') != 480 or result.get('seed') != 2026102905 or
                result.get('tokenizer') != {'rows': 32, 'any_id': 2151, 'some_id': 2070} or
                result.get('state_sha256_before') != result.get('state_sha256_after') or
                not re.fullmatch('[0-9a-f]{64}', result.get('state_sha256_before', '')) or
                result.get('cpu_threads') != 1 or result.get('cpu_interop_threads') != 1 or
                result.get('attention') != 'eager' or result.get('cuda_initialized') is not False):
            raise ValueError('saved runtime/state metadata differs')
    opener = gzip.open if a.events.suffix == '.gz' else open
    with opener(a.events, 'rb') as stream:
        report = run(manifest, result, stream, a.width, a.vocab)
    numeric_mode()
    if (sha(a.manifest), sha(a.events), sha(a.result)) != (manifest_sha, events_sha, result_sha):
        raise ValueError('input bytes changed during replay')
    if not a.synthetic and sha(a.started) != a.expected_started_sha256:
        raise ValueError('started record changed during replay')
    report.update({'synthetic': a.synthetic, 'source_commit': result.get('source_commit'),
                   'manifest_sha256': manifest_sha, 'events_sha256': events_sha,
                   'result_sha256': result_sha,
                   'identity_scope': ('Recorded source/asset/version pins and start/result consistency; '
                                      'not an attestation that these bytes executed or were loaded.')})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
