"""Independent saved-logit audit for the post-hoc BERT negation study.

Read only public cases and already recorded logits; never load a model. This
recomputes vocabulary probabilities, conditional changes and matched-control
cosines independently, and compares selected bytes to a proposed public subset.
It does not attest execution or re-audit the full activation/landing witnesses.
"""
import argparse
import base64
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import struct

CUES = ('not', 'never', 'just', 'also')
SITES = ('cue2', 'object2', 'mask2', 'cue5')
TRANSFERS = (('not', 'just'), ('just', 'not'), ('never', 'just'), ('just', 'never'),
             ('also', 'just'), ('just', 'also'), ('also', 'not'), ('also', 'never'))
PIN_JOURNAL = 'a67b5ea0ab7d4f7078bab57ad315f9bcf90302be674e8723c05c0785a011ccd1'
PIN_MANIFEST = '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff'
PIN_RESULT = '20bcd1777e343a268e73042e664bfeb0603a01791ae0b16d908f5d5ea6d08f9b'
PIN_VOCAB = '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'


def check(condition, text):
    if not condition:
        raise ValueError(text)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def denormal_guard():
    raw = bytes.fromhex('0100000000000000')
    value = struct.unpack('<d', raw)[0]
    negative_raw = bytes.fromhex('0100000000000080')
    negative = struct.unpack('<d', negative_raw)[0]
    check(value > 0 and struct.pack('<d', value * 1.) == raw
          and negative < 0 and struct.pack('<d', negative * 1.) == negative_raw
          and struct.pack('<d', float('5e-324')) == raw
          and struct.pack('<d', float('-5e-324')) == negative_raw,
          'binary64 subnormals are not preserved')
    tiny_f32 = struct.unpack('<f', bytes.fromhex('01000000'))[0]
    neg_f32 = struct.unpack('<f', bytes.fromhex('01000080'))[0]
    check(tiny_f32 == 2. ** -149 and neg_f32 == -(2. ** -149),
          'binary32 decoding subnormals are not preserved')


def words(raw, width):
    check(type(raw) is bytes and len(raw) == 4 * width, 'saved f32 vector length')
    values = struct.unpack('<' + str(width) + 'f', raw)
    check(all(math.isfinite(x) for x in values), 'nonfinite saved f32')
    return values


def probability(logits, excluded=()):
    excluded = frozenset(excluded)
    peak = max(x for i, x in enumerate(logits) if i not in excluded)
    weights = [0. if i in excluded else math.exp(x - peak) for i, x in enumerate(logits)]
    divisor = math.fsum(weights)
    check(divisor > 0 and math.isfinite(divisor), 'invalid softmax')
    return [value / divisor for value in weights]


def vector_cosine(a, b):
    check(len(a) == len(b), 'cosine width')
    aa, bb = math.hypot(*a), math.hypot(*b)
    if aa == 0. or bb == 0.:
        return None
    value = math.fsum((x / aa) * (y / bb) for x, y in zip(a, b))
    return max(-1., min(1., value))


def rank(values, token):
    return 1 + sum(x > values[token] or (x == values[token] and i < token) for i, x in enumerate(values))


def state(logits, probabilities, vocabulary):
    ordered = heapq.nsmallest(5, range(len(logits)), key=lambda i: (-logits[i], i))
    return {'top': [{'id': i, 'token': vocabulary[i], 'logit': logits[i], 'probability': probabilities[i]} for i in ordered],
            'any_rank': rank(logits, 2151), 'some_rank': rank(logits, 2070),
            'any_probability': probabilities[2151], 'some_probability': probabilities[2070]}


def measure(clean, patched, vocabulary):
    p, q = probability(clean), probability(patched)
    pc, qc = probability(clean, (2151, 2070)), probability(patched, (2151, 2070))
    delta = [b - a for a, b in zip(p, q)]
    dc = [b - a for a, b in zip(pc, qc)]
    all_absolute = math.fsum(abs(x) for x in delta)
    pair_absolute = abs(delta[2151]) + abs(delta[2070])
    other_absolute = math.fsum(abs(x) for i, x in enumerate(delta) if i not in (2151, 2070))
    movers = heapq.nsmallest(5, (i for i, value in enumerate(delta)
                                 if i not in (2151, 2070) and value != 0),
                            key=lambda i: (-abs(delta[i]), i))
    data = {'clean': state(clean, p, vocabulary), 'patched': state(patched, q, vocabulary),
            'total_variation': all_absolute / 2,
            'other_conditional_total_variation': math.fsum(abs(x) for x in dc) / 2,
            'any_some_absolute_movement_share': pair_absolute / all_absolute if all_absolute else None,
            'other_absolute_movement_share': other_absolute / all_absolute if all_absolute else None,
            'any_some_margin_change': math.fsum((patched[2151], -patched[2070], -clean[2151], clean[2070])),
            'any_probability_change': delta[2151], 'some_probability_change': delta[2070],
            'top_other_probability_movers': [{'id': i, 'token': vocabulary[i], 'change': delta[i]} for i in movers]}
    return data, delta, dc


def same(actual, expected, label):
    if type(actual) is dict and type(expected) is dict:
        check(set(actual) == set(expected), label + ' keys')
        for k in actual:
            same(actual[k], expected[k], label + '.' + k)
    elif type(actual) is list and type(expected) is list:
        check(len(actual) == len(expected), label + ' length')
        for a, b in zip(actual, expected):
            same(a, b, label)
    elif type(actual) in (int, float) and type(expected) in (int, float):
        check(math.isfinite(actual) and math.isfinite(expected), label + ' finite')
        if actual == 0 or expected == 0:
            check(actual == expected, label + ' zero')
        else:
            check(math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-15), label + ' numeric')
    else:
        check(type(actual) is type(expected) and actual == expected, label + ' value/type')


def describe(values):
    valid = [v for v in values if v is not None]
    return {'count': len(valid), 'median': statistics.median(valid) if valid else None,
            'minimum': min(valid) if valid else None, 'maximum': max(valid) if valid else None}


def aggregate(cases):
    output = {}
    for split in ('development', 'evaluation'):
        selected = [case for case in cases if case['split'] == split]
        output[split] = {}
        for site in SITES:
            output[split][site] = {}
            for direction in ('negative_to_just', 'just_to_negative'):
                items = []
                for case in selected:
                    data = case['sites'][site]
                    for pair in data['paired']:
                        if pair['direction'] == direction:
                            a, b = pair['primary'], pair['also_control']
                            items.append((pair, data['actions'][a['donor'] + '->' + a['target']],
                                          data['actions'][b['donor'] + '->' + b['target']]))
                result = {'contexts': len(selected), 'comparisons': len(items)}
                for metric in ('total_variation', 'other_conditional_total_variation',
                               'any_some_absolute_movement_share', 'any_some_margin_change'):
                    result[metric] = {'primary': describe([x[metric] for _, x, _ in items]),
                                      'also_control': describe([x[metric] for _, _, x in items]),
                                      'primary_greater_than_control': sum(a[metric] > b[metric]
                                          for _, a, b in items if a[metric] is not None and b[metric] is not None)}
                result['any_some_margin_change']['primary_positive_and_exceeds_absolute_control'] = sum(
                    a['any_some_margin_change'] > 0 and a['any_some_margin_change'] > abs(b['any_some_margin_change'])
                    for _, a, b in items)
                result['any_some_margin_change']['control_absolute_at_least_primary'] = sum(
                    abs(b['any_some_margin_change']) >= abs(a['any_some_margin_change']) for _, a, b in items)
                for metric in ('full_probability_displacement_cosine', 'other_conditional_displacement_cosine'):
                    result[metric] = describe([entry[metric] for entry, _, _ in items])
                for name, index in (('primary', 1), ('also_control', 2)):
                    result[name + '_top1_changes'] = sum(item[index]['clean']['top'][0]['id'] !=
                                                          item[index]['patched']['top'][0]['id'] for item in items)
                output[split][site][direction] = result
    return output


def inspect_case(records, case, original, vocabulary):
    width = len(vocabulary)
    clean = {cue: words(records['clean', cue], width) for cue in CUES}
    sites = {}
    for site in SITES:
        actions, delta, conditional = {}, {}, {}
        for donor, target in TRANSFERS:
            key = donor + '->' + target
            patched = words(records['transfer', site, donor, target], width)
            actions[key], delta[key], conditional[key] = measure(clean[target], patched, vocabulary)
        pairs = []
        for cue in ('not', 'never'):
            for direction, donor, target in (('negative_to_just', cue, 'just'), ('just_to_negative', 'just', cue)):
                first, control = donor + '->' + target, 'also->' + target
                pairs.append({'negative_cue': cue, 'direction': direction,
                    'primary': {'donor': donor, 'target': target}, 'also_control': {'donor': 'also', 'target': target},
                    'full_probability_displacement_cosine': vector_cosine(delta[first], delta[control]),
                    'other_conditional_displacement_cosine': vector_cosine(conditional[first], conditional[control])})
        sites[site] = {'actions': actions, 'paired': pairs}
    return {'case_id': case['id'], 'split': case['split'], 'radius': original['radius'], 'sites': sites}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    denormal_guard()
    for path, pin in ((args.events, PIN_JOURNAL), (args.manifest, PIN_MANIFEST),
                      (args.result, PIN_RESULT), (args.vocab, PIN_VOCAB)):
        check(sha_file(path) == pin, 'pinned input differs: ' + path.name)
    manifest, original = json.loads(args.manifest.read_bytes()), json.loads(args.result.read_bytes())
    report = json.loads(args.report.read_bytes())
    vocabulary = args.vocab.read_text(encoding='utf-8').splitlines()
    check(len(vocabulary) == 30522 and len(set(vocabulary)) == 30522, 'vocabulary differs')
    check(len(manifest['cases']) == len(original['cases']) == len(report['cases']) == 8,
          'case inventory differs')
    for key, expected in (('events_sha256', PIN_JOURNAL), ('manifest_sha256', PIN_MANIFEST),
                          ('original_result_sha256', PIN_RESULT), ('vocab_sha256', PIN_VOCAB)):
        check(report['inputs'][key] == expected, 'report input pin differs: ' + key)
    check(sha_file(args.subset) == report['subset']['sha256'], 'subset hash differs')
    with gzip.open(args.subset, 'rb') as stream:
        subset = stream.read(288 * 4 * len(vocabulary) + 1)
    check(len(subset) == report['subset']['uncompressed_bytes'], 'subset length differs')
    cases, event_count, matching, subset_offset, physical = [], 0, 0, 0, 0
    with gzip.open(args.events, 'rb') as journal:
        for index, (case, original_case) in enumerate(zip(manifest['cases'], original['cases'])):
            expected = []
            expected.extend([('clean', cue) for cue in CUES])
            expected.append(('radius',))
            for site in SITES:
                expected.extend(('self', site, cue) for cue in CUES)
                expected.extend(('transfer', site, donor, target) for donor, target in TRANSFERS)
            expected.extend(('repeat', cue) for cue in CUES)
            records, clean, event_indices = {}, {}, {}
            for item in expected:
                line = journal.readline(2 * 1024 * 1024 + 1)
                check(line.endswith(b'\n') and len(line) <= 2 * 1024 * 1024, 'missing or oversized journal record')
                event = json.loads(line)
                check(event.get('kind') == item[0] and event.get('case_id') == case['id'], 'case or kind differs')
                if item[0] == 'radius':
                    check(event['radius'] == original_case['radius'], 'radius differs')
                    event_count += 1
                    continue
                if item[0] == 'clean' or item[0] == 'repeat':
                    cue = item[1]
                    check(event.get('cue') == cue, 'cue differs')
                elif item[0] == 'self':
                    site, cue = item[1:]
                    check(event.get('site') == site and event.get('target') == cue, 'self identity differs')
                else:
                    site, donor, cue = item[1:]
                    check(event.get('site') == site and event.get('donor') == donor
                          and event.get('target') == cue, 'transfer identity differs')
                ids = list(case['just_ids'])
                ids[4] = manifest['cues'][cue]
                check(event.get('input_ids') == ids, 'saved model input differs')
                raw = base64.b64decode(event['mask_logits_f32le'], validate=True)
                check(len(raw) == 4 * len(vocabulary), 'saved full-logit length differs')
                if item[0] == 'clean':
                    records[item], clean[cue] = raw, raw
                    event_indices[item] = event_count
                elif item[0] == 'transfer':
                    if item[1] == 'cue5':
                        check(raw == clean[cue], 'final cue control differs')
                    records[item] = raw
                    event_indices[item] = event_count
                else:
                    check(raw == clean[cue], 'self or repeat full logits differ')
                event_count += 1
            reported_index = report['subset']['index'][index]
            check(reported_index['case_id'] == case['id'] and len(reported_index['entries']) == 36, 'index case differs')
            seen = {}
            for entry, expected_key in zip(reported_index['entries'], records):
                key = tuple(entry['key'])
                check(key == expected_key and entry['source_event_index'] == event_indices[key], 'subset source index differs')
                off, raw = entry['offset'], records[key]
                check(type(off) is int and off >= 0 and subset[off:off + len(raw)] == raw
                      and hashlib.sha256(raw).hexdigest() == entry['sha256'], 'subset bytes differ from source')
                if off == subset_offset:
                    seen[off] = raw
                    subset_offset += len(raw)
                    physical += 1
                else:
                    check(off in seen and seen[off] == raw, 'subset offset reuse differs')
                matching += 1
            computed = inspect_case(records, case, original_case, vocabulary)
            same(computed, report['cases'][index], 'case ' + case['id'])
            cases.append(computed)
        check(not journal.read(1) and event_count == 456, 'journal tail or count differs')
    check(subset_offset == len(subset), 'unused subset bytes')
    same(aggregate(cases), report['summary'], 'summary')
    denormal_guard()
    out = {'status': 'pass', 'cases': len(cases), 'journal_events': event_count,
           'selected_logical_vectors_compared': matching, 'subset_sha256': report['subset']['sha256'],
           'physical_unique_vectors': physical,
           'report_sha256': sha_file(args.report), 'journal_sha256': PIN_JOURNAL,
           'scope': 'Independent saved-logit comparisons and original-to-subset bytes, not execution or activation-witness attestation.'}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(out, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(out, sort_keys=True))


if __name__ == '__main__':
    main()
