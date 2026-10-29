"""Post-hoc full-vocabulary replay of a fixed saved negation-cue experiment."""
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
import sys
import tracemalloc

CUES = ('not', 'never', 'just', 'also')
SITES = ('cue2', 'object2', 'mask2', 'cue5')
TRANSFERS = (('not', 'just'), ('just', 'not'), ('never', 'just'), ('just', 'never'),
             ('also', 'just'), ('just', 'also'), ('also', 'not'), ('also', 'never'))
EVENTS_SHA256 = 'a67b5ea0ab7d4f7078bab57ad315f9bcf90302be674e8723c05c0785a011ccd1'
EVENTS_BYTES = 85368356
MANIFEST_SHA256 = '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff'
RESULT_SHA256 = '20bcd1777e343a268e73042e664bfeb0603a01791ae0b16d908f5d5ea6d08f9b'
VOCAB_SHA256 = '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'
SOURCE_COMMIT = 'c72d120e4c22a56a9eda051893430631fb963c70'
VOCAB_SIZE = 30522
MAX_LINE_BYTES = 2 * 1024 * 1024


def _require_binary64_subnormals():
    """Fail closed if the calling thread flushes binary64 subnormal values."""
    normal = math.ldexp(1.0, -1022)
    positive, negative = normal / 2.0, -normal / 2.0
    tiny = float.fromhex('0x0.0000000000001p-1022')
    if (positive == 0.0 or negative == 0.0 or tiny == 0.0 or
            positive * 2.0 != normal or negative * 2.0 != -normal or tiny * 1.0 == 0.0):
        raise ValueError('binary64 subnormal arithmetic is flushed')


def decode_f32(raw, count):
    """Decode finite little-endian binary32 without a binary32 conversion step."""
    if len(raw) != 4 * count:
        raise ValueError('mask-logit byte count differs')
    result = []
    for (bits,) in struct.iter_unpack('<I', raw):
        exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
        if exponent == 255:
            raise ValueError('nonfinite saved logit')
        value = (math.ldexp(fraction, -149) if exponent == 0 else
                 math.ldexp((1 << 23) | fraction, exponent - 150))
        result.append(-value if bits >> 31 else value)
    return tuple(result)


def _softmax(logits, excluded=()):
    excluded = set(excluded)
    selected = [value for i, value in enumerate(logits) if i not in excluded]
    if not selected:
        raise ValueError('no vocabulary remains')
    maximum = max(selected)
    weights = [0.0 if i in excluded else math.exp(value - maximum)
               for i, value in enumerate(logits)]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _rank(logits, token_id):
    value = logits[token_id]
    return 1 + sum(x > value or (x == value and i < token_id) for i, x in enumerate(logits))


def _top(logits, probabilities, vocabulary, n=5):
    ids = heapq.nlargest(min(n, len(logits)), range(len(logits)), key=lambda i: (logits[i], -i))
    return [{'id': i, 'token': vocabulary[i], 'logit': logits[i], 'probability': probabilities[i]} for i in ids]


def compare_logits(clean, patched, *, any_id, some_id, vocabulary):
    """Return scalar comparisons and the two patched probability vectors."""
    _require_binary64_subnormals()
    if (not isinstance(clean, (tuple, list)) or not isinstance(patched, (tuple, list)) or
            len(clean) != len(patched) or len(clean) != len(vocabulary) or len(clean) < 3):
        raise ValueError('matching nonempty full-vocabulary vectors required')
    if (type(any_id) is not int or type(some_id) is not int or any_id == some_id or
            not 0 <= any_id < len(clean) or not 0 <= some_id < len(clean)):
        raise ValueError('invalid comparison token IDs')
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for vector in (clean, patched) for value in vector):
        raise ValueError('finite logits required')
    p, q = _softmax(clean), _softmax(patched)
    p_other, q_other = _softmax(clean, (any_id, some_id)), _softmax(patched, (any_id, some_id))
    change = [y - x for x, y in zip(p, q)]
    other_change = [y - x for x, y in zip(p_other, q_other)]
    absolute = math.fsum(abs(x) for x in change)
    group_absolute = abs(change[any_id]) + abs(change[some_id])
    other_absolute = math.fsum(abs(x) for i, x in enumerate(change) if i not in (any_id, some_id))
    movers = heapq.nlargest(min(5, len(clean) - 2),
                           (i for i in range(len(clean)) if i not in (any_id, some_id) and change[i] != 0),
                           key=lambda i: (abs(change[i]), -i))
    def state(logits, probs):
        return {'top': _top(logits, probs, vocabulary),
                'any_rank': _rank(logits, any_id), 'some_rank': _rank(logits, some_id),
                'any_probability': probs[any_id], 'some_probability': probs[some_id]}
    margin_change = math.fsum((patched[any_id], -patched[some_id], -clean[any_id], clean[some_id]))
    result = {
        'clean': state(clean, p), 'patched': state(patched, q),
        'total_variation': absolute / 2.0,
        'other_conditional_total_variation': math.fsum(abs(x) for x in other_change) / 2.0,
        'any_some_absolute_movement_share': group_absolute / absolute if absolute else None,
        'other_absolute_movement_share': other_absolute / absolute if absolute else None,
        'any_some_margin_change': margin_change,
        'any_probability_change': change[any_id], 'some_probability_change': change[some_id],
        'top_other_probability_movers': [{'id': i, 'token': vocabulary[i], 'change': change[i]}
                                         for i in movers],
    }
    return result, q, q_other


def displacement_cosine(clean, first, second):
    _require_binary64_subnormals()
    if not len(clean) or len(clean) != len(first) or len(clean) != len(second):
        raise ValueError('matching probability vectors required')
    if any(type(x) not in (int, float) or not math.isfinite(x)
           for vector in (clean, first, second) for x in vector):
        raise ValueError('finite probability vectors required')
    a, b = [x - p for p, x in zip(clean, first)], [x - p for p, x in zip(clean, second)]
    if any(not math.isfinite(x) for x in a + b):
        raise ValueError('displacement outside numeric range')
    sa, sb = max(abs(x) for x in a), max(abs(x) for x in b)
    if not sa or not sb:
        return None
    a, b = [x / sa for x in a], [x / sb for x in b]
    norm = math.sqrt(math.fsum(x*x for x in a)) * math.sqrt(math.fsum(x*x for x in b))
    return max(-1.0, min(1.0, math.fsum(x*y for x, y in zip(a, b)) / norm))


def _event_logits(event, vocab_size):
    try:
        raw = base64.b64decode(event['mask_logits_f32le'], validate=True)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('invalid saved mask logits') from exc
    if len(raw) != 4 * vocab_size:
        raise ValueError('saved mask-logit length differs')
    return raw


def scan_case_records(lines, case, saved, cues, any_id, some_id, vocab_size):
    """Check one complete saved schedule and return the 36 relevant vectors."""
    case_id, records, clean, controls = case['id'], {}, {}, {'self': 0, 'repeat': 0, 'final_cue': 0}
    if (saved.get('case_id') != case_id or saved.get('split') != case['split'] or
            saved.get('forwards') != 56):
        raise ValueError('saved case identity differs')
    def row(cue):
        ids = list(case['just_ids'])
        ids[4] = cues[cue]
        return ids
    def take(kind, **identity):
        try:
            event = json.loads(next(lines))
        except (StopIteration, ValueError, TypeError) as exc:
            raise ValueError('missing or malformed saved event') from exc
        if (event.get('kind') != kind or event.get('case_id') != case_id or
                any(event.get(key) != value for key, value in identity.items())):
            raise ValueError('saved event order or identity differs')
        return event
    for cue in CUES:
        event = take('clean', cue=cue, input_ids=row(cue))
        raw = _event_logits(event, vocab_size)
        logits = decode_f32(raw, vocab_size)
        margin = math.fsum((logits[any_id], -logits[some_id]))
        if not math.isclose(margin, saved['clean_margins'][cue], rel_tol=0, abs_tol=1e-10):
            raise ValueError('clean saved margin differs')
        clean[cue], records[('clean', cue)] = raw, raw
    radius = take('radius')['radius']
    if (type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0 or
            radius != saved['radius']):
        raise ValueError('saved requested radius differs')
    for site in SITES:
        for cue in CUES:
            event = take('self', site=site, target=cue, input_ids=row(cue))
            if _event_logits(event, vocab_size) != clean[cue]:
                raise ValueError('saved self control differs')
            controls['self'] += 1
        expected = saved['effects'][site]
        if len(expected) != len(TRANSFERS):
            raise ValueError('saved transfer inventory differs')
        for index, (donor, target) in enumerate(TRANSFERS):
            event = take('transfer', site=site, donor=donor, target=target, input_ids=row(target))
            raw = _event_logits(event, vocab_size)
            logits, baseline = decode_f32(raw, vocab_size), decode_f32(clean[target], vocab_size)
            effect = math.fsum((logits[any_id], -logits[some_id], -baseline[any_id], baseline[some_id]))
            entry = expected[index]
            if (entry.get('donor') != donor or entry.get('target') != target or
                    not math.isclose(effect, entry.get('delta', float('nan')), rel_tol=0, abs_tol=1e-10)):
                raise ValueError('saved transfer margin differs')
            actual, error = event.get('actual_norm'), event.get('relative_error')
            if (event.get('requested_radius') != radius or type(actual) not in (int, float) or
                    not math.isfinite(actual) or actual <= 0 or type(error) not in (int, float) or
                    not math.isfinite(error) or abs(actual-radius)/radius > .001 or
                    not math.isclose(error, abs(actual-radius)/radius, rel_tol=1e-7, abs_tol=1e-12)):
                raise ValueError('recorded dose fields differ')
            if site == 'cue5':
                if raw != clean[target]:
                    raise ValueError('final-cue full-logit control differs')
                controls['final_cue'] += 1
            records[('transfer', site, donor, target)] = raw
    for cue in CUES:
        event = take('repeat', cue=cue, input_ids=row(cue))
        if _event_logits(event, vocab_size) != clean[cue]:
            raise ValueError('saved repeat differs')
        controls['repeat'] += 1
    return records, controls


def _selected_keys(case_index):
    base = case_index * 57
    result = [(('clean', cue), base + index) for index, cue in enumerate(CUES)]
    for site_index, site in enumerate(SITES):
        result.extend((('transfer', site, donor, target), base + 9 + 12 * site_index + index)
                      for index, (donor, target) in enumerate(TRANSFERS))
    return result


def write_subset_case(stream, records, case_index, offset, vocab_size):
    """Write one case's unique selected vectors and record their source positions."""
    expected = _selected_keys(case_index)
    if set(records) != {key for key, _ in expected}:
        raise ValueError('selected vector inventory differs')
    seen, entries = {}, []
    for key, source_index in expected:
        raw = records[key]
        if type(raw) is not bytes or len(raw) != 4 * vocab_size:
            raise ValueError('selected vector length differs')
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            saved_offset, saved_raw = seen[sha]
            if saved_raw != raw:
                raise ValueError('distinct vectors have the same digest')
        else:
            saved_offset = offset
            if stream.write(raw) != len(raw):
                raise ValueError('short subset write')
            seen[sha] = (offset, raw)
            offset += len(raw)
        entries.append({'key': list(key), 'source_event_index': source_index,
                        'offset': saved_offset, 'sha256': sha})
    return entries, offset


def read_subset_case(stream, entries, case_index, offset, vocab_size):
    """Reconstruct one case from its ordered, case-local deduplicated vectors."""
    expected = _selected_keys(case_index)
    if type(entries) is not list or len(entries) != len(expected):
        raise ValueError('selected subset inventory differs')
    by_offset, records = {}, {}
    for entry, (key, source_index) in zip(entries, expected):
        if (type(entry) is not dict or entry.get('key') != list(key) or
                entry.get('source_event_index') != source_index or
                type(entry.get('offset')) is not int or type(entry.get('sha256')) is not str):
            raise ValueError('selected subset identity differs')
        position = entry['offset']
        if position == offset:
            raw = stream.read(4 * vocab_size)
            if len(raw) != 4 * vocab_size:
                raise ValueError('selected subset is truncated')
            by_offset[position] = raw
            offset += len(raw)
        elif position in by_offset:
            raw = by_offset[position]
        else:
            raise ValueError('selected subset offset differs')
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('selected subset vector digest differs')
        records[key] = raw
    return records, offset


def analyze_case(records, case, saved, any_id, some_id, vocabulary):
    """Describe every transfer and paired same-recipient active comparison."""
    if set(records) != {key for key, _ in _selected_keys(0)}:
        raise ValueError('selected case vectors differ')
    size = len(vocabulary)
    clean = {cue: decode_f32(records[('clean', cue)], size) for cue in CUES}
    for cue in CUES:
        margin = math.fsum((clean[cue][any_id], -clean[cue][some_id]))
        if not math.isclose(margin, saved['clean_margins'][cue], rel_tol=0, abs_tol=1e-10):
            raise ValueError('selected clean margin differs from original saved result')
    clean_probability = {cue: _softmax(clean[cue]) for cue in CUES}
    clean_other = {cue: _softmax(clean[cue], (any_id, some_id)) for cue in CUES}
    sites = {}
    for site in SITES:
        actions, probabilities, other_probabilities = {}, {}, {}
        for index, (donor, target) in enumerate(TRANSFERS):
            logits = decode_f32(records[('transfer', site, donor, target)], size)
            metric, probability, other = compare_logits(clean[target], logits, any_id=any_id,
                                                         some_id=some_id, vocabulary=vocabulary)
            if not math.isclose(metric['any_some_margin_change'], saved['effects'][site][index]['delta'],
                                rel_tol=0, abs_tol=1e-10):
                raise ValueError('derived margin differs from original saved result')
            key = donor + '->' + target
            actions[key], probabilities[key], other_probabilities[key] = metric, probability, other
        pairs = []
        for cue in ('not', 'never'):
            for direction, donor, target in (('negative_to_just', cue, 'just'),
                                             ('just_to_negative', 'just', cue)):
                primary_key, also_key = donor + '->' + target, 'also->' + target
                pairs.append({
                    'negative_cue': cue, 'direction': direction,
                    'primary': {'donor': donor, 'target': target},
                    'also_control': {'donor': 'also', 'target': target},
                    'full_probability_displacement_cosine': displacement_cosine(
                        clean_probability[target], probabilities[primary_key], probabilities[also_key]),
                    'other_conditional_displacement_cosine': displacement_cosine(
                        clean_other[target], other_probabilities[primary_key], other_probabilities[also_key]),
                })
        sites[site] = {'actions': actions, 'paired': pairs}
    return {'case_id': case['id'], 'split': case['split'], 'radius': saved['radius'], 'sites': sites}


def _describe(values):
    values = [x for x in values if x is not None]
    return {'count': len(values), 'median': statistics.median(values) if values else None,
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None}


def summarize_cases(cases):
    """Summarize fixed comparisons; the repeated also->just arm is not independent."""
    output = {}
    for split in ('development', 'evaluation'):
        selected = [case for case in cases if case['split'] == split]
        output[split] = {}
        for site in SITES:
            output[split][site] = {}
            for direction in ('negative_to_just', 'just_to_negative'):
                pairs = []
                for case in selected:
                    block = case['sites'][site]
                    for pair in block['paired']:
                        if pair['direction'] != direction:
                            continue
                        a, b = pair['primary'], pair['also_control']
                        primary = block['actions'][a['donor'] + '->' + a['target']]
                        control = block['actions'][b['donor'] + '->' + b['target']]
                        pairs.append((pair, primary, control))
                summary = {'contexts': len(selected), 'comparisons': len(pairs)}
                for metric in ('total_variation', 'other_conditional_total_variation',
                               'any_some_absolute_movement_share', 'any_some_margin_change'):
                    summary[metric] = {
                        'primary': _describe([a[metric] for _, a, _ in pairs]),
                        'also_control': _describe([b[metric] for _, _, b in pairs]),
                        'primary_greater_than_control': sum(
                            a[metric] > b[metric] for _, a, b in pairs
                            if a[metric] is not None and b[metric] is not None),
                    }
                margin = summary['any_some_margin_change']
                margin['primary_positive_and_exceeds_absolute_control'] = sum(
                    a['any_some_margin_change'] > 0 and
                    a['any_some_margin_change'] > abs(b['any_some_margin_change']) for _, a, b in pairs)
                margin['control_absolute_at_least_primary'] = sum(
                    abs(b['any_some_margin_change']) >= abs(a['any_some_margin_change']) for _, a, b in pairs)
                for metric in ('full_probability_displacement_cosine', 'other_conditional_displacement_cosine'):
                    summary[metric] = _describe([pair[metric] for pair, _, _ in pairs])
                for name, index in (('primary', 1), ('also_control', 2)):
                    summary[name + '_top1_changes'] = sum(
                        pair[index]['clean']['top'][0]['id'] != pair[index]['patched']['top'][0]['id']
                        for pair in pairs)
                output[split][site][direction] = summary
    return output


def _read_pinned(path, expected, limit):
    with open(path, 'rb') as source:
        data = source.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('pinned input bytes differ')
    return data


def _read_inputs(manifest_path, result_path, vocabulary_path):
    manifest = json.loads(_read_pinned(manifest_path, MANIFEST_SHA256, 128 * 1024))
    result = json.loads(_read_pinned(result_path, RESULT_SHA256, 256 * 1024))
    vocab = _read_pinned(vocabulary_path, VOCAB_SHA256, 512 * 1024).decode('utf-8').splitlines()
    cases = manifest['cases']
    if (len(vocab) != VOCAB_SIZE or len(set(vocab)) != VOCAB_SIZE or
            len(cases) != 8 or [c['id'] for c in cases] != [f'negation-{i:02d}' for i in range(1, 9)] or
            [c['split'] for c in cases] != ['development'] * 2 + ['evaluation'] * 6 or
            manifest['scored_tokens'] != {'any': 2151, 'some': 2070} or
            manifest['cues'] != {'not': 2025, 'never': 2196, 'just': 2074, 'also': 2036} or
            result['status'] != 'complete' or result['forwards'] != 448 or
            result['source_commit'] != SOURCE_COMMIT or result['events_sha256'] != EVENTS_SHA256 or
            result['source_sha256']['experiments/bert_negation_public_cases_v1.json'] != MANIFEST_SHA256 or
            result['assets']['vocab.txt']['sha256'] != VOCAB_SHA256 or len(result['cases']) != 8):
        raise ValueError('fixed input schema or identity differs')
    return manifest, result, tuple(vocab)


def _hash_stream(stream, *, expected_bytes=None):
    stream.seek(0)
    digest, size = hashlib.sha256(), 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if expected_bytes is not None and size > expected_bytes:
            raise ValueError('file exceeds pinned size')
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest(), size


def _bounded_lines(stream):
    for _ in range(456):
        line = stream.readline(MAX_LINE_BYTES + 1)
        if not line or len(line) > MAX_LINE_BYTES or not line.endswith(b'\n'):
            raise ValueError('missing, incomplete or oversized journal line')
        yield line
    if stream.read(1):
        raise ValueError('extra journal records or bytes')


def _check_prediction(case_results, saved_result):
    scores = []
    for row, saved in zip(case_results, saved_result['cases']):
        site = row['sites']['cue2']['actions']
        score = math.fsum((
            math.fsum((site[cue + '->just']['any_some_margin_change'],
                       -site['just->' + cue]['any_some_margin_change'])) / 2
            for cue in ('not', 'never'))) / 2
        if not math.isclose(score, saved['directional_score'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('saved case directional score differs')
        if row['split'] == 'evaluation':
            scores.append(score)
    mean, positives = math.fsum(scores) / 6, sum(score > 0 for score in scores)
    prediction = saved_result['directional_prediction']
    if (not math.isclose(mean, prediction['mean'], rel_tol=0, abs_tol=1e-10) or
            positives != prediction['positive_cases'] or
            (mean > .02 and positives >= 4) is not prediction['passed']):
        raise ValueError('saved original aggregate decision differs')


def analyze_journal(events_path, manifest_path, result_path, vocabulary_path, subset_path):
    """Verify the fixed journal, compute comparisons and save exact selected f32 vectors."""
    _require_binary64_subnormals()
    manifest, original, vocabulary = _read_inputs(manifest_path, result_path, vocabulary_path)
    case_results, index_cases, offset = [], [], 0
    controls = {'self': 0, 'repeat': 0, 'final_cue': 0}
    with open(events_path, 'rb') as raw_events:
        if _hash_stream(raw_events, expected_bytes=EVENTS_BYTES) != (EVENTS_SHA256, EVENTS_BYTES):
            raise ValueError('journal hash or length differs')
        with open(subset_path, 'xb') as destination:
            with gzip.GzipFile(filename='', fileobj=destination, mode='wb', mtime=0, compresslevel=6) as subset:
                with gzip.GzipFile(fileobj=raw_events, mode='rb') as events:
                    lines = _bounded_lines(events)
                    for case_index, (case, saved) in enumerate(zip(manifest['cases'], original['cases'])):
                        records, checked = scan_case_records(lines, case, saved, manifest['cues'],
                                                             2151, 2070, VOCAB_SIZE)
                        for key in controls:
                            controls[key] += checked[key]
                        entries, offset = write_subset_case(subset, records, case_index, offset, VOCAB_SIZE)
                        index_cases.append({'case_id': case['id'], 'entries': entries})
                        case_results.append(analyze_case(records, case, saved, 2151, 2070, vocabulary))
                    if next(lines, None) is not None:
                        raise ValueError('extra journal records')
        if _hash_stream(raw_events, expected_bytes=EVENTS_BYTES) != (EVENTS_SHA256, EVENTS_BYTES):
            raise ValueError('journal changed during analysis')
    _check_prediction(case_results, original)
    with open(subset_path, 'rb') as source:
        subset_sha, subset_bytes = _hash_stream(source)
    _require_binary64_subnormals()
    return {
        'schema': 'bert-negation-distribution-posthoc-v1',
        'scope': ('Post-hoc descriptive saved-logit analysis; no model forward, independent execution '
                  'attestation or population inference.'),
        'inputs': {'events_sha256': EVENTS_SHA256, 'events_bytes': EVENTS_BYTES,
                   'manifest_sha256': MANIFEST_SHA256, 'original_result_sha256': RESULT_SHA256,
                   'vocab_sha256': VOCAB_SHA256, 'original_source_commit': SOURCE_COMMIT},
        'verification': {'journal_records': 456, 'forwards': 448, 'case_count': 8,
                         'full_logit_self_controls': controls['self'],
                         'full_logit_repeat_controls': controls['repeat'],
                         'full_logit_final_cue_controls': controls['final_cue'],
                         'journal_sha256_checked_before_and_after': True,
                         'scope': 'Saved schedule, mask logits, recorded doses and score; not a full activation replay.'},
        'subset': {'format': 'case-local-deduplicated-concatenated-f32le-gzip-v1',
                   'sha256': subset_sha, 'compressed_bytes': subset_bytes,
                   'uncompressed_bytes': offset, 'vocab_size': VOCAB_SIZE,
                   'selected_event_count': 288, 'stored_vector_count': offset // (4 * VOCAB_SIZE),
                   'index': index_cases,
                   'scope': 'Exact selected mask-logit bytes and source event positions; source linkage is reported.'},
        'cases': case_results, 'summary': summarize_cases(case_results),
        'limits': {'max_journal_line_bytes': MAX_LINE_BYTES, 'max_records': 456,
                   'max_selected_vectors_per_case': 36, 'vocab_size': VOCAB_SIZE,
                   'note': 'Logical input caps and one-case buffering; not a measured process RSS bound.'},
    }


def _same_nested(actual, expected):
    if type(actual) is dict and type(expected) is dict:
        return set(actual) == set(expected) and all(_same_nested(actual[k], expected[k]) for k in actual)
    if type(actual) is list and type(expected) is list:
        return len(actual) == len(expected) and all(_same_nested(a, b) for a, b in zip(actual, expected))
    if type(actual) in (float, int) and type(expected) in (float, int):
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected,
                                                                                   rel_tol=1e-12, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


def replay_subset(subset_path, report_path, manifest_path, result_path, vocabulary_path):
    """Recompute published metrics from selected vectors; this cannot authenticate the source journal."""
    _require_binary64_subnormals()
    manifest, original, vocabulary = _read_inputs(manifest_path, result_path, vocabulary_path)
    with open(report_path, 'rb') as source:
        data = source.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError('reference report exceeds limit')
    reference = json.loads(data)
    info = reference['subset']
    if (reference.get('schema') != 'bert-negation-distribution-posthoc-v1' or
            reference['inputs']['events_sha256'] != EVENTS_SHA256 or
            info['format'] != 'case-local-deduplicated-concatenated-f32le-gzip-v1' or
            info['vocab_size'] != VOCAB_SIZE or info['selected_event_count'] != 288 or
            len(info['index']) != 8 or not 0 < info['uncompressed_bytes'] <= 288 * 4 * VOCAB_SIZE or
            info['uncompressed_bytes'] % (4 * VOCAB_SIZE) or
            info['stored_vector_count'] != info['uncompressed_bytes'] // (4 * VOCAB_SIZE) or
            not 0 < info['compressed_bytes'] <= 40 * 1024 * 1024):
        raise ValueError('subset metadata differs')
    cases, offset = [], 0
    with open(subset_path, 'rb') as raw:
        if _hash_stream(raw, expected_bytes=info['compressed_bytes']) != (info['sha256'], info['compressed_bytes']):
            raise ValueError('subset hash or length differs')
        with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
            for i, (case, saved, index) in enumerate(zip(manifest['cases'], original['cases'], info['index'])):
                if index['case_id'] != case['id']:
                    raise ValueError('subset case identity differs')
                records, offset = read_subset_case(stream, index['entries'], i, offset, VOCAB_SIZE)
                for donor, target in TRANSFERS:
                    if records[('transfer', 'cue5', donor, target)] != records[('clean', target)]:
                        raise ValueError('subset final-cue control differs')
                cases.append(analyze_case(records, case, saved, 2151, 2070, vocabulary))
            if stream.read(1):
                raise ValueError('subset contains extra uncompressed bytes')
        if offset != info['uncompressed_bytes'] or _hash_stream(raw) != (info['sha256'], info['compressed_bytes']):
            raise ValueError('subset byte count or hash differs')
    _check_prediction(cases, original)
    summary = summarize_cases(cases)
    if not _same_nested(cases, reference['cases']) or not _same_nested(summary, reference['summary']):
        raise ValueError('subset-derived metrics differ from saved report')
    _require_binary64_subnormals()
    return {'status': 'subset_replay_pass', 'cases': len(cases), 'selected_event_references': 288,
            'stored_unique_case_local_vectors': info['stored_vector_count'],
            'subset_sha256': info['sha256'], 'uncompressed_bytes': offset,
            'scope': 'Selected vector and metric consistency only; not original-journal or execution attestation.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    journal = sub.add_parser('journal', help='verify the original journal and create a selected-vector subset')
    journal.add_argument('--events', required=True)
    journal.add_argument('--subset-output', required=True)
    journal.add_argument('--output', required=True)
    journal.add_argument('--trace-memory', action='store_true',
                         help='record Python-traced heap peak; can make the replay much slower')
    replay = sub.add_parser('subset', help='recompute the saved metrics from a selected-vector subset')
    replay.add_argument('--subset', required=True)
    replay.add_argument('--index', required=True)
    for item in (journal, replay):
        item.add_argument('--manifest', required=True)
        item.add_argument('--result', required=True)
        item.add_argument('--vocab', required=True)
    args = parser.parse_args(argv)
    if args.mode == 'journal':
        if args.trace_memory:
            tracemalloc.start()
        report = analyze_journal(args.events, args.manifest, args.result, args.vocab, args.subset_output)
        peak = None
        if args.trace_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        report['execution'] = {'python_version': sys.version.split()[0],
                               'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               'python_traced_heap_peak_bytes': peak,
                               'memory_note': 'Optional traced peak excludes serialization, native allocations and interpreter baseline; no RSS bound is claimed.'}
        with open(args.output, 'xb') as target:
            target.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))
    else:
        print(json.dumps(replay_subset(args.subset, args.index, args.manifest, args.result, args.vocab),
                         sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
