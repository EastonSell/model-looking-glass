"""Construct public fixed noun-substitution inputs with tokenizer data only.

The caller must independently observe publication of the exact protocol and
reviewed inventory revision. This helper does not query protected data or
authorize a model forward.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

PROTOCOL_SHA = 'cc43267dbb412234a31844e837263541d0e16445eb9d2cb45fcb9ce7a9614796'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
BINDING_SHA = '23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0'
DIRECTION_CASE_SHA = 'a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13'
SUBJECT_CASE_SHA = '5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e'
SUPPORT_SHA = '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'
SUCCESSOR_SHA = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
MODEL_ID = 'distilbert/distilgpt2'
MODEL_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
WEIGHTS_SHA = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
FRAMES = ('Before sunrise,', 'After sunset,')
CONDITIONS = ('none', 'attractor_singular', 'attractor_plural')
REGULAR = (('hand', 'hands'), ('arm', 'arms'), ('leg', 'legs'), ('eye', 'eyes'),
           ('ear', 'ears'), ('finger', 'fingers'), ('toe', 'toes'))
IRREGULAR = (('foot', 'feet'), ('tooth', 'teeth'))
ATTRACTORS = (('stone', 'stones'), ('wall', 'walls'), ('gate', 'gates'),
              ('lamp', 'lamps'), ('rock', 'rocks'), ('box', 'boxes'))
ANSWERS = ((' is', 318), (' are', 389), (' was', 373), (' were', 547),
           (' the', 262), (' a', 257), (' in', 287), (' on', 319))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit, expected=None):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or (expected is not None and sha(raw) != expected):
        raise ValueError('input byte bound or digest differs')
    return raw


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def token_row(value):
    if (type(value) is not list or not 1 <= len(value) <= 128
            or any(type(token) is not int or not 0 <= token < 50257 for token in value)):
        raise ValueError('invalid token row')
    return tuple(value)


def select_pairs(tokenizer, pool, count, forbidden):
    selected, log = [], []
    for pair in pool:
        ids = [encode(tokenizer, ' ' + form) for form in pair]
        if any(len(ids_for_form) != 1 for ids_for_form in ids):
            reason = 'not_one_token'
        elif ids[0][0] == ids[1][0]:
            reason = 'same_token'
        elif any(value[0] in forbidden for value in ids):
            reason = 'previous_or_selected_token'
        else:
            reason = 'selected'
            forbidden.update(value[0] for value in ids)
            selected.append({'forms': list(pair), 'token_ids': [value[0] for value in ids]})
        log.append({'forms': list(pair), 'token_ids': ids, 'decision': reason})
        if len(selected) == count:
            return selected, log
    raise ValueError('fixed token pool exhausted')


def prior_tokens(binding, direction_cases, subject_cases):
    subjects, attractors = set(), set()
    for row in binding['cases']:
        subjects.add(row['input_ids'][row['subject_index']])
        attractors.add(row['input_ids'][-1])
    for row in direction_cases['cases']:
        if row['role'] == 'target':
            subjects.add(row['token_ids'][row['subject_index']])
        attractors.add(row['token_ids'][-1])
    for row in subject_cases['cases']:
        if row['role'] == 'target':
            subjects.add(row['token_ids'][row['subject_index']])
        if row['condition'] in ('mass', 'attractor_singular', 'attractor_plural'):
            attractors.add(row['token_ids'][-1])
    return subjects, attractors


def validate_groups(cases):
    """Own-base continuations are intentional; cross-case relations are not."""
    groups = []
    for row in cases:
        base = token_row(row['token_ids'])
        group = [base] + [base + (answer,) for _, answer in ANSWERS]
        if len(set(group)) != len(group):
            raise ValueError('duplicate own continuation')
        groups.append(group)
    for i, group in enumerate(groups):
        for other in groups[i + 1:]:
            if any(related(a, b) for a in group for b in other):
                raise ValueError('distinct proposed cases overlap')
    return groups


def inventory_sequences(payload):
    if type(payload) is not dict or type(payload.get('records')) is not list:
        raise ValueError('invalid exposure inventory')
    result = set()
    for record in payload['records']:
        row = token_row(record['token_ids'])
        result.add(row)
        candidates = record.get('candidate_token_ids', [])
        groups = record.get('candidate_token_groups', {})
        if type(candidates) is not list or type(groups) is not dict:
            raise ValueError('invalid candidate metadata')
        values = list(candidates)
        for group in groups.values():
            if type(group) is not list:
                raise ValueError('invalid candidate group')
            values.extend(group)
        if record.get('target_token_id') is not None:
            values.append(record['target_token_id'])
        if any(type(value) is not int or not 0 <= value < 50257 for value in values):
            raise ValueError('invalid candidate token')
        result.update(row + (value,) for value in values)
    return result


def verify_inventory(raw, expected_sha, expected_version):
    if sha(raw) != expected_sha:
        raise ValueError('inventory digest differs')
    payload = json.loads(raw)
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=True, allow_nan=False).encode('ascii')
    identity = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
                'tokenizer_id': MODEL_ID, 'tokenizer_revision': MODEL_REVISION,
                'tokenizer_sha256': TOKENIZER_SHA, 'weights_sha256': WEIGHTS_SHA}
    if (raw != canonical or payload.get('version') != expected_version
            or payload.get('identity') != identity):
        raise ValueError('inventory canonical bytes, version or model identity differ')
    return payload


def build_cases(tokenizer, binding, direction_cases, subject_cases):
    old_subjects, old_attractors = prior_tokens(binding, direction_cases, subject_cases)
    used = set(old_subjects)
    regular, rlog = select_pairs(tokenizer, REGULAR, 2, used)
    irregular, ilog = select_pairs(tokenizer, IRREGULAR, 2, used)
    attractors, alog = select_pairs(tokenizer, ATTRACTORS, 2, used | old_attractors)
    nouns = {'R1': regular[0], 'R2': regular[1], 'I1': irregular[0], 'I2': irregular[1]}
    for text, expected in ANSWERS:
        if encode(tokenizer, text) != [expected]:
            raise ValueError('answer token differs')
    cases = []
    for frame_index, frame in enumerate(FRAMES):
        prefix = encode(tokenizer, frame + ' the')
        for condition_index, condition in enumerate(CONDITIONS):
            suffix = (' nearby' if condition == 'none' else
                      ' near the ' + attractors[frame_index]['forms'][condition_index - 1])
            block = []
            for morphology_index, (morphology, keys) in enumerate((('regular', ('R1', 'R2')),
                                                                   ('irregular', ('I1', 'I2')))):
                for lemma_index, key in enumerate(keys):
                    for number_index, number in enumerate(('singular', 'plural')):
                        item = nouns[key]
                        text = frame + ' the ' + item['forms'][number_index] + suffix
                        ids = encode(tokenizer, text)
                        index = len(prefix)
                        if (not 1 <= len(ids) <= 12 or ids[:index] != prefix
                                or index >= len(ids) - 1 or ids[index] != item['token_ids'][number_index]):
                            raise ValueError('target subject coordinate or length differs')
                        if any(encode(tokenizer, text + answer) != ids + [target]
                               for answer, target in ANSWERS):
                            raise ValueError('target continuation boundary differs')
                        block.append({'case_id': f'noun-sub-f{frame_index+1}-c{condition_index}-{key}-{number}',
                                      'frame_index': frame_index, 'condition_index': condition_index,
                                      'condition': condition, 'role': 'target',
                                      'morphology': morphology, 'morphology_index': morphology_index,
                                      'lemma_index': lemma_index, 'subject_key': key,
                                      'subject_number': number, 'subject_form': item['forms'][number_index],
                                      'text': text, 'token_ids': ids, 'subject_index': index,
                                      'final_index': len(ids) - 1})
            for row in block[1:]:
                a, b, index = block[0]['token_ids'], row['token_ids'], block[0]['subject_index']
                if (row['subject_index'] != index or len(a) != len(b) or a[index] == b[index]
                        or a[:index] + a[index+1:] != b[:index] + b[index+1:]):
                    raise ValueError('target variants differ beyond one subject token')
            text = frame + suffix
            ids = encode(tokenizer, text)
            if (not 1 <= len(ids) <= 12
                    or any(encode(tokenizer, text + answer) != ids + [target]
                           for answer, target in ANSWERS)):
                raise ValueError('fragment length or answer boundary differs')
            block.append({'case_id': f'noun-sub-f{frame_index+1}-c{condition_index}-context',
                          'frame_index': frame_index, 'condition_index': condition_index,
                          'condition': condition, 'role': 'context', 'morphology': None,
                          'morphology_index': None, 'lemma_index': None, 'subject_key': None,
                          'subject_number': None, 'text': text, 'token_ids': ids,
                          'subject_index': None, 'final_index': len(ids) - 1})
            cases.extend(block)
        for index in range(9):
            a = cases[frame_index * 27 + 9 + index]['token_ids']
            b = cases[frame_index * 27 + 18 + index]['token_ids']
            if len(a) != len(b) or a[:-1] != b[:-1] or a[-1] == b[-1]:
                raise ValueError('locatives differ beyond their final noun')
    if len(cases) != 54:
        raise ValueError('case count differs')
    groups = validate_groups(cases)
    selection = {'regular': regular, 'irregular': irregular, 'attractors': attractors,
                 'eligibility': {'regular': rlog, 'irregular': ilog, 'attractors': alog}}
    return cases, groups, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'protocol-revision', 'tokenizer-json', 'prior-binding',
                 'prior-direction-cases', 'prior-subject-cases', 'inventory',
                 'inventory-version', 'inventory-revision', 'inventory-sha256',
                 'natural-support', 'natural-successor', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if (re.fullmatch('[0-9a-f]{40}', args.protocol_revision) is None
            or re.fullmatch('[0-9a-f]{40}', args.inventory_revision) is None
            or re.fullmatch('[0-9a-f]{64}', args.inventory_sha256) is None):
        raise ValueError('invalid caller-supplied revision or digest')
    read(args.protocol, 256 * 1024, PROTOCOL_SHA)
    tokenizer_raw = read(args.tokenizer_json, 2 * 1024 * 1024, TOKENIZER_SHA)
    binding = json.loads(read(args.prior_binding, 1024 * 1024, BINDING_SHA))
    direction = json.loads(read(args.prior_direction_cases, 1024 * 1024, DIRECTION_CASE_SHA))
    subjects = json.loads(read(args.prior_subject_cases, 1024 * 1024, SUBJECT_CASE_SHA))
    inv_raw = read(args.inventory, 16 * 1024 * 1024, args.inventory_sha256)
    inventory = verify_inventory(inv_raw, args.inventory_sha256, args.inventory_version)
    support = json.loads(read(args.natural_support, 1024 * 1024, SUPPORT_SHA))
    successor = json.loads(read(args.natural_successor, 1024 * 1024, SUCCESSOR_SHA))
    support_rows = [token_row(row) for row in support['rows']]
    successor_rows = [token_row(row['input_ids']) for row in successor['cases']]
    if len(support_rows) != 128 or len(successor_rows) != 256:
        raise ValueError('natural public support counts differ')
    from tokenizers import Tokenizer, __version__
    if __version__ != '0.22.2':
        raise ValueError('tokenizers version differs')
    tokenizer = Tokenizer.from_str(tokenizer_raw.decode('utf-8'))
    cases, groups, selection = build_cases(tokenizer, binding, direction, subjects)
    proposed = [row for group in groups for row in group]
    known = inventory_sequences(inventory)
    for label, other in (('inventory', known), ('natural support', support_rows),
                         ('natural successor', successor_rows)):
        if any(related(row, item) for row in proposed for item in other):
            raise ValueError('public prefix collision: ' + label)
    rows = [row['token_ids'] for row in cases]
    rows_raw = json.dumps(rows, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    result = {'schema': 'distilgpt2-agreement-noun-substitution-cases-v1',
              'scope': 'Public prospective nonblind tokenizer-only inputs; no model forward.',
              'publication_requirement': 'Caller independently verifies protocol publication before construction.',
              'protocol_revision': args.protocol_revision, 'protocol_sha256': PROTOCOL_SHA,
              'case_preparer_sha256': sha(read(__file__, 1024 * 1024)),
              'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
              'tokenizer_sha256': TOKENIZER_SHA, 'tokenizers_version': __version__,
              'inventory_revision': args.inventory_revision, 'inventory_version': args.inventory_version,
              'inventory_sha256': args.inventory_sha256, 'inventory_sequences_checked': len(known),
              'natural_support_sha256': SUPPORT_SHA, 'natural_successor_sha256': SUCCESSOR_SHA,
              'pending_public_checks': 'Separate independently reviewed pending-input comparison required.',
              'custody_status': 'Fresh natural and injection reports required before forwarding.',
              'answer_ids': [value for _, value in ANSWERS], 'expected_forwards': 486,
              'ordered_rows_sha256': sha(rows_raw), 'selection': selection, 'cases': cases}
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with Path(args.output).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short case-file write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'case_sha256': sha(data), 'rows_sha256': sha(rows_raw),
                      'cases': len(cases), 'known_sequences': len(known)}, sort_keys=True))


if __name__ == '__main__':
    main()
