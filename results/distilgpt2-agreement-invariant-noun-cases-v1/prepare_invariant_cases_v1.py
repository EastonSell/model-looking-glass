"""Tokenizer-only preparer for the fixed invariant-noun study.

Publication and currentness of the supplied public artifacts, custodian
observations and model execution are separate operator responsibilities.
No model library is loaded by this file.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import types

PROTOCOL_SHA = 'be7ab65c18fb706cd7e833edc0c32a6b833172b3b1996ac3ec94f268f1c39147'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
SCHEDULE_SHA = 'bdb9f77cf12b191f2a713324077f6f175b1facdce7989ff79fe0c7390aec21a4'
PRIOR_SHAS = ('23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0',
              'a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13',
              '5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e',
              '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe')
SUPPORT_SHAS = ('5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6',
                '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4')
MODEL = 'distilbert/distilgpt2'
REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
WEIGHTS_SHA = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
ANSWERS = ((' is', 318), (' are', 389), (' was', 373), (' were', 547),
           (' the', 262), (' a', 257), (' in', 287), (' on', 319))
POOLS = {'discovery': ('sheep', 'deer', 'fish', 'moose', 'elk', 'cod'),
         'evaluation': ('bison', 'squid', 'shrimp', 'carp', 'aircraft',
                        'spacecraft', 'swine', 'offspring')}
FRAMES = {'discovery': ('Before breakfast,', 'After dinner,'),
          'evaluation': ('At midnight,', 'During autumn,')}
ATTRACTORS = (('gate', 'gates'), ('lamp', 'lamps'), ('rock', 'rocks'),
              ('tree', 'trees'), ('arch', 'arches'), ('box', 'boxes'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit, digest):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or sha(raw) != digest:
        raise ValueError('input digest or bound mismatch')
    return raw


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def old_tokens(binding, direction, subject, substitution):
    old_subjects, old_attractors = set(), set()
    for row in binding['cases']:
        old_subjects.add(row['input_ids'][row['subject_index']])
        old_attractors.add(row['input_ids'][-1])
    for row in direction['cases']:
        if row['role'] == 'target':
            old_subjects.add(row['token_ids'][row['subject_index']])
        old_attractors.add(row['token_ids'][-1])
    for bundle in (subject, substitution):
        for row in bundle['cases']:
            if row['role'] == 'target':
                old_subjects.add(row['token_ids'][row['subject_index']])
            if row['condition'] != 'none':
                old_attractors.add(row['token_ids'][-1])
    return old_subjects, old_attractors


def token_forms(tokenizer, tokens):
    return {tokenizer.decode([token]).strip().casefold() for token in tokens}


def select_nouns(tokenizer, old_subjects):
    used_tokens = set(old_subjects)
    used_forms = token_forms(tokenizer, old_subjects)
    selected, decisions = {}, {}
    for split, pool in POOLS.items():
        selected[split], decisions[split] = [], []
        for form in pool:
            ids = encode(tokenizer, ' ' + form)
            a, b = encode(tokenizer, ' this ' + form), encode(tokenizer, ' these ' + form)
            if len(ids) != 1:
                reason = 'not_one_token'
            elif a != [428, *ids] or b != [777, *ids]:
                reason = 'determiner_alignment'
            elif ids[0] in used_tokens or form.casefold() in used_forms:
                reason = 'previous_or_selected_subject'
            elif not all(encode(tokenizer, prefix + answer) == row + [answer_id]
                         for prefix, row in ((' this ' + form, a), (' these ' + form, b))
                         for answer, answer_id in ANSWERS):
                reason = 'answer_boundary'
            else:
                reason = 'selected'
                selected[split].append({'form': form, 'token_id': ids[0]})
                used_tokens.add(ids[0])
                used_forms.add(form.casefold())
            decisions[split].append({'form': form, 'ids': ids, 'decision': reason})
            if len(selected[split]) == 2:
                break
        if len(selected[split]) != 2:
            raise ValueError('fixed noun pool exhausted')
    return selected, decisions


def select_attractors(tokenizer, old_subjects, old_attractors, nouns):
    forbidden = set(old_subjects) | set(old_attractors)
    forbidden.update(noun['token_id'] for pairs in nouns.values() for noun in pairs)
    forms = token_forms(tokenizer, old_subjects | old_attractors)
    forms.update(noun['form'].casefold() for pairs in nouns.values() for noun in pairs)
    selected, decisions = [], []
    for pair in ATTRACTORS:
        ids = [encode(tokenizer, ' ' + form) for form in pair]
        if any(len(row) != 1 for row in ids):
            reason = 'not_one_token'
        elif ids[0][0] == ids[1][0] or any(row[0] in forbidden for row in ids):
            reason = 'previous_or_selected_token'
        elif len(set(x.casefold() for x in pair)) != 2 or any(x.casefold() in forms for x in pair):
            reason = 'previous_or_selected_form'
        else:
            reason = 'selected'
            selected.append({'forms': list(pair), 'token_ids': [row[0] for row in ids]})
            forbidden.update(row[0] for row in ids)
            forms.update(x.casefold() for x in pair)
        decisions.append({'forms': list(pair), 'ids': ids, 'decision': reason})
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError('fixed attractor pool exhausted')
    return selected, decisions


def case_row(tokenizer, identity, split, role, frame, context, noun, determiner, text):
    ids = encode(tokenizer, text)
    if not 1 <= len(ids) <= 12 or any(type(x) is not int or not 0 <= x < 50257 for x in ids):
        raise ValueError('invalid case token row')
    if any(encode(tokenizer, text + answer) != ids + [answer_id] for answer, answer_id in ANSWERS):
        raise ValueError('answer boundary mismatch')
    row = {'case_id': identity, 'split': split, 'role': role, 'frame_index': frame,
           'context_index': context, 'noun_index': noun, 'determiner': determiner,
           'text': text, 'token_ids': ids, 'final_index': len(ids) - 1}
    return row


def build(tokenizer, binding, direction, subject, substitution, schedule):
    if any(encode(tokenizer, answer) != [token] for answer, token in ANSWERS):
        raise ValueError('fixed answer token mismatch')
    if encode(tokenizer, ' this') != [428] or encode(tokenizer, ' these') != [777]:
        raise ValueError('fixed determiner token mismatch')
    old_subjects, old_attractors = old_tokens(binding, direction, subject, substitution)
    nouns, noun_decisions = select_nouns(tokenizer, old_subjects)
    attractors, attractor_decisions = select_attractors(tokenizer, old_subjects, old_attractors, nouns)
    rows = {}
    for split_index, split in enumerate(('discovery', 'evaluation')):
        for f, frame in enumerate(FRAMES[split]):
            pair = attractors[split_index * 2 + f]
            for c in range(3):
                suffix = ' nearby' if c == 0 else ' near the ' + pair['forms'][c - 1]
                for n, noun in enumerate(nouns[split]):
                    for det in ('this', 'these'):
                        identity = f'{split}_target_f{f}_c{c}_n{n}_{det}'
                        text = f'{frame} {det} {noun["form"]}{suffix}'
                        row = case_row(tokenizer, identity, split, 'target', f, c, n, det, text)
                        det_id = 428 if det == 'this' else 777
                        if row['token_ids'].count(det_id) != 1 or row['token_ids'].count(noun['token_id']) != 1:
                            raise ValueError('ambiguous target coordinate')
                        row['determiner_index'] = row['token_ids'].index(det_id)
                        row['subject_index'] = row['token_ids'].index(noun['token_id'])
                        rows[identity] = row
                identity = f'{split}_fragment_f{f}_c{c}'
                rows[identity] = case_row(tokenizer, identity, split, 'fragment', f, c, None, None,
                                          frame + suffix)
            for det in ('this', 'these'):
                identity = f'{split}_demonstrative_f{f}_{det}'
                row = case_row(tokenizer, identity, split, 'demonstrative', f, 0, None, det,
                               f'{frame} {det} nearby')
                det_id = 428 if det == 'this' else 777
                if row['token_ids'].count(det_id) != 1:
                    raise ValueError('ambiguous demonstrative coordinate')
                row['determiner_index'] = row['token_ids'].index(det_id)
                rows[identity] = row
    ordered_ids = [identity for identity, _ in schedule.cases()]
    if len(rows) != 68 or set(rows) != set(ordered_ids) or len(ordered_ids) != 68:
        raise ValueError('case/schedule identities differ')
    for split in ('discovery', 'evaluation'):
        for f in range(2):
            first = rows[f'{split}_demonstrative_f{f}_this']
            second = rows[f'{split}_demonstrative_f{f}_these']
            d = first['determiner_index']
            if (second['determiner_index'] != d or len(first['token_ids']) != len(second['token_ids'])
                    or first['token_ids'][:d] + first['token_ids'][d + 1:]
                    != second['token_ids'][:d] + second['token_ids'][d + 1:]):
                raise ValueError('demonstrative comparator alignment differs')
            for c in range(3):
                base = f'{split}_target_f{f}_c{c}_n'
                group = [rows[base + str(n) + '_' + det] for n in range(2) for det in ('this', 'these')]
                d, n = group[0]['determiner_index'], group[0]['subject_index']
                if d == n or any(len(row['token_ids']) != len(group[0]['token_ids'])
                                 or row['determiner_index'] != d or row['subject_index'] != n
                                 for row in group):
                    raise ValueError('factorial token positions differ')
                for row in group:
                    if any(a != b for i, (a, b) in enumerate(zip(group[0]['token_ids'], row['token_ids']))
                           if i not in (d, n)):
                        raise ValueError('factorial token alignment differs')
            for n in range(2):
                for det in ('this', 'these'):
                    singular = rows[f'{split}_target_f{f}_c1_n{n}_{det}']['token_ids']
                    plural = rows[f'{split}_target_f{f}_c2_n{n}_{det}']['token_ids']
                    if singular[:-1] != plural[:-1] or singular[-1] == plural[-1]:
                        raise ValueError('attractor alignment differs')
            singular = rows[f'{split}_fragment_f{f}_c1']['token_ids']
            plural = rows[f'{split}_fragment_f{f}_c2']['token_ids']
            if singular[:-1] != plural[:-1] or singular[-1] == plural[-1]:
                raise ValueError('fragment attractor alignment differs')
    ordered = [rows[identity] for identity in ordered_ids]
    groups = make_groups(ordered)
    return ordered, groups, {'nouns': nouns, 'noun_decisions': noun_decisions,
                             'attractors': attractors, 'attractor_decisions': attractor_decisions}


def token_row(row):
    if (type(row) is not list or not 1 <= len(row) <= 128
            or any(type(x) is not int or not 0 <= x < 50257 for x in row)):
        raise ValueError('invalid public token row')
    return tuple(row)


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def make_groups(cases):
    groups = []
    for row in cases:
        base = token_row(row['token_ids'])
        group = [base] + [base + (token,) for _, token in ANSWERS]
        if len(set(group)) != len(group):
            raise ValueError('duplicate own continuation')
        groups.append(group)
    for index, group in enumerate(groups):
        if any(related(a, b) for other in groups[index + 1:] for a in group for b in other):
            raise ValueError('cross-case prefix collision')
    return groups


def inventory_sequences(payload):
    if type(payload) is not dict or type(payload.get('records')) is not list:
        raise ValueError('invalid inventory')
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
                raise ValueError('invalid grouped candidates')
            values.extend(group)
        if record.get('target_token_id') is not None:
            values.append(record['target_token_id'])
        if any(type(value) is not int or not 0 <= value < 50257 for value in values):
            raise ValueError('invalid candidate token')
        result.update(row + (value,) for value in values)
    return result


def verify_inventory(raw, expected_version):
    payload = json.loads(raw)
    identity = {'model_id': MODEL, 'model_revision': REVISION, 'tokenizer_id': MODEL,
                'tokenizer_revision': REVISION, 'tokenizer_sha256': TOKENIZER_SHA,
                'weights_sha256': WEIGHTS_SHA}
    if (compact(payload) != raw or payload.get('version') != expected_version
            or payload.get('identity') != identity):
        raise ValueError('inventory identity, version or canonical bytes differ')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('protocol', 'protocol-revision', 'schedule', 'tokenizer-json',
                'prior-binding', 'prior-direction-cases', 'prior-subject-cases',
                'prior-substitution-cases', 'inventory', 'inventory-version',
                'inventory-revision', 'inventory-sha256', 'natural-support',
                'natural-successor', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if PROTOCOL_SHA is None:
        raise ValueError('protocol not yet frozen and published')
    if (not re.fullmatch('[0-9a-f]{40}', args.protocol_revision)
            or not re.fullmatch('[0-9a-f]{40}', args.inventory_revision)
            or not re.fullmatch('[0-9a-f]{64}', args.inventory_sha256)):
        raise ValueError('invalid revision or digest')
    read(args.protocol, 256 * 1024, PROTOCOL_SHA)
    schedule_raw = read(args.schedule, 256 * 1024, SCHEDULE_SHA)
    schedule = types.ModuleType('_frozen_invariant_schedule')
    exec(compile(schedule_raw, args.schedule, 'exec', dont_inherit=True), schedule.__dict__)
    tokenizer_raw = read(args.tokenizer_json, 2 * 1024 * 1024, TOKENIZER_SHA)
    paths = (args.prior_binding, args.prior_direction_cases, args.prior_subject_cases,
             args.prior_substitution_cases)
    prior = [json.loads(read(path, 2 * 1024 * 1024, digest)) for path, digest in zip(paths, PRIOR_SHAS)]
    inventory = verify_inventory(read(args.inventory, 32 * 1024 * 1024, args.inventory_sha256),
                                 args.inventory_version)
    support = json.loads(read(args.natural_support, 2 * 1024 * 1024, SUPPORT_SHAS[0]))
    successor = json.loads(read(args.natural_successor, 2 * 1024 * 1024, SUPPORT_SHAS[1]))
    support_rows = [token_row(x) for x in support['rows']]
    successor_rows = [token_row(x['input_ids']) for x in successor['cases']]
    if len(support_rows) != 128 or len(successor_rows) != 256:
        raise ValueError('public support counts differ')
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise ValueError('unexpected model import')
    from tokenizers import Tokenizer, __version__
    if __version__ != '0.22.2':
        raise ValueError('tokenizer version differs')
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise ValueError('unexpected model import')
    tokenizer = Tokenizer.from_str(tokenizer_raw.decode('utf-8'))
    cases, groups, selection = build(tokenizer, *prior, schedule)
    proposed = [sequence for group in groups for sequence in group]
    known = inventory_sequences(inventory)
    for label, other in (('inventory', known), ('natural support', support_rows),
                         ('natural successor', successor_rows)):
        if any(related(row, item) for row in proposed for item in other):
            raise ValueError('public prefix collision: ' + label)
    actions = schedule.actions()
    action_digest = sha(json.dumps(actions, separators=(',', ':'), ensure_ascii=True).encode('ascii'))
    if action_digest != 'fbf526fd867ed48127c24b3b35004727b1757782e90798d60dcfdec37ac9b2f6':
        raise ValueError('fixed action digest differs')
    row_digest = sha(json.dumps([row['token_ids'] for row in cases],
                                separators=(',', ':'), ensure_ascii=True).encode('ascii'))
    result = {'schema': 'agreement-invariant-noun-cases-v1', 'protocol_revision': args.protocol_revision,
              'protocol_sha256': PROTOCOL_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
              'model_id': MODEL, 'model_revision': REVISION, 'tokenizers_version': __version__,
              'inventory_revision': args.inventory_revision, 'inventory_version': args.inventory_version,
              'inventory_sha256': args.inventory_sha256, 'inventory_sequences_checked': len(known),
              'prior_bundle_sha256': list(PRIOR_SHAS), 'public_support_sha256': list(SUPPORT_SHAS),
              'selection': selection, 'answers': [value for _, value in ANSWERS],
              'ordered_rows_sha256': row_digest, 'actions_sha256': action_digest,
              'expected_forwards': len(actions), 'cases': cases,
              'pending_and_custody': 'Separate public pending-input comparisons and fresh custody required.'}
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with Path(args.output).open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'case_sha256': sha(raw), 'ordered_rows_sha256': row_digest,
                      'actions_sha256': action_digest, 'cases': len(cases),
                      'inventory_sequences_checked': len(known), 'forwards': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
