"""Prepare fixed public agreement cases with a pinned tokenizer, never a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

PROTOCOL_REVISION = '54f0f8d3ac7e60efc3005f0f49415715ba34ff47'
PROTOCOL_SHA = 'bd39b4556023110d0de88fb21381a0a3ce1e17cad862cdd0259b67849f540714'
REFERENCE_SHA = 'bdb31d5b74b6cff17ed2a683415beab22e36b101220257d774909be6f9f804bd'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
OLD_BINDING_SHA = '23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0'
OLD_CASES_SHA = 'a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13'
SUPPORT_SHA = '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'
SUCCESSOR_SHA = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
FULL_TOP_SHA = '4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261'
FRAMES = ('At dawn,', 'At dusk,', 'At noon,', 'In summer,')
CONDITIONS = ('none', 'mass', 'attractor_singular', 'attractor_plural')
REGULAR = (('horse','horses'), ('bird','birds'), ('duck','ducks'), ('cow','cows'),
           ('pig','pigs'), ('lion','lions'), ('bear','bears'), ('frog','frogs'))
IRREGULAR = (('mouse','mice'), ('goose','geese'), ('wolf','wolves'), ('ox','oxen'),
             ('calf','calves'), ('louse','lice'))
ATTRACTORS = (('bridge','bridges'), ('road','roads'), ('ship','ships'), ('hill','hills'),
              ('garden','gardens'), ('box','boxes'), ('car','cars'), ('table','tables'))
ANSWERS = ((' is',318), (' are',389), (' was',373), (' were',547),
           (' the',262), (' a',257), (' in',287), (' on',319))
ASSIGNMENT = (('R1','I1'), ('R1','I2'), ('R2','I1'), ('R2','I2'))


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


def select_pairs(tokenizer, pool, count, forbidden):
    chosen, log = [], []
    for pair in pool:
        ids = [encode(tokenizer, ' ' + form) for form in pair]
        if any(len(x) != 1 for x in ids):
            reason = 'not_one_token'
        elif ids[0][0] == ids[1][0]:
            reason = 'same_token'
        elif any(x[0] in forbidden for x in ids):
            reason = 'previous_or_selected_token'
        else:
            reason = 'selected'
            forbidden.update(x[0] for x in ids)
            chosen.append({'forms': list(pair), 'token_ids': [x[0] for x in ids]})
        log.append({'forms': list(pair), 'token_ids': ids, 'decision': reason})
        if len(chosen) == count:
            return chosen, log
    raise ValueError('fixed token pool exhausted')


def _rows(value, count):
    if type(value) is not list or len(value) != count:
        raise ValueError('unexpected public row count')
    result = []
    for row in value:
        if (type(row) is not list or not 1 <= len(row) <= 128
                or any(type(t) is not int or not 0 <= t < 50257 for t in row)):
            raise ValueError('invalid public token row')
        result.append(tuple(row))
    return result


def inventory_sequences(payload):
    if type(payload) is not dict or type(payload.get('records')) is not list:
        raise ValueError('invalid public inventory')
    sequences = set()
    for record in payload['records']:
        row = _rows([record.get('token_ids')], 1)[0]
        sequences.add(row)
        ids = list(record.get('candidate_token_ids', []))
        groups = record.get('candidate_token_groups', {})
        if type(groups) is not dict or any(type(x) is not list for x in groups.values()):
            raise ValueError('invalid candidate groups')
        for values in groups.values():
            ids.extend(values)
        if record.get('target_token_id') is not None:
            ids.append(record['target_token_id'])
        if any(type(t) is not int or not 0 <= t < 50257 for t in ids):
            raise ValueError('invalid inventory candidate token')
        sequences.update(row + (token,) for token in ids)
    return sequences


def build_cases(tokenizer, prior_binding, prior_cases):
    old_subjects = {row['input_ids'][row['subject_index']] for row in prior_binding['cases']}
    old_subjects.update(row['token_ids'][row['subject_index']]
                        for row in prior_cases['cases'] if row['role'] == 'target')
    old_attractors = {row['token_ids'][-1] for row in prior_cases['cases']}
    used = set(old_subjects)
    regular, regular_log = select_pairs(tokenizer, REGULAR, 2, used)
    irregular, irregular_log = select_pairs(tokenizer, IRREGULAR, 2, used)
    attrs, attr_log = select_pairs(tokenizer, ATTRACTORS, 4, used | old_attractors)
    pairs = {'R1': regular[0], 'R2': regular[1], 'I1': irregular[0], 'I2': irregular[1]}
    for text, token in ANSWERS:
        if encode(tokenizer, text) != [token]:
            raise ValueError('fixed answer token differs')
    water = encode(tokenizer, ' water')
    if len(water) != 1:
        raise ValueError('mass noun is not one token')
    cases = []
    for frame_index, frame in enumerate(FRAMES):
        prefix = encode(tokenizer, frame + ' the')
        for condition_index, condition in enumerate(CONDITIONS):
            suffix = (' nearby' if condition == 'none' else
                      ' near the water' if condition == 'mass' else
                      ' near the ' + attrs[frame_index]['forms'][condition_index - 2])
            pair_rows = []
            for pair_index, key in enumerate(ASSIGNMENT[frame_index]):
                pair = pairs[key]
                for number_index, number in enumerate(('singular', 'plural')):
                    text = frame + ' the ' + pair['forms'][number_index] + suffix
                    ids = encode(tokenizer, text)
                    index = len(prefix)
                    if (not 1 <= len(ids) <= 12 or ids[:index] != prefix
                            or ids[index] != pair['token_ids'][number_index]):
                        raise ValueError('subject token or input length differs')
                    for answer, target in ANSWERS:
                        if encode(tokenizer, text + answer) != ids + [target]:
                            raise ValueError('answer continuation boundary differs')
                    row = {'case_id': f'agreement-subject-f{frame_index+1:02d}-c{condition_index}-{key}-{number}',
                           'frame_index': frame_index, 'condition': condition,
                           'condition_index': condition_index, 'role': 'target', 'pair_index': pair_index,
                           'subject_key': key, 'subject_number': number,
                           'morphology': 'regular' if pair_index == 0 else 'irregular',
                           'subject_lemma': pair['forms'][0], 'subject_form': pair['forms'][number_index],
                           'text': text, 'token_ids': ids, 'subject_index': index, 'final_index': len(ids)-1}
                    cases.append(row)
                    pair_rows.append(row)
            for i in (0, 2):
                a, b = pair_rows[i:i+2]
                if (len(a['token_ids']) != len(b['token_ids']) or a['subject_index'] != b['subject_index']
                        or a['token_ids'][:a['subject_index']] != b['token_ids'][:b['subject_index']]
                        or a['token_ids'][a['subject_index']+1:] != b['token_ids'][b['subject_index']+1:]):
                    raise ValueError('subject pair does not match outside subject')
            text = frame + suffix
            ids = encode(tokenizer, text)
            if not 1 <= len(ids) <= 12:
                raise ValueError('fragment length differs')
            for answer, target in ANSWERS:
                if encode(tokenizer, text + answer) != ids + [target]:
                    raise ValueError('fragment continuation boundary differs')
            cases.append({'case_id': f'agreement-subject-f{frame_index+1:02d}-c{condition_index}-context',
                          'frame_index': frame_index, 'condition': condition,
                          'condition_index': condition_index, 'role': 'context', 'pair_index': None,
                          'subject_key': None, 'subject_number': None, 'morphology': None,
                          'text': text, 'token_ids': ids, 'subject_index': None, 'final_index': len(ids)-1})
        for offset in range(5):
            base = cases[frame_index*20 + 5 + offset]['token_ids']
            for cond in (2,3):
                other = cases[frame_index*20 + cond*5 + offset]['token_ids']
                if len(base) != len(other) or base[:-1] != other[:-1] or base[-1] == other[-1]:
                    raise ValueError('mass/count locatives do not match outside final token')
        a, b = cases[frame_index*20 + 2*5 + 4]['token_ids'], cases[frame_index*20 + 3*5 + 4]['token_ids']
        if len(a) != len(b) or a[:-1] != b[:-1] or a[-1] == b[-1]:
            raise ValueError('numbered fragments do not match outside final token')
    rows = [tuple(row['token_ids']) for row in cases]
    if len(rows) != 80 or any(related(a,b) for i,a in enumerate(rows) for b in rows[i+1:]):
        raise ValueError('new case rows overlap one another')
    return cases, {'regular': regular, 'irregular': irregular, 'attractors': attrs,
                   'eligibility': {'regular': regular_log, 'irregular': irregular_log, 'attractors': attr_log}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'reference', 'tokenizer-json', 'prior-binding', 'prior-cases',
                 'inventory', 'natural-support', 'natural-successor', 'full-top', 'output',
                 'inventory-version', 'inventory-revision', 'inventory-sha256'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.inventory_revision) or not re.fullmatch('[0-9a-f]{64}', a.inventory_sha256):
        raise ValueError('invalid operator-supplied inventory identity')
    read(a.protocol, 256*1024, PROTOCOL_SHA)
    read(a.reference, 1024*1024, REFERENCE_SHA)
    token_raw = read(a.tokenizer_json, 2*1024*1024, TOKENIZER_SHA)
    prior_binding = json.loads(read(a.prior_binding, 1024*1024, OLD_BINDING_SHA))
    prior_cases = json.loads(read(a.prior_cases, 1024*1024, OLD_CASES_SHA))
    inventory_raw = read(a.inventory, 16*1024*1024, a.inventory_sha256)
    inventory = json.loads(inventory_raw)
    if (inventory.get('version') != a.inventory_version or inventory_raw !=
            json.dumps(inventory, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()):
        raise ValueError('inventory version or canonical bytes differ')
    support = json.loads(read(a.natural_support, 1024*1024, SUPPORT_SHA))
    successor = json.loads(read(a.natural_successor, 1024*1024, SUCCESSOR_SHA))
    full_top = json.loads(read(a.full_top, 1024*1024, FULL_TOP_SHA))
    known_support = _rows(support['rows'], 128)
    known_successor = _rows([row['input_ids'] for row in successor['cases']], 256)
    known_full_top = _rows([row['token_ids'] for row in full_top['cases']], 48)
    from tokenizers import Tokenizer, __version__
    if __version__ != '0.22.2':
        raise ValueError('pinned tokenizers version differs')
    tokenizer = Tokenizer.from_str(token_raw.decode('utf-8'))
    cases, selection = build_cases(tokenizer, prior_binding, prior_cases)
    known_inventory = inventory_sequences(inventory)
    rows = [tuple(case['token_ids']) for case in cases]
    proposed = [row + suffix for row in rows for suffix in ((),) + tuple((v,) for _,v in ANSWERS)]
    for label, known in (('inventory', known_inventory), ('natural support', known_support),
                         ('natural successor pool', known_successor), ('full-top', known_full_top)):
        if any(related(row, other) for row in proposed for other in known):
            raise ValueError('public prefix collision: ' + label)
    row_raw = json.dumps([list(row) for row in rows], separators=(',', ':')).encode('ascii')
    result = {
        'schema': 'distilgpt2-agreement-subject-dependence-cases-v1',
        'scope': 'Public prospective nonblind tokenizer-only inputs; no model forward.',
        'protocol_revision': PROTOCOL_REVISION, 'protocol_sha256': PROTOCOL_SHA,
        'reference_sha256': REFERENCE_SHA, 'case_preparer_sha256': sha(Path(__file__).read_bytes()),
        'publication_requirement': 'Caller independently verifies protocol/reference publication before construction.',
        'model_id': 'distilbert/distilgpt2', 'model_revision': '2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'tokenizer_sha256': TOKENIZER_SHA, 'tokenizers_version': __version__,
        'inventory_version': a.inventory_version, 'inventory_revision': a.inventory_revision,
        'inventory_sha256': a.inventory_sha256, 'inventory_sequences_checked': len(known_inventory),
        'public_checks': {'natural_support_sha256': SUPPORT_SHA, 'natural_support_rows': 128,
                          'natural_successor_sha256': SUCCESSOR_SHA, 'natural_successor_rows': 256,
                          'full_top_sha256': FULL_TOP_SHA, 'full_top_rows': 48},
        'private_status': 'Pending separate current custodian report before any forward.',
        'answer_ids': [x[1] for x in ANSWERS], 'expected_forwards': 720,
        'ordered_rows_sha256': sha(row_raw), 'selection': selection, 'cases': cases,
    }
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with Path(a.output).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short case-file write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'case_sha256': sha(data), 'rows_sha256': sha(row_raw),
                      'cases': len(cases), 'known_sequences': len(known_inventory)}, sort_keys=True))


if __name__ == '__main__':
    main()
