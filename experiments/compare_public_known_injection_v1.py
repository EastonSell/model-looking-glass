"""No-model, public-only complete-row prefix comparison for fixed cases."""
import hashlib
import json
import sys
from pathlib import Path


def load(path, expected):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError('public input digest mismatch')
    return json.loads(data)


def row(value):
    if (type(value) is not list or not value
            or any(type(token) is not int or token < 0 for token in value)):
        raise ValueError('invalid public token row')
    return tuple(value)


def related(a, b):
    return a[:min(len(a), len(b))] == b[:min(len(a), len(b))]


def main(case_path, inventory_path, original_path, successor_path, destination):
    cases = load(case_path, 'f1e1043fc302e701b1224248122efc7f79938665ce12cd48ebb7dd713b353789')['cases']
    inventory = load(inventory_path, '4a4ed0127d18a7c5a8f6472af619b82895cb4efa6c07314a48a463510b1878b2')
    original = load(original_path, '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6')
    successor = load(successor_path, '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4')
    if len(cases) != 16 or len(inventory['records']) != 981:
        raise ValueError('unexpected public inventory shape')
    proposed = []
    for case in cases:
        base = row(case['token_ids'])
        if case['candidate_token_ids'] != [318, 389]:
            raise ValueError('candidate continuation mismatch')
        proposed.extend((base, base + (318,), base + (389,)))
    for i in range(len(cases)):
        for j in range(i):
            if related(row(cases[i]['token_ids']), row(cases[j]['token_ids'])):
                raise ValueError('proposed inputs share a complete-row prefix')
    foreign = []
    for entry in inventory['records']:
        base = row(entry['token_ids'])
        foreign.append(base)
        tokens = []
        if 'target_token_id' in entry:
            tokens.append(entry['target_token_id'])
        tokens.extend(entry.get('candidate_token_ids', []))
        for group in entry.get('candidate_token_groups', {}).values():
            tokens.extend(group)
        if any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('unsupported public continuation metadata')
        foreign.extend(base + (t,) for t in tokens)
    for raw in original['rows']:
        base = row(raw)
        foreign.extend((base, *(base + (t,) for t in (318, 389, 373, 547))))
    for entry in successor['cases']:
        base = row(entry['input_ids'])
        foreign.extend((base, *(base + (t,) for t in (318, 389, 373, 547))))
    conflicts = [(i, j) for i, a in enumerate(proposed) for j, b in enumerate(foreign)
                 if related(a, b)]
    if conflicts:
        raise ValueError(f'public full-row prefix conflicts: {conflicts}')
    report = {'schema': 'public-known-injection-prefix-comparison-v1',
              'case_sha256': 'f1e1043fc302e701b1224248122efc7f79938665ce12cd48ebb7dd713b353789',
              'inventory_kind': 'preliminary v32 candidate, not a merge/currentness claim',
              'inventory_sha256': '4a4ed0127d18a7c5a8f6472af619b82895cb4efa6c07314a48a463510b1878b2',
              'original_public_support_sha256': '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6',
              'successor_public_support_sha256': '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4',
              'proposed_sequences': len(proposed), 'foreign_sequences': len(foreign),
              'bidirectional_pairs': len(proposed) * len(foreign),
              'within_proposed_input_pairs': 120, 'conflicts': 0,
              'scope': 'public data only; no protected split, unrecorded history or pretraining checked'}
    data = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode('utf-8')
    with Path(destination).open('xb') as out:
        out.write(data)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main(*sys.argv[1:])
