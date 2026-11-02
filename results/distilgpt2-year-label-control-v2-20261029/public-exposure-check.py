"""Check selected already-exposed year/label rows against a public inventory.

No model or protected rows are read. This checks known public records only.
"""
import argparse
import hashlib
import json
from pathlib import Path

CASE_SHA = '07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'
SUCCESSOR_SHA = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
GROUPS = ('middle_ids', 'above_ids', 'local_below_ids', 'local_above_ids')


def read(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('public input hash differs')
    return json.loads(raw)


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def tokens(value):
    if type(value) is not list or not value or any(type(x) is not int or x < 0 for x in value):
        raise ValueError('invalid public token row')
    return tuple(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-version', required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--public-successor', type=Path, required=True)
    args = parser.parse_args()
    inventory = read(args.inventory, args.inventory_sha256)
    data = read(args.cases, CASE_SHA)
    successor = read(args.public_successor, SUCCESSOR_SHA)
    if inventory.get('version') != args.inventory_version or type(inventory.get('records')) is not list:
        raise ValueError('inventory version or records differ')
    rows = data['cases']
    if len(rows) != 48 or [x['family_index'] for x in rows] != [i//12 for i in range(48)]:
        raise ValueError('case ordering differs')
    selected = rows[12:]
    if len(selected) != 36 or len({x['case_id'] for x in selected}) != 36:
        raise ValueError('selected identities differ')
    own_ids, statuses = set(), {}
    for case in selected:
        names = {'planned-' + case['case_id'], 'observed-' + case['case_id']}
        matches = [record for record in inventory['records'] if record.get('record_id') in names]
        if len(matches) != 1:
            raise ValueError('own record is missing or repeated')
        record = matches[0]
        valid = ((record['record_id'].startswith('planned-') and record.get('category') == 'planned')
                 or (record['record_id'].startswith('observed-') and record.get('category') == 'observed-saved'))
        if (not valid or tokens(record['token_ids']) != tokens(case['input_ids'])
                or record.get('candidate_token_groups') != {k: case[k] for k in GROUPS}):
            raise ValueError('own record or candidate groups differ')
        own_ids.add(record['record_id'])
        statuses[record['category']] = statuses.get(record['category'], 0) + 1

    candidate_rows = []
    for case in selected:
        row = tokens(case['input_ids'])
        candidate_rows.append(row)
        for group in GROUPS:
            for token in case[group]:
                if type(token) is not int or token < 0:
                    raise ValueError('invalid selected candidate')
                candidate_rows.append(row + (token,))
    foreign = [record for record in inventory['records'] if record.get('record_id') not in own_ids]
    foreign_rows = []
    for record in foreign:
        row = tokens(record['token_ids'])
        foreign_rows.append(row)
        target = record.get('target_token_id')
        if target is not None:
            if type(target) is not int or target < 0:
                raise ValueError('invalid recorded target')
            foreign_rows.append(row + (target,))
        for token in record.get('candidate_token_ids', []):
            if type(token) is not int or token < 0:
                raise ValueError('invalid recorded candidate')
            foreign_rows.append(row + (token,))
        groups = record.get('candidate_token_groups', {})
        if type(groups) is not dict:
            raise ValueError('invalid recorded candidate groups')
        for values in groups.values():
            for token in values:
                if type(token) is not int or token < 0:
                    raise ValueError('invalid recorded candidate group')
                foreign_rows.append(row + (token,))
    if any(related(a, b) for a in candidate_rows for b in foreign_rows):
        raise ValueError('selected input or continuation has a known foreign relation')
    input_rows = [tokens(row['input_ids']) for row in selected]
    if any(related(a, b) for i, a in enumerate(input_rows) for b in input_rows[i+1:]):
        raise ValueError('selected public inputs have a mutual relation')

    successor_rows = [tokens(row['input_ids']) for row in successor['cases']]
    if len(successor_rows) != 256 or len(set(successor_rows)) != 256:
        raise ValueError('public successor pool differs')
    extensions = successor_rows + [row+(token,) for row in successor_rows for token in (318, 389, 373, 547)]
    if any(related(a, b) for a in candidate_rows for b in extensions):
        raise ValueError('selected input or continuation relates to the public successor pool')
    result = {'schema': 'year-label-v2-public-exposure-check-v1',
              'inventory_version': args.inventory_version, 'inventory_sha256': args.inventory_sha256,
              'cases_sha256': CASE_SHA, 'successor_inputs_sha256': SUCCESSOR_SHA,
              'selected_inputs': len(selected), 'selected_input_and_continuation_entries': len(candidate_rows),
              'own_record_statuses': statuses, 'foreign_records': len(foreign),
              'foreign_input_and_extension_entries': len(foreign_rows),
              'public_successor_input_and_extension_entries': len(extensions),
              'foreign_prefix_relations': 0, 'selected_input_prefix_relations': 0,
              'public_successor_prefix_relations': 0,
              'scope': 'Public inputs and recorded extensions only; own rows already observed in v1. No protected split or unknown outside/pretraining exposure is checked.'}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
