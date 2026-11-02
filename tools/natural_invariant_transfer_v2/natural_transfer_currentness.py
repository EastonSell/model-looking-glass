"""Compare the fixed public successor cases with foreign public inventory rows."""
import argparse
import hashlib
import json
from pathlib import Path


CASE_SHA = '188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b'
ANSWER_IDS = (318, 389, 373, 547, 262, 257, 287, 319)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def row(values):
    if (type(values) not in (list, tuple) or not values
            or any(type(value) is not int or not 0 <= value < 50257 for value in values)):
        raise ValueError('invalid public token row')
    return tuple(values)


def expanded(record):
    base = row(record['token_ids'])
    bases = [base]
    if 'attention_mask' in record:
        mask = record['attention_mask']
        if (type(mask) is not list or len(mask) != len(base)
                or any(type(x) is not int or x not in (0, 1) for x in mask)):
            raise ValueError('invalid attention mask')
        active = tuple(x for x, keep in zip(base, mask) if keep)
        if active and active != base:
            bases.append(active)
    candidates = []
    if 'target_token_id' in record:
        candidates.append(record['target_token_id'])
    candidates.extend(record.get('candidate_token_ids', ()))
    groups = record.get('candidate_token_groups', {})
    if type(groups) is not dict:
        raise ValueError('invalid candidate groups')
    for group in groups.values():
        candidates.extend(group)
    if any(type(x) is not int or not 0 <= x < 50257 for x in candidates):
        raise ValueError('invalid continuation')
    result = []
    for value in bases:
        result.append(value)
        result.extend(value + (x,) for x in candidates)
    if 'scored_prefix_token_ids' in record:
        scored = row(record['scored_prefix_token_ids'])
        choices = record.get('scored_candidate_token_ids', ())
        if any(type(x) is not int or not 0 <= x < 50257 for x in choices):
            raise ValueError('invalid scored continuation')
        result.append(scored)
        result.extend(scored + (x,) for x in choices)
    return result


def compare(case_bytes, inventory_bytes):
    if digest(case_bytes) != CASE_SHA:
        raise ValueError('fixed public case bytes differ')
    cases = json.loads(case_bytes)['cases']
    inventory = json.loads(inventory_bytes)
    records = inventory['records']
    if type(cases) is not list or len(cases) != 68 or type(records) is not list:
        raise ValueError('invalid case or inventory size')
    ids = {'planned-' + c['case_id'] for c in cases}
    if len(ids) != 68:
        raise ValueError('duplicate own case ID')
    record_ids = [r['record_id'] for r in records]
    if len(set(record_ids)) != len(record_ids) or not ids <= set(record_ids):
        raise ValueError('missing or duplicate reserved records')
    indexed = {record['record_id']: record for record in records}
    for case in cases:
        record = indexed['planned-' + case['case_id']]
        if (row(record['token_ids']) != row(case['token_ids'])
                or record.get('public_text') != case['text'] or record.get('case_role') != case['role']
                or type(record.get('candidate_token_ids')) is not list
                or any(type(x) is not int for x in record['candidate_token_ids'])
                or tuple(record['candidate_token_ids']) != ANSWER_IDS
                or any(name in record for name in ('attention_mask', 'target_token_id', 'candidate_token_groups',
                                                   'scored_prefix_token_ids', 'scored_candidate_token_ids'))
                or record.get('category') not in ('planned', 'observed-reported', 'observed-saved')):
            raise ValueError('own reservation identity differs')
    proposals = []
    for case in cases:
        base = row(case['token_ids'])
        proposals.append(base)
        proposals.extend(base + (token,) for token in ANSWER_IDS)
    foreign = []
    for record in records:
        if record['record_id'] not in ids:
            foreign.extend(expanded(record))
    # Exact complete-row prefix relations, in both directions. Own reserved
    # rows are exempt by their fixed IDs only, never by token similarity.
    relations = 0
    examples = []
    for own in proposals:
        for other in foreign:
            if own[:len(other)] == other or other[:len(own)] == own:
                relations += 1
                if len(examples) < 4:
                    examples.append({'own_length': len(own), 'foreign_length': len(other)})
    return {'schema': 'natural-invariant-transfer-currentness-v1',
            'case_sha256': digest(case_bytes), 'inventory_sha256': digest(inventory_bytes),
            'case_count': len(cases), 'own_reservation_count': len(ids),
            'own_sequences': len(proposals), 'foreign_sequences': len(foreign),
            'bidirectional_pairs': len(proposals) * len(foreign), 'relations': relations,
            'examples': examples,
            'scope': 'Known public inventory only; no protected, unknown or pretraining exposure is established.'}


def compare_pending(case_bytes, bundle_bytes):
    """Compare a known fixed public pending bundle and its declared continuations."""
    if digest(case_bytes) != CASE_SHA:
        raise ValueError('fixed public case bytes differ')
    cases = json.loads(case_bytes)['cases']
    bundle = json.loads(bundle_bytes)
    schema = bundle.get('schema')
    foreign = []
    if schema == 'france-full-top-fixed-cases-v3':
        if (type(bundle.get('rows')) is not list or len(bundle['rows']) != 27
                or type(bundle.get('projection_basis')) is not list or len(bundle['projection_basis']) != 4):
            raise ValueError('unexpected France case count')
        for item in bundle['rows'] + bundle['projection_basis']:
            base = row(item['token_ids'])
            mask = item['attention_mask']
            if type(mask) is not list or len(mask) != len(base) or any(type(x) is not int or x != 1 for x in mask):
                raise ValueError('unexpected France attention mask')
            foreign.append(base)
            continuations = item['continuations']
            if type(continuations) is not list:
                raise ValueError('invalid France continuations')
            for continuation in continuations:
                token = continuation['token_id']
                candidate = row(continuation['appended_token_ids'])
                if type(token) is not int or candidate != base + (token,):
                    raise ValueError('invalid France continuation')
                foreign.append(candidate)
        if len(foreign) != 73:
            raise ValueError('unexpected France continuation count')
    elif schema == 'distilgpt2-paired-continuation-cases-v1':
        rows = bundle.get('cases')
        if type(rows) is not list or len(rows) != 64:
            raise ValueError('unexpected copying case count')
        for item in rows:
            base = row(item['token_ids'])
            choices = (item['candidate_f_id'], item['candidate_g_id'])
            if any(type(x) is not int or not 0 <= x < 50257 for x in choices) or choices[0] == choices[1]:
                raise ValueError('invalid copying continuations')
            foreign.append(base)
            foreign.extend(base + (x,) for x in choices)
    else:
        raise ValueError('unsupported pending input schema')
    proposals = []
    for case in cases:
        base = row(case['token_ids'])
        proposals.append(base)
        proposals.extend(base + (token,) for token in ANSWER_IDS)
    relations = sum(own[:len(other)] == other or other[:len(own)] == own
                    for own in proposals for other in foreign)
    return {'schema': 'natural-invariant-transfer-pending-comparison-v1',
            'case_sha256': digest(case_bytes), 'bundle_sha256': digest(bundle_bytes),
            'bundle_schema': bundle['schema'], 'own_sequences': len(proposals),
            'foreign_sequences': len(foreign), 'bidirectional_pairs': len(proposals)*len(foreign),
            'relations': relations,
            'scope': 'Public fixed pending bundle only; no model use or private exposure is established.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = args.inventory.read_bytes()
    if digest(inventory) != args.inventory_sha256:
        raise ValueError('inventory SHA-256 differs')
    report = compare(args.cases.read_bytes(), inventory)
    data = (json.dumps(report, sort_keys=True, indent=2) + '\n').encode()
    with args.output.open('xb') as output:
        if output.write(data) != len(data):
            raise OSError('short report write')
    print(json.dumps({'report_sha256': digest(data), 'relations': report['relations'],
                      'bidirectional_pairs': report['bidirectional_pairs']}, sort_keys=True))
    return 0 if report['relations'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
