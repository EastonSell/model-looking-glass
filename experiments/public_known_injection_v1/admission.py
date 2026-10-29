"""Model-free admission against a hash-pinned public reservation."""
import hashlib
import json
from pathlib import Path


def related(left, right):
    return left[:min(len(left), len(right))] == right[:min(len(left), len(right))]


def verify_inventory(inventory_path, expected_sha, cases):
    data = Path(inventory_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('public inventory digest differs')
    payload = json.loads(data)
    if payload.get('version') != 'distilgpt2-development-exposures-v34' or len(cases) != 16:
        raise ValueError('unexpected reservation version or case count')
    records = payload['records']
    ids = [record['record_id'] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate public record identity')
    own = {'planned-' + case['case_id']: case for case in cases}
    if len(own) != 16:
        raise ValueError('duplicate own case identity')
    seen, foreign = set(), []
    for record in records:
        record_id = record['record_id']
        base = tuple(record['token_ids'])
        if record_id in own:
            case = own[record_id]
            if (record['category'] != 'planned' or list(base) != case['token_ids']
                    or record.get('case_role') != case['split']
                    or record.get('candidate_token_ids') != [318, 389]):
                raise ValueError('own reservation differs or is not planned')
            seen.add(record_id)
            continue
        foreign.append(base)
        answers = list(record.get('candidate_token_ids', []))
        if 'target_token_id' in record:
            answers.append(record['target_token_id'])
        for group in record.get('candidate_token_groups', {}).values():
            answers.extend(group)
        if any(type(token) is not int or token < 0 for token in answers):
            raise ValueError('invalid public continuation')
        foreign.extend(base + (token,) for token in answers)
    if seen != set(own):
        raise ValueError('own reservation missing')
    rows = [tuple(case['token_ids']) for case in cases]
    if any(related(a, b) for index, a in enumerate(rows) for b in rows[index + 1:]):
        raise ValueError('own inputs share a full-row prefix')
    proposals = [sequence for row in rows for sequence in (row, row + (318,), row + (389,))]
    if any(related(proposal, other) for proposal in proposals for other in foreign):
        raise ValueError('public foreign input/continuation conflict')
    return {'reservation_sha256': expected_sha, 'own_planned_rows': len(seen),
            'proposal_sequences': len(proposals), 'foreign_sequences': len(foreign),
            'bidirectional_pairs': len(proposals) * len(foreign), 'conflicts': 0,
            'scope': 'recorded public inventory only; no protected or unknown exposure attestation'}
