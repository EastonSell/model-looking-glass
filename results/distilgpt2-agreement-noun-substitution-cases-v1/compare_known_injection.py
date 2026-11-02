"""Compare fixed public study rows with a pending tokenizer-only public bundle."""
import hashlib
import json
import sys
from pathlib import Path


def load(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('input hash mismatch')
    return json.loads(data)


def related(left, right):
    return left[:min(len(left), len(right))] == right[:min(len(left), len(right))]


def main(case_path, pending_path, output_path):
    case_sha = 'f1e1043fc302e701b1224248122efc7f79938665ce12cd48ebb7dd713b353789'
    pending_sha = '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe'
    cases, pending = load(case_path, case_sha)['cases'], load(pending_path, pending_sha)
    if len(cases) != 16 or len(pending['cases']) != 54 or pending['answer_ids'] != [318, 389, 373, 547, 262, 257, 287, 319]:
        raise ValueError('pending case shape changed')
    proposed, other = [], []
    for item in cases:
        base = tuple(item['token_ids'])
        proposed.extend((base, base + (318,), base + (389,)))
    for item in pending['cases']:
        base = tuple(item['token_ids'])
        other.append(base)
        other.extend(base + (token,) for token in pending['answer_ids'])
    if any(related(a, b) for a in proposed for b in other):
        raise ValueError('known input/continuation prefix conflict')
    report = {'schema': 'known-injection-pending-public-comparison-v1',
              'case_sha256': case_sha, 'pending_sha256': pending_sha,
              'proposed_sequences': len(proposed), 'pending_sequences': len(other),
              'bidirectional_pairs': len(proposed) * len(other), 'conflicts': 0,
              'scope': 'tokenizer-only pending bundle; no protected split or execution attestation'}
    data = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode('utf-8')
    with Path(output_path).open('xb') as out:
        out.write(data)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main(*sys.argv[1:])
