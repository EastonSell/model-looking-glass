"""Compare the frozen public case bundle to the public short-service rows."""
import argparse
import hashlib
import json
from pathlib import Path

CASE_SHA = '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe'
SERVICE_SHA = '47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c'


def read(path, expected):
    with Path(path).open('rb') as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('public input digest or size differs')
    return json.loads(raw)


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def compare(case_doc, service_doc):
    rows, answers = case_doc['cases'], case_doc['answer_ids']
    if len(rows) != 54 or answers != [318, 389, 373, 547, 262, 257, 287, 319]:
        raise ValueError('unexpected fixed case count or answers')
    proposed = [tuple(row['token_ids']) + suffix for row in rows
                for suffix in ((),) + tuple((answer,) for answer in answers)]
    service = set()
    for pair in service_doc['comparisons']:
        for name in ('reference', 'service'):
            item = pair[name]
            ids, mask = item['input_ids'], item['attention_mask']
            if (type(ids) is not list or type(mask) is not list or len(ids) != len(mask)
                    or not ids or any(type(x) is not int or not 0 <= x < 50257 for x in ids)
                    or any(type(x) is not int or x not in (0, 1) for x in mask)):
                raise ValueError('invalid service input or mask')
            count = sum(mask)
            if mask != [1] * count + [0] * (len(mask) - count):
                raise ValueError('noncontiguous service mask')
            service.add(tuple(ids))
            service.add(tuple(ids[:count]))
    if len(proposed) != 486 or len(service) != 6:
        raise ValueError('unexpected proposed or service sequence count')
    comparisons = sum(1 for row in proposed for other in service if related(row, other))
    return {'schema': 'agreement-substitution-short-service-comparison-v1',
            'case_sha256': CASE_SHA, 'service_sha256': SERVICE_SHA,
            'proposed_sequences': len(proposed), 'unique_active_or_full_service_rows': len(service),
            'bidirectional_comparisons': len(proposed) * len(service),
            'relations': comparisons, 'status': 'clear' if comparisons == 0 else 'collision',
            'scope': 'Public token rows only; service active/padded pairs have intentional internal prefixes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--service', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(read(args.cases, CASE_SHA), read(args.service, SERVICE_SHA))
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode('ascii')
    with Path(args.output).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short write')
    print(hashlib.sha256(raw).hexdigest())
    if result['status'] != 'clear':
        raise SystemExit('public prefix collision')


if __name__ == '__main__':
    main()
