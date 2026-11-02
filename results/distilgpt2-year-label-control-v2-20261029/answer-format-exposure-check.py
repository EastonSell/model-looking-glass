"""Compare two fixed public case sets; no model or protected data is read."""
import argparse
import hashlib
import json
from pathlib import Path

YEAR_SHA = '07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'
ANSWER_SHA = 'cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a'
GROUPS = ('middle_ids', 'above_ids', 'local_below_ids', 'local_above_ids')


def read(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('public case bytes differ')
    return json.loads(raw)


def row(value):
    if type(value) is not list or not value or any(type(x) is not int or x < 0 for x in value):
        raise ValueError('invalid token row')
    return tuple(value)


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year-cases', type=Path, required=True)
    parser.add_argument('--answer-cases', type=Path, required=True)
    args = parser.parse_args()
    years = read(args.year_cases, YEAR_SHA)['cases']
    answers = read(args.answer_cases, ANSWER_SHA)['cases']
    if len(years) != 48 or [x['family_index'] for x in years] != [i // 12 for i in range(48)]:
        raise ValueError('year case ordering differs')
    if len(answers) != 64 or len({x['case_id'] for x in answers}) != 64:
        raise ValueError('answer case identities differ')
    selected = years[12:]
    if len({x['case_id'] for x in selected}) != 36:
        raise ValueError('selected year identities differ')
    year_rows = []
    for case in selected:
        base = row(case['input_ids'])
        year_rows.append(base)
        for name in GROUPS:
            if len(case[name]) != 3:
                raise ValueError('candidate group length differs')
            for token in case[name]:
                if type(token) is not int or token < 0:
                    raise ValueError('invalid year candidate')
                year_rows.append(base + (token,))
    answer_rows = []
    for case in answers:
        base = row(case['token_ids'])
        answer_rows.append(base)
        labels = case['labels']
        if set(labels) != {'factual', 'counterfactual', 'madrid', 'stockholm'}:
            raise ValueError('answer candidate labels differ')
        for label in labels.values():
            token = label['token_id']
            if type(token) is not int or token < 0 or row(label['full_token_ids']) != base + (token,):
                raise ValueError('answer continuation differs')
            answer_rows.append(base + (token,))
    if len(year_rows) != 468 or len(answer_rows) != 320:
        raise ValueError('comparison size differs')
    if any(related(a, b) for a in year_rows for b in answer_rows):
        raise ValueError('a public full-row prefix relation exists')
    print(json.dumps({'schema': 'year-label-v2-answer-format-public-check-v1',
                      'year_case_sha256': YEAR_SHA, 'answer_case_sha256': ANSWER_SHA,
                      'year_inputs': len(selected), 'year_input_and_continuation_entries': len(year_rows),
                      'answer_inputs': len(answers), 'answer_input_and_continuation_entries': len(answer_rows),
                      'bidirectional_prefix_comparisons': len(year_rows) * len(answer_rows),
                      'relations': 0,
                      'scope': 'Public cases and listed next-token continuations only; no protected or unknown outside exposure is checked.'},
                     sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
