"""Create the planned public-input reservation from fixed case/scan bytes."""
import argparse
import hashlib
import json
from pathlib import Path

CASE_SHA = '284fc9adc0cc3caa52ea5d122c1f08db50c8dda2d881297e2dcd1f2229cb6386'
COMPARISON_SHA = 'f030cf522295013708e98a7e5b83dc9bebf94d4cec5b492f970c3d4380c314fa'


def project(rows):
    if len(rows) != 55 or len({row['id'] for row in rows}) != 55:
        raise ValueError('expected 55 distinct planned case IDs')
    return [{'record_id': 'planned-antonym-preference-v1-' + row['id'],
             'case_id': row['id'], 'object': row['object'], 'role': row['role'],
             'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
             'mask_index': row['mask_index'], 'status': 'planned'} for row in rows]


def build(case_raw, comparison_raw):
    if hashlib.sha256(case_raw).hexdigest() != CASE_SHA or hashlib.sha256(comparison_raw).hexdigest() != COMPARISON_SHA:
        raise ValueError('fixed case or public comparison bytes differ')
    cases, check = json.loads(case_raw), json.loads(comparison_raw)
    if check.get('candidate_sha256') != CASE_SHA or check.get('related_pairs') != [] or check.get('candidate_rows') != 55:
        raise ValueError('public comparison does not clear the fixed cases')
    return {'schema': 'distilbert-antonym-preference-public-exposures-v1',
            'protocol_commit': cases['protocol_commit'],
            'case_path': 'experiments/bert_antonym_preference_public_cases_v1.json',
            'case_sha256': CASE_SHA,
            'comparison_path': 'experiments/bert_antonym_preference_public_comparison_v1.json',
            'comparison_sha256': COMPARISON_SHA, 'planned_rows': 55,
            'records': project(cases['cases']),
            'scope': ('Planned public inputs only; no model forward, complete exposure history, '
                      'private data or pretraining novelty is established.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = (json.dumps(build(args.cases.read_bytes(), args.comparison.read_bytes()),
                      sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as out:
        if out.write(raw) != len(raw):
            raise OSError('short exposure write')
    print('exposure SHA-256', hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
