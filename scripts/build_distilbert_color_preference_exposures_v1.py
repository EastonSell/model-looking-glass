"""Record the fixed preference-case inputs as planned, not observed."""
import argparse
import hashlib
import json
from pathlib import Path

CASE_SHA256 = 'f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10'
COMPARISON_SHA256 = '6c29d8875238448a8fc20887832b02d058b6f76fcab0a41fe6385050e932a1c3'
PROTOCOL_COMMIT = 'd94594f69426aa5e41a7cf90acd1e98be42f9087'


def read_pinned(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('input bytes differ from the frozen identity')
    return json.loads(raw)


def build(case_path, comparison_path):
    case = read_pinned(case_path, CASE_SHA256)
    comparison = read_pinned(comparison_path, COMPARISON_SHA256)
    if (case['protocol_commit'] != PROTOCOL_COMMIT or len(case['cases']) != 37 or
            comparison['case_sha256'] != CASE_SHA256 or comparison['new_rows'] != 37 or
            comparison['full_or_sep_removed_prefix_collisions'] != 0):
        raise ValueError('case or comparison identity differs')
    records = []
    for row in case['cases']:
        records.append({'record_id': 'planned-color-preference-v2-' + row['id'],
                        'case_id': row['id'], 'role': row['role'], 'object': row['object'],
                        'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
                        'mask_index': row['mask_index'], 'status': 'planned'})
    if len({row['record_id'] for row in records}) != 37:
        raise ValueError('duplicate case ID')
    return {'schema': 'distilbert-color-preference-public-exposures-v1',
            'protocol_commit': PROTOCOL_COMMIT,
            'case_path': 'experiments/bert_color_preference_public_cases_v2.json',
            'case_sha256': CASE_SHA256,
            'comparison_path': 'experiments/bert_color_preference_public_comparison_v2.json',
            'comparison_sha256': COMPARISON_SHA256, 'planned_rows': 37,
            'records': records,
            'scope': ('Planned public inputs only; this does not record a model forward, '
                      'prove unknown or private history, or independently authorize execution.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = build(args.cases, args.comparison)
    with args.output.open('xb') as stream:
        stream.write((json.dumps(document, sort_keys=True, indent=2) + '\n').encode())


if __name__ == '__main__':
    main()
