"""Export fixed case and pair tables from a pinned independent selected replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected-report', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    raw = args.selected_report.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_sha256:
        raise ValueError('independent selected replay differs from the supplied pin')
    data = json.loads(raw)
    if data.get('forwards') != 720 or len(data.get('cases', [])) != 80 or len(data.get('pairs', [])) != 32:
        raise ValueError('selected replay dimensions differ')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'case-responses.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['case_id', 'frame', 'condition', 'role', 'number', 'subject_key']
        fields += [f'{arm}_{margin}' for arm in ('clean', 'u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')
                   for margin in ('present', 'past', 'the_vs_a', 'in_vs_on')]
        fields += [f'{kind}_{direction}_{margin}' for kind in ('odd', 'even') for direction in ('u', 'v1', 'v2')
                   for margin in ('present', 'past', 'the_vs_a', 'in_vs_on')]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for case in data['cases']:
            row = {key: case.get(key) for key in fields[:6]}
            for arm in ('clean', 'u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus'):
                for margin, value in case['margins'][arm].items():
                    row[f'{arm}_{margin}'] = value
            for kind in ('odd', 'even'):
                for direction in ('u', 'v1', 'v2'):
                    for margin, value in case[kind][direction].items():
                        row[f'{kind}_{direction}_{margin}'] = value
            writer.writerow(row)
    with (args.output_dir / 'pair-contrasts.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['frame', 'condition', 'subject_key', 'singular', 'plural', 'clean_correct',
                  'k_u', 'k_v1', 'k_v2', 'g_u', 'g_v1', 'g_v2']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for pair in data['pairs']:
            row = {key: pair[key] for key in fields[:6]}
            row.update({f'{metric}_{direction}': pair[metric][direction]
                        for metric in ('k', 'g') for direction in ('u', 'v1', 'v2')})
            writer.writerow(row)
    pairs = {(x['frame'], x['subject_key'], x['condition']): x for x in data['pairs']}
    with (args.output_dir / 'within-pair-condition-contrasts.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['frame', 'subject_key', 'contrast', 'k_u', 'k_v1', 'k_v2', 'g_u', 'g_v1', 'g_v2']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for frame, subject in sorted({(x['frame'], x['subject_key']) for x in data['pairs']}):
            for left, right in (('attractor_singular', 'none'), ('attractor_plural', 'none'),
                                ('attractor_plural', 'attractor_singular'), ('mass', 'none')):
                a, b = pairs[(frame, subject, left)], pairs[(frame, subject, right)]
                row = {'frame': frame, 'subject_key': subject, 'contrast': left + ' - ' + right}
                row.update({f'{metric}_{direction}': a[metric][direction] - b[metric][direction]
                            for metric in ('k', 'g') for direction in ('u', 'v1', 'v2')})
                writer.writerow(row)


if __name__ == '__main__':
    main()
