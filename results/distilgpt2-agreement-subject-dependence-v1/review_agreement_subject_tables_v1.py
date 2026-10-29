"""Check public CSV exports against the independently replayed selected values."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MARGINS = ('present', 'past', 'the_vs_a', 'in_vs_on')
ARMS = ('clean', 'u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')
DIRECTIONS = ('u', 'v1', 'v2')
CONTRASTS = (('attractor_singular', 'none'), ('attractor_plural', 'none'),
             ('attractor_plural', 'attractor_singular'), ('mass', 'none'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def csv_file(path, fields):
    raw = path.read_bytes()
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError('CSV columns differ')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('invalid CSV row')
    return rows, sha(raw)


def equal_text(actual, expected):
    if actual != ('' if expected is None else str(expected)):
        raise ValueError('CSV identity or boolean differs')


def equal_number(actual, expected):
    if float(actual) != expected:
        raise ValueError('CSV numeric value differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--tables', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    raw = args.report.read_bytes()
    if sha(raw) != '5534757670fe16a6d751b88cc8847a7fe5f9071ac6fbc60564ba376a92e65b17':
        raise ValueError('independent report differs')
    report = json.loads(raw)
    if (report['event_sha256'] != 'f1e3acf77eff4dee33affa4ccda0b6b05add7bf3e14bdde07e96b2e9dcef234b'
            or report['forwards'] != 720 or len(report['cases']) != 80 or len(report['pairs']) != 32):
        raise ValueError('independent report dimensions differ')
    identities = ['case_id', 'frame', 'condition', 'role', 'number', 'subject_key']
    case_fields = identities + [f'{a}_{m}' for a in ARMS for m in MARGINS]
    case_fields += [f'{kind}_{d}_{m}' for kind in ('odd', 'even') for d in DIRECTIONS for m in MARGINS]
    case_rows, case_sha = csv_file(args.tables / 'case-responses.csv', case_fields)
    if len(case_rows) != 80:
        raise ValueError('case count differs')
    for row, case in zip(case_rows, report['cases']):
        for key in identities:
            equal_text(row[key], case.get(key))
        for arm in ARMS:
            for margin in MARGINS:
                equal_number(row[f'{arm}_{margin}'], case['margins'][arm][margin])
        for kind in ('odd', 'even'):
            for direction in DIRECTIONS:
                for margin in MARGINS:
                    equal_number(row[f'{kind}_{direction}_{margin}'], case[kind][direction][margin])
    pair_ids = ['frame', 'condition', 'subject_key', 'singular', 'plural', 'clean_correct']
    metric_fields = [f'{metric}_{direction}' for metric in ('k', 'g') for direction in DIRECTIONS]
    pair_rows, pair_sha = csv_file(args.tables / 'pair-contrasts.csv', pair_ids + metric_fields)
    if len(pair_rows) != 32:
        raise ValueError('pair count differs')
    pairs = {}
    for row, pair in zip(pair_rows, report['pairs']):
        for key in pair_ids:
            equal_text(row[key], pair[key])
        for metric in ('k', 'g'):
            for direction in DIRECTIONS:
                equal_number(row[f'{metric}_{direction}'], pair[metric][direction])
        key = (pair['frame'], pair['subject_key'], pair['condition'])
        if key in pairs:
            raise ValueError('duplicate pair identity')
        pairs[key] = pair
    contrast_rows, contrast_sha = csv_file(args.tables / 'within-pair-condition-contrasts.csv',
        ['frame', 'subject_key', 'contrast'] + metric_fields)
    keys = sorted({(x['frame'], x['subject_key']) for x in report['pairs']})
    expected = [(frame, subject, left, right) for frame, subject in keys for left, right in CONTRASTS]
    if len(keys) != 8 or len(contrast_rows) != len(expected) or len(expected) != 32:
        raise ValueError('contrast count differs')
    for row, (frame, subject, left, right) in zip(contrast_rows, expected):
        equal_text(row['frame'], frame)
        equal_text(row['subject_key'], subject)
        equal_text(row['contrast'], left + ' - ' + right)
        a, b = pairs[(frame, subject, left)], pairs[(frame, subject, right)]
        for metric in ('k', 'g'):
            for direction in DIRECTIONS:
                equal_number(row[f'{metric}_{direction}'], a[metric][direction] - b[metric][direction])
    result = {'status': 'pass', 'independent_report_sha256': sha(raw), 'case_rows': len(case_rows),
              'pair_rows': len(pair_rows), 'within_pair_condition_contrasts': len(contrast_rows),
              'case_responses_sha256': case_sha, 'pair_contrasts_sha256': pair_sha,
              'within_pair_condition_contrasts_sha256': contrast_sha,
              'scope': 'CSV mapping and arithmetic from saved selected-logit report; no model execution or full-vocabulary validation.'}
    encoded = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as f:
        f.write(encoded)
    print(json.dumps({'sha256': sha(encoded), 'bytes': len(encoded)}))


if __name__ == '__main__':
    main()
