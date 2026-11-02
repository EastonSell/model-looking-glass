"""Extract signed case, dose and downstream tables from an already-audited run.

This is a presentation helper, not an independent model or execution audit.
It preserves all cases, both signs and nulls; conclusions remain descriptive.
"""
import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require_subnormals():
    raw = bytes.fromhex('0100000000000000')
    value = struct.unpack('<d', raw)[0]
    if not (value > 0. and struct.pack('<d', value * 1.) == raw and struct.pack('<d', float('5e-324')) == raw):
        raise ValueError('binary64 subnormal preservation is unavailable')


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty output table')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def even_norm(encoded):
    raw = base64.b64decode(encoded, validate=True)
    if not raw or len(raw) % 8:
        raise ValueError('invalid saved float64 vector')
    values = (x[0] for x in struct.iter_unpack('<d', raw))
    return math.sqrt(math.fsum(value ** 2 for value in values))


def summarize(result, journals):
    effects = {(e['recipient'], e['site'], e['mode']): e for e in result['effects']}
    expected_effects = 144 * len(journals)
    if len(effects) != expected_effects:
        raise ValueError('effect identity inventory differs')
    doses = {}
    for records in journals.values():
        for record in records:
            if record.get('event') == 'forward_complete' and record.get('dose') is not None:
                site = ('early_ingredient' if record['layer'] == 0 else
                        {7: 'late_ingredient', 1: 'late_the', 3: 'late_tasted'}[record['position']])
                key = (record['row_id'], site, record['mode'])
                if key in doses:
                    raise ValueError('duplicate dose entry')
                doses[key] = record['dose']
    if len(doses) != 128 * len(journals):
        raise ValueError('dose inventory differs')
    cells = result['comparisons']
    by_cell = {(x['recipient'], x['group']): x for x in cells}
    if len(by_cell) != 16 * len(journals):
        raise ValueError('case inventory differs')
    cases = []
    for cell in cells:
        for axis in ('lex', 'context', 'orth1', 'orth2'):
            plus, minus = (effects[cell['recipient'], cell['group'], axis + '_' + sign] for sign in ('plus', 'minus'))
            dp, dm = (doses[cell['recipient'], cell['group'], axis + '_' + sign] for sign in ('plus', 'minus'))
            value, projection = cell['axes'][axis], cell['axes'][axis]['output_projection']
            cases.append({'stem': cell['stem'], 'template': cell['template'], 'recipient': cell['recipient'],
                'direction': cell['direction'], 'site': cell['group'], 'axis': axis,
                'eligible_descriptive': cell['eligible_descriptive'], 'baseline_margin': plus['baseline_margin'],
                'plus_margin': plus['patched_margin'], 'minus_margin': minus['patched_margin'],
                'plus_effect': plus['effect'], 'minus_effect': minus['effect'],
                'plus_sweet_bits': plus['summary']['sweet_bits'], 'plus_sour_bits': plus['summary']['sour_bits'],
                'minus_sweet_bits': minus['summary']['sweet_bits'], 'minus_sour_bits': minus['summary']['sour_bits'],
                'odd': value['odd'], 'even': value['even'], 'donor_signed_odd': value['signed_odd'],
                'taste_specificity': cell['taste_specificity'], 'context_specificity': cell['context_specificity'],
                'tv_plus': value['tv_plus'], 'tv_minus': value['tv_minus'],
                'output_projection': projection['coefficient'], 'output_cosine': projection['cosine'],
                'output_norm': projection['vector_norm'], 'output_residual_norm': projection['residual_norm'],
                'requested_norm': dp['requested_norm'], 'plus_delivered_norm': dp['delivered_norm'],
                'minus_delivered_norm': dm['delivered_norm'], 'plus_relative_error': dp['relative_error'],
                'minus_relative_error': dm['relative_error'], 'recipient_norm': dp['recipient_norm'],
                'plus_to_recipient_norm': dp['dose_to_recipient_norm'],
                'minus_to_recipient_norm': dm['dose_to_recipient_norm'],
                'plus_lexical_cosine': dp['lexical_cosine'], 'minus_lexical_cosine': dm['lexical_cosine'],
                'plus_context_cosine': dp['context_cosine'], 'minus_context_cosine': dm['context_cosine']})
    downstream = []
    for item in result['propagation']:
        cell = by_cell[item['recipient'], 'early_ingredient']
        for axis in ('lex', 'context', 'orth1', 'orth2'):
            details = cell['axes'][axis]['later'][item['group']]
            downstream.append({'stem': item['stem'], 'template': item['template'], 'recipient': item['recipient'],
                'direction': item['direction'], 'later_position': item['group'], 'axis': axis,
                'projection': item[axis + '_projection'], 'cosine': item[axis + '_cosine'],
                'odd_norm': item[axis + '_norm'], 'even_norm': even_norm(details['even_float64_le_base64']),
                'clean_direction_norm': details['projection']['direction_norm'],
                'propagation_specificity': item['propagation_specificity']})
    radii = [entry['radius'] for entry in result['doses']]
    ratios = [value['dose_to_recipient_norm'] for value in doses.values() if value['dose_to_recipient_norm'] is not None]
    summary = {'forwards': result['completed_forwards'], 'stems': len(journals), 'cases': len(cells),
        'case_axis_rows': len(cases), 'downstream_axis_rows': len(downstream),
        'eligibility_descriptive': result['eligibility_descriptive'],
        'radius_min': min(radii), 'radius_max': max(radii),
        'delivered_ratio_min': min(ratios) if ratios else None, 'delivered_ratio_max': max(ratios) if ratios else None,
        'maximum_relative_dose_error': max(value['relative_error'] for value in doses.values()),
        'taste_aggregates': result['taste_aggregates'], 'propagation_aggregates': result['propagation_aggregates'],
        'scope': 'Descriptive saved-record extraction; all cases, negative values, nulls and both action signs retained.'}
    return cases, downstream, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require_subnormals()
    audit = json.loads(args.audit.read_bytes())
    if audit.get('status') != 'pass':
        raise ValueError('the independent audit is not a pass')
    if args.fixture is not None:
        if args.run_dir is not None:
            raise ValueError('choose fixture or run')
        raw = args.fixture.read_bytes()
        if audit.get('input_sha256') != sha(raw):
            raise ValueError('fixture audit binding differs')
        data = json.loads(raw)
        result, journals = data['result'], {'toy': data['events']}
    else:
        if args.run_dir is None:
            raise ValueError('run directory required')
        raw = (args.run_dir / 'result.json').read_bytes()
        if audit.get('result_sha256') != sha(raw):
            raise ValueError('result audit binding differs')
        result, journals = json.loads(raw), {}
        for name, expected in audit['journal_sha256'].items():
            data = (args.run_dir / name).read_bytes()
            if sha(data) != expected:
                raise ValueError('journal audit binding differs')
            journals[name[:-6]] = [json.loads(line) for line in data.splitlines()]
    cases, downstream, summary = summarize(result, journals)
    require_subnormals()
    args.output_dir.mkdir(parents=False, exist_ok=False)
    write_csv(args.output_dir / 'all-case-axis.csv', cases)
    write_csv(args.output_dir / 'all-downstream-axis.csv', downstream)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
                                                    encoding='utf-8')
    print(json.dumps({'case_axis_rows': len(cases), 'downstream_axis_rows': len(downstream),
                      'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
