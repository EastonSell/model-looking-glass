"""Audit closed copying intervention arm order and summaries, without full logits.

This checks recorded comparison metadata and baseline classifications; it does
not authenticate execution, reconstruct logits, or certify intervention landing.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

BASELINE_RESULT_SHA = 'fcee5b994d9f8d70c0d9c35eae62da8827f5cb92008c9db709c8beb9a733aa99'
BASELINE_AUDIT_SHA = '584c0ccf1ad11c5ad4d857bfe73c3a5d70b138fd1c85b5bee6aac6152ba6b186'
CASES_SHA = '96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734'
GROUPS = list(range(2, 25, 2))
HEADS = ('h2', 'h10', 'h7')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum, expected=None):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or (expected is not None and digest(raw) != expected):
        raise ValueError('file size or digest differs')
    return raw


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError('nonfinite JSON')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('invalid numeric field')
    return float(value)


def close(actual, expected):
    return math.isclose(number(actual), float(expected), rel_tol=1e-13, abs_tol=1e-15)


def audit(data, support):
    if (data.get('schema') != 'distilgpt2-copy-behavior-intervention-result-v1'
            or data.get('status') != 'complete' or set(support) != set(GROUPS)):
        raise ValueError('result or support inventory differs')
    records, families = data['records'], data['families']
    if type(records) is not list or type(families) is not list or len(families) != 12:
        raise ValueError('fixed family inventory differs')
    cursor, planned_eligible, reports, eligible = 0, 0, [], []
    for group, family in zip(GROUPS, families, strict=True):
        if (family['group'] != group or family['baseline_status'] != support[group]
                or support[group] not in ('supported', 'unsupported')):
            raise ValueError('family baseline binding differs')
        plan, assessment, summary = family['output_plan'], family['output_assessment'], family['summary']
        if plan['status'] not in ('eligible', 'dose-ineligible'):
            raise ValueError('unknown plan status')
        planned = plan['status'] == 'eligible'
        planned_eligible += int(planned)
        expected = [(v, kind) for v in range(8) for kind in ('clean', 'noop')]
        for variant in range(8):
            order = ('out_h2', 'out_h10', 'out_h7')
            shift = (8 * (group - 1) + variant) % 3
            if planned:
                expected.extend((variant, kind) for kind in order[shift:] + order[:shift])
            expected.append((variant, 'clean_after'))
        rows = records[cursor:cursor + len(expected)]; cursor += len(expected)
        if (len(rows) != len(expected) or any(row['group'] != group for row in rows)
                or [(r['variant'], r['action']) for r in rows] != expected):
            raise ValueError('fixed arm schedule differs')
        variants = [{r['action']: r for r in rows if r['variant'] == v} for v in range(8)]
        for item in variants:
            identities = {(r['case_id'], r['target_id']) for r in item.values()}
            if len(identities) != 1:
                raise ValueError('case identity changed within a prompt')
        actual_status = assessment['status']
        if (actual_status not in ('eligible', 'dose-ineligible')
                or (not planned and actual_status != 'dose-ineligible')
                or summary['dose_status'] != actual_status
                or summary['baseline_status'] != support[group]
                or summary['group'] != group
                or number(summary['q_out']) != number(plan['q_out'])):
            raise ValueError('dose or summary status differs')
        report = {'group': group, 'baseline_status': support[group], 'plan_status': plan['status'],
                  'dose_status': actual_status, 'dose_reason': assessment.get('reason'),
                  'q_out': plan['q_out'], 'recorded_actions': len(rows)}
        if actual_status == 'dose-ineligible':
            if (summary['status'] != 'dose-ineligible' or summary['contrast'] is not None
                    or summary['contrast_terms'] is not None or summary['contrast_sign'] is not None):
                raise ValueError('dose-ineligible family was promoted')
            reports.append(report); continue
        if not planned or summary['status'] != 'evaluable':
            raise ValueError('eligible family lacks active arms')
        changes, terms, head_d = {}, {}, {}
        for head in HEADS:
            changes[head] = [number(v['out_' + head]['comparison']['target_log_probability_delta']) for v in variants]
            terms[head] = []
            for rotation in range(2):
                row = changes[head][4*rotation:4*rotation+4]
                terms[head].extend((row[0], row[0], row[0], -row[1], -row[2], -row[3]))
                exact = (3*Fraction.from_float(row[0]) - sum((Fraction.from_float(x) for x in row[1:]), Fraction())) / 3
                if not close(summary['rotations'][rotation][head]['target_log_probability_D'], exact):
                    raise ValueError('rotation-level log-probability effect differs')
            head_d[head] = sum((Fraction.from_float(x) for x in terms[head]), Fraction()) / 6
            if not close(summary['D'][head], head_d[head]):
                raise ValueError('family head effect differs')
            logits = [number(v['out_' + head]['comparison']['target_logit_delta']) for v in variants]
            logit_terms = []
            for rotation in range(2):
                row = logits[4*rotation:4*rotation+4]
                logit_terms.extend((row[0], row[0], row[0], -row[1], -row[2], -row[3]))
            exact_logit = sum((Fraction.from_float(x) for x in logit_terms), Fraction()) / 6
            if not close(summary['target_logit_D'][head], exact_logit):
                raise ValueError('family target-logit effect differs')
        expected_terms = terms['h10'] + [-x for x in terms['h2']]
        supplied = summary['contrast_terms']
        if (type(supplied) is not list or len(supplied) != 24
                or [number(x).hex() for x in supplied] != [x.hex() for x in expected_terms]):
            raise ValueError('contrast source terms differ')
        contrast = head_d['h10'] - head_d['h2']
        sign = 1 if contrast > 0 else -1 if contrast < 0 else 0
        if (type(summary['contrast_sign']) is not int or summary['contrast_sign'] != sign
                or not close(summary['contrast'], contrast)
                or not close(summary['h10_minus_h7'], head_d['h10'] - head_d['h7'])
                or not close(summary['h2_minus_h7'], head_d['h2'] - head_d['h7'])):
            raise ValueError('family contrast or sign differs')
        report.update({'contrast': summary['contrast'], 'contrast_sign': sign, 'D': summary['D']})
        reports.append(report); eligible.append((support[group], contrast))
    if cursor != len(records):
        raise ValueError('extra or missing action records')
    expected_forwards = 96 * 3 + planned_eligible * 24
    expected_bytes = 96 * (3*201_028 + 3*33_792) + planned_eligible * 24 * (201_028 + 3*33_792)
    if data['counts'] != {'submitted': expected_forwards, 'model_returns': expected_forwards,
                          'returned': expected_forwards, 'retained': expected_bytes}:
        raise ValueError('forward or retained-byte counts differ')
    supported = [v for status, v in eligible if status == 'supported']
    unsupported = [v for status, v in eligible if status == 'unsupported']
    def descriptor(values, reported):
        n = len(values); positive = sum(x > 0 for x in values)
        mean = sum(values, Fraction()) / n if n else None
        sign = (1 if mean > 0 else -1 if mean < 0 else 0) if n else None
        if (type(reported['count']) is not int or reported['count'] != n
                or type(reported['positive']) is not int or reported['positive'] != positive
                or reported['mean_sign'] != sign
                or ((mean is None and reported['mean_contrast'] is not None)
                    or (mean is not None and not close(reported['mean_contrast'], mean)))):
            raise ValueError('aggregate denominator or effect differs')
        return n, positive, sign
    decision = data['decision']
    n, positive, sign = descriptor(supported, decision['primary'])
    descriptor([v for _, v in eligible], decision['all_dose_eligible'])
    descriptor(unsupported, decision['baseline_unsupported_dose_eligible'])
    status = ('ineligible' if n < 6 else 'criterion-met' if sign > 0 and 4*positive >= 3*n
              else 'criterion-not-met')
    if (decision['primary_status'] != status or decision['dose_ineligible_groups'] !=
            [r['group'] for r in reports if r['dose_status'] == 'dose-ineligible']):
        raise ValueError('primary criterion or ineligible inventory differs')
    return {'status': 'saved-summaries-consistent', 'families': reports, 'decision': decision,
            'counts': data['counts'], 'scope': 'Recorded comparison metadata and fixed family accounting only; no full-logit replay, execution or landing attestation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'baseline-result', 'baseline-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.run_dir, args.baseline_result, args.baseline_audit, args.output)):
        raise ValueError('absolute paths required')
    baseline = parse(read(args.baseline_result, 8*1024*1024, BASELINE_RESULT_SHA))
    audit_data = parse(read(args.baseline_audit, 8*1024*1024, BASELINE_AUDIT_SHA))
    left, right = baseline['summary']['families'], audit_data['summary']['families']
    pairs = [(r['group'], r['baseline_status']) for r in left]
    if (pairs != [(r['group'], r['baseline_status']) for r in right]
            or audit_data.get('cases_sha256') != CASES_SHA or audit_data.get('synthetic_fixture') is not False):
        raise ValueError('baseline classification binding differs')
    result_raw = read(args.run_dir / 'result.json', 8*1024*1024, args.result_sha256)
    result = parse(result_raw)
    manifest_raw = read(args.run_dir / 'manifest.json', 2*1024*1024)
    manifest = parse(manifest_raw)
    if (result.get('manifest_sha256') != digest(manifest_raw) or manifest.get('cases_sha256') != CASES_SHA
            or manifest.get('baseline_hashes', {}).get('result.json') != BASELINE_RESULT_SHA
            or manifest.get('baseline_independent_audit_sha256') != BASELINE_AUDIT_SHA):
        raise ValueError('run manifest binding differs')
    support = {group: status for group, status in pairs if group in GROUPS}
    if len(support) != 12:
        raise ValueError('held-out baseline inventory differs')
    report = audit(result, support)
    report.update({'result_sha256': digest(result_raw), 'manifest_sha256': digest(manifest_raw),
                   'baseline_result_sha256': BASELINE_RESULT_SHA, 'baseline_audit_sha256': BASELINE_AUDIT_SHA})
    encoded = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        if stream.write(encoded) != len(encoded):
            raise OSError('short report write')
    print(json.dumps({'status': report['status'], 'sha256': digest(encoded)}))


if __name__ == '__main__':
    main()
