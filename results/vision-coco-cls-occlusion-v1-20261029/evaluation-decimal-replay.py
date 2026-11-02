"""Replay CLS/occlusion evaluation logit contrasts from closed float32 bytes.

This is a descriptive numerical check, not a landing, dose, custody, or
execution audit. Inputs must be quiescent. It checks the previously audited
screen selection and reports the stored dose status without validating it.
"""

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct


EXPECTED_SCREEN_AUDIT = 'a7a58e437bb539e11d60799d6fc7d43b4de3a062f9b18b37237c3dd05d097459'
EXPECTED_TERMINAL = '4f308d497117323ca6a50519c547b674145f89dcf09e2683138f41ca9bc4fa67'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'input exceeds its byte limit')
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def logits(raw):
    require(len(raw) == 4000, 'logit byte count')
    values = []
    for (word,) in struct.iter_unpack('<I', raw):
        exponent, fraction = (word >> 23) & 255, word & 0x7fffff
        require(exponent != 255, 'nonfinite logit')
        mantissa = fraction if exponent == 0 else (1 << 23) | fraction
        value = Decimal(mantissa) * (Decimal(2) ** (-149 if exponent == 0 else exponent - 150))
        values.append(-value if word & 0x80000000 else value)
    return values


def log_group(raw, category):
    values = logits(raw)
    require(category in (17, 18), 'category differs')
    group = values[281:286] if category == 17 else values[151:269]
    maximum, group_maximum = max(values), max(group)
    return (group_maximum - maximum + sum((x-group_maximum).exp() for x in group).ln()
            - sum((x-maximum).exp() for x in values).ln())


def main(data, screen_audit, report):
    screen_raw = read(screen_audit, 1024 * 1024)
    require(digest(screen_raw) == EXPECTED_SCREEN_AUDIT, 'screen audit digest')
    screen = json.loads(screen_raw)
    ids = screen['selected_ids']
    require(len(ids) == 8 and len(set(ids)) == 8 and
            len(screen['initial_check']['initials']) == 24, 'selected prefix differs')
    terminal_raw = read(data / 'terminal.json', 1024 * 1024)
    require(digest(terminal_raw) == EXPECTED_TERMINAL, 'terminal digest')
    terminal = json.loads(terminal_raw)
    require(terminal['saved_forwards'] == terminal['submissions'] == 184 and
            terminal['selected_ids'] == ids, 'closed schedule differs')
    records = {}
    for index in range(1, 185):
        row = json.loads(read(data / f'forward-{index:03d}.json', 1024 * 1024))
        raw = read(data / f'forward-{index:03d}.bin', 1024 * 1024)
        require(row['submission'] == index and row['sha256'] == digest(raw), 'saved forward digest')
        layout = row['layout']['logits']
        require(layout == {'bytes': 4000, 'offset': 0, 'shape': [1, 1000]}, 'logit layout')
        require(row['label'] not in records, 'duplicate forward label')
        records[row['label']] = raw[:4000]
    output, max_drop_error, max_contrast_error = [], 0.0, 0.0
    with localcontext() as context:
        context.prec = 60
        for ident in ids:
            saved = json.loads(read(data / f'case-{ident}.json', 1024 * 1024))
            require(saved['image_id'] == ident, 'case identity')
            category = saved['category_id']
            clean = log_group(records[f'initial:{ident}:clean'], category)
            labels = {'animal_mask': f'initial:{ident}:animal',
                      'background_mask': f'initial:{ident}:background'}
            for name in ('cls_animal', 'cls_background', 'spatial_animal', 'spatial_background',
                         'random_cls', 'random_animal', 'random_background'):
                labels[name] = f'case-{ident}:{name}'
            drops = {name: clean-log_group(records[label], category) for name, label in labels.items()}
            for name, value in drops.items():
                stored = saved['conditions'][name]['drop']
                require(type(stored) in (int, float) and math.isfinite(stored), 'reported drop')
                error = abs(float(value)-stored)
                max_drop_error = max(max_drop_error, error)
                require(error <= 1e-12, 'saved drop differs')
            b = drops['animal_mask']-drops['background_mask']
            tc = drops['cls_animal']-drops['cls_background']
            ts = drops['spatial_animal']-drops['spatial_background']
            d = abs(b-ts)-abs(b-tc)
            for key, value in (('occlusion_contrast', b), ('cls_contrast', tc),
                               ('spatial_contrast', ts), ('residual_advantage', d)):
                stored = saved['comparisons'][key]
                require(type(stored) in (int, float) and math.isfinite(stored), 'reported contrast')
                error = abs(float(value)-stored)
                max_contrast_error = max(max_contrast_error, error)
                require(error <= 1e-12, 'saved contrast differs')
            dose = saved['dose']
            require(dose['status'] == 'limited-dose' and dose['q'] > 0 and
                    all(value > 0 for value in dose['receiver_norms'].values()), 'reported dose status')
            output.append({'image_id': ident, 'category_id': category,
                           'drops': {name: float(value) for name, value in drops.items()},
                           'occlusion_contrast': float(b), 'cls_contrast': float(tc),
                           'spatial_contrast': float(ts), 'residual_advantage': float(d),
                           'reported_q_over_receiver_norm': {
                               key: dose['q']/value for key, value in dose['receiver_norms'].items()},
                           'reported_dose_status': dose['status']})
    result = {'schema': 'vision-cls-occlusion-evaluation-decimal-review-v1', 'precision': 60,
              'screen_cross_phase_audit_sha256': digest(screen_raw), 'terminal_sha256': digest(terminal_raw),
              'max_abs_drop_error': max_drop_error, 'max_abs_contrast_error': max_contrast_error,
              'cases': output,
              'scope': 'Saved-logit arithmetic only. Dose statuses and ratios are reported values, not audited; '
                       'no intervention landing, source, execution, timing, or population claim.'}
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    with report.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'short report write')
    print(json.dumps({'report_sha256': digest(raw), 'max_abs_drop_error': max_drop_error,
                      'max_abs_contrast_error': max_contrast_error, 'cases': len(output)}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('screen_audit', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    main(args.data, args.screen_audit, args.report)
