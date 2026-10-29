"""Build a descriptive constant output-slope reference from exposed data.

This reads the eight previously published context +/- u logits, never a model
or a prospective evaluation case. The result is an analysis-only binary64
vector; it is not a new model intervention or a calibrated randomization null.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

EVENT_SHA = '7c8be25d480d986b087052685720ba7d2813c53d27509dcc81e67f9e2049eca6'
REPORT_SHA = '8b816905a3009e212ab138c62cbb24f1dd06d53079e228c283bb5d8f6ae1842f'
WIDTH = 50257
IDS = tuple('agreement-direction-context-%02d' % i for i in range(1, 9))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def derive(pairs, doses):
    if len(pairs) != 8 or len(doses) != 8:
        raise ValueError('the reference needs all eight fixed development contexts')
    width = len(pairs[0][0])
    if width == 0 or any(len(a) != width or len(b) != width for a, b in pairs):
        raise ValueError('inconsistent vectors')
    if any(type(q) is not float or not math.isfinite(q) or q <= 0 for q in doses):
        raise ValueError('invalid development dose')
    if any(not math.isfinite(x) for pair in pairs for vector in pair for x in vector):
        raise ValueError('nonfinite development logit')
    # Written operation order is part of the reference: each difference is
    # divided by (2*q) in binary64, then fsum in fixed context order, then /8.
    return tuple(math.fsum((plus[j] - minus[j]) / (2.0 * q)
                           for (plus, minus), q in zip(pairs, doses)) / 8.0
                 for j in range(width))


def bounded(path, limit):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('input exceeds bound')
    return data


def build(attempt, report_path, output):
    event_raw = bounded(attempt / 'events.jsonl', 4 * 1024 * 1024)
    report_raw = bounded(report_path, 4 * 1024 * 1024)
    if sha(event_raw) != EVENT_SHA or sha(report_raw) != REPORT_SHA:
        raise ValueError('the development journal or report hash differs')
    events = [json.loads(line) for line in event_raw.splitlines()]
    report = json.loads(report_raw)
    contexts = report.get('contexts')
    if type(contexts) is not list or [x.get('case_id') for x in contexts] != list(IDS):
        raise ValueError('development context order differs')
    records = {}
    for row in events:
        event = row.get('event', {})
        key = (event.get('case_id'), event.get('action'))
        if key[0] in IDS and key[1] in ('u_plus', 'u_minus'):
            if key in records:
                raise ValueError('duplicate development action')
            records[key] = row
    if len(records) != 16:
        raise ValueError('missing development action')
    pairs, doses, sources = [], [], []
    for row in contexts:
        case_id, q = row['case_id'], row['dose']
        values, refs = [], []
        for action in ('u_plus', 'u_minus'):
            record = records[case_id, action]
            event, ref = record['event'], record['tensors']['last_logits']
            if (event.get('all_sequence_logits_finite') is not True
                    or event['dose']['requested_norm'] != q
                    or ref.get('dtype') != 'float32-le' or ref.get('shape') != [WIDTH]
                    or type(ref.get('bytes')) is not int or ref['bytes'] != WIDTH * 4
                    or type(ref.get('sha256')) is not str or len(ref['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in ref['sha256'])):
                raise ValueError('invalid development event or payload reference')
            raw = bounded(attempt / 'tensors' / (ref['sha256'] + '.f32le'), WIDTH * 4)
            if len(raw) != WIDTH * 4 or sha(raw) != ref['sha256']:
                raise ValueError('development logit bytes differ')
            values.append(struct.unpack('<%df' % WIDTH, raw))
            refs.append(ref['sha256'])
        pairs.append(tuple(values))
        doses.append(q)
        sources.append({'case_id': case_id, 'dose': q, 'plus_sha256': refs[0], 'minus_sha256': refs[1]})
    reference = derive(pairs, doses)
    raw = struct.pack('<%dd' % WIDTH, *reference)
    if output.exists():
        raise FileExistsError('reference output directory must be new')
    output.mkdir()
    vector_path = output / 'output-slope.f64le'
    vector_path.write_bytes(raw)
    if vector_path.read_bytes() != raw:
        raise OSError('written output slope differs')
    manifest = {
        'schema': 'agreement-context-output-slope-reference-v1',
        'scope': 'Descriptive first-order output-space reference from eight exposed contexts; not an intervention or inferential null.',
        'event_sha256': EVENT_SHA, 'science_report_sha256': REPORT_SHA,
        'construction': 'slope[j] = math.fsum((plus[j] - minus[j]) / (2.0 * q) in fixed context order) / 8.0',
        'dtype': 'float64-le', 'length': WIDTH, 'vector_bytes': len(raw), 'vector_sha256': sha(raw),
        'sources': sources,
        'present_margin_per_unit': reference[318] - reference[389],
        'past_margin_per_unit': reference[373] - reference[547],
    }
    manifest_raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    (output / 'manifest.json').write_bytes(manifest_raw)
    return {'vector_sha256': sha(raw), 'manifest_sha256': sha(manifest_raw),
            'present_margin_per_unit': manifest['present_margin_per_unit'],
            'past_margin_per_unit': manifest['past_margin_per_unit']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.attempt, args.report, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
