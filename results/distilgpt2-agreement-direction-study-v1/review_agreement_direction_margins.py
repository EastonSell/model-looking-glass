"""Independent two-logit replay of a closed public agreement-direction run.

This checks the pinned event bytes and each retained last-logit digest. It
does not establish how activations were produced, process cleanup, or exposure
history, and is narrower than the full saved-witness science reader.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


CASE_SHA = 'a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13'
EVENT_SHA = '7c8be25d480d986b087052685720ba7d2813c53d27509dcc81e67f9e2049eca6'
SCIENCE_SHA = '8b816905a3009e212ab138c62cbb24f1dd06d53079e228c283bb5d8f6ae1842f'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def decode(bits):
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7fffff
    check(exponent != 255, 'nonfinite selected logit')
    value = (math.ldexp(float(mantissa), -149) if exponent == 0
             else math.ldexp(float((1 << 23) + mantissa), exponent - 150))
    return math.copysign(value, -1.0 if bits >> 31 else 1.0)


def top_id(raw):
    best, winner = -1, None
    for index, (bits,) in enumerate(struct.iter_unpack('<I', raw)):
        check(((bits >> 23) & 255) != 255, 'nonfinite full last-logit vector')
        order = (0x80000000 if (bits & 0x7fffffff) == 0 else
                 (~bits & 0xffffffff) if bits >> 31 else bits ^ 0x80000000)
        if order > best:
            best, winner = order, index
    return winner


def audit(attempt, case_file, reader_report):
    case_raw = case_file.read_bytes()
    check(hashlib.sha256(case_raw).hexdigest() == CASE_SHA, 'case digest differs')
    cases = json.loads(case_raw)['cases']
    check(len(cases) == 40 and len({x['case_id'] for x in cases}) == 40, 'case inventory differs')
    events_raw = (attempt / 'events.jsonl').read_bytes()
    check(hashlib.sha256(events_raw).hexdigest() == EVENT_SHA, 'event digest differs')
    records = [json.loads(line) for line in events_raw.splitlines()]
    check(len(records) == 809, 'journal event count differs')
    events = [record for record in records if record['event']['kind'] == 'forward']
    check(len(events) == 784, 'forward count differs')
    margins, tops, cache = {}, {}, {}
    for index, record in enumerate(events):
        event, reference = record['event'], record['tensors']['last_logits']
        check(type(event['index']) is int and event['index'] == index, 'forward sequence differs')
        key = (event['case_id'], event['action'])
        check(key not in margins, 'duplicate case/action')
        check(reference['dtype'] == 'float32-le' and reference['shape'] == [50257]
              and reference['bytes'] == 50257 * 4, 'logit layout differs')
        sha = reference['sha256']
        check(type(sha) is str and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
              'invalid tensor digest')
        if sha not in cache:
            raw = (attempt / 'tensors' / (sha + '.f32le')).read_bytes()
            check(len(raw) == 50257 * 4 and hashlib.sha256(raw).hexdigest() == sha,
                  'logit bytes differ from reference')
            is_bits, are_bits = (struct.unpack_from('<I', raw, 4 * token)[0] for token in (318, 389))
            cache[sha] = (decode(is_bits) - decode(are_bits), top_id(raw))
        margins[key], tops[key] = cache[sha]

    def response(case, direction):
        name = case['case_id']
        return (margins[name, direction + '_plus'] - margins[name, direction + '_minus']) / 2.0

    target = cases[:32]
    context = cases[32:]
    check(all(row['role'] == 'target' and row['family_index'] == i // 4 and row['variant'] == i % 4
              for i, row in enumerate(target)) and all(row['role'] == 'context' for row in context),
          'case ordering differs')

    def signed(rows):
        u = math.fsum(response(row, 'u') for row in rows) / len(rows)
        c1 = math.fsum(abs(response(row, 'v1')) for row in rows) / len(rows)
        c2 = math.fsum(abs(response(row, 'v2')) for row in rows) / len(rows)
        return {'n': len(rows), 'mean_u': u, 'mean_abs_v1': c1, 'mean_abs_v2': c2,
                'passed': u >= 0.05 and u > c1 and u > c2}

    classes = {label: signed([row for row in target if row['subject_number'] == label])
               for label in ('singular', 'plural')}
    families = [{'family': n, **signed(target[n * 4:n * 4 + 4])} for n in range(8)]
    pairs = []
    for family in range(8):
        for variant, morphology in ((0, 'regular'), (2, 'irregular')):
            singular, plural = (target[4 * family + variant + offset]['case_id'] for offset in (0, 1))
            clean_gap = margins[singular, 'clean'] - margins[plural, 'clean']
            losses = {direction: clean_gap - (margins[singular, 'center_' + direction]
                                               - margins[plural, 'center_' + direction])
                      for direction in ('u', 'v1', 'v2')}
            pairs.append({'family': family, 'morphology': morphology, **losses,
                          'passed': losses['u'] >= 0.05 and losses['u'] > abs(losses['v1'])
                                    and losses['u'] > abs(losses['v2'])})
    p_u = math.fsum(row['u'] for row in pairs) / 16
    p_v1 = math.fsum(abs(row['v1']) for row in pairs) / 16
    p_v2 = math.fsum(abs(row['v2']) for row in pairs) / 16
    passed_pairs = sum(row['passed'] for row in pairs)
    projection = {'n': 16, 'mean_u': p_u, 'mean_abs_v1': p_v1, 'mean_abs_v2': p_v2,
                  'pairs': pairs, 'pairs_passed': passed_pairs,
                  'passed': p_u >= 0.10 and p_u > p_v1 and p_u > p_v2 and passed_pairs >= 12}
    family_count = sum(row['passed'] for row in families)
    signed_result = {'classes': classes, 'families': families, 'families_passed': family_count,
                     'passed': all(row['passed'] for row in classes.values()) and family_count >= 6}
    reader_raw = reader_report.read_bytes()
    check(hashlib.sha256(reader_raw).hexdigest() == SCIENCE_SHA, 'science report digest differs')
    reader = json.loads(reader_raw)
    check(reader['signed_prediction'] == signed_result and reader['projection_prediction'] == projection,
          'independent selected-logit decisions differ from science reader')
    context_u = [response(row, 'u') for row in context]
    clean_correct = sum((1 if row['subject_number'] == 'singular' else -1)
                        * margins[row['case_id'], 'clean'] > 0 for row in target)
    reversals = {arm: sum((margins[row['case_id'], 'clean'] > 0 > margins[row['case_id'], arm]
                           or margins[row['case_id'], 'clean'] < 0 < margins[row['case_id'], arm])
                          for row in target) for arm in ('same_diff', 'opp_same', 'opp_diff')}
    top1_changes = sum(tops[row['case_id'], 'clean'] != tops[row['case_id'], 'u_plus'] for row in target)
    return {'case_sha256': CASE_SHA, 'event_sha256': EVENT_SHA,
            'scientific_report_sha256': hashlib.sha256(reader_raw).hexdigest(),
            'forwards': len(events), 'unique_logit_payloads': len(cache),
            'signed_prediction': signed_result, 'projection_prediction': projection,
            'context_u_odd_mean': math.fsum(context_u) / 8,
            'context_u_odd_min': min(context_u), 'context_u_odd_max': max(context_u),
            'baseline_correct': clean_correct, 'donor_margin_reversals': reversals,
            'u_plus_top1_changes': top1_changes,
            'scope': ('Selected present-tense logits and full-logit top-1 only; '
                      'not process, intervention, private-split or exposure authentication.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--science-report', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.attempt, args.cases, args.science_report)
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if args.output:
        with args.output.open('xb') as stream:
            check(stream.write(raw) == len(raw), 'short report write')
            stream.flush()
    print(json.dumps({'forwards': result['forwards'], 'scientific_report_sha256': result['scientific_report_sha256'],
                      'signed_passed': result['signed_prediction']['passed'],
                      'projection_passed': result['projection_prediction']['passed'],
                      'context_u_odd_mean': result['context_u_odd_mean'],
                      'baseline_correct': result['baseline_correct'], 'donor_reversals': result['donor_margin_reversals'],
                      'u_plus_top1_changes': result['u_plus_top1_changes']}, sort_keys=True))


if __name__ == '__main__':
    main()
