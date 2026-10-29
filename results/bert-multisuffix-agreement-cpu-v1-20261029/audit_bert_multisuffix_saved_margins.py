"""Independent selected-logit replay for a closed multisuffix attempt.

Checks fixed action identities, two selected binary32 logits and the
prespecified donorward contrast; it does not replay geometry, full outputs,
process execution or source custody. Expected hashes come from a separate
closed observation. No model or GPU is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

CASE_SHA = 'bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e'
VARIANTS = ('terminal', 'period', 'present', 'outside', 'awake')
VOCAB = 30522
IS, ARE = 2003, 2024


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def strict(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'cases', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('events-sha256', 'logits-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    root, output = a.attempt.resolve(strict=True), a.output.resolve()
    require(output != root and not output.is_relative_to(root) and not output.exists(),
            'fresh output outside attempt required')
    require(all(re.fullmatch(r'[0-9a-f]{64}', x) for x in (a.events_sha256, a.logits_sha256)),
            'digest syntax differs')
    ep, lp = root / 'events.jsonl', root / 'logits.f32'
    require(sha(a.cases) == CASE_SHA and sha(ep) == a.events_sha256 and sha(lp) == a.logits_sha256,
            'input digest differs')
    cases = strict(a.cases.read_bytes())['cases']
    require(len(cases) == 32 and all(type(c['index']) is int and c['index'] == i for i, c in enumerate(cases)),
            'case sequence differs')
    with ep.open('rb') as stream:
        events = [strict(line) for line in stream]
    require(len(events) == 1088 and lp.stat().st_size == len(events) * VOCAB * 4, 'incomplete event/logit count')
    schedule = [(c, variant, 'clean') for c in cases for variant in VARIANTS]
    for case in cases:
        for vi, variant in enumerate(VARIANTS):
            arms = ['self', 'opposite', 'same', 'random']
            if vi:
                arms.append('fixed_terminal_opposite')
            schedule.extend((case, variant, arm) for arm in (*arms, 'clean_after'))
    require(len(schedule) == len(events), 'schedule length differs')
    values, raw_selected = {}, {}
    with lp.open('rb') as stream:
        for index, (case, variant, arm) in enumerate(schedule):
            event = events[index]
            require(type(event.get('event_index')) is int and event['event_index'] == index
                    and type(event.get('case_index')) is int and event['case_index'] == case['index']
                    and event.get('case_id') == case['case_id'] and event.get('variant') == variant
                    and event.get('arm') == arm and event.get('returned_status') == 'returned'
                    and event.get('full_logits_finite') is True
                    and type(event.get('logits_offset')) is int and event['logits_offset'] == index * VOCAB * 4,
                    'action identity differs')
            raw = stream.read(VOCAB * 4)
            require(len(raw) == VOCAB * 4, 'short logits row')
            selected = (raw[IS * 4:IS * 4 + 4], raw[ARE * 4:ARE * 4 + 4])
            two = tuple(struct.unpack('<f', value)[0] for value in selected)
            require(all(math.isfinite(x) for x in two), 'nonfinite selected logit')
            key = (case['index'], variant, arm)
            values[key], raw_selected[key] = two[0] - two[1], selected
            if arm in ('self', 'clean_after'):
                require(selected == raw_selected[(case['index'], variant, 'clean')], 'selected control bits differ')
            if arm in ('opposite', 'fixed_terminal_opposite', 'same'):
                expected = case['same_donor_index'] if arm == 'same' else case['opposite_donor_index']
                require(type(event.get('donor_index')) is int and event['donor_index'] == expected,
                        'donor identity differs')
            if arm == 'random':
                require(type(event.get('random_seed')) is int and event['random_seed'] == 20261029 + case['index'],
                        'random seed differs')
    rows = []
    for case in cases:
        i, sign = case['index'], (-1 if case['number'] == 0 else 1)
        row = {'case_id': case['case_id'], 'number': case['number'], 'variants': {}}
        for vi, variant in enumerate(VARIANTS):
            baseline = values[i, variant, 'clean']
            arms = ('opposite', 'same', 'random') + (('fixed_terminal_opposite',) if vi else ())
            effects = {arm: sign * (values[i, variant, arm] - baseline) for arm in arms}
            row['variants'][variant] = {'clean_margin': baseline, 'donorward': effects,
                'opposite_minus_same': effects['opposite'] - effects['same'],
                'opposite_minus_random': effects['opposite'] - effects['random']}
        rows.append(row)
    summary = {}
    for variant in VARIANTS:
        variants = [row['variants'][variant] for row in rows]
        same = [v['opposite_minus_same'] for v in variants]
        random = [v['opposite_minus_random'] for v in variants]
        positive = sum(x > 0 and y > 0 for x, y in zip(same, random))
        mean_same, mean_random = math.fsum(same) / 32, math.fsum(random) / 32
        summary[variant] = {'mean_opposite_minus_same': mean_same, 'mean_opposite_minus_random': mean_random,
                            'positive_against_both': positive,
                            'criterion': mean_same > 0 and mean_random > 0 and positive >= 24}
    require(sha(a.cases) == CASE_SHA and sha(ep) == a.events_sha256 and sha(lp) == a.logits_sha256,
            'input changed during replay')
    report = {'schema': 'independent-multisuffix-saved-margins-v1', 'case_sha256': CASE_SHA,
              'events_sha256': a.events_sha256, 'logits_sha256': a.logits_sha256, 'events': len(events),
              'cases': rows, 'summary': summary,
              'scope': 'Selected is/are logits and fixed donor contrasts only; geometry, full-output finiteness, execution and custody require separate review.'}
    data = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with output.open('xb') as stream:
        require(stream.write(data) == len(data), 'short report write')
    print('independent_saved_margins', hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
