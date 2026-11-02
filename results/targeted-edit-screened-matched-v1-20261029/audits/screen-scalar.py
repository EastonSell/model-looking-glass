"""Model-free scalar review of a saved, public targeted-edit baseline screen."""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_artifact(base, entry, info):
    data = (base / entry).read_bytes()
    if len(data) != info['bytes'] or sha256(data) != info['sha256']:
        raise ValueError(f'artifact bytes or SHA mismatch: {entry}')
    return data


def audit(case_file, phase_file, screen_dir):
    case_bytes = case_file.read_bytes()
    phase_bytes = phase_file.read_bytes()
    phase = json.loads(phase_bytes)
    selection_bytes = verify_artifact(
        screen_dir.parent, 'screen/screen-selection.json',
        phase['artifacts']['screen/screen-selection.json'])
    wrapper = json.loads(selection_bytes)
    selection = wrapper['selection']
    if sha256(case_bytes) != wrapper['case_sha256']:
        raise ValueError('case inventory SHA mismatch')
    cases = [c for c in json.loads(case_bytes)['cases'] if c['role'] == 'target']
    if len(cases) != 32 or len(selection['rows']) != 32:
        raise ValueError('expected exactly 32 screen targets and selection rows')
    screen_bytes = verify_artifact(screen_dir, 'screen.f32', wrapper['artifacts']['screen.f32'])
    observations_bytes = verify_artifact(
        screen_dir, 'observations.jsonl', wrapper['artifacts']['observations.jsonl'])
    verify_artifact(screen_dir, 'state-original.json', wrapper['artifacts']['state-original.json'])
    for name in ('screen/screen.f32', 'screen/observations.jsonl', 'screen/state-original.json'):
        verify_artifact(screen_dir.parent, name, phase['artifacts'][name])
    observations = [json.loads(line) for line in observations_bytes.splitlines()]
    stride = 50257 * 4
    if len(observations) != 32 or len(screen_bytes) != 32 * stride:
        raise ValueError('screen/observation shape mismatch')

    strata = ('s1', 's2', 's3', 's4')
    eligible_by_stratum = {key: [] for key in strata}
    rows = []
    for i, (case, declared, observed) in enumerate(zip(cases, selection['rows'], observations)):
        if case['stratum'] not in eligible_by_stratum:
            raise ValueError('unknown stratum')
        continuations = case['continuations']
        if ([x['text'] for x in continuations] != [' London', ' Paris'] or
                [x['ids'] for x in continuations] != [[3576], [6342]]):
            raise ValueError('unexpected candidate token identity')
        raw = screen_bytes[i * stride:(i + 1) * stride]
        row_hash = sha256(raw)
        logits = struct.unpack('<50257f', raw)
        if not all(math.isfinite(x) for x in logits):
            raise ValueError(f'nonfinite full-vocabulary row {i}')
        margin = float(logits[3576]) - float(logits[6342])
        eligible = margin < 0.0
        top_id = max(range(len(logits)), key=lambda j: (logits[j], -j))
        expected = {'case_id': case['case_id'], 'stratum': case['stratum'], 'offset': i,
                    'logit_byte_offset': i * stride, 'logit_bytes': stride,
                    'logit_sha256': row_hash, 'baseline_margin': margin, 'eligible': eligible}
        if any(declared.get(k) != v for k, v in expected.items()):
            raise ValueError(f'declared row mismatch at {i}')
        observed_expected = {'case_id': case['case_id'], 'index': i, 'offset': i * stride,
                             'bytes': stride, 'sha256': row_hash}
        if any(observed.get(k) != v for k, v in observed_expected.items()):
            raise ValueError(f'observation mismatch at {i}')
        if eligible:
            eligible_by_stratum[case['stratum']].append(i)
        rows.append({'case_id': case['case_id'], 'offset': i, 'stratum': case['stratum'],
                     'baseline_margin': margin, 'eligible': eligible,
                     'top_token_id': top_id, 'row_sha256': row_hash})

    if Counter(c['stratum'] for c in cases) != {key: 8 for key in strata}:
        raise ValueError('case stratum counts differ from protocol')
    chosen = []
    for n in range(8):
        for key in strata:
            available = eligible_by_stratum[key]
            if n < len(available):
                chosen.append(available[n])
            if len(chosen) == 8:
                break
        if len(chosen) == 8:
            break
    chosen_ids = [cases[i]['case_id'] for i in chosen]
    eligible_counts = {key: len(eligible_by_stratum[key]) for key in strata}
    selected_counts = {key: sum(cases[i]['stratum'] == key for i in chosen) for key in strata}
    expected_summary = {'eligible_count': sum(eligible_counts.values()),
                        'eligible_counts_by_stratum': eligible_counts,
                        'selected_count': len(chosen), 'selected_counts_by_stratum': selected_counts,
                        'selected_offsets': chosen, 'selected_ids': chosen_ids,
                        'formal_target_eligible': len(chosen) == 8}
    if any(selection.get(k) != v for k, v in expected_summary.items()):
        raise ValueError('round-robin summary mismatch')
    for i, row in enumerate(rows):
        row['selected'] = i in chosen
        if selection['rows'][i]['selected'] != row['selected']:
            raise ValueError(f'selection flag mismatch at {i}')
    top_counts = Counter('London' if r['top_token_id'] == 3576 else
                         'Paris' if r['top_token_id'] == 6342 else 'other' for r in rows)
    return {
        'schema': 'screened-target-edit-independent-scalar-audit-v1',
        'status': 'passed',
        'scope': ('Post-run, model-free saved-byte and selection check; this script was written '
                  'after the baseline screen was observed. It does not establish source chronology, '
                  'actual execution, restoration or model identity.'),
        'audit_source_sha256': sha256(Path(__file__).read_bytes()),
        'case_sha256': sha256(case_bytes), 'phase_sha256': sha256(phase_bytes),
        'selection_sha256': sha256(selection_bytes), 'screen_sha256': sha256(screen_bytes),
        'observations_sha256': sha256(observations_bytes),
        'vocabulary_size': 50257, 'checked_finite_logits': 32 * 50257,
        **expected_summary,
        'top_token_category_counts': {key: top_counts[key] for key in ('London', 'Paris', 'other')},
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--screen-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.cases, args.phase, args.screen_dir)
    with args.report.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(f"{result['status']}: {len(result['rows'])} rows, "
          f"{result['eligible_count']} eligible, {result['selected_count']} selected")


if __name__ == '__main__':
    main()
