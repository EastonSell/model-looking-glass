"""Audit saved public evidence for the v30 status overlay without a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f'digest mismatch: {path.name}')
    return data


def need(value, message):
    if not value:
        raise ValueError(message)


def study(attempt, cases, *, receipt_sha, manifest_sha, journal_sha, v2):
    receipt = json.loads(read(attempt / 'supervisor.json', receipt_sha))
    run = attempt / 'run'
    actual = {p.name for p in run.iterdir() if p.is_file()}
    need(actual == set(receipt['run_sha256']), 'receipt run-file set differs')
    for name, digest in receipt['run_sha256'].items():
        read(run / name, digest)
    manifest = json.loads(read(run / 'manifest.json', manifest_sha))
    expected = cases[12:] if v2 else cases
    need(manifest['cases'] == expected, 'manifest case rows differ')
    journal = read(run / 'attempt.jsonl', journal_sha)
    events = [json.loads(line) for line in journal.splitlines()]
    complete = [event for event in events if event.get('event') == 'complete']
    need(len(complete) == (288 if v2 else 168), 'complete action count differs')
    actual_actions = Counter((event['case_id'], event['action']) for event in complete)
    all_actions = ('clean', 'noop', 'self', 'year_final', 'label_final',
                   'year_at_zero', 'rotated_year_final', 'clean_after')
    wanted = Counter()
    for index, case in enumerate(expected):
        for action in (all_actions if v2 or index < 12 else ('clean', 'noop')):
            wanted[(case['case_id'], action)] += 1
    need(actual_actions == wanted, 'per-case actions differ')
    by_case = {case['case_id']: [] for case in expected}
    for event in complete:
        by_case[event['case_id']].extend(event['raw'].values())
    for case_id, descriptors in by_case.items():
        raw = (run / (case_id + '.f32')).read_bytes()
        offset = 0
        for descriptor in sorted(descriptors, key=lambda d: d['offset']):
            need(descriptor['offset'] == offset, 'raw segment gap or overlap')
            end = offset + descriptor['bytes']
            need(end <= len(raw) and sha(raw[offset:end]) == descriptor['sha256'], 'raw segment differs')
            offset = end
        need(offset == len(raw), 'raw stream coverage differs')
    need(receipt['process_exit_code'] == (0 if v2 else 1), 'observed exit differs')
    need(receipt['failure_file_present'] is (not v2) and receipt['interrupted'] is False,
         'failure or interruption state differs')
    return {'cases': len(expected), 'completed_actions': len(complete),
            'raw_files': len(expected), 'receipt_run_files': len(actual)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year-cases', required=True, type=Path)
    p.add_argument('--year-v1', required=True, type=Path)
    p.add_argument('--year-v2', required=True, type=Path)
    p.add_argument('--full-top-cases', required=True, type=Path)
    p.add_argument('--full-top-manifest', required=True, type=Path)
    p.add_argument('--full-top-attempt', required=True, type=Path)
    args = p.parse_args()
    years = json.loads(read(args.year_cases, '07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'))['cases']
    need(len(years) == 48 and len({x['case_id'] for x in years}) == 48, 'year cases differ')
    one = study(args.year_v1, years, v2=False,
                receipt_sha='f3b418c7760ed23e2f3c3f90bd39b33538b2b9913ffad2dcf0dd7546742f55bf',
                manifest_sha='d43d181e1c7046c8e97e6310bd3edd38fe85089e0e69347cd2ddb2728d29d47a',
                journal_sha='a2f5ba4f980bde155bf4eb5729f154f7422956eb9c0920421f13327038e057f5')
    two = study(args.year_v2, years, v2=True,
                receipt_sha='8e41226e78793580e09985232e8dd68415ac54c5863c4157189e08835d387978',
                manifest_sha='6a8ade8ab8bdcddc8a686c15f1ddb43b8db9a125d83bb49825fa8f2639510903',
                journal_sha='0c5132426cf1e855c93fa9f3fd4828f6c734d971522efa3a3f84ae354ff24f75')
    top = json.loads(read(args.full_top_cases, '4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261'))['cases']
    top_manifest = json.loads(read(args.full_top_manifest, '458c934bbe74ad01fd0d6abe6d4686aa1e47cbb993f4b76e04d451d07b3e56bb'))
    need(top_manifest['case_sha256'] == '4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261',
         'full-top manifest case binding differs')
    targets = [x for x in top if x['role'] != 'locality']
    localities = [x for x in top if x['role'] == 'locality']
    need(len(targets) == 36 and len(localities) == 12, 'full-top case counts differ')
    phase = json.loads(read(args.full_top_attempt / 'phase-record.json', '646aab4c7ee527c687c99c06408db4fc81850670f23c49a5db46f1bfd8f089f1'))
    need(phase['manifest_sha256'] == '458c934bbe74ad01fd0d6abe6d4686aa1e47cbb993f4b76e04d451d07b3e56bb',
         'full-top phase manifest binding differs')
    journal = read(args.full_top_attempt / 'screen/observations.jsonl', '28f29deac63fab2599a67ebf4537f14723e374377f2205c2cfe08300f7de7e57')
    selection = json.loads(read(args.full_top_attempt / 'screen/screen-selection.json', 'b41bc1240a8b43fa3810ab7bd27412d7a07341df4b79e7fbdef61d32d4f9ead9'))
    raw = read(args.full_top_attempt / 'screen/screen.f32', '62a04b0d13472a882ed56c9248391c73456cf4836007869e8e786fad517cf5b4')
    rows = [json.loads(line) for line in journal.splitlines()]
    need(len(rows) == 36 and {x['case_id'] for x in rows} == {x['case_id'] for x in targets},
         'full-top target identities differ')
    need(len({x['case_id'] for x in rows}) == 36, 'full-top target identities repeat')
    offset = 0
    for index, record in enumerate(rows):
        need(record['index'] == index and record['offset'] == offset and record['bytes'] == 201028,
             'full-top raw segment order differs')
        end = offset + record['bytes']
        need(end <= len(raw) and sha(raw[offset:end]) == record['sha256'], 'full-top raw segment differs')
        offset = end
    need(offset == len(raw), 'full-top raw coverage differs')
    need(phase['status'] == 'complete' and phase['child']['selection_status'] == 'ineligible'
         and selection['selection']['status'] == 'ineligible', 'full-top outcome differs')
    result = {'schema': 'exposure-v30-evidence-audit-v1', 'year_v1': one, 'year_v2': two,
              'full_top_targets': len(targets), 'full_top_localities_not_in_journal': len(localities),
              'full_top_raw_bytes': len(raw),
              'scope': 'Saved-byte consistency and cooperative process records; no model or protected data, independent execution attestation, or complete exposure-history claim. V1 private stderr is not read.'}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
