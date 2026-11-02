"""Check closed invariant-v2 file/receipt consistency without model execution.

External hashes are supplied by the caller. The result does not authenticate
execution, establish descendant cleanup, or check scientific outcomes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


def digest(path):
    h = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
            count += len(chunk)
    return h.hexdigest(), count


def ordinary(path, *, directory=False):
    info = path.lstat()
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not expected or stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('unsupported filesystem entry')


def checked_json(path, expected):
    ordinary(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('pinned receipt differs')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'supervisor-sha256', 'observer-exit', 'observer-exit-sha256',
                 'prelaunch-sha256', 'reservation-sha256', 'operator-exposure-sha256',
                 'source-revision', 'source-tree', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    for value in (args.supervisor_sha256, args.observer_exit_sha256, args.prelaunch_sha256,
                  args.reservation_sha256, args.operator_exposure_sha256):
        if re.fullmatch(r'[0-9a-f]{64}', value) is None:
            raise ValueError('invalid external SHA-256')
    if any(re.fullmatch(r'[0-9a-f]{40}', value) is None for value in (args.source_revision, args.source_tree)):
        raise ValueError('invalid source identity')
    outer = Path(args.attempt).resolve(strict=True)
    run = outer / 'run'
    ordinary(outer, directory=True)
    ordinary(run, directory=True)
    s = checked_json(outer / 'supervisor.json', args.supervisor_sha256)
    observed = checked_json(Path(args.observer_exit), args.observer_exit_sha256)
    child_path = outer / 'child-custody.json'
    child_sha, _ = digest(child_path)
    child = checked_json(child_path, child_sha)
    expected_binding = {'prelaunch_sha256': args.prelaunch_sha256,
                        'reservation_sha256': args.reservation_sha256,
                        'latest_exposure_sha256': args.operator_exposure_sha256}
    if (observed.get('schema') != 'agreement-invariant-v2-owner-exit-observation-v1'
            or type(observed.get('observer_exit_code')) is not int or observed['observer_exit_code'] != 0
            or type(observed.get('child_exit_code')) is not int or observed['child_exit_code'] != 0
            or observed.get('supervisor_sha256') != args.supervisor_sha256
            or observed.get('child_custody_sha256') != child_sha
            or observed.get('prelaunch_sha256') != args.prelaunch_sha256
            or observed.get('reservation_sha256') != args.reservation_sha256
            or observed.get('operator_exposure_sha256') != args.operator_exposure_sha256
            or observed.get('source_revision') != args.source_revision
            or observed.get('source_tree') != args.source_tree):
        raise ValueError('external observer record differs')
    if (s.get('schema') != 'agreement-direction-gated-observation-v1' or s.get('clean') is not True
            or s.get('validation_only') is not False or s.get('child_binding_matches') is not True
            or s.get('post_source_check') != 'matched' or s.get('post_errors') != []
            or s.get('run_inventory_error') is not None or s.get('child') != child
            or s.get('child_sha256') != child_sha or s.get('source_revision') != args.source_revision
            or s.get('source_tree') != args.source_tree or s.get('prelaunch_sha256') != args.prelaunch_sha256
            or s.get('completion_statuses') != {'entry_result': 'complete_candidate', 'terminal': 'complete_candidate'}):
        raise ValueError('supervisor record differs')
    result = s.get('supervision_result', {})
    if (result.get('clean') is not True or type(result.get('exit_code')) is not int or result['exit_code'] != 0
            or result.get('timed_out') is not False or result.get('pid') != child.get('owner_pid')
            or result.get('run_id') != child.get('run_id') or observed.get('owner_pid') != child.get('owner_pid')
            or observed.get('run_id') != child.get('run_id') or child.get('mode') != 'study'
            or child.get('prelaunch_sha256') != args.prelaunch_sha256):
        raise ValueError('direct child identity or exit differs')
    records = s.get('run_files')
    if type(records) is not list or len(records) != observed.get('run_files'):
        raise ValueError('run inventory count differs')
    seen, total = {}, 0
    for row in records:
        name = row.get('path') if type(row) is dict else None
        if type(name) is not str or not name or '\\' in name:
            raise ValueError('invalid run path')
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or name in seen:
            raise ValueError('unsafe or repeated run path')
        path = run.joinpath(*rel.parts)
        ordinary(path)
        sha, size = digest(path)
        if type(row.get('bytes')) is not int or row['bytes'] != size or row.get('sha256') != sha:
            raise ValueError('run file bytes differ: ' + name)
        seen[name] = sha
        total += size
    actual = set()
    for path in run.rglob('*'):
        ordinary(path, directory=path.is_dir())
        if path.is_file():
            actual.add(path.relative_to(run).as_posix())
    if actual != set(seen):
        raise ValueError('run inventory is incomplete or has extra files')
    required = {'prelaunch.json': args.prelaunch_sha256,
                'reconstructed-exposure.json': args.reservation_sha256,
                'latest-exposure.json': args.operator_exposure_sha256,
                'attempt/events.jsonl': observed.get('events_sha256'),
                'attempt/started.json': observed.get('started_sha256'),
                'attempt/terminal.json': observed.get('terminal_sha256'),
                'entry-result.json': observed.get('entry_result_sha256'),
                'model-admission.json': observed.get('model_admission_sha256')}
    if any(seen.get(name) != sha for name, sha in required.items()):
        raise ValueError('run/observer binding differs')
    entry = checked_json(run / 'entry-result.json', seen['entry-result.json'])
    terminal = checked_json(run / 'attempt/terminal.json', seen['attempt/terminal.json'])
    if (entry.get('status') != 'complete_candidate' or entry.get('model_disposed') is not True
            or entry.get('binding') != expected_binding or entry.get('owner_pid') != child['owner_pid']
            or entry.get('owner_run_id') != child['run_id'] or entry.get('worker', {}).get('forwards') != 612
            or terminal.get('status') != 'complete_candidate' or terminal.get('events') != 613
            or terminal.get('storage_accounting_complete') is not True
            or observed.get('reported_forwards') != 612 or observed.get('event_count') != 613
            or observed.get('storage_accounting_complete') is not True):
        raise ValueError('complete candidate accounting differs')
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or os.path.lexists(output):
        raise ValueError('a fresh absolute output is required')
    report = {'schema': 'agreement-invariant-v2-independent-file-receipt-v1',
              'status': 'consistent_closed_candidate', 'run_files_verified': len(seen),
              'run_file_bytes_verified': total, 'events': 613, 'reported_forwards': 612,
              'observer_exit_code': 0, 'direct_child_exit_code': 0,
              'source_revision': args.source_revision, 'source_tree': args.source_tree,
              'prelaunch_sha256': args.prelaunch_sha256, 'reservation_sha256': args.reservation_sha256,
              'operator_exposure_sha256': args.operator_exposure_sha256,
              'events_sha256': seen['attempt/events.jsonl'], 'terminal_sha256': seen['attempt/terminal.json'],
              'supervisor_sha256': args.supervisor_sha256, 'observer_exit_sha256': args.observer_exit_sha256,
              'scope': ('Saved file hashes and cooperative direct-exit records only; not authenticated '
                        'execution, scientific outcomes, protected-history clearance or descendant drain.')}
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')
    with output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short report write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'run_files_verified': len(seen)}))


if __name__ == '__main__':
    main()
