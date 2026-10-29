"""Independently check a closed agreement-direction supervisor receipt.

This reads saved public and operator records only. It does not authenticate
execution, independently observe the outer parent's exit, inspect private
splits, or recompute the scientific outcome. Paths passed to it are trusted.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def same_json(left, right):
    options = {'sort_keys': True, 'separators': (',', ':'), 'allow_nan': False}
    return json.dumps(left, **options) == json.dumps(right, **options)


def ordinary(path, directory=False):
    info = path.lstat()
    valid_kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    require(valid_kind and not stat.S_ISLNK(info.st_mode)
            and not (getattr(info, 'st_file_attributes', 0) & 0x400),
            'aliased or unexpected filesystem entry')
    return info


def bounded(path, cap):
    info = ordinary(path)
    require(info.st_size <= cap, 'bounded input too large')
    with path.open('rb') as f:
        raw = f.read(cap + 1)
    require(len(raw) == info.st_size and len(raw) <= cap, 'bounded input changed')
    return raw


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def finite(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON value')
        return value

    def reject(_):
        raise ValueError('nonfinite JSON value')

    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                      parse_float=finite, parse_constant=reject)


def inventory(root):
    ordinary(root, directory=True)
    rows, count, total = [], 0, 0
    pending = [(root, 0)]
    while pending:
        folder, depth = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                count += 1
                require(count <= 15000, 'too many output entries')
                path = Path(entry.path)
                info = path.lstat()
                require(not stat.S_ISLNK(info.st_mode)
                        and not (getattr(info, 'st_file_attributes', 0) & 0x400),
                        'aliased output entry')
                if stat.S_ISDIR(info.st_mode):
                    require(depth < 8, 'output nesting exceeds limit')
                    pending.append((path, depth + 1))
                else:
                    require(stat.S_ISREG(info.st_mode), 'unexpected output entry')
                    total += info.st_size
                    require(total <= 1024 ** 3, 'output size exceeds limit')
                    h, observed = hashlib.sha256(), 0
                    with path.open('rb') as stream:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            observed += len(chunk)
                            require(observed <= info.st_size, 'output grew during read')
                            h.update(chunk)
                    require(observed == info.st_size and path.stat().st_size == info.st_size,
                            'output changed during read')
                    rows.append({'path': path.relative_to(root).as_posix(),
                                 'bytes': observed, 'sha256': h.hexdigest()})
    return sorted(rows, key=lambda row: row['path'])


def checkout_source(checkout, expected_revision, expected_tree):
    ordinary(checkout, directory=True)
    env = {key: val for key, val in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': 'NUL',
                'GIT_CONFIG_SYSTEM': 'NUL', 'GIT_GRAFT_FILE': 'NUL',
                'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0',
                'GIT_OPTIONAL_LOCKS': '0'})
    def git(*args):
        return subprocess.check_output(['git', '--no-replace-objects', '-C', str(checkout), *args],
                                       env=env, timeout=60, stderr=subprocess.DEVNULL)
    require(git('rev-parse', '--verify', 'HEAD').strip().decode() == expected_revision,
            'checkout revision differs')
    require(git('show', '-s', '--format=%T', 'HEAD').strip().decode() == expected_tree,
            'checkout tree differs')
    require(not git('status', '--porcelain=v1', '--untracked-files=all').strip(),
            'checkout is not clean')
    entries = git('ls-tree', '-rz', '--full-tree', expected_revision).split(b'\0')
    sources = {}
    for entry in entries:
        if not entry:
            continue
        header, raw_path = entry.split(b'\t', 1)
        mode, kind, blob = header.decode('ascii').split()
        name = raw_path.decode('utf-8')
        if not ((name.startswith('looking_glass/') and name.endswith('.py'))
                or name == 'tools/agreement_direction_cpu_v1.py'):
            continue
        require(mode in ('100644', '100755') and kind == 'blob' and '..' not in Path(name).parts,
                'unexpected source tree entry')
        raw = bounded(checkout / name, 8 * 1024 * 1024)
        actual_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
        require(actual_blob == blob, 'checkout source differs from committed blob')
        sources[name] = digest(raw)
    return sources


def check_admission(admission, owner, source_hashes, binding, child, pre):
    require(admission['source_revision'] == owner['source_revision']
            and admission['source_tree'] == owner['source_tree'] and admission['source_sha256'] == source_hashes
            and admission['binding'] == binding
            and type(admission['forwards']) is int and admission['forwards'] == 0
            and type(admission['owner_pid']) is int and admission['owner_pid'] == child['owner_pid']
            and admission['owner_run_id'] == child['run_id'], 'admission identity differs')
    for key in ('python', 'torch', 'transformers', 'tokenizers', 'numpy', 'huggingface_hub'):
        require(admission['runtime_versions'][key] == pre['runtime'][key], 'runtime version differs')


def audit(outer, expected_prelaunch, expected_supervisor, checkout, validation=False,
          parent_exit_path=None, expected_parent_exit=None):
    ordinary(outer, directory=True)
    receipt_raw = bounded(outer / 'supervisor.json', 16 * 1024 * 1024)
    pre_raw = bounded(outer / 'pre-run.json', 256 * 1024)
    child_raw = bounded(outer / 'child-custody.json', 256 * 1024)
    require(digest(receipt_raw) == expected_supervisor, 'supervisor receipt pin differs')
    pre, child, receipt = (strict_json(raw) for raw in (pre_raw, child_raw, receipt_raw))
    require(digest(pre_raw) == expected_prelaunch == receipt['prelaunch_sha256'], 'prelaunch binding differs')
    require(pre['schema'] == 'distilgpt2-agreement-direction-prelaunch-v1', 'prelaunch schema differs')
    owner, exposure = pre['owner'], pre['exposure']
    require(receipt['schema'] == 'agreement-direction-gated-observation-v1'
            and receipt['source_revision'] == owner['source_revision']
            and receipt['source_tree'] == owner['source_tree'], 'supervisor source binding differs')
    require(receipt['validation_only'] is validation and receipt['clean'] is True
            and receipt['child_binding_matches'] is True and same_json(receipt['child'], child)
            and receipt['child_sha256'] == digest(child_raw), 'supervisor or child binding differs')
    supervised = receipt['supervision_result']
    require(type(supervised) is dict and supervised['clean'] is True
            and type(supervised['exit_code']) is int and supervised['exit_code'] == 0
            and type(supervised['pid']) is int and supervised['pid'] > 0
            and supervised['timed_out'] is False and supervised['reason'] == 'clean device teardown',
            'direct-child supervision was not clean')
    require(child['schema'] == 'agreement-direction-supervised-child-v1'
            and child['mode'] == ('validate_only' if validation else 'study')
            and child['prelaunch_sha256'] == expected_prelaunch
            and type(child['owner_pid']) is int and child['owner_pid'] > 0
            and child['owner_pid'] == supervised['pid']
            and type(child['run_id']) is str and re.fullmatch('[0-9a-f]{32}', child['run_id'])
            and child['run_id'] == supervised['run_id'], 'child identity differs')
    require(receipt['supervision_error_type'] is None and receipt['recovery'] == []
            and receipt['post_errors'] == [] and receipt['post_source_check'] == 'matched'
            and receipt['run_inventory_error'] is None, 'recorded postchecks were not clean')
    start = datetime.datetime.fromisoformat(receipt['start_utc'])
    end = datetime.datetime.fromisoformat(receipt['end_utc'])
    require(start.tzinfo is not None and end.tzinfo is not None and start <= end, 'recorded time order differs')
    source_hashes = checkout_source(checkout, owner['source_revision'], owner['source_tree'])
    require(source_hashes[owner['entry_path']] == owner['entry_script_sha256']
            and source_hashes[owner['executor_path']] == owner['executor_sha256']
            and all(source_hashes.get(path) == h for path, h in owner['supervision_sha256'].items()),
            'published selected source hashes differ')
    run = outer / 'run'
    if validation:
        require(not run.exists() and receipt['run_files'] == []
                and receipt['completion_statuses'] == {'terminal': None, 'entry_result': None},
                'validation created a study output')
        run_files = []
    else:
        run_files = inventory(run)
        require(same_json(run_files, receipt['run_files']), 'run file inventory or hashes differ')
        require(receipt['completion_statuses'] == {'terminal': 'complete_candidate',
                                                   'entry_result': 'complete_candidate'},
                'recorded candidate status differs')
        require(not (run / 'entry-failure.json').exists(), 'failure marker coexists with candidate')
        require(bounded(run / 'prelaunch.json', 256 * 1024) == pre_raw, 'retained prelaunch differs')
        require(digest(bounded(run / 'latest-exposure.json', 4 * 1024 * 1024))
                == exposure['latest_sha256'], 'child exposure snapshot differs')
        natural_raw = bounded(run / 'natural-comparison.json', 256 * 1024)
        require(digest(natural_raw) == pre['custody']['natural']['comparison_receipt_sha256'],
                'natural comparison receipt differs')
        natural = strict_json(natural_raw)
        require(digest(bounded(run / 'natural-comparison-inputs.json', 2 * 1024 * 1024))
                == natural['public_input_file_sha256'], 'natural comparison inputs differ')
        admission = strict_json(bounded(run / 'admission.json', 4 * 1024 * 1024))
        entry = strict_json(bounded(run / 'entry-result.json', 256 * 1024))
        model = strict_json(bounded(run / 'model-admission.json', 256 * 1024))
        started = strict_json(bounded(run / 'attempt' / 'started.json', 256 * 1024))
        terminal = strict_json(bounded(run / 'attempt' / 'terminal.json', 256 * 1024))
        binding = {'prelaunch_sha256': expected_prelaunch, 'reservation_sha256': exposure['reservation_sha256'],
                   'latest_exposure_sha256': exposure['latest_sha256']}
        check_admission(admission, owner, source_hashes, binding, child, pre)
        ckpt = model['checkpoint']
        expected_runtime = {'threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
                            'float32_matmul_precision': 'highest', 'calling_thread_denormal_product_bits': 0x00400000}
        require(model['status'] == 'loaded_before_forward' and model['binding'] == binding
                and model['case_sha256'] == pre['cases']['sha256']
                and same_json(model['runtime'], expected_runtime)
                and ckpt['model_id'] == pre['checkpoint']['model_id']
                and ckpt['model_revision'] == pre['checkpoint']['revision']
                and ckpt['weights_sha256'] == pre['checkpoint']['weights_sha256']
                and ckpt['asset_sha256']['tokenizer.json'] == pre['checkpoint']['tokenizer_sha256']
                and ckpt['device'] == 'cpu', 'model admission differs')
        require(started == {'scope': 'public prospective agreement-direction', 'external_binding': binding,
                            'source_revision': owner['source_revision']}, 'journal start differs')
        worker = entry['worker']
        require(entry['status'] == worker['status'] == terminal['status'] == 'complete_candidate'
                and entry['binding'] == binding and entry['model_disposed'] is True
                and type(entry['owner_pid']) is int and entry['owner_pid'] == child['owner_pid']
                and entry['owner_run_id'] == child['run_id']
                and type(worker['forwards']) is int and worker['forwards'] == 784
                and same_json(terminal['detail'], worker) and type(terminal['events']) is int and terminal['events'] == 809
                and terminal['storage_accounting_complete'] is True, 'candidate or cleanup records differ')
        require(type(terminal['event_log_bytes']) is int
                and terminal['event_log_bytes'] == (run / 'attempt' / 'events.jsonl').stat().st_size
                and type(terminal['retained_tensor_bytes']) is int
                and type(worker['retained_tensor_bytes']) is int
                and terminal['retained_tensor_bytes'] == worker['retained_tensor_bytes'],
                'saved journal accounting differs')
    all_files = inventory(outer)
    before_receipt = sum(row['bytes'] for row in all_files if row['path'] != 'supervisor.json')
    require(type(receipt['physical_bytes_before_receipt']) is int
            and before_receipt == receipt['physical_bytes_before_receipt'], 'recorded physical inventory differs')
    require(not any(row['path'].startswith('empty-pycache/') for row in all_files), 'bytecode prefix is not empty')
    exit_digest = None
    if parent_exit_path is not None:
        require(not validation and type(expected_parent_exit) is str, 'unexpected parent exit record')
        exit_raw = bounded(parent_exit_path, 256 * 1024)
        exit_digest = digest(exit_raw)
        require(exit_digest == expected_parent_exit, 'parent exit record pin differs')
        observed = strict_json(exit_raw)
        require(observed['schema'] == 'agreement-direction-observer-exit-observation-v1'
                and type(observed['observer_exit_code']) is int and observed['observer_exit_code'] == 0
                and observed['supervisor_receipt_sha256'] == digest(receipt_raw)
                and observed['child_custody_sha256'] == digest(child_raw)
                and observed['entry_result_sha256'] == digest(bounded(run / 'entry-result.json', 256 * 1024))
                and observed['journal_terminal_sha256'] == digest(bounded(run / 'attempt' / 'terminal.json', 256 * 1024))
                and observed['journal_events_sha256'] == digest(bounded(run / 'attempt' / 'events.jsonl', 64 * 1024 * 1024))
                and type(observed['reported_forwards']) is int and observed['reported_forwards'] == 784
                and type(observed['reported_journal_events']) is int and observed['reported_journal_events'] == 809,
                'parent exit observation differs from closed records')
        observed_at = datetime.datetime.fromisoformat(observed['observed_utc'].replace('Z', '+00:00'))
        require(observed_at.tzinfo is not None and end <= observed_at, 'parent observation time differs')
    else:
        require(expected_parent_exit is None, 'parent exit pin supplied without record')
    return {'mode': 'validation_only' if validation else 'closed_candidate',
            'prelaunch_sha256': digest(pre_raw), 'supervisor_sha256': digest(receipt_raw),
            'source_revision': owner['source_revision'], 'source_tree': owner['source_tree'],
            'source_files': len(source_hashes), 'run_files': len(run_files),
            'physical_bytes_before_receipt': before_receipt,
            'recorded_child_exit_zero': True, 'recorded_child_identity_matches': True,
            'recorded_postchecks_clean': True,
            'operator_parent_exit_record_consistent': parent_exit_path is not None,
            'operator_parent_exit_record_sha256': exit_digest,
            'scope': ('Recorded consistency only; parent exit is an operator observation, not independently '
                      'authenticated execution, descendant drain, scientific replay or complete exposure proof.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', required=True, type=Path)
    parser.add_argument('--checkout', required=True, type=Path)
    parser.add_argument('--prelaunch-sha256', required=True)
    parser.add_argument('--supervisor-sha256', required=True)
    parser.add_argument('--parent-exit', type=Path)
    parser.add_argument('--parent-exit-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{64}', args.prelaunch_sha256) is not None, 'invalid prelaunch pin')
    require(re.fullmatch('[0-9a-f]{64}', args.supervisor_sha256) is not None, 'invalid supervisor pin')
    require(args.parent_exit_sha256 is None or re.fullmatch('[0-9a-f]{64}', args.parent_exit_sha256),
            'invalid parent exit pin')
    report = audit(args.outer, args.prelaunch_sha256, args.supervisor_sha256, args.checkout,
                   args.validation_only, args.parent_exit, args.parent_exit_sha256)
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if args.output is not None:
        with args.output.open('xb') as stream:
            require(stream.write(raw) == len(raw), 'short report write')
            stream.flush()
            os.fsync(stream.fileno())
    print(raw.decode('utf-8'), end='')


if __name__ == '__main__':
    main()
