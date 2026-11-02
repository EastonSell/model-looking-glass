"""Build a source-bound operator record from separately observed public inputs.

This does not discover current exposure or custody status. The caller must
obtain and inspect those observations, and must not reuse an uncertain attempt.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import types


def load_captured(name, path, raw):
    """Execute the bytes already verified by the caller, never a cached pyc."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bounded(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('input exceeds byte bound')
    return raw


def main():
    parser = argparse.ArgumentParser()
    for name in ('repository', 'base-python', 'bound-exposure', 'operator-exposure',
                 'exposure-observation', 'natural-observation', 'injection-observation',
                 'observer', 'observer-support', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--source-publication', required=True)
    parser.add_argument('--expected-source-revision', required=True)
    parser.add_argument('--expected-source-tree', required=True)
    parser.add_argument('--expected-entry-sha256', required=True)
    parser.add_argument('--expected-worker-sha256', required=True)
    parser.add_argument('--reader-status', choices=('under_review', 'reviewed'), required=True)
    parser.add_argument('--purpose', choices=('rehearsal', 'study'), required=True)
    args = parser.parse_args()
    root = args.repository.resolve(strict=True)
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    environment['GIT_NO_REPLACE_OBJECTS'] = '1'

    def git(*parts):
        return subprocess.check_output(['git', *parts], cwd=root, env=environment, timeout=60).decode().strip()

    revision, tree = git('rev-parse', 'HEAD'), git('show', '-s', '--format=%T', 'HEAD')
    if (revision != args.expected_source_revision or tree != args.expected_source_tree
            or git('status', '--porcelain', '--untracked-files=normal')):
        raise ValueError('owner checkout must be clean')
    entry_path = root / 'tools/agreement_substitution_cpu_v1.py'
    worker_path = root / 'looking_glass/agreement_substitution_executor_v1.py'
    entry_raw = bounded(entry_path, 8 * 1024 * 1024)
    worker_raw = bounded(worker_path, 8 * 1024 * 1024)
    observer_path = args.observer.resolve(strict=True)
    support_path = args.observer_support.resolve(strict=True)
    observer_raw = bounded(observer_path, 1024 * 1024)
    support_raw = bounded(support_path, 1024 * 1024)
    if (digest(entry_raw) != args.expected_entry_sha256 or digest(worker_raw) != args.expected_worker_sha256
            or digest(observer_raw) != 'a51a3a0c9185041e25cba8371bba60e616fdf4e4d1512cb5397c40c04c72a8c5'
            or digest(support_raw) != '4e3e88634871fcd0d820ea649f7159952d5e6a26fe3a8c098ceb641f4dc28584'):
        raise ValueError('published owner or reviewed observer bytes differ')
    for relative, raw in (('tools/agreement_substitution_cpu_v1.py', entry_raw),
                          ('looking_glass/agreement_substitution_executor_v1.py', worker_raw)):
        committed = subprocess.check_output(['git', 'show', revision + ':' + relative],
                                            cwd=root, env=environment, timeout=60)
        if committed != raw:
            raise ValueError('owner bytes differ from committed source')
    entry = load_captured('_substitution_prelaunch_entry', entry_path, entry_raw)
    support = load_captured('_substitution_prelaunch_support', support_path, support_raw)
    if not entry._digest(entry.RESERVATION_REVISION, 40) or not isinstance(entry.RESERVATION_MODULE, str):
        raise ValueError('the published reservation is not pinned')
    supervision = ('looking_glass/__init__.py', 'looking_glass/supervisor.py',
                   'looking_glass/cpu_study_supervision.py', 'looking_glass/device_lease.py',
                   'looking_glass/device_identity.py')
    for relative in (entry_path.relative_to(root).as_posix(), worker_path.relative_to(root).as_posix(), *supervision):
        disk = support.bounded_read(root / relative, 8 * 1024 * 1024)
        committed = subprocess.check_output(['git', 'show', revision + ':' + relative],
                                            cwd=root, env=environment, timeout=60)
        if committed != disk:
            raise ValueError('source differs from committed checkout')
    subprocess.run(['git', 'merge-base', '--is-ancestor', entry.DIRECTION_REVISION, revision], cwd=root,
                   env=environment, timeout=60, check=True)
    # These published commitments are separate from the compact owner branch:
    # require exact Git blobs, and have the operator verify publication/currentness.
    for source_rev, relative, expected in (
            (entry.PROTOCOL_REVISION, entry.PROTOCOL_PATH, entry.PROTOCOL_SHA256),
            (entry.CASE_REVISION, entry.CASE_PATH, entry.CASE_SHA256),
            (entry.CASE_REVISION, entry.NATURAL_RECEIPT_PATH, entry.NATURAL_RECEIPT_SHA256),
            (entry.RESERVATION_REVISION, entry.RESERVATION_MODULE, None)):
        local = bounded(root / relative, 8 * 1024 * 1024)
        public = subprocess.check_output(['git', 'show', source_rev + ':' + relative],
                                         cwd=root, env=environment, timeout=60)
        if local != public or (expected is not None and digest(local) != expected):
            raise ValueError('published commitment bytes differ')
    natural_raw = support.bounded_read(args.natural_observation, 256 * 1024)
    injection_raw = support.bounded_read(args.injection_observation, 256 * 1024)
    exposure_raw = support.bounded_read(args.exposure_observation, 256 * 1024)
    natural, injection, exposure = map(json.loads, (natural_raw, injection_raw, exposure_raw))
    bound_raw = support.bounded_read(args.bound_exposure, 16 * 1024 * 1024)
    operator_raw = support.bounded_read(args.operator_exposure, 16 * 1024 * 1024)
    if digest(bound_raw) != entry.RESERVATION_SHA256 or digest(operator_raw) != exposure['operator_sha256']:
        raise ValueError('exposure observation does not match supplied bytes')
    if exposure['reconstructed_sha256'] != entry.RESERVATION_SHA256:
        raise ValueError('exposure observation does not bind the reconstructed inventory')
    record = {
        'schema': 'distilgpt2-agreement-substitution-prelaunch-v1',
        'purpose': args.purpose,
        'owner': {'source_revision': revision, 'source_tree': tree,
                  'executor_path': worker_path.relative_to(root).as_posix(),
                  'executor_sha256': digest(worker_raw),
                  'entry_path': entry_path.relative_to(root).as_posix(),
                  'entry_script_sha256': digest(entry_raw),
                  'supervision_sha256': {path: digest((root / path).read_bytes()) for path in supervision},
                  'source_publication': args.source_publication},
        'protocol': {'revision': entry.PROTOCOL_REVISION, 'path': entry.PROTOCOL_PATH,
                     'sha256': entry.PROTOCOL_SHA256},
        'cases': {'revision': entry.CASE_REVISION, 'path': entry.CASE_PATH,
                  'sha256': entry.CASE_SHA256, 'ordered_rows_sha256': entry.ROWS_SHA256},
        'directions': {'revision': entry.DIRECTION_REVISION, 'manifest_sha256': entry.DIRECTION_MANIFEST_SHA256,
                       **{name + '_sha256': value for name, value in entry.DIRECTION_SHA256.items()}},
        'reference': {'revision': entry.PROTOCOL_REVISION, 'path': entry.REFERENCE_PATH,
                      'sha256': entry.REFERENCE_SHA256, 'manifest_sha256': entry.REFERENCE_MANIFEST_SHA256},
        'checkpoint': {'model_id': entry.MODEL_ID, 'revision': entry.MODEL_REVISION,
                       'weights_sha256': entry.WEIGHTS_SHA256, 'tokenizer_sha256': entry.TOKENIZER_SHA256},
        'exposure': {'reservation_revision': entry.RESERVATION_REVISION,
                     'reservation_sha256': entry.RESERVATION_SHA256,
                     'reservation_version': entry.RESERVATION_VERSION,
                     'latest_revision': entry.RESERVATION_REVISION,
                     'latest_sha256': entry.RESERVATION_SHA256,
                     'latest_version': entry.RESERVATION_VERSION,
                     'latest_module': entry.RESERVATION_MODULE,
                     'operator_revision': exposure['operator_revision'],
                     'operator_sha256': exposure['operator_sha256'],
                     'operator_version': exposure['operator_version'],
                     'operator_observation_sha256': digest(exposure_raw),
                     'operator_refreshed_utc': exposure['refreshed_utc'],
                     'binding_scope': 'The child reconstructs the recorded reservation; any later current inventory is an operator observation.'},
        'custody': {'natural': {'status': 'clear', 'case_sha256': entry.CASE_SHA256,
                                'support_receipt_sha256': entry.NATURAL_RECEIPT_SHA256,
                                'scope': entry.NATURAL_SCOPE, 'observation_sha256': digest(natural_raw),
                                'refreshed_utc': natural['refreshed_utc']},
                    'injection': {'status': 'no_protected_rows', 'direct_target_conflict': 'none reported',
                                  'observation_sha256': digest(injection_raw),
                                  'refreshed_utc': injection['refreshed_utc']}},
        'runtime': {**entry.VERSIONS, 'cpu_threads': 1, 'cpu_interop_threads': 1,
                    'claim_cuda_visibility': '', 'environment': entry.REQUIRED_ENV,
                    'base_python_sha256': digest(support.bounded_read(args.base_python.resolve(strict=True), 16 * 1024 * 1024)),
                    'inherited_environment_sha256': support.inherited_environment_hashes()},
        'observer': {'sha256': digest(observer_raw),
                     'support_sha256': digest(support_raw),
                     'mode': 'gated isolated direct child',
                     'scope': 'Cooperative direct-child observation, not authenticated execution or descendant drain.'},
        'limits': entry.LIMITS,
        'independent_reader_status': args.reader_status,
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Operator/custodian reports do not authenticate execution, private membership or complete exposure history.',
    }
    entry.validate_prelaunch(record)
    cases = entry.verify_case_document(support.bounded_read(root / entry.CASE_PATH, 2 * 1024 * 1024))
    entry.verify_custody(record, (root / entry.NATURAL_RECEIPT_PATH).resolve(),
                         args.natural_observation.resolve(), args.injection_observation.resolve())
    entry.verify_operator_inventory(record, args.operator_exposure.resolve(),
                                    args.exposure_observation.resolve(), bound_raw, cases)
    raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with args.output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short prelaunch write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'source_revision': revision, 'source_tree': tree,
                      'prelaunch_sha256': digest(raw), 'purpose': args.purpose}, sort_keys=True))


if __name__ == '__main__':
    main()
