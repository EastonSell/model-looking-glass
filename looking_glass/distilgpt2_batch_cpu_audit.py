"""Read-only checks for the fixed DistilGPT-2 CPU batch evidence.

These checks establish internal consistency, not truthful execution or general
batch separability. No model, CUDA library, or pickle is loaded here.
"""

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import zipfile
import zlib
from datetime import datetime, timedelta
from pathlib import Path


class EvidenceError(ValueError):
    """Evidence is missing, malformed, or internally inconsistent."""


WIDTH = 50257
OUTPUT_CAP = 8 * 1024 * 1024
ARCHIVE_CAP = 16 * 1024 * 1024
CALLS = (('serial-0-0', 1), ('serial-0-1', 1), ('mixed-0', 2),
         ('serial-1-0', 1), ('serial-1-1', 1), ('mixed-1', 2))
VERSION = 'distilgpt2-batch-cpu-v1'
REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
MODEL_ID = 'distilbert/distilgpt2'
SEED = 2026102808
ASSETS = {
    'config.json': (762, '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a'),
    'generation_config.json': (124, 'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805'),
    'merges.txt': (456318, '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
    'model.safetensors': (352824413, 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'),
    'tokenizer.json': (1355256, '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'),
    'tokenizer_config.json': (26, '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f'),
    'vocab.json': (1042301, '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'),
}
SOURCE_PATHS = (
    'docs/DISTILGPT2_BATCH_CPU_V1.md', 'tools/run_distilgpt2_batch_cpu_v1.py',
    'looking_glass/__init__.py', 'looking_glass/distilgpt2_batch_cpu_v1.py',
    'looking_glass/batch_validation.py', 'looking_glass/microbatch.py',
    'looking_glass/pretrained.py', 'looking_glass/resident_worker.py',
    'tests/test_distilgpt2_batch_cpu_v1.py',
)
DISCOVERY = ([50256, 101, 201, 301], [50256, 102, 202, 302])
CASES = (([50256, 103, 203, 303], [50256, 104, 204, 304]),
         ([50256, 105, 205, 305], [50256, 106, 206, 306]))


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        raise EvidenceError('invalid finite JSON value') from None


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _row_digest(row):
    return _digest([{'name': 'attention_mask', 'dtype': 'int64', 'shape': [1, 4],
                     'values': [[1, 1, 1, 1]]},
                    {'name': 'input_ids', 'dtype': 'int64', 'shape': [1, 4],
                     'values': [row]}])


def _expected_groups():
    return [{'group_id': f'group-{group + 1}', 'cases': [
        {'case_id': f'case-{group + 1}-{index + 1}', 'owner': ('team-a', 'team-b')[index],
         'input_sha256': _row_digest(row), 'row_sha256': [_row_digest(row)]}
        for index, row in enumerate(rows)]} for group, rows in enumerate(CASES)]


def _expected_binding(runtime, sources, role):
    control = role == 'batch_mean'
    model = 'batch-mean-negative-control' if control else MODEL_ID
    width = 1 if control else WIDTH
    checkpoint = 'toy-v1' if control else REVISION
    output_bytes = 2 * 4 * width * 4
    return {
        'profile_id': f'{VERSION}-{role}', 'certification_id': f'{VERSION}-{role}-sample',
        'model_id': model, 'model_fingerprint': _digest({'model_id': model, 'revision': checkpoint,
                                                        'assets': {} if control else ASSETS}),
        'checkpoint': checkpoint, 'checkpoint_sha256': _digest('toy-v1') if control else ASSETS['model.safetensors'][1],
        'adapter': f'{VERSION}-{role}', 'adapter_sha256': _digest({'adapter': VERSION, 'role': role,
                                                                   'source_sha256': sources}),
        'runtime_id': 'sha256:' + _digest(runtime), 'backend': 'cpu-eager', 'device': 'cpu',
        'model_dtype': 'float32', 'output_dtype': 'float32',
        'input_schema': [['input_ids', 'int64'], ['attention_mask', 'int64']], 'input_shape': [1, 4],
        'output_width': width, 'approved_rows': [2], 'max_requests': 2, 'max_per_owner': 1,
        'max_batch_rows': 2, 'max_token_positions': 8, 'max_logical_input_bytes': 128,
        'max_logical_output_bytes': output_bytes, 'max_logical_total_bytes': 128 + output_bytes,
        'seed': SEED, 'serial_atol': 1e-6, 'serial_rtol': 1e-5, 'mode': 'eval',
        'deterministic': True, 'batch_separable': True, 'rng_policy': 'none',
        'cache_policy': 'disabled', 'mutable_state_policy': 'none', 'tracing': False, 'intervention': False,
    }


def audit_manifest(manifest, expected_sha256):
    """Recompute the frozen protocols and fixed inputs against an external pin."""
    fields = {'version', 'source_commit', 'runtime', 'source_sha256', 'asset_sha256', 'raw_output_cap_bytes',
              'primary_protocol', 'reversal_protocol', 'batch_mean_protocol', 'manifest_sha256'}
    _require(type(manifest) is dict and set(manifest) == fields and manifest['version'] == VERSION
             and type(manifest['source_commit']) is str
             and re.fullmatch(r'[0-9a-f]{40}', manifest['source_commit']) is not None
             and _sha(expected_sha256) and manifest['manifest_sha256'] == expected_sha256,
             'frozen manifest identity is invalid')
    _require(_digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) == expected_sha256,
             'frozen manifest hash differs from the independent pin')
    _require(type(manifest['raw_output_cap_bytes']) is int and manifest['raw_output_cap_bytes'] == OUTPUT_CAP
             and manifest['asset_sha256'] == {name: item[1] for name, item in ASSETS.items()},
             'checkpoint assets or output cap differ from the fixed protocol')
    sources, runtime = manifest['source_sha256'], manifest['runtime']
    runtime_fields = {'python', 'platform', 'machine', 'processor', 'python_byteorder', 'isolated_python',
                      'no_site', 'dont_write_bytecode', 'pycache_prefix_sha256', 'python_path_sha256',
                      'torch', 'transformers', 'numpy', 'torch_build_sha256', 'torch_cpu_capability',
                      'intraop_threads', 'interop_threads', 'deterministic_algorithms', 'deterministic_warn_only',
                      'float32_matmul_precision', 'mkldnn_enabled', 'mkldnn_deterministic', 'default_dtype',
                      'default_device', 'cpu_autocast_enabled', 'cuda_initialized',
                      'denormal_preserving_calling_thread', 'environment'}
    _require(type(sources) is dict and set(sources) == set(SOURCE_PATHS) | {'installed-transformers-modeling_gpt2.py'}
             and all(_sha(value) for value in sources.values())
             and type(runtime) is dict and set(runtime) == runtime_fields,
             'source or runtime identity is incomplete')
    fixed_runtime = {'python_byteorder': 'little', 'isolated_python': True, 'no_site': True,
                     'dont_write_bytecode': True, 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
                     'intraop_threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
                     'default_dtype': 'torch.float32', 'default_device': 'cpu', 'cpu_autocast_enabled': False,
                     'cuda_initialized': False, 'denormal_preserving_calling_thread': True}
    _require(all(key in runtime and _canonical(runtime[key]) == _canonical(value)
                 for key, value in fixed_runtime.items())
             and _sha(runtime.get('pycache_prefix_sha256')) and _sha(runtime.get('torch_build_sha256'))
             and type(runtime.get('python_path_sha256')) is list and runtime['python_path_sha256']
             and all(_sha(value) for value in runtime['python_path_sha256'])
             and type(runtime.get('environment')) is dict
             and set(runtime['environment']) == {'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                  'OMP_DYNAMIC', 'MKL_DYNAMIC'}
             and runtime['environment']['CUDA_VISIBLE_DEVICES'] == '-1'
             and all(value is None or type(value) is str for value in runtime['environment'].values())
             and all(type(runtime[key]) is bool for key in ('deterministic_warn_only', 'mkldnn_enabled',
                                                            'mkldnn_deterministic'))
             and runtime['float32_matmul_precision'] in ('highest', 'high', 'medium')
             and all(type(runtime[key]) is str and runtime[key] for key in
                     ('python', 'platform', 'machine', 'numpy', 'torch_cpu_capability'))
             and type(runtime['processor']) is str,
             'runtime settings differ from the fixed CPU protocol')
    discovery = sorted(_row_digest(row) for row in DISCOVERY)
    for name, role in (('primary_protocol', 'primary'), ('reversal_protocol', 'row_reversal'),
                       ('batch_mean_protocol', 'batch_mean')):
        expected = {'version': 'model-looking-glass-batch-validation-v1',
                    'binding': _expected_binding(runtime, sources, role),
                    'discovery_row_hashes': discovery, 'groups': _expected_groups()}
        expected['protocol_sha256'] = _digest(expected)
        _require(_canonical(manifest[name]) == _canonical(expected), 'frozen inputs, binding or protocol differ')
    return {'manifest_sha256': expected_sha256, 'source_commit': manifest['source_commit']}


def _hash_file(path, limit):
    h = hashlib.sha256()
    total = 0
    try:
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                total += len(chunk)
                _require(total <= limit, 'file exceeds the audit limit')
                h.update(chunk)
    except EvidenceError:
        raise
    except (OSError, ValueError):
        raise EvidenceError('cannot read evidence file') from None
    return h.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _decode_json(data):
    try:
        _require(len(data) <= 8 * 1024 * 1024, 'JSON evidence exceeds the audit limit')
        return json.loads(data, object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(EvidenceError('nonfinite JSON')))
    except EvidenceError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise EvidenceError('cannot parse evidence JSON') from None


def _load_json(path):
    try:
        with Path(path).open('rb') as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
    except (OSError, ValueError):
        raise EvidenceError('cannot read evidence JSON') from None
    return _decode_json(data)


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
    except (TypeError, ValueError, OverflowError):
        return False


def _relative_name(value):
    return (type(value) is str and value and '\\' not in value and ':' not in value and '\x00' not in value
            and not value.startswith('/') and all(part not in ('', '.', '..') for part in value.split('/')))


def verify_repository(repository, manifest_path, manifest, *, manifest_commit, manifest_repo_path,
                      verify_worktree=False):
    """Check committed manifest/source blobs and ancestry, with optional checkout bytes."""
    _require(type(manifest_commit) is str and re.fullmatch(r'[0-9a-f]{40}', manifest_commit) is not None
             and _relative_name(manifest_repo_path) and type(verify_worktree) is bool,
             'a valid independent manifest commit and relative path are required')
    _require(type(manifest) is dict and type(manifest.get('source_sha256')) is dict,
             'manifest source identity is incomplete')
    source = manifest.get('source_commit')
    _require(type(source) is str and re.fullmatch(r'[0-9a-f]{40}', source) is not None
             and source != manifest_commit, 'source and manifest commits must be distinct')
    root = Path(repository)
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], check=True, timeout=60,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        git('merge-base', '--is-ancestor', source, manifest_commit)
        _require(not git('ls-tree', '-z', source, '--', manifest_repo_path),
                 'manifest path already existed in source commit')
        def blob(commit, path, limit):
            entry = git('ls-tree', '-z', commit, '--', path)
            fields = entry.rstrip(b'\0').split(b'\t', 1)
            _require(len(fields) == 2 and fields[1] == path.encode('utf-8')
                     and fields[0].split(b' ', 2)[:2] in ([b'100644', b'blob'], [b'100755', b'blob']),
                     'expected tracked file is absent or not an ordinary blob')
            size = git('cat-file', '-s', f'{commit}:{path}').strip()
            _require(size.isdigit() and int(size) <= limit, 'committed blob exceeds the audit limit')
            return git('cat-file', 'blob', f'{commit}:{path}')
        committed = blob(manifest_commit, manifest_repo_path, 8 * 1024 * 1024)
        with Path(manifest_path).open('rb') as stream:
            supplied = stream.read(8 * 1024 * 1024 + 1)
        _require(len(supplied) <= 8 * 1024 * 1024 and supplied == committed,
                 'manifest bytes differ from the committed blob')
        _require(_canonical(_decode_json(committed)) == _canonical(manifest),
                 'supplied manifest object differs from the committed bytes')
        for path in SOURCE_PATHS:
            expected = manifest['source_sha256'][path]
            _require(_sha(expected) and hashlib.sha256(blob(source, path, 64 * 1024 * 1024)).hexdigest() == expected,
                     'recorded source digest differs from its committed blob')
            if verify_worktree:
                _require(_hash_file(root / path, 64 * 1024 * 1024) == expected,
                         'local working source bytes differ from the recorded blob')
        if verify_worktree:
            git('merge-base', '--is-ancestor', manifest_commit, 'HEAD')
            _require(blob('HEAD', manifest_repo_path, 8 * 1024 * 1024) == committed,
                     'checkout manifest differs from the committed blob')
    except EvidenceError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, subprocess.SubprocessError):
        raise EvidenceError('cannot verify source and manifest Git history') from None
    return {'manifest_commit': manifest_commit, 'source_commit': source,
            'source_blobs_checked': len(SOURCE_PATHS), 'working_source_checked': verify_worktree,
            'publication_before_execution_independently_proven': False}


def _observations():
    result = []
    for group, rows in enumerate(CASES):
        for name, ids in ((f'serial-{group}-0', [rows[0]]),
                          (f'serial-{group}-1', [rows[1]]), (f'mixed-{group}', list(rows))):
            result.append({'request_id': name, 'input_ids_shape': [len(ids), 4],
                           'attention_mask_shape': [len(ids), 4], 'input_ids_dtype': 'torch.int64',
                           'attention_mask_dtype': 'torch.int64', 'input_ids_device': 'cpu',
                           'attention_mask_device': 'cpu', 'input_ids_values': ids,
                           'attention_mask_values': [[1] * 4 for _ in ids]})
    return result


def _nonnegative(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        return False


def _audit_report(report, protocol, status, case_status):
    fields = {'version', 'protocol_sha256', 'status', 'complete', 'calls', 'groups',
              'tested_rows', 'error', 'evidence_sha256'}
    _require(type(report) is dict and set(report) == fields
             and report['version'] == 'model-looking-glass-batch-evidence-v1'
             and report['protocol_sha256'] == protocol['protocol_sha256']
             and report['status'] == status and report['complete'] is True
             and _canonical(report['calls']) == _canonical({'serial': 4, 'mixed': 2})
             and _canonical(report['tested_rows']) == _canonical([2]) and report['error'] is None
             and _sha(report['evidence_sha256'])
             and _digest({k: v for k, v in report.items() if k != 'evidence_sha256'}) == report['evidence_sha256'],
             'comparison report identity or completion is invalid')
    _require(type(report['groups']) is list and len(report['groups']) == 2,
             'comparison report group coverage differs')
    metrics = {'status', 'max_abs_error', 'absolute_error_overflow',
               'max_relative_error', 'relative_error_nonfinite_reason'}
    for group, expected in zip(report['groups'], protocol['groups']):
        _require(type(group) is dict and set(group) == {'group_id', 'cases'}
                 and group['group_id'] == expected['group_id'] and type(group['cases']) is list
                 and len(group['cases']) == 2, 'comparison report group differs')
        for case, frozen in zip(group['cases'], expected['cases']):
            _require(type(case) is dict and set(case) == set(frozen) | metrics
                     and all(_canonical(case[key]) == _canonical(value) for key, value in frozen.items())
                     and case['status'] == case_status and type(case['absolute_error_overflow']) is bool,
                     'comparison case or fixed input identity differs')
            absolute, relative = case['max_abs_error'], case['max_relative_error']
            # All compared values are finite float32. Their differences and
            # nonzero-denominator ratios fit comfortably in Python binary64.
            _require(case['absolute_error_overflow'] is False and _nonnegative(absolute),
                     'invalid float32 absolute comparison metric')
            _require((relative is None and case['relative_error_nonfinite_reason'] == 'zero_reference')
                     or (_nonnegative(relative) and case['relative_error_nonfinite_reason'] is None),
                     'invalid relative comparison metric')
            _require(case_status != 'mismatch' or absolute > 0,
                     'a nonzero discrepancy is required for a mismatch')
            _require(absolute != 0 or (relative == 0 and case['relative_error_nonfinite_reason'] is None),
                     'zero absolute error requires zero relative error')


def audit_run(directory, manifest, *, expected_manifest_sha256, manifest_commit, manifest_repo_path):
    """Check a saved terminal and archive; numerical replay is a separate check."""
    audit_manifest(manifest, expected_manifest_sha256)
    _require(type(manifest_commit) is str and re.fullmatch(r'[0-9a-f]{40}', manifest_commit) is not None
             and _relative_name(manifest_repo_path), 'independent manifest commit and path are required')
    root = Path(directory)
    try:
        names = {item.name for item in root.iterdir()}
    except OSError:
        raise EvidenceError('cannot inspect evidence directory') from None
    _require(not ({'summary.json', 'failure.json'} <= names), 'contradictory terminal records')
    _require('attempt.json' in names, 'attempt record is missing')
    attempt = _load_json(root / 'attempt.json')
    _require(type(attempt) is dict and set(attempt) ==
             {'version', 'status', 'manifest_sha256', 'manifest_commit', 'started_utc'}
             and attempt['version'] == VERSION and attempt['status'] == 'started'
             and attempt['manifest_sha256'] == expected_manifest_sha256
             and attempt['manifest_commit'] == manifest_commit and _timestamp(attempt['started_utc']),
             'attempt record differs from the frozen identity')
    if 'summary.json' not in names:
        if 'failure.json' in names:
            failure = _load_json(root / 'failure.json')
            _require(type(failure) is dict and set(failure) ==
                     {'version', 'status', 'exception_type', 'cleanup_exception_type',
                      'observed_model_calls', 'finished_utc'}
                     and failure['version'] == VERSION and failure['status'] == 'incomplete'
                     and type(failure['exception_type']) is str
                     and (failure['cleanup_exception_type'] is None or type(failure['cleanup_exception_type']) is str)
                     and type(failure['observed_model_calls']) is list and _timestamp(failure['finished_utc']),
                     'failure record is malformed')
        return {'status': 'incomplete', 'failure_record_present': 'failure.json' in names,
                'manifest_sha256': expected_manifest_sha256}
    _require(names == {'attempt.json', 'primary-report.json', 'logits.zip', 'summary.json'},
             'successful terminal has missing or extra artifacts')
    summary, report = _load_json(root / 'summary.json'), _load_json(root / 'primary-report.json')
    fields = {'version', 'status', 'complete', 'manifest_sha256', 'manifest_commit', 'manifest_repo_path',
              'model_metadata', 'model_state_sha256_before', 'model_state_sha256_after', 'observed_model_calls',
              'output_archive', 'controls', 'primary_evidence_sha256', 'finished_utc'}
    _require(type(summary) is dict and set(summary) == fields and summary['version'] == VERSION
             and summary['status'] in ('passed', 'failed', 'incomplete') and type(summary['complete']) is bool
             and summary['manifest_sha256'] == expected_manifest_sha256
             and summary['manifest_commit'] == manifest_commit and summary['manifest_repo_path'] == manifest_repo_path
             and _timestamp(summary['finished_utc']), 'terminal identity or status is malformed')
    _require(datetime.fromisoformat(summary['finished_utc']) >= datetime.fromisoformat(attempt['started_utc']),
             'recorded finish precedes recorded start')
    if summary['status'] != 'passed':
        _require(summary['complete'] is (summary['status'] == 'failed'),
                 'nonpassing terminal has contradictory completion')
        return {'status': summary['status'], 'manifest_sha256': expected_manifest_sha256,
                'limitations': ['nonpassing results were not promoted to a successful audit']}
    _require(summary['complete'] is True
             and _sha(summary['model_state_sha256_before'])
             and summary['model_state_sha256_before'] == summary['model_state_sha256_after']
             and _canonical(summary['observed_model_calls']) == _canonical(_observations()),
             'terminal status, model state or ordered input observations differ')
    metadata = summary['model_metadata']
    expected_metadata = {'model_id': MODEL_ID, 'model_revision': REVISION, 'tokenizer_id': MODEL_ID,
                         'tokenizer_revision': REVISION, 'weights_sha256': ASSETS['model.safetensors'][1],
                         'asset_sha256': {name: item[1] for name, item in ASSETS.items()},
                         'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
                         'device': 'cpu', 'dtype': 'float32', 'attention_implementation': 'eager',
                         'use_cache': False, 'parameter_count': 81912576}
    _require(type(metadata) is dict and all(key in metadata and _canonical(metadata[key]) == _canonical(value)
                                            for key, value in expected_metadata.items()),
             'recorded checkpoint metadata differs from the fixed protocol')
    _audit_report(report, manifest['primary_protocol'], 'passed', 'matched')
    _require(summary['primary_evidence_sha256'] == report['evidence_sha256'],
             'primary report digest differs from terminal')
    controls = summary['controls']
    _require(type(controls) is dict and set(controls) == {'positive_playback', 'reversed_playback', 'batch_mean'}
             and _canonical(controls['positive_playback']) == _canonical(report),
             'positive playback differs from the saved report')
    _audit_report(controls['reversed_playback'], manifest['reversal_protocol'], 'failed', 'mismatch')
    _audit_report(controls['batch_mean'], manifest['batch_mean_protocol'], 'failed', 'mismatch')
    archive = audit_archive(root / 'logits.zip', summary['output_archive'])
    return {'status': 'structurally_consistent', 'recorded_status': 'passed',
            'manifest_sha256': expected_manifest_sha256, 'manifest_commit': manifest_commit,
            'archive': archive, 'numerical_metrics_independently_recomputed': False,
            'limitations': ['source and runtime assertions do not attest execution',
                            'saved hashes do not prove the model computed these values',
                            'this audit does not recompute the tolerance comparisons']}


def audit_archive(path, info):
    """Check exact ordered members and raw bytes without trusting ZIP metadata alone."""
    _require(type(info) is dict and set(info) == {'raw_bytes', 'archive_sha256', 'entries'}
             and type(info['raw_bytes']) is int and _sha(info['archive_sha256'])
             and type(info['entries']) is list and len(info['entries']) == len(CALLS),
             'archive inventory is malformed')
    expected_total = sum(rows * 4 * WIDTH * 4 for _, rows in CALLS)
    _require(info['raw_bytes'] == expected_total and expected_total <= OUTPUT_CAP,
             'raw archive size differs from the fixed protocol')
    _require(_hash_file(path, ARCHIVE_CAP) == info['archive_sha256'], 'archive hash differs')
    members = []
    try:
        with zipfile.ZipFile(path, 'r') as archive:
            expected_names = [name + '.f32' for name, _ in CALLS]
            _require(archive.namelist() == expected_names, 'archive member order or names differ')
            for entry, (name, rows) in zip(info['entries'], CALLS):
                size = rows * 4 * WIDTH * 4
                _require(type(entry) is dict and set(entry) ==
                         {'request_id', 'member', 'shape', 'dtype', 'bytes', 'sha256'}
                         and entry['request_id'] == name and entry['member'] == name + '.f32'
                         and _canonical(entry['shape']) == _canonical([rows, 4, WIDTH])
                         and entry['dtype'] == 'little-endian-float32'
                         and type(entry['bytes']) is int and entry['bytes'] == size
                         and _sha(entry['sha256']), 'archive entry differs from the fixed protocol')
                member = archive.getinfo(entry['member'])
                _require(not member.is_dir() and not (member.flag_bits & 1)
                         and member.compress_type == zipfile.ZIP_DEFLATED and member.file_size == size,
                         'archive member metadata is invalid')
                raw = archive.read(member)
                _require(len(raw) == size and hashlib.sha256(raw).hexdigest() == entry['sha256'],
                         'archived output bytes differ from their digest')
                _require(all(math.isfinite(value[0]) for value in struct.iter_unpack('<f', raw)),
                         'archived logits contain nonfinite values')
                members.append({'request_id': name, 'shape': [rows, 4, WIDTH],
                                'bytes': size, 'sha256': entry['sha256']})
    except EvidenceError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile, EOFError, zlib.error):
        raise EvidenceError('invalid output archive') from None
    return {'archive_sha256': info['archive_sha256'], 'raw_bytes': expected_total, 'members': members}


def compare_archives(first, second):
    """Compare independently audited uncompressed members; ZIP headers may vary."""
    _require(type(first) is dict and type(second) is dict and 'members' in first and 'members' in second,
             'archive audit reports are required')
    return {'matching_member_hashes': _canonical(first['members']) == _canonical(second['members']),
            'archive_hashes_match': first.get('archive_sha256') == second.get('archive_sha256')}


def compare_runs(first_directory, second_directory, manifest, *, expected_manifest_sha256,
                 manifest_commit, manifest_repo_path):
    """Compare two separately saved records without running a model."""
    arguments = {'expected_manifest_sha256': expected_manifest_sha256, 'manifest_commit': manifest_commit,
                 'manifest_repo_path': manifest_repo_path}
    first = audit_run(first_directory, manifest, **arguments)
    second = audit_run(second_directory, manifest, **arguments)
    if first['status'] != 'structurally_consistent' or second['status'] != 'structurally_consistent':
        return {'status': 'incomplete_or_nonpassing', 'reference_status': first['status'],
                'replay_status': second['status']}
    a, b = Path(first_directory), Path(second_directory)
    primary_a, primary_b = _load_json(a / 'primary-report.json'), _load_json(b / 'primary-report.json')
    summary_a, summary_b = _load_json(a / 'summary.json'), _load_json(b / 'summary.json')
    archive = compare_archives(first['archive'], second['archive'])
    checks = {'raw_member_hashes_match': archive['matching_member_hashes'],
              'primary_reports_match': _canonical(primary_a) == _canonical(primary_b),
              'controls_match': _canonical(summary_a['controls']) == _canonical(summary_b['controls']),
              'model_state_hashes_match': summary_a['model_state_sha256_before'] == summary_b['model_state_sha256_before']}
    return {'status': 'matching_saved_checks' if all(checks.values()) else 'different_saved_checks',
            'checks': checks, 'archive_container_hashes_match': archive['archive_hashes_match'],
            'limitations': ['matching hashes do not attest independent execution',
                            'the saved comparisons are not independently numerically recomputed here']}


def verify_assets(directory):
    """Hash the seven locally supplied pinned assets; no network access is used."""
    root = Path(directory)
    try:
        for name, (size, expected) in ASSETS.items():
            path = root / name
            _require(path.stat().st_size == size and _hash_file(path, size) == expected,
                     'local checkpoint asset differs from the fixed pin')
    except EvidenceError:
        raise
    except (OSError, ValueError):
        raise EvidenceError('cannot inspect local checkpoint assets') from None
    return {'assets_checked': len(ASSETS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='private or published reference run directory')
    parser.add_argument('--manifest', required=True, help='saved frozen manifest file')
    parser.add_argument('--expected-manifest-sha256', required=True, help='separately retained manifest digest')
    parser.add_argument('--manifest-commit', required=True, help='independently selected manifest Git commit')
    parser.add_argument('--manifest-repo-path', required=True, help='manifest path relative to the repository')
    parser.add_argument('--repository', required=True, help='repository containing source and manifest Git history')
    parser.add_argument('--verify-worktree', action='store_true', help='also compare current working source bytes')
    parser.add_argument('--asset-directory', help='optional pinned local checkpoint snapshot')
    parser.add_argument('--transformers-source', help='optional installed modeling_gpt2.py file')
    parser.add_argument('--replay-directory', help='optional second completed run to compare')
    args = parser.parse_args(argv)
    try:
        manifest = _load_json(args.manifest)
        audit_manifest(manifest, args.expected_manifest_sha256)
        history = verify_repository(args.repository, args.manifest, manifest,
                                    manifest_commit=args.manifest_commit,
                                    manifest_repo_path=args.manifest_repo_path,
                                    verify_worktree=args.verify_worktree)
        local = {'assets': 'not_checked', 'transformers_source': 'not_checked'}
        if args.asset_directory:
            verify_assets(args.asset_directory)
            local['assets'] = 'matched'
        if args.transformers_source:
            _require(_hash_file(args.transformers_source, 64 * 1024 * 1024) ==
                     manifest['source_sha256']['installed-transformers-modeling_gpt2.py'],
                     'local Transformers source differs from recorded identity')
            local['transformers_source'] = 'matched'
        arguments = {'expected_manifest_sha256': args.expected_manifest_sha256,
                     'manifest_commit': args.manifest_commit, 'manifest_repo_path': args.manifest_repo_path}
        result = audit_run(args.directory, manifest, **arguments)
        output = {'history': history, 'local_checks': local, 'reference': result}
        if args.replay_directory:
            output['comparison'] = compare_runs(args.directory, args.replay_directory, manifest, **arguments)
    except KeyboardInterrupt:
        print(json.dumps({'status': 'cancelled', 'error': 'evidence verification interrupted'}, sort_keys=True))
        return 130
    except Exception:
        # Paths and other local details can be private. The command-line tool
        # returns a generic failure; callers of the Python API control logging.
        print(json.dumps({'status': 'invalid', 'error': 'evidence verification failed'}, sort_keys=True))
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    success = (result['status'] == 'structurally_consistent'
               and ('comparison' not in output or output['comparison']['status'] == 'matching_saved_checks'))
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
