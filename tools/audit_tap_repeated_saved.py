"""Independent, CPU-only audit of one closed provisional timing session.

This recomputes saved timing arithmetic and source-blob identity. It does not
prove process reaping, activity-screen truth, GPU isolation, or execution of
the recorded operations; those require the separate lifecycle review.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import statistics
import struct
import subprocess

PROTOCOL = 'gpu-tap-latency-repeated-v1.2'
V13_PROTOCOL = 'gpu-tap-latency-repeated-v1.3'
V13_SOURCE_COMMIT = '0c14778447ed6dcbc4890984641370659a09ffef'
V13_PROTOCOL_SOURCE_SHA = '86b475a31d60a2158ee0de3ce0f4fe65068443335927bc0307d0daa8c1bf7503'
SCHEDULE_SHA = 'ce2f990a6937da86565a21c52f13ad54f11b1e683421a8a2060e544694fe6bf9'
GIB = 1024 ** 3
HEADROOM = OUTSIDE_CAP = 256 * 1024 ** 2
MODES = ('none', 'direct', 'gradient')
IDENTITY_FIELDS = ('run_id', 'supervisor_run_id', 'owner_pid', 'owner_birth',
                   'session_index', 'source_commit', 'gpu_uuid')
ASSET_HASHES = {
    'config.json': '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a',
    'generation_config.json': 'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'model.safetensors': 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
}
EXPECTED_MODEL = {'model_id': 'distilbert/distilgpt2', 'model_revision': '2290a62682d06624634c1f46a6ad5be0f47f38aa',
    'weights_sha256': ASSET_HASHES['model.safetensors'], 'asset_sha256': ASSET_HASHES,
    'parameter_count': 81912576, 'attention_implementation': 'eager', 'dtype': 'float32',
    'use_cache': False, 'input_ids': [15496, 995],
    'input_ids_sha256_le_i64': hashlib.sha256(struct.pack('<2q', 15496, 995)).hexdigest(), 'device': 'cuda:0'}
EXPECTED_RUNTIME = {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15',
    'torch_version': '2.8.0+cu128', 'cuda_build': '12.8', 'transformers_version': '4.57.3',
    'loader_versions': {'huggingface_hub': '0.36.2', 'safetensors': '0.8.0', 'tokenizers': '0.22.2', 'numpy': '2.4.6'},
    'allocator_backend': 'native', 'allocator_configuration': 'backend:native', 'device': 'cuda:0',
    'dtype': 'float32', 'attention_implementation': 'eager', 'eval': True, 'use_cache': False,
    'cpu_threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True, 'matmul_tf32': False,
    'cudnn_tf32': False, 'cudnn_benchmark': False, 'cudnn_deterministic': True, 'seed': 17,
    'cublas_workspace_config': ':4096:8', 'allocator_fraction': 0.25, 'cyclic_gc_enabled': False,
    'autocast_enabled': False, 'overwrite_module_params_on_conversion': False,
    'swap_module_params_on_conversion': False}


def fail(message):
    raise ValueError(message)


def check_protocol(expected):
    if type(expected) is not str or expected not in (PROTOCOL, V13_PROTOCOL):
        fail('unsupported timing protocol')
    return expected


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail('duplicate JSON field')
            result[key] = value
        return result
    def constant(_value):
        fail('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def read(path, limit):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        fail('empty or oversized input')
    return data


def record(path, limit=256 * 1024):
    data = read(path, limit)
    value = parse(data)
    if type(value) is not dict or canonical(value) != data:
        fail('noncanonical record')
    return value, data


def same(actual, expected):
    return canonical(actual) == canonical(expected)


def memory(row, total, cap):
    keys = ('free_bytes', 'total_bytes', 'allocated_bytes', 'reserved_bytes')
    if type(row) is not dict or set(row) != set(keys) or any(type(row[k]) is not int for k in keys):
        fail('invalid memory fields')
    free, seen_total, allocated, reserved = (row[k] for k in keys)
    reserve = max(GIB, (15 * total + 99) // 100)
    required = reserve + HEADROOM + OUTSIDE_CAP + max(0, cap - reserved)
    if (seen_total != total or not 0 <= allocated <= reserved <= total
            or not 0 <= free <= total or free + reserved > total
            or reserved > cap - HEADROOM or free < required):
        fail('memory snapshot outside admission bounds')


def expected_order(schedule):
    result = []
    for stage, rounds in (('warmup', schedule['warmup_rounds']),
                          ('measured', schedule['sessions'][0]['measured_rounds'])):
        for number, modes in enumerate(rounds):
            if type(modes) is not list or len(modes) != 3 or set(modes) != set(MODES):
                fail('invalid schedule round')
            for position, mode in enumerate(modes):
                result.append({'stage': stage, 'round': number, 'position': position,
                               'mode': mode, 'observation_id': f'{stage}:{number}:{position}'})
    if len(result) != 42:
        fail('wrong schedule length')
    return result


def audit_rows(rows, schedule, candidate, *, expected_protocol=PROTOCOL):
    check_protocol(expected_protocol)
    if type(rows) is not list or len(rows) != 86:
        fail('journal is not complete')
    total, cap = candidate['cuda_total_bytes'], candidate['cap_bytes']
    if type(total) is not int or total <= 0 or type(cap) is not int or cap != total // 4:
        fail('invalid device total or allocator cap')
    for key in ('admission_memory', 'memory_before_transfer'):
        memory(candidate[key], total, cap)
        if candidate[key]['free_bytes'] < 3 * GIB:
            fail('setup memory below minimum')
    if (type(candidate.get('free_before_model_transfer')) is not int
            or candidate['free_before_model_transfer'] != candidate['memory_before_transfer']['free_bytes']):
        fail('transfer-memory mismatch')
    clock = candidate.get('clock', {})
    resolution = clock.get('resolution_seconds')
    if (clock.get('name') != 'perf_counter_ns' or type(clock.get('implementation')) is not str
            or not clock['implementation'] or clock.get('monotonic') is not True
            or clock.get('adjustable') is not False or type(resolution) not in (int, float)
            or not math.isfinite(resolution) or not 0 < resolution <= 1e-6):
        fail('invalid measurement clock')
    minimum_wall = math.ceil(100 * resolution * 1_000_000_000)
    order = [None] + expected_order(schedule)
    measured, observed, previous_end, previous_guard_end = [], [], None, None
    for index, expected in enumerate(order):
        row, post = rows[2 * index:2 * index + 2]
        obs_id = 'preflight' if expected is None else expected['observation_id']
        stage = 'preflight' if expected is None else expected['stage']
        if (type(row) is not dict or type(post) is not dict or row.get('stage') != stage
                or row.get('observation_id') != obs_id or row.get('postcheck_status') != 'pending'
                or post.get('stage') != 'postcheck' or post.get('observation_id') != obs_id
                or post.get('status') != 'passed'):
            fail('observation or postcheck mismatch')
        memory(row.get('memory_before'), total, cap)
        memory(post.get('memory_after'), total, cap)
        observed.extend((row['memory_before'], post['memory_after']))
        peak_a, peak_r = row.get('peak_allocated_bytes'), row.get('peak_reserved_bytes')
        if (type(peak_a) is not int or type(peak_r) is not int or not 0 <= peak_a <= peak_r <= cap - HEADROOM
                or peak_a < max(row['memory_before']['allocated_bytes'], post['memory_after']['allocated_bytes'])
                or peak_r < max(row['memory_before']['reserved_bytes'], post['memory_after']['reserved_bytes'])):
            fail('invalid peak memory')
        start, end, completed = row.get('started_ns'), row.get('ended_ns'), post.get('completed_ns')
        if (any(type(x) is not int for x in (start, end, completed))
                or not 0 <= start < end <= completed < 2 ** 63 or completed > end + 2_000_000_000
                or (previous_end is not None and start < previous_end)):
            fail('invalid observation chronology')
        prior_end = previous_end
        previous_end = completed
        if expected is None:
            checks = row.get('checks', {})
            if (type(checks) is not dict or checks.get('outputs_finite') is not True or checks.get('outputs_exact') is not True
                    or checks.get('values_exact') is not True or checks.get('id_value_absent') is not True
                    or not same(checks.get('preflight_modes'), ['none', 'id', 'direct', 'gradient'])
                    or not same(checks.get('capture_modes'), ['id', 'direct', 'gradient'])
                    or checks.get('run_id') != f'{expected_protocol}-session-0'
                    or not same(checks.get('trace_sequences'), [0])
                    or not same(checks.get('output_shape'), [1, 2, 50257])
                    or checks.get('output_dtype') != 'torch.float32' or checks.get('output_device') != 'cuda:0'
                    or type(checks.get('events_per_mode')) is not int or checks['events_per_mode'] != 1
                    or type(checks.get('value_tensor_bytes')) is not int or checks['value_tensor_bytes'] != 3072
                    or not same(checks.get('event_identities'), [['transformer.h.0', 'block', 0, 0, 0, [0], [0], [1]]])
                    or not same(checks.get('value_layouts'), [{'shape': [1, 1, 768], 'dtype': 'torch.float32', 'device': 'cuda:0'}])
                    or not same(checks.get('comparison'), {'operator': 'torch.equal', 'tolerance': 'none',
                        'guarantees_bitwise_identity': False, 'subnormal_handling': 'backend-dependent'})):
                fail('preflight checks incomplete')
            ms, me, gl, gu = (row.get(k) for k in ('memory_snapshot_started_ns',
                              'memory_snapshot_ended_ns', 'gate_lower_ns', 'gate_upper_ns'))
            if (any(type(value) is not int for value in (ms, me, gl, gu))
                    or not 0 <= ms <= me <= gl <= start <= gu < 2 ** 63
                    or gu - gl > 250_000_000 or start - ms > 2_000_000_000):
                fail('preflight admission interval differs')
            continue
        if any(type(row.get(k)) is not type(v) or row.get(k) != v for k, v in expected.items()):
            fail('frozen sample order mismatch')
        if type(row.get('session')) is not int or row['session'] != 0:
            fail('sample session mismatch')
        values = (row.get('completion_ns'), row.get('guarded_started_ns'), row.get('guarded_ended_ns'),
                  row.get('guarded_wall_ns'), row.get('wall_ns'), row.get('postcheck_completed_ns'))
        if any(type(x) is not int for x in values):
            fail('invalid timing fields')
        finish, gs, ge, gw, wall, recorded_post = values
        event = row.get('device_event_ms')
        if (not 0 <= gs <= start < finish <= end <= completed <= ge < 2 ** 63
                or (prior_end is not None and gs < prior_end)
                or (previous_guard_end is not None and gs < previous_guard_end)
                or finish - start != wall or ge - gs != gw or recorded_post != completed
                or wall < minimum_wall or type(event) not in (int, float) or not math.isfinite(event)
                or event < 0 or event * 1_000_000 > wall + 100_000
                or not same(row.get('memory_after'), post['memory_after'])
                or post.get('device_work_complete') is not True):
            fail('timing, completion or memory mismatch')
        previous_guard_end = ge
        trace = row['mode'] != 'none'
        if (row.get('output_finite') is not True or row.get('default_stream') is not True
                or not same(row.get('output_shape'), [1, 2, 50257]) or row.get('output_dtype') != 'torch.float32'
                or row.get('output_device') != 'cuda:0' or type(row.get('events')) is not int
                or row['events'] != int(trace) or type(row.get('recorded_tensor_bytes')) is not int
                or row['recorded_tensor_bytes'] != (3072 if trace else 0)
                or row.get('capture_finite') is not (True if trace else None)):
            fail('output or stream check missing')
        if trace:
            if (row.get('trace_run_id') != f'{expected_protocol}-session-0' or row.get('trace_mode') != row['mode']
                    or not same(row.get('trace_identity'), ['transformer.h.0', 'block', 0, 0, 0, [0], [0], [1]])
                    or type(row.get('trace_sequence')) is not int or row['trace_sequence'] != 0
                    or not same(row.get('value_shape'), [1, 1, 768]) or row.get('value_dtype') != 'torch.float32'
                    or row.get('value_device') != 'cuda:0'):
                fail('trace attribution differs')
        elif any(row.get(k) is not None for k in ('trace_identity', 'trace_sequence', 'trace_run_id',
                                                   'trace_mode', 'value_shape', 'value_dtype', 'value_device')):
            fail('unexpected trace in baseline')
        if expected['stage'] == 'measured':
            measured.append(row)
    if (type(candidate.get('preflight_started_ns')) is not int or type(candidate.get('workload_ended_ns')) is not int
            or candidate['preflight_started_ns'] != rows[0]['started_ns']
            or candidate['workload_ended_ns'] != rows[-1]['completed_ns']):
        fail('journal endpoint mismatch')
    groups = {}
    for row in measured:
        groups.setdefault(row['round'], {})[row['mode']] = row
    if set(groups) != set(range(12)) or any(set(group) != set(MODES) for group in groups.values()):
        fail('measured pairs incomplete')
    pairs = []
    for number in range(12):
        group = groups[number]
        n, d, g = (group[mode]['wall_ns'] for mode in MODES)
        pairs.append({'round': number, 'gradient_over_direct': g / d, 'direct_over_none': d / n,
                      'gradient_minus_direct_ms': (g - d) / 1_000_000,
                      'direct_minus_none_ms': (d - n) / 1_000_000,
                      'guarded_wall_ns': {mode: group[mode]['guarded_wall_ns'] for mode in MODES}})
    summary = {'session': 0, 'warmup_samples': 6, 'measured_samples': 36, 'paired_rounds': pairs}
    for name in ('gradient_over_direct', 'direct_over_none', 'gradient_minus_direct_ms', 'direct_minus_none_ms'):
        summary['median_' + name] = statistics.median(row[name] for row in pairs)
    for field in ('wall_ns', 'device_event_ms', 'guarded_wall_ns'):
        summary['mode_' + field + '_medians'] = {
            mode: statistics.median(row[field] for row in measured if row['mode'] == mode) for mode in MODES}
    samples = [row for row in rows if row['stage'] in ('preflight', 'warmup', 'measured')]
    all_saved = [candidate['admission_memory'], candidate['memory_before_transfer']] + observed
    resources = {'maximum_recorded_reset_peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in samples),
                 'maximum_recorded_reset_peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in samples),
                 'minimum_free_bytes_in_admission_pretransfer_and_journal': min(r['free_bytes'] for r in all_saved),
                 'guarded_wall_ns_medians': summary['mode_guarded_wall_ns_medians']}
    return summary, resources


def git(git_exe, checkout, *args):
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT') if key in os.environ}
    env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_COUNT='0', GIT_TERMINAL_PROMPT='0')
    return subprocess.run([str(git_exe), '--no-replace-objects', *args], cwd=checkout, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30).stdout


def verify_source(checkout, git_exe, candidate, started, manifest):
    commit = candidate['source_commit']
    if type(commit) is not str or re.fullmatch('[0-9a-f]{40}', commit) is None:
        fail('invalid source revision')
    if git(git_exe, checkout, 'rev-parse', 'HEAD').decode().strip() != commit:
        fail('checkout revision differs')
    tree = git(git_exe, checkout, 'rev-parse', 'HEAD:').decode().strip()
    if tree != started.get('source_tree') or git(git_exe, checkout, 'status', '--porcelain', '--untracked-files=all'):
        fail('checkout tree or status differs')
    raw_tree = git(git_exe, checkout, 'ls-tree', '-r', '-z', '--full-tree', commit)
    entries = {}
    for item in raw_tree.split(b'\0'):
        if not item:
            continue
        info, path = item.split(b'\t', 1)
        mode, kind, blob = info.split(b' ')
        entries[path.decode()] = (mode.decode(), kind.decode(), blob.decode())
    sources = manifest.get('source_files')
    if type(sources) is not dict or not sources:
        fail('missing source inventory')
    for name, metadata in sources.items():
        p = PurePosixPath(name)
        if type(name) is not str or str(p) != name or p.is_absolute() or any(part in ('', '.', '..') for part in p.parts):
            fail('invalid source path')
        if (type(metadata) is not dict or set(metadata) != {'mode', 'sha256'}
                or metadata['mode'] not in ('100644', '100755')):
            fail('invalid source metadata')
        path = checkout.joinpath(*p.parts)
        if not path.is_file() or not path.resolve(strict=True).is_relative_to(checkout):
            fail('source outside checkout')
        for item in (path, *path.parents):
            if item == checkout:
                break
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                fail('indirect source path')
        raw = read(path, 4 * 1024 * 1024)
        digest = hashlib.sha256(raw).hexdigest()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if digest != metadata['sha256'] or entries.get(name) != (metadata['mode'], 'blob', blob):
            fail('source bytes or Git blob differ')
    source = read(checkout / 'looking_glass/tap_repeated_gpu.py', 4 * 1024 * 1024)
    tree_ast = ast.parse(source)
    bindings = [node for node in ast.walk(tree_ast) if isinstance(node, ast.Name)
                and node.id == '_SOURCE_PATHS' and isinstance(node.ctx, ast.Store)]
    values = [ast.literal_eval(node.value) for node in tree_ast.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '_SOURCE_PATHS']
    if (len(bindings) != 1 or len(values) != 1 or type(values[0]) is not tuple
            or len(values[0]) != len(set(values[0])) or set(values[0]) != set(sources)):
        fail('source inventory differs from controller')
    return tree, len(sources)


def check_private_identity(candidate):
    for key in ('run_id', 'supervisor_run_id'):
        if type(candidate.get(key)) is not str or re.fullmatch('[0-9a-f]{32}', candidate[key]) is None:
            fail('invalid private run identity')
    pid, birth, uuid = (candidate.get(key) for key in ('owner_pid', 'owner_birth', 'gpu_uuid'))
    if (type(pid) is not int or not 0 < pid < 2 ** 32 or type(birth) is not str
            or re.fullmatch('[1-9][0-9]{0,19}', birth) is None or int(birth) >= 2 ** 64
            or type(uuid) is not str
            or re.fullmatch('GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}', uuid) is None):
        fail('invalid private owner identity')


def check_v13_source(checkout, manifest):
    """Check the verified source tree declares the intended v1.3 schedule and grant."""
    name = 'looking_glass/tap_repeated_protocol.py'
    if name not in manifest['source_files']:
        fail('versioned protocol source is absent from the verified inventory')
    source = read(checkout / name, 4 * 1024 * 1024)
    if hashlib.sha256(source).hexdigest() != V13_PROTOCOL_SOURCE_SHA:
        fail('verified v1.3 protocol source differs from the frozen bytes')
    tree = ast.parse(source)
    expected = {'EXPERIMENT_PROTOCOL': V13_PROTOCOL, 'SCHEDULE_SHA256': SCHEDULE_SHA,
                'START_GRANT_MAX_NS': 1_000_000_000}
    for name, value in expected.items():
        stores = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
                  and node.id == name and isinstance(node.ctx, ast.Store)]
        assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == name]
        try:
            actual = ast.literal_eval(assignments[0]) if len(assignments) == 1 else None
        except (ValueError, TypeError):
            actual = None
        if len(stores) != 1 or type(actual) is not type(value) or actual != value:
            fail('verified source does not declare the v1.3 protocol')


def check_v13_barriers(run_dir, candidate, rows, public):
    """Recheck saved grant, preflight and cleanup consistency, not process execution."""
    grant, _ = record(run_dir / 'start-grant.json', 4096)
    finished, _ = record(run_dir / 'finished.json', 4096)
    cleanup, _ = record(run_dir / 'cleanup-grant.json', 4096)
    identity = ('session_index', 'owner_pid', 'owner_birth', 'supervisor_run_id', 'challenge',
                'gpu_uuid', 'cuda_total_bytes', 'activity_smi_total_bytes', 'clock', 'runtime')
    timestamps = ('ready_ns', 'received_ns', 'issued_ns', 'expires_ns',
                  'initial_oldest_query_started_ns', 'initial_latest_query_ended_ns',
                  'initial_process_started_ns', 'initial_process_ended_ns')
    grant_keys = {'schema', *identity, *timestamps}
    finish_keys = {'schema', *identity, 'grant_seen_ns', 'preflight_started_ns', 'workload_ended_ns'}
    cleanup_keys = {'schema', *identity, 'workload_ended_ns', 'issued_ns', 'expires_ns', 'screen_availability'}
    if (set(grant) != grant_keys or grant['schema'] != 'tap-repeated-start-grant-v1'
            or set(finished) != finish_keys or finished['schema'] != 'tap-repeated-workload-finished-v1'
            or set(cleanup) != cleanup_keys or cleanup['schema'] != 'tap-repeated-cleanup-grant-v1'
            or any(not same(finished.get(key), grant.get(key)) or not same(cleanup.get(key), grant.get(key))
                   for key in identity)
            or any(not same(grant.get(key), candidate.get(key)) for key in
                   ('session_index', 'owner_pid', 'owner_birth', 'supervisor_run_id', 'gpu_uuid', 'cuda_total_bytes'))
            or not same(grant.get('clock'), candidate.get('clock'))
            or grant['clock'].get('implementation') != 'QueryPerformanceCounter()'
            or not same(grant.get('runtime'), {'platform': 'win32', 'implementation': 'cpython',
                                             'python_version': '3.11.15'})
            or type(grant.get('challenge')) is not str
            or re.fullmatch('[0-9a-f]{64}', grant['challenge']) is None
            or type(grant.get('activity_smi_total_bytes')) is not int
            or grant['activity_smi_total_bytes'] != public['slot']['device_totals']['nvidia_smi_bytes']
            or any(type(grant[key]) is not int for key in timestamps)):
        fail('saved v1.3 barrier identity differs')
    ready, received, issued, expires, oldest, latest, process_start, process_end = (grant[key] for key in timestamps)
    first = rows[0]
    seen, start, end = (finished.get(key) for key in ('grant_seen_ns', 'preflight_started_ns', 'workload_ended_ns'))
    lower, upper, memory_start, memory_end = (first.get(key) for key in
        ('gate_lower_ns', 'gate_upper_ns', 'memory_snapshot_started_ns', 'memory_snapshot_ended_ns'))
    cleanup_issued, cleanup_expires = cleanup.get('issued_ns'), cleanup.get('expires_ns')
    values = (seen, start, end, lower, upper, memory_start, memory_end, cleanup_issued, cleanup_expires,
              cleanup.get('workload_ended_ns'))
    if any(type(value) is not int for value in values):
        fail('saved v1.3 barrier timestamps are invalid')
    margin = 2 * max(1, math.ceil(grant['clock']['resolution_seconds'] * 1_000_000_000))
    expected_expiry = min(issued + 1_000_000_000, oldest + 2_000_000_000 - margin,
                          process_start + 2_000_000_000 - margin)
    if (not 0 <= ready + margin <= received <= oldest <= latest < process_start <= process_end <= issued < expires
            or received - ready > 10_000_000_000 or expires != expected_expiry
            or not issued + margin <= seen <= memory_start <= memory_end <= lower <= start <= upper <= expires < 2 ** 63
            or upper - lower > 250_000_000 or start - memory_start > 2_000_000_000
            or latest + margin > lower or process_end + margin > lower
            or upper - oldest + margin > 2_000_000_000 or upper - process_start + margin > 2_000_000_000
            or start != first['started_ns'] or end != rows[-1]['completed_ns']
            or cleanup['workload_ended_ns'] != end
            or not end + margin <= cleanup_issued < cleanup_expires < 2 ** 63
            or cleanup_expires != cleanup_issued + 5_000_000_000
            or cleanup['screen_availability'] != public['slot']['screen']['availability']):
        fail('saved v1.3 grant or preflight interval differs')


def audit(run_dir, public_path, checkout, git_exe, controller_exit, *, expected_protocol=PROTOCOL):
    check_protocol(expected_protocol)
    if type(controller_exit) is not int or controller_exit != 0:
        fail('controller did not exit successfully')
    if os.path.lexists(Path(str(public_path) + '.incomplete.json')):
        fail('an incomplete sidecar exists')
    result, public_raw = record(public_path)
    if result.get('status') not in ('completed_provisional', 'completed_exploratory_contaminated'):
        fail('public result is not a completed session')
    started, _ = record(run_dir / 'started.json')
    manifest, manifest_raw = record(run_dir / 'protected-manifest/source-manifest.json')
    candidate, candidate_raw = record(run_dir / 'candidate.json')
    final, _ = record(run_dir / 'final.json')
    check_private_identity(candidate)
    if (manifest.get('schema') != 'tap-repeated-source-manifest-v1'
            or started.get('schema') != 'tap-repeated-controller-start-v1'
            or candidate.get('schema') != 'tap-repeated-candidate-v1'
            or final.get('schema') != 'tap-repeated-final-v1'
            or manifest.get('source_commit') != candidate.get('source_commit')
            or manifest.get('run_id') != candidate.get('run_id')
            or started.get('source_manifest_sha256') != hashlib.sha256(manifest_raw).hexdigest()
            or started.get('run_id') != candidate.get('run_id')
            or type(started.get('session')) is not int or started['session'] != 0
            or started.get('source_commit') != candidate.get('source_commit')
            or final.get('candidate_sha256') != hashlib.sha256(candidate_raw).hexdigest()
            or any(type(final.get(k)) is not type(candidate.get(k)) or final.get(k) != candidate.get(k)
                   for k in IDENTITY_FIELDS)):
        fail('record identity or digest mismatch')
    if (candidate.get('protocol') != expected_protocol or type(candidate.get('session_index')) is not int
            or candidate['session_index'] != 0
            or type(candidate.get('journal_rows')) is not int or candidate['journal_rows'] != 86
            or candidate.get('whole_run_claims_held') is not True
            or final.get('unsafe') is not False or final.get('synchronized_disposal') is not True
            or final.get('claims_released_after_disposal') is not True
            or final.get('device_lease_held_at_attestation') is not True
            or any(type(final.get(k)) is not int or final[k] != 0 for k in
                   ('allocated_bytes', 'reserved_bytes', 'active_memory_leases', 'trace_claim_remaining_bytes'))):
        fail('owner completion evidence incomplete')
    if expected_protocol == V13_PROTOCOL and candidate.get('source_commit') != V13_SOURCE_COMMIT:
        fail('v1.3 execution source differs from the frozen attempt')
    if (not same(candidate.get('model'), EXPECTED_MODEL) or not same(candidate.get('runtime'), EXPECTED_RUNTIME)
            or type(candidate.get('trace_claim_bytes')) is not int or candidate['trace_claim_bytes'] != 128 * 1024):
        fail('model, runtime or trace claim differs')
    tree, source_count = verify_source(checkout, git_exe, candidate, started, manifest)
    if expected_protocol == V13_PROTOCOL:
        check_v13_source(checkout, manifest)
    if result.get('source_commit') != candidate['source_commit'] or result.get('source_tree') != tree:
        fail('public source identity differs')
    schedule_raw = read(checkout / 'evidence/gpu-tap-latency-repeated-v1/schedule.json', 32 * 1024)
    if hashlib.sha256(schedule_raw).hexdigest() != SCHEDULE_SHA or candidate.get('schedule_sha256') != SCHEDULE_SHA:
        fail('schedule hash differs')
    schedule = parse(schedule_raw)
    journal = read(run_dir / 'journal.jsonl', 8 * 1024 * 1024)
    if hashlib.sha256(journal).hexdigest() != candidate.get('journal_sha256'):
        fail('journal digest differs')
    lines = journal.splitlines(keepends=True)
    rows = [parse(line) for line in lines]
    if any(canonical(row) != line for row, line in zip(rows, lines)):
        fail('noncanonical journal row')
    summary, resources = audit_rows(rows, schedule, candidate, expected_protocol=expected_protocol)
    slot = result.get('slot', {})
    screen = slot.get('screen', {})
    device = result.get('device', {})
    if (set(result) != {'schema', 'protocol', 'status', 'source_commit', 'source_tree', 'session', 'slot',
                         'device', 'resources', 'coordination', 'limitations'}
            or result['schema'] != 'tap-repeated-provisional-session-v1'
            or type(slot) is not dict or set(slot) != {'session', 'state', 'screen', 'summary', 'device_totals'}
            or type(screen) is not dict or set(screen) != {'status', 'availability', 'contamination_reasons',
                'unavailable_reasons', 'before_utilization_percent', 'after_utilization_percent',
                'telemetry_samples', 'foreign_process_count'}
            or type(device) is not dict or set(device) != {'name', 'driver_point_observation', 'compute_capability'}
            or type(result.get('coordination')) is not dict
            or set(result['coordination']) != {'gpu_slot', 'cpu_context'}
            or result['coordination'].get('gpu_slot') != 'operator_coordinated'
            or result['coordination'].get('cpu_context') not in ('targeted_pause', 'targeted_pause_other_work', 'unknown')
            or type(result.get('limitations')) is not list or any(type(x) is not str for x in result['limitations'])
            or type(device.get('driver_point_observation')) is not str
            or re.fullmatch('[0-9]+(?:\\.[0-9]+){1,3}', device['driver_point_observation']) is None
            or result.get('protocol') != expected_protocol or type(result.get('session')) is not int or result['session'] != 0
            or slot.get('state') != 'complete' or type(slot.get('session')) is not int or slot['session'] != 0
            or not same(slot.get('summary'), summary) or not same(result.get('resources'), resources)
            or device.get('name') != candidate.get('cuda_device_name')
            or not same(device.get('compute_capability'), candidate.get('cuda_compute_capability'))
            or screen.get('availability') != 'complete'
            or screen.get('status') not in ('no_contamination_observed', 'contaminated')
            or (result['status'] == 'completed_provisional') != (screen['status'] == 'no_contamination_observed')):
        fail('public summary or status differs')
    totals = slot['device_totals']
    if (type(totals) is not dict or set(totals) != {'cuda_bytes', 'nvidia_smi_bytes', 'maximum_difference_bytes'}
            or any(type(totals[k]) is not int for k in totals)
            or totals['cuda_bytes'] != candidate['cuda_total_bytes'] or totals['nvidia_smi_bytes'] <= 0
            or totals['maximum_difference_bytes'] != 2 * 1024 ** 2
            or abs(totals['cuda_bytes'] - totals['nvidia_smi_bytes']) > totals['maximum_difference_bytes']):
        fail('public device totals differ')
    telemetry = screen['telemetry_samples']
    if (type(telemetry) is not list or type(screen['foreign_process_count']) is not int
            or screen['foreign_process_count'] < 0
            or any(type(screen[k]) is not list or any(type(x) is not int or not 0 <= x <= 100 for x in screen[k])
                   for k in ('before_utilization_percent', 'after_utilization_percent'))
            or any(type(screen[k]) is not list or any(type(x) is not str for x in screen[k])
                   for k in ('contamination_reasons', 'unavailable_reasons'))):
        fail('public activity projection differs')
    point_keys = {'query_started_ns', 'query_ended_ns', 'observed_ns', 'total_bytes',
                  'used_bytes', 'free_bytes', 'utilization_percent'}
    if any(type(point) is not dict or set(point) != point_keys
           or any(type(point[k]) is not int for k in point_keys) for point in telemetry):
        fail('public telemetry projection differs')
    for forbidden in ('gpu_uuid', 'owner_pid', 'supervisor_run_id', 'foreign_process_ids', 'run_directory'):
        if ('"' + forbidden + '"').encode() in public_raw:
            fail('private field in public projection')
    def strings(value):
        if type(value) is str:
            yield value
        elif type(value) is dict:
            for key, item in value.items():
                yield from strings(key)
                yield from strings(item)
        elif type(value) is list:
            for item in value:
                yield from strings(item)
    public_strings = tuple(strings(result))
    if any(re.search(r'(?i)\b[a-z]:[\\/]', value) for value in public_strings):
        fail('path in public projection')
    for private in (candidate['gpu_uuid'], candidate['run_id'], candidate['supervisor_run_id'], str(run_dir)):
        if (type(private) is str and private
                and any(private.casefold() in value.casefold() for value in public_strings)):
            fail('private identifier in public projection')
    if expected_protocol == V13_PROTOCOL:
        check_v13_barriers(run_dir, candidate, rows, result)
    output = {'schema': 'tap-v12-independent-timing-audit-v1', 'status': 'passed',
            'source_commit': candidate['source_commit'], 'source_tree': tree,
            'source_files_checked': source_count, 'journal_sha256': candidate['journal_sha256'],
            'schedule_sha256': SCHEDULE_SHA, 'measured_samples': 36,
            'screen_status_as_reported': screen['status'],
            'median_gradient_over_direct': summary['median_gradient_over_direct'],
            'median_direct_over_none': summary['median_direct_over_none'],
            'scope': 'saved source and timing arithmetic only; process closure and telemetry require separate review'}
    if expected_protocol == V13_PROTOCOL:
        output.update(schema='tap-v13-independent-timing-audit-v1', protocol=V13_PROTOCOL,
                      pooling='single_session_only')
    return output


def audit_incomplete(run_dir, public_path, controller_exit, *, expected_protocol=PROTOCOL):
    """Describe existing partial records without treating them as measured results."""
    check_protocol(expected_protocol)
    if expected_protocol == V13_PROTOCOL and type(controller_exit) is not int:
        fail('controller exit observation must be an integer')
    def check_available_identity(value):
        if (('protocol' in value and value['protocol'] != V13_PROTOCOL)
                or ('source_commit' in value and value['source_commit'] != V13_SOURCE_COMMIT)):
            fail('incomplete record names a different protocol or source')
    sidecar = Path(str(public_path) + '.incomplete.json')
    selected = sidecar if os.path.lexists(sidecar) else public_path
    status = None
    if selected.is_file():
        value, _ = record(selected)
        status = value.get('status')
        if expected_protocol == V13_PROTOCOL:
            check_available_identity(value)
    if status != 'incomplete' and controller_exit == 0:
        fail('no incomplete outcome was established')
    if expected_protocol == V13_PROTOCOL:
        for name in ('candidate.json', 'started.json'):
            path = run_dir / name
            if path.is_file():
                try:
                    observed, _ = record(path)
                except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
                    observed = None
                if observed is not None:
                    check_available_identity(observed)
    path = run_dir / 'journal.jsonl'
    count, digest, torn = 0, None, False
    if path.is_file():
        with path.open('rb') as stream:
            raw = stream.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            fail('partial journal exceeds its bound')
        digest = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines(keepends=True)[:86]:
            if not line.endswith(b'\n'):
                torn = True
                break
            try:
                value = parse(line)
                if type(value) is not dict or canonical(value) != line:
                    torn = True
                    break
            except (TypeError, ValueError, UnicodeError, RecursionError):
                torn = True
                break
            if expected_protocol == V13_PROTOCOL:
                checks = value.get('checks')
                preflight_id = checks.get('run_id') if type(checks) is dict else None
                trace_id = value.get('trace_run_id')
                for identity in (preflight_id, trace_id):
                    if identity is not None and (type(identity) is not str
                            or re.fullmatch(re.escape(V13_PROTOCOL) + r'-session-[0-4]', identity) is None):
                        fail('incomplete journal names a different protocol')
            count += 1
    output = {'schema': 'tap-v12-independent-timing-audit-v1', 'status': 'incomplete',
            'controller_exit_as_reported': controller_exit, 'parseable_journal_rows_unvalidated': count,
            'partial_journal_sha256': digest, 'partial_or_invalid_final_line': torn,
            'scope': 'partial records only; no completed timing, process-closure or telemetry claim'}
    if expected_protocol == V13_PROTOCOL:
        output.update(schema='tap-v13-independent-timing-audit-v1', expected_protocol=V13_PROTOCOL,
                      protocol_verification='not_established')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--public-result', required=True, type=Path)
    parser.add_argument('--checkout', required=True, type=Path)
    parser.add_argument('--git', required=True, type=Path)
    parser.add_argument('--controller-exit', required=True, type=int)
    parser.add_argument('--protocol', choices=(PROTOCOL, V13_PROTOCOL), default=PROTOCOL)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    public = args.public_result.absolute()
    sidecar = Path(str(public) + '.incomplete.json')
    if args.controller_exit != 0 or os.path.lexists(sidecar) or not public.is_file():
        result = audit_incomplete(run_dir, public, args.controller_exit, expected_protocol=args.protocol)
    else:
        value, _ = record(public)
        if value.get('status') == 'incomplete':
            result = audit_incomplete(run_dir, public, args.controller_exit, expected_protocol=args.protocol)
        else:
            result = audit(run_dir, public, args.checkout.resolve(strict=True),
                           args.git.resolve(strict=True), args.controller_exit, expected_protocol=args.protocol)
    print(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
