"""Replay one public short-service result and its CPU reference using saved bytes.

No private candidate, PID, lease, or process receipt is read. Source and hash
commitments do not independently establish execution or native CUDA cleanup.
The CPU/GPU comparison is descriptive, not a preregistered cross-runtime gate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

CPU_SHA = 'e56f9860b42eab1f7651d988b8120884d22e2e140e22e12bd87ceb462f794051'
MANIFEST_SHA = '47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c'
INVENTORY_SHA = 'e586bae02145a62a7163eb030485fa85a33f9e94cd0949a9f88f3f74c7de194b'
VOCAB = 50257
ATOL, RTOL = 1e-6, 1e-5
GIB = 1024 ** 3
NAMES = ('reference-alpha', 'reference-beta', 'client-alpha', 'client-beta')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum, expected=None):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum and (expected is None or sha(raw) == expected),
            'file size or hash differs')
    return raw


def parse(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, 'duplicate JSON key')
            obj[key] = value
        return obj
    def invalid(_):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def same(left, right):
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return canonical(left) == canonical(right)


def comparison(left, right):
    require(type(left) is bytes and type(right) is bytes and len(left) == len(right)
            and len(left) > 0 and len(left) % 4 == 0, 'float32 layout differs')
    count = violations = 0
    maximum = 0.0
    for (a,), (b,) in zip(struct.iter_unpack('<f', left), struct.iter_unpack('<f', right), strict=True):
        require(math.isfinite(a) and math.isfinite(b), 'nonfinite saved logit')
        distance = abs(a - b)
        maximum = max(maximum, distance)
        violations += distance > ATOL + RTOL * abs(b)
        count += 1
    return {'count': count, 'violations': violations, 'max_abs': maximum,
            'atol': ATOL, 'rtol': RTOL}


def public_memory(gpu_dir, measurement):
    rows, hashes = measurement['memory'], measurement['memory_sha256']
    require(type(rows) is list and len(rows) == 21 and type(hashes) is dict,
            'memory inventory differs')
    names = [f'memory-{i:03d}.json' for i in range(len(rows))]
    require(set(hashes) == set(names), 'memory file names differ')
    fields = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
              'peak_allocated_bytes', 'peak_reserved_bytes')
    total = None
    stages = []
    for name, row in zip(names, rows, strict=True):
        raw = read(gpu_dir / name, 4096, hashes[name])
        require(type(row) is dict and set(row) == set(fields) | {'stage'}
                and same(row, parse(raw)) and type(row['stage']) is str
                and all(type(row[key]) is int and row[key] >= 0 for key in fields),
                'memory row differs')
        if total is None:
            total = row['device_total_bytes']
        require(row['device_total_bytes'] == total and 7 * GIB <= total <= 9 * GIB
                and row['device_free_bytes'] <= total
                and row['allocated_bytes'] <= row['reserved_bytes']
                and row['device_free_bytes'] + row['reserved_bytes'] <= total
                and row['allocated_bytes'] <= row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
                and row['reserved_bytes'] <= row['peak_reserved_bytes'] <= total // 4,
                'inconsistent saved CUDA counters')
        stages.append(row['stage'])
    ordered = ('before-model-load', 'after-model-load', 'before-reference-alpha',
               'after-reference-alpha', 'before-reference-beta', 'after-reference-beta',
               'before-two-clients', 'after-two-clients', 'after-worker-close')
    require(all(stages.count(stage) == 1 for stage in ordered)
            and [stage for stage in stages if stage in ordered] == list(ordered)
            and all(stage in ordered or stage == 'admission' for stage in stages),
            'memory stage order differs')
    cleanup_raw = read(gpu_dir / 'cleanup.json', 4096, measurement['cleanup_sha256'])
    cleanup = parse(cleanup_raw)
    require(same(cleanup, measurement['cleanup'])
            and same(cleanup.get('after'), {'allocated_bytes': 0, 'reserved_bytes': 0})
            and cleanup.get('model_released') is True and cleanup.get('input_output_tensors_released') is True
            and type(cleanup.get('accounting_leases_after')) is int and cleanup['accounting_leases_after'] == 0
            and type(cleanup.get('host_claims_after')) is int and cleanup['host_claims_after'] == 0,
            'cleanup projection differs')
    return {'count': len(rows), 'cuda_total_bytes': total,
            'minimum_sampled_device_free_bytes': min(row['device_free_bytes'] for row in rows),
            'maximum_process_peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in rows),
            'maximum_process_peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in rows),
            'cleanup_sha256': sha(cleanup_raw), 'cleanup_after': cleanup['after'],
            'model_and_known_tensors_released_recorded': True, 'accounting_leases_after': 0}


def audit(cpu_dir, gpu_dir, manifest_path, source, bucket, result_sha):
    require(type(bucket) is int and bucket in (16, 32), 'unsupported bucket')
    require(type(source) is str and re.fullmatch(r'[0-9a-f]{40}', source) is not None
            and type(result_sha) is str and re.fullmatch(r'[0-9a-f]{64}', result_sha) is not None,
            'exact source and result hash required')
    cpu_raw = read(cpu_dir / 'candidate.json', 128 * 1024, CPU_SHA)
    cpu = parse(cpu_raw)
    require(cpu['schema'] == 'short-service-cpu-reference-v1' and cpu['status'] == 'passed'
            and cpu['inventory_sha256'] == INVENTORY_SHA, 'CPU reference identity differs')
    manifest = parse(read(manifest_path, 65536, MANIFEST_SHA))
    manifest_index = {item['id']: item for item in manifest['comparisons']}
    require(len(manifest_index) == len(manifest['comparisons']), 'duplicate input case')
    result_raw = read(gpu_dir / 'result.json', 1024 * 1024, result_sha)
    result = parse(result_raw)
    require(result.get('schema') == 'short-service-gpu-observation-v1' and result.get('status') == 'passed'
            and type(result.get('bucket')) is int and result['bucket'] == bucket
            and result.get('source_commit') == source, 'public result identity differs')
    measurement = result['measurement']
    require(type(measurement.get('bucket')) is int and measurement['bucket'] == bucket,
            'measurement bucket differs')
    for field in ('candidate_sha256', 'owner_receipt_sha256', 'source_manifest_sha256'):
        require(type(measurement.get(field)) is str
                and re.fullmatch(r'[0-9a-f]{64}', measurement[field]) is not None,
                'private commitment missing')
    fields = ('model_id', 'model_revision', 'asset_sha256', 'weights_sha256', 'parameter_count',
              'model_class', 'dtype', 'attention_implementation', 'use_cache', 'torch_version',
              'transformers_version')
    require(all(field in cpu['model_metadata'] and field in measurement['model_metadata']
                and same(cpu['model_metadata'][field], measurement['model_metadata'][field])
                for field in fields), 'recorded model metadata differs')
    case_ids = ('synthetic-16', 'gas') if bucket == 16 else ('synthetic-32', 'ligand')
    roles = ('reference', 'service')
    cases = measurement['cases']
    require(type(cases) is list and len(cases) == 2
            and [case['case_id'] for case in cases] == list(case_ids), 'case order differs')
    cpu_index = {(entry['case'], entry['role']): entry for entry in cpu['entries']}
    require(len(cpu_index) == len(cpu['entries']), 'duplicate CPU case')
    cpu_rows = []
    for case, role, gpu_case in zip(case_ids, roles, cases, strict=True):
        entry = cpu_index[(case, role)]
        stem = case + '-' + role
        start = parse(read(cpu_dir / (stem + '-start.json'), 32768))
        require(same(start, {'case': case, 'role': role, **manifest_index[case][role]})
                and same(gpu_case, {'case_id': case, 'input_ids': start['input_ids'],
                                    'attention_mask': start['attention_mask'],
                                    'active_position': sum(start['attention_mask']) - 1})
                and same(entry['shape'], [1, bucket, VOCAB]) and entry['filename'] == stem + '.f32'
                and type(entry['bytes']) is int and entry['bytes'] == bucket * VOCAB * 4,
                'CPU/GPU inputs or layout differ')
        raw = read(cpu_dir / entry['filename'], bucket * VOCAB * 4, entry['sha256'])
        require(len(raw) == entry['bytes'], 'CPU logit size differs')
        cpu_rows.append(raw)
    entry_hashes = measurement['model_entry_sha256']
    require(type(entry_hashes) is dict and set(entry_hashes) == {f'forward-{i:03d}-entry.json' for i in range(1, 5)},
            'model entry inventory differs')
    entry_cases = []
    cases_by_id = {case['case_id']: case for case in cases}
    for number in range(1, 5):
        name = f'forward-{number:03d}-entry.json'
        entry = parse(read(gpu_dir / name, 4096, entry_hashes[name]))
        case = cases_by_id[entry['case_id']]
        require(same(entry, {'entry': number, 'shape': [1, bucket], 'dtype': 'torch.int64',
                             'worker_device': 'cuda:0', 'case_id': case['case_id'],
                             'input_ids': case['input_ids'], 'attention_mask': case['attention_mask']}),
                'model entry differs from declared case')
        entry_cases.append(case['case_id'])
    require(entry_cases[:2] == list(case_ids) and sorted(entry_cases[2:]) == sorted(case_ids),
            'model entry ordering differs')
    hashes = measurement['output_sha256']
    require(type(hashes) is dict and set(hashes) == set(NAMES), 'GPU output inventory differs')
    payloads = {}
    for name in NAMES:
        raw = read(gpu_dir / (name + '.f32'), bucket * VOCAB * 4, hashes[name])
        require(len(raw) == bucket * VOCAB * 4, 'GPU full-logit size differs')
        payloads[name] = raw
    positions = [case['active_position'] for case in cases]
    def row(name, position):
        start = position * VOCAB * 4
        return payloads[name][start:start + VOCAB * 4]
    references = [row('reference-alpha', positions[0]), row('reference-beta', positions[1])]
    separation = comparison(*references)
    route = []
    require(separation['violations'] > 0, 'serial references are not distinguishable')
    for index, team in enumerate(('alpha', 'beta')):
        name = 'client-' + team
        full = comparison(payloads[name], payloads['reference-' + team])
        own = comparison(row(name, positions[index]), references[index])
        other = comparison(row(name, positions[index]), references[1 - index])
        require(full['violations'] == own['violations'] == 0 and other['violations'] > 0,
                'public routing comparison differs')
        route.append({'team': team, 'active_position': positions[index], 'full': full,
                      'own_last': own, 'other_last': other})
    require(same({'reference_separation': separation, 'comparisons': route}, measurement['routing_readback']),
            'saved routing summary differs')
    cross = []
    for index, name in enumerate(NAMES):
        case = index % 2
        start = positions[case] * VOCAB * 4
        cross.append({'gpu_output': name, 'case': case_ids[case], 'cpu_role': roles[case],
                      'gpu_sha256': sha(payloads[name]),
                      'full': {**comparison(payloads[name], cpu_rows[case]), 'reference': 'saved_cpu'},
                      'last_active': {**comparison(payloads[name][start:start + VOCAB * 4],
                                                   cpu_rows[case][start:start + VOCAB * 4]),
                                      'reference': 'saved_cpu'}})
    return {'schema': 'short-service-public-cpu-gpu-replay-v1', 'status': 'descriptive',
            'bucket': bucket, 'source_commit': source, 'gpu_result_sha256': result_sha,
            'cpu_candidate_sha256': sha(cpu_raw), 'manifest_sha256': MANIFEST_SHA,
            'private_candidate_sha256_commitment': measurement['candidate_sha256'],
            'memory': public_memory(gpu_dir, measurement), 'routing': {'reference_separation': separation,
                                                                      'comparisons': route},
            'cpu_gpu_comparisons': cross,
            'scope': ['saved-byte comparison; CPU is the descriptive cross-runtime reference',
                      'full comparisons include padded positions; last-active comparisons do not',
                      'CPU/GPU tolerance is descriptive, not a preregistered cross-runtime pass gate',
                      'memory free is sampled device-wide; allocator peaks are process-local',
                      'cleanup is an owner record, not proof of native or global device teardown',
                      'private hashes are commitments; this replay does not authenticate execution or process custody']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-dir', type=Path, required=True)
    parser.add_argument('--gpu-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--bucket', type=int, choices=(16, 32), required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(all(path.is_absolute() for path in (args.cpu_dir, args.gpu_dir, args.manifest, args.output)),
            'absolute input and output paths are required')
    require(not os.path.lexists(args.output) and args.output.parent.is_dir(), 'fresh report path required')
    result = audit(args.cpu_dir, args.gpu_dir, args.manifest, args.source, args.bucket, args.result_sha256)
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'short report write')
        stream.flush()
        os.fsync(stream.fileno())
    require(args.output.read_bytes() == raw, 'report readback differs')
    print(json.dumps({'status': result['status'], 'sha256': sha(raw)}))


if __name__ == '__main__':
    main()
