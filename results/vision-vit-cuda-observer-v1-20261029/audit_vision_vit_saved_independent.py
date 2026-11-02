"""Read-only, stdlib audit of one closed exposed-pair ViT CUDA observation.

The script never imports a model library or queries a GPU. It verifies saved
bits and source identities; it does not attest execution or native teardown.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

if not __debug__:
    raise RuntimeError('Run without Python optimization so evidence checks execute.')

LABELS = ('cat-clean', 'dog-clean', 'cat-self', 'cat-gpu-donor', 'cat-cpu-donor', 'cat-repeat')
TOKENS = (161, 162, 175, 176)
SITE = 'vit.encoder.layer.2'
REFERENCE = '3df19e35e8550f456015f3af3a9ce5536d0387fbf19cca7c58f6dd25fa975fa0'
STATE = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
MIB, GIB = 1024**2, 1024**3


def read(path, maximum=MIB):
    data = path.read_bytes()
    assert 0 < len(data) <= maximum, path.name
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_file(path, maximum=MIB):
    data = json.loads(read(path, maximum))
    assert type(data) is dict
    return data


def decode(data):
    assert len(data) % 4 == 0
    result = []
    for (bits,) in struct.iter_unpack('<I', data):
        exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
        assert exponent != 255, 'nonfinite float32 encoding'
        mantissa = fraction if exponent == 0 else (1 << 23) | fraction
        value = math.ldexp(float(mantissa), -149 if exponent == 0 else exponent - 150)
        result.append(-value if bits >> 31 else value)
    return result


def margin(data):
    values = decode(data)
    assert len(values) == 1000
    def log_sum(group):
        maximum = max(group)
        return maximum, math.log(math.fsum(math.exp(value - maximum) for value in group))
    dog, dog_sum = log_sum(values[151:269])
    cat, cat_sum = log_sum(values[281:286])
    return (dog - cat) + dog_sum - cat_sum


def diff(a, b):
    left, right = decode(a), decode(b)
    assert len(left) == len(right) and len(left) > 0
    changes = [abs(x - y) for x, y in zip(left, right, strict=True)]
    return {'max_abs': max(changes), 'mean_abs': math.fsum(changes) / len(changes), 'count': len(changes)}


def close(x, y):
    assert type(x) in (int, float) and type(y) in (int, float)
    assert math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=1e-13, abs_tol=1e-12)


def reference_files(directory):
    ref_raw = read(directory / 'reference.json')
    assert sha(ref_raw) == REFERENCE
    ref = json.loads(ref_raw)
    expected = {'cat-pixels.f32', 'dog-pixels.f32'} | {
        f'cpu-forward-{i:03d}.{ext}' for i in (1, 2, 24) for ext in ('bin', 'json')}
    assert ref['status'] == 'complete' and ref['model_forwards'] == 0
    assert ref['tokens'] == list(TOKENS) and set(ref['files']) == expected
    files = {}
    for name in expected:
        data = read(directory / name)
        assert len(data) == ref['files'][name]['bytes'] and sha(data) == ref['files'][name]['sha256']
        files[name] = data
    selected = b''.join(files['cpu-forward-002.bin'][4000 + i * 768:4000 + (i + 1) * 768] for i in TOKENS)
    assert selected == files['cpu-forward-024.bin'][312736:315808] and len(selected) == 3072
    return files


def checkpoint_state_digest(raw):
    assert len(raw) >= 8
    header_length = struct.unpack('<Q', raw[:8])[0]
    assert 0 < header_length <= len(raw) - 8
    header = json.loads(raw[8:8 + header_length])
    data = memoryview(raw)[8 + header_length:]
    tensors = {name: item for name, item in header.items() if name != '__metadata__'}
    assert len(tensors) == 200
    digest, spans = hashlib.sha256(), []
    for name, item in sorted(tensors.items()):
        assert item['dtype'] == 'F32' and type(item['shape']) is list
        assert all(type(value) is int and value >= 0 for value in item['shape'])
        start, end = item['data_offsets']
        assert type(start) is int and type(end) is int and 0 <= start <= end <= len(data)
        assert end - start == 4 * math.prod(item['shape'])
        spans.append((start, end))
        digest.update(name.encode() + b'\0torch.float32\0')
        digest.update(json.dumps(item['shape'], separators=(',', ':')).encode() + b'\0')
        digest.update(data[start:end])
    spans.sort()
    assert spans[0][0] == 0 and spans[-1][1] == len(data)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    return digest.hexdigest()


def forward(directory, index, device='cuda:0'):
    label = LABELS[index - 1]
    row = object_file(directory / f'forward-{index:03d}.json')
    assert type(row['submission']) is int and row['submission'] == index and row['label'] == label
    assert row['dtype'] == 'little-endian-float32' and row['file'] == f'forward-{index:03d}.bin'
    raw = read(directory / row['file'])
    assert sha(raw) == row['sha256']
    shapes = {'logits': [1, 1000], 'before_all': [1, 197, 192]}
    if index in (3, 4, 5):
        shapes.update({'after_all': [1, 197, 192]})
        for i in range(4):
            shapes.update({f'before_{i}': [192], f'after_{i}': [192]})
        if index in (4, 5):
            shapes['requested'] = [4, 192]
    assert set(row['layout']) == set(shapes)
    spans, tensors = [], {}
    for name, shape in shapes.items():
        item = row['layout'][name]
        size, offset = 4 * math.prod(shape), item['offset']
        assert type(offset) is int and offset >= 0 and type(item['bytes']) is int and item['bytes'] == size
        assert item['shape'] == shape and all(type(value) is int for value in item['shape'])
        assert offset + size <= len(raw)
        spans.append((offset, offset + size))
        tensors[name] = raw[offset:offset + size]
    spans.sort()
    assert spans[0][0] == 0 and spans[-1][1] == len(raw)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    returned = object_file(directory / f'return-{index:03d}.json', 16384)
    returned_raw = read(directory / f'return-{index:03d}.bin', 16384)
    assert returned['label'] == label and returned['submission'] == index
    assert returned['file'] == f'return-{index:03d}.bin' and returned['shape'] == [1, 1000]
    assert returned['dtype'] == 'little-endian-float32' and returned['sha256'] == sha(returned_raw)
    assert returned_raw == tensors['logits']
    close(row['dog_minus_cat_log_odds'], margin(tensors['logits']))
    planned = [('before_all', 'capture', None)]
    if index in (3, 4, 5):
        for i, token in enumerate(TOKENS):
            planned.extend(((f'before_{i}', 'capture', token), (f'action_{i}', 'patch', token),
                            (f'after_{i}', 'capture', token)))
        planned.append(('after_all', 'capture', None))
    events = row['metadata']['steps']
    assert len(events) == len(planned)
    for order, ((name, operation, token), event) in enumerate(zip(planned, events, strict=True)):
        assert event['name'] == name and event['operation'] == operation and event['order'] == order
        assert event['module_path'] == SITE and event['forward_index'] == 0 and event['invocation'] == 0
        assert event['output_shape'] == [1, 197, 192] and event['dtype'] == 'torch.float32'
        assert event['device'] == device and event.get('token_index') == token
        assert event.get('batch_index') == (None if token is None else 0)
        assert event.get('source_capture') == (name.replace('action_', 'before_')
                                               if index == 3 and operation == 'patch' else None)
    if index in (3, 4, 5):
        before = tensors['before_all']
        requested = (b''.join(tensors[f'before_{i}'] for i in range(4)) if index == 3 else tensors['requested'])
        expected = bytearray(before)
        for i, token in enumerate(TOKENS):
            start = token * 768
            assert tensors[f'before_{i}'] == before[start:start + 768]
            expected[start:start + 768] = requested[i * 768:(i + 1) * 768]
            assert tensors[f'after_{i}'] == expected[start:start + 768]
        assert tensors['after_all'] == bytes(expected)
    return row, tensors


def audit(repo, directory, controller, reference, assets, git, expected_commit):
    started, owner = (object_file(directory / name) for name in ('started.json', 'owner.json'))
    candidate_raw = read(directory / 'candidate.json')
    candidate, final = json.loads(candidate_raw), object_file(directory / 'final.json')
    supervision, outcome = (object_file(directory / name) for name in ('supervision.json', 'outcome.json'))
    assert outcome['status'] == 'complete' and outcome['forward_count'] == 6
    assert not (directory / 'failure.json').exists()
    assert not list(directory.glob('*.partial'))
    for stem in ('forward', 'return'):
        for extension in ('json', 'bin'):
            assert {p.name for p in directory.glob(f'{stem}-*.{extension}')} == {
                f'{stem}-{index:03d}.{extension}' for index in range(1, 7)}
    assert all(record['source_commit'] == expected_commit for record in (started, owner, candidate, final, outcome))
    assert all(record['run_id'] == started['run_id'] for record in (owner, candidate, final, supervision, outcome))
    assert all(record['supervisor_run_id'] == supervision['supervisor_run_id'] for record in (owner, candidate, final))
    assert all(type(record['pid']) is int and record['pid'] == supervision['pid'] for record in (owner, candidate, final))
    assert supervision['clean'] is True and supervision['timed_out'] is False and supervision['exit_code'] == 0
    assert outcome['supervisor'] == {'clean': True, 'timed_out': False, 'exit_code': 0}
    assert final['candidate_sha256'] == outcome['candidate_sha256'] == sha(candidate_raw)
    assert final['active_memory_leases'] == 0 and outcome['telemetry_error_type'] is None
    assert outcome['owner_presence_after_exit']['query'] == 'ok'
    assert outcome['owner_presence_after_exit']['owner_pid_listed'] is False
    controller_start = object_file(controller / 'start.json', 16384)
    controller_exit_raw = read(controller / 'exit.json', 16384)
    controller_exit = json.loads(controller_exit_raw)
    reconciliation_raw = read(controller / 'coordinator-reconciliation.json', 16384)
    reconciliation = json.loads(reconciliation_raw)
    assert controller_start['source_commit'] == controller_exit['source_commit'] == expected_commit
    assert type(controller_start['pid']) is int and controller_start['pid'] == controller_exit['pid']
    assert controller_exit['exit_code'] == 0 and controller_exit['reaped'] is True
    assert controller_exit['wrapper_interrupted'] is False and controller_exit['first_forward_observed'] is True
    assert controller_start['bootstrap_help_rehearsal'] is controller_exit['bootstrap_help_rehearsal'] is False
    assert controller_exit['stdout_sha256'] == sha((controller / 'stdout.log').read_bytes())
    assert controller_exit['stderr_sha256'] == sha((controller / 'stderr.log').read_bytes())
    assert reconciliation['source_commit'] == expected_commit
    assert reconciliation['controller_exit_observation_sha256'] == sha(controller_exit_raw)
    assert reconciliation['candidate_sha256'] == sha(candidate_raw)
    for key, filename in (('outcome_sha256', 'outcome.json'), ('owner_sha256', 'owner.json'),
                          ('final_sha256', 'final.json'), ('supervision_sha256', 'supervision.json')):
        assert reconciliation[key] == sha(read(directory / filename))
    assert reconciliation['reported_forward_count'] == 6 and reconciliation['saved_active_memory_leases'] == 0
    assert reconciliation['saved_supervisor_clean_exit'] is True
    assert reconciliation['marker_and_attestation_absent_at_unlocked_point'] is True
    assert reconciliation['prior_cuda_cleanup_verified'] is False
    source = started['source_sha256']
    assert len(source) == 17
    for name, expected in source.items():
        local = read(repo / name, 4 * MIB)
        committed = subprocess.check_output([str(git), 'show', expected_commit + ':' + name], cwd=repo)
        assert local == committed and sha(local) == expected
    manifest = object_file(repo / 'experiments/vision_coco_spatial_v1.json')
    asset_hashes = manifest['model_files']
    assert set(asset_hashes) == {'config.json', 'preprocessor_config.json', 'model.safetensors'}
    assert started['asset_sha256'] == asset_hashes and started['reference_sha256'] == REFERENCE
    asset_data = {name: read(assets / name, 32 * MIB) for name in asset_hashes}
    for name, expected in asset_hashes.items():
        assert sha(asset_data[name]) == expected
    assert checkpoint_state_digest(asset_data['model.safetensors']) == STATE
    ref = reference_files(reference)
    assert sha(ref['cat-pixels.f32']) == 'e531519426eff5db71bac0fb3247417a894949db5dc785fc0a88b9f11081a8ed'
    assert sha(ref['dog-pixels.f32']) == 'e63ad3e6f0a3a9b39f44e9828bc8be94e9eb3905bf5b4da11e19c91567c5c09a'
    measurement = candidate['measurement']
    assert measurement['reference_sha256'] == REFERENCE and measurement['model_state_sha256'] == STATE
    rows, tensors = zip(*(forward(directory, i) for i in range(1, 7)), strict=True)
    assert list(rows) == measurement['forwards'] and measurement['forward_count'] == 6
    cat, dog, self_obs, gpu_patch, cpu_patch, repeat = tensors
    assert all(item['before_all'] == cat['before_all'] for item in (self_obs, gpu_patch, cpu_patch, repeat))
    assert self_obs['after_all'] == cat['before_all']
    assert self_obs['logits'] == repeat['logits'] == cat['logits']
    assert gpu_patch['requested'] == b''.join(dog['before_all'][t * 768:(t + 1) * 768] for t in TOKENS)
    assert cpu_patch['requested'] == ref['cpu-forward-024.bin'][312736:315808]
    comparisons = {
        'cpu_gpu_cat_logits': diff(ref['cpu-forward-001.bin'][:4000], cat['logits']),
        'cpu_gpu_dog_logits': diff(ref['cpu-forward-002.bin'][:4000], dog['logits']),
        'cpu_gpu_cat_early': diff(ref['cpu-forward-001.bin'][4000:155296], cat['before_all']),
        'cpu_gpu_dog_early': diff(ref['cpu-forward-002.bin'][4000:155296], dog['before_all']),
        'cpu_gpu_cpu_donor_logits': diff(ref['cpu-forward-024.bin'][:4000], cpu_patch['logits']),
        'gpu_native_vs_cpu_donor_logits': diff(gpu_patch['logits'], cpu_patch['logits']),
        'cpu_gpu_donor_tokens': diff(ref['cpu-forward-024.bin'][312736:315808], gpu_patch['requested'])}
    assert set(comparisons) == set(measurement['comparisons'])
    for name, calculated in comparisons.items():
        saved = measurement['comparisons'][name]
        assert calculated['count'] == saved['count']
        close(calculated['max_abs'], saved['max_abs'])
        close(calculated['mean_abs'], saved['mean_abs'])
    margins = {label: margin(value['logits']) for label, value in zip(LABELS, tensors, strict=True)}
    margins.update(cpu_cat=margin(ref['cpu-forward-001.bin'][:4000]),
                   cpu_dog=margin(ref['cpu-forward-002.bin'][:4000]),
                   cpu_cat_cpu_donor=margin(ref['cpu-forward-024.bin'][:4000]))
    assert set(margins) == set(measurement['margins'])
    for label in margins:
        close(margins[label], measurement['margins'][label])
    runtime = measurement['runtime']
    assert runtime['device'] == 'cuda:0' and runtime['device_name'] == 'NVIDIA GeForce RTX 2070 SUPER'
    assert runtime['compute_capability'] == [7, 5] and runtime['allocator'] == 'native'
    assert runtime['dtype'] == 'float32' and runtime['attention'] == 'eager' and runtime['seed'] == 17
    total, cap, commitment = runtime['device_total_bytes'], runtime['allocator_cap_bytes'], runtime['commitment_bytes']
    assert cap == total // 4 and commitment == cap + 256 * MIB
    assert runtime['torch'] == '2.8.0+cu128' and runtime['transformers'] == '4.57.3'
    assert runtime['matmul_tf32'] is runtime['cudnn_tf32'] is runtime['cudnn_benchmark'] is False
    assert runtime['cudnn_deterministic'] is runtime['deterministic_algorithms'] is True
    assert runtime['autocast_enabled'] is False and runtime['threads'] == runtime['interop_threads'] == 1
    assert runtime['float32_matmul_precision'] == 'highest'
    numeric_keys = ('threads', 'interop_threads', 'deterministic_algorithms', 'matmul_tf32', 'cudnn_tf32',
                    'cudnn_benchmark', 'cudnn_deterministic', 'autocast_enabled', 'float32_matmul_precision')
    assert measurement['numeric_settings_after'] == {key: runtime[key] for key in numeric_keys}
    assert runtime['gpu_trace_claim_bytes'] == runtime['host_trace_claim_bytes'] == 16 * MIB
    assert measurement['cleanup_weakrefs'] == {'model': False, 'cat_pixels': False, 'dog_pixels': False}
    stages = ['before_model_load', 'before_model_copy', 'after_model_copy'] + [
        f'after_forward_{i}' for i in range(1, 7)] + ['after_disposal', 'after_cleanup']
    assert measurement['stages'] == stages
    snapshots = []
    reserve = max(GIB, (15 * total + 99) // 100)
    for i, name in enumerate(stages, 1):
        row = object_file(directory / f'stage-{i:02d}.json', 16384)
        assert row['stage'] == name and row['peak_scope'] == 'since_pre_model_reset'
        memory = row['memory']
        vals = [memory[k] for k in ('device_free_bytes', 'device_total_bytes', 'allocated_bytes',
                                    'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes')]
        assert all(type(value) is int and value >= 0 for value in vals)
        free, observed_total, allocated, reserved, peak_allocated, peak_reserved = vals
        assert observed_total == total and free + reserved <= total and allocated <= reserved
        assert allocated <= peak_allocated <= peak_reserved <= cap - 256 * MIB
        assert reserved <= peak_reserved and free >= reserve + commitment and reserved + reserve + commitment <= total
        snapshots.append(memory)
    assert snapshots[0] == measurement['initial_memory'] and snapshots[-1] == measurement['cleanup_memory']
    assert snapshots[-1]['allocated_bytes'] == snapshots[-1]['reserved_bytes'] == 0
    assert all(a['peak_allocated_bytes'] <= b['peak_allocated_bytes'] and a['peak_reserved_bytes'] <= b['peak_reserved_bytes']
               for a, b in zip(snapshots, snapshots[1:]))
    record_hashes = {name: sha(read(directory / name)) for name in
        ['started.json', 'owner.json', 'candidate.json', 'final.json', 'supervision.json', 'outcome.json']}
    record_hashes.update({f'{stem}-{index:03d}.{extension}': sha(read(directory / f'{stem}-{index:03d}.{extension}'))
                          for stem in ('forward', 'return') for index in range(1, 7)
                          for extension in ('bin', 'json')})
    record_hashes.update({f'stage-{index:02d}.json': sha(read(directory / f'stage-{index:02d}.json'))
                          for index in range(1, 12)})
    return {'schema': 'vision-vit-cuda-independent-saved-audit-v1',
            'status': 'saved_observation_consistent', 'source_commit': expected_commit,
            'auditor_sha256': sha(Path(__file__).read_bytes()), 'reference_sha256': REFERENCE,
            'checkpoint_state_sha256': STATE, 'asset_sha256': asset_hashes, 'record_sha256': record_hashes,
            'controller_exit_sha256': sha(controller_exit_raw), 'coordinator_reconciliation_sha256': sha(reconciliation_raw),
            'forward_count': len(rows), 'same_gpu_self_repeat_and_landing': True,
            'margins': margins, 'comparisons': comparisons,
            'cat_gpu_donor_minus_clean_margin': margins['cat-gpu-donor'] - margins['cat-clean'],
            'cat_cpu_donor_minus_clean_margin': margins['cat-cpu-donor'] - margins['cat-clean'],
            'memory': {'sample_count': len(snapshots),
                       'max_allocator_peak_allocated_bytes': max(s['peak_allocated_bytes'] for s in snapshots),
                       'max_allocator_peak_reserved_bytes': max(s['peak_reserved_bytes'] for s in snapshots),
                       'minimum_sampled_device_free_bytes': min(s['device_free_bytes'] for s in snapshots),
                       'allocator_cleanup_allocated_bytes': snapshots[-1]['allocated_bytes'],
                       'allocator_cleanup_reserved_bytes': snapshots[-1]['reserved_bytes']},
            'external_device_points': {'before': outcome['nvidia_smi_before'], 'after': outcome['nvidia_smi_after']},
            'limitations': ['one previously exposed pair; descriptive CPU/GPU comparison',
                            'source and saved records do not independently attest execution',
                            'allocator peaks are process-local; device-free values are point observations',
                            'zero allocator counters do not prove native CUDA cleanup or global idleness']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--git', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.repo, args.run_dir, args.controller, args.reference, args.assets, args.git, args.commit),
                     sort_keys=True, allow_nan=False))
