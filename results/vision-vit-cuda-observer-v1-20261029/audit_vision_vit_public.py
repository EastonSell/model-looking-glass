"""Replay a path-free projection of the exposed-pair ViT CUDA evidence.

This checks public saved data only. Original process identity and execution
records are private; their hashes are commitments, not public attestation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

if not __debug__:
    raise RuntimeError('Run without Python optimization so evidence checks execute.')

CORE_SHA = '2845ad54ed0c8cb1a4c83f29c10b07957380df6562e509b2bd3586b72f35a662'
CLOSURE_SHA = '99b001f63ddb02f10b6e0c77f83474532468be07f73431851e60f71233d3ec1b'
COORDINATOR_SHA = '8dba3b14a9ca74df4a11ffb0d16e3a3931f394a2ca1ce5e66101f155a56dec80'
INDEPENDENT_SHA = 'ad704a1d514943afd7720177d51db65facf235ca36282229213e97e44c8b83a3'
COMMIT = '7df264b4c7d76ab6f9b360cba9ecaa17867d92da'
MIB, GIB = 1024**2, 1024**3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'run-dir', 'closure', 'coordinator', 'independent-report', 'reference', 'assets', 'git'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    core_path = Path(__file__).resolve().with_name('audit_vision_vit_saved_independent.py')
    assert digest(core_path.read_bytes()) == CORE_SHA
    core = runpy.run_path(str(core_path), run_name='_checked_vision_saved_core')
    closure_raw, coordinator_raw = args.closure.read_bytes(), args.coordinator.read_bytes()
    independent_raw = args.independent_report.read_bytes()
    assert digest(closure_raw) == CLOSURE_SHA and digest(coordinator_raw) == COORDINATOR_SHA
    assert digest(independent_raw) == INDEPENDENT_SHA
    closure, coordinator = json.loads(closure_raw), json.loads(coordinator_raw)
    independent = json.loads(independent_raw)
    assert independent['auditor_sha256'] == CORE_SHA and independent['source_commit'] == COMMIT
    assert independent['reference_sha256'] == core['REFERENCE']
    assert closure['schema'] == 'vision-vit-cuda-public-closure-v1'
    assert closure['source_commit'] == coordinator['source_commit'] == COMMIT
    assert closure['original_candidate_sha256'] == coordinator['candidate_sha256']
    assert closure['original_outcome_sha256'] == coordinator['outcome_sha256']
    assert coordinator['reported_forward_count'] == 6 and coordinator['saved_active_memory_leases'] == 0
    assert coordinator['saved_supervisor_clean_exit'] is True
    assert coordinator['marker_and_attestation_absent_at_unlocked_point'] is True
    assert coordinator['prior_cuda_cleanup_verified'] is False
    assert closure['reference_sha256'] == core['REFERENCE'] and len(closure['source_sha256']) == 17
    for name, expected in closure['source_sha256'].items():
        local = core['read'](args.repo / name, 4 * MIB)
        committed = subprocess.check_output([str(args.git), 'show', COMMIT + ':' + name], cwd=args.repo)
        assert local == committed and digest(local) == expected
    manifest = core['object_file'](args.repo / 'experiments/vision_coco_spatial_v1.json')
    hashes = manifest['model_files']
    assert hashes == closure['asset_sha256'] and set(hashes) == {'config.json', 'preprocessor_config.json', 'model.safetensors'}
    asset_data = {name: core['read'](args.assets / name, 32 * MIB) for name in hashes}
    assert all(digest(asset_data[name]) == expected for name, expected in hashes.items())
    assert core['checkpoint_state_digest'](asset_data['model.safetensors']) == core['STATE']
    reference = core['reference_files'](args.reference)
    for stem in ('forward', 'return'):
        for extension in ('json', 'bin'):
            assert {p.name for p in args.run_dir.glob(f'{stem}-*.{extension}')} == {
                f'{stem}-{i:03d}.{extension}' for i in range(1, 7)}
            for i in range(1, 7):
                name = f'{stem}-{i:03d}.{extension}'
                assert digest(core['read'](args.run_dir / name)) == independent['record_sha256'][name]
    assert not list(args.run_dir.glob('*.partial'))
    rows, tensors = zip(*(core['forward'](args.run_dir, i) for i in range(1, 7)), strict=True)
    measurement = closure['measurement']
    assert list(rows) == measurement['forwards'] and measurement['forward_count'] == 6
    assert measurement['reference_sha256'] == core['REFERENCE'] and measurement['model_state_sha256'] == core['STATE']
    assert measurement['same_gpu_self_repeat_and_landing'] is True
    cat, dog, self_obs, gpu_patch, cpu_patch, repeat = tensors
    assert all(item['before_all'] == cat['before_all'] for item in (self_obs, gpu_patch, cpu_patch, repeat))
    assert self_obs['after_all'] == cat['before_all'] and self_obs['logits'] == repeat['logits'] == cat['logits']
    assert gpu_patch['requested'] == b''.join(dog['before_all'][t * 768:(t + 1) * 768] for t in core['TOKENS'])
    assert cpu_patch['requested'] == reference['cpu-forward-024.bin'][312736:315808]
    diff, margin, close = core['diff'], core['margin'], core['close']
    comparisons = {
        'cpu_gpu_cat_logits': diff(reference['cpu-forward-001.bin'][:4000], cat['logits']),
        'cpu_gpu_dog_logits': diff(reference['cpu-forward-002.bin'][:4000], dog['logits']),
        'cpu_gpu_cat_early': diff(reference['cpu-forward-001.bin'][4000:155296], cat['before_all']),
        'cpu_gpu_dog_early': diff(reference['cpu-forward-002.bin'][4000:155296], dog['before_all']),
        'cpu_gpu_cpu_donor_logits': diff(reference['cpu-forward-024.bin'][:4000], cpu_patch['logits']),
        'gpu_native_vs_cpu_donor_logits': diff(gpu_patch['logits'], cpu_patch['logits']),
        'cpu_gpu_donor_tokens': diff(reference['cpu-forward-024.bin'][312736:315808], gpu_patch['requested'])}
    assert set(comparisons) == set(measurement['comparisons'])
    for name, calculated in comparisons.items():
        saved = measurement['comparisons'][name]
        assert calculated['count'] == saved['count']
        close(calculated['max_abs'], saved['max_abs'])
        close(calculated['mean_abs'], saved['mean_abs'])
    margins = {label: margin(value['logits']) for label, value in zip(core['LABELS'], tensors, strict=True)}
    margins.update(cpu_cat=margin(reference['cpu-forward-001.bin'][:4000]),
                   cpu_dog=margin(reference['cpu-forward-002.bin'][:4000]),
                   cpu_cat_cpu_donor=margin(reference['cpu-forward-024.bin'][:4000]))
    assert set(margins) == set(measurement['margins'])
    for label in margins:
        close(margins[label], measurement['margins'][label])
    runtime = measurement['runtime']
    total, cap, commitment = runtime['device_total_bytes'], runtime['allocator_cap_bytes'], runtime['commitment_bytes']
    assert runtime['device'] == 'cuda:0' and runtime['device_name'] == 'NVIDIA GeForce RTX 2070 SUPER'
    assert runtime['compute_capability'] == [7, 5] and runtime['allocator'] == 'native'
    assert cap == total // 4 and commitment == cap + 256 * MIB
    assert runtime['torch'] == '2.8.0+cu128' and runtime['transformers'] == '4.57.3'
    assert runtime['matmul_tf32'] is runtime['cudnn_tf32'] is runtime['cudnn_benchmark'] is False
    assert runtime['cudnn_deterministic'] is runtime['deterministic_algorithms'] is True
    assert runtime['autocast_enabled'] is False and runtime['threads'] == runtime['interop_threads'] == 1
    numeric = ('threads', 'interop_threads', 'deterministic_algorithms', 'matmul_tf32', 'cudnn_tf32',
               'cudnn_benchmark', 'cudnn_deterministic', 'autocast_enabled', 'float32_matmul_precision')
    assert measurement['numeric_settings_after'] == {key: runtime[key] for key in numeric}
    assert measurement['cleanup_weakrefs'] == {'model': False, 'cat_pixels': False, 'dog_pixels': False}
    names = ['before_model_load', 'before_model_copy', 'after_model_copy'] + [
        f'after_forward_{i}' for i in range(1, 7)] + ['after_disposal', 'after_cleanup']
    assert measurement['stages'] == names
    assert {p.name for p in args.run_dir.glob('stage-*.json')} == {f'stage-{i:02d}.json' for i in range(1, 12)}
    snapshots, reserve = [], max(GIB, (15 * total + 99) // 100)
    for i, name in enumerate(names, 1):
        assert digest(core['read'](args.run_dir / f'stage-{i:02d}.json', 16384)) == independent['record_sha256'][f'stage-{i:02d}.json']
        row = core['object_file'](args.run_dir / f'stage-{i:02d}.json', 16384)
        assert row['stage'] == name and row['peak_scope'] == 'since_pre_model_reset'
        memory = row['memory']
        fields = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
                  'peak_allocated_bytes', 'peak_reserved_bytes')
        values = [memory[key] for key in fields]
        assert all(type(value) is int and value >= 0 for value in values)
        free, observed_total, allocated, reserved, peak_allocated, peak_reserved = values
        assert observed_total == total and free + reserved <= total and allocated <= reserved
        assert allocated <= peak_allocated <= peak_reserved <= cap - 256 * MIB and reserved <= peak_reserved
        assert free >= reserve + commitment and reserved + reserve + commitment <= total
        snapshots.append(memory)
    assert snapshots[0] == measurement['initial_memory'] and snapshots[-1] == measurement['cleanup_memory']
    assert snapshots[-1]['allocated_bytes'] == snapshots[-1]['reserved_bytes'] == 0
    assert all(a['peak_allocated_bytes'] <= b['peak_allocated_bytes'] and a['peak_reserved_bytes'] <= b['peak_reserved_bytes']
               for a, b in zip(snapshots, snapshots[1:]))
    assert set(margins) == set(independent['margins'])
    for label, value in margins.items():
        close(value, independent['margins'][label])
    assert set(comparisons) == set(independent['comparisons'])
    for name, value in comparisons.items():
        saved = independent['comparisons'][name]
        assert value['count'] == saved['count']
        close(value['max_abs'], saved['max_abs'])
        close(value['mean_abs'], saved['mean_abs'])
    result = {'schema': 'vision-vit-cuda-public-saved-replay-v1', 'status': 'saved_data_consistent',
        'source_commit': COMMIT, 'public_closure_sha256': CLOSURE_SHA, 'coordinator_sha256': COORDINATOR_SHA,
        'independent_report_sha256': INDEPENDENT_SHA, 'replay_source_sha256': digest(Path(__file__).read_bytes()),
        'original_candidate_sha256_commitment': closure['original_candidate_sha256'],
        'original_outcome_sha256_commitment': closure['original_outcome_sha256'],
        'reference_sha256': core['REFERENCE'], 'checkpoint_state_sha256': core['STATE'],
        'forward_count': 6, 'same_gpu_self_repeat_and_landing': True,
        'margins': independent['margins'], 'comparisons': independent['comparisons'],
        'memory': {'sample_count': 11, 'max_allocator_peak_allocated_bytes': max(x['peak_allocated_bytes'] for x in snapshots),
                   'max_allocator_peak_reserved_bytes': max(x['peak_reserved_bytes'] for x in snapshots),
                   'minimum_sampled_device_free_bytes': min(x['device_free_bytes'] for x in snapshots),
                   'allocator_cleanup_allocated_bytes': snapshots[-1]['allocated_bytes'],
                   'allocator_cleanup_reserved_bytes': snapshots[-1]['reserved_bytes']},
        'limitations': ['one previously exposed pair; descriptive CPU/GPU comparison',
                        'published floating summaries are pinned values; local recomputation uses absolute 1e-12 and relative 1e-13 tolerances',
                        'private candidate and process identity hashes are commitments, not publicly verifiable execution evidence',
                        'allocator peaks are process-local; device-free values are point observations',
                        'zero allocator counters do not prove native CUDA cleanup or global idleness']}
    assert result['memory'] == independent['memory']
    sys.stdout.buffer.write(json.dumps(result, sort_keys=True, allow_nan=False).encode('utf-8') + b'\n')


if __name__ == '__main__':
    main()
