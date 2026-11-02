"""CPU-only reconciliation of one saved ViT CUDA observation.

This checks saved identity, byte inventories, direct-process receipts and memory
arithmetic. It is not model execution, process-tree, global-device-idleness or
native-cleanup attestation, and sampled memory is not a worst-case bound.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

COMMIT = '7df264b4c7d76ab6f9b360cba9ecaa17867d92da'
REFERENCE = '3df19e35e8550f456015f3af3a9ce5536d0387fbf19cca7c58f6dd25fa975fa0'
CASE = '8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207'
STATE = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
CPU_TERMINAL = '2b61dfc59a8ab6c1072293e97d8e8ad42ceea8c4c1a0956bc9e7c992c2f3a9cc'
LABELS = ('cat-clean', 'dog-clean', 'cat-self', 'cat-gpu-donor', 'cat-cpu-donor', 'cat-repeat')
STAGES = ('before_model_load', 'before_model_copy', 'after_model_copy',
          *(f'after_forward_{i}' for i in range(1, 7)), 'after_disposal', 'after_cleanup')
MIB, GIB = 1024 ** 2, 1024 ** 3


def require(value, message):
    if not value:
        raise ValueError(message)


def pairs(values):
    out = {}
    for key, value in values:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            digest.update(block)
    return digest.hexdigest()


def timestamp(value):
    require(type(value) is str, 'missing timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'timestamp lacks timezone')
    return parsed


def audit(run, observation, repo, reference, assets, prior, preflight, post_exit, prior_failure, observer_source):
    run, observation, repo, reference, assets, prior, prior_failure = map(
        Path, (run, observation, repo, reference, assets, prior, prior_failure))
    started, owner, supervisor, candidate, final, outcome = [read_json(run / name) for name in
        ('started.json', 'owner.json', 'supervision.json', 'candidate.json', 'final.json', 'outcome.json')]
    attempt, controller_start, controller_exit, reconciliation, release = [read_json(observation / name) for name in
        ('attempt.json', 'start.json', 'exit.json', 'coordinator-reconciliation.json', 'slot-release.json')]
    pre = read_json(Path(preflight))
    require(file_sha(Path(preflight)) == 'ed442819637a6761ac113d987c3039cd53a8a4bbff4ee0ee838c1daa3ffa1ae8', 'preflight changed')
    require(pre['status'] == 'complete' and type(pre['model_forwards']) is int and pre['model_forwards'] == 0
            and pre['cuda_capable_dependencies_imported'] is False
            and pre['source_commit'] == COMMIT and pre['reference_sha256'] == REFERENCE, 'invalid preflight')
    require(all(x.get('source_commit') == COMMIT for x in
                (started, owner, candidate, final, outcome, attempt, controller_start, controller_exit, reconciliation, release)),
            'source identity differs')
    require(started['source_sha256'] == pre['source_sha256'] and started['asset_sha256'] == pre['asset_sha256']
            and started['reference_sha256'] == REFERENCE, 'preflight/start identity differs')
    source = (repo / 'looking_glass/vision_vit_cuda_observer.py').read_bytes()
    source_ast = ast.parse(source)
    assignments = [node for node in source_ast.body if isinstance(node, ast.Assign)
                   and any(isinstance(x, ast.Name) and x.id == '_SOURCE_PATHS' for x in node.targets)]
    require(len(assignments) == 1, 'source allowlist unavailable')
    expected_source = ast.literal_eval(assignments[0].value)
    require(type(expected_source) is tuple and len(expected_source) == 17
            and set(expected_source) == set(started['source_sha256']), 'source inventory differs')
    git = subprocess.run(['git', '--no-replace-objects', '-C', str(repo), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, timeout=30, check=True)
    require(git.stdout.strip() == COMMIT, 'checkout commit differs')
    for name in expected_source:
        local = (repo / name).read_bytes()
        committed = subprocess.run(['git', '--no-replace-objects', '-C', str(repo), 'show', COMMIT + ':' + name],
                                   capture_output=True, timeout=30, check=True).stdout
        require(local == committed and sha(local) == started['source_sha256'][name], 'committed source differs')
    case_raw = (repo / 'experiments/vision_coco_spatial_v1.json').read_bytes()
    require(sha(case_raw) == CASE, 'case manifest differs')
    manifest = json.loads(case_raw, object_pairs_hook=pairs)
    require(manifest['model_files'] == started['asset_sha256'], 'asset pins differ')
    for name, expected in manifest['model_files'].items():
        require(file_sha(assets / name) == expected, 'offline checkpoint asset differs')
    pair = next(x for x in manifest['pairs'] if x['pair_id'] == 'pair-04')
    require(pair['split'] == 'heldout' and pair['box_tokens'] == [161, 162, 175, 176], 'case selection differs')
    require(file_sha(reference / 'reference.json') == REFERENCE, 'reference manifest differs')
    ref = read_json(reference / 'reference.json')
    require(ref['status'] == 'complete' and type(ref['model_forwards']) is int and ref['model_forwards'] == 0
            and ref['pair_id'] == 'pair-04'
            and ref['case_sha256'] == CASE and ref['cpu_terminal_sha256'] == CPU_TERMINAL
            and ref['tokens'] == pair['box_tokens'], 'reference metadata differs')
    expected_files = {'cat-pixels.f32', 'dog-pixels.f32'} | {
        f'cpu-forward-{i:03d}.{ext}' for i in (1, 2, 24) for ext in ('json', 'bin')}
    require(set(ref['files']) == expected_files, 'reference inventory differs')
    for name, value in ref['files'].items():
        data = (reference / name).read_bytes()
        require(type(value['bytes']) is int and len(data) == value['bytes'] and sha(data) == value['sha256'],
                'reference file differs')
        if name.startswith('cpu-forward-'):
            require(data == (prior / name.removeprefix('cpu-')).read_bytes(), 'prior CPU evidence differs')
    require(file_sha(prior / 'terminal.json') == CPU_TERMINAL, 'prior CPU terminal differs')
    for side in ('cat', 'dog'):
        item = pair[side]
        require(ref['files'][side + '-pixels.f32']['image_id'] == item['image_id']
                and ref['files'][side + '-pixels.f32']['sha256'] == item['pixel_sha256'], 'pixel identity differs')
        jpeg = assets / 'images' / item['file_name']
        require(jpeg.stat().st_size == item['jpeg_bytes'] and file_sha(jpeg) == item['jpeg_sha256'], 'JPEG differs')
    dog = (reference / 'cpu-forward-002.bin').read_bytes()
    patch = (reference / 'cpu-forward-024.bin').read_bytes()
    require(b''.join(dog[4000 + i * 768:4000 + (i + 1) * 768] for i in pair['box_tokens']) == patch[312736:315808],
            'saved CPU donor differs')
    run_id = started['run_id']
    require([x.get('schema') for x in (started, owner, candidate, final, outcome)] ==
            ['vision-vit-cuda-start-v1', 'vision-vit-cuda-owner-v1', 'vision-vit-cuda-candidate-v1',
             'vision-vit-cuda-final-v1', 'vision-vit-cuda-result-v1'], 'receipt schema differs')
    require(type(run_id) is str and re.fullmatch(r'[0-9a-f]{32}', run_id) is not None, 'invalid run identity')
    require(all(x.get('run_id') == run_id for x in (owner, supervisor, candidate, final, outcome)), 'run IDs differ')
    require(type(owner['pid']) is int and owner['pid'] > 0
            and all(x.get('pid') == owner['pid'] for x in (supervisor, candidate, final))
            and all(x.get('supervisor_run_id') == owner['supervisor_run_id'] for x in (supervisor, candidate, final)),
            'owner and supervisor identities differ')
    require(supervisor['clean'] is True and type(supervisor['exit_code']) is int and supervisor['exit_code'] == 0
            and supervisor['timed_out'] is False and outcome['status'] == 'complete'
            and outcome['supervisor'] == {'clean': True, 'exit_code': 0, 'timed_out': False}
            and outcome['supervisor_error_type'] is None and outcome['telemetry_error_type'] is None,
            'supervision did not complete')
    require(type(controller_start['pid']) is int and controller_start['pid'] > 0
            and controller_exit['pid'] == controller_start['pid']
            and type(controller_exit['exit_code']) is int and controller_exit['exit_code'] == 0
            and controller_exit['reaped'] is True and controller_exit['wrapper_interrupted'] is False
            and controller_exit['bootstrap_help_rehearsal'] is False
            and controller_exit['first_forward_observed'] is True, 'controller exit differs')
    require(file_sha(Path(observer_source)) == '40315f631cf1951317162013a8cee7c5b8fb9e0a463364ceff837b9df9bfb867',
            'outer observer source differs')
    require(controller_exit['stdout_sha256'] == file_sha(observation / 'stdout.log')
            and controller_exit['stderr_sha256'] == file_sha(observation / 'stderr.log')
            and (observation / 'stderr.log').read_bytes() == b'', 'controller logs differ')
    require(timestamp(attempt['utc']) <= timestamp(controller_start['utc']) <= timestamp(started['started_utc'])
            <= timestamp(outcome['nvidia_smi_after']['timestamp_utc']) <= timestamp(controller_exit['utc'])
            <= timestamp(reconciliation['observed_utc']) <= timestamp(release['recorded_utc']), 'receipt order differs')
    candidate_sha = file_sha(run / 'candidate.json')
    require(final['candidate_sha256'] == outcome['candidate_sha256'] == reconciliation['candidate_sha256'] == candidate_sha
            and type(final['active_memory_leases']) is int and final['active_memory_leases'] == 0,
            'candidate/final binding differs')
    for field, path in [('controller_exit_observation_sha256', observation / 'exit.json'),
                        ('final_sha256', run / 'final.json'), ('outcome_sha256', run / 'outcome.json'),
                        ('owner_sha256', run / 'owner.json'), ('supervision_sha256', run / 'supervision.json')]:
        require(reconciliation[field] == file_sha(path), 'coordinator evidence differs')
    require(reconciliation['marker_and_attestation_absent_at_unlocked_point'] is True
            and reconciliation['prior_cuda_cleanup_verified'] is False
            and reconciliation['saved_supervisor_clean_exit'] is True
            and type(reconciliation['saved_active_memory_leases']) is int
            and reconciliation['saved_active_memory_leases'] == 0
            and reconciliation['reported_forward_count'] == 6, 'coordinator reconciliation differs')
    require(release['slot_released'] is True and release['cuda_native_cleanup_verified'] is False
            and release['next_owner_requires_fresh_admission'] is True
            and release['reconciliation_sha256'] == file_sha(observation / 'coordinator-reconciliation.json'),
            'slot release differs')
    post = read_json(Path(post_exit))
    require(file_sha(Path(post_exit)) == release['post_exit_inventory_sha256']
            == '99739bd3d75b9dde9f6530d944c60131cfcbe2886398bf34d43d6f8e8637b902'
            and post['source_commit'] == COMMIT and post['controller_exit_sha256'] == file_sha(observation / 'exit.json')
            and post['saved_owner_supervisor_identities_reconcile'] is True
            and post['saved_supervisor_clean_exit_zero'] is True and post['controller_exit_observation_reaped_zero'] is True
            and post['state_observation']['marker_absent'] is True
            and post['state_observation']['attestation_absent'] is True, 'post-exit point observation differs')
    for name, expected in post['private_run_file_sha256'].items():
        require(name in ('candidate.json', 'final.json', 'outcome.json', 'owner.json', 'supervision.json')
                and file_sha(run / name) == expected, 'post-exit inventory binding differs')
    require(set(post['private_run_file_sha256']) ==
            {'candidate.json', 'final.json', 'outcome.json', 'owner.json', 'supervision.json'},
            'post-exit inventory set differs')
    prior_exit = read_json(prior_failure / 'exit.json')
    prior_start = read_json(prior_failure / 'start.json')
    require(file_sha(prior_failure / 'exit.json') == '506d27835dc2ccb7a5d94e50b01cc97cab586c627e9358dbf5a739919af49b2d'
            and prior_exit['source_commit'] == COMMIT and prior_exit['pid'] == prior_start['pid']
            and type(prior_exit['exit_code']) is int and prior_exit['exit_code'] == 1
            and prior_exit['reaped'] is True and prior_exit['first_forward_observed'] is False
            and prior_exit['stderr_sha256'] == file_sha(prior_failure / 'stderr.log')
            and prior_exit['stdout_sha256'] == file_sha(prior_failure / 'stdout.log'),
            'prior failed wrapper evidence differs')
    m = candidate['measurement']
    require(type(m['forward_count']) is int and m['forward_count'] == outcome['forward_count'] == 6
            and m['reference_sha256'] == REFERENCE and m['model_state_sha256'] == STATE
            and m['same_gpu_self_repeat_and_landing'] is True and len(m['forwards']) == 6, 'measurement identity differs')
    rt = m['runtime']
    fixed = {'allocator': 'native', 'attention': 'eager', 'autocast_enabled': False,
             'cuda_build': '12.8', 'cudnn_benchmark': False, 'cudnn_deterministic': True,
             'cudnn_tf32': False, 'deterministic_algorithms': True, 'device': 'cuda:0',
             'device_name': 'NVIDIA GeForce RTX 2070 SUPER', 'dtype': 'float32',
             'float32_matmul_precision': 'highest', 'interop_threads': 1, 'matmul_tf32': False,
             'python': '3.11.15', 'seed': 17, 'threads': 1, 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
             'gpu_trace_claim_bytes': 16 * MIB, 'host_trace_claim_bytes': 16 * MIB}
    require(all(k in rt and type(rt[k]) is type(v) and rt[k] == v for k, v in fixed.items())
            and rt['compute_capability'] == [7, 5], 'runtime policy differs')
    for name, value in m['numeric_settings_after'].items():
        require(name in rt and type(rt[name]) is type(value) and rt[name] == value, 'numeric settings changed')
    total = rt['device_total_bytes']; cap = rt['allocator_cap_bytes']; commitment = rt['commitment_bytes']
    require(all(type(v) is int for v in (total, cap, commitment)) and 7 * GIB <= total <= 9 * GIB
            and cap == int(.25 * total) and commitment == cap + 256 * MIB, 'allocator policy differs')
    for inv in (outcome['nvidia_smi_before'], outcome['nvidia_smi_after']):
        require(inv['device_name'] == rt['device_name'] and type(inv['total_bytes']) is int
                and abs(inv['total_bytes'] - total) <= MIB and type(inv['free_bytes']) is int
                and type(inv['used_bytes']) is int and 0 <= inv['free_bytes'] <= inv['total_bytes']
                and 0 <= inv['used_bytes'] <= inv['total_bytes'], 'management inventory differs')
    require(outcome['nvidia_smi_before']['driver_version'] == outcome['nvidia_smi_after']['driver_version'],
            'management driver version changed')
    records = []
    for i, label in enumerate(LABELS, 1):
        meta = read_json(run / f'forward-{i:03d}.json')
        ret = read_json(run / f'return-{i:03d}.json')
        require(meta == m['forwards'][i - 1] and meta['label'] == ret['label'] == label
                and type(meta['submission']) is int and type(ret['submission']) is int
                and meta['submission'] == ret['submission'] == i,
                'forward ordering differs')
        require(meta['file'] == f'forward-{i:03d}.bin' and ret['file'] == f'return-{i:03d}.bin', 'file identity differs')
        data, result = (run / meta['file']).read_bytes(), (run / ret['file']).read_bytes()
        require(len(data) <= MIB and sha(data) == meta['sha256'] and len(result) == 4000
                and sha(result) == ret['sha256'] and result == data[:4000] and ret['shape'] == [1, 1000]
                and ret['dtype'] == meta['dtype'] == 'little-endian-float32', 'forward/return bytes differ')
        offset = 0
        for entry in sorted(meta['layout'].values(), key=lambda value: value['offset']):
            require(type(entry['offset']) is int and type(entry['bytes']) is int and entry['offset'] == offset
                    and entry['bytes'] > 0 and type(entry['shape']) is list
                    and all(type(x) is int and x > 0 for x in entry['shape'])
                    and math.prod(entry['shape']) * 4 == entry['bytes'], 'binary layout differs')
            offset += entry['bytes']
        require(offset == len(data), 'binary layout incomplete')
        md = meta['metadata']
        require(type(md['capture_bytes']) is int and type(md['source_bytes']) is int
                and 0 <= md['capture_bytes'] + md['source_bytes'] <= rt['gpu_trace_claim_bytes']
                and len(data) == 4000 + md['capture_bytes'] + md['source_bytes'], 'declared trace geometry differs')
        records.append({'label': label, 'saved_bytes': len(data), 'capture_bytes': md['capture_bytes'],
                        'source_bytes': md['source_bytes']})
    require(tuple(m['stages']) == STAGES, 'stage names differ')
    peaks, free_values = [], []
    for i, name in enumerate(STAGES, 1):
        stage = read_json(run / f'stage-{i:02d}.json')
        require(stage['stage'] == name and stage['peak_scope'] == 'since_pre_model_reset', 'stage differs')
        x = stage['memory']; keys = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
                                    'peak_allocated_bytes', 'peak_reserved_bytes')
        require(set(x) == set(keys) and all(type(x[k]) is int and x[k] >= 0 for k in keys), 'memory fields differ')
        free, current, reserved, pa, pr = (x['device_free_bytes'], x['allocated_bytes'], x['reserved_bytes'],
                                          x['peak_allocated_bytes'], x['peak_reserved_bytes'])
        reserve = max(GIB, (15 * total + 99) // 100)
        require(x['device_total_bytes'] == total and free + reserved <= total and current <= reserved
                and current <= pa <= pr and reserved <= pr <= cap - 256 * MIB
                and free >= reserve + commitment and reserved + reserve + commitment <= total,
                'memory/headroom arithmetic differs')
        if peaks:
            require(pa >= peaks[-1][0] and pr >= peaks[-1][1], 'cumulative peaks decreased')
        peaks.append((pa, pr)); free_values.append(free)
        if i == 1: require(x == m['initial_memory'], 'initial memory differs')
        if i == 11: require(x == m['cleanup_memory'] and current == reserved == 0, 'cleanup memory differs')
    require(m['cleanup_weakrefs'] == {'model': False, 'cat_pixels': False, 'dog_pixels': False}
            and all(value is False for value in m['cleanup_weakrefs'].values()), 'cleanup weakrefs differ')
    expected_run = {'started.json', 'owner.json', 'supervision.json', 'candidate.json', 'final.json', 'outcome.json'}
    expected_run |= {f'{stem}-{i:03d}.{ext}' for i in range(1, 7) for stem in ('forward', 'return') for ext in ('json', 'bin')}
    expected_run |= {f'stage-{i:02d}.json' for i in range(1, 12)}
    require({p.name for p in run.iterdir() if p.is_file()} == expected_run, 'run artifact inventory differs')
    require(all(not any(p.iterdir()) for p in run.iterdir() if p.is_dir()), 'run contains nonempty directory')
    return {'schema': 'independent-vit-cuda-provenance-v1', 'status': 'saved_evidence_consistent',
            'source_commit': COMMIT, 'source_file_count': len(expected_source), 'asset_count': len(manifest['model_files']),
            'reference_file_count': len(expected_files), 'saved_forwards': len(records), 'memory_stages': len(peaks),
            'max_saved_capture_and_source_bytes': max(x['capture_bytes'] + x['source_bytes'] for x in records),
            'gpu_and_host_trace_claim_bytes_each': 16 * MIB, 'allocator_cap_bytes': cap,
            'logical_gpu_commitment_bytes': commitment, 'observed_peak_allocated_bytes': max(x[0] for x in peaks),
            'observed_peak_reserved_bytes': max(x[1] for x in peaks), 'minimum_sampled_free_bytes': min(free_values),
            'cleanup_allocator_allocated_bytes': 0, 'cleanup_allocator_reserved_bytes': 0,
            'controller_and_owner_direct_exit_zero': True,
            'earlier_wrapper_failed_before_first_saved_forward': True,
            'limitations': ['source/receipts assume trusted quiescent files, not execution attestation',
                            'direct-child observation does not prove descendant drain or native CUDA cleanup',
                            'allocator peaks and free memory are sampled observations, not a device-wide bound',
                            'trace claims do not cap all host RAM or non-PyTorch device allocations',
                            'case pair was previously exposed; external data and pretraining provenance are not established']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'observation', 'repo', 'reference', 'assets', 'prior', 'preflight', 'post-exit',
                 'prior-failure', 'observer-source', 'output'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    result = audit(a.run, a.observation, a.repo, a.reference, a.assets, a.prior, a.preflight,
                   a.post_exit, a.prior_failure, a.observer_source)
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with Path(a.output).open('xb') as out:
        require(out.write(raw) == len(raw), 'short report write')
    print(json.dumps({'status': result['status'], 'report_sha256': sha(raw), 'saved_forwards': result['saved_forwards']}))


if __name__ == '__main__':
    main()
