"""Retain bounded full-logit CPU references for fixed public short inputs.

Inputs must already be reserved. This is a correctness observation, not a
GPU profile, throughput benchmark, or independent execution attestation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


MANIFEST_SHA256 = '47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c'
INVENTORY_SHA256 = 'e586bae02145a62a7163eb030485fa85a33f9e94cd0949a9f88f3f74c7de194b'
MIN_DISK_FREE = 2 * 1024 ** 3
MAX_TENSOR_BYTES = 32 * 50257 * 4
SOURCE_PATHS = ('looking_glass/__init__.py', 'looking_glass/pretrained.py',
                'looking_glass/short_service_cpu_reference.py',
                'looking_glass/short_service_pilot_core.py',
                'looking_glass/broker.py', 'looking_glass/owner_completion.py',
                'looking_glass/resident_worker.py', 'looking_glass/budgeted_worker.py',
                'looking_glass/memory_budget.py', 'looking_glass/grouped_observation.py', 'looking_glass/fair_queue.py',
                'experiments/short_cpu_service_reference_inputs_v1.json',
                'tests/test_short_service_cpu_reference.py', 'docs/SHORT_GPU_SERVICE_PILOT_V1.md',
                'pyproject.toml')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_new(path, data):
    with Path(path).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short evidence write')
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write_new(path, (json.dumps(value, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False) + '\n').encode('utf-8'))


def _read_inputs(manifest_path, inventory_path):
    def read_bounded(path, limit):
        with Path(path).open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('input file exceeds its bound')
        return raw
    manifest_raw = read_bounded(manifest_path, 64 * 1024)
    inventory_raw = read_bounded(inventory_path, 2 * 1024 ** 2)
    if _sha(manifest_raw) != MANIFEST_SHA256 or _sha(inventory_raw) != INVENTORY_SHA256:
        raise ValueError('fixed manifest or reservation inventory differs')
    manifest, inventory = json.loads(manifest_raw), json.loads(inventory_raw)
    cases = manifest['comparisons']
    if [item['id'] for item in cases] != ['synthetic-16', 'synthetic-32', 'gas', 'ligand']:
        raise ValueError('fixed comparison order differs')
    records = inventory['records']
    identifiers = [record['record_id'] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('reservation identities repeat')
    index = {record['record_id']: record for record in records}
    mapping = {'synthetic-16': ('planned-service-synthetic-mask-16',) * 2,
               'synthetic-32': ('planned-service-synthetic-mask-32',) * 2,
               'gas': ('planned-short-science-gas-active', 'planned-short-science-gas-padded'),
               'ligand': ('planned-short-science-ligand-active', 'planned-short-science-ligand-padded')}
    for case in cases:
        for role, record_id in zip(('reference', 'service'), mapping[case['id']], strict=True):
            value, record = case[role], index.get(record_id)
            if (type(record) is not dict or record.get('category') != 'planned'
                    or record.get('token_ids') != value['input_ids']
                    or record.get('attention_mask') != value['attention_mask']
                    or type(record.get('token_ids')) is not list or type(record.get('attention_mask')) is not list
                    or any(type(item) is not int for item in record['token_ids'] + record['attention_mask'])):
                raise ValueError('an exact fixed input/mask reservation is missing')
    return manifest, inventory


def _source_identity(repo, git_exe, expected_commit):
    if re.fullmatch(r'[0-9a-f]{40}', expected_commit) is None:
        raise ValueError('an exact source commit is required')
    git = Path(git_exe)
    if not git.is_absolute() or not git.resolve(strict=True).is_file():
        raise ValueError('an absolute Git executable is required')
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0'})
    def git_call(*args):
        return subprocess.run([str(git), '--no-replace-objects', '-C', str(repo), *args],
                              env=env, capture_output=True, check=True, timeout=15).stdout
    actual = git_call('rev-parse', '--verify', 'HEAD').decode('ascii').strip()
    if actual != expected_commit or git_call('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('a clean exact source checkout is required')
    hashes = {}
    for name in SOURCE_PATHS:
        raw = (repo / name).read_bytes()
        if len(raw) > 4 * 1024 ** 2 or raw != git_call('show', actual + ':' + name):
            raise RuntimeError('source bytes differ from the committed checkout')
        hashes[name] = _sha(raw)
    return hashes


def _sequence(bundle, cases, output):
    import torch
    from .short_service_pilot_core import compare_f32

    model = bundle.model
    if (sys.byteorder != 'little' or model.training is not False or model.config.use_cache is not False
            or model.config.vocab_size != 50257 or model.config.n_positions < 32
            or getattr(model.config, '_attn_implementation', None) != 'eager'):
        raise RuntimeError('model or byte order differs from the fixed CPU contract')
    if any(value.device.type != 'cpu' or value.dtype != torch.float32 for value in model.parameters()):
        raise RuntimeError('model parameters are not CPU float32')
    torch.random.default_generator.manual_seed(29)
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError('CPU thread setting differs')
    entries, retained = [], {}
    for case_id in ('synthetic-16', 'gas', 'synthetic-32', 'ligand'):
        case = next(item for item in cases if item['id'] == case_id)
        roles = ('reference',) if case['reference'] == case['service'] else ('reference', 'service')
        for role in roles:
            item = case[role]
            ids = torch.tensor([item['input_ids']], dtype=torch.int64)
            mask = torch.tensor([item['attention_mask']], dtype=torch.int64)
            name = f'{case_id}-{role}'
            _write_json(output / (name + '-start.json'), {'case': case_id, 'role': role,
                        'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']})
            with torch.inference_mode():
                returned = model(input_ids=ids, attention_mask=mask, use_cache=False)
                logits = returned.logits
            if (type(logits) is not torch.Tensor or logits.device.type != 'cpu' or logits.dtype != torch.float32
                    or tuple(logits.shape) != (1, ids.shape[1], 50257)):
                raise RuntimeError('returned logits do not match the expected full shape')
            with torch.inference_mode(False), torch.no_grad():
                saved = logits.detach().contiguous().clone()
            raw = saved.numpy().tobytes(order='C')
            if len(raw) != ids.shape[1] * 50257 * 4 or len(raw) > MAX_TENSOR_BYTES:
                raise RuntimeError('full-logit bytes exceed the fixed bound')
            _write_new(output / (name + '.f32'), raw)
            if not bool(torch.isfinite(saved).all()):
                raise RuntimeError('saved logits are not finite')
            entry = {'case': case_id, 'role': role, 'shape': list(saved.shape),
                     'bytes': len(raw), 'sha256': _sha(raw), 'filename': name + '.f32'}
            _write_json(output / (name + '.json'), entry)
            entries.append(entry)
            retained[(case_id, role)] = saved
    comparisons = []
    for case_id in ('gas', 'ligand'):
        reference = retained[(case_id, 'reference')]
        padded = retained[(case_id, 'service')][:, :reference.shape[1], :].contiguous()
        value = compare_f32(padded, reference)
        comparisons.append({'case': case_id, 'comparison': value})
    return entries, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--git', required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise RuntimeError('CPU-only execution requires CUDA_VISIBLE_DEVICES empty or -1')
    repo = Path(__file__).resolve(strict=True).parents[1]
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new absolute directory')
    hashes = _source_identity(repo, args.git, args.expected_commit)
    manifest, _ = _read_inputs(repo / 'experiments/short_cpu_service_reference_inputs_v1.json', args.inventory)
    cache = Path(args.cache_dir).resolve(strict=True)
    if not cache.is_dir() or any(shutil.disk_usage(path).free < MIN_DISK_FREE
                                 for path in (output.parent, cache)):
        raise RuntimeError('available disk is below the fixed reserve')
    output.mkdir()
    _write_json(output / 'started.json', {'schema': 'short-service-cpu-reference-start-v1',
                'source_commit': args.expected_commit, 'source_sha256': hashes,
                'manifest_sha256': MANIFEST_SHA256, 'inventory_sha256': INVENTORY_SHA256})
    try:
        from .pretrained import load_distilgpt2
        bundle = load_distilgpt2(cache_dir=cache, allow_download=False)
        entries, comparisons = _sequence(bundle, manifest['comparisons'], output)
        status = 'passed' if all(item['comparison']['violations'] == 0 for item in comparisons) else 'failed'
        _write_json(output / 'result.json', {'schema': 'short-service-cpu-reference-v1', 'status': status,
                    'model_metadata': bundle.metadata, 'entries': entries, 'padding_comparisons': comparisons,
                    'source_commit': args.expected_commit, 'inventory_sha256': INVENTORY_SHA256})
        return 0 if status == 'passed' else 1
    except BaseException as exc:
        try:
            _write_json(output / 'failure.json', {'schema': 'short-service-cpu-reference-failure-v1',
                        'status': 'incomplete', 'error_type': type(exc).__name__})
        except Exception:
            pass
        raise


if __name__ == '__main__':
    raise SystemExit(main())
