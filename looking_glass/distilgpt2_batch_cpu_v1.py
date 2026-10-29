"""Pre-frozen sampled DistilGPT-2 CPU batch correctness experiment.

Run in a fresh CPU-only process. This is not a performance benchmark or an
attestation mechanism. See docs/DISTILGPT2_BATCH_CPU_V1.md before use.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import batch_validation as validation
from .microbatch import BatchProfile, ForwardContract
from .pretrained import ASSETS, MODEL_ID, REVISION, load_distilgpt2
from .resident_worker import ForwardRequest, ResidentWorker


VERSION = 'distilgpt2-batch-cpu-v1'
SEED = 2026102808
OUTPUT_CAP = 8 * 1024 * 1024
WIDTH = 50257
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    'docs/DISTILGPT2_BATCH_CPU_V1.md',
    'tools/run_distilgpt2_batch_cpu_v1.py',
    'looking_glass/__init__.py',
    'looking_glass/distilgpt2_batch_cpu_v1.py',
    'looking_glass/batch_validation.py',
    'looking_glass/microbatch.py',
    'looking_glass/pretrained.py',
    'looking_glass/resident_worker.py',
    'tests/test_distilgpt2_batch_cpu_v1.py',
)
DISCOVERY = ([50256, 101, 201, 301], [50256, 102, 202, 302])
CASES = (([50256, 103, 203, 303], [50256, 104, 204, 304]),
         ([50256, 105, 205, 305], [50256, 106, 206, 306]))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inputs(row):
    return {'input_ids': torch.tensor([row], dtype=torch.int64, device='cpu'),
            'attention_mask': torch.ones((1, 4), dtype=torch.int64, device='cpu')}


def groups():
    return [{'group_id': f'group-{g + 1}', 'cases': [
        {'case_id': f'case-{g + 1}-{c + 1}', 'owner': ('team-a', 'team-b')[c],
         'inputs': inputs(row)} for c, row in enumerate(rows)]}
        for g, rows in enumerate(CASES)]


def _profile(runtime, sources, *, control=None):
    model = 'batch-mean-negative-control' if control == 'batch_mean' else MODEL_ID
    width = 1 if control == 'batch_mean' else WIDTH
    role = control or 'primary'
    adapter_sha = digest({'adapter': VERSION, 'role': role, 'source_sha256': sources})
    model_sha = digest({'model_id': model, 'revision': REVISION if control != 'batch_mean' else 'toy-v1',
                        'assets': ASSETS if control != 'batch_mean' else {}})
    checkpoint_sha = ASSETS['model.safetensors'][1] if control != 'batch_mean' else digest('toy-v1')
    contract = ForwardContract(
        model_id=model, checkpoint=REVISION if control != 'batch_mean' else 'toy-v1',
        adapter=f'{VERSION}-{role}', device='cpu', model_dtype=torch.float32,
        runtime_id='sha256:' + digest(runtime), seed=SEED, deterministic=True, batch_separable=True,
        mode='eval', rng_policy='none', cache_policy='disabled', mutable_state_policy='none',
        tracing=False, intervention=False)
    output_bytes = 2 * 4 * width * 4
    profile = BatchProfile(
        profile_id=f'{VERSION}-{role}', certification_id=f'{VERSION}-{role}-sample',
        contract=contract, input_schema=(('input_ids', torch.int64), ('attention_mask', torch.int64)),
        input_shape=(1, 4), output_dtype=torch.float32, output_width=width,
        serial_atol=1e-6, serial_rtol=1e-5, approved_rows=(2,), max_requests=2,
        max_per_owner=1, max_batch_rows=2, max_token_positions=8,
        max_logical_input_bytes=128, max_logical_output_bytes=output_bytes,
        max_logical_total_bytes=128 + output_bytes)
    return validation.binding_from_profile(profile, model_fingerprint=model_sha,
        checkpoint_sha256=checkpoint_sha, adapter_sha256=adapter_sha, backend='cpu-eager')


def freeze(runtime, sources, *, source_commit):
    if type(source_commit) is not str or re.fullmatch(r'[0-9a-f]{40}', source_commit) is None:
        raise ValueError('source commit must be a full git SHA-1')
    try:
        runtime = json.loads(canonical(runtime))
        sources = json.loads(canonical(sources))
    except (TypeError, ValueError) as exc:
        raise ValueError('runtime and sources must be finite JSON data') from exc
    if type(runtime) is not dict or type(sources) is not dict or not sources:
        raise ValueError('runtime and source digests are required')
    if any(type(k) is not str or type(v) is not str or re.fullmatch(r'[0-9a-f]{64}', v) is None
           for k, v in sources.items()):
        raise ValueError('invalid source digest')
    manifest = {'version': VERSION, 'source_commit': source_commit, 'runtime': runtime,
                'source_sha256': sources, 'asset_sha256': {k: v[1] for k, v in ASSETS.items()},
                'raw_output_cap_bytes': OUTPUT_CAP}
    for key, control in (('primary_protocol', None), ('reversal_protocol', 'row_reversal'),
                         ('batch_mean_protocol', 'batch_mean')):
        binding = _profile(runtime, sources, control=control)
        discovery = [row_hash for row in DISCOVERY
                     for row_hash in validation.input_row_digests(binding, inputs(row))]
        manifest[key] = validation.freeze_protocol(binding, groups(), discovery_row_hashes=discovery)
    manifest['manifest_sha256'] = digest(manifest)
    return manifest


def verify_manifest(manifest, runtime, sources):
    if type(manifest) is not dict or 'source_commit' not in manifest:
        raise ValueError('invalid manifest')
    expected = freeze(runtime, sources, source_commit=manifest['source_commit'])
    if canonical(manifest) != canonical(expected):
        raise ValueError('manifest, runtime, source, or fixed protocol mismatch')


def source_hashes():
    from transformers.models.gpt2 import modeling_gpt2
    result = {name: file_sha256(ROOT / name) for name in SOURCE_PATHS}
    result['installed-transformers-modeling_gpt2.py'] = file_sha256(modeling_gpt2.__file__)
    return result


def verify_git_sources(source_commit):
    """Check the clean checkout contains the exact tracked source-commit blobs."""
    if type(source_commit) is not str or re.fullmatch(r'[0-9a-f]{40}', source_commit) is None:
        raise ValueError('invalid source commit')
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                              check=True, text=True).stdout.strip()
    try:
        git('merge-base', '--is-ancestor', source_commit, 'HEAD')
        if git('status', '--porcelain', '--untracked-files=all'):
            raise ValueError('checkout must be clean before the experiment')
        for path in SOURCE_PATHS:
            if git('rev-parse', f'{source_commit}:{path}') != git('rev-parse', f'HEAD:{path}'):
                raise ValueError('tracked source differs from source commit')
            entry = git('ls-tree', source_commit, '--', path)
            if not (entry.startswith('100644 blob ') or entry.startswith('100755 blob ')):
                raise ValueError('source must be a tracked ordinary file')
            blob = subprocess.run(['git', 'cat-file', 'blob', f'{source_commit}:{path}'],
                cwd=ROOT, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != file_sha256(ROOT / path):
                raise ValueError('on-disk source differs from the committed blob')
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('source commit is not an ancestor of a clean checkout') from exc


def configure_runtime():
    """Configure a fresh, isolated CPU-only interpreter before freezing or running."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('launch with CUDA_VISIBLE_DEVICES=-1')
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or not sys.pycache_prefix
            or not os.path.isabs(sys.pycache_prefix)
            or sys._xoptions.get('pycache_prefix') != sys.pycache_prefix
            or os.path.lexists(sys.pycache_prefix)):
        raise RuntimeError('use the -I -S -B source-only bootstrap with a nonexistent pycache prefix')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    if not torch.set_flush_denormal(False):
        raise RuntimeError('runtime cannot configure denormal handling')
    validation._require_denormals()
    torch.random.default_generator.manual_seed(SEED)
    return runtime_info()


def runtime_info():
    validation._require_denormals()
    if sys.byteorder != 'little' or torch.is_autocast_enabled('cpu'):
        raise RuntimeError('this experiment requires little endian and disabled CPU autocast')
    import numpy
    import transformers
    if torch.__version__ != '2.8.0+cu128' or transformers.__version__ != '4.57.3':
        raise RuntimeError('runtime versions differ from the frozen experiment environment')
    return {
        'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
        'processor': platform.processor(), 'python_byteorder': sys.byteorder,
        'isolated_python': bool(sys.flags.isolated), 'no_site': bool(sys.flags.no_site),
        'dont_write_bytecode': sys.dont_write_bytecode,
        'pycache_prefix': sys.pycache_prefix,
        'python_path': sys.path,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'numpy': numpy.__version__, 'torch_build': torch.__config__.show(),
        'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'mkldnn_enabled': torch.backends.mkldnn.enabled,
        'mkldnn_deterministic': torch.backends.mkldnn.deterministic,
        'default_dtype': str(torch.get_default_dtype()),
        'default_device': str(torch.get_default_device()),
        'cpu_autocast_enabled': torch.is_autocast_enabled('cpu'),
        'denormal_preserving_calling_thread': True,
        'environment': {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_DYNAMIC', 'MKL_DYNAMIC')},
    }


def _observation(request_id, frame):
    ids, mask = frame.f_locals.get('input_ids'), frame.f_locals.get('attention_mask')
    if type(ids) is not torch.Tensor or type(mask) is not torch.Tensor:
        raise RuntimeError('model-entry inputs are unavailable')
    return {'request_id': request_id, 'input_ids_shape': list(ids.shape),
            'attention_mask_shape': list(mask.shape), 'input_ids_dtype': str(ids.dtype),
            'attention_mask_dtype': str(mask.dtype), 'input_ids_device': str(ids.device),
            'attention_mask_device': str(mask.device)}


def _expected_observation(request_id, rows):
    return {'request_id': request_id, 'input_ids_shape': [rows, 4],
            'attention_mask_shape': [rows, 4], 'input_ids_dtype': 'torch.int64',
            'attention_mask_dtype': 'torch.int64', 'input_ids_device': 'cpu',
            'attention_mask_device': 'cpu'}


def expected_calls():
    return [_expected_observation(request_id, rows) for group in range(2)
            for request_id, rows in ((f'serial-{group}-0', 1), (f'serial-{group}-1', 1),
                                     (f'mixed-{group}', 2))]


def observations_complete(calls):
    return calls == expected_calls()


class ObservedExecutor:
    """Record model entry, then retain at most cap_bytes of completed CPU logits."""
    def __init__(self, worker, model, *, output_width, cap_bytes):
        self.worker, self.model, self.output_width = worker, model, output_width
        self.cap_bytes = cap_bytes
        self.calls, self.outputs = [], {}
        self.raw_bytes = 0
        self.code = type(model).forward.__code__

    def __call__(self, call_inputs, *, request_id, seed):
        observed = []
        def profiler(frame, event, arg):
            if event == 'call' and frame.f_code is self.code and frame.f_locals.get('self') is self.model:
                observed.append(_observation(request_id, frame))
        previous = sys.getprofile()
        try:
            sys.setprofile(profiler)
            result = self.worker.execute(ForwardRequest(request_id, call_inputs, seed=seed))
        finally:
            sys.setprofile(previous)
            self.calls.extend(observed)
        rows = call_inputs['input_ids'].shape[0]
        if observed != [_expected_observation(request_id, rows)]:
            raise RuntimeError('model-entry observation is missing, duplicated, or incompatible')
        if result.request_id != request_id:
            raise RuntimeError('worker returned a different request ID')
        value = result.logits
        if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                or value.dtype != torch.float32 or tuple(value.shape) != (rows, 4, self.output_width)
                or value.requires_grad or not bool(torch.isfinite(value).all())):
            raise RuntimeError('invalid completed logits')
        size = value.numel() * value.element_size()
        if request_id in self.outputs or self.raw_bytes + size > self.cap_bytes:
            raise RuntimeError('duplicate output or retained raw-output cap exceeded')
        self.outputs[request_id] = value
        self.raw_bytes += size
        return value


def save_outputs(path, outputs, *, cap_bytes):
    if type(outputs) is not dict or type(cap_bytes) is not int or cap_bytes <= 0:
        raise ValueError('invalid output archive arguments')
    size = 0
    for name, value in outputs.items():
        if (type(name) is not str or re.fullmatch(r'(serial-[01]-[01]|mixed-[01])', name) is None
                or type(value) is not torch.Tensor or value.device.type != 'cpu'
                or value.dtype != torch.float32 or value.layout != torch.strided):
            raise ValueError('invalid raw output')
        size += value.numel() * value.element_size()
    if size > cap_bytes:
        raise ValueError('raw output cap exceeded')
    entries = []
    with zipfile.ZipFile(path, mode='x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in outputs.items():
            raw = value.contiguous().numpy().astype('<f4', copy=False).tobytes(order='C')
            member = f'{name}.f32'
            archive.writestr(member, raw)
            entries.append({'request_id': name, 'member': member, 'shape': list(value.shape),
                            'dtype': 'little-endian-float32', 'bytes': len(raw),
                            'sha256': hashlib.sha256(raw).hexdigest()})
    return {'raw_bytes': size, 'archive_sha256': file_sha256(path), 'entries': entries}


def load_outputs(path, info):
    import numpy as np
    if file_sha256(path) != info['archive_sha256'] or info['raw_bytes'] > OUTPUT_CAP:
        raise ValueError('archive checksum or cap mismatch')
    result = {}
    with zipfile.ZipFile(path) as archive:
        entries = info['entries']
        if (len({entry['member'] for entry in entries}) != len(entries)
                or archive.namelist() != [entry['member'] for entry in entries]):
            raise ValueError('archive membership mismatch')
        total = 0
        for entry in entries:
            name, shape = entry['request_id'], entry['shape']
            if (type(name) is not str or re.fullmatch(r'(serial-[01]-[01]|mixed-[01])', name) is None
                    or type(shape) is not list or len(shape) != 3
                    or any(type(v) is not int or v <= 0 for v in shape)
                    or name in result or entry['member'] != f'{name}.f32'
                    or entry['dtype'] != 'little-endian-float32'
                    or entry['bytes'] != 4 * shape[0] * shape[1] * shape[2]
                    or archive.getinfo(entry['member']).file_size != entry['bytes']):
                raise ValueError('invalid archived output metadata')
            total += entry['bytes']
            if total > OUTPUT_CAP or total > info['raw_bytes']:
                raise ValueError('raw archive exceeds its declared cap')
            raw = archive.read(entry['member'])
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('output checksum mismatch')
            result[name] = torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy().reshape(shape))
    if sum(value.numel() * value.element_size() for value in result.values()) != info['raw_bytes']:
        raise ValueError('raw output length mismatch')
    return result


def _playback(outputs, *, reversed_rows):
    def execute(call_inputs, *, request_id, seed):
        output = outputs[request_id]
        return output.flip(0) if reversed_rows and request_id.startswith('mixed-') else output
    return execute


def _run(protocol, execute):
    return validation.run_validation(protocol, protocol['binding'], groups(), execute,
        expected_protocol_sha256=protocol['protocol_sha256'])


def run_controls(manifest, outputs):
    def batch_mean(call_inputs, *, request_id, seed):
        values = call_inputs['input_ids'].to(torch.float32)
        return (values - values.mean(dim=0, keepdim=True)).unsqueeze(-1)
    return {'positive_playback': _run(manifest['primary_protocol'], _playback(outputs, reversed_rows=False)),
            'reversed_playback': _run(manifest['reversal_protocol'], _playback(outputs, reversed_rows=True)),
            'batch_mean': _run(manifest['batch_mean_protocol'], batch_mean)}


def state_sha256(model):
    """Stream named tensor state without copying a full large tensor to bytes."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if value.device.type != 'cpu' or value.layout != torch.strided or not value.is_contiguous():
            raise RuntimeError('model state must be contiguous CPU tensors')
        header = canonical({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)})
        h.update(len(header).to_bytes(8, 'little'))
        h.update(header)
        view = memoryview(value.detach().view(torch.uint8).reshape(-1).numpy())
        h.update(len(view).to_bytes(8, 'little'))
        for offset in range(0, len(view), 1024 * 1024):
            h.update(view[offset:offset + 1024 * 1024])
    return h.hexdigest()


def write_json(path, data):
    with open(path, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def write_summary_atomic(path, data):
    """Publish one complete terminal summary with an atomic rename."""
    temp = path.with_suffix('.json.tmp')
    with open(temp, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.replace(temp, path)


def execute_experiment(manifest, *, cache_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=False, exist_ok=False)
    write_json(output_dir / 'attempt.json', {'version': VERSION, 'status': 'started',
        'manifest_sha256': manifest['manifest_sha256'],
        'started_utc': datetime.now(timezone.utc).isoformat()})
    executor = worker = None
    close_attempted = False
    try:
        verify_git_sources(manifest['source_commit'])
        runtime = configure_runtime()
        sources = source_hashes()
        verify_manifest(manifest, runtime, sources)
        loaded = load_distilgpt2(cache_dir=str(cache_dir), allow_download=False)
        metadata, model = loaded.metadata, loaded.model
        if (metadata['model_id'] != MODEL_ID or metadata['model_revision'] != REVISION
                or metadata['asset_sha256'] != manifest['asset_sha256']
                or loaded.tokenizer.eos_token_id != 50256 or model.config.use_cache is not False
                or model.config._attn_implementation != 'eager'):
            raise RuntimeError('loaded checkpoint, tokenizer, or model policy differs from protocol')
        before = state_sha256(model)
        worker = ResidentWorker(MODEL_ID, model, device='cpu', max_batch_size=2,
                                max_sequence_length=4, max_input_bytes=128,
                                max_output_bytes=2 * 4 * WIDTH * 4)
        executor = ObservedExecutor(worker, model, output_width=WIDTH, cap_bytes=OUTPUT_CAP)
        report = _run(manifest['primary_protocol'], executor)
        after = state_sha256(model)
        verify_manifest(manifest, runtime_info(), source_hashes())
        validation.save_evidence(output_dir / 'primary-report.json', report,
            protocol=manifest['primary_protocol'],
            expected_protocol_sha256=manifest['primary_protocol']['protocol_sha256'])
        output_info = save_outputs(output_dir / 'logits.zip', executor.outputs, cap_bytes=OUTPUT_CAP)
        archived = load_outputs(output_dir / 'logits.zip', output_info)
        controls = run_controls(manifest, archived)
        complete = (report['complete'] and observations_complete(executor.calls) and before == after
                    and controls['positive_playback'] == report
                    and controls['reversed_playback']['complete'] and controls['batch_mean']['complete'])
        def all_mismatch(control):
            return (control['status'] == 'failed'
                    and all(case['status'] == 'mismatch' for group in control['groups']
                            for case in group['cases']))
        passed = (complete and report['status'] == 'passed'
                  and all_mismatch(controls['reversed_playback'])
                  and all_mismatch(controls['batch_mean']))
        summary = {'version': VERSION, 'status': 'passed' if passed else 'failed' if complete else 'incomplete',
                   'complete': complete, 'manifest_sha256': manifest['manifest_sha256'],
                   'model_metadata': metadata, 'model_state_sha256_before': before,
                   'model_state_sha256_after': after, 'observed_model_calls': executor.calls,
                   'output_archive': output_info, 'controls': controls,
                   'primary_evidence_sha256': report['evidence_sha256'],
                   'finished_utc': datetime.now(timezone.utc).isoformat()}
        close_attempted = True
        worker.close()
        worker = None
        write_summary_atomic(output_dir / 'summary.json', summary)
        return summary
    except BaseException as exc:
        cleanup_error = None
        if worker is not None and not close_attempted:
            try:
                worker.close()
            except BaseException as close_exc:
                cleanup_error = type(close_exc).__name__
        if not (output_dir / 'summary.json').exists():
            try:
                write_json(output_dir / 'failure.json', {'version': VERSION, 'status': 'incomplete',
                    'exception_type': type(exc).__name__, 'cleanup_exception_type': cleanup_error,
                    'observed_model_calls': executor.calls if executor is not None else [],
                    'finished_utc': datetime.now(timezone.utc).isoformat()})
            except BaseException:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('freeze')
    prepare.add_argument('--source-commit', required=True)
    prepare.add_argument('--manifest', required=True)
    run = sub.add_parser('run')
    run.add_argument('--manifest', required=True)
    run.add_argument('--expected-manifest-sha256', required=True)
    run.add_argument('--cache-dir', required=True)
    run.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        verify_git_sources(args.source_commit)
        manifest = freeze(configure_runtime(), source_hashes(), source_commit=args.source_commit)
        write_json(args.manifest, manifest)
        print(manifest['manifest_sha256'])
    else:
        with open(args.manifest, encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest.get('manifest_sha256') != args.expected_manifest_sha256:
            raise ValueError('manifest digest differs from separately supplied expected digest')
        result = execute_experiment(manifest, cache_dir=args.cache_dir, output_dir=args.output_dir)
        print(json.dumps({'status': result['status'], 'complete': result['complete']}))
        if result['status'] != 'passed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
