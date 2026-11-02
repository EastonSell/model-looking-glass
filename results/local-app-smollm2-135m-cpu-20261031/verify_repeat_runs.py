"""Bounded offline same-process HTTP A/B/A verification; not a long-term leak proof.

Requires already-installed app dependencies and the exact existing local checkpoint.
No downloads, training, checkpoint writes or recovery from quarantined execution.
"""
import argparse
import gc
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
import weakref

import verify_reference_vectors as provenance


def rss_bytes():
    """Linux current process RSS, not a portable allocator or GPU measurement."""
    try:
        fields = Path('/proc/self/statm').read_text().split()
        return int(fields[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def hooks(model):
    return {name: {attribute: tuple(getattr(module, attribute).keys()) for attribute in
                  ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks')}
            for name, module in model.named_modules()}


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: verification assertions must remain enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New JSON file; never overwritten')
    parser.add_argument('--implementation-root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--cycles', type=int, default=20, help='1..200 measured A/B/A cycles after one warmup cycle')
    args = parser.parse_args()
    if not 1 <= args.cycles <= 200:
        raise ValueError('cycles must be from 1 to 200')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    implementation = Path(args.implementation_root).expanduser().resolve()
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('output must be a new file in an existing directory')
    if output.resolve().is_relative_to(checkpoint) or Path(tempfile.gettempdir()).resolve().is_relative_to(checkpoint):
        raise ValueError('evidence and temporary output must not be inside checkpoint')
    if provenance.file_hash(provenance.HISTORICAL_RECORD) != provenance.HISTORICAL_RECORD_SHA256:
        raise RuntimeError('historical manifest record changed')
    historical = json.loads(provenance.HISTORICAL_RECORD.read_text())['source']
    if historical['repository'] != 'HuggingFaceTB/SmolLM2-135M' or historical['revision'] != provenance.REVISION:
        raise RuntimeError('unexpected checkpoint identity')
    assets = provenance.verify_checkpoint(checkpoint, historical['asset_manifest'])
    scripts = {p.name: provenance.file_hash(p) for p in (Path(__file__).resolve(), Path(provenance.__file__).resolve())}
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(implementation))
    import torch, transformers
    from looking_glass import app, app_runtime, app_storage
    for module in (app, app_runtime, app_storage):
        if not Path(module.__file__).resolve().is_relative_to(implementation):
            raise RuntimeError('wrong implementation import origin')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    sources = provenance.implementation_hashes(implementation)
    settings = {'max_new_tokens': 2, 'max_trace_bytes': 16 * 1024**2}
    prompts = ['Gravity is', 'Electricity is', 'Gravity is']
    baselines, samples, released, recoveries = {}, [], 0, 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='mlg-repeat-verification-') as temporary:
        state = app.AppState(export_root=Path(temporary) / 'exports')
        server = app.create_server(state)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        def call(path, body=None, expected=200):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=60)
            headers = {'X-MLG-Token': state.token}
            if body is not None: headers['Content-Type'] = 'application/json'
            try:
                connection.request('GET' if body is None else 'POST', path,
                                   body=None if body is None else json.dumps(body), headers=headers)
                response = connection.getresponse(); data = response.read()
                if response.status != expected:
                    raise RuntimeError(f'{path}: expected {expected}, got {response.status}: {data[:300]!r}')
                return json.loads(data)
            finally:
                connection.close()
        def fingerprint():
            with state.lock:
                run = state.store._run
                values = state.store._values
                events = state.store._events
                steps = [{key: step.get(key) for key in ('app_forward_index', 'input_token_ids',
                          'input_attention_mask', 'selected_token_id', 'use_cache')} for step in run['steps']]
                assert 1 <= len(steps) <= 2 and len(values) == 90 * len(steps)
                tensors = []
                for value, event in zip(values, events):
                    assert value.dtype == torch.float32 and value.device.type == 'cpu'
                    raw = value.view(torch.int32).numpy().astype('<i4', copy=False).tobytes()
                    metadata = event['metadata']
                    tensors.append({'shape': list(value.shape), 'sha256': hashlib.sha256(raw).hexdigest(),
                                    'module_path': metadata['module_path'], 'output_path': metadata['output_path'],
                                    'layer_index': metadata['layer_index'], 'app_forward_index': metadata['app_forward_index']})
                return {'answer': run['answer'], 'steps': steps, 'tensors': tensors}
        try:
            call('/api/load', {'path': str(checkpoint), 'device': 'cpu'})
            model = state.runtime.model; binding = state.runtime.capture_binding()
            before_parameters = provenance.state_hashes(model); before_hooks = hooks(model)
            assert all(not slots for module in before_hooks.values() for slots in module.values())
            assert model.lm_head.weight is model.model.embed_tokens.weight
            for cycle in range(args.cycles + 1):
                for position, prompt in enumerate(prompts):
                    if position == 1 and cycle in (0, args.cycles):
                        previous = call('/api/status')['run']; original = fingerprint()
                        call('/api/run', {'prompt': prompt, 'settings': {'max_trace_bytes': 2**40}}, expected=400)
                        assert call('/api/status')['run'] == previous and fingerprint() == original
                        assert state.runtime.capture_binding() is binding
                        recoveries += 1
                    old = [weakref.ref(value) for value in state.store._values]
                    summary = call('/api/run', {'prompt': prompt, 'settings': settings})
                    gc.collect()
                    assert all(reference() is None for reference in old), 'superseded capture tensor remains retained'
                    released += len(old)
                    current = fingerprint()
                    if prompt not in baselines: baselines[prompt] = current
                    assert current == baselines[prompt], 'same input changed output IDs or captured bits'
                    assert state.runtime.model is model and state.runtime.capture_binding() is binding
                    assert hooks(model) == before_hooks and state.runtime.info()['state'] == 'ready'
                    if cycle:
                        samples.append({'cycle': cycle, 'position': position, 'prompt': prompt,
                                        'rss_bytes_after_gc': rss_bytes(), 'event_count': summary['event_count'],
                                        'capture_bytes': summary['total_bytes'], 'fingerprint_sha256': provenance.digest(current)})
            assert provenance.state_hashes(model) == before_parameters, 'model parameters or buffers changed'
            final = [weakref.ref(value) for value in state.store._values]
            call('/api/clear', {}); gc.collect()
            assert all(reference() is None for reference in final)
            released += len(final)
            assert state.runtime.model is model and state.runtime.capture_binding() is binding
            assert hooks(model) == before_hooks and not (Path(temporary) / 'exports').exists()
            call('/api/unload', {})
            assert state.runtime.info() is None
        finally:
            server.shutdown(); worker.join(5); server.server_close()
            if worker.is_alive(): raise RuntimeError('HTTP server did not stop')
    assert provenance.verify_checkpoint(checkpoint, historical['asset_manifest']) == assets
    assert provenance.implementation_hashes(implementation) == sources
    assert all(provenance.file_hash(Path(__file__).with_name(name)) == digest for name, digest in scripts.items())
    rss = [sample['rss_bytes_after_gc'] for sample in samples if sample['rss_bytes_after_gc'] is not None]
    report = {'status': 'passed', 'model': historical['repository'], 'revision': historical['revision'],
              'device': 'cpu', 'dtype': 'float32', 'python': platform.python_version(),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'loaded_once_no_restart': True, 'warmup_runs': 3, 'measured_runs': len(samples),
              'measured_cycles': args.cycles, 'sequence': prompts, 'settings': settings,
              'same_input_output_and_all_capture_bits_equal': True,
              'model_parameter_and_buffer_hashes_unchanged': True, 'no_residual_hooks': True,
              'released_capture_tensors_verified_by_weakref': released,
              'recoverable_preflight_failures_followed_by_success': recoveries,
              'explicit_clear_releases_last_capture_without_retiring_model': True,
              'checkpoint_assets_unchanged': True, 'implementation_sha256': sources,
              'verification_script_sha256': scripts,
              'historical_manifest_record_sha256': provenance.HISTORICAL_RECORD_SHA256,
              'checkpoint_file_sha256': assets,
              'baseline_fingerprints': {prompt: provenance.digest(data) for prompt, data in baselines.items()},
              'rss_observation': None if not rss else {'source': 'Linux /proc/self/statm current process RSS after GC',
                 'first_bytes': rss[0], 'last_bytes': rss[-1], 'min_bytes': min(rss), 'max_bytes': max(rss),
                 'last_minus_first_bytes': rss[-1] - rss[0],
                 'interpretation': 'Finite observational trace; allocator caching/RSS is not a leak proof or total-memory guarantee.'},
              'samples': samples, 'elapsed_seconds_not_benchmark': round(time.monotonic() - started, 3),
              'scope': 'One pinned CPU checkpoint and short deterministic prompts in one process. No quality, unlimited-session, GPU, browser, or restart-free recovery-from-quarantine claim.'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'measured_runs', 'released_capture_tensors_verified_by_weakref', 'rss_observation')}))


if __name__ == '__main__':
    main()
