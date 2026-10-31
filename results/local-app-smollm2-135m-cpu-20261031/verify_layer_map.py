"""Offline captured-layer navigation checks, not a rendered-browser or GPU test.

Uses the exact pinned local SmolLM2 assets; never downloads or edits a checkpoint.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading

import verify_reference_vectors as provenance


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: verification assertions must remain enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New JSON file; never overwritten')
    parser.add_argument('--implementation-root', default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    implementation = Path(args.implementation_root).expanduser().resolve()
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('output must be a new file in an existing directory')
    if output.resolve().is_relative_to(checkpoint) or Path(tempfile.gettempdir()).resolve().is_relative_to(checkpoint):
        raise ValueError('evidence and temporary output must not be inside checkpoint')
    assert provenance.file_hash(provenance.HISTORICAL_RECORD) == provenance.HISTORICAL_RECORD_SHA256
    historical = json.loads(provenance.HISTORICAL_RECORD.read_text())['source']
    assert historical['repository'] == 'HuggingFaceTB/SmolLM2-135M' and historical['revision'] == provenance.REVISION
    assets = provenance.verify_checkpoint(checkpoint, historical['asset_manifest'])
    scripts = {p.name: provenance.file_hash(p) for p in (Path(__file__).resolve(), Path(provenance.__file__).resolve())}
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(implementation))
    import torch, transformers
    from looking_glass import app, app_runtime, app_storage
    for module in (app, app_runtime, app_storage):
        assert Path(module.__file__).resolve().is_relative_to(implementation)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    sources = provenance.implementation_hashes(implementation)
    static = {p.name: provenance.file_hash(p) for p in (implementation / 'looking_glass/app_static').iterdir() if p.is_file()}
    with tempfile.TemporaryDirectory(prefix='mlg-layer-verification-') as temporary:
        state = app.AppState(export_root=Path(temporary) / 'exports')
        server = app.create_server(state); worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        def call(path, body=None, expected=200):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=60)
            headers = {'X-MLG-Token': state.token}
            if body is not None: headers['Content-Type'] = 'application/json'
            try:
                connection.request('GET' if body is None else 'POST', path,
                                   body=None if body is None else json.dumps(body), headers=headers)
                response = connection.getresponse(); data = response.read()
                assert response.status == expected, (path, response.status, data[:200])
                return json.loads(data)
            finally:
                connection.close()
        def bits():
            return [hashlib.sha256(value.view(torch.int32).numpy().astype('<i4', copy=False).tobytes()).hexdigest()
                    for value in state.store._values]
        def maps(capture):
            result = []
            for step in run['steps']:
                forward = step['app_forward_index']; layers = []
                for offset in (0, 12, 24):
                    page = call(f'/api/layer-map?forward={forward}&offset={offset}&limit=12&capture_id={capture}')
                    assert page.pop('capture_id') == capture
                    assert page['total_layers'] == 30 and page['known_events'] == page['selected_forward_events'] == 90
                    assert page['unmapped_events'] == page['missing_forward_events'] == 0
                    assert page['known_events'] + page['other_forward_events'] == run['event_count']
                    layers.extend(page['layers']); result.append(page)
                assert [layer['layer_index'] for layer in layers] == list(range(30))
                assert all(layer['stack'] == 'input' and len(layer['nodes']) == 3 for layer in layers)
                actual = {node['module_path'] for layer in layers for node in layer['nodes']}
                assert actual == {f'model.layers.{i}{suffix}' for i in range(30) for suffix in ('', '.self_attn', '.mlp')}
            return result
        try:
            call('/api/load', {'path': str(checkpoint), 'device': 'cpu'})
            model = state.runtime.model; parameters = provenance.state_hashes(model)
            run = call('/api/run', {'prompt': 'Gravity is', 'settings': {'max_new_tokens': 2}})
            assert 1 <= len(run['steps']) <= 2 and run['event_count'] == 90 * len(run['steps'])
            original = bits(); mapped = maps(run['capture_id']); representatives = []
            for page in mapped:
                for layer in page['layers']:
                    for node in layer['nodes']:
                        event = node['representative_event']; index = event['index']
                        assert event['matched_vector_index'] == 0 and node['event_count'] == 1
                        step = next(step for step in run['steps'] if step['app_forward_index'] == page['selected_forward'])
                        assert node['vector_count'] == len(step['input_token_ids'])
                        assert node['tensor_bytes'] == node['vector_count'] * 576 * 4
                        raw = call(f'/api/tensor?event={index}&vector=0&limit=4096&capture_id=' + run['capture_id'])
                        coordinate = raw['coordinate']; value = state.store._values[index].reshape(-1, 576)[0]
                        expected_bits = [f'0x{int(item) & 0xffffffff:08x}' for item in value.view(torch.int32).numpy()]
                        assert raw['raw_bits'] == expected_bits and len(raw['values']) == 576
                        assert coordinate['status'] == 'known' and coordinate['module_path'] == node['module_path']
                        assert coordinate['app_forward_index'] == page['selected_forward'] and coordinate['layer_index'] == layer['layer_index']
                        assert event['metadata']['output_path'] == ([0] if node['module_path'].endswith('.self_attn') else [])
                        representatives.append((index, raw))
            first = representatives[0][0]
            pin = call('/api/pin-reference', {'capture_id': run['capture_id'], 'event': first, 'vector': 0, 'expected_reference_id': None})
            paired = call(f'/api/compare-vector?event={first}&vector=0&capture_id={run["capture_id"]}&reference_id={pin["reference_id"]}')
            assert paired['metrics']['raw_bit_changed_count'] == 0
            assert bits() == original and provenance.state_hashes(model) == parameters
            assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
            assert not state.store.export_root.exists()
            saved = call('/api/save', {'name': 'layer-map-check', 'capture_id': run['capture_id']})
            del model, value  # The verifier must not retain model/capture aliases across explicit unload/clear.
            call('/api/unload', {}); call('/api/clear', {})
            opened = call('/api/open', {'path': saved['path']})
            assert opened['capture_id'] != run['capture_id'] and maps(opened['capture_id']) == mapped
            for index, raw in representatives:
                assert call(f'/api/tensor?event={index}&vector=0&limit=4096&capture_id=' + opened['capture_id']) == raw
            assert bits() == original and state.runtime.model is None
            call('/api/layer-map?capture_id=' + run['capture_id'], expected=400)
        finally:
            server.shutdown(); worker.join(5); server.server_close()
            assert not worker.is_alive()
    assert provenance.verify_checkpoint(checkpoint, historical['asset_manifest']) == assets
    assert provenance.implementation_hashes(implementation) == sources
    assert all(provenance.file_hash(implementation / 'looking_glass/app_static' / name) == digest for name, digest in static.items())
    assert all(provenance.file_hash(Path(__file__).with_name(name)) == digest for name, digest in scripts.items())
    report = {'status': 'passed', 'model': historical['repository'], 'revision': provenance.REVISION,
              'device': 'cpu', 'dtype': 'float32', 'python': platform.python_version(), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'prompt': 'Gravity is', 'forward_count': len(run['steps']),
              'captured_events': run['event_count'], 'retained_tensor_bytes': run['total_bytes'],
              'captured_layers_per_forward': 30, 'exact_nodes_per_forward': 90,
              'representative_vectors_checked_before_and_after_reopen': len(representatives), 'features_per_vector': 576,
              'all_retained_tensor_hashes_unchanged': True, 'parameters_and_buffers_unchanged': True,
              'first_representative_self_reference_exact': True, 'stale_capture_revision_rejected': True,
              'map_metadata_identical_after_reopen': True, 'checkpoint_assets_unchanged': True,
              'checkpoint_file_sha256': assets, 'implementation_sha256': sources, 'static_asset_sha256': static,
              'verification_script_sha256': scripts, 'historical_manifest_record_sha256': provenance.HISTORICAL_RECORD_SHA256,
              'map_response_sha256_with_execution_ids_not_portable': provenance.digest(mapped),
              'scope': 'Authenticated HTTP and captured metadata/values from one CPU checkpoint. No browser rendering, GPU, neuron connectivity, semantic layout, or causal explanation claim.'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'forward_count', 'representative_vectors_checked_before_and_after_reopen')}))


if __name__ == '__main__':
    main()
