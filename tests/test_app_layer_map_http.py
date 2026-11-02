"""Authenticated layer-map navigation to actual local CPU capture payloads."""
import base64
import hashlib
import json
import threading

import pytest
import torch

from looking_glass.app import AppState, create_server
from test_app_http import call, server


def test_layer_map_requires_current_capture_revision(server):
    code, body, _ = call(server, '/api/run', 'POST', {'prompt': 'fixture', 'settings': {}})
    assert code == 200
    capture = json.loads(body)['capture_id']
    assert call(server, '/api/layer-map')[0] == 400
    assert call(server, '/api/layer-map?capture_id=stale')[0] == 400
    assert call(server, '/api/layer-map?capture_id=' + capture, token=False)[0] == 403
    code, body, _ = call(server, '/api/layer-map?capture_id=' + capture)
    assert code == 200, body
    data = json.loads(body)
    assert data['capture_id'] == capture and data['layers'] == []
    assert data['total_all_events'] == data['missing_forward_events'] == 1
    for suffix in ('&forward=-1', '&forward=true', '&limit=51', '&offset=-1', '&forward=0&forward=1'):
        assert call(server, '/api/layer-map?capture_id=' + capture + suffix)[0] == 400
    call(server, '/api/clear', 'POST', {})
    assert call(server, '/api/layer-map?capture_id=' + capture)[0] == 400


@pytest.mark.parametrize('family', ['gpt2', 'distilbert', 't5', 'vit', 'llama'])
def test_layer_map_actual_boundaries_vectors_reference_and_reopen(tmp_path, family):
    from test_app_runtime import local_checkpoint, image_bytes
    from test_app_llama_http import llama_checkpoint
    checkpoint = tmp_path / 'model'
    if family == 'llama': llama_checkpoint(checkpoint, 2)
    else: local_checkpoint(checkpoint, family)
    state = AppState(export_root=tmp_path / 'exports')
    httpd = create_server(state); worker = threading.Thread(target=httpd.serve_forever, daemon=True); worker.start()
    endpoint = state, httpd
    def request(path, body=None, expected=200):
        code, data, _ = call(endpoint, path, 'GET' if body is None else 'POST', body)
        assert code == expected, data
        return json.loads(data)
    def bits():
        return [hashlib.sha256(value.view(torch.int32).numpy().astype('<i4', copy=False).tobytes()).hexdigest()
                for value in state.store._values]
    try:
        request('/api/load', {'path': str(checkpoint), 'device': 'cpu'})
        body = {'prompt': 'hello [MASK]' if family == 'distilbert' else 'hello world', 'settings': {'max_new_tokens': 2}}
        if family == 'vit': body['image_base64'] = base64.b64encode(image_bytes()).decode()
        run = request('/api/run', body); capture = run['capture_id']; original = bits()
        mapped = request('/api/layer-map?forward=0&capture_id=' + capture)
        assert mapped['capture_id'] == capture
        assert mapped['total_layers'] == (3 if family == 't5' else 2)  # T5 fixture has 2 encoder + 1 decoder layers.
        selected = [row for row in state.store._events if row['metadata']['app_forward_index'] == 0]
        assert mapped['known_events'] == mapped['selected_forward_events'] == len(selected)
        assert mapped['unmapped_events'] == mapped['missing_forward_events'] == 0
        assert mapped['other_forward_events'] + mapped['known_events'] == run['event_count']
        assert mapped['total_all_events'] == run['event_count']
        assert sum(node['event_count'] for layer in mapped['layers'] for node in layer['nodes']) == len(selected)
        assert sum(node['vector_count'] for layer in mapped['layers'] for node in layer['nodes']) == sum(e['vector_count'] for e in selected)
        assert sum(node['tensor_bytes'] for layer in mapped['layers'] for node in layer['nodes']) == sum(e['tensor_bytes'] for e in selected)
        for layer in mapped['layers']:
            assert len(layer['nodes']) == 3
            for node in layer['nodes']:
                event = node['representative_event']; index = event['index']
                assert event['matched_vector_index'] == 0
                raw = request(f'/api/tensor?event={index}&vector=0&capture_id={capture}')
                coordinate = raw['coordinate']
                assert coordinate['status'] == 'known'
                assert coordinate['module_path'] == node['module_path']
                assert coordinate['stack'] == layer['stack'] and coordinate['layer_index'] == layer['layer_index']
                assert coordinate['app_forward_index'] == 0
                assert raw == state.store.tensor(index, 0)
                assert event['metadata']['output_path'] == node['output_path']
        assert bits() == original  # No reduction, recomputation or tensor mutation to draw the map.
        pages = [request(f'/api/layer-map?forward=0&offset={i}&limit=1&capture_id={capture}')['layers'][0]
                 for i in range(mapped['total_layers'])]
        assert pages == mapped['layers']
        first = mapped['layers'][0]['nodes'][0]['representative_event']['index']
        pin_body = {'capture_id': capture, 'event': first, 'vector': 0, 'expected_reference_id': None}
        if family in ('gpt2', 'llama'):
            pin = request('/api/pin-reference', pin_body)
            pair = request(f'/api/compare-vector?reference_id={pin["reference_id"]}&capture_id={capture}&event={first}&vector=0')
            assert pair['metrics']['raw_bit_changed_count'] == 0
        else:
            request('/api/pin-reference', pin_body, expected=400)  # Map navigation does not broaden reference authority.
        exported = request('/api/save', {'name': family, 'capture_id': capture})
        request('/api/unload', {}); request('/api/clear', {})
        opened = request('/api/open', {'path': exported['path']})
        reopened = request('/api/layer-map?forward=0&capture_id=' + opened['capture_id'])
        assert reopened.pop('capture_id') != mapped.pop('capture_id')
        assert reopened == mapped and bits() == original
        request('/api/layer-map?capture_id=' + capture, expected=400)
    finally:
        httpd.shutdown(); worker.join(5); httpd.server_close()
        assert not worker.is_alive()
