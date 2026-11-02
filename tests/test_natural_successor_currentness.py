"""Fabricated official responses; no model, claim or private input."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'tools/natural_successor_currentness.py'
spec = importlib.util.spec_from_file_location('_successor_currentness_test', SOURCE)
currentness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(currentness)


@pytest.fixture
def observation_inputs():
    raw = json.dumps({'version': 'distilgpt2-development-exposures-v27', 'records': []},
                     sort_keys=True, separators=(',', ':')).encode('ascii')
    source = b'# fabricated reviewed exposure module\n'
    head = 'a' * 40
    def api(url):
        if url.endswith('/commits/main'):
            return {'sha': head}
        assert url.endswith('/contents/looking_glass?ref=' + head)
        return [{'name': 'pretrained_exposures_v26.py', 'type': 'file'},
                {'name': 'pretrained_exposures_v27.py', 'type': 'file'}]
    def read(url):
        assert url.endswith('/' + head + '/looking_glass/pretrained_exposures_v27.py')
        return source
    return {'version': 27, 'source_sha256': hashlib.sha256(source).hexdigest(),
            'inventory_sha256': hashlib.sha256(raw).hexdigest(), 'inventory_bytes': raw,
            'api_reader': api, 'source_reader': read}


def test_exact_source_payload_and_stable_official_head(observation_inputs):
    result = currentness.observe(**observation_inputs)
    assert result == {'schema': 'mlg-natural-successor-currentness-observation-v1',
                      'head': 'a' * 40, 'version': 27, 'source_sha256': observation_inputs['source_sha256'],
                      'inventory_sha256': observation_inputs['inventory_sha256']}


@pytest.mark.parametrize('change', ['source', 'payload', 'version', 'newer', 'head', 'listing'])
def test_changed_or_unavailable_state_stops(observation_inputs, change):
    values = dict(observation_inputs)
    if change == 'source':
        values['source_reader'] = lambda url: b'changed source'
    elif change == 'payload':
        values['inventory_bytes'] += b' '
    elif change == 'version':
        values['version'] = 26
    elif change == 'newer':
        original = values['api_reader']
        def newer(url):
            result = original(url)
            return result + [{'name': 'pretrained_exposures_v28.py', 'type': 'file'}] if type(result) is list else result
        values['api_reader'] = newer
    elif change == 'head':
        calls = []
        original = values['api_reader']
        def moved(url):
            if url.endswith('/commits/main'):
                calls.append(1)
                return {'sha': ('a' if len(calls) == 1 else 'b') * 40}
            return original(url)
        values['api_reader'] = moved
    else:
        values['api_reader'] = lambda url: {'sha': 'a' * 40} if url.endswith('/commits/main') else []
    with pytest.raises(currentness.CurrentnessError):
        currentness.observe(**values)


def test_duplicate_keys_and_noncanonical_payload_stop(observation_inputs):
    for raw in (b'{"version":"x","version":"distilgpt2-development-exposures-v27"}',
                observation_inputs['inventory_bytes'] + b'\n'):
        with pytest.raises(currentness.CurrentnessError):
            currentness.observe(**dict(observation_inputs, inventory_bytes=raw,
                                       inventory_sha256=hashlib.sha256(raw).hexdigest()))
