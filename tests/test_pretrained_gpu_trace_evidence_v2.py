"""Fabricated CPU checks for the bounded raw trace evidence writer."""

import hashlib
import importlib
import json

import pytest
import torch


def _writer(directory):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    return module.TensorEvidence(directory, torch=torch, hidden=4, vocab=7)


def _logits():
    values = torch.arange(35, dtype=torch.float32).reshape(1, 5, 7)
    values[0, 0, 0] = -0.0
    values[0, 0, 1] = torch.frombuffer(bytearray.fromhex('01000000'), dtype=torch.float32)[0]
    return values


def _captures():
    return [torch.full((1, 2, 4), float(index), dtype=torch.float32) for index in range(9)]


def _write_all(writer):
    actions = [('logit', 1, 'baseline_cold'), ('logit', 2, 'baseline_warm'),
               ('logit', 3, 'reference'), ('capture', 'reference', 0),
               ('logit', 4, 'reference'), ('capture', 'reference', 1),
               ('logit', 5, 'direct'), ('capture', 'direct', 0),
               ('logit', 6, 'direct'), ('capture', 'direct', 1),
               ('logit', 7, 'gradient'), ('capture', 'gradient', 0),
               ('logit', 8, 'gradient'), ('capture', 'gradient', 1),
               ('logit', 9, 'id'), ('logit', 10, 'id')]
    for kind, first, second in actions:
        if kind == 'logit':
            writer.write_logit(first, second, _logits())
        else:
            writer.write_capture(first, second, _captures())


def _complete(directory):
    writer = _writer(directory)
    _write_all(writer)
    return writer.finish()


def test_completed_evidence_preserves_ordered_float32_bytes_and_digests(tmp_path):
    binding = _complete(tmp_path)
    manifest_bytes = (tmp_path / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert binding['manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert binding['payload_bytes'] == 10 * 140 + 6 * 288
    assert len(manifest['logits']) == 10 and len(manifest['captures']) == 6
    raw_logits = (tmp_path / 'logits-001.f32').read_bytes()
    assert len(raw_logits) == 140 and raw_logits[:4] == bytes.fromhex('00000080')
    assert raw_logits[4:8] == bytes.fromhex('01000000')
    first_capture = (tmp_path / 'reference-1.f32').read_bytes()
    assert len(first_capture) == 288 and first_capture[:32] == bytes(32)
    first_event = manifest['captures'][0]['events'][0]
    assert first_event['sha256'] == hashlib.sha256(bytes(32)).hexdigest()
    assert (first_event['offset'], first_event['length'], first_event['layer_index'],
            first_event['module_kind'], first_event['token_positions']) == (0, 32, 0, 'attn', [0, 1])


def test_wrong_geometry_is_rejected_before_evidence_file_exists(tmp_path):
    writer = _writer(tmp_path)
    with pytest.raises(ValueError):
        writer.write_logit(1, 'baseline_cold', torch.zeros((1, 5, 8)))
    assert not (tmp_path / 'logits-001.f32').exists()
    assert not (tmp_path / 'manifest.json').exists()


def test_interrupted_file_sync_preserves_partial_and_refuses_later_work(tmp_path, monkeypatch):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    writer = _writer(tmp_path)
    monkeypatch.setattr(module.os, 'fsync', lambda _fd: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        writer.write_logit(1, 'baseline_cold', _logits())
    assert (tmp_path / 'logits-001.f32.partial').exists()
    with pytest.raises(RuntimeError):
        writer.write_logit(2, 'baseline_warm', _logits())
    assert not (tmp_path / 'logits-002.f32').exists()
    assert not (tmp_path / 'manifest.json').exists()


def test_failed_partial_unlink_preserves_both_names_and_stops_writer(tmp_path, monkeypatch):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    writer = _writer(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'unlink', lambda _path: (_ for _ in ()).throw(PermissionError()))
        with pytest.raises(PermissionError):
            writer.write_logit(1, 'baseline_cold', _logits())
    assert (tmp_path / 'logits-001.f32.partial').exists()
    assert (tmp_path / 'logits-001.f32').exists()
    with pytest.raises(RuntimeError):
        writer.write_logit(2, 'baseline_warm', _logits())
    assert not (tmp_path / 'logits-002.f32').exists() and not (tmp_path / 'manifest.json').exists()


def test_failed_manifest_unlink_stops_before_success_binding(tmp_path, monkeypatch):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    writer = _writer(tmp_path)
    _write_all(writer)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'unlink', lambda _path: (_ for _ in ()).throw(PermissionError()))
        with pytest.raises(OSError):
            writer.finish()
    assert (tmp_path / 'manifest.json').exists() and (tmp_path / 'manifest.json.partial').exists()
    assert writer.failed and not writer.finished
    with pytest.raises(RuntimeError):
        writer.finish()


def _candidate_records(directory):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    logits = [item['sha256'] for item in manifest['logits']]
    reference = [event['sha256'] for entry in manifest['captures'] if entry['mode'] == 'reference'
                 for event in entry['events']]
    modes = {}
    for mode in ('direct', 'gradient'):
        events = [event for entry in manifest['captures'] if entry['mode'] == mode for event in entry['events']]
        offset = 4 if mode == 'direct' else 6
        modes[mode] = {'logit_sha256': logits[offset:offset + 2], 'events': [{'metadata': {key: item[key] for key in
            ('sequence', 'forward_index', 'layer_index', 'module_kind', 'module_path',
             'output_path', 'token_positions')}, 'value_sha256': item['sha256']} for item in events]}
    modes['id'] = {'logit_sha256': logits[8:10], 'events': [{'value_sha256': None} for _ in range(18)]}
    result = {'baseline_logit_sha256': logits[:2], 'reference_logit_sha256': logits[2:4],
              'reference_value_sha256': reference, 'modes': modes}
    return logits, result


def test_raw_reader_reconstructs_all_logit_and_capture_bit_comparisons(tmp_path):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    binding = _complete(tmp_path)
    logits, result = _candidate_records(tmp_path)
    report = module.validate_tensor_evidence(tmp_path, binding, logits, result, hidden=4, vocab=7)
    assert report == {'forward_count': 10, 'capture_event_count': 54,
                      'logit_bit_comparisons': 9, 'capture_bit_comparisons': 36,
                      'payload_bytes': 3128}


@pytest.mark.parametrize('kind', ['truncated', 'extra'])
def test_raw_reader_rejects_wrong_file_length(tmp_path, kind):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    binding = _complete(tmp_path)
    logits, result = _candidate_records(tmp_path)
    path = tmp_path / 'logits-003.f32'
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] if kind == 'truncated' else raw + b'\0')
    with pytest.raises(ValueError):
        module.validate_tensor_evidence(tmp_path, binding, logits, result, hidden=4, vocab=7)


def test_raw_reader_rejects_coherently_rehashed_signed_zero_mismatch(tmp_path):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    binding = _complete(tmp_path)
    logits, result = _candidate_records(tmp_path)
    path = tmp_path / 'gradient-1.f32'
    raw = bytearray(path.read_bytes())
    raw[3] = 0x80  # Positive zero becomes negative zero; ordinary float equality would miss it.
    path.write_bytes(raw)
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(item for item in manifest['captures'] if item['mode'] == 'gradient'
                 and item['forward_index'] == 0)
    entry['sha256'] = hashlib.sha256(raw).hexdigest()
    entry['events'][0]['sha256'] = hashlib.sha256(raw[:32]).hexdigest()
    result['modes']['gradient']['events'][0]['value_sha256'] = entry['events'][0]['sha256']
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    manifest_path.write_bytes(manifest_bytes)
    binding['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    with pytest.raises(ValueError, match='bits differ'):
        module.validate_tensor_evidence(tmp_path, binding, logits, result, hidden=4, vocab=7)


def test_raw_reader_rejects_reordered_segments_even_with_matching_new_hashes(tmp_path):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    binding = _complete(tmp_path)
    logits, result = _candidate_records(tmp_path)
    path = tmp_path / 'direct-1.f32'
    raw = path.read_bytes()
    raw = raw[32:64] + raw[:32] + raw[64:]
    path.write_bytes(raw)
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(item for item in manifest['captures'] if item['mode'] == 'direct'
                 and item['forward_index'] == 0)
    entry['sha256'] = hashlib.sha256(raw).hexdigest()
    for index in range(2):
        digest = hashlib.sha256(raw[index * 32:(index + 1) * 32]).hexdigest()
        entry['events'][index]['sha256'] = digest
        result['modes']['direct']['events'][index]['value_sha256'] = digest
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    manifest_path.write_bytes(manifest_bytes)
    binding['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    with pytest.raises(ValueError, match='bits differ'):
        module.validate_tensor_evidence(tmp_path, binding, logits, result, hidden=4, vocab=7)


def test_raw_reader_rejects_coherently_rehashed_nonfinite_logits(tmp_path):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    binding = _complete(tmp_path)
    logits, result = _candidate_records(tmp_path)
    path = tmp_path / 'logits-005.f32'
    raw = bytes.fromhex('0000807f') + path.read_bytes()[4:]
    path.write_bytes(raw)
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    manifest['logits'][4]['sha256'] = digest
    logits[4] = digest
    result['modes']['direct']['logit_sha256'][0] = digest
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    manifest_path.write_bytes(manifest_bytes)
    binding['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    with pytest.raises(ValueError, match='finite'):
        module.validate_tensor_evidence(tmp_path, binding, logits, result, hidden=4, vocab=7)


def test_raw_reader_rejects_boolean_alias_for_candidate_event_index(tmp_path):
    module = importlib.import_module('looking_glass.pretrained_gpu_trace_evidence_v2')
    binding = _complete(tmp_path)
    logits, result = _candidate_records(tmp_path)
    result['modes']['direct']['events'][1]['metadata']['sequence'] = True
    with pytest.raises(ValueError, match='binding'):
        module.validate_tensor_evidence(tmp_path, binding, logits, result, hidden=4, vocab=7)
