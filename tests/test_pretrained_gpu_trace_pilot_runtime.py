"""CPU fakes for phase measurements, input guards and failure custody."""

from types import SimpleNamespace

import pytest
import torch

from looking_glass.pretrained_gpu_trace_pilot_runtime import PilotExecutor, phase_estimate


GIB = 1024 ** 3


def memory():
    return {'free_bytes': 7 * GIB, 'total_bytes': 8 * GIB, 'allocated_bytes': 0,
            'reserved_bytes': 0, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}


def make_executor(tmp_path, *, check=lambda: None):
    calls, roots = [], {}
    executor = PilotExecutor(torch=torch, check=check, synchronize=lambda: calls.append('sync'),
        snapshot=lambda: (calls.append('snapshot') or memory()),
        reset_peaks=lambda: calls.append('reset'), directory=tmp_path, roots=roots,
        device='cpu', vocab_size=3)
    return executor, roots, calls


def test_fixed_phase_estimate_adds_declared_work_and_conservative_margin():
    before, after = memory(), memory()
    after.update(free_bytes=before['free_bytes'] - 100, peak_allocated_bytes=200,
                 peak_reserved_bytes=300)
    result = phase_estimate('direct', before, after)
    assert result['declared_bytes'] == 80 + 1005140 + 131072
    assert result['observed_increment_bytes'] == 300
    assert result['estimated_incremental_bytes'] == result['basis_bytes'] + result['margin_bytes']
    assert result['margin_bytes'] == 32 * 1024 ** 2 + (result['basis_bytes'] + 9) // 10


def test_phase_order_synchronizes_and_resets_before_the_work(tmp_path):
    executor, roots, calls = make_executor(tmp_path)
    executor.begin('load')
    executor.end('load', {'model_loaded': True})
    assert calls[:3] == ['sync', 'snapshot', 'reset']
    assert calls[-2:] == ['sync', 'snapshot']
    assert (tmp_path / 'phase-01-start.json').is_file() and (tmp_path / 'phase-01.json').is_file()
    with pytest.raises(RuntimeError, match='order'):
        executor.begin('gradient')
    assert not roots['pending']


def test_observed_owner_loss_after_forward_keeps_output_and_stops_device_callbacks(tmp_path):
    state = {'held': True}
    def check():
        if not state['held']:
            raise RuntimeError('owner lost')
    executor, roots, calls = make_executor(tmp_path, check=check)
    executor.begin('load')
    executor.end('load', {'model_loaded': True})
    executor.begin('baseline_cold')
    prior = len(calls)
    output = SimpleNamespace(logits=torch.ones((1, 5, 3)))
    def invoke():
        state['held'] = False
        return output
    with pytest.raises(RuntimeError, match='owner lost'):
        executor.forward('baseline_cold', invoke)
    assert roots['output'] is output and roots['pending'] is True
    assert len(calls) == prior and not (tmp_path / 'forward-001.json').exists()


def test_failed_trace_clear_preserves_prior_capture_objects(tmp_path):
    executor, roots, _ = make_executor(tmp_path)
    trace = SimpleNamespace(events=[])
    executor.bind_trace('direct', trace)
    event = object()
    trace.events.append(event)
    trace.events.clear()
    assert roots['quarantined_events'] == [event] and trace.events == []


def test_failed_forward_record_is_not_counted_and_keeps_output_root(tmp_path, monkeypatch):
    from looking_glass import pretrained_gpu_trace_pilot_runtime as module
    executor, roots, _ = make_executor(tmp_path)
    executor.begin('load')
    executor.end('load', {'model_loaded': True})
    executor.begin('baseline_cold')
    output = SimpleNamespace(logits=torch.ones((1, 5, 3)))
    original = module.write_json_new
    def write(path, value):
        if path.name == 'forward-001.json':
            raise OSError('fabricated write failure')
        return original(path, value)
    monkeypatch.setattr(module, 'write_json_new', write)
    with pytest.raises(OSError, match='write failure'):
        executor.forward('baseline_cold', lambda: output)
    assert executor.forward_count == 0 and roots['output'] is output
    assert not (tmp_path / 'forward-001.json').exists()


def test_input_mutation_is_rejected_before_next_phase(tmp_path):
    executor, roots, calls = make_executor(tmp_path)
    executor.begin('load')
    executor.end('load', {'model_loaded': True})
    ids = torch.tensor([[464, 6766, 318, 4171, 13]], dtype=torch.int64)
    mask = torch.ones_like(ids)
    executor.verify_inputs(ids, mask)
    ids[0, 0] += 1
    resets = calls.count('reset')
    with pytest.raises(ValueError, match='token IDs'):
        executor.begin('baseline_cold')
    assert calls.count('reset') == resets and roots['pending'] is True


@pytest.mark.parametrize('change', [
    {'free_bytes': 9 * GIB}, {'allocated_bytes': 1},
    {'allocated_bytes': 1, 'reserved_bytes': 1, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 1},
    {'reserved_bytes': 1, 'peak_reserved_bytes': 0},
])
def test_inconsistent_observation_is_rejected_before_estimation(change):
    before, after = memory(), memory()
    after.update(change)
    with pytest.raises(ValueError, match='inconsistent'):
        phase_estimate('direct', before, after)


def test_real_tiny_model_composes_all_seven_phases_and_ten_forward_records(tmp_path):
    from copy import deepcopy
    import json
    import gc
    from transformers import GPT2Config, GPT2LMHeadModel
    from looking_glass.bert_gpu_evidence import write_json_new
    from looking_glass.pretrained_gpu_trace_pilot import _validate_saved, _weakrefs_for_disposal
    from looking_glass.pretrained_gpu_trace_pilot_core import run_fixed_trace
    from looking_glass.pretrained_gpu_trace_evidence_v2 import TensorEvidence
    from looking_glass.pretrained import ASSETS, MODEL_ID, REVISION
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=6767, n_positions=16, n_embd=8, n_layer=6, n_head=2,
                        use_cache=False, _attn_implementation='eager')
    model = GPT2LMHeadModel(config).eval()
    ids = torch.tensor([[464, 6766, 318, 4171, 13]], dtype=torch.int64)
    mask = torch.ones_like(ids)
    executor, roots, _ = make_executor(tmp_path)
    executor.vocab_size = model.config.vocab_size
    executor.begin('load')
    executor.end('load', {'parameter_count': 81912576, 'weights_sha256': ASSETS['model.safetensors'][1],
                          'before_transfer': memory()})
    executor.verify_inputs(ids, mask)
    (tmp_path / 'raw').mkdir()
    executor.bind_evidence(TensorEvidence(tmp_path / 'raw', torch=torch, hidden=8, vocab=6767))
    result = run_fixed_trace(model, ids, mask, executor, study_id='cpu-composition')
    evidence = executor.finish_evidence()
    assert executor.index == 7 and executor.forward_count == 10
    assert len(list(tmp_path.glob('phase-??.json'))) == 7
    assert len(list(tmp_path.glob('forward-???.json'))) == 10
    assert len(roots['reference']) == 18 and len(roots['traces']) == 3
    assert result['direct_gradient_equal'] is True and not roots['pending']
    write_json_new(tmp_path / 'input-setup.json', {'schema': 'pretrained-gpu-trace-input-v1',
        'input_ids': [464, 6766, 318, 4171, 13], 'attention_mask': [1] * 5,
        'shape': [1, 5], 'dtype': 'int64', 'memory': memory()})
    measurement = {'forward_count': 10, 'phases': executor.phases, 'result': result, 'tensor_evidence': evidence,
        'runtime': {'device_total_bytes': 8 * GIB, 'nvidia_total_bytes': 8 * GIB,
                    'allocator_cap_bytes': 1536 * 1024 ** 2, 'allocator_fraction': 0.1875,
                    'reserve_bytes': (8 * GIB * 15 + 99) // 100, 'allocator_backend': 'native',
                    'device': 'cpu', 'device_uuid': 'fake', 'device_name': 'fake',
                    'compute_capability': [7, 5], 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
                    'transformers': '4.57.3', 'python': '3.11.0', 'seed': 0, 'dtype': 'float32',
                    'default_dtype': 'torch.float32', 'attention': 'eager', 'use_cache': False,
                    'inference_mode': True, 'autocast_enabled': False, 'matmul_tf32': False,
                    'cudnn_tf32': False, 'threads': 1, 'interop_threads': 1,
                    'deterministic_algorithms': False, 'cudnn_benchmark': False, 'cudnn_deterministic': False,
                    'resident_commitment_bytes': 512 * 1024 ** 2, 'job_commitment_bytes': GIB,
                    'gpu_trace_claim_bytes': 131072, 'host_trace_claim_bytes': 4 * 1024 ** 2},
        'model': {'model_id': MODEL_ID, 'model_revision': REVISION, 'tokenizer_id': MODEL_ID,
                  'tokenizer_revision': REVISION, 'parameter_count': 81912576,
                  'weights_sha256': ASSETS['model.safetensors'][1],
                  'asset_sha256': {name: digest for name, (_, digest) in ASSETS.items()},
                  'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
                  'dtype': 'float32', 'attention_implementation': 'eager', 'use_cache': False},
        'cleanup': memory(), 'model_input_trace_weakrefs_dead': True, 'active_memory_leases': 0,
        'runtime_binding_rechecked': True}
    assert _validate_saved(tmp_path, measurement, hidden=8, vocab=6767, device='cpu',
                           study_id='cpu-composition') == 10
    raw_path = tmp_path / 'raw' / 'gradient-2.f32'
    original_raw = raw_path.read_bytes()
    raw_path.write_bytes(original_raw[:-1])
    with pytest.raises(ValueError, match='length'):
        _validate_saved(tmp_path, measurement, hidden=8, vocab=6767, device='cpu',
                        study_id='cpu-composition')
    raw_path.write_bytes(original_raw)
    for field, value in (('seed', 1), ('dtype', 'float16'), ('use_cache', True),
                         ('matmul_tf32', True), ('host_trace_claim_bytes', 0)):
        bad = deepcopy(measurement)
        bad['runtime'][field] = value
        with pytest.raises(ValueError, match='runtime'):
            _validate_saved(tmp_path, bad, hidden=8, vocab=6767, device='cpu', study_id='cpu-composition')
    bad = deepcopy(measurement)
    bad['model']['weights_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='model'):
        _validate_saved(tmp_path, bad, hidden=8, vocab=6767, device='cpu', study_id='cpu-composition')
    input_path = tmp_path / 'input-setup.json'
    original_input = input_path.read_bytes()
    bad_input = json.loads(original_input)
    bad_input['attention_mask'][0] = True
    input_path.write_text(json.dumps(bad_input), encoding='utf-8')
    with pytest.raises(ValueError, match='input'):
        _validate_saved(tmp_path, measurement, hidden=8, vocab=6767, device='cpu', study_id='cpu-composition')
    input_path.write_bytes(original_input)
    phase_path = tmp_path / 'phase-05.json'
    original_phase = phase_path.read_bytes()
    bad = deepcopy(measurement)
    bad['result']['modes']['direct']['events'][1]['metadata']['sequence'] = True
    bad['phases'][4]['proof']['events'][1]['metadata']['sequence'] = True
    phase_path.write_text(json.dumps(bad['phases'][4]), encoding='utf-8')
    with pytest.raises(ValueError, match='event'):
        _validate_saved(tmp_path, bad, hidden=8, vocab=6767, device='cpu', study_id='cpu-composition')
    phase_path.write_bytes(original_phase)
    roots['model'] = model
    refs = _weakrefs_for_disposal(roots)
    roots['reference'] = roots['reference_lease'] = roots['reference_handles'] = None
    roots['traces'].clear()
    roots['quarantined_events'].clear()
    roots['cpu_logits'] = roots['input_ids'] = roots['attention_mask'] = None
    roots['model'] = None
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del model, ids, mask
        assert all(ref() is None for ref in refs)
    finally:
        if was_enabled:
            gc.enable()


def test_partial_reference_evidence_failure_stops_before_the_next_forward(tmp_path, monkeypatch):
    from transformers import GPT2Config, GPT2LMHeadModel
    from looking_glass import pretrained_gpu_trace_evidence_v2 as evidence_module
    from looking_glass.pretrained_gpu_trace_pilot_core import run_fixed_trace
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6767, n_positions=16, n_embd=8, n_layer=6,
                                      n_head=2, use_cache=False, _attn_implementation='eager')).eval()
    ids = torch.tensor([[464, 6766, 318, 4171, 13]], dtype=torch.int64)
    mask = torch.ones_like(ids)
    executor, roots, _ = make_executor(tmp_path)
    executor.vocab_size = 6767
    executor.begin('load')
    executor.end('load', {'model_loaded': True})
    executor.verify_inputs(ids, mask)
    raw = tmp_path / 'raw'
    raw.mkdir()
    writer = evidence_module.TensorEvidence(raw, torch=torch, hidden=8, vocab=6767)
    executor.bind_evidence(writer)
    original = evidence_module._publish_new
    def fail_after_partial(path, data):
        if path.name == 'reference-1.f32':
            path.with_name(path.name + '.partial').write_bytes(data[:16])
            raise OSError('fabricated partial write')
        return original(path, data)
    monkeypatch.setattr(evidence_module, '_publish_new', fail_after_partial)
    with pytest.raises(OSError, match='partial write'):
        run_fixed_trace(model, ids, mask, executor, study_id='partial-cpu')
    assert executor.forward_count == 3 and executor.current == 'reference' and writer.failed
    assert len(roots['reference']) == 9
    assert (raw / 'reference-1.f32.partial').read_bytes()
    assert not (raw / 'manifest.json').exists() and not (tmp_path / 'forward-004.json').exists()
