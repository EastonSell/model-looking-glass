"""CPU-only checks for the FLAN-T5 owned-device observation boundary."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import weakref

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration


def subject():
    try:
        return importlib.import_module('looking_glass.flan_t5_device_observation')
    except ModuleNotFoundError as error:
        if error.name != 'looking_glass.flan_t5_device_observation':
            raise
        pytest.fail('the device-aware FLAN-T5 observer is unavailable')


def tiny_t5():
    config = T5Config(vocab_size=40, d_model=16, d_kv=4, d_ff=32,
                      num_layers=1, num_decoder_layers=1, num_heads=4,
                      dropout_rate=0.0, decoder_start_token_id=0,
                      pad_token_id=0, eos_token_id=1, use_cache=False)
    previous = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(23)
        return T5ForConditionalGeneration(config).eval()
    finally:
        torch.random.set_rng_state(previous)


def pinned_meta_t5():
    config = T5Config(vocab_size=32128, d_model=512, d_kv=64, d_ff=1024,
                      num_layers=8, num_decoder_layers=8, num_heads=6,
                      feed_forward_proj='gated-gelu', tie_word_embeddings=False,
                      dropout_rate=0.1, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    with torch.device('meta'):
        return T5ForConditionalGeneration(config).eval()


def raw(tensor):
    return tensor.detach().view(torch.int32).contiguous().numpy().tobytes()


def test_cpu_prototype_captures_t5_patch_and_host_owned_full_outputs():
    api = subject()
    model = tiny_t5()
    source = {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]}
    target = {'input_ids': [10, 1]}
    donor = torch.arange(16, dtype=torch.float32) / 4
    stages = []
    custody = api.ObservationCustody()
    result = api.observe_cpu(model, source, target, action='patch', token_index=1,
                             value=donor, custody=custody, check_owner=lambda: None,
                             synchronize=stages.append)

    def patch_hook(module, inputs, output):
        modified = output.clone()
        modified[0, 1] = donor
        return modified

    handle = model.encoder.final_layer_norm.register_forward_hook(patch_hook)
    try:
        with torch.inference_mode():
            reference = model(input_ids=torch.tensor([[8, 9, 1]]), attention_mask=torch.ones((1, 3), dtype=torch.long),
                              decoder_input_ids=torch.tensor([[0, 10]]), decoder_attention_mask=torch.ones((1, 2), dtype=torch.long),
                              use_cache=False, return_dict=True)
    finally:
        handle.remove()
    assert raw(result.logits) == raw(reference.logits)
    assert raw(result.state) == raw(reference.encoder_last_hidden_state)
    assert raw(result.before[0, 1]) != raw(donor)
    assert raw(result.after[0, 1]) == raw(donor) == raw(result.state[0, 1])
    assert raw(result.before[0, 0]) == raw(result.after[0, 0])
    assert raw(result.before[0, 2]) == raw(result.after[0, 2])
    assert result.source_ids == (8, 9, 1) and result.decoder_ids == (0, 10)
    assert all(t.device.type == 'cpu' and t.is_contiguous() for t in
               (result.logits, result.state, result.before, result.after))
    assert [step['device'] for step in result.metadata['steps']] == ['cpu'] * 3
    assert stages == ['output_live', 'captures_live', 'device_refs_disposed']
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    donor.add_(20)
    assert raw(result.after[0, 1]) != raw(donor)


def test_unhooked_clean_forward_retains_full_logits_and_eos_inputs():
    api = subject()
    model = tiny_t5()
    source = {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]}
    target = {'input_ids': [10, 1]}
    result = api.observe_cpu(model, source, target, action='clean', custody=api.ObservationCustody(),
                             check_owner=lambda: None, synchronize=lambda stage: None)
    with torch.inference_mode():
        expected = model(input_ids=torch.tensor([[8, 9, 1]]), attention_mask=torch.ones((1, 3), dtype=torch.long),
                         decoder_input_ids=torch.tensor([[0, 10]]), decoder_attention_mask=torch.ones((1, 2), dtype=torch.long),
                         use_cache=False, return_dict=True)
    assert result.before is result.after is result.metadata is None
    assert raw(result.logits) == raw(expected.logits) and raw(result.state) == raw(expected.encoder_last_hidden_state)
    assert result.source_ids == (8, 9, 1) and result.source_mask == (1, 1, 1)
    assert result.decoder_ids == (0, 10) and result.decoder_mask == (1, 1)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_inputs_remain_cpu_when_torch_default_factory_device_is_meta():
    api = subject()
    model = tiny_t5()
    with torch.device('meta'):
        result = api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                                 {'input_ids': [10, 1]}, action='clean', custody=api.ObservationCustody(),
                                 check_owner=lambda: None, synchronize=lambda stage: None)
    assert result.logits.device.type == result.state.device.type == 'cpu'


def test_corrupt_plan_coordinate_fails_before_observation_is_returned(monkeypatch):
    api = subject()
    from looking_glass.intervention_plan import ActivationPlan

    original = ActivationPlan.metadata.fget

    def incorrect_metadata(plan):
        metadata = original(plan)
        metadata['steps'][1]['token_index'] = 2
        return metadata

    monkeypatch.setattr(ActivationPlan, 'metadata', property(incorrect_metadata))
    model = tiny_t5()
    with pytest.raises(ValueError, match='metadata'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='patch', token_index=1,
                        value=torch.ones(16), custody=api.ObservationCustody(),
                        check_owner=lambda: None, synchronize=lambda stage: None)


def test_success_releases_forward_roots_before_final_synchronization():
    api = subject()
    model = tiny_t5()
    custody = api.ObservationCustody()
    references = []

    def synchronize(stage):
        if stage == 'output_live':
            references.extend(weakref.ref(value) for value in (
                custody.input_ids, custody.decoder_mask, custody.source,
                custody.plan_sources[0], custody.plan_captures['before'], custody.output.logits))
            assert all(ref() is not None for ref in references)
        elif stage == 'device_refs_disposed':
            assert all(ref() is None for ref in references)

    result = api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                             {'input_ids': [10, 1]}, action='patch', token_index=1,
                             value=torch.ones(16), custody=custody,
                             check_owner=lambda: None, synchronize=synchronize)
    assert result.logits.device.type == 'cpu'


def test_failed_live_synchronization_keeps_output_and_plan_roots():
    api = subject()
    model = tiny_t5()
    custody = api.ObservationCustody()
    def fail(stage):
        assert stage == 'output_live'
        raise RuntimeError('injected failed drain')
    with pytest.raises(RuntimeError, match='failed drain'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='patch', token_index=1,
                        value=torch.ones(16), custody=custody,
                        check_owner=lambda: None, synchronize=fail)
    assert custody.output is not None and custody.input_ids is not None
    assert len(custody.plan_sources) == 1
    assert set(custody.plan_captures) == {'before', 'after'}
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_final_synchronization_failure_retains_complete_host_evidence():
    api = subject()
    model = tiny_t5()
    custody = api.ObservationCustody()
    def synchronize(stage):
        if stage == 'device_refs_disposed':
            raise RuntimeError('injected final drain failure')
    with pytest.raises(RuntimeError, match='final drain'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='patch', token_index=1,
                        value=torch.ones(16), custody=custody, check_owner=lambda: None,
                        synchronize=synchronize)
    result = custody.host_result
    assert result is not None and result.logits.device.type == 'cpu'
    assert tuple(result.logits.shape) == (1, 2, 40)
    assert raw(result.after[0, 1]) == raw(torch.ones(16))
    assert result.metadata['steps'][1]['token_index'] == 1


def test_failed_capture_synchronization_keeps_already_copied_host_tensors():
    api = subject()
    model = tiny_t5()
    custody = api.ObservationCustody()
    def synchronize(stage):
        if stage == 'captures_live':
            raise RuntimeError('injected capture drain failure')
    with pytest.raises(RuntimeError, match='capture drain'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='patch', token_index=1,
                        value=torch.ones(16), custody=custody, check_owner=lambda: None,
                        synchronize=synchronize)
    assert custody.host_result is None
    assert tuple(custody.host_logits.shape) == (1, 2, 40)
    assert raw(custody.host_after[0, 1]) == raw(torch.ones(16))
    assert custody.host_state.device.type == custody.host_before.device.type == 'cpu'


@pytest.mark.parametrize('action', ['capture', 'null', 'zero'])
def test_unmodified_and_zero_controls_use_full_same_forward_witnesses(action):
    api = subject()
    model = tiny_t5()
    options = {} if action == 'capture' else {'token_index': 1}
    result = api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                             {'input_ids': [10, 1]}, action=action, custody=api.ObservationCustody(),
                             check_owner=lambda: None, synchronize=lambda stage: None, **options)
    if action == 'capture':
        assert result.after is None and raw(result.before) == raw(result.state)
    elif action == 'null':
        assert raw(result.before) == raw(result.after) == raw(result.state)
    else:
        assert raw(result.after[0, 1]) == bytes(16 * 4)
        assert raw(result.before[0, 0]) == raw(result.after[0, 0])
        assert raw(result.before[0, 2]) == raw(result.after[0, 2])


def test_noncontiguous_patch_preserves_raw_signed_zero_and_subnormal_bits():
    api = subject()
    model = tiny_t5()
    values = [0, -2147483648, 1, -2147483647] * 4
    interleaved = torch.zeros(32, dtype=torch.int32)
    interleaved[::2] = torch.tensor(values, dtype=torch.int32)
    donor = interleaved[::2].view(torch.float32)
    assert not donor.is_contiguous()
    result = api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                             {'input_ids': [10, 1]}, action='patch', token_index=1, value=donor,
                             custody=api.ObservationCustody(), check_owner=lambda: None,
                             synchronize=lambda stage: None)
    assert raw(result.after[0, 1]) == raw(donor)


def test_model_input_mutation_fails_after_forward_and_keeps_output_rooted():
    api = subject()
    model = tiny_t5()
    original = model.forward
    def mutate(*args, **kwargs):
        kwargs['input_ids'][0, 0] = 3
        return original(*args, **kwargs)
    model.forward = mutate
    custody = api.ObservationCustody()
    with pytest.raises(ValueError, match='input or mask changed'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='capture', custody=custody,
                        check_owner=lambda: None, synchronize=lambda stage: None)
    assert custody.output is not None and custody.input_ids is not None


def test_noncontiguous_return_is_rejected_and_rooted_before_host_copy(monkeypatch):
    api = subject()
    model, custody = tiny_t5(), api.ObservationCustody()
    original = model.forward
    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits = output.logits.transpose(1, 2).contiguous().transpose(1, 2)
        assert tuple(output.logits.shape) == (1, 2, 40) and not output.logits.is_contiguous()
        return output
    monkeypatch.setattr(model, 'forward', forward)
    with pytest.raises(ValueError, match='bounded float32'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='clean', custody=custody,
                        check_owner=lambda: None, synchronize=lambda stage: None)
    assert custody.output is not None and custody.host_logits_bits is None


def test_cuda_entry_rejects_unsupervised_use_before_touching_device(monkeypatch):
    api = subject()
    def forbidden(*args, **kwargs):
        pytest.fail('CUDA was touched before supervised ownership')
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    with pytest.raises((ValueError, RuntimeError), match='supervised'):
        api.observe_cuda_owned(None, {}, {}, action='capture', custody=api.ObservationCustody(),
                               session=None, device_lease=None, memory_lease=None,
                               record_stage=lambda stage: None)


def test_private_observer_rejects_cuda_without_owned_entry(monkeypatch):
    api = subject()
    def forbidden(*args, **kwargs):
        pytest.fail('CUDA was touched without the owned entry')
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    with pytest.raises(RuntimeError, match='supervised'):
        api._observe(None, {}, {}, action='capture', token_index=None, value=None,
                     custody=api.ObservationCustody(), check_owner=lambda: None,
                     synchronize=lambda stage: None, device='cuda:0')


def test_owner_callback_false_is_rejected_before_model_forward(monkeypatch):
    api = subject()
    model = tiny_t5()
    def forbidden(*args, **kwargs):
        pytest.fail('forward ran after ownership was rejected')
    monkeypatch.setattr(model, 'forward', forbidden)
    with pytest.raises(RuntimeError, match='owner'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='capture', custody=api.ObservationCustody(),
                        check_owner=lambda: False, synchronize=lambda stage: None)


def test_cuda_owner_check_requires_active_device_and_host_trace_claims(monkeypatch):
    api = subject()
    from looking_glass.device_lease import DeviceLease
    from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget
    from looking_glass.supervisor import OwnerSession

    gib = 1024 ** 3
    session, device_lease = object.__new__(OwnerSession), object.__new__(DeviceLease)
    session._identity = object()
    session._runtime_verified = True
    monkeypatch.setattr(OwnerSession, 'check', lambda self, lease: None)
    budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                          max_gpu_trace_bytes=api.GPU_TRACE_CLAIM,
                          max_host_trace_bytes=api.HOST_TRACE_CLAIM)
    memory = budget.acquire(2 * gib, max_gpu_trace_bytes=api.GPU_TRACE_CLAIM,
                            max_host_trace_bytes=api.HOST_TRACE_CLAIM)
    with pytest.raises(RuntimeError, match='claims'):
        api._cuda_owner_check(session, device_lease, memory)
    memory.reserve_trace('gpu', api.GPU_TRACE_CLAIM)
    memory.reserve_trace('host', api.HOST_TRACE_CLAIM)
    api._cuda_owner_check(session, device_lease, memory)
    memory.release_trace('host', api.HOST_TRACE_CLAIM)
    with pytest.raises(RuntimeError, match='claims'):
        api._cuda_owner_check(session, device_lease, memory)
    memory.release_trace('gpu', api.GPU_TRACE_CLAIM)
    memory.release()


def test_cuda_endpoint_records_synchronized_allocator_and_device_values_before_validation():
    api = subject()
    events = []
    class FakeCuda:
        def synchronize(self, index): events.append(('sync', index))
        def mem_get_info(self, index): return (5_000, 8_000)
        def memory_allocated(self, index): return 100
        def memory_reserved(self, index): return 200
        def max_memory_allocated(self, index): return 150
        def max_memory_reserved(self, index): return 250
    cuda = FakeCuda()
    observations = []
    api._record_cuda_stage(cuda, lambda: None, 'output_live',
                           lambda stage, value: observations.append((stage, value)))
    assert events == [('sync', 0)]
    assert observations == [('output_live', {'device_free_bytes': 5000, 'device_total_bytes': 8000,
        'allocated_bytes': 100, 'reserved_bytes': 200, 'peak_allocated_bytes': 150,
        'peak_reserved_bytes': 250})]
    cuda.memory_reserved = lambda index: 9_000
    with pytest.raises(ValueError, match='memory'):
        api._record_cuda_stage(cuda, lambda: None, 'bad',
                               lambda stage, value: observations.append((stage, value)))
    assert observations[-1][0] == 'bad' and observations[-1][1]['reserved_bytes'] == 9_000


def test_recorder_cannot_change_the_memory_values_used_for_validation():
    api = subject()
    class FakeCuda:
        def synchronize(self, index): pass
        def mem_get_info(self, index): return (5_000, 8_000)
        def memory_allocated(self, index): return 100
        def memory_reserved(self, index): return 9_000
        def max_memory_allocated(self, index): return 150
        def max_memory_reserved(self, index): return 250
    def mutate(stage, value):
        value['reserved_bytes'] = 200
    with pytest.raises(ValueError, match='memory'):
        api._record_cuda_stage(FakeCuda(), lambda: None, 'bad', mutate)


@pytest.mark.parametrize('old,new', [(0, False), (100, 100.0)])
def test_recorder_cannot_change_counter_types_without_changing_values(old, new):
    api = subject()
    class FakeCuda:
        def synchronize(self, index): pass
        def mem_get_info(self, index): return (5_000, 8_000)
        def memory_allocated(self, index): return old
        def memory_reserved(self, index): return 200
        def max_memory_allocated(self, index): return 150
        def max_memory_reserved(self, index): return 250
    def mutate(stage, value):
        value['allocated_bytes'] = new
    with pytest.raises(ValueError, match='memory'):
        api._record_cuda_stage(FakeCuda(), lambda: None, 'bad', mutate)


def test_cuda_numeric_state_rejects_backend_drift(monkeypatch):
    api = subject()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 1)
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: True)
    monkeypatch.setattr(torch, 'is_deterministic_algorithms_warn_only_enabled', lambda: False)
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', False)
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', True)
    def check_in_own_thread():
        assert torch.set_flush_denormal(False)
        with torch.device('meta'):
            api._cuda_numeric_state()
        monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', True)
        with pytest.raises(RuntimeError, match='numerical'):
            api._cuda_numeric_state()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(check_in_own_thread).result(timeout=10)


def test_owner_revocation_after_forward_preserves_return_and_skips_sync(monkeypatch):
    api = subject()
    model = tiny_t5()
    original, live, stages = model.forward, {'value': True}, []
    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        live['value'] = False
        return result
    def check():
        if not live['value']:
            raise RuntimeError('owner revoked')
    monkeypatch.setattr(model, 'forward', revoke)
    custody = api.ObservationCustody()
    with pytest.raises(RuntimeError, match='owner revoked'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='patch', token_index=1,
                        value=torch.ones(16), custody=custody, check_owner=check,
                        synchronize=stages.append)
    assert custody.output is not None and len(custody.plan_sources) == 1
    assert set(custody.plan_captures) == {'before', 'after'} and stages == []


def test_owner_revocation_after_cpu_bit_copy_keeps_that_copy_rooted(monkeypatch):
    api = subject()
    model, custody = tiny_t5(), api.ObservationCustody()
    original_to, copied, revoked = torch.Tensor.to, [], {'value': False}
    def to(self, *args, **kwargs):
        result = original_to(self, *args, **kwargs)
        if self.dtype == torch.int32 and tuple(self.shape) == (1, 2, 40) and args and args[0] == 'cpu':
            copied.append(weakref.ref(result))
            revoked['value'] = True
        return result
    def check():
        if revoked['value']:
            raise RuntimeError('owner revoked after host copy')
    monkeypatch.setattr(torch.Tensor, 'to', to)
    with pytest.raises(RuntimeError, match='after host copy'):
        api.observe_cpu(model, {'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]},
                        {'input_ids': [10, 1]}, action='clean', custody=custody,
                        check_owner=check, synchronize=lambda stage: None)
    assert len(copied) == 1 and copied[0]() is not None
    assert custody.host_logits_bits is copied[0]()
    assert custody.host_logits_bits.device.type == 'cpu' and custody.host_logits_bits.dtype == torch.int32


@pytest.mark.parametrize('source,target', [
    ({'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]}, {'input_ids': [1]}),
    ({'input_ids': [8, 0, 1], 'attention_mask': [1, 1, 1]}, {'input_ids': [10, 1]}),
    ({'input_ids': [8, 9, 1], 'attention_mask': [True, 1, 1]}, {'input_ids': [10, 1]}),
    ({'input_ids': [8, 9, 1], 'attention_mask': [1, 1, 1]}, {'input_ids': [True, 1]}),
])
def test_invalid_fixed_ids_are_rejected_before_a_forward(source, target, monkeypatch):
    api = subject()
    model = tiny_t5()
    def forbidden(*args, **kwargs):
        pytest.fail('invalid model inputs reached the forward')
    monkeypatch.setattr(model, 'forward', forbidden)
    with pytest.raises(ValueError, match='contract'):
        api.observe_cpu(model, source, target, action='clean', custody=api.ObservationCustody(),
                        check_owner=lambda: None, synchronize=lambda stage: None)


def test_one_observation_claim_cannot_be_reused_concurrently_or_sequentially():
    api = subject()
    from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget
    gib = 1024 ** 3
    budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                          max_gpu_trace_bytes=api.GPU_TRACE_CLAIM,
                          max_host_trace_bytes=api.HOST_TRACE_CLAIM)
    memory = budget.acquire(api.GPU_TRACE_CLAIM, max_gpu_trace_bytes=api.GPU_TRACE_CLAIM,
                            max_host_trace_bytes=api.HOST_TRACE_CLAIM)
    memory.reserve_trace('gpu', api.GPU_TRACE_CLAIM)
    memory.reserve_trace('host', api.HOST_TRACE_CLAIM)
    barrier = Barrier(2)
    def attempt():
        barrier.wait(timeout=5)
        try:
            api._consume_observation_claim(memory, api.ObservationCustody())
            return 'accepted'
        except RuntimeError:
            return 'rejected'
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ['accepted', 'rejected']
    with pytest.raises(RuntimeError, match='already'):
        api._consume_observation_claim(memory, api.ObservationCustody())
    memory.release_trace('gpu', api.GPU_TRACE_CLAIM)
    memory.release_trace('host', api.HOST_TRACE_CLAIM)
    memory.release()


def test_cuda_entry_consumes_its_claim_even_if_forward_is_skipped(monkeypatch):
    api = subject()
    from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget
    gib = 1024 ** 3
    budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                          max_gpu_trace_bytes=api.GPU_TRACE_CLAIM,
                          max_host_trace_bytes=api.HOST_TRACE_CLAIM)
    memory = budget.acquire(api.GPU_TRACE_CLAIM, max_gpu_trace_bytes=api.GPU_TRACE_CLAIM,
                            max_host_trace_bytes=api.HOST_TRACE_CLAIM)
    memory.reserve_trace('gpu', api.GPU_TRACE_CLAIM)
    memory.reserve_trace('host', api.HOST_TRACE_CLAIM)
    model = pinned_meta_t5()
    monkeypatch.setattr(api, '_cuda_owner_check', lambda *args: None)
    monkeypatch.setattr(api, '_cuda_numeric_state', lambda: None)
    monkeypatch.setattr(api, '_cuda_numerics', lambda: {})
    monkeypatch.setattr(api, '_observe', lambda *args, **kwargs: 'synthetic observation')
    options = dict(action='capture', session=object(), device_lease=object(), memory_lease=memory,
                   record_stage=lambda stage, data: None)
    custody = api.ObservationCustody()
    assert api.observe_cuda_owned(model, {}, {}, custody=custody, **options) == 'synthetic observation'
    assert custody.consumed_lease is memory
    with pytest.raises(RuntimeError, match='already'):
        api.observe_cuda_owned(model, {}, {}, custody=api.ObservationCustody(), **options)
    memory.release_trace('gpu', api.GPU_TRACE_CLAIM)
    memory.release_trace('host', api.HOST_TRACE_CLAIM)
    memory.release()


def test_pinned_flan_requires_eight_layers_and_an_untied_output_head():
    api = subject()
    model = pinned_meta_t5()
    api._validate_flan_checkpoint_model(model)
    model.config.num_layers = 6
    with pytest.raises(ValueError, match='pinned'):
        api._validate_flan_checkpoint_model(model)
    model.config.num_layers = 8
    model.config.num_decoder_layers = 6
    with pytest.raises(ValueError, match='pinned'):
        api._validate_flan_checkpoint_model(model)
    model.config.num_decoder_layers = 8
    original = model.encoder.block
    model.encoder.block = torch.nn.ModuleList(list(original)[:-1])
    with pytest.raises(ValueError, match='pinned'):
        api._validate_flan_checkpoint_model(model)
    model.encoder.block = original
    original_decoder = model.decoder.block
    model.decoder.block = torch.nn.ModuleList(list(original_decoder)[:-1])
    with pytest.raises(ValueError, match='pinned'):
        api._validate_flan_checkpoint_model(model)
    model.decoder.block = original_decoder
    model.lm_head.weight = torch.nn.Parameter(model.shared.weight.detach())
    assert model.lm_head.weight is not model.shared.weight
    assert model.lm_head.weight.untyped_storage() is model.shared.weight.untyped_storage()
    with pytest.raises(ValueError, match='pinned'):
        api._validate_flan_checkpoint_model(model)


def test_one_custody_cannot_bind_a_second_observation_lease():
    api = subject()
    from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget
    gib = 1024 ** 3
    budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                          max_gpu_trace_bytes=2 * api.GPU_TRACE_CLAIM,
                          max_host_trace_bytes=2 * api.HOST_TRACE_CLAIM)
    leases = [budget.acquire(api.GPU_TRACE_CLAIM, max_gpu_trace_bytes=api.GPU_TRACE_CLAIM,
                             max_host_trace_bytes=api.HOST_TRACE_CLAIM) for _ in range(2)]
    custody = api.ObservationCustody()
    api._consume_observation_claim(leases[0], custody)
    with pytest.raises(RuntimeError, match='custody'):
        api._consume_observation_claim(leases[1], custody)
    assert api._observation_claim_is_bound(leases[0], custody)
    assert not api._observation_claim_is_bound(leases[1], custody)
    for lease in leases:
        lease.release()
