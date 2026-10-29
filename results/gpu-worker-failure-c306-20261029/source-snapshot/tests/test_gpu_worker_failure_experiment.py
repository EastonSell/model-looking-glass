"""CPU-only checks for the separate synthetic worker failure exercise."""
import gc
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget
from looking_glass.worker_failure_experiment import (
    TraceCustody, WorkerCustody, run_trace_probe, run_worker_probe,
)


def tiny_model():
    config = GPT2Config(vocab_size=64, n_positions=8, n_ctx=8, n_embd=16,
                        n_layer=1, n_head=2, use_cache=False)
    config._attn_implementation = 'eager'
    torch.random.default_generator.manual_seed(17)
    return GPT2LMHeadModel(config).eval()


def budget():
    memory = CudaMemorySnapshot(3 * 1024**3, 4 * 1024**3, 0, 0)
    return MemoryBudget(lambda: memory, max_gpu_trace_bytes=64, max_host_trace_bytes=64)


def test_trace_success_and_failure_keep_capture_until_completion_then_release():
    model = tiny_model()
    ledger = budget()
    resident = ledger.acquire(32 * 1024**2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    sync_calls = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for injected in (False, True, False):
            custody = TraceCustody()
            report = run_trace_probe(model, budget=ledger, device='cpu', guard=lambda: None,
                                     synchronize=lambda: sync_calls.append('sync'),
                                     custody=custody, inject_cancellation=injected)
            assert report['outcome'] == ('synthetic_trace_cancellation' if injected else 'success')
            assert report['event_bytes'] == 64 and report['capture_retained_until_sync'] is True
            assert report['capture_released'] is True and report['trace_claim_released'] is True
            assert custody.capture_ref() is None and custody.lease.released is True
            assert custody.input_ref() is None and custody.output_ref() is None and custody.logits_ref() is None
            assert report['trace_resources_retained_until_sync'] is True
            assert report['trace_resources_released'] is True
            assert ledger.active_leases() == (resident,)
    finally:
        if was_enabled:
            gc.enable()
    assert len(sync_calls) == 6


def test_trace_failed_synchronization_retains_capture_and_claim():
    model = tiny_model()
    ledger = budget()
    resident = ledger.acquire(32 * 1024**2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    custody = TraceCustody()
    def fail_sync():
        raise RuntimeError('simulated synchronization failure')
    with pytest.raises(RuntimeError, match='synchronization failure'):
        run_trace_probe(model, budget=ledger, device='cpu', guard=lambda: None,
                        synchronize=fail_sync, custody=custody, inject_cancellation=True)
    assert custody.events and custody.capture_ref() is not None
    assert custody.input_ref() is not None and custody.output_ref() is not None and custody.logits_ref() is not None
    assert custody.lease.trace_bytes('host') == 64 and not custody.lease.released
    assert set(ledger.active_leases()) == {resident, custody.lease}


def test_interrupted_report_stops_before_request_and_roots_resources():
    model = tiny_model()
    ledger = budget()
    resident = ledger.acquire(32 * 1024**2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    custody = WorkerCustody()
    def fail_record(stage, value):
        raise OSError('simulated evidence write failure')
    with pytest.raises(OSError, match='evidence write failure'):
        run_worker_probe(model, budget=ledger, resident_lease=resident,
                         device='cpu', guard=lambda: None,
                         checkpoint=lambda stage: {'stage': stage},
                         record=fail_record, custody=custody)
    assert custody.budget is ledger and custody.resident_lease is resident
    assert isinstance(custody.error, OSError)
    assert custody.model.invocations == 0
    assert set(ledger.active_leases()) == {resident, custody.workspace}
    assert custody.broker.close(timeout=5)


def test_real_worker_cancel_then_recovery_then_simulated_oom_fails_closed():
    model = tiny_model()
    ledger = budget()
    resident = ledger.acquire(32 * 1024**2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    checkpoints = []
    def checkpoint(stage):
        checkpoints.append(stage)
        return {'stage': stage}
    records = []
    custody = WorkerCustody()
    report = run_worker_probe(model, budget=ledger, resident_lease=resident,
                              device='cpu', guard=lambda: None, checkpoint=checkpoint,
                              record=lambda stage, value: records.append((stage, value)),
                              custody=custody)
    assert report['baseline'] == 'succeeded'
    assert report['cancellation']['accepted'] is True
    assert report['cancellation']['outcome'] == 'cancelled'
    assert report['post_cancel'] == 'succeeded'
    assert report['simulated_oom']['injection_reached'] is True
    assert report['simulated_oom']['private_forward_error_type'] == 'OutOfMemoryError'
    assert report['simulated_oom']['worker_unsafe'] is True
    assert report['simulated_oom']['subsequent_work_refused'] is True
    assert report['simulated_oom']['active_claim_count'] == 2
    assert report['simulated_oom']['no_explicit_checkpoint_after_unsafe'] is True
    assert checkpoints == ['before_worker', 'after_baseline', 'after_cancellation',
                           'after_post_cancel', 'before_simulated_oom']
    assert [stage for stage, _ in records] == ['before_worker', 'baseline', 'after_baseline',
        'cancellation', 'after_cancellation', 'post_cancel', 'after_post_cancel',
        'before_simulated_oom', 'simulated_oom']
    assert custody.budget is ledger and custody.resident_lease is resident
    assert not custody.broker._thread.is_alive()
