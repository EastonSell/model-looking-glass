"""Fabricated CPU checks of the GPU pilot's serial-reference/routing core."""
from types import SimpleNamespace

import pytest
import torch

from looking_glass.budgeted_worker import WorkloadProfile
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from looking_glass.owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
from looking_glass.resident_worker import ResidentWorker
from looking_glass.short_cpu_service import make_short_input
from looking_glass.short_service_pilot_core import SerialRoutingPilot, compare_f32


class Model(torch.nn.Module):
    def __init__(self, vocab, *, collapse=False, misroute=False):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.vocab, self.collapse, self.misroute, self.calls = vocab, collapse, misroute, 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        ids = torch.zeros_like(input_ids) if self.collapse else input_ids
        if self.misroute and self.calls > 2:
            ids = torch.where(ids == 1, 2, torch.where(ids == 2, 1, ids))
        return {'logits': ids.unsqueeze(-1).to(torch.float32) +
                          torch.arange(self.vocab, dtype=torch.float32).view(1, 1, -1) / 1000}


def make_pilot(bucket, *, vocab=41, collapse=False, misroute=False):
    model_id = 'a' * 64
    model = Model(vocab, collapse=collapse, misroute=misroute).eval()
    resident = ResidentWorker(model_id, model, device='cpu', max_batch_size=1,
                              max_sequence_length=bucket, max_input_bytes=bucket * 16,
                              max_output_bytes=bucket * vocab * 4)
    budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                          max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    load = budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    workspace = budget.acquire(16 * 1024 * 1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    profile = WorkloadProfile('fake', model_id, 'cpu', (1, bucket),
        (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64')),
        bucket * vocab * 4, 16 * 1024 * 1024)
    authority = OwnerCompletionAuthority()
    worker = LifetimeWorkspaceWorker(resident, budget, load, workspace,
                                     profiles=(profile,), authority=authority)
    rows = ([1] * bucket, [2] * (bucket - 1))
    cases = tuple({name: torch.tensor(value, dtype=torch.int64)
                   for name, value in make_short_input(row, bucket_length=bucket,
                                                       vocab_size=vocab, pad_token_id=vocab - 1).items()}
                  for row in rows)
    stages, saved = [], {}
    pilot = SerialRoutingPilot(worker, authority, bucket=bucket, vocab=vocab, pad=vocab - 1,
        seed=29, cases=cases, on_stage=stages.append,
        on_tensor=lambda name, value: saved.setdefault(name, value.detach().clone()), timeout=10)
    return model, resident, budget, authority, pilot, stages, saved


@pytest.mark.parametrize('bucket', [16, 32])
def test_two_distinct_clients_match_complete_serial_references(bucket):
    model, resident, budget, authority, pilot, stages, saved = make_pilot(bucket)
    result = pilot.run()
    assert result['status'] == 'passed' and result['broker_jobs'] == 4 and model.calls == 4
    assert [item['active_position'] for item in result['comparisons']] == [bucket - 1, bucket - 2]
    assert result['reference_separation']['violations'] > 0
    assert set(saved) == {'reference-alpha', 'reference-beta', 'client-alpha', 'client-beta'}
    assert all(value.shape == (1, bucket, 41) for value in saved.values())
    assert torch.equal(saved['reference-alpha'], saved['client-alpha'])
    assert torch.equal(saved['reference-beta'], saved['client-beta'])
    assert stages == ['before-reference-alpha', 'after-reference-alpha', 'before-reference-beta',
                      'after-reference-beta', 'before-two-clients', 'after-two-clients']
    pilot.close()
    assert not authority.unsafe


def test_indistinguishable_serial_references_fail_before_client_submissions():
    model, resident, budget, authority, pilot, stages, saved = make_pilot(16, collapse=True)
    with pytest.raises(RuntimeError, match='indistinguishable'):
        pilot.run()
    assert model.calls == 2 and authority.unsafe
    assert set(saved) == {'reference-alpha', 'reference-beta'}
    with pytest.raises(RuntimeError, match='uncertain'):
        pilot.close()


def test_wrong_client_outputs_are_rejected_and_owner_becomes_unsafe():
    model, resident, budget, authority, pilot, stages, saved = make_pilot(16, misroute=True)
    with pytest.raises(RuntimeError, match='distinct serial reference'):
        pilot.run()
    assert model.calls == 4 and authority.unsafe
    with pytest.raises(RuntimeError, match='uncertain'):
        pilot.close()


@pytest.mark.parametrize('wrong_job', [999, True])
def test_outcome_job_identity_must_match_the_exact_handle(wrong_job):
    from dataclasses import replace
    model, resident, budget, authority, pilot, stages, saved = make_pilot(16)
    pilot.run()
    original = pilot.outcomes[0]
    handle = pilot.handles[0]
    class ForgedHandle:
        job_id = handle.job_id
        def wait(self, timeout):
            return replace(original, job_id=wrong_job)
    with pytest.raises(RuntimeError, match='identity'):
        pilot._accept(ForgedHandle(), team='alpha', request_id='reference-alpha')
    pilot.close()


def test_full_vocabulary_payload_is_bounded_and_distinct():
    model, resident, budget, authority, pilot, stages, saved = make_pilot(32, vocab=50257)
    result = pilot.run()
    assert result['comparisons'][0]['full']['count'] == 32 * 50257
    assert all(value.numel() * value.element_size() == 32 * 50257 * 4 for value in saved.values())
    pilot.close()


def test_comparator_rejects_nonfinite_and_uses_explicit_tolerance():
    baseline = torch.tensor([1., 1.], dtype=torch.float32)
    within = torch.tensor([1. + 1e-6, 1.], dtype=torch.float32)
    outside = torch.tensor([1. + 2e-5, 1.], dtype=torch.float32)
    assert compare_f32(within, baseline)['violations'] == 0
    assert compare_f32(outside, baseline)['violations'] == 1
    with pytest.raises(ValueError, match='nonfinite'):
        compare_f32(torch.tensor([float('nan'), 1.]), baseline)


def test_comparator_decodes_subnormal_bits_even_with_flush_enabled():
    import math
    from looking_glass.short_service_pilot_core import _decode_f32_bits
    raw = torch.tensor([1, -2147483647, -2147483648], dtype=torch.int32).view(torch.float32)
    zero = torch.zeros(3, dtype=torch.float32)
    try:
        if not torch.set_flush_denormal(True):
            pytest.skip('this runtime cannot control float32 flush mode')
        decoded = _decode_f32_bits(raw)
        result = compare_f32(raw, zero, atol=0., rtol=0.)
        assert result['violations'] == 2 and result['max_abs'] == 2. ** -149
        assert math.copysign(1., float(decoded[2])) == -1.
    finally:
        torch.set_flush_denormal(False)
