import pytest
import torch

from looking_glass.tap_repeated_cpu_trace import cpu_trace_preflight
from looking_glass import tap_repeated_protocol as protocol
from looking_glass import tap_repeated_runner as runner


def test_real_cpu_gpt2_direct_and_gradient_hooks_match_without_cuda(monkeypatch):
    transformers = pytest.importorskip('transformers')
    config = transformers.GPT2Config(vocab_size=15497, n_embd=16, n_layer=1, n_head=2, n_positions=8)
    config._attn_implementation = 'eager'
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = transformers.GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[15496, 995]], dtype=torch.int64)
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('must not query CUDA'))
        report = cpu_trace_preflight(model, ids, session=2)
        class Journal:
            def __init__(self): self.rows = []
            def append(self, row): self.rows.append(row)
        device = runner._CPUTraceDevice(2, model, ids)
        guard, journal = runner._GuardedDevice(device, runner._SimulatedOwner()), Journal()
        runner._preflight(guard, journal, 2, 8 * protocol.GIB, 2 * protocol.GIB)
        for position, mode in enumerate(protocol.MODES):
            item = {'stage': 'measured', 'round': 0, 'position': position, 'mode': mode,
                    'observation_id': f'measured:0:{position}'}
            runner._sample(guard, journal, item, 2, 8 * protocol.GIB, 2 * protocol.GIB, 1)
        assert journal.rows[0]['checks']['value_layouts'][0]['shape'] == [1, 1, 16]
        samples = [row for row in journal.rows if row['stage'] == 'measured']
        assert [row['events'] for row in samples] == [0, 1, 1]
        assert [row['recorded_tensor_bytes'] for row in samples] == [0, 64, 64]
        assert all(row['output_shape'] == [1, 2, 15497] and row['output_finite'] for row in samples)
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
        with pytest.raises(ValueError, match='fixed CPU token IDs'):
            cpu_trace_preflight(model, torch.tensor([[15496, 1]]), session=2)
        model.train()
        with pytest.raises(ValueError, match='evaluation-mode'):
            cpu_trace_preflight(model, ids, session=2)
    finally:
        torch.set_num_threads(previous)
    assert report['cpu_only'] is True and report['gpu_eligible'] is False
    assert report['pretrained_identity_verified'] is False and report['matches_protocol_layout'] is False
    assert report['outputs_equal'] and report['captures_equal']
    assert report['event_identity'] == ['transformer.h.0', 'block', 0, 0, 0, [0], [0], [1]]
    assert report['captured_shape'] == [1, 1, 16] and report['captured_bytes'] == 64
    assert not torch.cuda.is_initialized()
