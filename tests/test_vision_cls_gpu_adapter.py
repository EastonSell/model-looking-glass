"""CPU-only tests for the fixed exposed-pair CUDA forward boundary."""

import gc
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest
import torch

from looking_glass import vision_coco_cls_gpu_adapter as adapter
from looking_glass.vision_coco_study import EARLY, Evidence


class FakeModel(torch.nn.Module):
    def __init__(self, fail_after_capture=False):
        super().__init__()
        self.vit = torch.nn.Module()
        self.vit.encoder = torch.nn.Module()
        self.vit.encoder.layer = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
        self.fail_after_capture = fail_after_capture
        self.calls = 0

    def forward(self, pixel_values, return_dict=True, output_hidden_states=False, output_attentions=False):
        assert return_dict is True and output_hidden_states is False and output_attentions is False
        self.calls += 1
        value = torch.arange(197 * 192, dtype=torch.float32).reshape(1, 197, 192) / 10000
        value = value + pixel_values.reshape(-1)[0]
        for layer in self.vit.encoder.layer:
            value = layer(value)
        if self.fail_after_capture:
            raise torch.cuda.OutOfMemoryError('fabricated failure after captures')
        logits = torch.arange(1000, dtype=torch.float32).reshape(1, 1000) / 1000 + value.sum() / 100000
        return FakeOutput(logits)


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


def test_adapter_import_does_not_import_torch_in_fresh_process():
    root = str(Path(__file__).resolve().parents[1])
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "import looking_glass.vision_coco_cls_gpu_adapter; "
            "assert 'torch' not in sys.modules")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, root],
                            env=environment, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_full_state_source_and_capture_overlap_is_bounded(tmp_path):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    calls = []
    worker = adapter.ForwardAdapter(check=lambda: calls.append('check'),
                                    synchronize=lambda: calls.append('sync'), device='cpu')
    source = torch.full((1, 197, 192), 3.0)
    logits, captures, row = worker(model, torch.zeros(1, 3, 224, 224), evidence, 'full-swap',
                                   path=EARLY, tokens=None, replacement=source, action='patch')
    assert torch.equal(captures['after_all'], source)
    assert row['metadata']['source_bytes'] == 151296
    assert row['metadata']['capture_bytes'] == 302592
    assert row['layout']['requested']['bytes'] == 151296
    assert row['layout']['before_all']['bytes'] == 151296
    assert row['layout']['after_all']['bytes'] == 151296
    assert logits.shape == (1, 1000)
    assert worker.pending is False and worker.failed is False and worker.roots is None
    assert calls.count('sync') >= 2
    assert not any(module._forward_hooks for module in model.modules())


def test_four_token_plan_preserves_unselected_values(tmp_path):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    tokens = [90, 91, 104, 105]
    source = torch.full((4, 192), -2.0)
    _, captures, row = worker(model, torch.zeros(1, 3, 224, 224), evidence, 'spatial',
                              path=EARLY, tokens=tokens, replacement=source, action='patch')
    before, after = captures['before_all'], captures['after_all']
    expected = before.clone()
    expected[0, tokens] = source
    assert torch.equal(after, expected)
    assert row['metadata']['source_bytes'] == 3072
    assert row['metadata']['capture_bytes'] == 308736


def test_failed_completion_retains_roots_and_forbids_reuse(tmp_path):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    syncs = []
    def fail_sync():
        syncs.append('sync')
        raise RuntimeError('fabricated completion failure')
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=fail_sync, device='cpu')
    with pytest.raises(RuntimeError, match='completion failure'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'failed')
    assert worker.pending and worker.failed
    assert worker.roots.output is not None and worker.roots.plan is not None
    assert {'early', 'late'} <= set(worker.roots.capture_roots)
    assert model.calls == 1 and syncs == ['sync']
    with pytest.raises(RuntimeError, match='unsafe'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'forbidden')
    assert model.calls == 1 and syncs == ['sync']


def test_oom_after_capture_keeps_plan_and_does_not_sync(tmp_path):
    model = FakeModel(fail_after_capture=True).eval()
    evidence = Evidence(tmp_path / 'data', 34)
    syncs = []
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: syncs.append('sync'), device='cpu')
    with pytest.raises(torch.cuda.OutOfMemoryError):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'oom')
    assert worker.pending and worker.failed and worker.roots.plan is not None
    assert {'early', 'late'} <= set(worker.roots.capture_roots)
    assert syncs == []


def test_loss_during_completion_stops_before_later_device_work(tmp_path):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    state = {'held': True, 'syncs': 0}
    def check():
        if not state['held']:
            raise RuntimeError('owner revoked')
    def revoke_during_sync():
        state['syncs'] += 1
        state['held'] = False
    worker = adapter.ForwardAdapter(check=check, synchronize=revoke_during_sync, device='cpu')
    with pytest.raises(RuntimeError, match='owner revoked'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'revoked')
    assert state['syncs'] == 1 and model.calls == 1
    assert worker.failed and worker.pending and worker.roots.output is not None
    assert {'early', 'late'} <= set(worker.roots.capture_roots)
    with pytest.raises(RuntimeError, match='unsafe'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'forbidden')
    assert state['syncs'] == 1 and model.calls == 1


def test_non_cpu_handoff_fails_before_scientific_helpers(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    monkeypatch.setattr(adapter, '_copy_to_cpu',
                        lambda value: torch.empty(value.shape, dtype=value.dtype, device='meta'))
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    with pytest.raises(ValueError, match='logit copy did not reach CPU'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'bad-copy')
    assert worker.pending and worker.failed and worker.roots.output is not None
    assert not (evidence.root / 'forward-001.json').exists()


def test_non_cpu_capture_fails_before_scientific_handoff(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    copies = []
    def copy(value):
        copies.append(1)
        if len(copies) == 2:
            return torch.empty(value.shape, dtype=value.dtype, device='meta')
        return value.detach().to('cpu').contiguous()
    monkeypatch.setattr(adapter, '_copy_to_cpu', copy)
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    with pytest.raises(ValueError, match='scientific handoff requires CPU tensors'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'bad-capture-copy')
    assert worker.pending and worker.failed and worker.roots.output is not None
    assert (evidence.root / 'return-001.bin').is_file()
    assert not (evidence.root / 'forward-001.json').exists()


def test_model_mutation_of_distinct_device_input_is_detected(tmp_path, monkeypatch):
    class MutatingModel(FakeModel):
        def forward(self, pixel_values, return_dict=True, output_hidden_states=False, output_attentions=False):
            pixel_values.add_(1.0)
            return super().forward(pixel_values, return_dict=return_dict,
                                   output_hidden_states=output_hidden_states,
                                   output_attentions=output_attentions)

    pixels = torch.zeros(1, 3, 224, 224)
    model = MutatingModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    monkeypatch.setattr(adapter, '_transfer_to_device', lambda value, device: value.detach().clone())
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    with pytest.raises(ValueError, match='model modified input pixels'):
        worker(model, pixels, evidence, 'mutated-input')
    assert not pixels.any() and model.calls == 1
    assert worker.failed and worker.pending and worker.roots.pixels is not None
    assert (evidence.root / 'forward-001.bin').is_file()
    assert evidence.active['saved_file'] == 'forward-001.bin'
    assert not (evidence.root / 'forward-001.json').exists()


def test_failed_input_readback_keeps_completed_activation_bytes(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    copied = []
    def copy(value):
        copied.append(1)
        if len(copied) == 4:  # logits, early, late, then the device input
            raise RuntimeError('fabricated input readback failure')
        return value.detach().to('cpu').contiguous()
    monkeypatch.setattr(adapter, '_copy_to_cpu', copy)
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    with pytest.raises(RuntimeError, match='input readback failure'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'failed-input-readback')
    assert len(copied) == 4 and model.calls == 1 and worker.failed and worker.pending
    assert worker.roots.pixels is not None and worker.roots.output is not None
    assert (evidence.root / 'forward-001.bin').is_file()
    assert evidence.active['saved_file'] == 'forward-001.bin'
    assert not (evidence.root / 'forward-001.json').exists()


def test_validation_failure_after_saved_row_cannot_be_promoted_or_reused(tmp_path, monkeypatch):
    from looking_glass import vision_coco_study

    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    def reject_metadata(*_args):
        raise ValueError('fabricated metadata rejection')
    monkeypatch.setattr(vision_coco_study, '_check_metadata', reject_metadata)
    with pytest.raises(ValueError, match='metadata rejection'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'unverified-row')
    assert (evidence.root / 'forward-001.json').is_file()
    assert (evidence.root / 'forward-001.bin').is_file()
    assert len(evidence.records) == 1 and model.calls == 1
    assert worker.pending and worker.failed and worker.roots.output is not None
    with pytest.raises(RuntimeError, match='unsafe'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'forbidden')
    assert model.calls == 1


def test_revocation_during_cpu_guard_stops_before_transfer(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    state = {'held': True, 'transfers': 0}
    def check():
        if not state['held']:
            raise RuntimeError('owner revoked')
    def transfer(value, device):
        state['transfers'] += 1
        return value.detach().to(device)
    monkeypatch.setattr(adapter, '_transfer_to_device', transfer)
    worker = adapter.ForwardAdapter(check=check, synchronize=lambda: None, device='cpu')
    with pytest.raises(RuntimeError, match='owner revoked'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'revoked-before-transfer',
               numeric_guard=lambda: state.update(held=False))
    assert state['transfers'] == 0 and model.calls == 0 and worker.failed


@pytest.mark.parametrize(('tokens', 'shape'), [
    (None, (1, 197, 193)), ([0], (2, 192)), ([90, 91, 104, 105], (5, 192)),
])
def test_wrong_replacement_geometry_stops_before_any_transfer(tmp_path, monkeypatch, tokens, shape):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    transfers = []
    monkeypatch.setattr(adapter, '_transfer_to_device',
                        lambda value, device: (transfers.append(1), value.detach().to(device))[1])
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    with pytest.raises(ValueError, match='replacement geometry'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'oversized',
               path=EARLY, tokens=tokens, replacement=torch.zeros(shape), action='patch')
    assert transfers == [] and model.calls == 0 and worker.failed


@pytest.mark.parametrize('tokens', [[False], [0.0], (0,), [90, 91, 104, True]])
def test_invalid_token_types_stop_before_any_transfer(tmp_path, monkeypatch, tokens):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    transfers = []
    monkeypatch.setattr(adapter, '_transfer_to_device',
                        lambda value, device: (transfers.append(1), value.detach().to(device))[1])
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    with pytest.raises(ValueError, match='token inventory'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'invalid-tokens',
               path=EARLY, tokens=tokens, replacement=torch.zeros(1, 192), action='patch')
    assert transfers == [] and model.calls == 0 and worker.failed


def test_revocation_during_first_transfer_stops_before_replacement(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    state = {'held': True, 'transfers': 0}
    def check():
        if not state['held']:
            raise RuntimeError('owner revoked')
    def transfer(value, device):
        state['transfers'] += 1
        state['held'] = False
        return value.detach().to(device)
    monkeypatch.setattr(adapter, '_transfer_to_device', transfer)
    worker = adapter.ForwardAdapter(check=check, synchronize=lambda: None, device='cpu')
    with pytest.raises(RuntimeError, match='owner revoked'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'revoked-transfer',
               path=EARLY, tokens=None, replacement=torch.zeros(1, 197, 192), action='patch')
    assert state['transfers'] == 1 and model.calls == 0 and worker.roots.pixels is not None
    assert worker.roots.replacement is None and worker.failed


def test_revocation_during_logit_copy_stops_before_capture_copies(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    state = {'held': True, 'copies': 0}
    def check():
        if not state['held']:
            raise RuntimeError('owner revoked')
    def copy(value):
        state['copies'] += 1
        state['held'] = False
        return value.detach().to('cpu').contiguous()
    monkeypatch.setattr(adapter, '_copy_to_cpu', copy)
    worker = adapter.ForwardAdapter(check=check, synchronize=lambda: None, device='cpu')
    with pytest.raises(RuntimeError, match='owner revoked'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'revoked-copy')
    assert state['copies'] == 1 and model.calls == 1 and worker.roots.output is not None
    assert worker.failed and worker.pending


def test_revocation_during_first_capture_stops_before_next_copy(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    state = {'held': True, 'copies': 0}
    def check():
        if not state['held']:
            raise RuntimeError('owner revoked')
    def copy(value):
        state['copies'] += 1
        if state['copies'] == 2:
            state['held'] = False
        return value.detach().to('cpu').contiguous()
    monkeypatch.setattr(adapter, '_copy_to_cpu', copy)
    worker = adapter.ForwardAdapter(check=check, synchronize=lambda: None, device='cpu')
    with pytest.raises(RuntimeError, match='owner revoked'):
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'revoked-capture')
    assert state['copies'] == 2 and model.calls == 1
    assert worker.failed and worker.pending and worker.roots.output is not None


def test_successful_full_state_drain_drops_per_forward_device_roots(tmp_path, monkeypatch):
    model = FakeModel().eval()
    evidence = Evidence(tmp_path / 'data', 34)
    refs = {}
    worker = None
    def transfer(value, _device):
        return value.detach().clone()
    def copy(value):
        return value.detach().to('cpu').contiguous().clone()
    def sync():
        roots = worker.roots
        if not refs and roots is not None and roots.output is not None:
            refs['pixels'] = weakref.ref(roots.pixels)
            refs['replacement'] = weakref.ref(roots.replacement)
            refs['plan'] = weakref.ref(roots.plan)
            refs['output'] = weakref.ref(roots.output)
            refs['logits'] = weakref.ref(roots.output.logits)
            for index, value in enumerate(roots.sources):
                refs[f'source-{index}'] = weakref.ref(value)
            for name, value in roots.capture_roots.items():
                refs[f'capture-{name}'] = weakref.ref(value)
    monkeypatch.setattr(adapter, '_transfer_to_device', transfer)
    monkeypatch.setattr(adapter, '_copy_to_cpu', copy)
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=sync, device='cpu')
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        _, captures, _ = worker(model, torch.zeros(1, 3, 224, 224), evidence, 'full-swap',
                                path=EARLY, tokens=None, replacement=torch.ones(1, 197, 192), action='patch')
        assert captures['before_all'].device.type == captures['after_all'].device.type == 'cpu'
        assert refs and all(ref() is None for ref in refs.values())
        worker(model, torch.zeros(1, 3, 224, 224), evidence, 'next-clean')
        assert model.calls == 2 and not worker.failed
    finally:
        if was_enabled:
            gc.enable()


def test_scientific_runner_composes_with_34_bounded_cpu_fake_forwards(tmp_path):
    from looking_glass.vision_coco_cls_study import run_comparison

    class PairModel(FakeModel):
        def forward(self, pixel_values, return_dict=True, output_hidden_states=False, output_attentions=False):
            assert return_dict is True and output_hidden_states is False and output_attentions is False
            self.calls += 1
            value = torch.arange(197 * 192, dtype=torch.float32).reshape(1, 197, 192) / 10000
            value = value + pixel_values.reshape(-1)[0]
            for layer in self.vit.encoder.layer:
                value = layer(value)
            logits = torch.full((1, 1000), -5.0)
            if float(value[0, 0, 0]) < .5:
                logits[:, 281:286] = 5.0
            else:
                logits[:, 151:269] = 5.0
            return FakeOutput(logits)

    model = PairModel().eval()
    pixels = {index: torch.full((1, 3, 224, 224), value)
              for index, value in ((1, .1), (2, 1.0), (3, .2), (4, 1.2))}
    pair = {'pair_id': 'pair-00', 'split': 'heldout', 'box_tokens': [90, 91, 104, 105],
            'cat': {'image_id': 1}, 'dog': {'image_id': 2},
            'cat_donor': {'image_id': 3}, 'dog_donor': {'image_id': 4}}
    evidence = Evidence(tmp_path / 'data', 34)
    worker = adapter.ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    rows = run_comparison(model, [pair], pixels, evidence, forward_fn=worker)
    assert model.calls == evidence.submissions == len(evidence.records) == 34
    assert rows[0]['dose_matchability'] == {'cat': 'matchable', 'dog': 'matchable'}
    assert not worker.pending and not worker.failed and worker.roots is None
    assert max(sum(item['bytes'] for item in row['layout'].values()) for row in evidence.records) == 457888
