"""CPU-only consistency tests for the fixed exposed-pair GPU owner."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from looking_glass import vision_coco_cls_exposed_gpu as owner
from looking_glass.vision_coco_cls_gpu_adapter import ForwardAdapter
from looking_glass.vision_coco_cls_study import run_comparison
from looking_glass.vision_coco_study import Evidence


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.nn.Module()
        self.vit.encoder = torch.nn.Module()
        self.vit.encoder.layer = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])

    def forward(self, pixel_values, return_dict=True, output_hidden_states=False, output_attentions=False):
        assert return_dict and not output_hidden_states and not output_attentions
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


def _measurement_with_stages(root, results, count):
    total, cap = 8 * owner.GIB, 2 * owner.GIB
    memory = {'device_free_bytes': 7 * owner.GIB, 'device_total_bytes': total,
              'allocated_bytes': 0, 'reserved_bytes': 0, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
    stages = (['before_model_load', 'before_model_transfer', 'after_model_transfer']
              + [f'after_forward_{index}' for index in range(1, count + 1)]
              + ['after_final_state_check', 'after_model_disposal', 'after_cleanup'])
    for index, name in enumerate(stages, 1):
        (root / f'stage-{index:03d}.json').write_text(json.dumps(
            {'stage': name, 'memory': memory, 'peak_scope': 'since_pre_model_reset',
             'host_memory': {'working_set_bytes': 100, 'peak_working_set_bytes': 200,
                             'private_bytes': 300}}), encoding='utf-8')
    runtime = {'device_total_bytes': total, 'allocator_cap_bytes': cap,
        'commitment_bytes': cap + owner.HEADROOM, 'gpu_trace_claim_bytes': owner.TRACE,
        'host_trace_claim_bytes': owner.TRACE, 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
        'transformers': '4.57.3', 'numpy': '2.4.6', 'device': 'cuda:0',
        'device_name': 'NVIDIA GeForce RTX 2070 SUPER', 'compute_capability': [7, 5],
        'dtype': 'float32', 'attention': 'eager', 'inference_mode': True, 'default_dtype': 'torch.float32',
        'binary64_rounding': 'nearest-even', 'denormals_preserved': True,
        'seed': 17, 'nvidia_total_bytes': total,
        'deterministic_algorithms': True, 'matmul_tf32': False, 'cudnn_tf32': False,
        'cudnn_benchmark': False, 'cudnn_deterministic': True, 'autocast_enabled': False,
        'allocator': 'native', 'threads': 1, 'interop_threads': 1}
    return {'reference_sha256': owner.REFERENCE_SHA, 'model_state_sha256': owner.STATE_SHA,
            'results': results, 'forward_count': count, 'model_and_parameter_weakrefs_dead': True,
            'runtime': runtime, 'stages': stages, 'initial_memory': memory,
            'memory_before_transfer': memory, 'cleanup_memory': memory}


def test_owner_module_import_leaves_torch_absent_in_fresh_process():
    root = str(Path(__file__).resolve().parents[1])
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "import looking_glass.vision_coco_cls_exposed_gpu; assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, root],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_two_volume_capacity_accounts_for_output_and_workspace():
    assert owner._capacity_policy(3 * owner.GIB, owner.GIB, False) == 3 * owner.GIB
    assert owner._capacity_policy(4 * owner.GIB, 4 * owner.GIB, True) == 4 * owner.GIB
    for output, workspace, same in ((3 * owner.GIB - 1, owner.GIB, False),
                                    (3 * owner.GIB, owner.GIB - 1, False),
                                    (4 * owner.GIB - 1, 4 * owner.GIB - 1, True)):
        with pytest.raises(RuntimeError, match='headroom'):
            owner._capacity_policy(output, workspace, same)
    for output, workspace, same in ((True, owner.GIB, False), (3 * owner.GIB, False, False),
                                    (3 * owner.GIB, owner.GIB, 0)):
        with pytest.raises(ValueError, match='observation'):
            owner._capacity_policy(output, workspace, same)


def test_runtime_temp_and_caches_must_be_inside_the_attempt(tmp_path, monkeypatch):
    run = tmp_path / 'attempt'
    run.mkdir()
    for name in ('temp', 'hf-cache', 'cuda-cache'):
        (run / name).mkdir()
    for key, name in (('TEMP', 'temp'), ('TMP', 'temp'),
                      ('HF_HOME', 'hf-cache'), ('CUDA_CACHE_PATH', 'cuda-cache')):
        monkeypatch.setenv(key, str(run / name))
    owner._validate_runtime_dirs(run)
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    with pytest.raises(ValueError, match='temporary/cache'):
        owner._validate_runtime_dirs(run)
    monkeypatch.delenv('HF_HOME')
    with pytest.raises(ValueError, match='temporary/cache'):
        owner._validate_runtime_dirs(run)


@pytest.mark.skipif(os.name != 'nt', reason='the fixed host uses the Windows CRT')
def test_real_host_rounding_guard_rejects_directed_mode_in_subprocess():
    root = str(Path(__file__).resolve().parents[1])
    code = r'''
import ctypes, sys
sys.path.insert(0, sys.argv[1])
from looking_glass.vision_coco_recognized_screen import check_rounding
control = ctypes.CDLL('msvcrt')._controlfp
control.argtypes = (ctypes.c_uint, ctypes.c_uint)
control.restype = ctypes.c_uint
before = control(0, 0)
try:
    control(0x100, 0x300)
    try:
        check_rounding()
    except ValueError:
        pass
    else:
        raise AssertionError('directed rounding was accepted')
finally:
    control(before, 0x300)
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, root],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_fixed_conditional_schedule():
    both = owner._labels({'cat': 'matchable', 'dog': 'matchable'})
    one = owner._labels({'cat': 'matchable', 'dog': 'dose-ineligible'})
    neither = owner._labels({'cat': 'dose-unmatchable', 'dog': 'dose-ineligible'})
    assert len(both) == 34 and len(one) == 28 and len(neither) == 22
    assert both[:4] == ['pair-00:cat:initial', 'pair-00:dog:initial',
                        'pair-00:cat_donor:initial', 'pair-00:dog_donor:initial']
    assert both[-1] == 'pair-00:dog:post_clean'
    with pytest.raises(ValueError):
        owner._labels({'cat': True, 'dog': 'matchable'})


def test_actual_fabricated_runner_saved_rows_replay_and_tampering_rejects(tmp_path):
    model = FakeModel().eval()
    pixels = {index: torch.full((1, 3, 224, 224), value)
              for index, value in ((1, .1), (2, 1.0), (3, .2), (4, 1.2))}
    pair = {'pair_id': 'pair-00', 'split': 'heldout', 'box_tokens': [90, 91, 104, 105],
            'cat': {'image_id': 1}, 'dog': {'image_id': 2},
            'cat_donor': {'image_id': 3}, 'dog_donor': {'image_id': 4}}
    evidence = Evidence(tmp_path / 'data', 34)
    adapter = ForwardAdapter(check=lambda: None, synchronize=lambda: None, device='cpu')
    results = run_comparison(model, [pair], pixels, evidence, forward_fn=adapter)
    measurement = _measurement_with_stages(tmp_path, results, evidence.submissions)
    assert owner._validate_saved(tmp_path, measurement) == 34
    path = tmp_path / 'data' / 'forward-006.json'
    saved = path.read_bytes()
    row = json.loads(saved)
    del row['layout']['before_all']
    path.write_text(json.dumps(row), encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        owner._validate_saved(tmp_path, measurement)
    path.write_bytes(saved)
    binary = tmp_path / 'data' / 'forward-006.bin'
    value = binary.read_bytes()
    binary.write_bytes(bytes([value[0] ^ 1]) + value[1:])
    with pytest.raises(ValueError, match='identity'):
        owner._validate_saved(tmp_path, measurement)
    binary.write_bytes(value)
    stage = tmp_path / 'stage-001.json'
    point = json.loads(stage.read_bytes())
    point['memory']['device_total_bytes'] -= 1
    stage.write_text(json.dumps(point), encoding='utf-8')
    with pytest.raises(ValueError, match='memory counters'):
        owner._validate_saved(tmp_path, measurement)


def test_diff_requires_complete_finite_logit_vectors():
    import struct
    left = struct.pack('<1000f', *([1.0] * 1000))
    right = struct.pack('<1000f', *([1.5] * 1000))
    assert owner._diff(left, right) == {'max_abs': .5, 'mean_abs': .5, 'count': 1000}
    with pytest.raises(ValueError):
        owner._diff(left[:-4], right)
    with pytest.raises(ValueError):
        owner._diff(left, struct.pack('<1000f', *([float('nan')] + [1.5] * 999)))


def _fake_device_setup(monkeypatch, *, fail_on=None, rounding_drift_on=None, eval_drift_on=None):
    import transformers
    from looking_glass import vision_coco_cls_gpu_adapter as adapter_module
    from looking_glass import vision_coco_recognized_screen
    calls, observed = [], []
    rounding = {'nearest': True}
    def check_rounding():
        if not rounding['nearest']:
            raise ValueError('binary64 round-to-nearest is unavailable')
    monkeypatch.setattr(vision_coco_recognized_screen, 'check_rounding', check_rounding)

    class DeviceFakeModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=192, num_hidden_layers=12, num_labels=1000,
                                          patch_size=16, image_size=224, _attn_implementation='eager')
            self.forward_count = 0

        def to(self, device):
            assert device == 'cuda:0'
            calls.append('model-transfer')
            return self

        def forward(self, *args, **kwargs):
            self.forward_count += 1
            calls.append(f'forward-{self.forward_count}')
            if self.forward_count == fail_on:
                raise torch.cuda.OutOfMemoryError('fabricated OOM')
            result = super().forward(*args, **kwargs)
            if self.forward_count == rounding_drift_on:
                rounding['nearest'] = False
            if self.forward_count == eval_drift_on:
                self.vit.encoder.layer[0].train(True)
            return result

    def load(*_args, **_kwargs):
        model = DeviceFakeModel()
        observed.append(weakref.ref(model))
        return model, {}

    class CPUAdapter(ForwardAdapter):
        def __init__(self, **kwargs):
            kwargs['device'] = 'cpu'
            super().__init__(**kwargs)

    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(transformers.ViTForImageClassification, 'from_pretrained', load)
    monkeypatch.setattr(adapter_module, 'ForwardAdapter', CPUAdapter)
    monkeypatch.setattr(owner, '_held', lambda _session, _lease: None)
    monkeypatch.setattr(owner, '_runtime_uuid', lambda _torch, _uuid: 'GPU-fake')
    monkeypatch.setattr(owner, '_state_digest', lambda _model, check=None: owner.STATE_SHA)
    monkeypatch.setattr(owner.OwnerSession, 'verify_runtime_binding', lambda *_a, **_k: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _i: SimpleNamespace(
        name='NVIDIA GeForce RTX 2070 SUPER', major=7, minor=5, total_memory=8 * owner.GIB, uuid='fake'))
    monkeypatch.setattr(torch.cuda, 'get_allocator_backend', lambda: 'native')
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *_a: calls.append('set-cap'))
    monkeypatch.setattr(torch.cuda, 'get_per_process_memory_fraction', lambda _i: .25)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda _i: calls.append('reset-peak'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _i: calls.append('sync'))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda _i: (7 * owner.GIB, 8 * owner.GIB))
    for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda _i: 0)
    monkeypatch.setattr(torch.cuda, 'manual_seed', lambda _seed: calls.append('seed'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('empty-cache'))
    monkeypatch.setattr(torch._C, '_cuda_clearCublasWorkspaces', lambda: calls.append('clear-workspace'))
    monkeypatch.setattr(torch, 'set_num_threads', lambda _n: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda _n: None)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 1)
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', lambda _v: None)
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: True)
    monkeypatch.setattr(owner, '_ROOT', dict(owner._ROOT, entered=False, pending=False, gc_disabled=False,
        session=None, lease=None, budget=None, memory_lease=None, claims_ready=False, model=None,
        adapter=None, evidence=None, failure=None, diagnostic_failure=None, completed_forwards=0))
    monkeypatch.setenv('LOOKING_GLASS_VISION_SMI_TOTAL', str(8 * owner.GIB))
    pair = {'pair_id': 'pair-00', 'split': 'heldout', 'box_tokens': [90, 91, 104, 105]}
    pair.update({role: {'image_id': image_id} for role, image_id in owner.ROLES.items()})
    files = {}
    for role, value in (('cat', .1), ('dog', 1.0), ('cat_donor', .2), ('dog_donor', 1.2)):
        files[f'{role}-pixels.f32'] = torch.full((1, 3, 224, 224), value).numpy().tobytes()
        files[f'{role}-cpu-logits.f32'] = bytes(4000)
    return calls, observed, {'pair': pair}, files


def test_fabricated_device_owner_runs_and_drains_34_actions(tmp_path, monkeypatch):
    calls, refs, reference, files = _fake_device_setup(monkeypatch)
    result = owner._measure(object(), object(), torch, tmp_path, reference, files, tmp_path)
    assert result['forward_count'] == 34 and owner._validate_saved(tmp_path, result) == 34
    assert calls.count('model-transfer') == 1 and calls.count('reset-peak') == 1
    assert len([call for call in calls if call.startswith('forward-')]) == 34
    assert 'clear-workspace' in calls and result['cleanup_memory']['reserved_bytes'] == 0
    assert refs[0]() is None and owner._ROOT['model'] is None and not owner._ROOT['pending']
    assert owner._ROOT['budget'].active_leases() == (owner._ROOT['memory_lease'],)


def test_fabricated_oom_stops_device_work_and_retains_claim(tmp_path, monkeypatch):
    calls, refs, reference, files = _fake_device_setup(monkeypatch, fail_on=3)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        owner._measure(object(), object(), torch, tmp_path, reference, files, tmp_path)
    assert calls[-1] == 'forward-3' and 'clear-workspace' not in calls
    assert refs[0]() is not None and owner._ROOT['pending'] and owner._ROOT['adapter'].failed
    assert owner._ROOT['budget'].active_leases() == (owner._ROOT['memory_lease'],)
    assert (tmp_path / 'data' / 'forward-002.bin').is_file()
    assert not (tmp_path / 'candidate.json').exists()


def test_changed_provider_total_stops_before_allocator_or_model_work(tmp_path, monkeypatch):
    calls, refs, reference, files = _fake_device_setup(monkeypatch)
    monkeypatch.setenv('LOOKING_GLASS_VISION_SMI_TOTAL', str(8 * owner.GIB - 2 * owner.MIB - 1))
    with pytest.raises(RuntimeError, match='device totals differ'):
        owner._measure(object(), object(), torch, tmp_path, reference, files, tmp_path)
    assert 'set-cap' not in calls and 'model-transfer' not in calls and refs == []
    assert owner._ROOT['pending'] and owner._ROOT['model'] is None


def test_rounding_drift_after_forward_preserves_bytes_and_stops(tmp_path, monkeypatch):
    calls, refs, reference, files = _fake_device_setup(monkeypatch, rounding_drift_on=1)
    with pytest.raises(ValueError, match='round-to-nearest'):
        owner._measure(object(), object(), torch, tmp_path, reference, files, tmp_path)
    assert calls.count('forward-1') == 1 and 'clear-workspace' not in calls
    assert refs[0]() is not None and owner._ROOT['pending'] and owner._ROOT['adapter'].failed
    assert (tmp_path / 'data' / 'forward-001.bin').is_file()
    assert not (tmp_path / 'data' / 'forward-001.json').exists()
    assert owner._ROOT['budget'].active_leases() == (owner._ROOT['memory_lease'],)


def test_eval_state_drift_after_forward_preserves_bytes_and_stops(tmp_path, monkeypatch):
    calls, refs, reference, files = _fake_device_setup(monkeypatch, eval_drift_on=1)
    with pytest.raises(RuntimeError, match='evaluation state changed'):
        owner._measure(object(), object(), torch, tmp_path, reference, files, tmp_path)
    assert calls.count('forward-1') == 1 and 'clear-workspace' not in calls
    assert refs[0]() is not None and owner._ROOT['pending'] and owner._ROOT['adapter'].failed
    assert (tmp_path / 'data' / 'forward-001.bin').is_file()
    assert not (tmp_path / 'data' / 'forward-001.json').exists()


def test_changed_default_dtype_rejects_before_model_load(tmp_path, monkeypatch):
    calls, refs, reference, files = _fake_device_setup(monkeypatch)
    monkeypatch.setattr(torch, 'get_default_dtype', lambda: torch.float64)
    with pytest.raises(RuntimeError, match='numeric execution settings differ'):
        owner._measure(object(), object(), torch, tmp_path, reference, files, tmp_path)
    assert refs == [] and 'model-transfer' not in calls
    assert owner._ROOT['budget'].active_leases() == (owner._ROOT['memory_lease'],)


def test_owner_loss_during_state_copy_stops_before_next_parameter(monkeypatch):
    model = torch.nn.Linear(2, 2)
    state = {'held': True, 'copies': 0}
    def check():
        if not state['held']:
            raise RuntimeError('owner revoked')
    def copy(value):
        state['copies'] += 1
        state['held'] = False
        return value.detach().cpu().contiguous()
    monkeypatch.setattr(owner, '_state_copy', copy)
    with pytest.raises(RuntimeError, match='owner revoked'):
        owner._state_digest(model, check=check)
    assert state['copies'] == 1


@pytest.mark.parametrize('supervisor_result', ['raises', 'unreaped', 'cancel'])
def test_uncertain_supervisor_never_claims_post_exit_or_queries_gpu(tmp_path, monkeypatch, supervisor_result):
    commit = 'a' * 40
    identity = SimpleNamespace(lock_path=tmp_path / 'gpu.lock', marker_path=tmp_path / 'gpu.unsafe.json')
    seen = []
    monkeypatch.setattr(owner, '_startup', lambda: None)
    monkeypatch.setattr(owner, '_pristine', lambda: None)
    monkeypatch.setattr(owner, '_workspace_root', lambda *_a: tmp_path)
    monkeypatch.setattr(owner, '_capacity', lambda *_a: {'disk_free_bytes': 4 * owner.GIB})
    monkeypatch.setattr(owner, '_source_identity', lambda *_a: (commit, {'source': 'fixed'}))
    monkeypatch.setattr(owner, '_read_reference', lambda *_a: ({}, {}))
    monkeypatch.setattr(owner, '_asset_inventory', lambda *_a: {})
    monkeypatch.setattr(owner, 'resolve_device_identity', lambda *_a, **_k: identity)
    monkeypatch.setattr(owner, '_smi_snapshot', lambda _smi: (seen.append('query') or
        ('GPU-fake', {'timestamp_utc': '2026-10-29T12:00:00Z', 'total_bytes': 8 * owner.GIB})))
    def supervise(*_args, **_kwargs):
        identity.marker_path.write_text('possibly-running', encoding='ascii')
        if supervisor_result == 'raises':
            raise RuntimeError('fabricated interruption after grant')
        if supervisor_result == 'cancel':
            raise KeyboardInterrupt()
        return SimpleNamespace(run_id='fake', pid=123, exit_code=None, clean=False,
                               timed_out=False, reason='unreaped')
    monkeypatch.setattr(owner, 'run_supervised', supervise)
    monkeypatch.setenv('PROGRAMFILES', str(tmp_path))
    monkeypatch.setenv('LOOKING_GLASS_VISION_GIT_EXE', str(tmp_path / 'git.exe'))
    monkeypatch.setenv('LOOKING_GLASS_VISION_SOURCE_COMMIT', commit)
    if supervisor_result == 'cancel':
        with pytest.raises(KeyboardInterrupt):
            owner.controller_main(tmp_path, tmp_path, tmp_path, tmp_path / 'attempt', tmp_path, 'fake-smi')
        result = json.loads((tmp_path / 'attempt' / 'outcome.json').read_bytes())
    else:
        result = owner.controller_main(tmp_path, tmp_path, tmp_path, tmp_path / 'attempt', tmp_path, 'fake-smi')
    assert result['status'] == 'incomplete' and seen == ['query']
    assert result['nvidia_smi_after'] is None and result['owner_presence_after_exit']['query'] == 'not_attempted'
    assert result['marker_absent_at_post_exit_point'] is None
    assert result['attestation_absent_at_post_exit_point'] is None
    assert identity.marker_path.read_text(encoding='ascii') == 'possibly-running'
