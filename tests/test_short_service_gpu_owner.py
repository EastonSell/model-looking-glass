"""Fabricated CPU-only owner boundary tests; no checkpoint or GPU access."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from looking_glass import short_service_gpu_owner as owner


def test_fixed_input_selection_preserves_mask_and_active_position():
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'experiments/short_cpu_service_reference_inputs_v1.json').read_bytes())
    sixteen = owner._cases(manifest, 16)
    thirty_two = owner._cases(manifest, 32)
    assert [row['case_id'] for row in sixteen] == ['synthetic-16', 'gas']
    assert [row['active_position'] for row in sixteen] == [15, 14]
    assert [row['case_id'] for row in thirty_two] == ['synthetic-32', 'ligand']
    assert [row['active_position'] for row in thirty_two] == [31, 22]
    assert sixteen[1]['attention_mask'][-1] == 0
    assert thirty_two[1]['attention_mask'][-9:] == [0] * 9
    with pytest.raises(ValueError):
        owner._cases(manifest, True)
    manifest['comparisons'][2]['service']['attention_mask'][0] = 0
    with pytest.raises(ValueError):
        owner._cases(manifest, 16)


def memory_record(stage, *, free=6 * owner.GIB, peak=0):
    return {'stage': stage, 'device_free_bytes': free, 'device_total_bytes': 8 * owner.GIB,
            'allocated_bytes': 0, 'reserved_bytes': 0,
            'peak_allocated_bytes': peak, 'peak_reserved_bytes': peak}


def write_memory(tmp_path, records):
    for index, record in enumerate(records):
        (tmp_path / f'memory-{index:03d}.json').write_text(json.dumps(record), encoding='utf-8')


def test_memory_saved_order_reset_and_headroom_are_checked(tmp_path):
    records = [memory_record('before-model-load'), memory_record('after-model-load')]
    records += [memory_record(stage) for stage in owner.STAGES]
    records += [memory_record('after-worker-close')]
    write_memory(tmp_path, records)
    assert owner._validate_memory(records, tmp_path)
    reordered = [dict(item) for item in records]
    reordered[2], reordered[3] = reordered[3], reordered[2]
    write_memory(tmp_path, reordered)
    with pytest.raises(RuntimeError):
        owner._validate_memory(reordered, tmp_path)
    invalid = [dict(item) for item in records]
    invalid[2]['allocated_bytes'] = True
    write_memory(tmp_path, invalid)
    with pytest.raises(RuntimeError):
        owner._validate_memory(invalid, tmp_path)
    bad_headroom = [dict(item) for item in records]
    bad_headroom[0]['device_free_bytes'] = 4 * owner.GIB
    write_memory(tmp_path, bad_headroom)
    with pytest.raises(RuntimeError):
        owner._validate_memory(bad_headroom, tmp_path)


def test_controller_disk_guard_rejects_low_space(monkeypatch):
    class Usage:
        free = owner.MIN_DISK - 1
    monkeypatch.setattr(owner.shutil, 'disk_usage', lambda _path: Usage())
    with pytest.raises(RuntimeError, match='disk'):
        owner._disk_guard(Path('.'))


def test_external_storage_binding_detects_replacement_and_volume_limits(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    parent = tmp_path / 'evidence'
    cache = tmp_path / 'cache'
    temp = tmp_path / 'temp'
    for path in (workspace, parent, cache, temp):
        path.mkdir()
    run = parent / 'run'
    run.mkdir()
    monkeypatch.setenv('TEMP', str(temp))
    monkeypatch.setenv('TMP', str(temp))
    monkeypatch.setattr(owner.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=3 * owner.GIB))
    binding = owner._storage_snapshot(workspace, run, cache)
    raw = owner.g._canonical(binding)
    (run / 'storage-binding.json').write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert owner._storage_guard(workspace, run, cache, digest)['workspace_free_bytes'] == 3 * owner.GIB
    with pytest.raises(RuntimeError, match='binding bytes'):
        owner._storage_guard(workspace, run, cache, '0' * 64)
    monkeypatch.setattr(owner.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=owner.MIN_WORKSPACE_DISK - 1
                                                     if Path(path) == workspace else 3 * owner.GIB))
    with pytest.raises(RuntimeError, match='workspace residual'):
        owner._storage_guard(workspace, run, cache, digest)
    monkeypatch.setattr(owner.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=3 * owner.GIB))
    temp.rename(tmp_path / 'old-temp')
    temp.mkdir()
    with pytest.raises(RuntimeError, match='identity changed'):
        owner._storage_guard(workspace, run, cache, digest)


def test_storage_failure_precedes_any_model_or_cuda_work(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'experiments/short_cpu_service_reference_inputs_v1.json').read_bytes())
    monkeypatch.setattr(owner, '_read_inputs', lambda *_: (manifest, {}))
    monkeypatch.setattr(owner.g, '_configure', lambda *_: pytest.fail('CUDA setup was reached'))
    def reject():
        raise RuntimeError('fabricated storage identity change')
    with pytest.raises(RuntimeError, match='storage identity'):
        owner._owner_run(None, None, run_dir=tmp_path, cache=tmp_path, inventory=tmp_path,
                         bucket=16, driver='fake', sources={}, _worker_device='cpu', _check_storage=reject)


def test_storage_rejects_volume_mismatch_and_replaced_output(tmp_path, monkeypatch):
    workspace, parent, cache, temp = (tmp_path / name for name in ('workspace', 'output', 'cache', 'temp'))
    for path in (workspace, parent, cache, temp):
        path.mkdir()
    run = parent / 'run'
    run.mkdir()
    monkeypatch.setenv('TEMP', str(temp))
    monkeypatch.setenv('TMP', str(temp))
    original = owner._plain_directory
    def different_cache(path):
        value = original(path)
        if Path(path) == cache:
            value['device'] += 1
        return value
    monkeypatch.setattr(owner, '_plain_directory', different_cache)
    with pytest.raises(ValueError, match='share a volume'):
        owner._storage_base(workspace, parent, cache)
    monkeypatch.setattr(owner, '_plain_directory', original)
    monkeypatch.setattr(owner.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=3 * owner.GIB))
    raw = owner.g._canonical(owner._storage_snapshot(workspace, run, cache))
    digest = hashlib.sha256(raw).hexdigest()
    (run / 'storage-binding.json').write_bytes(raw)
    run.rename(parent / 'old-run')
    run.mkdir()
    (run / 'storage-binding.json').write_bytes(raw)
    with pytest.raises(RuntimeError, match='identity changed'):
        owner._storage_guard(workspace, run, cache, digest)


def test_candidate_requires_clean_exit_matching_input_source_and_memory(tmp_path, monkeypatch):
    from looking_glass.pretrained import ASSETS, MODEL_ID, REVISION
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'experiments/short_cpu_service_reference_inputs_v1.json').read_bytes())
    cases = owner._cases(manifest, 16)
    memory = [memory_record('before-model-load'), memory_record('after-model-load')]
    memory += [memory_record(stage) for stage in owner.STAGES]
    memory += [memory_record('after-worker-close')]
    write_memory(tmp_path, memory)
    runtime = {'physical_gpu_uuid': 'GPU-fake', 'driver': '123.45', 'total_bytes': 8 * owner.GIB}
    (tmp_path / 'private-runtime.json').write_text(json.dumps(runtime), encoding='utf-8')
    monkeypatch.setattr(owner.g, '_validate_runtime', lambda *_: None)
    monkeypatch.setattr(owner.g, '_public_runtime', lambda value: value)
    monkeypatch.setattr(owner, 'verify_payloads', lambda *_args, **_kwargs:
                        {'reference_separation': {}, 'comparisons': []})
    routing = {'status': 'passed', 'bucket': 16, 'vocab': 50257, 'broker_jobs': 4,
               'reference_separation': {}, 'comparisons': [],
               'full_logit_sha256': {name: 'a' * 64 for name in owner.NAMES}}
    source = {'source.py': 'b' * 64}
    monkeypatch.setattr(owner.g, '_verify_child_sources', lambda *_: source)
    (tmp_path / 'owner.json').write_text(json.dumps({'run_id': 'run', 'supervisor_run_id': 'supervisor',
                 'pid': 123, 'source_manifest_sha256': 'd' * 64}), encoding='utf-8')
    entry_cases = (cases[0], cases[1], cases[1], cases[0])
    entries = [{'entry': index, 'shape': [1, 16], 'dtype': 'torch.int64', 'worker_device': 'cuda:0',
                'case_id': case['case_id'], 'input_ids': case['input_ids'],
                'attention_mask': case['attention_mask']} for index, case in enumerate(entry_cases, 1)]
    for entry in entries:
        (tmp_path / f"forward-{entry['entry']:03d}-entry.json").write_text(json.dumps(entry), encoding='utf-8')
    cleanup = {'before': {'allocated_bytes': 0, 'reserved_bytes': 1024},
               'after': {'allocated_bytes': 0, 'reserved_bytes': 0},
               'api': 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces',
               'model_released': True, 'input_output_tensors_released': True,
               'host_claims_after': 0, 'accounting_leases_after': 0}
    (tmp_path / 'cleanup.json').write_text(json.dumps(cleanup), encoding='utf-8')
    result = {'status': 'passed', 'worker_device': 'cuda:0', 'bucket': 16,
              'cases': json.loads(json.dumps(cases)),
              'model_entries': entries,
              'source_sha256': source, 'manifest_sha256': owner.MANIFEST_SHA256,
              'inventory_sha256': owner.INVENTORY_SHA256, 'routing': routing,
              'host_available_bytes': 4 * owner.GIB, 'host_before_load_bytes': 4 * owner.GIB,
              'host_claim_bytes': owner.HOST_CLAIM, 'runtime_redacted': runtime, 'memory': memory, 'cleanup': cleanup,
              'model_metadata': {'model_id': MODEL_ID, 'model_revision': REVISION,
                                 'parameter_count': 1,
                                 'weights_sha256': ASSETS['model.safetensors'][1],
                                 'asset_sha256': {name: digest for name, (_, digest) in ASSETS.items()}}}
    candidate = {'schema': 'short-service-gpu-candidate-v1', 'run_id': 'run',
                 'supervisor_run_id': 'supervisor', 'pid': 123, 'source_commit': 'c' * 40,
                 'source_manifest_sha256': 'd' * 64, 'result': result}
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps(candidate), encoding='utf-8')
    receipt = SimpleNamespace(run_id='supervisor', pid=123, exit_code=0, clean=True, timed_out=False)
    kwargs = {'run_id': 'run', 'commit': 'c' * 40, 'supervised': receipt, 'bucket': 16,
              'source_manifest_sha': 'd' * 64, 'sources': source, 'expected_cases': cases,
              'expected_gpu_uuid': 'GPU-fake', 'expected_driver': '123.45'}
    assert owner._promote(tmp_path, **kwargs)['bucket'] == 16
    receipt.exit_code = False
    with pytest.raises(ValueError):
        owner._promote(tmp_path, **kwargs)
    receipt.exit_code = 0
    candidate['result']['cases'][1]['attention_mask'][-1] = 1
    path.write_text(json.dumps(candidate), encoding='utf-8')
    with pytest.raises(RuntimeError, match='identity'):
        owner._promote(tmp_path, **kwargs)
    (tmp_path / 'failure.json').write_text('{}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='failure'):
        owner._promote(tmp_path, **kwargs)


def test_isolated_project_import_does_not_import_model_dependencies(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    site = Path(torch.__file__).resolve().parent.parent
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    command = [getattr(sys, '_base_executable', None) or sys.executable, '-I', '-S', '-B',
               '-X', 'pycache_prefix=' + str(prefix), str(repo / 'scripts/short_service_gpu_bootstrap.py'),
               '--site-packages', str(site), '--check-project']
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'status': 'ok', 'model_dependencies_loaded': False}
    assert not list(prefix.iterdir())


@pytest.mark.parametrize('failure', ['none', 'recorder', 'input', 'close', 'cleanup', 'claim'])
def test_fabricated_owner_graph_success_and_failure_custody(tmp_path, monkeypatch, failure):
    """Use the real serial worker and broker with fake telemetry and a CPU model."""
    from looking_glass import pretrained
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'experiments/short_cpu_service_reference_inputs_v1.json').read_bytes())
    monkeypatch.setattr(owner, '_read_inputs', lambda *_: (manifest, {}))
    monkeypatch.setattr(owner, '_disk_guard', lambda *_: [owner.MIN_DISK])
    monkeypatch.setattr(owner, '_available_host', lambda: 4 * owner.GIB)
    monkeypatch.setattr(owner.g, '_ROOTS', [None] * 25)
    monkeypatch.setattr(owner, '_HOLD', [None] * 16)
    monkeypatch.setattr(owner.g, '_CUDA_RESULTS', [None] * 8)
    monkeypatch.setattr(owner.g, '_configure', lambda *_: None)
    monkeypatch.setattr(owner.g, '_bind_device', lambda *_: SimpleNamespace(total_memory=8 * owner.GIB))
    monkeypatch.setattr(owner.g, '_live_runtime', lambda *_: {'test': True})
    monkeypatch.setattr(owner.g, '_validate_runtime', lambda *_: None)
    monkeypatch.setattr(owner.g, '_public_runtime', lambda value: value)
    monkeypatch.setattr(owner.g, '_assert_no_child_processes', lambda: None)
    def fake_cuda(_session, _lease, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except BaseException as exc:
            owner.g._ROOTS[15] = exc
            if owner.g._ROOTS[20] is not None:
                owner.g._ROOTS[20].mark_unsafe()
            raise
    monkeypatch.setattr(owner.g, '_cuda', fake_cuda)
    def fake_convert(_session, _lease, model, *, device, dtype):
        model.to(device=device, dtype=dtype)
        owner.g._ROOTS[16] = model
    monkeypatch.setattr(owner.g, '_convert_model', fake_convert)
    def fake_cleanup(*_):
        if failure == 'cleanup':
            raise RuntimeError('fabricated cleanup failure')
        return {'before': {'allocated_bytes': 0, 'reserved_bytes': 0},
                'after': {'allocated_bytes': 0, 'reserved_bytes': 0},
                'api': 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces'}
    monkeypatch.setattr(owner.g, '_guarded_cleanup', fake_cleanup)
    monkeypatch.setattr(owner.OwnerSession, 'check', lambda *_: None)
    monkeypatch.setattr(owner.gc, 'isenabled', lambda: False)
    for name, fn in {'is_available': lambda: True, 'get_allocator_backend': lambda: 'native',
                     'set_per_process_memory_fraction': lambda *_: None,
                     'get_per_process_memory_fraction': lambda *_: .25,
                     'manual_seed': lambda *_: None, 'synchronize': lambda *_: None,
                     'reset_peak_memory_stats': lambda *_: None}.items():
        monkeypatch.setattr(torch.cuda, name, fn)
    monkeypatch.setattr(owner.g, '_snapshot', lambda *_: {
        'device_free_bytes': 6 * owner.GIB, 'device_total_bytes': 8 * owner.GIB,
        'allocated_bytes': 0, 'reserved_bytes': 0, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0})

    calls = []
    model_ref = [None]
    bundle_ref = [None]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                          vocab_size=50257, n_positions=1024)

        def get_input_embeddings(self):
            return SimpleNamespace(weight=self.weight)

        def get_output_embeddings(self):
            return SimpleNamespace(weight=self.weight)

        def forward(self, input_ids, attention_mask):
            calls.append(1)
            values = torch.arange(50257, dtype=torch.float32).view(1, 1, -1) / 1000
            return {'logits': input_ids.unsqueeze(-1).to(torch.float32) + values}

    class Bundle:
        pass
    def fake_load(**_):
        model = Model().eval()
        bundle = Bundle()
        bundle.model = model
        bundle.metadata = {'asset_sha256': {'fake': 'abc'}, 'parameter_count': 1}
        model_ref[0], bundle_ref[0] = weakref.ref(model), weakref.ref(bundle)
        return bundle
    monkeypatch.setattr(pretrained, 'load_distilgpt2', fake_load)
    saved = {}
    def recorder(_directory, _cache, name, raw, _bucket):
        if failure == 'recorder' and saved:
            raise OSError('fabricated recorder failure')
        saved[name] = len(raw)
    monkeypatch.setattr(owner, '_write_tensor', recorder)
    if failure == 'input':
        from looking_glass.resident_worker import ResidentWorker
        original = ResidentWorker._perform_with_custody
        def mutate_before_device(self, request, custody=None):
            request.inputs['input_ids'][0, 0] = 7
            return original(self, request, custody)
        monkeypatch.setattr(ResidentWorker, '_perform_with_custody', mutate_before_device)
    if failure == 'close':
        from looking_glass.resident_worker import ResidentWorker
        def fail_close(_self):
            raise RuntimeError('fabricated resident close failure')
        monkeypatch.setattr(ResidentWorker, 'close', fail_close)
    if failure == 'claim':
        from looking_glass.memory_budget import MemoryLease
        original_release = MemoryLease.release
        release_calls = []
        def fail_second_claim(self):
            release_calls.append(self)
            if len(release_calls) == 2:
                raise RuntimeError('fabricated claim release failure')
            return original_release(self)
        monkeypatch.setattr(MemoryLease, 'release', fail_second_claim)
    session, lease = object(), object()
    owner.g._ROOTS[0], owner.g._ROOTS[1] = session, lease
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        if failure == 'recorder':
            with pytest.raises(OSError, match='recorder failure'):
                owner._owner_run(session, lease, run_dir=tmp_path, cache=tmp_path,
                                 inventory=tmp_path / 'unused', bucket=16, driver='fake', sources={},
                                 _worker_device='cpu')
        elif failure == 'input':
            with pytest.raises(Exception):
                owner._owner_run(session, lease, run_dir=tmp_path, cache=tmp_path,
                                 inventory=tmp_path / 'unused', bucket=16, driver='fake', sources={},
                                 _worker_device='cpu')
        elif failure in ('close', 'cleanup', 'claim'):
            with pytest.raises(RuntimeError, match='fabricated'):
                owner._owner_run(session, lease, run_dir=tmp_path, cache=tmp_path,
                                 inventory=tmp_path / 'unused', bucket=16, driver='fake', sources={},
                                 _worker_device='cpu')
        else:
            result = owner._owner_run(session, lease, run_dir=tmp_path, cache=tmp_path,
                                      inventory=tmp_path / 'unused', bucket=16, driver='fake', sources={},
                                      _worker_device='cpu')
    finally:
        torch.set_num_threads(old_threads)
    if failure == 'recorder':
        assert len(calls) == 2 and list(saved) == ['reference-alpha']
        assert owner.g._ROOTS[20].unsafe is True
        assert len(list(tmp_path.glob('forward-*-entry.json'))) == 2
    elif failure == 'input':
        assert not calls and not saved
        assert owner.g._ROOTS[20].unsafe is True
        assert not list(tmp_path.glob('forward-*-entry.json'))
        invalid = json.loads((tmp_path / 'forward-001-entry-invalid.json').read_text(encoding='utf-8'))
        assert invalid['reason'] == 'input-or-mask-mismatch' and invalid['input_ids'][0] == 7
    elif failure in ('close', 'cleanup', 'claim'):
        assert len(calls) == 4 and set(saved) == set(owner.NAMES)
        assert owner.g._ROOTS[20].unsafe is True
        assert not (tmp_path / 'cleanup.json').exists()
    else:
        assert len(calls) == 4 and set(saved) == set(owner.NAMES)
        assert result['status'] == 'passed' and result['worker_device'] == 'cpu'
        assert len(result['model_entries']) == 4
        assert [item['stage'] for item in result['memory'] if item['stage'] in owner.STAGES] == list(owner.STAGES)
        assert owner.g._ROOTS[20].unsafe is False and model_ref[0]() is None
    assert owner._HOLD[5] is not None
    if failure == 'none':
        assert owner.g._ROOTS[4] is None and owner.g._ROOTS[3].released and owner.g._ROOTS[19].released
    elif failure not in ('cleanup', 'claim'):
        assert owner.g._ROOTS[4] is bundle_ref[0]() and model_ref[0]() is not None
        assert not owner.g._ROOTS[3].released and not owner.g._ROOTS[19].released
    elif failure == 'cleanup':
        assert not owner.g._ROOTS[3].released and not owner.g._ROOTS[19].released
    else:
        assert owner.g._ROOTS[3].released and not owner.g._ROOTS[19].released
