"""CPU-only contracts for the bounded, fail-stop GPU exercise."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

from looking_glass.worker_failure_gpu import (_DeviceCalls, _snapshot, _write_new, _check_disk,
                                               _check_claims, _device_evidence, owner_main)


def test_device_self_report_binds_cap_readback_and_runtime_versions():
    total, cap = 8 * 1024**3, 512 * 1024**2
    value = _device_evidence('a' * 32, total, cap, cap / total, '2.8.0+cu128', '12.8', '4.57.3')
    assert value['run_id'] == 'a' * 32 and value['runtime_uuid_verified'] is True
    assert value['cuda_total_bytes'] == total and value['allocator_cap_bytes'] == cap
    assert value['allocator_fraction_readback'] == 0.0625
    assert value['torch_version'] == '2.8.0+cu128' and value['cuda_build'] == '12.8'
    assert value['transformers_version'] == '4.57.3'
    with pytest.raises(ValueError):
        _device_evidence('a' * 32, total, cap, .5, '2.8.0+cu128', '12.8', '4.57.3')


def test_disk_preflight_rejects_below_two_gib_before_any_output(tmp_path):
    with pytest.raises(RuntimeError, match='disk'):
        _check_disk(tmp_path, usage=lambda path: SimpleNamespace(free=2 * 1024**3 - 1))
    assert list(tmp_path.iterdir()) == []
    assert _check_disk(tmp_path, usage=lambda path: SimpleNamespace(free=2 * 1024**3)) == 2 * 1024**3


def test_guard_rejects_replaced_or_released_trace_claim():
    resident, trace = object(), object()
    ledger = SimpleNamespace(active_leases=lambda: (resident, trace))
    _check_claims(ledger, resident, trace)
    with pytest.raises(RuntimeError, match='claim'):
        _check_claims(ledger, resident, object())
    with pytest.raises(RuntimeError, match='claim'):
        _check_claims(SimpleNamespace(active_leases=lambda: (resident,)), resident, trace)


def test_guard_rejects_undercharged_trace_even_when_lease_is_active():
    from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget
    ledger = MemoryBudget(lambda: CudaMemorySnapshot(3 * 1024**3, 4 * 1024**3, 0, 0),
                          max_gpu_trace_bytes=64, max_host_trace_bytes=0)
    resident = ledger.acquire(32 * 1024**2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    trace = ledger.acquire(16 * 1024**2, max_gpu_trace_bytes=64, max_host_trace_bytes=0)
    with pytest.raises(RuntimeError, match='trace'):
        _check_claims(ledger, resident, trace, gpu_trace_bytes=64)
    trace.reserve_trace('gpu', 64)
    _check_claims(ledger, resident, trace, gpu_trace_bytes=64)
    trace.release_trace('gpu', 64)
    with pytest.raises(RuntimeError, match='trace'):
        _check_claims(ledger, resident, trace, gpu_trace_bytes=64)
    _check_claims(ledger, resident, trace, gpu_trace_bytes=0)


def test_owner_loss_after_device_return_roots_result_and_stops_next_call():
    class Resource:
        pass
    weak = []
    checks = 0
    def guard():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError('owner lost after device return')
    def operation():
        value = Resource()
        weak.append(weakref.ref(value))
        return value
    calls = _DeviceCalls(guard)
    with pytest.raises(RuntimeError, match='owner lost'):
        calls.call(operation)
    assert calls.unsafe is True and calls.pending is True
    assert weak[0]() is calls.return_root
    with pytest.raises(RuntimeError):
        calls.call(lambda: pytest.fail('must not issue another operation'))


def test_snapshot_stops_after_bad_allocated_counter_before_reserved_query():
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda index: calls.append('sync'),
        mem_get_info=lambda index: (calls.append('memory') or (3000, 4000)),
        memory_allocated=lambda index: (calls.append('allocated') or -1),
        memory_reserved=lambda index: (calls.append('reserved') or 0),
    )
    with pytest.raises(ValueError):
        _snapshot(lambda: None, SimpleNamespace(cuda=cuda), 4000)
    assert calls == ['sync', 'memory', 'allocated']


def test_owner_revocation_stops_before_next_cuda_query():
    calls = []
    def guard():
        if calls and calls[-1] == 'memory':
            raise RuntimeError('owner lost')
    cuda = SimpleNamespace(
        synchronize=lambda index: calls.append('sync'),
        mem_get_info=lambda index: (calls.append('memory') or (3000, 4000)),
        memory_allocated=lambda index: (calls.append('allocated') or 0),
        memory_reserved=lambda index: (calls.append('reserved') or 0),
    )
    with pytest.raises(RuntimeError, match='owner lost'):
        _snapshot(guard, SimpleNamespace(cuda=cuda), 4000)
    assert calls == ['sync', 'memory']


def test_private_record_is_bounded_create_only(tmp_path):
    target = tmp_path / 'record.json'
    digest = _write_new(target, {'status': 'incomplete'})
    assert len(digest) == 64 and json.loads(target.read_text()) == {'status': 'incomplete'}
    with pytest.raises(FileExistsError):
        _write_new(target, {'status': 'changed'})
    with pytest.raises(ValueError):
        _write_new(tmp_path / 'large.json', {'value': 'x' * 70000})
    assert not (tmp_path / 'large.json').exists()


def test_disabled_owner_never_imports_cuda_dependencies(tmp_path, monkeypatch):
    import looking_glass.worker_failure_gpu as owner
    monkeypatch.setattr(owner, '_ENABLED', False)
    assert owner_main(str(tmp_path), str(tmp_path), '0' * 32, '0' * 40, 'unused') == 1
    assert not list(tmp_path.iterdir())


def test_owner_exception_keeps_failure_root_and_writes_only_incomplete_record(tmp_path, monkeypatch):
    import looking_glass.worker_failure_gpu as owner
    state = {'entered': False, 'directory': tmp_path, 'stage': 'trace',
             'failure': None, 'diagnostic': None}
    monkeypatch.setattr(owner, '_ENABLED', True)
    monkeypatch.setattr(owner, '_CUSTODY', state)
    def fail(*args):
        raise OSError('sensitive private detail')
    monkeypatch.setattr(owner, '_execute_owner', fail, raising=False)
    assert owner.owner_main(str(tmp_path), str(tmp_path), '0' * 32, '0' * 40, 'unused') == 1
    assert isinstance(state['failure'], OSError)
    data = json.loads((tmp_path / 'failure.json').read_text())
    assert data == {'schema': 'worker-failure-owner-incomplete-v1', 'status': 'incomplete',
                    'stage': 'trace', 'error_type': 'OSError', 'clean_attested': False}
    assert 'sensitive' not in (tmp_path / 'failure.json').read_text()
    assert not (tmp_path / 'terminal.json').exists()


def test_pinned_cpu_imports_work_with_python_subprocess_creation_denied():
    root = Path(__file__).resolve().parents[1]
    site = root.parents[1] / '.venv/Lib/site-packages'
    code = ("import importlib.util,subprocess,sys; "
            "spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[3]); "
            "bootstrap=importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap); "
            "bootstrap._prime_windows_platform(); "
            "sys.path.insert(0,sys.argv[1]); sys.path.append(sys.argv[2]); "
            "from looking_glass.worker_failure_gpu import _ForbidChildProcess; "
            "subprocess.Popen=_ForbidChildProcess; import torch,transformers; "
            "from transformers import GPT2Config,GPT2LMHeadModel; "
            "from looking_glass.worker_failure_gpu import _forbid_collection; import gc; "
            "gc.disable(); gc.collect=_forbid_collection; "
            "model=GPT2LMHeadModel(GPT2Config(vocab_size=64,n_positions=8,n_ctx=8,n_embd=16,n_layer=1,n_head=2,use_cache=False)).eval(); "
            "assert torch.cuda.is_initialized() is False; "
            "assert str(torch.__version__) == '2.8.0+cu128'; "
            "assert str(transformers.__version__) == '4.57.3'")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    uname = {'system': 'Windows', 'node': 'cpu-test', 'release': '11', 'version': '10.0.26200',
             'machine': 'AMD64', 'processor': 'cpu-test'}
    env['LOOKING_GLASS_FAILURE_VALIDATION_PLATFORM_UNAME'] = json.dumps(uname, sort_keys=True, separators=(',', ':'))
    child = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code, str(root), str(site),
                            str(root / 'scripts/worker_failure_gpu_bootstrap.py')],
                           capture_output=True, text=True, env=env, timeout=45)
    assert child.returncode == 0, child.stderr


def test_actual_trace_then_worker_sequence_on_tiny_cpu_model():
    root = Path(__file__).resolve().parents[1]
    code = r'''import gc,json,sys
sys.path.insert(0,sys.argv[1])
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from looking_glass.memory_budget import CudaMemorySnapshot,MemoryBudget
from looking_glass.worker_failure_gpu import _exercise_sequence
config=GPT2Config(vocab_size=64,n_positions=8,n_ctx=8,n_embd=16,n_layer=1,n_head=2,use_cache=False)
config._attn_implementation='eager'
model=GPT2LMHeadModel(config).eval()
budget=MemoryBudget(lambda: CudaMemorySnapshot(3*1024**3,4*1024**3,0,0),max_gpu_trace_bytes=0,max_host_trace_bytes=64)
resident=budget.acquire(32*1024**2,max_gpu_trace_bytes=0,max_host_trace_bytes=0)
state={'traces':[],'current_trace':None,'worker':None,'stage':None}
records=[]
def guard():
    trace=state['current_trace']
    worker=state['worker']
    temporary=trace.lease if trace is not None else (None if worker is None else worker.workspace)
    expected={resident} if temporary is None else {resident,temporary}
    if set(budget.active_leases()) != expected: raise RuntimeError('claim mismatch')
    if trace is not None and temporary is not None and temporary.trace_bytes('host') != trace.charge_expected:
        raise RuntimeError('trace charge mismatch')
gc.disable()
result=_exercise_sequence(model=model,budget=budget,resident_lease=resident,device='cpu',
    guard=guard,synchronize=lambda: None,snapshot=lambda:{'cpu_only':True},
    record=lambda stage,value: records.append(stage),state=state)
print(json.dumps({'records':records,'report':result,'claims':len(budget.active_leases())}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    child = subprocess.run([sys.executable, '-c', code, str(root)], capture_output=True, text=True,
                           env=env, timeout=60)
    assert child.returncode == 0, child.stderr
    report = json.loads(child.stdout)
    assert report['records'] == ['trace_0', 'after_trace_0', 'trace_1', 'after_trace_1',
        'trace_2', 'after_trace_2', 'before_worker', 'baseline', 'after_baseline',
        'cancellation', 'after_cancellation', 'post_cancel', 'after_post_cancel',
        'before_simulated_oom', 'simulated_oom']
    assert report['report']['simulated_oom']['worker_unsafe'] is True
    assert report['report']['simulated_oom']['subsequent_work_refused'] is True
    assert report['claims'] == 2
