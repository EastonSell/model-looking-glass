"""Exercise actual broker retirement before a subsequent worker checkpoint."""
import os
from pathlib import Path
import subprocess
import sys


def test_no_checkpoint_overtakes_a_real_broker_retirement():
    root = Path(__file__).resolve().parents[1]
    code = r'''import sys,threading
sys.path.insert(0,sys.argv[1])
from transformers import GPT2Config,GPT2LMHeadModel
from looking_glass.memory_budget import CudaMemorySnapshot,MemoryBudget
from looking_glass.worker_failure_experiment import WorkerCustody,run_worker_probe
config=GPT2Config(vocab_size=64,n_positions=8,n_ctx=8,n_embd=16,n_layer=1,n_head=2,use_cache=False)
config._attn_implementation='eager'
model=GPT2LMHeadModel(config).eval()
budget=MemoryBudget(lambda:CudaMemorySnapshot(3*1024**3,4*1024**3,0,0),max_gpu_trace_bytes=0,max_host_trace_bytes=0)
resident=budget.acquire(32*1024**2,max_gpu_trace_bytes=0,max_host_trace_bytes=0)
custody=WorkerCustody()
entered=threading.Event()
release=threading.Event()
overtook=threading.Event()
checkpoints=[]
errors=[]
def checkpoint(stage):
    checkpoints.append(stage)
    if stage=='after_baseline': overtook.set()
    if stage=='before_worker':
        original=custody.broker._retire_job
        def delayed(job):
            entered.set()
            if not release.wait(10): raise RuntimeError('test retirement timeout')
            return original(job)
        custody.broker._retire_job=delayed
    return {'stage':stage}
def run():
    try: run_worker_probe(model,budget=budget,resident_lease=resident,device='cpu',
        guard=lambda:None,checkpoint=checkpoint,record=lambda stage,value:None,custody=custody)
    except BaseException as error: errors.append(error)
thread=threading.Thread(target=run)
thread.start()
if not entered.wait(20):
    release.set()
    raise AssertionError('real broker did not reach retirement')
premature=overtook.wait(1.0)
release.set()
thread.join(60)
if thread.is_alive() or errors: raise AssertionError('worker did not complete: '+repr(errors))
if premature: raise AssertionError('checkpoint overtook broker retirement')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    child = subprocess.run([sys.executable, '-B', '-c', code, str(root)], capture_output=True, text=True,
                           env=env, timeout=90)
    assert child.returncode == 0, child.stderr


def test_retirement_timeout_marks_owner_unsafe_before_any_guard():
    import threading
    from types import MethodType, SimpleNamespace
    import pytest
    from looking_glass.broker import Broker
    from looking_glass.owner_completion import OwnerCompletionAuthority
    from looking_glass.worker_failure_experiment import WorkerCustody, _await_broker_retirement
    root = WorkerCustody()
    root.authority = OwnerCompletionAuthority()
    broker = SimpleNamespace(_condition=threading.Condition(), quarantined=False,
        _fatal_cleanup_failure=False, _closed=False, _unavailable=None,
        _thread=SimpleNamespace(is_alive=lambda: True), _running=None, _preparing=0,
        _pending_slots=0, _fair_preparations=set(), _fair_handoff=None, _queue=[], _job_roots=[object()],
        _completion_authority=root.authority)
    broker._wait_condition_locked = MethodType(Broker._wait_condition_locked, broker)
    root.broker = broker
    calls = []
    with pytest.raises(TimeoutError):
        _await_broker_retirement(root, lambda: calls.append('guard'), timeout=.01)
    assert root.authority.unsafe is True
    assert calls == []
