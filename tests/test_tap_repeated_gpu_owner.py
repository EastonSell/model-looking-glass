"""CPU-only tests for the disabled fixed owner orchestration."""

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
import weakref

import pytest

from looking_glass import tap_repeated_gpu_owner as subject
from looking_glass import tap_repeated_protocol as protocol


def test_import_and_disabled_entry_do_not_load_torch_or_touch_device(tmp_path):
    source = ("import sys; from looking_glass import tap_repeated_gpu_owner as o; "
              "assert 'torch' not in sys.modules; "
              "assert o.owner_main('x','y','z','c','g') == 1; "
              "assert 'torch' not in sys.modules")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    result = subprocess.run([sys.executable, '-c', source], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_journal_is_bounded_canonical_durable_and_single_use(tmp_path):
    path = tmp_path / 'journal.jsonl'
    journal = subject._Journal(path)
    for index in range(86):
        journal.append({'z': index, 'a': True})
    digest = journal.close()
    raw = path.read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest()
    assert len(raw.splitlines()) == 86 and raw.startswith(b'{"a":true,"z":0}\n')
    with pytest.raises(RuntimeError):
        journal.append({'a': 1})
    with pytest.raises(FileExistsError):
        subject._Journal(path)


def test_assigned_cuda_return_remains_rooted_if_postcheck_fails(monkeypatch):
    from looking_glass.supervisor import OwnerSession
    calls = []
    def check(_session, _lease):
        calls.append('check')
        if len(calls) == 2:
            raise RuntimeError('owner lost')
    monkeypatch.setattr(OwnerSession, 'check', check)
    custody = dict(subject._CUSTODY, pending_return=None)
    monkeypatch.setattr(subject, '_CUSTODY', custody)
    class Result:
        pass
    value = Result()
    ref = weakref.ref(value)
    with pytest.raises(RuntimeError, match='owner lost') as caught:
        subject._device_call(object(), object(), lambda: value)
    caught.value.__traceback__ = None
    del caught, value
    assert ref() is custody['pending_return']


def test_initial_snapshot_rejects_bad_values_before_later_queries(monkeypatch):
    calls = []
    class Cuda:
        def mem_get_info(self, _device): calls.append('free'); return (1, 3)
        def memory_allocated(self, _device): calls.append('allocated'); return 0
        def memory_reserved(self, _device): calls.append('reserved'); return 0
    monkeypatch.setattr(subject, '_device_call', lambda _s, _l, fn, *args: fn(*args))
    with pytest.raises(ValueError, match='free/total'):
        subject._initial_snapshot(types.SimpleNamespace(cuda=Cuda()), None, None,
                                  8 * protocol.GIB, 2 * protocol.GIB)
    assert calls == ['free']


def test_runtime_record_requires_exact_boolean_and_integer_types():
    expected = subject._expected_runtime()
    assert subject._exact_mapping(dict(expected), expected)
    bad = dict(expected, use_cache=0)
    assert not subject._exact_mapping(bad, expected)
    bad = dict(expected, cpu_threads=True)
    assert not subject._exact_mapping(bad, expected)


def test_failure_projection_keeps_only_bounded_safe_partial_values(monkeypatch):
    custody = dict(subject._CUSTODY, stage='samples',
        journal=types.SimpleNamespace(count=4, closed=False),
        lean=types.SimpleNamespace(partial={'stage': 'running', 'mode': 'direct', 'started_ns': 123,
            'completion_ns': True, 'device_event_ms': float('nan'), 'private': 'PRIVATE-PATH-SECRET',
            'memory_before': {'free_bytes': 10, 'total_bytes': 20, 'allocated_bytes': 2, 'reserved_bytes': 3}}))
    monkeypatch.setattr(subject, '_CUSTODY', custody)
    record = subject._failure_record(RuntimeError('PRIVATE-ERROR'))
    assert record['journal_rows_fsynced_before_failure'] == 4
    assert record['partial']['started_ns'] == 123 and 'completion_ns' not in record['partial']
    assert 'device_event_ms' not in record['partial'] and 'PRIVATE' not in json.dumps(record)
    assert record['partial']['memory_before']['free_bytes'] == 10


def test_cache_is_bound_to_the_shared_offline_workspace(monkeypatch, tmp_path):
    cache = tmp_path / 'work' / 'hf-cache' / 'hub'
    cache.mkdir(parents=True)
    session = types.SimpleNamespace(identity=types.SimpleNamespace(workspace_root=tmp_path))
    monkeypatch.setenv('HF_HOME', str(cache.parent))
    for name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE'):
        monkeypatch.delenv(name, raising=False)
    subject._validate_cache(session, cache.resolve())
    monkeypatch.setenv('HF_HOME', os.path.relpath(cache.parent))
    with pytest.raises(RuntimeError, match='home'):
        subject._validate_cache(session, cache.resolve())
    monkeypatch.setenv('HF_HOME', str(cache.parent))
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path))
    with pytest.raises(RuntimeError, match='alternate'):
        subject._validate_cache(session, cache.resolve())


def test_cache_rejects_reparse_components(monkeypatch, tmp_path):
    cache = tmp_path / 'work' / 'hf-cache' / 'hub'
    cache.mkdir(parents=True)
    session = types.SimpleNamespace(identity=types.SimpleNamespace(workspace_root=tmp_path))
    monkeypatch.setenv('HF_HOME', str(cache.parent))
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == cache.parent:
            return types.SimpleNamespace(st_mode=result.st_mode,
                st_file_attributes=getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))
        return result
    monkeypatch.setattr(Path, 'lstat', lstat)
    with pytest.raises(RuntimeError, match='indirect'):
        subject._validate_cache(session, cache)


def _synthetic_execution(monkeypatch, tmp_path, *, fail_second_sample, version_error=None):
    """Run the actual sequence using fakes; no device or real supervisor."""
    from looking_glass import (device_lease, gpu_oneshot, supervisor, tap_repeated_admission,
        tap_repeated_child_barrier, tap_repeated_cuda_backend, tap_repeated_disposal,
        tap_repeated_lean, tap_repeated_model, tap_repeated_owner, tap_repeated_preflight,
        tap_repeated_timing)

    fixture_path = Path(__file__).with_name('test_tap_repeated_protocol.py')
    spec = importlib.util.spec_from_file_location('_gpu_owner_protocol_fixture', fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    schedule = protocol.load_schedule(fixture.SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    rows = fixture._attempt(schedule, 0, binding)['journal']
    calls = []
    total, cap = 8 * protocol.GIB, 2 * protocol.GIB
    memory = {'free_bytes': 6 * protocol.GIB, 'total_bytes': total,
              'allocated_bytes': 128 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}

    class FakeLease:
        @staticmethod
        def acquire(_path, timeout): calls.append('lease'); return FakeLease()
        def trace_bytes(self, _location): return 0

    class Session:
        run_id = 'b' * 32
        device_lease_path = tmp_path / 'device.lock'
        identity = types.SimpleNamespace(gpu_uuid='GPU-00000000-0000-0000-0000-000000000001')
        @staticmethod
        def from_environment(): calls.append('session'); return Session()
        @staticmethod
        def check(_session, _lease): calls.append('owner_check')
        @staticmethod
        def verify_runtime_binding(_session, _lease, **_kwargs): calls.append('bind_runtime')
        def send_startup(self, _lease, _payload):
            assert (tmp_path / 'journal.jsonl').read_bytes() == b''
            calls.append('ready')
        @staticmethod
        def attest_clean(_session, _lease): calls.append('attest')

    class Admission:
        def __init__(self, *, check_owner, snapshot, total_bytes, cap_bytes):
            self.snapshot, self.state = snapshot, 'new'
            self.lease = FakeLease()
            self.budget = types.SimpleNamespace(active_leases=lambda: ())
        def acquire(self):
            calls.append('admit')
            assert self.snapshot() == memory
            self.state = 'active'
            return dict(memory)
        def verify(self):
            calls.append('verify_claim')
            assert self.state == 'active'

    class Operations:
        def __init__(self, _session, _lease, admitted):
            self.admitted, self.unsafe = admitted, False
        def bind_session(self, _slot): calls.append('bind_session')
        def require_disabled_gc(self): assert not gc.isenabled()

    class Runtime:
        def __init__(self, _operations): pass
        def initialize(self): calls.append('initialize')
        def synchronize(self): calls.append('synchronize')

    class Sample:
        def __init__(self, *_args, **_kwargs): pass

    class Setup:
        def __init__(self, _sample):
            self.partial = {'memory_before_transfer': memory}
        def run(self, _cache): calls.append('model'); return {'input_ids': [15496, 995]}

    class Preflight:
        def __init__(self, _sample): pass
        def prepare(self): calls.append('prepare')
        def run(self, gate):
            calls.append('preflight')
            gate(rows[0]['gate_lower_ns'], rows[0]['gate_upper_ns'])
            return rows[0], rows[1]

    class Lean:
        def __init__(self, _sample): self.index, self.partial = 0, None
        def sample_mode(self, mode):
            self.index += 1
            calls.append('sample')
            self.partial = {'stage': 'running', 'mode': mode, 'started_ns': 123}
            if fail_second_sample and self.index == 2:
                raise KeyboardInterrupt('private test failure')
            row, post = rows[2 * self.index:2 * self.index + 2]
            assert row['mode'] == mode
            return {**row, 'memory_after': post['memory_after'], 'postcheck_completed_ns': post['completed_ns']}

    class Barriers:
        def __init__(self, _ready): pass
        def accept_start(self, _raw, **_kwargs): calls.append('start_grant')
        def authorize_preflight(self, _lower, _upper): calls.append('authorize')
        def publish_finished(self, path, **_kwargs):
            calls.append('finished')
            Path(path).write_text('finished', encoding='utf-8')
        def accept_cleanup(self, _raw, **_kwargs): calls.append('cleanup_grant')

    class Disposal:
        def __init__(self, operations, **_kwargs):
            self.operations = operations
            self.state = 'ready'
            self.counters = None
        def finish(self):
            calls.append('dispose')
            self.operations.admitted.state = 'released'
            self.state = 'released'
            self.counters = {'allocated_bytes': 0, 'reserved_bytes': 0}

    def device_count():
        calls.append('device_count')
        return 1
    cuda = types.SimpleNamespace(device_count=device_count,
        get_device_properties=lambda _i: types.SimpleNamespace(uuid='fake', total_memory=total,
                                                                name='CPU fake', major=7, minor=5),
        get_allocator_backend=lambda: 'native', mem_get_info=lambda _i: (memory['free_bytes'], total),
        memory_allocated=lambda _i: memory['allocated_bytes'], memory_reserved=lambda _i: memory['reserved_bytes'],
        get_per_process_memory_fraction=lambda _i: 0.25, manual_seed=lambda _seed: None,
        is_initialized=lambda: False, empty_cache=lambda: None)
    torch = types.ModuleType('torch')
    torch.cuda, torch.__version__, torch.version = cuda, '2.8.0+cu128', types.SimpleNamespace(cuda='12.8')
    torch._C = types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: None)
    if version_error == 'torch': torch.__version__ = '0.0'
    if version_error == 'cuda': torch.version.cuda = '0.0'
    if version_error == 'workspace_cleanup': torch._C._cuda_clearCublasWorkspaces = None
    torch.__future__ = types.SimpleNamespace(get_overwrite_module_params_on_conversion=lambda: False,
                                            get_swap_module_params_on_conversion=lambda: False)
    torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False, benchmark=False, deterministic=True))
    torch.random = types.SimpleNamespace(default_generator=types.SimpleNamespace(manual_seed=lambda _seed: None))
    torch.set_num_threads = torch.set_num_interop_threads = torch.use_deterministic_algorithms = lambda _v: None
    torch.get_num_threads = torch.get_num_interop_threads = lambda: 1
    torch.are_deterministic_algorithms_enabled = lambda: True
    torch.is_autocast_enabled = lambda _device: False
    transformers = types.ModuleType('transformers')
    transformers.__version__ = '4.57.3'
    if version_error == 'transformers': transformers.__version__ = '0.0'
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'backend:native')
    monkeypatch.setattr(subject, '_environment', lambda *_args: (0, 'c' * 64))
    monkeypatch.setattr(subject, '_workspace', lambda _session: Path(__file__).resolve().parents[1])
    monkeypatch.setattr(subject, '_validate_cache', lambda _session, _cache: None)
    versions = dict(subject._expected_runtime()['loader_versions'])
    if version_error == 'loader': versions['safetensors'] = '0.0'
    monkeypatch.setattr(subject, '_dependency_versions', lambda: versions)
    monkeypatch.setattr(subject, '_clock', lambda: (fixture.CLOCK, {'platform': 'win32',
        'implementation': 'cpython', 'python_version': '3.11.15'}))
    monkeypatch.setattr(subject, '_OWNER_ENABLED', True)
    monkeypatch.setattr(subject, '_CUSTODY', dict(subject._CUSTODY, entered=False, directory=None, failure=None,
                                                  diagnostic_failure=None))
    monkeypatch.setattr(device_lease, 'DeviceLease', FakeLease)
    monkeypatch.setattr(supervisor, 'OwnerSession', Session)
    monkeypatch.setattr(supervisor, '_supervisor_birth', lambda: '123')
    monkeypatch.setattr(gpu_oneshot, '_runtime_uuid', lambda _torch, _uuid: Session.identity.gpu_uuid)
    monkeypatch.setattr(tap_repeated_admission, '_Admission', Admission)
    monkeypatch.setattr(tap_repeated_owner, '_OwnerOperations', Operations)
    monkeypatch.setattr(tap_repeated_cuda_backend, '_CudaRuntime', Runtime)
    monkeypatch.setattr(tap_repeated_timing, '_TimedSample', Sample)
    monkeypatch.setattr(tap_repeated_model, '_FixedModelSetup', Setup)
    monkeypatch.setattr(tap_repeated_preflight, '_FixedPreflight', Preflight)
    monkeypatch.setattr(tap_repeated_lean, '_LeanTimedSample', Lean)
    monkeypatch.setattr(tap_repeated_child_barrier, 'ChildBarriers', Barriers)
    monkeypatch.setattr(tap_repeated_child_barrier, 'wait_frame', lambda *_a, **_k: (b'grant', 1))
    monkeypatch.setattr(tap_repeated_disposal, '_Disposal', Disposal)
    for module, flag in ((tap_repeated_owner, '_OPERATIONS_ENABLED'),
                         (tap_repeated_cuda_backend, '_CUDA_BACKEND_ENABLED'),
                         (tap_repeated_timing, '_TIMING_ENABLED'),
                         (tap_repeated_model, '_MODEL_SETUP_ENABLED'), (tap_repeated_lean, '_LEAN_ENABLED')):
        monkeypatch.setattr(module, flag, getattr(module, flag))
    was_enabled, collect = gc.isenabled(), gc.collect
    try:
        result = subject.owner_main(str(tmp_path), str(tmp_path), 'd' * 32, 'a' * 40, str(tmp_path / 'git'))
    finally:
        gc.collect = collect
        if was_enabled:
            gc.enable()
    return result, calls


def test_cpu_fake_owner_preserves_producer_barrier_and_cleanup_order(monkeypatch, tmp_path):
    result, calls = _synthetic_execution(monkeypatch, tmp_path, fail_second_sample=False)
    assert result == 0
    assert calls.count('sample') == 42
    assert calls.index('admit') < calls.index('model') < calls.index('prepare') < calls.index('ready')
    assert calls.index('start_grant') < calls.index('preflight') < calls.index('sample')
    assert calls.index('verify_claim') < calls.index('finished') < calls.index('cleanup_grant')
    assert calls.index('cleanup_grant') < calls.index('dispose') < calls.index('attest')
    candidate_raw = (tmp_path / 'candidate.json').read_bytes()
    candidate = json.loads(candidate_raw)
    final = json.loads((tmp_path / 'final.json').read_bytes())
    assert candidate['journal_rows'] == 86 and candidate['whole_run_claims_held'] is True
    assert candidate['cuda_device_name'] == 'CPU fake' and candidate['cuda_compute_capability'] == [7, 5]
    assert candidate['admission_memory'] == candidate['memory_before_transfer']
    assert candidate['journal_sha256'] == hashlib.sha256((tmp_path / 'journal.jsonl').read_bytes()).hexdigest()
    assert final['candidate_sha256'] == hashlib.sha256(candidate_raw).hexdigest()
    assert final['claims_released_after_disposal'] and final['device_lease_held_at_attestation']


def test_cpu_fake_owner_stops_on_sample_interruption_without_disposal(monkeypatch, tmp_path):
    result, calls = _synthetic_execution(monkeypatch, tmp_path, fail_second_sample=True)
    assert result == 1 and calls.count('sample') == 2
    assert 'finished' not in calls and 'dispose' not in calls and 'attest' not in calls
    assert not (tmp_path / 'candidate.json').exists() and not (tmp_path / 'final.json').exists()
    assert len((tmp_path / 'journal.jsonl').read_bytes().splitlines()) == 4
    failure = json.loads((tmp_path / 'failure.json').read_bytes())
    assert failure['stage'] == 'samples' and failure['error_type'] == 'KeyboardInterrupt'
    assert failure['journal_rows_fsynced_before_failure'] == 4
    assert failure['partial'] == {'source': 'lean', 'stage': 'running', 'mode': 'direct', 'started_ns': 123}


@pytest.mark.parametrize('version_error', ['torch', 'cuda', 'transformers', 'loader', 'workspace_cleanup'])
def test_cpu_fake_owner_rejects_runtime_mismatch_before_admission_and_model(monkeypatch, tmp_path, version_error):
    result, calls = _synthetic_execution(monkeypatch, tmp_path, fail_second_sample=False,
                                         version_error=version_error)
    assert result == 1
    assert 'admit' not in calls and 'model' not in calls
    if version_error not in ('transformers', 'loader'):
        assert 'device_count' not in calls
    else:
        assert calls.index('bind_runtime') > calls.index('device_count')
    assert 'dispose' not in calls and 'attest' not in calls
