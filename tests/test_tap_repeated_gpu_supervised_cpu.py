"""Exercise the actual owner loop and parent with a CPU-only fake device.

The timing and model callbacks are fakes. The Windows Job, owner session,
admission, barriers, durable journal, disposal and parent replay are real.
This is not a GPU execution, model correctness or timing measurement.
"""

import importlib.util
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import time

import pytest

from looking_glass import tap_repeated_parent as parent
from looking_glass import tap_repeated_protocol as protocol
from looking_glass.device_identity import resolve_device_identity
from looking_glass.supervisor import StartupChannel
from looking_glass.tap_repeated_activity import ParentActivity
from scripts.v2_source_guards import WindowsJob


_GPU = 'GPU-00000000-0000-0000-0000-000000000001'


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='the parent uses Windows Job and pinned QPC')
@pytest.mark.parametrize('source_bootstrap', [False, True], ids=['direct', 'source-bootstrap'])
def test_actual_owner_parent_journal_assembler_and_reap_with_fake_device(monkeypatch, tmp_path, source_bootstrap):
    from looking_glass import tap_repeated_gpu_evidence as evidence

    fixture_path = Path(__file__).with_name('test_tap_repeated_protocol.py')
    spec = importlib.util.spec_from_file_location('_timing_cpu_fixture', fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    schedule = protocol.load_schedule(fixture.SCHEDULE)
    seed_binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40,
                    'model_sha256': 'c' * 64, 'runtime_sha256': 'd' * 64,
                    'device_sha256': 'e' * 64}
    rows = fixture._attempt(schedule, 0, seed_binding)['journal']
    (tmp_path / 'template.json').write_text(json.dumps(rows), encoding='utf-8')
    code = textwrap.dedent(r'''
        import copy, json, os, sys, time, types
        from pathlib import Path
        folder = Path(sys.argv[1])
        rows = json.loads((folder / 'template.json').read_text(encoding='utf-8'))
        gpu, total, gib = 'GPU-00000000-0000-0000-0000-000000000001', 8 * 1024**3, 1024**3
        memory = {'free_bytes': 6 * gib, 'total_bytes': total,
                  'allocated_bytes': 384 * 1024**2, 'reserved_bytes': 512 * 1024**2}
        cleaned = [False]
        def cleanup(): cleaned[0] = True
        cuda = types.SimpleNamespace(
            is_initialized=lambda: False, device_count=lambda: 1,
            get_device_properties=lambda _i: types.SimpleNamespace(uuid=gpu, total_memory=total,
                                                                    name='CPU fake', major=7, minor=5),
            get_allocator_backend=lambda: 'native', mem_get_info=lambda _i: (memory['free_bytes'], total),
            memory_allocated=lambda _i: 0 if cleaned[0] else memory['allocated_bytes'],
            memory_reserved=lambda _i: 0 if cleaned[0] else memory['reserved_bytes'],
            get_per_process_memory_fraction=lambda _i: 0.25, manual_seed=lambda _seed: None,
            empty_cache=cleanup)
        torch = types.ModuleType('torch')
        torch.cuda, torch.__version__, torch.version = cuda, '2.8.0+cu128', types.SimpleNamespace(cuda='12.8')
        torch._C = types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: None)
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
        sys.modules['torch'], sys.modules['transformers'] = torch, transformers
        from looking_glass import (gpu_oneshot, tap_repeated_gpu_owner as subject,
            tap_repeated_cuda_backend as backend, tap_repeated_model as fixed_model,
            tap_repeated_preflight as preflight, tap_repeated_lean as lean)
        subject._OWNER_ENABLED = True
        subject._environment = lambda *_args: (0, 'b' * 64)
        subject._workspace = lambda _session: Path.cwd()
        subject._validate_cache = lambda _session, _cache: None
        subject._dependency_versions = lambda: dict(subject._expected_runtime()['loader_versions'])
        gpu_oneshot._runtime_uuid = lambda _torch, value: value
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native'

        class Resource:
            pass
        class Runtime:
            device, vocab_size, hidden_size = 'cuda:0', 50257, 768
            def __init__(self, operations): self.operations = operations
            def initialize(self):
                roots = self.operations.roots
                roots.default_stream, roots.start_event, roots.end_event = Resource(), Resource(), Resource()
            def synchronize(self): return None
        class Setup:
            def __init__(self, sample):
                self.sample = sample
                self.partial = {'memory_before_transfer': dict(memory)}
            def run(self, _cache):
                roots = self.sample.operations.roots
                roots.model, roots.input_ids = Resource(), Resource()
                roots.converted = roots.model
                return {'weights_sha256': 'c' * 64, 'input_ids': [15496, 995]}
        class Preflight:
            def __init__(self, sample): self.sample = sample
            def prepare(self): return None
            def run(self, gate):
                row, post = copy.deepcopy(rows[0]), copy.deepcopy(rows[1])
                memory_started = time.perf_counter_ns()
                memory_ended = time.perf_counter_ns()
                lower = time.perf_counter_ns()
                upper = lower + 250_000_000
                gate(lower, upper)
                started = time.perf_counter_ns()
                ended, completed = started + 100_000, started + 200_000
                while time.perf_counter_ns() < completed: time.sleep(.001)
                row.update(started_ns=started, ended_ns=ended, gate_lower_ns=lower, gate_upper_ns=upper,
                           memory_snapshot_started_ns=memory_started, memory_snapshot_ended_ns=memory_ended)
                post['completed_ns'] = completed
                self.sample._preflight_verified = True
                return row, post
        class Lean:
            def __init__(self, _sample): self.index = 0
            def sample_mode(self, mode):
                self.index += 1
                row, post = copy.deepcopy(rows[2*self.index]), copy.deepcopy(rows[2*self.index+1])
                assert row['mode'] == mode
                start = time.perf_counter_ns()
                completion, end, post_ns, guarded_end = (start+100_000, start+200_000,
                                                          start+300_000, start+400_000)
                while time.perf_counter_ns() < guarded_end: time.sleep(.001)
                row.update(started_ns=start, completion_ns=completion, ended_ns=end, wall_ns=100_000,
                           guarded_started_ns=start-1000, guarded_ended_ns=guarded_end,
                           guarded_wall_ns=401_000, device_event_ms=.05,
                           memory_after=post['memory_after'], postcheck_completed_ns=post_ns)
                return row
        backend._CudaRuntime, fixed_model._FixedModelSetup = Runtime, Setup
        preflight._FixedPreflight, lean._LeanTimedSample = Preflight, Lean
        code = subject.owner_main(str(folder), str(folder), 'c' * 32, 'a' * 40, str(folder / 'fake-git'))
        os._exit(code)
    ''')
    monkeypatch.setattr(parent, '_PARENT_LAUNCH_ENABLED', True)
    identity = resolve_device_identity(tmp_path, _GPU, physical_uuids=(_GPU,), environ={'CUDA_VISIBLE_DEVICES': _GPU})
    info = time.get_clock_info('perf_counter')
    clock = {'name': 'perf_counter_ns', 'monotonic': info.monotonic, 'adjustable': info.adjustable,
             'implementation': info.implementation, 'resolution_seconds': info.resolution}
    runtime = {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}
    class ObservedJob(WindowsJob):
        def terminate_and_wait(self, timeout=5):
            self.before_stop = self.process_counts()
            return super().terminate_and_wait(timeout=timeout)
    channel, job = StartupChannel(max_bytes=4096), ObservedJob()
    repo = Path(__file__).resolve().parents[1]
    env = {'SystemRoot': os.environ['SystemRoot'], 'PYTHONPATH': str(repo), 'CUDA_VISIBLE_DEVICES': _GPU}
    commit, tree = 'a' * 40, 'b' * 40
    command = [sys._base_executable, '-B', '-c', code, str(tmp_path)]
    if source_bootstrap:
        # Run the actual isolated source-bound bootstrap and dispatcher, then
        # import the pinned CPU dependencies after the fake lease is acquired.
        # Device and model operations remain fakes; no checkpoint is loaded.
        try:
            pinned = (metadata.version('torch') == '2.8.0+cu128'
                      and metadata.version('transformers') == '4.57.3')
        except metadata.PackageNotFoundError:
            pinned = False
        if not pinned:
            pytest.skip('pinned CPU dependency imports are unavailable')
        bootstrap = repo / 'scripts' / 'tap_repeated_gpu_bootstrap.py'
        assert bootstrap.is_file()
        git = Path(shutil.which('git')).resolve(strict=True)
        commit = subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        tree = subprocess.check_output([git, 'rev-parse', 'HEAD:'], cwd=repo, text=True).strip()
        paths = runpy.run_path(str(bootstrap), run_name='_test_bootstrap_support')['_verify_source'](
            repo, commit, git)
        source_files = {}
        for relative in paths:
            mode = subprocess.check_output([git, 'ls-tree', commit, '--', relative], cwd=repo).split(b' ', 1)[0]
            assert mode in (b'100644', b'100755')
            source_files[relative] = {'mode': mode.decode('ascii'),
                                      'sha256': hashlib.sha256((repo / relative).read_bytes()).hexdigest()}
        manifest = {'schema': 'tap-repeated-source-manifest-v1', 'run_id': 'c' * 32,
                    'source_commit': commit, 'source_files': source_files}
        manifest_raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        protected = tmp_path / 'protected-manifest'
        protected.mkdir()
        (protected / 'source-manifest.json').write_bytes(manifest_raw)
        prefix = tmp_path / 'pycache'
        prefix.mkdir()
        env.pop('PYTHONPATH')
        env['LOOKING_GLASS_TAP_REPEATED_MANIFEST_SHA256'] = hashlib.sha256(manifest_raw).hexdigest()
        uname = platform.uname()
        fields = {key: getattr(uname, key) for key in
                  ('system', 'node', 'release', 'version', 'machine', 'processor')}
        env['LOOKING_GLASS_TAP_REPEATED_PLATFORM_UNAME'] = json.dumps(fields, sort_keys=True, separators=(',', ':'))
        needle = "sys.modules['torch'], sys.modules['transformers'] = torch, transformers\n"
        end = "code = subject.owner_main(str(folder), str(folder), 'c' * 32, 'a' * 40, str(folder / 'fake-git'))\nos._exit(code)\n"
        before, after = code.split(needle, 1)
        install, tail = after.split(end, 1)
        assert not tail
        source_code = before + '\ndef install():\n' + textwrap.indent(install, '    ') + textwrap.dedent('''
            import_line = None
            def trace(frame, event, arg):
                global import_line
                if (event == 'call' and frame.f_globals.get('__name__') == 'looking_glass.tap_repeated_gpu_owner'
                        and frame.f_code.co_name == 'owner_main'):
                    install()
                    import inspect
                    lines, first = inspect.getsourcelines(frame.f_globals['_execute_owner'])
                    matches = [first + i for i, line in enumerate(lines) if line.strip() == 'import torch']
                    if len(matches) != 1:
                        raise RuntimeError('owner dependency import boundary changed')
                    import_line = matches[0]
                elif (event == 'line' and frame.f_globals.get('__name__') == 'looking_glass.tap_repeated_gpu_owner'
                        and frame.f_code.co_name == '_execute_owner' and frame.f_lineno == import_line):
                    from looking_glass.supervisor import OwnerSession
                    OwnerSession.check(frame.f_locals['session'], frame.f_locals['lease'])
                    if 'torch' in sys.modules or 'transformers' in sys.modules:
                        raise RuntimeError('dependency import occurred before the fake lease')
                    import torch as actual_torch
                    import transformers as actual_transformers
                    if (str(actual_torch.__version__) != '2.8.0+cu128'
                            or str(actual_transformers.__version__) != '4.57.3'
                            or actual_torch.cuda.is_initialized()):
                        raise RuntimeError('pinned CPU dependencies were not imported safely')
                    (folder / 'pinned-import-proof.json').write_text(json.dumps({
                        'lease_checked': True, 'cuda_initialized': False,
                        'torch': str(actual_torch.__version__),
                        'transformers': str(actual_transformers.__version__)}), encoding='utf-8')
                    sys.modules['torch'], sys.modules['transformers'] = torch, transformers
                    sys.settrace(None)
                return trace
            sys.settrace(trace)
            import runpy
            bootstrap, site, commit, git = sys.argv[2:]
            sys.argv = [bootstrap, '--bootstrap-site', site, '--source-commit', commit,
                        '--git-executable', git, '--owner', str(folder), str(folder), 'c' * 32, commit, git]
            runpy.run_path(bootstrap, run_name='__main__')
        ''')
        site = Path(sysconfig.get_paths()['purelib']).resolve(strict=True)
        command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}',
                   '-c', source_code, str(tmp_path), str(bootstrap), str(site), commit, str(git)]
    supervision = parent._Supervision(command,
        channel=channel, identity=identity, device_lease_path=identity.lock_path,
        marker_path=identity.marker_path, cwd=repo, env=env, timeout=90, process_job=job)
    def query(kind):
        return b'' if kind == 'process' else f'{_GPU},8192,1024,7168,0\n'.encode()
    activity = ParentActivity(gpu_uuid=_GPU, query=query)
    journal = parent._PrivateJournal(tmp_path / 'activity.jsonl')
    try:
        result = parent._drive_session(session_index=0, challenge='b' * 64, identity=identity,
            channel=channel, supervision=supervision, activity=activity, journal=journal,
            attempt_dir=tmp_path, expected_smi_total=8 * 1024**3, schedule=schedule, parent_clock=clock,
            parent_runtime=runtime, exclusive=True, clock=time.perf_counter_ns, sleep=time.sleep,
            startup_timeout=30, workload_timeout=30, reap_timeout=30)
        assert result['supervision'].clean and result['supervision'].exit_code == 0
        assert job.before_stop == (1, 0)
        assert result['state'] == 'pending_owner_evidence' and len(result['journal']) == 86
        assert result['screen']['availability'] == 'complete'
        candidate_raw, final_raw = (tmp_path / 'candidate.json').read_bytes(), (tmp_path / 'final.json').read_bytes()
        candidate = json.loads(candidate_raw)
        attempt, binding = evidence.assemble_private_attempt(parent=result, candidate_raw=candidate_raw,
            final_raw=final_raw, study_run_id='c'*32, source_commit=commit, source_tree=tree,
            expected_model=candidate['model'], expected_runtime=candidate['runtime'], parent_runtime=runtime,
            expected_device_name='CPU fake', observed_driver='555.55',
            clock=clock, exclusive=True, source_exit_contract=True)
        replay = protocol.summarize_attempts(schedule, [attempt], trusted_binding=binding)
        assert replay['complete_sessions'] == 1 and replay['aggregate'] is None
        assert replay['slots'][0]['summary']['measured_samples'] == 36
        assert len(replay['slots'][0]['summary']['paired_rounds']) == 12
        assert not identity.marker_path.exists() and not identity.attestation_path.exists()
        if source_bootstrap:
            proof = json.loads((tmp_path / 'pinned-import-proof.json').read_text(encoding='utf-8'))
            assert proof == {'lease_checked': True, 'cuda_initialized': False,
                             'torch': '2.8.0+cu128', 'transformers': '4.57.3'}
    finally:
        channel.abort()
        job.close()
