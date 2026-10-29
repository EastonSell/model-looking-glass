"""CPU-only checks for the supervised GPU-owner composition.

The fake runtime reports a fictitious GPU and never touches CUDA.  A test
claiming actual runtime/device verification would need a separately approved
hardware window and measured profiles.
"""

import threading
import unittest
import gc
import ctypes
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import weakref
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass.memory_budget import AdmissionError, CudaMemorySnapshot
from looking_glass import gpu_service
from looking_glass.gpu_service import (_canonical_runtime_uuid, _run_bound_owner,
    _TorchServiceBackend, _validate_service_keys, ProfileFiles, _load_profile_registry)


GIB = 1024 ** 3
LOAD_COMMITMENT = 2 * GIB
JOB_COMMITMENT = 512 * 1024 ** 2
UUID = 'GPU-00000000-0000-0000-0000-000000000001'

_CUDA_PATCHES = []
_ORIGINAL_CUDA_VISIBLE_DEVICES = None


def _block_cuda_calls(*args, **kwargs):
    raise AssertionError('CPU-only GPU-service tests must never call CUDA')


def _install_cuda_guard(*, hide_device):
    """Fail immediately on high-level CUDA or low-level torch CUDA calls."""
    if hide_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    for name in dir(torch.cuda):
        value = getattr(torch.cuda, name)
        if (callable(value) and not (isinstance(value, type)
                                    and issubclass(value, BaseException))):
            item = patch.object(torch.cuda, name, _block_cuda_calls)
            item.start()
            _CUDA_PATCHES.append(item)
    for name in dir(torch._C):
        value = getattr(torch._C, name)
        if name.startswith('_cuda_') and callable(value) and not isinstance(value, type):
            item = patch.object(torch._C, name, _block_cuda_calls)
            item.start()
            _CUDA_PATCHES.append(item)
    original_to = torch.Tensor.to
    def cpu_only_to(tensor, *args, **kwargs):
        destination = kwargs.get('device')
        if destination is None and args:
            destination = args[0]
        if (isinstance(destination, (str, torch.device))
                and str(destination).split(':', 1)[0] == 'cuda'):
            return _block_cuda_calls()
        return original_to(tensor, *args, **kwargs)
    item = patch.object(torch.Tensor, 'to', cpu_only_to)
    item.start()
    _CUDA_PATCHES.append(item)
    for loader_name in ('CDLL', 'WinDLL', 'PyDLL', 'OleDLL'):
        original_loader = getattr(ctypes, loader_name, None)
        if original_loader is None:
            continue
        def cpu_only_load(name, *args, _original=original_loader, **kwargs):
            candidate = str(name).lower()
            if any(piece in candidate for piece in ('nvcuda', 'nvml', 'libcuda', 'nvidia-ml')):
                return _block_cuda_calls()
            return _original(name, *args, **kwargs)
        item = patch.object(ctypes, loader_name, cpu_only_load)
        item.start()
        _CUDA_PATCHES.append(item)
    original_popen = subprocess.Popen
    def cpu_only_popen(args, *pargs, **kwargs):
        executable = args[0] if isinstance(args, (list, tuple)) and args else args
        if isinstance(executable, (str, bytes, os.PathLike)) and 'nvidia-smi' in os.fsdecode(executable).lower():
            return _block_cuda_calls()
        return original_popen(args, *pargs, **kwargs)
    item = patch.object(subprocess, 'Popen', cpu_only_popen)
    item.start()
    _CUDA_PATCHES.append(item)


def setUpModule():
    global _ORIGINAL_CUDA_VISIBLE_DEVICES
    _ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    _install_cuda_guard(hide_device=True)


def tearDownModule():
    for item in reversed(_CUDA_PATCHES):
        item.stop()
    _CUDA_PATCHES.clear()
    if _ORIGINAL_CUDA_VISIBLE_DEVICES is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = _ORIGINAL_CUDA_VISIBLE_DEVICES


class FakeSession:
    def __init__(self, events, *, reject_check=False, reject_binding=False):
        self.events = events
        self.identity = SimpleNamespace(gpu_uuid=UUID)
        self.reject_check = reject_check
        self.reject_binding = reject_binding

    def check(self, lease):
        self.events.append('session-check')
        if self.reject_check:
            raise RuntimeError('no supervised lease')

    def verify_runtime_binding(self, lease, *, runtime_uuid, device_count):
        self.events.append('verify-binding')
        if self.reject_binding or runtime_uuid != UUID or device_count != 1:
            raise RuntimeError('identity mismatch')

    def attest_clean(self, lease):
        self.events.append('attest')


@dataclass(frozen=True)
class FakeProfile:
    estimated_incremental_bytes: int


@dataclass
class FakeResult:
    status: str = 'clean'
    teardown_verified: bool = True


class FakeWorker:
    quarantined = False


class FakeBroker:
    quarantined = False


def _reset_fake_owner():
    # Only fake CPU fixtures may reset process-lifetime custody. A real owner
    # must exit instead of clearing a quarantine or reusing this process.
    gpu_service._QUARANTINE_HOLDS.clear()
    gpu_service._QUARANTINED.clear()
    gpu_service._OWNER_CUSTODY[:] = [None] * len(gpu_service._OWNER_CUSTODY)
    gpu_service._OWNER_CLAIMED = False
    gpu_service._OWNER_UNSAFE = False
    gpu_service._OWNER_DRAIN_PENDING = False


class UuidRendering:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class RuntimeUuidTests(unittest.TestCase):
    def test_cpu_guard_blocks_runtime_driver_and_tensor_copy(self):
        import torch
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '')
        guarded = [torch.cuda.device_count, torch.cuda.get_device_properties,
                   lambda: torch.tensor([1]).to('cuda:0'),
                   lambda: ctypes.CDLL('nvcuda.dll'),
                   lambda: subprocess.Popen(['nvidia-smi'])]
        if hasattr(torch._C, '_cuda_init'):
            guarded.append(torch._C._cuda_init)
        for fn in guarded:
            with self.subTest(fn=fn), self.assertRaisesRegex(AssertionError, 'never call CUDA'):
                fn()

    def test_requires_exact_runtime_uuid_object_and_full_lowercase_rendering(self):
        raw = 'abcdefab-1234-5678-9abc-abcdef123456'
        self.assertEqual(_canonical_runtime_uuid(UuidRendering(raw), UuidRendering), 'GPU-' + raw)
        for invalid in ('GPU-' + raw, raw.upper(), raw[:-1], raw + '\n'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _canonical_runtime_uuid(UuidRendering(invalid), UuidRendering)
        with self.assertRaises(TypeError):
            _canonical_runtime_uuid(raw, UuidRendering)


@dataclass(frozen=True)
class FakeKey:
    operation: str
    warmup_state: str
    model_id: str = 'a' * 64
    device: str = 'cuda:0'
    device_uuid: str = UUID
    device_total_bytes: int = 8 * GIB
    checkpoint_sha256: str = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
    checkpoint_revision: str = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
    model_config_sha256: str = '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a'
    model_dtype: str = 'torch.float32'
    compute_dtype: str = 'torch.float32'
    attention_backend: str = 'eager'
    model_mode: str = 'eval'
    grad_mode: str = 'inference'
    autocast: str = 'disabled'
    use_cache: bool = False
    compiled: bool = False
    capture_mode: str = 'none'
    max_gpu_trace_bytes: int = 0
    max_host_trace_bytes: int = 0
    batch_size: int = 1
    sequence_length: int = 2
    input_dtypes: tuple = (('input_ids', 'torch.int64'),)
    output_limit_bytes: int = 512 * 1024
    warmup_protocol_sha256: str = '84b21a9ba63413bb9394dbac6d31dbc08ce3171508a25477f44e596be579760c'
    capture_policy_sha256: str = '1ec2c8752f73cab7403fa9012e0796de77fdd87eb395a7cbc796044bf1643250'


def service_keys():
    load = FakeKey('model_load', 'cold', batch_size=0, sequence_length=0,
                   input_dtypes=(), output_limit_bytes=0)
    cold = FakeKey('forward', 'cold')
    return load, cold, replace(cold, warmup_state='warm')


class KeyContractTests(unittest.TestCase):
    def test_only_the_exact_pinned_signature_and_state_transition_are_supported(self):
        identity = SimpleNamespace(gpu_uuid=UUID)
        load, cold, warm = service_keys()
        _validate_service_keys(identity, (load, cold, warm))
        invalid = [
            (load, cold, cold),
            (load, replace(cold, sequence_length=8), replace(warm, sequence_length=8)),
            (load, cold, replace(warm, input_dtypes=(('attention_mask', 'torch.int64'),))),
            (replace(load, model_id='b' * 64), cold, warm),
            (load, replace(cold, attention_backend='sdpa'), replace(warm, attention_backend='sdpa')),
            (load, replace(cold, use_cache=True), replace(warm, use_cache=True)),
            (load, replace(cold, capture_mode='direct'), replace(warm, capture_mode='direct')),
            (load, replace(cold, warmup_protocol_sha256='f' * 64),
             replace(warm, warmup_protocol_sha256='f' * 64)),
            (replace(load, capture_policy_sha256='f' * 64), cold, warm),
            (replace(load, device_total_bytes=9 * GIB), cold, warm),
            (replace(load, device_total_bytes=True), cold, warm),
            tuple(replace(key, device_total_bytes=6 * GIB) for key in (load, cold, warm)),
            (load, replace(cold, output_limit_bytes=2 * 1024 * 1024),
             replace(warm, output_limit_bytes=2 * 1024 * 1024)),
        ]
        for keys in invalid:
            with self.subTest(keys=keys):
                with self.assertRaises(ValueError):
                    _validate_service_keys(identity, keys)


def measured_keys_and_registry(*, source_sha=None):
    from looking_glass.resource_profile import (WorkloadKey, MemorySample,
        ProfileProvenance, MeasuredResourceProfile, ProfileRegistry,
        encode_evidence, model_runtime_fingerprint)
    keys = []
    for template in service_keys():
        fields = asdict(template)
        fields.update(source_sha256=source_sha or hashlib.sha256(b'fake-source-manifest').hexdigest(),
                      runtime_settings_sha256='2' * 64,
                      torch_version='2.8.0+cu128', transformers_version='4.57.3',
                      cuda_runtime_version='12.8', driver_version='570.0',
                      python_version='3.11.15', device_name='fake CPU-test device',
                      compute_capability=(7, 5), device_total_bytes=8 * GIB)
        fields['model_id'] = model_runtime_fingerprint(fields)
        keys.append(WorkloadKey(**fields))
    records = []
    for index, key in enumerate(keys):
        before = CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0)
        after = CudaMemorySnapshot(7 * GIB - 1024, 8 * GIB, 1024, 2048)
        sample = MemorySample(key.fingerprint(), f'fake-{index}', '5' * 64, 17,
                              'completed', before, after, 1024, 2048)
        evidence = encode_evidence(key, (sample,))
        provenance = ProfileProvenance('resource-profile-v1', hashlib.sha256(b'fake-source-manifest').hexdigest(),
            hashlib.sha256(evidence).hexdigest(), '2026-10-28T00:00:00Z',
            'cooperative-exclusive', 'synchronize-reset-synchronize')
        records.append(MeasuredResourceProfile.build(key, (sample,), provenance,
                                                       margin_bytes=1024, margin_bps=0))
    registry = ProfileRegistry(tuple(records), approved_profile_ids=tuple(p.profile_id for p in records))
    return tuple(keys), registry


def write_profile_files(directory, *, source_sha=None):
    from looking_glass.resource_profile import encode_evidence
    keys, registry = measured_keys_and_registry(source_sha=source_sha)
    records = []
    for index, key in enumerate(keys):
        profile = registry.resolve(key)
        document = Path(directory) / f'profile-{index}.json'
        evidence = Path(directory) / f'evidence-{index}.json'
        manifest = Path(directory) / f'manifest-{index}.json'
        document.write_text(profile.to_json(), encoding='utf-8')
        evidence.write_bytes(encode_evidence(key, profile.samples))
        manifest.write_bytes(b'fake-source-manifest')
        records.append(ProfileFiles(document, evidence, manifest, profile.profile_id))
    return tuple(records), keys


class CpuProfilePreflightTests(unittest.TestCase):
    def test_profile_source_key_must_match_the_saved_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            files, _ = write_profile_files(directory, source_sha='1' * 64)
            with self.assertRaisesRegex(ValueError, 'source manifest'):
                _load_profile_registry(files, SimpleNamespace(gpu_uuid=UUID))
    def test_only_independently_pinned_untampered_profiles_form_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            files, keys = write_profile_files(directory)
            identity = SimpleNamespace(gpu_uuid=UUID)
            registry, declared = _load_profile_registry(files, identity)
            self.assertEqual(declared, keys)
            self.assertEqual(registry.resolve(keys[0]).profile_id, files[0].approved_profile_id)
            files[1].evidence.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'evidence'):
                _load_profile_registry(files, identity)

    def test_missing_profile_or_untrusted_pin_fails_before_any_owner_start(self):
        with tempfile.TemporaryDirectory() as directory:
            files, _ = write_profile_files(directory)
            identity = SimpleNamespace(gpu_uuid=UUID)
            with self.assertRaises(ValueError):
                _load_profile_registry(files[:2], identity)
            changed_pin = replace(files[0], approved_profile_id='f' * 64)
            with self.assertRaisesRegex(ValueError, 'approved'):
                _load_profile_registry((changed_pin,) + files[1:], identity)
            files[2].document.unlink()
            with self.assertRaises(FileNotFoundError):
                _load_profile_registry(files, identity)


class SharedBootstrapSmokeTests(unittest.TestCase):
    """Opt-in CPU checkpoint smoke for the shared, reviewed child guard."""

    def test_offline_pinned_model_load_under_guarded_imports(self):
        """Optional local-cache smoke, with real weights and no CUDA access.

        The temporary manifest records the dry run's resolved import origins,
        distribution metadata and torch DLLs; it is a test fixture, not a
        source profile or production preflight. Hashing DLLs costs several GiB
        of disk I/O, so this is opt-in rather than part of every test run.
        """
        if os.environ.get('MLG_RUN_OFFLINE_STACK_SMOKE') != '1':
            self.skipTest('set MLG_RUN_OFFLINE_STACK_SMOKE=1 for the offline stack smoke')
        from looking_glass.gpu_bootstrap import prepare_owner_launch
        site = Path(sys.prefix) / 'Lib' / 'site-packages'
        environment = Path(sys.prefix).resolve()
        checkpoint_cache = Path(__file__).resolve().parents[3] / 'hf-cache' / 'hub'
        if not site.is_dir() or not checkpoint_cache.is_dir():
            self.skipTest('the shared Windows venv or offline checkpoint cache is unavailable')
        guard_and_load = '''\
import os, sys
guard = getattr(sys, '_looking_glass_import_guard', None)
if guard is not None:
    guard.assert_bound(os.environ['MLG_SOURCE_MANIFEST_SHA256'])
    guard.assert_torch_dll_inventory(os.environ['MLG_SOURCE_MANIFEST_SHA256'])
import ctypes, subprocess, torch
def deny(*args, **kwargs):
    raise AssertionError('CPU checkpoint smoke reached CUDA')
def denied(name):
    def fail(*args, **kwargs):
        raise AssertionError('CPU checkpoint smoke reached CUDA: ' + name)
    return fail
for name in dir(torch.cuda):
    value = getattr(torch.cuda, name)
    if name == '_is_compiled':
        # This wheel is CUDA-built. Return its build metadata without asking
        # the driver; Transformers imports this check even in a CPU load.
        setattr(torch.cuda, name, lambda: True)
    elif callable(value) and not isinstance(value, type):
        setattr(torch.cuda, name, denied('torch.cuda.' + name))
for name in dir(torch._C):
    value = getattr(torch._C, name)
    if name.startswith('_cuda_') and callable(value) and not isinstance(value, type):
        setattr(torch._C, name, denied('torch._C.' + name))
original_to = torch.Tensor.to
def cpu_only_to(tensor, *args, **kwargs):
    destination = kwargs.get('device', args[0] if args else None)
    if isinstance(destination, (str, torch.device)) and str(destination).split(':', 1)[0] == 'cuda':
        deny()
    return original_to(tensor, *args, **kwargs)
torch.Tensor.to = cpu_only_to
torch.Tensor.cuda = deny
for name in ('CDLL', 'WinDLL', 'PyDLL', 'OleDLL'):
    loader = getattr(ctypes, name, None)
    if loader is not None:
        def guarded_load(path, *args, _loader=loader, **kwargs):
            if any(piece in str(path).lower() for piece in ('nvcuda', 'nvml', 'libcuda', 'nvidia-ml')):
                deny()
            return _loader(path, *args, **kwargs)
        setattr(ctypes, name, guarded_load)
original_popen = subprocess.Popen
def guarded_popen(args, *pargs, **kwargs):
    executable = args[0] if isinstance(args, (list, tuple)) and args else args
    if isinstance(executable, (str, bytes, os.PathLike)) and 'nvidia-smi' in os.fsdecode(executable).lower():
        deny()
    return original_popen(args, *pargs, **kwargs)
subprocess.Popen = guarded_popen
from looking_glass.pretrained import ASSETS, load_distilgpt2
bundle = load_distilgpt2(cache_dir=CHECKPOINT_CACHE, allow_download=False)
assert type(bundle.model).__name__ == 'GPT2LMHeadModel'
assert bundle.metadata['weights_sha256'] == ASSETS['model.safetensors'][1]
assert not bundle.model.training and bundle.model.config.use_cache is False
assert next(bundle.model.parameters()).device.type == 'cpu'
assert bundle.tokenizer('Hello', add_special_tokens=False)['input_ids']
print('verified-offline-cpu-model')
'''.replace('CHECKPOINT_CACHE', repr(str(checkpoint_cache)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, private = root / 'repo', root / 'private'
            repo.mkdir()
            private.mkdir()
            package = repo / 'looking_glass'
            package.mkdir()
            (package / '__init__.py').write_text('', encoding='utf-8')
            (package / 'gpu_service.py').write_text(guard_and_load, encoding='utf-8')
            (package / 'pretrained.py').write_bytes(
                (Path(__file__).resolve().parents[1] / 'looking_glass' / 'pretrained.py').read_bytes())
            system_root = os.environ.get('SYSTEMROOT')
            if not system_root:
                self.skipTest('the Windows system path is unavailable')
            trusted_path = str((Path(system_root) / 'System32').resolve(strict=True))
            env = {key: value for key, value in os.environ.items()
                   if key.upper() in {'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP'}}
            env.update({'CUDA_VISIBLE_DEVICES': '', 'PATH': trusted_path,
                        'PYTHONHASHSEED': '17', 'PYTHONNOUSERSITE': '1',
                        'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            base = Path(sys._base_executable).resolve(strict=True)
            dry_source = (
                'import sys,json,importlib.machinery,runpy\n'
                f'sys.path.extend([{str(repo)!r}, {str(site)!r}])\n'
                'found = set()\n'
                'class RecordingFinder:\n'
                ' def find_spec(self, fullname, path=None, target=None):\n'
                '  spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)\n'
                '  if spec is not None and spec.origin: found.add(spec.origin)\n'
                '  return spec\n'
                'sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), RecordingFinder())\n'
                "runpy.run_module('looking_glass.gpu_service',run_name='__main__',alter_sys=True)\n"
                'found.update(str(getattr(m,"__file__","")) for m in '
                'list(sys.modules.values()) if getattr(m,"__file__",None))\n'
                'print(json.dumps(sorted(found)))\n'
            )
            dry = subprocess.run([str(base), '-B', '-S', '-P', '-c', dry_source],
                                 env=env, cwd=str(root), capture_output=True, text=True,
                                 timeout=180)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            paths = {Path(name).resolve() for name in json.loads(dry.stdout.splitlines()[-1])}
            for metadata in site.glob('*.dist-info'):
                paths.update(path.resolve() for path in metadata.rglob('*') if path.is_file())
            paths.update(path.resolve() for path in (site / 'torch' / 'lib').glob('*.dll'))
            def record(path, base_path):
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                return {'path': path.relative_to(base_path).as_posix(),
                        'size': path.stat().st_size, 'sha256': digest.hexdigest()}
            files = [record(path, environment) for path in paths
                     if path.is_file() and path.is_relative_to(environment)]
            self.assertTrue(files)
            repository_files = [record(repo / name, repo) for name in (
                'looking_glass/__init__.py', 'looking_glass/gpu_service.py',
                'looking_glass/pretrained.py')]
            base_digest = hashlib.sha256(base.read_bytes()).hexdigest()
            document = {'schema': 'resource-bootstrap-source-v1', 'git_commit': '0' * 40,
                        'git_tree': '1' * 40,
                        'python_executable': {'size': base.stat().st_size, 'sha256': base_digest},
                        'repository_files': sorted(repository_files, key=lambda item: item['path']),
                        'distributions': [{'name': 'test-dry-run', 'version': '0',
                                           'files': sorted(files, key=lambda item: item['path'])}]}
            encoded = json.dumps(document, sort_keys=True, ensure_ascii=False,
                                 separators=(',', ':')).encode('utf-8')
            launch = prepare_owner_launch(project_root=repo, site_packages=site,
                environment_root=environment, private_parent=private, gpu_uuid=UUID,
                module='looking_glass.gpu_service', arguments=(), source_manifest=encoded,
                trusted_path=trusted_path, inherited_environment=env)
            launch.verify_ready()
            child_env = dict(launch.environment, CUDA_VISIBLE_DEVICES='')
            verified = subprocess.run(launch.command, env=child_env, cwd=str(root),
                                      capture_output=True, text=True, timeout=300)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertIn('verified-offline-cpu-model', verified.stdout)
            self.assertFalse(any(launch.pycache_prefix.iterdir()))


class ConcreteBackendContractTests(unittest.TestCase):
    def test_runtime_binding_stops_if_supervision_is_lost_between_queries(self):
        import torch
        events = []
        session = FakeSession(events)
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        def count():
            events.append('count')
            session.reject_check = True
            return 1
        def properties(index):
            events.append('properties')
        with patch.object(torch.cuda, 'device_count', side_effect=count), \
             patch.object(torch.cuda, 'get_device_properties', side_effect=properties):
            with self.assertRaisesRegex(RuntimeError, 'supervised lease'):
                backend.runtime_binding()
        self.assertEqual(events.count('count'), 1)
        self.assertNotIn('properties', events)

    def test_telemetry_stops_if_supervision_is_lost_between_queries(self):
        import torch
        events = []
        session = FakeSession(events)
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        def available():
            events.append('availability')
            session.reject_check = True
            return True
        with patch.object(torch.cuda, 'is_available', side_effect=available), \
             patch.object(torch.cuda, 'mem_get_info', side_effect=lambda *args: events.append('memory')):
            with self.assertRaisesRegex(RuntimeError, 'supervised lease'):
                backend.telemetry()
        self.assertEqual(events.count('availability'), 1)
        self.assertNotIn('memory', events)

    def test_synchronization_fails_if_supervision_is_lost_during_call(self):
        import torch
        events = []
        session = FakeSession(events)
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        def synchronize(index):
            events.append('sync')
            session.reject_check = True
        with patch.object(torch.cuda, 'synchronize', side_effect=synchronize):
            with self.assertRaisesRegex(RuntimeError, 'supervised lease'):
                backend.synchronize()
        self.assertEqual(events.count('sync'), 1)

    def test_runtime_count_cannot_query_properties_after_quarantine(self):
        import torch
        events, latched = [], [False]
        backend = _TorchServiceBackend(session=FakeSession(events), device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        def count():
            latched[0] = True
            return 1
        def properties(index):
            events.append('properties')
        with patch.object(gpu_service._QUARANTINED, 'is_set', side_effect=lambda: latched[0]), \
             patch.object(torch.cuda, 'device_count', side_effect=count), \
             patch.object(torch.cuda, 'get_device_properties', side_effect=properties):
            with self.assertRaisesRegex(RuntimeError, 'quarantined'):
                backend.runtime_binding()
        self.assertNotIn('properties', events)

    def test_telemetry_stops_between_individual_cuda_calls_on_quarantine(self):
        import torch
        events, latched = [], [False]
        backend = _TorchServiceBackend(session=FakeSession(events), device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        def available():
            latched[0] = True
            return True
        def memory(*args):
            events.append('memory-query')
            return (7 * GIB, 8 * GIB)
        with patch.object(gpu_service._QUARANTINED, 'is_set', side_effect=lambda: latched[0]), \
             patch.object(torch.cuda, 'is_available', side_effect=available), \
             patch.object(torch.cuda, 'mem_get_info', side_effect=memory):
            with self.assertRaisesRegex(RuntimeError, 'quarantined'):
                backend.telemetry()
        self.assertNotIn('memory-query', events)

    def test_concrete_runtime_check_rejects_latched_owner_before_cuda_query(self):
        events = []
        backend = _TorchServiceBackend(session=FakeSession(events), device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        backend.runtime_binding = lambda: events.append('cuda-query') or (UUID, 1)
        with patch.object(gpu_service._QUARANTINED, 'is_set', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'quarantined'):
                backend._check_runtime()
        self.assertNotIn('cuda-query', events)

    def test_actual_request_factory_only_accepts_profiled_token_values(self):
        import torch
        from looking_glass.memory_budget import MemoryBudget
        session = FakeSession([])
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        backend._resolved_keys = ()
        backend.runtime_binding = lambda: (UUID, 1)
        backend.synchronize = lambda: None
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                return self
        class WarmupWorker:
            quarantined = False
            def execute(self, request):
                self.warmup = request
                return SimpleNamespace(logits=torch.zeros((1, 2, 50257), dtype=torch.float32))
        class FakeServer:
            def __init__(self, broker, **kwargs):
                self.request_factory = kwargs['request_factory']
        factory = lambda request_id, inputs, seed: (request_id, inputs, seed)
        workers = []
        def make_worker(*args):
            worker = WarmupWorker()
            workers.append(worker)
            return worker
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: factory)
        load = SimpleNamespace(key=SimpleNamespace(model_id='c' * 64))
        cold = SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT)
        warm = SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT)
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', return_value=SimpleNamespace(model=CpuModel())), \
             patch('looking_glass.resident_worker.ResidentWorker', return_value=object()), \
             patch('looking_glass.budgeted_worker.DeviceAdmission', return_value=object()), \
             patch('looking_glass.budgeted_worker.BudgetedWorker', side_effect=make_worker), \
             patch('looking_glass.broker.Broker', return_value=object()), \
             patch('looking_glass.ipc.IPCServer', FakeServer):
            _, _, server = backend.construct(budget, lease, (load, cold, warm))
        request_factory = server.request_factory
        self.assertEqual(workers[0].warmup.seed, 17)
        self.assertEqual(workers[0].warmup.inputs['input_ids'].tolist(), [[15496, 995]])
        for seed in (29, 43, 71):
            self.assertEqual(request_factory('r', {'input_ids': [[15496, 995]]}, seed),
                             ('r', {'input_ids': [[15496, 995]]}, seed))
        for invalid in ({'input_ids': [[15496, 996]]}, {'input_ids': [[15496.0, 995]]},
                        {'input_ids': [[15496]]}, {'input_ids': [[15496, 995]], 'mask': [[1, 1]]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                request_factory('r', invalid, 29)
        for invalid_seed in (0, 17, 29.0, True):
            with self.subTest(seed=invalid_seed), self.assertRaises(ValueError):
                request_factory('r', {'input_ids': [[15496, 995]]}, invalid_seed)

    def test_invalid_team_configuration_is_rejected_before_any_gpu_work(self):
        for teams in ((), ('bad name',), ('alpha', 'alpha'), ('x' * 65,),
                      tuple(f'team{i}' for i in range(129)), (['unhashable'],)):
            with self.subTest(teams=teams), self.assertRaises(ValueError):
                _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
                    cache_dir=Path.cwd(), team_ids=teams,
                    live_key_builder=lambda identity: (), verify_loaded_model=lambda model: True,
                    verify_device_model=lambda *args: True)

    def test_actual_cpu_model_verifier_is_required(self):
        with self.assertRaisesRegex(TypeError, 'model verifier'):
            _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
                cache_dir=Path.cwd(), team_ids=('alpha',),
                live_key_builder=lambda identity: (), verify_loaded_model=None,
                verify_device_model=lambda *args: True)

    def test_actual_device_model_verifier_is_required(self):
        for verifier in (None, False):
            with self.subTest(verifier=verifier), self.assertRaisesRegex(TypeError, 'device model verifier'):
                _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
                    cache_dir=Path.cwd(), team_ids=('alpha',),
                    live_key_builder=lambda identity: (), verify_loaded_model=lambda model: True,
                    verify_device_model=verifier)

    def test_cpu_load_is_seeded_immediately_before_offline_loader(self):
        import torch
        from looking_glass.memory_budget import MemoryBudget
        events = []
        backend = _TorchServiceBackend(session=FakeSession(events), device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        backend._resolved_keys = ()
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        profiles = (None, SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT),
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT))
        generator = SimpleNamespace(manual_seed=lambda seed: events.append(('seed', seed)))
        def cpu_load(**kwargs):
            events.append(('load', kwargs['allow_download']))
            raise ValueError('stop after CPU loader entry')
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch.object(torch.random, 'default_generator', generator), \
             patch('looking_glass.pretrained.load_distilgpt2', side_effect=cpu_load):
            with self.assertRaisesRegex(ValueError, 'stop after CPU loader entry'):
                backend.construct(budget, lease, profiles)
        self.assertEqual(events[-2:], [('seed', 17), ('load', False)])

    def test_construct_refuses_a_smaller_resident_lease_before_cpu_load(self):
        from looking_glass.memory_budget import MemoryBudget
        backend = _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        backend._resolved_keys = ()
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', side_effect=AssertionError('loader reached')):
            with self.assertRaisesRegex(RuntimeError, 'resident-load reservation'):
                backend.construct(budget, lease, (None, None, None))

    def test_copy_flags_and_postcopy_verifier_gate_worker_creation(self):
        import torch
        from looking_glass.memory_budget import MemoryBudget

        class WorkerReached(Exception):
            pass

        def run_case(initial_flags, *, after_copy=None, verification=True, other_copy=False,
                     quarantine_on_verification=False):
            events, flags, latched = [], list(initial_flags), [False]
            session, device_lease = FakeSession(events), object()

            class CpuModel:
                training = False
                config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                         vocab_size=50257, n_positions=1024)
                def to(self, **kwargs):
                    events.append('copy')
                    if after_copy is not None:
                        flags[:] = after_copy
                    return object() if other_copy else self

            model = CpuModel()

            def verify_device(received, identity, verified_session, verified_lease):
                events.append('verify-device')
                self.assertIs(received, model)
                self.assertIs(identity, session.identity)
                self.assertIs(verified_session, session)
                self.assertIs(verified_lease, device_lease)
                if quarantine_on_verification:
                    latched[0] = True
                if isinstance(verification, Exception):
                    raise verification
                return verification

            backend = _TorchServiceBackend(session=session, device_lease=device_lease,
                cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
                verify_loaded_model=lambda model: True, verify_device_model=verify_device)
            backend._resolved_keys = ()
            backend.runtime_binding = lambda: (UUID, 1)
            backend.synchronize = lambda: events.append('synchronize')
            budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                                  max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
            load = SimpleNamespace(key=SimpleNamespace(model_id='c' * 64))

            def make_resident(*args, **kwargs):
                events.append('worker')
                raise WorkerReached()

            with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
                 patch.object(gpu_service._QUARANTINED, 'is_set', side_effect=lambda: latched[0]), \
                 patch('looking_glass.pretrained.load_distilgpt2', return_value=SimpleNamespace(model=model)), \
                 patch('looking_glass.resident_worker.ResidentWorker', side_effect=make_resident), \
                 patch.object(torch.__future__, 'get_overwrite_module_params_on_conversion', side_effect=lambda: flags[0]), \
                 patch.object(torch.__future__, 'get_swap_module_params_on_conversion', side_effect=lambda: flags[1]):
                try:
                    backend.construct(budget, lease, (load,
                        SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT),
                        SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT)))
                except Exception as exc:
                    return events, exc
            self.fail('construction unexpectedly completed')

        for flags in ((True, False), (False, True), (None, False)):
            with self.subTest(initial_flags=flags):
                events, error = run_case(flags)
                self.assertIsInstance(error, RuntimeError)
                self.assertNotIn('copy', events)
                self.assertNotIn('worker', events)
        for flags in ((True, False), (False, True)):
            with self.subTest(changed_flags=flags):
                events, error = run_case((False, False), after_copy=flags)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn('copy', events)
                self.assertIn('synchronize', events)
                self.assertNotIn('verify-device', events)
                self.assertNotIn('worker', events)
        for result in (False, None, 1, ValueError('invalid post-copy model')):
            with self.subTest(device_verification=result):
                events, error = run_case((False, False), verification=result)
                self.assertIsInstance(error, (RuntimeError, ValueError))
                self.assertLess(events.index('copy'), events.index('synchronize'))
                self.assertLess(events.index('synchronize'), events.index('verify-device'))
                self.assertNotIn('worker', events)
        events, error = run_case((False, False), quarantine_on_verification=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertNotIn('worker', events)
        events, error = run_case((False, False), other_copy=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertNotIn('worker', events)
        events, error = run_case((False, False))
        self.assertIsInstance(error, WorkerReached)
        self.assertLess(events.index('copy'), events.index('synchronize'))
        self.assertLess(events.index('synchronize'), events.index('verify-device'))
        self.assertLess(events.index('verify-device'), events.index('worker'))
        self.assertEqual(events.count('synchronize'), 2)
        self.assertGreater(events.index('synchronize', events.index('verify-device')),
                           events.index('verify-device'))

    def test_load_cold_and_warm_profiles_must_all_match_live_keys_exactly(self):
        from looking_glass.resource_profile import ProfileRegistry, ProfileUnavailable
        keys, registry = measured_keys_and_registry()
        session = FakeSession([])
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha', 'beta'),
            live_key_builder=lambda identity: keys, verify_loaded_model=lambda model: True,
            verify_device_model=lambda *args: True)
        self.assertEqual(backend.live_keys(session.identity), keys)
        load, cold, warm = backend.resolve_profiles(registry, keys)
        self.assertEqual(load.key, keys[0])
        self.assertEqual(cold.model_id, keys[1].model_id)
        self.assertEqual(warm.model_id, keys[2].model_id)
        self.assertEqual(cold.shape, (1, 2))
        self.assertEqual(warm.shape, (1, 2))
        # The measured estimate is an eligibility check. The worker must
        # reserve the full predeclared job amount, even when the estimate is smaller.
        self.assertEqual(cold.estimated_incremental_bytes, JOB_COMMITMENT)
        self.assertEqual(warm.estimated_incremental_bytes, JOB_COMMITMENT)
        missing = ProfileRegistry((load,), approved_profile_ids=(load.profile_id,))
        with self.assertRaises(ProfileUnavailable):
            backend.resolve_profiles(missing, keys)

    def test_oversize_forward_estimate_is_rejected_before_loading(self):
        from looking_glass.resource_profile import ProfileRegistry
        keys, registry = measured_keys_and_registry()
        profiles = [registry.resolve(key) for key in keys]
        profiles[1] = replace(profiles[1], margin_bytes=JOB_COMMITMENT + 1)
        oversized = ProfileRegistry(tuple(profiles),
                                    approved_profile_ids=tuple(p.profile_id for p in profiles))
        backend = _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: keys,
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        with self.assertRaisesRegex(ValueError, 'commitment'):
            backend.resolve_profiles(oversized, keys)
        self.assertIsNone(backend._resolved_keys)

    def test_cold_and_warm_workers_hold_fixed_job_lease_during_execution(self):
        import torch
        from torch import nn
        from looking_glass.memory_budget import MemoryBudget
        from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker

        keys, registry = measured_keys_and_registry()
        backend = _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: keys,
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        _, cold, warm = backend.resolve_profiles(registry, keys)
        cold, warm = replace(cold, device='cpu'), replace(warm, device='cpu')
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        resident_lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        observed = []

        class TinyModel(nn.Module):
            def forward(self, input_ids):
                observed.append(tuple(sorted(lease.estimated_incremental_bytes
                                             for lease in budget.active_leases())))
                return input_ids.to(torch.float32).unsqueeze(-1)

        resident = ResidentWorker(cold.model_id, TinyModel().eval(), device='cpu',
                                  max_batch_size=1, max_sequence_length=2,
                                  max_input_bytes=64 * 1024, max_output_bytes=512 * 1024)
        admission = DeviceAdmission(budget, device='cpu', max_estimated_incremental_bytes=JOB_COMMITMENT)
        cold_worker = BudgetedWorker(resident, admission, (cold,))
        warm_worker = BudgetedWorker(resident, admission, (warm,))
        for worker, seed in ((cold_worker, 17), (warm_worker, 29)):
            request = ForwardRequest('fixed-test', {'input_ids': torch.tensor([[15496, 995]])}, seed)
            worker.execute(request)
            self.assertEqual(budget.active_leases(), (resident_lease,))
        self.assertEqual(observed, [(JOB_COMMITMENT, LOAD_COMMITMENT)] * 2)
        warm_worker.close()
        resident_lease.release()

    def test_owner_rechecks_supervision_after_cpu_loading_before_cuda_copy(self):
        from looking_glass.memory_budget import MemoryBudget
        events = []
        session = FakeSession(events)
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        backend._resolved_keys = ()
        backend.runtime_binding = lambda: (UUID, 1)
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                events.append('cuda-copy')
                raise AssertionError('copy must not happen after supervisor revocation')
        def cpu_load(**kwargs):
            events.append('cpu-load')
            session.reject_check = True
            return SimpleNamespace(model=CpuModel())
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', side_effect=cpu_load):
            with self.assertRaisesRegex(RuntimeError, 'supervised lease'):
                backend.construct(budget, lease, (None,
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT),
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT)))
        self.assertNotIn('cuda-copy', events)

    def test_owner_rechecks_live_headroom_after_cpu_loading_before_cuda_copy(self):
        from looking_glass.memory_budget import MemoryBudget
        events = []
        session = FakeSession(events)
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True, verify_device_model=lambda *args: True)
        backend._resolved_keys = ()
        backend.runtime_binding = lambda: (UUID, 1)
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                events.append('cuda-copy')
                raise AssertionError('copy must not happen without live headroom')
        free = [7 * GIB]
        def cpu_load(**kwargs):
            free[0] = GIB
            return SimpleNamespace(model=CpuModel())
        budget = MemoryBudget(lambda: CudaMemorySnapshot(free[0], 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', side_effect=cpu_load):
            with self.assertRaises(AdmissionError):
                backend.construct(budget, lease, (None,
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT),
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT)))
        self.assertNotIn('cuda-copy', events)

    def test_model_verification_failure_stops_before_cuda_copy(self):
        from looking_glass.memory_budget import MemoryBudget
        events = []
        session = FakeSession(events)
        def reject_model(model):
            raise ValueError('model bytes do not match checkpoint')
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=reject_model, verify_device_model=lambda *args: True)
        backend._resolved_keys = ()
        backend.runtime_binding = lambda: (UUID, 1)
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                events.append('cuda-copy')
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', return_value=SimpleNamespace(model=CpuModel())):
            with self.assertRaisesRegex(ValueError, 'do not match'):
                backend.construct(budget, lease, (None,
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT),
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT)))
        self.assertNotIn('cuda-copy', events)

    def test_model_verifier_must_explicitly_return_true(self):
        from looking_glass.memory_budget import MemoryBudget
        session = FakeSession([])
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: False, verify_device_model=lambda *args: True)
        backend._resolved_keys = ()
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', return_value=SimpleNamespace(model=CpuModel())):
            with self.assertRaisesRegex(RuntimeError, 'model verification'):
                backend.construct(budget, lease, (None,
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT),
                    SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT)))


class FakeBackend:
    def __init__(self, events, *, free=7 * GIB, missing=False,
                 quarantined=False, broker_quarantined=False, dangling=False, failure=False):
        self.events = events
        self.free = free
        self.missing = missing
        self.dangling = dangling
        self.failure = failure
        self.worker = FakeWorker()
        self.worker.quarantined = quarantined
        self.broker = FakeBroker()
        self.broker.quarantined = broker_quarantined
        self.budget = None
        self.load_lease = None
        self.keys = service_keys()

    def runtime_binding(self):
        self.events.append('runtime-binding')
        return UUID, 1

    def live_keys(self, identity):
        self.events.append('derive-live-keys')
        return self.keys

    def resolve_profiles(self, registry, keys):
        self.events.append('resolve-profiles')
        if self.missing:
            raise LookupError('no approved profile')
        return (FakeProfile(GIB), FakeProfile(JOB_COMMITMENT), FakeProfile(JOB_COMMITMENT))

    def telemetry(self):
        self.events.append('telemetry')
        return CudaMemorySnapshot(self.free, 8 * GIB, 0, 0)

    def construct(self, budget, load_lease, profiles):
        self.events.append('load-and-warmup')
        self.budget, self.load_lease = budget, load_lease
        assert load_lease in budget.active_leases()
        if self.dangling:
            budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        return self.worker, self.broker, object()

    def synchronize(self):
        self.events.append('synchronize')

    def serve(self, server, broker, worker, stop_event, announce, verify_drain):
        self.events.append('serve-and-stop')
        verified = verify_drain()
        return FakeResult('worker_failed' if self.failure else 'clean', verified)


class OwnerOrderTests(unittest.TestCase):
    def setUp(self):
        # These are CPU-only fake resources. A real GPU process is never reset
        # after quarantine; it must be replaced by the supervisor.
        _reset_fake_owner()

    def tearDown(self):
        _reset_fake_owner()

    def setup_run(self, **kwargs):
        events = []
        session = FakeSession(events)
        backend = FakeBackend(events, **kwargs)
        return events, session, backend

    def run_owner(self, session, backend):
        return _run_bound_owner(session, object(), backend=backend,
                                registry=object(), stop_event=threading.Event(),
                                announce=lambda server: None)

    def test_resident_lease_uses_fixed_commitment_not_smaller_profile_estimate(self):
        events, session, backend = self.setup_run()
        result = self.run_owner(session, backend)
        self.assertTrue(result.teardown_verified)
        self.assertEqual(backend.load_lease.estimated_incremental_bytes, LOAD_COMMITMENT)
        self.assertFalse(backend.load_lease.released)

    def test_oversize_load_profile_fails_before_memory_telemetry(self):
        events, session, backend = self.setup_run()
        backend.resolve_profiles = lambda registry, keys: (
            FakeProfile(LOAD_COMMITMENT + 1), FakeProfile(JOB_COMMITMENT), FakeProfile(JOB_COMMITMENT))
        with self.assertRaisesRegex(ValueError, 'commitment'):
            self.run_owner(session, backend)
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)

    def test_device_total_mismatch_prevents_initial_model_load(self):
        events, session, backend = self.setup_run()
        backend.telemetry = lambda: CudaMemorySnapshot(8 * GIB, 9 * GIB, 0, 0)
        with self.assertRaisesRegex(RuntimeError, 'device total'):
            self.run_owner(session, backend)
        self.assertNotIn('load-and-warmup', events)
        self.assertNotIn('attest', events)

    def test_changed_device_total_prevents_job_admission_after_load(self):
        events, session, backend = self.setup_run()
        def construct(budget, resident_lease, profiles):
            events.append('loaded')
            backend.telemetry = lambda: CudaMemorySnapshot(8 * GIB, 9 * GIB, 0, 0)
            budget.try_acquire(JOB_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            events.append('job-started')
        backend.construct = construct
        with self.assertRaisesRegex(RuntimeError, 'device total'):
            self.run_owner(session, backend)
        self.assertIn('loaded', events)
        self.assertNotIn('job-started', events)
        self.assertNotIn('attest', events)
        self.assertFalse(gpu_service._OWNER_CUSTODY[5].released)


    def test_failed_post_verifier_sync_retains_model_and_lease_without_attesting(self):
        events = []
        session, device_lease = FakeSession(events), object()
        keys = service_keys()
        backend = _TorchServiceBackend(session=session, device_lease=device_lease,
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: keys,
            verify_loaded_model=lambda model: True,
            verify_device_model=lambda *args: events.append('verified-device') or True)
        backend._resolved_keys = keys
        backend.runtime_binding = lambda: (UUID, 1)
        backend.telemetry = lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0)
        backend.resolve_profiles = lambda registry, selected: (
            SimpleNamespace(key=keys[0], estimated_incremental_bytes=GIB),
            SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT),
            SimpleNamespace(estimated_incremental_bytes=JOB_COMMITMENT))

        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                events.append('copy')
                return self

        model = CpuModel()
        model_ref = weakref.ref(model)
        syncs = []
        def sync():
            syncs.append(None)
            if len(syncs) == 2:
                raise RuntimeError('second sync failed')
        backend.synchronize = sync
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', return_value=SimpleNamespace(model=model)), \
             patch('looking_glass.resident_worker.ResidentWorker', side_effect=AssertionError('worker started')):
            with self.assertRaisesRegex(RuntimeError, 'second sync failed'):
                _run_bound_owner(session, device_lease, backend=backend, registry=object(),
                                 stop_event=threading.Event(), announce=lambda server: None)
        self.assertEqual(len(syncs), 2)
        self.assertIn('verified-device', events)
        self.assertNotIn('attest', events)
        self.assertTrue(gpu_service._owner_is_unsafe())
        lease = gpu_service._OWNER_CUSTODY[5]
        self.assertFalse(lease.released)
        lease_ref = weakref.ref(lease)
        del lease, model, backend
        gc.collect()
        self.assertIsNotNone(model_ref())
        self.assertIsNotNone(lease_ref())

    def test_supervisor_check_precedes_any_runtime_call(self):
        events, session, backend = self.setup_run()
        session.reject_check = True
        with self.assertRaisesRegex(RuntimeError, 'supervised lease'):
            self.run_owner(session, backend)
        self.assertEqual(events, ['session-check'])

    def test_identityless_supervision_is_not_accepted(self):
        events, session, backend = self.setup_run()
        session.identity = None
        with self.assertRaisesRegex(RuntimeError, 'identity-aware'):
            self.run_owner(session, backend)
        self.assertEqual(events, ['session-check'])

    def test_identity_mismatch_prevents_telemetry_and_loading(self):
        events, session, backend = self.setup_run()
        session.reject_binding = True
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            self.run_owner(session, backend)
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)

    def test_missing_profile_prevents_telemetry_and_loading(self):
        events, session, backend = self.setup_run(missing=True)
        with self.assertRaisesRegex(LookupError, 'no approved profile'):
            self.run_owner(session, backend)
        self.assertLess(events.index('verify-binding'), events.index('derive-live-keys'))
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)

    def test_unsupported_live_signature_is_rejected_before_profile_or_telemetry(self):
        events, session, backend = self.setup_run()
        load, cold, warm = backend.keys
        backend.keys = (load, replace(cold, sequence_length=8),
                        replace(warm, sequence_length=8))
        with self.assertRaisesRegex(ValueError, 'signature'):
            self.run_owner(session, backend)
        self.assertNotIn('resolve-profiles', events)
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)

    def test_headroom_is_checked_before_model_loading(self):
        events, session, backend = self.setup_run(free=GIB)
        with self.assertRaises(AdmissionError):
            self.run_owner(session, backend)
        self.assertNotIn('load-and-warmup', events)
        self.assertNotIn('attest', events)

    def test_quarantine_during_telemetry_stops_before_construct(self):
        events, session, backend = self.setup_run()
        original = backend.telemetry
        def telemetry_latches():
            value = original()
            gpu_service._QUARANTINED.set()
            return value
        backend.telemetry = telemetry_latches
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            self.run_owner(session, backend)
        self.assertNotIn('load-and-warmup', events)
        self.assertNotIn('attest', events)

    def test_quarantine_during_live_key_build_stops_profile_resolution(self):
        events, session, backend = self.setup_run()
        original = backend.live_keys
        def keys_latch(identity):
            keys = original(identity)
            gpu_service._QUARANTINED.set()
            return keys
        backend.live_keys = keys_latch
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            self.run_owner(session, backend)
        self.assertNotIn('resolve-profiles', events)
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)


    def test_resident_reservation_is_held_until_exit_and_attestation_is_last(self):
        events, session, backend = self.setup_run()
        result = self.run_owner(session, backend)
        self.assertTrue(result.teardown_verified)
        self.assertEqual(events.count('load-and-warmup'), 1)
        self.assertLess(events.index('verify-binding'), events.index('telemetry'))
        self.assertLess(events.index('telemetry'), events.index('load-and-warmup'))
        self.assertGreaterEqual(events.count('synchronize'), 1)
        self.assertEqual(events[-1], 'attest')
        self.assertEqual(backend.budget.active_leases(), (backend.load_lease,))
        self.assertFalse(backend.load_lease.released)

    def test_clean_owner_retains_load_lease_and_device_resources_until_exit(self):
        events, session, backend = self.setup_run()
        result = self.run_owner(session, backend)
        self.assertTrue(result.teardown_verified)
        worker_ref = weakref.ref(backend.worker)
        load_ref = weakref.ref(backend.load_lease)
        budget_ref = weakref.ref(backend.budget)
        del backend
        gc.collect()
        self.assertIsNotNone(worker_ref())
        self.assertIsNotNone(load_ref())
        self.assertIsNotNone(budget_ref())

    def test_clean_owner_cannot_be_reused_in_same_process(self):
        events, session, backend = self.setup_run()
        self.assertTrue(self.run_owner(session, backend).teardown_verified)
        with self.assertRaisesRegex(RuntimeError, 'already been used'):
            self.run_owner(session, FakeBackend(events))

    def test_workload_failure_can_have_verified_teardown_without_success(self):
        events, session, backend = self.setup_run(failure=True)
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'worker_failed')
        self.assertTrue(result.teardown_verified)
        self.assertIn('attest', events)

    def test_startup_failure_with_verified_drain_attests_without_claiming_success(self):
        events, session, backend = self.setup_run()
        def startup_failure(server, broker, worker, stop_event, announce, verify):
            return FakeResult('shutdown_incomplete', verify())
        backend.serve = startup_failure
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'shutdown_incomplete')
        self.assertTrue(result.teardown_verified)
        self.assertIn('attest', events)

    def test_quarantine_or_extra_lease_prevents_attestation(self):
        for kwargs in ({'quarantined': True}, {'broker_quarantined': True}, {'dangling': True}):
            with self.subTest(kwargs=kwargs):
                _reset_fake_owner()
                events, session, backend = self.setup_run(**kwargs)
                result = self.run_owner(session, backend)
                self.assertFalse(result.teardown_verified)
                self.assertNotIn('attest', events)

    def test_broker_quarantine_during_sync_prevents_attestation(self):
        events, session, backend = self.setup_run()
        def quarantine_during_sync():
            events.append('synchronize')
            backend.broker.quarantined = True
        backend.synchronize = quarantine_during_sync
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertTrue(gpu_service._QUARANTINED.is_set())
        self.assertNotIn('attest', events)

    def test_synchronization_failure_prevents_attestation(self):
        events, session, backend = self.setup_run()
        def fail_sync():
            events.append('synchronize')
            raise RuntimeError('device failed to drain')
        backend.synchronize = fail_sync
        result = self.run_owner(session, backend)
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('attest', events)
        self.assertFalse(backend.load_lease.released)

    def test_failed_drain_latches_and_retains_resources_until_process_exit(self):
        events, session, backend = self.setup_run()
        worker_ref = weakref.ref(backend.worker)
        backend.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('lost drain'))
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertTrue(gpu_service._QUARANTINED.is_set())
        del backend
        gc.collect()
        self.assertIsNotNone(worker_ref())
        next_backend = FakeBackend(events)
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            self.run_owner(session, next_backend)

    def test_failed_drain_retains_known_resources_if_error_bookkeeping_ooms(self):
        events, session, backend = self.setup_run()
        worker_ref = weakref.ref(backend.worker)
        class AppendFails(list):
            def append(self, value):
                raise MemoryError('injected bookkeeping failure')
        backend.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('lost drain'))
        with patch.object(gpu_service, '_QUARANTINE_HOLDS', AppendFails()):
            result = self.run_owner(session, backend)
            lease_ref = weakref.ref(backend.load_lease)
            self.assertEqual(result.status, 'quarantined')
            del backend
            gc.collect()
            self.assertIsNotNone(worker_ref())
            self.assertIsNotNone(lease_ref())
        self.assertNotIn('attest', events)

    def test_published_lease_survives_interrupted_acquire_receipt(self):
        from looking_glass.memory_budget import MemoryBudget
        events, session, backend = self.setup_run()
        refs = {}
        class InterruptBudget(MemoryBudget):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                refs['budget'] = weakref.ref(self)
            def acquire(self, *args, **kwargs):
                lease = super().acquire(*args, **kwargs)
                refs['lease'] = weakref.ref(lease)
                raise MemoryError('injected interruption after lease publication')
        class AppendFails(list):
            def append(self, value):
                raise MemoryError('injected bookkeeping failure')
        with patch.object(gpu_service, 'MemoryBudget', InterruptBudget), \
             patch.object(gpu_service, '_QUARANTINE_HOLDS', AppendFails()):
            with self.assertRaisesRegex(MemoryError, 'after lease publication'):
                self.run_owner(session, backend)
            del backend
            gc.collect()
            self.assertIsNotNone(refs['budget']())
            self.assertIsNotNone(refs['lease']())
        self.assertNotIn('attest', events)

    def test_first_failure_is_retained_if_later_quarantine_fails(self):
        events, session, backend = self.setup_run()
        refs, calls = {}, []
        class Payload:
            pass
        class DrainFailure(RuntimeError):
            pass
        def fail_sync():
            payload = Payload()
            refs['payload'] = weakref.ref(payload)
            exc = DrainFailure('first failure')
            exc.payload = payload
            raise exc
        class AppendFails(list):
            def append(self, value):
                raise MemoryError('injected bookkeeping failure')
        original_quarantine = gpu_service._quarantine
        def interrupted_quarantine(*args):
            calls.append(None)
            if len(calls) == 2:
                raise MemoryError('later quarantine failed')
            return original_quarantine(*args)
        backend.synchronize = fail_sync
        with patch.object(gpu_service, '_QUARANTINE_HOLDS', AppendFails()), \
             patch.object(gpu_service, '_quarantine', interrupted_quarantine):
            with self.assertRaisesRegex(MemoryError, 'later quarantine failed'):
                self.run_owner(session, backend)
            del backend
            gc.collect()
            self.assertIsNotNone(refs['payload']())
        self.assertEqual(len(calls), 3)
        self.assertNotIn('attest', events)

    def test_false_teardown_claim_is_independently_rechecked(self):
        events, session, backend = self.setup_run()
        def dishonest_serve(server, broker, worker, stop_event, announce, verify):
            worker.quarantined = True
            return FakeResult(teardown_verified=True)
        backend.serve = dishonest_serve
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('attest', events)

    def test_failed_drain_cannot_be_healed_by_later_clean_claim(self):
        events, session, backend = self.setup_run()
        calls = []
        def first_sync_fails():
            calls.append(None)
            if len(calls) == 1:
                raise RuntimeError('first synchronization failed')
        def falsely_clean(server, broker, worker, stop_event, announce, verify):
            self.assertFalse(verify())
            return FakeResult('clean', True)
        backend.synchronize = first_sync_fails
        backend.serve = falsely_clean
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertEqual(len(calls), 1)
        self.assertNotIn('attest', events)

    def test_failed_drain_stays_quarantined_if_event_notification_fails(self):
        events, session, backend = self.setup_run()
        calls = []
        def first_sync_fails():
            calls.append(None)
            if len(calls) == 1:
                raise RuntimeError('first synchronization failed')
        def falsely_clean(server, broker, worker, stop_event, announce, verify):
            try:
                verify()
            except MemoryError:
                pass
            return FakeResult('clean', True)
        backend.synchronize = first_sync_fails
        backend.serve = falsely_clean
        with patch.object(gpu_service._QUARANTINED, 'set', side_effect=MemoryError('notification')):
            result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertEqual(len(calls), 1)
        self.assertNotIn('attest', events)

    def test_pending_drain_survives_interrupted_latch_publication(self):
        events, session, backend = self.setup_run()
        syncs, notifications = [], []
        def fail_sync():
            syncs.append(None)
            raise RuntimeError('device drain failed')
        def falsely_clean(server, broker, worker, stop_event, announce, verify):
            try:
                verify()
            except KeyboardInterrupt:
                pass
            return FakeResult('clean', True)
        original = gpu_service._mark_unsafe
        def interrupted_mark():
            notifications.append(None)
            if len(notifications) == 1:
                raise KeyboardInterrupt('before latch publication')
            original()
        backend.synchronize = fail_sync
        backend.serve = falsely_clean
        with patch.object(gpu_service, '_mark_unsafe', interrupted_mark):
            result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertEqual(len(syncs), 1)
        self.assertNotIn('attest', events)

    def test_observed_broker_quarantine_cannot_be_healed_by_later_clean_claim(self):
        events, session, backend = self.setup_run()
        def falsely_clean(server, broker, worker, stop_event, announce, verify):
            broker.quarantined = True
            self.assertFalse(verify())
            broker.quarantined = False
            return FakeResult('clean', True)
        backend.serve = falsely_clean
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('synchronize', events)
        self.assertNotIn('attest', events)

    def test_quarantine_latched_during_sync_prevents_attestation(self):
        events, session, backend = self.setup_run()
        backend.synchronize = lambda: gpu_service._QUARANTINED.set()
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('attest', events)

    def test_revoked_supervision_during_load_keeps_marker(self):
        events, session, backend = self.setup_run()
        original = backend.construct
        def revoked(*args):
            components = original(*args)
            session.reject_check = True
            return components
        backend.construct = revoked
        result = self.run_owner(session, backend)
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('attest', events)


class CpuComposedServiceTests(unittest.TestCase):
    def setUp(self):
        _reset_fake_owner()

    def tearDown(self):
        _reset_fake_owner()

    def test_owner_roots_real_worker_buffers_after_fake_device_drain_failure(self):
        import torch
        from contextlib import nullcontext
        from unittest.mock import Mock
        from torch import nn
        from looking_glass import resident_worker as resident_module
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker, WorkerSynchronizationError

        refs = {}
        class ObservedModel(nn.Module):
            def forward(self, input_ids):
                logits = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
                refs['input'] = weakref.ref(input_ids)
                refs['logits'] = weakref.ref(logits)
                return {'logits': logits}
        class NoRetentionSpace:
            def append(self, value):
                raise MemoryError('diagnostic retention unavailable')
        class CustodyBackend(FakeBackend):
            def __init__(self, events):
                super().__init__(events)
                model = ObservedModel().eval()
                refs['model'] = weakref.ref(model)
                self.worker = ResidentWorker('fake-cuda', model)
                self.worker._device = torch.device('cuda:0')
                refs['worker'] = weakref.ref(self.worker)
            def serve(self, server, broker, worker, stop_event, announce, verify_drain):
                try:
                    worker.execute(ForwardRequest('fake-failure', {'input_ids': torch.tensor([[1, 2]])}))
                except WorkerSynchronizationError:
                    return FakeResult('quarantined', False)
                raise AssertionError('the fake device drain should have failed')
        def fake_to(tensor, *args, **kwargs):
            if kwargs.get('copy'):
                result = tensor.clone()
                refs['offload'] = weakref.ref(result)
                return result
            return tensor
        fake_process_event = threading.Event()
        backend = CustodyBackend([])
        with patch.object(resident_module, '_process_quarantined', fake_process_event), \
             patch.object(resident_module, '_quarantine_references', NoRetentionSpace()), \
             patch.object(fake_process_event, 'set', side_effect=MemoryError('notification unavailable')), \
             patch('torch.random.fork_rng', return_value=nullcontext()), \
             patch('torch.cuda.default_generators', [Mock()]), \
             patch.object(torch.Tensor, 'to', fake_to), \
             patch('torch.cuda.synchronize', side_effect=RuntimeError('fake drain failed')):
            result = _run_bound_owner(FakeSession([]), object(), backend=backend,
                registry=object(), stop_event=threading.Event(), announce=lambda server: None)
            self.assertEqual(result.status, 'quarantined')
            self.assertFalse(result.teardown_verified)
            del backend
            gc.collect()
            for name in ('model', 'worker', 'input', 'logits', 'offload'):
                with self.subTest(name=name):
                    self.assertIsNotNone(refs[name]())

    def test_one_warm_worker_serves_two_authenticated_teams_and_drains(self):
        import torch
        from torch import nn
        from looking_glass.broker import Broker
        from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
        from looking_glass.ipc import IPCClient, IPCRejected, IPCServer
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker
        from looking_glass.service import make_gpt2_request_factory, serve_owner

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)
                self.calls = 0
            def forward(self, input_ids):
                self.calls += 1
                return input_ids.to(torch.float32).unsqueeze(-1) + torch.tensor([0.0, 1.0])

        class CpuBackend(FakeBackend):
            def construct(self, budget, resident_lease, profiles):
                self.events.append('load-and-warmup')
                self.budget, self.load_lease = budget, resident_lease
                self.model = TinyModel().eval()
                resident = ResidentWorker('c' * 64, self.model, device='cpu',
                    max_batch_size=1, max_sequence_length=2, max_input_bytes=64,
                    max_output_bytes=1024)
                admission = DeviceAdmission(budget, device='cpu', max_estimated_incremental_bytes=2048)
                def profile(name):
                    return WorkloadProfile(name, 'c' * 64, 'cpu', (1, 2),
                        (('input_ids', 'torch.int64'),), 1024, 2048)
                cold = BudgetedWorker(resident, admission, (profile('cold'),))
                cold.execute(ForwardRequest('warmup', {'input_ids': torch.tensor([[1, 2]])}, 17))
                worker = BudgetedWorker(resident, admission, (profile('warm'),))
                broker = Broker(worker, max_pending=4, max_pending_per_team=2)
                factory = make_gpt2_request_factory(vocab_size=10, max_positions=10,
                                                    max_batch=1, max_tokens=2)
                server = IPCServer(broker, team_ids=('alpha', 'beta'), request_factory=factory,
                                   max_connections=4, max_batch=1, max_tokens=2)
                self.worker = worker
                return worker, broker, server

            def serve(self, server, broker, worker, stop_event, announce, verify_drain):
                return serve_owner(server, broker, worker, stop_event, announce=announce,
                                   verify_drain=verify_drain, shutdown_timeout=10)

        events, results = [], []
        backend, session = CpuBackend(events), FakeSession(events)
        stop = threading.Event()

        def announce(server):
            try:
                alpha = IPCClient(server.address, server.team_tokens['alpha']).submit(
                    team_id='alpha', request_id='a', inputs={'input_ids': [[1, 2]]})
                beta = IPCClient(server.address, server.team_tokens['beta']).submit(
                    team_id='beta', request_id='b', inputs={'input_ids': [[3, 4]]})
                with self.assertRaises(IPCRejected):
                    IPCClient(server.address, server.team_tokens['alpha']).submit(
                        team_id='beta', request_id='spoof', inputs={'input_ids': [[3, 4]]})
                results.extend((alpha.wait(10), beta.wait(10)))
            finally:
                stop.set()

        result = _run_bound_owner(session, object(), backend=backend, registry=object(),
                                  stop_event=stop, announce=announce)
        self.assertEqual(result.status, 'clean')
        self.assertTrue(result.teardown_verified)
        self.assertEqual([item['status'] for item in results], ['succeeded', 'succeeded'])
        self.assertEqual([item['result']['logits'] for item in results],
                         [[1.0, 2.0, 2.0, 3.0], [3.0, 4.0, 4.0, 5.0]])
        self.assertEqual(backend.model.calls, 3)
        self.assertEqual(backend.budget.active_leases(), (backend.load_lease,))


def _fake_supervised_child(mode):
    """A direct-child fake: real supervisor and locks, no actual CUDA calls."""
    # Preserve the fake visibility string for identity validation while
    # blocking all CUDA calls in this synthetic subprocess.
    _install_cuda_guard(hide_device=False)
    from looking_glass.device_lease import DeviceLease
    from looking_glass.supervisor import OwnerSession

    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=5)
    backend = FakeBackend([], failure=(mode == 'worker_failed'))
    if mode == 'sync_failed':
        backend.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('fake drain failure'))
    try:
        result = _run_bound_owner(session, lease, backend=backend, registry=object(),
                                  stop_event=threading.Event(), announce=lambda server: None)
    except BaseException:
        os._exit(4)
    # The OS closes the retained lease only on actual process exit. Failed
    # teardown must never continue the owner interpreter or run destructors.
    os._exit(0 if result.status == 'clean' and result.teardown_verified else 3)


class SupervisedSubprocessTests(unittest.TestCase):
    def test_real_supervisor_distinguishes_work_failure_from_unsafe_drain(self):
        from looking_glass.device_identity import resolve_device_identity
        from looking_glass.supervisor import run_supervised

        repo = Path(__file__).resolve().parent.parent
        site_packages = Path(sys.prefix) / 'Lib' / 'site-packages'
        if not site_packages.is_dir():
            self.skipTest('this subprocess test needs the Windows venv site-packages')
        base_python = getattr(sys, '_base_executable', None)
        self.assertTrue(base_python and Path(base_python).is_file())
        child_env = {'CUDA_VISIBLE_DEVICES': UUID,
                     'PYTHONPATH': os.pathsep.join((str(repo), str(site_packages)))}
        with tempfile.TemporaryDirectory() as directory:
            for mode, clean, exit_code in (('clean', True, 0),
                                           ('worker_failed', True, 3),
                                           ('sync_failed', False, 3)):
                with self.subTest(mode=mode):
                    root = Path(directory) / mode
                    root.mkdir()
                    with patch.dict(os.environ, child_env):
                        identity = resolve_device_identity(root, UUID, physical_uuids=(UUID,))
                        result = run_supervised(
                            [base_python, str(Path(__file__).resolve()), '--fake-owner', mode],
                            device_lease_path=identity.lock_path,
                            marker_path=identity.marker_path, identity=identity,
                            timeout=20, env=child_env, cwd=str(repo),
                        )
                    self.assertEqual(result.exit_code, exit_code)
                    self.assertEqual(result.clean, clean)
                    self.assertEqual(identity.marker_path.exists(), not clean)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fake-owner':
        _fake_supervised_child(sys.argv[2])
    else:
        unittest.main()
