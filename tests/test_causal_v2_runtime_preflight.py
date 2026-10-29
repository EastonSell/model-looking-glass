"""CPU-only preflight checks; no v2 cases or model forwards."""
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimePreflightTests(unittest.TestCase):
    def _child(self, script):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT)}
        with tempfile.TemporaryDirectory() as prefix:
            result = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={prefix}", "-c", script],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        if result.returncode == 77:
            self.skipTest("this preflight requires the protocol's Python and PyTorch versions")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_runtime_and_model_state_without_forward_or_rng_leak(self):
        self._child(r'''
import hashlib, platform, sys
from unittest.mock import patch
import torch
from looking_glass.causal_v2_preparation import MODEL_SEEDS
from looking_glass.causal_v2_model_identity import state_sha256
from looking_glass.causal_v2_runtime_preflight import inspect_runtime_and_models
from looking_glass.model import TinyResidual

if platform.python_version() != '3.11.15' or str(torch.__version__) != '2.8.0+cu128':
    raise SystemExit(77)
before = torch.random.get_rng_state().clone()
def forbidden_forward(self, *args, **kwargs):
    raise AssertionError('forward is forbidden')
with patch.object(TinyResidual, 'forward', forbidden_forward):
    policy, runtime, states = inspect_runtime_and_models()
assert torch.equal(before, torch.random.get_rng_state())
assert [item['seed'] for item in states] == list(MODEL_SEEDS)
assert all(set(item) == {'seed', 'sha256'} and len(item['sha256']) == 64 for item in states)
assert len({item['sha256'] for item in states}) == len(states)
with torch.device('cpu'), torch.random.fork_rng(devices=[]):
    for item in states:
        torch.random.default_generator.manual_seed(item['seed'])
        assert state_sha256(TinyResidual(width=8, depth=4).eval()) == item['sha256']
assert torch.equal(before, torch.random.get_rng_state())
assert runtime == {'python_version': platform.python_version(), 'torch_version': torch.__version__,
                   'system': platform.system(), 'machine': platform.machine(), 'byteorder': sys.byteorder,
                   'intraop_threads': 1, 'denormal_preserved': True,
                   'torch_build_sha256': hashlib.sha256(torch.__config__.show().encode('utf-8')).hexdigest()}
policy.verify()
assert not torch.cuda.is_initialized()
''')

    def test_rejects_gpu_initialization_and_unmasked_or_wrong_runtime_before_models(self):
        self._child(r'''
import os
from unittest.mock import patch
import torch
from looking_glass.causal_v2_runtime_preflight import PreflightError, inspect_runtime_and_models
from looking_glass.model import TinyResidual

with patch.object(TinyResidual, '__init__', side_effect=AssertionError('must not construct model')):
    for mode in ('cuda', 'mask', 'python', 'torch'):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0' if mode == 'mask' else '-1'}), \
             patch('torch.cuda.is_initialized', return_value=(mode == 'cuda')), \
             patch('looking_glass.causal_v2_runtime_preflight.platform.python_version',
                   return_value='0' if mode == 'python' else '3.11.15'), \
             patch('torch.__version__', '0' if mode == 'torch' else '2.8.0+cu128'):
            try: inspect_runtime_and_models()
            except PreflightError: pass
            else: raise AssertionError(f'{mode} failure was accepted')
''')

    def test_rejects_changed_model_state(self):
        self._child(r'''
from unittest.mock import patch
from looking_glass.causal_v2_runtime_preflight import PreflightError, inspect_runtime_and_models
import platform, torch
if platform.python_version() != '3.11.15' or str(torch.__version__) != '2.8.0+cu128':
    raise SystemExit(77)
with patch('looking_glass.causal_v2_runtime_preflight.state_sha256', side_effect=['a'*64, 'b'*64]):
    try: inspect_runtime_and_models()
    except PreflightError: pass
    else: raise AssertionError('inconsistent state accepted')
''')

    def test_backend_settings_are_bounded_and_preflight_rejects_changes(self):
        self._child(r'''
import os, platform, torch
from unittest.mock import patch
from looking_glass.causal_v2_runtime_preflight import (PreflightError, inspect_backend_settings,
                                                       inspect_runtime_and_models)
if platform.python_version() != '3.11.15' or str(torch.__version__) != '2.8.0+cu128':
    raise SystemExit(77)
before = inspect_backend_settings()
assert set(before) == {'interop_threads', 'float32_matmul_precision', 'cpu_capability', 'mkldnn_enabled',
                       'mkldnn_deterministic', 'deterministic_algorithms',
                       'deterministic_warn_only', 'thread_environment'}
assert set(before['thread_environment']) == {'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_DYNAMIC',
                                             'MKL_DYNAMIC', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'}
assert before['cpu_capability'] == torch.backends.cpu.get_cpu_capability()
torch.set_float32_matmul_precision('medium')
changed = inspect_backend_settings()
assert changed['float32_matmul_precision'] == 'medium' and changed != before
torch.set_float32_matmul_precision(before['float32_matmul_precision'])
for bad in ('x' * 129, 'C:/private/path', 'line\nbreak'):
    with patch.dict(os.environ, {'OMP_NUM_THREADS': bad}):
        try: inspect_backend_settings()
        except PreflightError: pass
        else: raise AssertionError('unsafe thread environment accepted')
from looking_glass.model import TinyResidual
original_init = TinyResidual.__init__
def mutating_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    torch.set_float32_matmul_precision('medium' if before['float32_matmul_precision'] != 'medium' else 'highest')
try:
    with patch.object(TinyResidual, '__init__', mutating_init):
        try: inspect_runtime_and_models()
        except PreflightError: pass
        else: raise AssertionError('backend change during preflight accepted')
finally:
    torch.set_float32_matmul_precision(before['float32_matmul_precision'])
''')

    def test_ambient_meta_default_is_not_used_or_changed(self):
        self._child(r'''
import platform, torch
from looking_glass.causal_v2_runtime_preflight import inspect_runtime_and_models
if platform.python_version() != '3.11.15' or str(torch.__version__) != '2.8.0+cu128':
    raise SystemExit(77)
torch.set_default_device('meta')
before = torch.random.get_rng_state().clone()
policy, runtime, states = inspect_runtime_and_models()
assert torch.get_default_device().type == 'meta'
assert torch.equal(before, torch.random.get_rng_state())
assert len(states) == 6 and runtime['intraop_threads'] == 1
policy.verify()
assert not torch.cuda.is_initialized()
''')

    def test_isolated_source_loader_imports_and_measures_without_forwards(self):
        code = r'''
import hashlib, importlib, importlib.util, json, pathlib, platform, sys
from unittest.mock import patch
checkout, prefix, venv = map(pathlib.Path, sys.argv[1:4])
dependencies = json.loads(sys.argv[4])
spec = importlib.util.spec_from_file_location('bootstrap_probe', checkout / 'scripts' / 'v2_source_bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
bootstrap._check_startup(checkout, prefix)
sources = {path.relative_to(checkout).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in (checkout / 'looking_glass').rglob('*.py')}
finder = bootstrap._ProjectFinder(bootstrap._source_map(checkout, sources))
bootstrap._add_dependencies(checkout, venv, dependencies)
sys.meta_path.insert(0, finder)
preflight = importlib.import_module('looking_glass.causal_v2_runtime_preflight')
importlib.import_module('looking_glass.causal_v2_study_executor')
importlib.import_module('looking_glass.causal_v2_scoring')
import torch
if platform.python_version() != '3.11.15' or str(torch.__version__) != '2.8.0+cu128':
    raise SystemExit(77)
model = importlib.import_module('looking_glass.model')
def forbidden_forward(self, *args, **kwargs):
    raise AssertionError('forward is forbidden')
with patch.object(model.TinyResidual, 'forward', forbidden_forward):
    policy, runtime, states = preflight.inspect_runtime_and_models()
inventory = finder.verify_inventory()
assert {'looking_glass/causal_v2_runtime_preflight.py', 'looking_glass/causal_v2_study_executor.py',
        'looking_glass/causal_v2_custodian_run.py', 'looking_glass/causal_v2_scoring.py',
        'looking_glass/model.py', 'looking_glass/causal_v2_runtime.py'} <= set(inventory)
assert len(states) == 6 and runtime['intraop_threads'] == 1 and not torch.cuda.is_initialized()
policy.verify()
bootstrap._check_search_path(checkout)
assert not any(prefix.iterdir())
'''
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
        venv = Path(sys.prefix).resolve()
        dependencies = [str(Path(path).resolve()) for path in site.getsitepackages()
                        if Path(path).is_dir() and Path(path).resolve().is_relative_to(venv)]
        with tempfile.TemporaryDirectory() as prefix:
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", f"pycache_prefix={prefix}",
                                     "-c", code, str(ROOT), prefix, str(venv), json.dumps(dependencies)],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        if result.returncode == 77:
            self.skipTest("this preflight requires the protocol's Python and PyTorch versions")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
