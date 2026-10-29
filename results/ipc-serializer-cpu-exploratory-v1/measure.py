"""Exploratory CPU encode timing; run from a committed checkout, not a GPU benchmark."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import random
import statistics
import struct
import subprocess
import sys
import time

repo = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo))
import numpy
import torch
from looking_glass.ipc import _encode_tensor_result
from looking_glass.resident_worker import ForwardResult

torch.set_num_threads(1)
assert torch.set_flush_denormal(False)
observed_utc = datetime.now(timezone.utc).isoformat()
seed = 6017
rng = random.Random(seed)
words = [1, 0x80000001, 0x007fffff, 0x807fffff, 0, 0x80000000, 0x3f800000, 0xbf800000]
while len(words) < 2 * 50257:
    word = rng.getrandbits(32)
    if ((word >> 23) & 0xff) != 0xff:
        words.append(word)
oracle = [struct.pack('>d', struct.unpack('>f', struct.pack('>I', word))[0]).hex() for word in words[:8]]
signed = [word if word < 2**31 else word - 2**32 for word in words]
tensor = torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(1, 2, 50257)
result = ForwardResult('fixed', tensor)
cap = 4 * 1024 * 1024
saved_numpy = sys.modules['numpy']
rows = []
correctness = {}
for flush in (False, True):
    assert torch.set_flush_denormal(flush)
    for phase, rounds in (('warmup', 3), ('measured', 12)):
        for number in range(rounds):
            modes = ['numpy', 'stdlib', 'legacy_lossy']
            rng.shuffle(modes)
            for position, mode in enumerate(modes):
                sys.modules['numpy'] = None if mode == 'stdlib' else saved_numpy
                started = time.perf_counter_ns()
                if mode == 'legacy_lossy':
                    assert bool(torch.isfinite(tensor).all())
                    values = tensor.reshape(-1).tolist()
                else:
                    values = _encode_tensor_result(result, cap)['logits']
                elapsed = time.perf_counter_ns() - started
                first = [struct.pack('>d', value).hex() for value in values[:8]]
                key = f'{mode}:flush_{flush}'
                correctness[key] = {'first_eight_equal': first == oracle,
                                    'first_eight_mismatches': sum(a != b for a, b in zip(first, oracle))}
                if mode != 'legacy_lossy' and first != oracle:
                    raise AssertionError('exact encoder disagreed with the fixed bit oracle')
                rows.append({'flush': flush, 'phase': phase, 'round': number, 'position': position,
                             'mode': mode, 'elapsed_ns': elapsed})
                del values
sys.modules['numpy'] = saved_numpy
clock = time.get_clock_info('perf_counter')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
source_hash = hashlib.sha256((repo / 'looking_glass' / 'ipc.py').read_bytes()).hexdigest()
medians = {f'{mode}:flush_{flush}': statistics.median(row['elapsed_ns'] for row in rows
           if row['mode'] == mode and row['flush'] == flush and row['phase'] == 'measured')
           for flush in (False, True) for mode in ('numpy', 'stdlib', 'legacy_lossy')}
cold_program = '''import json, sys, time
start = time.perf_counter_ns()
import torch
after_torch = time.perf_counter_ns()
numpy_preloaded = 'numpy' in sys.modules
import numpy
after_numpy = time.perf_counter_ns()
print(json.dumps({'torch_import_ns': after_torch - start, 'numpy_import_ns': after_numpy - after_torch,
                  'numpy_preloaded_by_torch': numpy_preloaded}))'''
fresh_process_imports = []
for _ in range(3):
    started = time.perf_counter_ns()
    child = subprocess.run([sys.executable, '-c', cold_program], check=True,
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
    elapsed = time.perf_counter_ns() - started
    fresh_process_imports.append({'process_elapsed_ns': elapsed, **json.loads(child.stdout)})
record = {'protocol': 'ipc-serializer-cpu-exploratory-v1', 'commit': commit,
          'source_sha256': source_hash,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'observed_utc': observed_utc, 'seed': seed, 'shape': [1, 2, 50257],
          'dtype': 'float32', 'clock': {'name': 'perf_counter_ns', 'implementation': clock.implementation,
                                        'resolution_seconds': clock.resolution},
          'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
          'numpy': numpy.__version__, 'torch_num_threads': torch.get_num_threads(),
          'correctness': correctness, 'median_ns': medians, 'rows': rows,
          'fresh_process_imports': fresh_process_imports,
          'scope': 'Warm in-process tensor-to-Python-number conversion only; no JSON framing, GPU, isolated host, or independent inputs. Fresh-process imports do not flush OS caches. Legacy conversion is lossy under flush-on and is not an equivalent implementation.'}
with output.open('x', encoding='utf-8') as destination:
    destination.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
assert not torch.cuda.is_initialized()
print(json.dumps({'output': str(output), 'median_ns': medians, 'correctness': correctness,
                  'fresh_process_imports': fresh_process_imports}, sort_keys=True))
