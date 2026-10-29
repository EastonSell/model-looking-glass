"""CPU-only startup checks; they do not load a checkpoint."""
import json
from pathlib import Path
import subprocess
import sys

import torch


def test_isolated_bootstrap_finds_site_without_loading_a_model(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    site = Path(torch.__file__).resolve().parent.parent
    prefix = tmp_path / 'empty-prefix'
    prefix.mkdir()
    command = [getattr(sys, '_base_executable', None) or sys.executable, '-I', '-S', '-B',
               '-X', 'pycache_prefix=' + str(prefix), str(repo / 'scripts/short_service_cpu_reference_bootstrap.py'),
               '--site-packages', str(site), '--check-only']
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'status': 'ok', 'torch_available': True}
    assert not list(prefix.iterdir())


def test_nonempty_bytecode_prefix_is_rejected(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    site = Path(torch.__file__).resolve().parent.parent
    prefix = tmp_path / 'used-prefix'
    prefix.mkdir()
    (prefix / 'stale').write_text('x')
    command = [getattr(sys, '_base_executable', None) or sys.executable, '-I', '-S', '-B',
               '-X', 'pycache_prefix=' + str(prefix), str(repo / 'scripts/short_service_cpu_reference_bootstrap.py'),
               '--site-packages', str(site), '--check-only']
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'empty directory' in result.stderr
