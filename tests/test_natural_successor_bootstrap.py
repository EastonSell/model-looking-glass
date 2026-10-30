"""No-model startup tests with fabricated source; no protected data."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest

try:
    from looking_glass import natural_successor_bootstrap as launch
except ImportError:
    launch = None


def run(tmp_path, source, digest, *, site=None, mode='-I', extra=(), overrides=None):
    # Fabricated owners use stdlib only; exercise the pinned Windows layout
    # without depending on this host's installed dependency layout.
    actual_site = tmp_path / 'Lib' / 'site-packages'
    actual_site.mkdir(parents=True, exist_ok=True)
    chosen_site = str(actual_site.resolve()) if site is None else site
    env = os.environ.copy()
    env['MLG_NATURAL_SITE'] = str(actual_site.resolve())
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['OMP_NUM_THREADS'] = '1'
    if overrides:
        env.update(overrides)
    prefix = str(tmp_path / 'nonexistent-pycache')
    return subprocess.run([sys._base_executable, mode, *extra, '-S', '-B', '-X', 'pycache_prefix=' + prefix,
                           '-c', launch.BOOTSTRAP, str(source), digest, chosen_site],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)


def test_isolated_source_first_bootstrap_adds_one_approved_site(tmp_path):
    assert launch is not None
    source = tmp_path / 'fake_owner.py'
    source.write_text("import os,sys\nprint(sys.path.count(os.path.realpath(os.environ['MLG_NATURAL_SITE'])))\n",
                      encoding='utf-8')
    result = run(tmp_path, source, hashlib.sha256(source.read_bytes()).hexdigest())
    assert result.returncode == 0 and result.stdout.strip() == b'1'


def test_wrong_hash_or_site_never_executes_owner(tmp_path):
    assert launch is not None
    source = tmp_path / 'fake_owner.py'
    marker = tmp_path / 'would-execute'
    source.write_text(f"open({str(marker)!r}, 'w').write('unsafe')\n", encoding='utf-8')
    result = run(tmp_path, source, '0' * 64)
    assert result.returncode != 0 and not marker.exists()
    result = run(tmp_path, source, hashlib.sha256(source.read_bytes()).hexdigest(), site=str(tmp_path))
    assert result.returncode != 0 and not marker.exists()
    result = run(tmp_path, source, hashlib.sha256(source.read_bytes()).hexdigest(),
                 overrides={'MKL_CBWR': 'COMPATIBLE'})
    assert result.returncode != 0 and not marker.exists()


@pytest.mark.parametrize('flag', ['-O', '-OO'])
def test_optimized_bootstrap_rejects_before_source(tmp_path, flag):
    assert launch is not None
    source = tmp_path / 'fake_owner.py'
    marker = tmp_path / 'would-execute'
    source.write_text(f"open({str(marker)!r}, 'w').write('unsafe')\n", encoding='utf-8')
    result = run(tmp_path, source, hashlib.sha256(source.read_bytes()).hexdigest(), extra=(flag,))
    assert result.returncode != 0 and not marker.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="frozen Windows startup integration")
def test_actual_frozen_startup_passes_only_in_no_model_validation(tmp_path):
    assert launch is not None
    frozen = os.environ.get('MLG_FROZEN_STARTUP_SOURCE')
    if not frozen:
        pytest.skip('set MLG_FROZEN_STARTUP_SOURCE for the pinned offline check')
    frozen_path = Path(frozen).resolve(strict=True)
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == (
        '1325af599ffce4f44963a23d14623938e1e038fb66fd65fc8003e7f50a478b5c')
    owner = Path(__file__).resolve().parents[1] / 'tools/natural_successor_startup_probe.py'
    site = str((Path(sysconfig.get_path('purelib'))).resolve())
    env = os.environ.copy()
    env.update({'MLG_NATURAL_SITE': site, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'})
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(tmp_path / 'unused'),
               '-c', launch.BOOTSTRAP, str(owner), hashlib.sha256(owner.read_bytes()).hexdigest(), site,
               '--validate-only', '--frozen-launcher', str(frozen_path), '--frozen-launcher-sha',
               hashlib.sha256(frozen_path.read_bytes()).hexdigest()]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0 and result.stdout.strip() == b'startup-passed; no-model'
