"""Behavioral tests for the isolated, source-bound repeated tap entry point."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO / 'scripts/tap_repeated_gpu_bootstrap.py'
MANIFEST = REPO / 'evidence/gpu-tap-latency-repeated-v1/schedule.json'
GIT = Path(shutil.which('git') or '')
SOURCE_PATHS = (
    'scripts/tap_repeated_gpu_bootstrap.py',
    'looking_glass/__init__.py',
    'looking_glass/tap_repeated_gpu.py',
    'evidence/gpu-tap-latency-repeated-v1/schedule.json',
)
PLATFORM_UNAME = {'system': 'Windows', 'node': 'fixture-node', 'release': '11',
                  'version': '10.0.26200', 'machine': 'AMD64', 'processor': 'fixture-processor'}


def git(cwd, *args):
    return subprocess.run([str(GIT), *args], cwd=cwd, check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def make_checkout(tmp_path, body='', bad_manifest=False):
    # This assertion is the expected initial red until the new bootstrap exists.
    assert BOOTSTRAP.is_file(), 'the source-bound repeated tap bootstrap is missing'
    checkout = tmp_path / 'checkout'
    for name in ('scripts', 'looking_glass', 'evidence/gpu-tap-latency-repeated-v1'):
        (checkout / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BOOTSTRAP, checkout / SOURCE_PATHS[0])
    (checkout / SOURCE_PATHS[1]).write_text('', encoding='utf8')
    owner = checkout / SOURCE_PATHS[2]
    owner.write_text('_SOURCE_PATHS = ' + repr(SOURCE_PATHS) + '\n' + body, encoding='utf8')
    manifest = MANIFEST.read_bytes() + (b' ' if bad_manifest else b'')
    (checkout / SOURCE_PATHS[3]).write_bytes(manifest)
    git(checkout, 'init', '-q')
    git(checkout, 'config', 'core.autocrlf', 'false')
    git(checkout, 'add', '--', '.')
    git(checkout, '-c', 'user.name=Probe', '-c', 'user.email=probe@example.invalid',
        'commit', '-qm', 'synthetic source')
    return checkout, git(checkout, 'rev-parse', 'HEAD')


def write_source_manifest(tmp_path, checkout, commit):
    files = {}
    for path in SOURCE_PATHS:
        raw = subprocess.run([str(GIT), '--no-replace-objects', 'show', commit + ':' + path],
                             cwd=checkout, check=True, capture_output=True).stdout
        tree = subprocess.run([str(GIT), '--no-replace-objects', 'ls-tree', commit, '--', path],
                              cwd=checkout, check=True, capture_output=True, text=True).stdout
        files[path] = {'mode': tree.split(' ', 1)[0], 'sha256': hashlib.sha256(raw).hexdigest()}
    manifest = {'schema': 'tap-repeated-source-manifest-v1', 'run_id': 'synthetic-run',
                'source_commit': commit, 'source_files': files}
    data = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    directory = tmp_path / 'run' / 'protected-manifest'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'source-manifest.json').write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def invoke(tmp_path, checkout, commit, *, site=None, owner_commit=None, owner_git=None, git_env=None,
           bootstrap_git=None, manifest_digest=None, uname=None):
    prefix = tmp_path / 'bytecode'
    prefix.mkdir(exist_ok=True)
    site = site or tmp_path / 'site'
    site.mkdir(exist_ok=True)
    output = tmp_path / 'invoked.txt'
    env = dict(os.environ, PROBE_OUT=str(output))
    env.update(git_env or {})
    digest = write_source_manifest(tmp_path, checkout, commit)
    env['LOOKING_GLASS_TAP_REPEATED_MANIFEST_SHA256'] = manifest_digest or digest
    env['LOOKING_GLASS_TAP_REPEATED_PLATFORM_UNAME'] = json.dumps(
        PLATFORM_UNAME if uname is None else uname, sort_keys=True, separators=(',', ':'))
    selected_git = str(bootstrap_git or GIT.resolve())
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(checkout / SOURCE_PATHS[0]), '--bootstrap-site', str(site),
               '--source-commit', commit, '--git-executable', selected_git,
               '--owner', str(tmp_path / 'run'), str(tmp_path / 'cache'), 'synthetic-run',
               owner_commit or commit, owner_git or selected_git]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    return result, output


def test_child_uses_manifest_without_invoking_git(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    # This plain file is not executable Git. Parent verification is separate
    # and uses the real Git binary; the child must not invoke this path.
    fake_git = tmp_path / 'not-git.exe'
    fake_git.write_bytes(b'not executable')
    result, output = invoke(tmp_path, checkout, commit, bootstrap_git=str(fake_git))
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'executed'


def test_child_rejects_wrong_source_manifest_digest(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    result, output = invoke(tmp_path, checkout, commit, manifest_digest='0' * 64)
    assert result.returncode != 0
    assert not output.exists()


def test_child_uses_parent_observed_windows_platform_without_a_shell(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os,platform,subprocess\n"
        "subprocess.Popen=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('spawn forbidden'))\n"
        "p=platform.uname()\nPath(os.environ['PROBE_OUT']).write_text(p.machine + '/' + p.version + '/' + platform.processor())\n")
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'AMD64/10.0.26200/fixture-processor'


def test_child_rejects_an_unbound_or_unsupported_platform(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    result, output = invoke(tmp_path, checkout, commit, uname=dict(PLATFORM_UNAME, machine='arm64'))
    assert result.returncode != 0
    assert not output.exists()


def test_clean_committed_source_runs_the_owner_after_binding(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text(os.environ['LOOKING_GLASS_TAP_REPEATED_SOURCE_COMMIT'])\n")
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == commit


@pytest.mark.parametrize(('ending', 'exit_code'), [
    ('', 0), ("raise RuntimeError('synthetic owner failure')\n", 1),
    ('raise SystemExit(0)\n', 0), ('raise SystemExit(7)\n', 1),
    ('raise KeyboardInterrupt()\n', 1),
])
def test_owner_entry_exits_without_python_shutdown_hooks(tmp_path, ending, exit_code):
    body = ("from pathlib import Path\nimport atexit, os\n"
            "path = Path(os.environ['PROBE_OUT'])\n"
            "atexit.register(lambda: Path(str(path) + '.exit').write_text('ran'))\n"
            "path.write_text('entered')\n")
    body += ending
    checkout, commit = make_checkout(tmp_path, body)
    result, output = invoke(tmp_path, checkout, commit)
    assert output.read_text() == 'entered'
    assert result.returncode == exit_code
    assert not Path(str(output) + '.exit').exists()


def test_changed_tracked_source_is_rejected_even_if_git_hides_the_dirty_flag(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    git(checkout, 'update-index', '--assume-unchanged', SOURCE_PATHS[2])
    with (checkout / SOURCE_PATHS[2]).open('a', encoding='utf8') as stream:
        stream.write('\n# changed after the commit\n')
    assert not git(checkout, 'status', '--porcelain', '--untracked-files=no')
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows worktree symlink emulation')
def test_committed_symlink_mode_is_rejected_even_for_regular_checkout_bytes(tmp_path):
    checkout, _ = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    git(checkout, 'config', 'core.symlinks', 'false')
    owner = SOURCE_PATHS[2]
    blob = git(checkout, 'hash-object', '-w', owner)
    git(checkout, 'update-index', '--cacheinfo', f'120000,{blob},{owner}')
    git(checkout, '-c', 'user.name=Probe', '-c', 'user.email=probe@example.invalid',
        'commit', '-qm', 'synthetic symlink mode')
    commit = git(checkout, 'rev-parse', 'HEAD')
    assert (checkout / owner).is_file() and not (checkout / owner).is_symlink()
    assert not git(checkout, 'status', '--porcelain', '--untracked-files=no')
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()


def test_git_replacement_ref_cannot_substitute_the_committed_owner(tmp_path):
    checkout, original = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('original')\n")
    owner = checkout / SOURCE_PATHS[2]
    owner.write_text('_SOURCE_PATHS = ' + repr(SOURCE_PATHS) + '\n'
        "from pathlib import Path\nimport os\nPath(os.environ['PROBE_OUT']).write_text('replacement')\n",
        encoding='utf8')
    git(checkout, 'add', '--', SOURCE_PATHS[2])
    git(checkout, '-c', 'user.name=Probe', '-c', 'user.email=probe@example.invalid',
        'commit', '-qm', 'replacement source')
    replacement = git(checkout, 'rev-parse', 'HEAD')
    git(checkout, 'replace', original, replacement)
    git(checkout, 'reset', '--hard', original)
    assert git(checkout, 'rev-parse', 'HEAD') == original
    assert not git(checkout, 'status', '--porcelain', '--untracked-files=no')
    result, output = invoke(tmp_path, checkout, original)
    assert result.returncode != 0
    assert not output.exists()


def test_inherited_git_directory_and_config_overrides_are_ignored(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    overrides = {'GIT_DIR': str(tmp_path / 'missing-git-dir'),
                 'GIT_WORK_TREE': str(tmp_path / 'wrong-work-tree'),
                 'GIT_INDEX_FILE': str(tmp_path / 'wrong-index'),
                 'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.bare', 'GIT_CONFIG_VALUE_0': 'true'}
    result, output = invoke(tmp_path, checkout, commit, git_env=overrides)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'executed'


def test_checkout_cannot_be_selected_as_dependency_directory(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    result, output = invoke(tmp_path, checkout, commit, site=checkout)
    assert result.returncode != 0
    assert not output.exists()


def test_owner_commit_must_match_preflight_commit(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    result, output = invoke(tmp_path, checkout, commit, owner_commit='0' * 40)
    assert result.returncode != 0
    assert not output.exists()


def test_owner_git_executable_must_match_preflight_executable(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    result, output = invoke(tmp_path, checkout, commit, owner_git=sys._base_executable)
    assert result.returncode != 0
    assert not output.exists()


def test_parent_receives_verified_commit_and_git_identity(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text(os.environ['LOOKING_GLASS_TAP_REPEATED_SOURCE_COMMIT'] + '\\n' + "
        "os.environ['LOOKING_GLASS_TAP_REPEATED_GIT_EXE'])\n")
    prefix, site, output = tmp_path / 'bytecode', tmp_path / 'site', tmp_path / 'parent.txt'
    prefix.mkdir()
    site.mkdir()
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
        str(checkout / SOURCE_PATHS[0]), '--bootstrap-site', str(site), '--source-commit', commit,
        '--git-executable', str(GIT.resolve()), '--workspace-root', str(tmp_path)], cwd=tmp_path,
        env=dict(os.environ, PROBE_OUT=str(output)), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines() == [commit, str(GIT.resolve())]


def test_later_annotated_source_tuple_rebinding_is_rejected(tmp_path):
    checkout, commit = make_checkout(tmp_path, "_SOURCE_PATHS: tuple = ('looking_glass/__init__.py',)\n"
        "from pathlib import Path\nimport os\nPath(os.environ['PROBE_OUT']).write_text('executed')\n")
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()


def test_committed_but_wrong_manifest_is_rejected_before_owner(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n", bad_manifest=True)
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()


def test_sourceless_dependency_is_rejected(tmp_path):
    checkout, commit = make_checkout(tmp_path, 'import torch\n')
    site = tmp_path / 'site'
    site.mkdir()
    source = site / 'torch.py'
    source.write_text("from pathlib import Path\nimport os\n"
                      "Path(os.environ['PROBE_OUT']).write_text('executed')\n", encoding='utf8')
    py_compile.compile(str(source), cfile=str(site / 'torch.pyc'), doraise=True)
    source.unlink()
    result, output = invoke(tmp_path, checkout, commit, site=site)
    assert result.returncode != 0
    assert not output.exists()


def test_numpy_import_before_source_preflight_is_rejected(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    prefix, site, output = tmp_path / 'bytecode', tmp_path / 'site', tmp_path / 'invoked.txt'
    prefix.mkdir()
    site.mkdir()
    digest = write_source_manifest(tmp_path, checkout, commit)
    runner = ("import runpy,sys,types; sys.modules['numpy']=types.ModuleType('numpy'); "
              "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
        '-c', runner, str(checkout / SOURCE_PATHS[0]), '--bootstrap-site', str(site),
        '--source-commit', commit, '--git-executable', str(GIT.resolve()), '--owner',
        str(tmp_path / 'run'), str(tmp_path / 'cache'), 'synthetic-run', commit, str(GIT.resolve())],
        cwd=tmp_path, env=dict(os.environ, PROBE_OUT=str(output),
            LOOKING_GLASS_TAP_REPEATED_MANIFEST_SHA256=digest,
            LOOKING_GLASS_TAP_REPEATED_PLATFORM_UNAME=json.dumps(PLATFORM_UNAME, sort_keys=True,
                separators=(',', ':'))), capture_output=True, timeout=60)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.skipif(not GIT.is_file(), reason='Git executable is unavailable')
def test_unlisted_project_module_cannot_shadow_a_bound_dependency(tmp_path):
    checkout, commit = make_checkout(tmp_path, 'import looking_glass.unlisted\n')
    (checkout / 'looking_glass/unlisted.py').write_text(
        "from pathlib import Path\nimport os\nPath(os.environ['PROBE_OUT']).write_text('executed')\n",
        encoding='utf8')
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()
