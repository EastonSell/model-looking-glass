"""Behavioral tests for the isolated, source-bound exposed-pair entry point."""

from __future__ import annotations

import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO / 'scripts/bert_gpu_exposed_pair_bootstrap.py'
MANIFEST = REPO / 'experiments/bert_agreement_public_cases_v1.json'
GIT = Path(shutil.which('git') or '')
SOURCE_PATHS = (
    'scripts/bert_gpu_exposed_pair_bootstrap.py',
    'looking_glass/__init__.py',
    'looking_glass/bert_gpu_exposed_pair.py',
    'experiments/bert_agreement_public_cases_v1.json',
)


def git(cwd, *args):
    return subprocess.run([str(GIT), *args], cwd=cwd, check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def make_checkout(tmp_path, body='', bad_manifest=False):
    # This assertion is the expected initial red until the new bootstrap exists.
    assert BOOTSTRAP.is_file(), 'the source-bound exposed-pair bootstrap is missing'
    checkout = tmp_path / 'checkout'
    for name in ('scripts', 'looking_glass', 'experiments'):
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


def invoke(tmp_path, checkout, commit, *, site=None, owner_commit=None, owner_git=None, git_env=None):
    prefix = tmp_path / 'bytecode'
    prefix.mkdir(exist_ok=True)
    site = site or tmp_path / 'site'
    site.mkdir(exist_ok=True)
    output = tmp_path / 'invoked.txt'
    env = dict(os.environ, PROBE_OUT=str(output))
    env.update(git_env or {})
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(checkout / SOURCE_PATHS[0]), '--bootstrap-site', str(site),
               '--source-commit', commit, '--git-executable', str(GIT.resolve()),
               '--owner', str(tmp_path / 'run'), str(tmp_path / 'cache'), 'synthetic-run',
               owner_commit or commit, owner_git or str(GIT.resolve())]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    return result, output


def test_clean_committed_source_runs_the_owner_after_binding(tmp_path):
    checkout, commit = make_checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text(os.environ['LOOKING_GLASS_PAIR_SOURCE_COMMIT'])\n")
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == commit


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
        "Path(os.environ['PROBE_OUT']).write_text(os.environ['LOOKING_GLASS_PAIR_SOURCE_COMMIT'] + '\\n' + "
        "os.environ['LOOKING_GLASS_PAIR_GIT_EXE'])\n")
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


@pytest.mark.skipif(not GIT.is_file(), reason='Git executable is unavailable')
def test_unlisted_project_module_cannot_shadow_a_bound_dependency(tmp_path):
    checkout, commit = make_checkout(tmp_path, 'import looking_glass.unlisted\n')
    (checkout / 'looking_glass/unlisted.py').write_text(
        "from pathlib import Path\nimport os\nPath(os.environ['PROBE_OUT']).write_text('executed')\n",
        encoding='utf8')
    result, output = invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()
