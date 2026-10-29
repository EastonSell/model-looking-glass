"""No-model tests for the ViT observer's isolated source bootstrap."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
GIT = Path(shutil.which("git") or "")
SOURCES = ("scripts/vision_vit_cuda_bootstrap.py", "looking_glass/__init__.py",
           "looking_glass/vision_vit_cuda_observer.py", "experiments/vision_coco_spatial_v1.json")


def _git(directory, *args):
    return subprocess.run([str(GIT), *args], cwd=directory, capture_output=True, text=True,
                          timeout=30, check=True).stdout.strip()


def _checkout(tmp_path, body, *, wrong_manifest=False):
    checkout = tmp_path / "checkout"
    for directory in ("scripts", "looking_glass", "experiments"):
        (checkout / directory).mkdir(parents=True)
    shutil.copyfile(REPO / SOURCES[0], checkout / SOURCES[0])
    (checkout / SOURCES[1]).write_text("", encoding="utf-8")
    (checkout / SOURCES[2]).write_text("_SOURCE_PATHS = " + repr(SOURCES) + "\n" + body, encoding="utf-8")
    raw = (REPO / SOURCES[3]).read_bytes() + (b" " if wrong_manifest else b"")
    (checkout / SOURCES[3]).write_bytes(raw)
    _git(checkout, "init", "-q")
    _git(checkout, "config", "core.autocrlf", "false")
    _git(checkout, "add", "--", ".")
    _git(checkout, "-c", "user.name=Probe", "-c", "user.email=probe@example.invalid", "commit", "-qm", "fake")
    return checkout, _git(checkout, "rev-parse", "HEAD")


def _invoke(tmp_path, checkout, commit):
    prefix, site = tmp_path / "bytecode", tmp_path / "site"
    prefix.mkdir()
    site.mkdir()
    output = tmp_path / "invoked.txt"
    command = [sys._base_executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(checkout / SOURCES[0]), "--bootstrap-site", str(site), "--source-commit", commit,
               "--git-executable", str(GIT.resolve()), "--owner", str(tmp_path / "run"),
               str(tmp_path / "assets"), "synthetic", commit, str(GIT.resolve())]
    result = subprocess.run(command, cwd=tmp_path, env=dict(os.environ, PROBE_OUT=str(output)),
                            capture_output=True, text=True, timeout=60)
    return result, output


@pytest.mark.skipif(not GIT.is_file(), reason="Git executable unavailable")
def test_exact_committed_source_and_identity_reach_owner(tmp_path):
    checkout, commit = _checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text(os.environ['LOOKING_GLASS_VISION_SOURCE_COMMIT'])\n")
    result, output = _invoke(tmp_path, checkout, commit)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == commit


@pytest.mark.skipif(not GIT.is_file(), reason="Git executable unavailable")
def test_source_drift_is_rejected_even_when_git_hides_it(tmp_path):
    checkout, commit = _checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n")
    _git(checkout, "update-index", "--assume-unchanged", SOURCES[2])
    with (checkout / SOURCES[2]).open("a", encoding="utf-8") as stream:
        stream.write("\n# drift\n")
    assert not _git(checkout, "status", "--porcelain", "--untracked-files=no")
    result, output = _invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.skipif(not GIT.is_file(), reason="Git executable unavailable")
def test_wrong_manifest_is_rejected_before_owner(tmp_path):
    checkout, commit = _checkout(tmp_path, "from pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('executed')\n", wrong_manifest=True)
    result, output = _invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.skipif(not GIT.is_file(), reason="Git executable unavailable")
def test_unlisted_project_module_cannot_execute(tmp_path):
    checkout, commit = _checkout(tmp_path, "import looking_glass.unlisted\n")
    (checkout / "looking_glass/unlisted.py").write_text(
        "from pathlib import Path\nimport os\nPath(os.environ['PROBE_OUT']).write_text('executed')\n",
        encoding="utf-8")
    result, output = _invoke(tmp_path, checkout, commit)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.skipif(not GIT.is_file(), reason="Git executable unavailable")
def test_checkout_root_cannot_shadow_stdlib(tmp_path):
    checkout, commit = _checkout(tmp_path, "import fractions\nfrom pathlib import Path\nimport os\n"
        "Path(os.environ['PROBE_OUT']).write_text('safe')\n")
    (checkout / "numbers.py").write_text("raise RuntimeError('checkout shadow executed')\n", encoding="utf-8")
    result, output = _invoke(tmp_path, checkout, commit)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "safe"
