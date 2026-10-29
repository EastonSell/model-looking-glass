"""Guarded Windows owner rehearsal; no v2 cases or model forwards."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("run_synthetic_v2.py", "v2_source_launcher.py", "v2_source_bootstrap.py", "v2_source_guards.py")
EXTRA = (
    "docs/CAUSAL_CONFIRMATION_V2.md",
    "docs/CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md",
    "docs/CAUSAL_CONFIRMATION_V2_SOURCE_CLARIFICATION.md",
    "docs/CAUSAL_CONFIRMATION_V2_SELF_PATCH_CLARIFICATION.md",
    "docs/CAUSAL_CONFIRMATION_V2_RESOURCE_CLARIFICATION.md",
    "results/synthetic-v2-exclusions-20261028/inputs.json",
    "results/synthetic-v2-resource-pilot-20261028/artifacts/pilot_inputs.json",
)


@pytest.fixture
def checkout(tmp_path):
    if os.name != "nt" or not shutil.which("git"):
        pytest.skip("requires Windows and git")
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("requires a virtualenv with torch")
    site = Path(next(iter(spec.submodule_search_locations))).parent.resolve()
    venv = Path(sys.prefix).resolve()
    if not site.is_relative_to(venv):
        pytest.skip("torch is not installed within the test virtualenv")
    repo = tmp_path / "checkout"
    (repo / "scripts").mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copyfile(ROOT / "scripts" / name, repo / "scripts" / name)
    shutil.copytree(ROOT / "looking_glass", repo / "looking_glass",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The fixture's root forward is deliberately fatal: a successful real
    # rehearsal proves no TinyResidual forward was invoked, independent of
    # the launcher's reported counter.
    model = repo / "looking_glass" / "model.py"
    source = model.read_bytes()
    header = b"    def forward(self, x, route=0, perturb=None):\n"
    assert source.count(header) == 1
    model.write_bytes(source.replace(header, header + b"        raise AssertionError('unexpected forward')\n"))
    for relative in EXTRA:
        dest = repo / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    for key, value in (("user.name", "Development Test"), ("user.email", "dev@example.invalid")):
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "--", "scripts", "looking_glass", "docs", "results"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fabricated owner source"],
                   check=True, capture_output=True)
    work = tmp_path / "work"
    work.mkdir()
    return repo, venv, site, work, tmp_path / "new-evidence"


def _run(fixture, mode, *, mask=True):
    repo, venv, site, work, output = fixture
    base_python = Path(sys.base_prefix) / "python.exe"
    args = [str(base_python), "-I", "-S", "-B", str(repo / "scripts" / "run_synthetic_v2.py"), mode,
            "--checkout", str(repo), "--venv-root", str(venv), "--dependency", str(site),
            "--work-dir", str(work), "--output", str(output), "--timeout", "60"]
    env = dict(os.environ)
    if mask:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return subprocess.run(args, capture_output=True, text=True, cwd=work, env=env, timeout=90)


def test_guarded_real_cpu_rehearsal_does_not_generate_cases_or_forwards(checkout):
    result = _run(checkout, "--rehearsal")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "rehearsal"
    assert report["root_forwards"] == 0
    assert report["child_reaped"] is True
    assert report["study_ready"] is False
    assert not checkout[4].exists()
    assert list(checkout[3].iterdir()) == []


def test_study_mode_requires_cpu_mask_before_source_or_case_work(checkout):
    result = _run(checkout, "--study", mask=False)
    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "failed"
    assert not checkout[4].exists()
    assert not checkout[4].with_name(checkout[4].name + ".attempt.json").exists()
    assert list(checkout[3].iterdir()) == []


def test_study_bootstrap_rejects_missing_guarded_start_before_project_import(checkout, tmp_path):
    repo, venv, site, work, output = checkout
    prefix = tmp_path / "empty-prefix"
    prefix.mkdir()
    base_python = Path(sys.base_prefix) / "python.exe"
    args = [str(base_python), "-I", "-S", "-B", "-X", f"pycache_prefix={prefix}",
            str(repo / "scripts" / "v2_source_bootstrap.py"), "--study-owner", "--checkout", str(repo),
            "--sources", str(tmp_path / "unused.json"), "--venv-root", str(venv),
            "--dependency", str(site), "--prefix", str(prefix), "--source-revision", "1" * 40,
            "--output", str(output)]
    result = subprocess.run(args, capture_output=True, text=True, cwd=work,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=15)
    assert result.returncode != 0
    assert json.loads(result.stderr)["error_type"] == "BootstrapError"
    assert not output.exists()
    assert not output.with_name(output.name + ".attempt.json").exists()


@pytest.mark.parametrize("script", ["run_synthetic_v2.py", "v2_source_launcher.py"])
def test_help_preserves_successful_exit(checkout, script):
    repo, _, _, work, _ = checkout
    base_python = Path(sys.base_prefix) / "python.exe"
    result = subprocess.run([str(base_python), "-I", "-S", "-B", str(repo / "scripts" / script), "--help"],
                            capture_output=True, text=True, cwd=work, timeout=15)
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert result.stderr == ""


def test_uncommitted_source_blocks_rehearsal_before_output(checkout):
    (checkout[0] / "looking_glass" / "causal_v2_study_entry.py").write_text("raise AssertionError()\n")
    result = _run(checkout, "--rehearsal")
    assert result.returncode != 0
    assert not checkout[4].exists()
    assert list(checkout[3].iterdir()) == []
