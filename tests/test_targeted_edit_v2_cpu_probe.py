"""Real direct-child no-checkpoint rehearsals for the v2 CPU study."""

import json
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pytest
import torch

from looking_glass.cpu_study_supervision import run_cpu_study
from looking_glass import targeted_edit_cpu_v1 as v1, targeted_edit_v2_edits as edits


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/targeted_edit_v2_cpu_probe.py"
REFERENCE = ROOT / "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"


def _matches_recorded_host():
    if os.name != "nt" or not REFERENCE.is_file():
        return False
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["environment"]
    return (platform.python_version() == reference["python_version"]
            and platform.platform() == reference["platform"]
            and platform.machine() == reference["machine"]
            and platform.processor() == reference["processor"]
            and str(torch.__version__) == reference["torch_version"])


def _command(tmp_path, repo=ROOT):
    study = tmp_path / "study"
    prefix = tmp_path / "prefix"
    study.mkdir()
    prefix.mkdir()
    report = tmp_path / "probe.json"
    site = Path(torch.__file__).resolve().parent.parent
    command = [sys._base_executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(repo / "scripts/targeted_edit_v2_cpu_probe.py"), "--repo", str(repo), "--site", str(site),
               "--study-root", str(study), "--report", str(report)]
    env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
             "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "PROCESSOR_IDENTIFIER") if name in os.environ}
    env.update({"PATH": str(Path(env.get("SYSTEMROOT", "C:/Windows")) / "System32"),
                "TEMP": str(tmp_path), "TMP": str(tmp_path), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    return command, env, study, prefix, report


def _copy_probe_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "looking_glass").mkdir()
    reference = repo / REFERENCE.relative_to(ROOT)
    reference.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts/targeted_edit_v2_cpu_probe.py")
    shutil.copy2(REFERENCE, reference)
    modules = ("__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
               "pretrained", "pretrained_exposures", "pretrained_exposures_v2", "targeted_edit_cpu_v1",
               "targeted_edit_cpu_v2", "targeted_edit_v2_edits")
    for module in modules:
        shutil.copy2(ROOT / "looking_glass" / (module + ".py"), repo / "looking_glass" / (module + ".py"))
    return repo


def test_probe_reference_is_bounded_and_pinned(tmp_path):
    spec = importlib.util.spec_from_file_location("targeted_edit_v2_cpu_probe_test", SCRIPT)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    repo = _copy_probe_repo(tmp_path)
    assert probe._load_reference(repo, v1, edits)["environment"]
    reference = repo / REFERENCE.relative_to(ROOT)
    reference.write_bytes(reference.read_bytes() + b" ")
    with pytest.raises(probe.ProbeError, match="digest"):
        probe._load_reference(repo, v1, edits)
    reference.write_bytes(b"x" * 65537)
    with pytest.raises(probe.ProbeError, match="oversized"):
        probe._load_reference(repo, v1, edits)


@pytest.mark.skipif(os.name != "nt", reason="Windows-only process and platform rehearsal")
def test_unsupervised_probe_cannot_claim_or_write_report(tmp_path):
    command, env, _study, prefix, report = _command(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert not report.exists()
    assert not any(prefix.iterdir())


@pytest.mark.skipif(not _matches_recorded_host(), reason="requires the recorded v1 host/runtime")
def test_supervised_no_model_probe_preserves_v1_runtime_and_reaps_child(tmp_path):
    # The isolated child resolves only explicitly allowed project sources.
    # A checkout-root stdlib shadow must never execute, even if it exists.
    repo = _copy_probe_repo(tmp_path)
    shadow_marker = tmp_path / "shadow-executed"
    for name in ("subprocess", "platform"):
        (repo / (name + ".py")).write_text(
            "from pathlib import Path\nPath(" + repr(str(shadow_marker)) + ").touch()\n"
            "raise RuntimeError('shadow module was executed')\n", encoding="utf-8")
    command, env, study, prefix, report = _command(tmp_path, repo)
    result = run_cpu_study(command, study_root=study, timeout=120, env=env, cwd=tmp_path)
    assert result.exit_code == 0 and result.clean and not result.timed_out
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["status"] == "matching_v1_runtime"
    assert value["owner_pid"] == result.pid and value["supervisor_run_id"] == result.run_id
    assert value["python_process_guard_installed"] is True
    assert value["model_dependency_absent_before_claim_and_mask"] is True
    assert value["cuda_visibility_before_model_imports"] == "-1"
    assert value["cuda_visibility_after_imports"] == "-1"
    assert value["native_descendant_absence_attested"] is False
    assert value["checkpoint_loaded"] is False and value["model_forwarded"] is False
    assert not shadow_marker.exists()
    assert not any(prefix.iterdir())


@pytest.mark.skipif(not _matches_recorded_host(), reason="requires the recorded v1 host/runtime")
@pytest.mark.parametrize("point", ["after_environment", "before_attestation"])
def test_mask_drift_fails_closed_even_if_report_exists(tmp_path, point):
    repo = _copy_probe_repo(tmp_path)
    exercised = tmp_path / "injected-drift"
    module = repo / "looking_glass/targeted_edit_v2_edits.py"
    code = "\nimport gc as _gc, os as _os\nfrom pathlib import Path as _Path\n"
    code += "from . import targeted_edit_cpu_v1 as _v1\n_original_environment = _v1._environment\n"
    code += "def _drifting_environment():\n    result = _original_environment()\n"
    if point == "after_environment":
        code += "    _Path(" + repr(str(exercised)) + ").touch()\n    _os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
    else:
        code += "    def _drift_gc():\n        _Path(" + repr(str(exercised)) + ").touch()\n"
        code += "        _os.environ['CUDA_VISIBLE_DEVICES'] = ''\n        return 0\n    _gc.collect = _drift_gc\n"
    code += "    return result\n_v1._environment = _drifting_environment\n"
    with module.open("a", encoding="utf-8") as handle:
        handle.write(code)
    command, env, study, prefix, report = _command(tmp_path, repo)
    result = run_cpu_study(command, study_root=study, timeout=120, env=env, cwd=tmp_path)
    assert exercised.exists(), "the injected drift must actually execute"
    assert result.exit_code != 0 and not result.clean and not result.timed_out
    assert (study / "cpu-study.unsafe.json").is_file()
    assert report.exists() is (point == "before_attestation")
    assert not any(prefix.iterdir())
