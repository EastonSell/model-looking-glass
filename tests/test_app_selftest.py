"""CPU contract checks do not substitute for the explicitly gated CUDA integration."""
import json
import subprocess
import sys

import pytest
import torch

from looking_glass import app_selftest as subject


def test_cpu_prerequisite_never_reports_pass_or_constructs_fixture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(subject, "_checkpoint", lambda *args: pytest.fail("must not build before CUDA prerequisite"))
    report = subject.run_selftest()
    assert report["status"] == "skipped"
    assert report["stages"] == [{"stage": "cuda_prerequisite", "status": "skipped"}]


@pytest.mark.parametrize("device,budget", [("cpu", 1024), ("auto", 1024), ("cuda:-1", 1024), ("cuda:0", 0), ("cuda:0", True)])
def test_invalid_request_is_failed_not_skipped(device, budget):
    assert subject.run_selftest(device, budget)["status"] == "failed"


@pytest.mark.parametrize("status,code", [("passed", 0), ("failed", 1), ("skipped", 2)])
def test_cli_classification_and_stdout_is_only_json(monkeypatch, capsys, status, code):
    def synthetic(*args):
        print("synthetic dependency diagnostic")
        return {"status": status, "fixture": "mocked CLI classification; not CUDA execution"}
    monkeypatch.setattr(subject, "run_selftest", synthetic)
    assert subject.main([]) == code
    streams = capsys.readouterr()
    assert json.loads(streams.out)["status"] == status
    assert "diagnostic" in streams.err


def _synthetic_prerequisite(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * 1024**3, 8 * 1024**3))


def test_stage_failure_stops_remaining_family_and_removes_temporary_directory(monkeypatch):
    _synthetic_prerequisite(monkeypatch)
    roots, called = [], []
    def failure(root, family, device, budget, stages):
        roots.append(root); called.append(family)
        (root / "synthetic-marker").write_text("test only")
        raise RuntimeError("/private/path should not be reported")
    monkeypatch.setattr(subject, "_family", failure)
    report = subject.run_selftest()
    assert report["status"] == "failed" and called == ["gpt2"]
    assert not roots[0].exists()
    assert "/private/path" not in json.dumps(report)


def test_synthetic_success_requires_all_three_families_and_selected_budget(monkeypatch):
    _synthetic_prerequisite(monkeypatch)
    calls = []
    def synthetic(root, family, device, budget, stages):
        calls.append((family, str(device), budget))
        stages.extend({"family": family, "stage": stage, "status": "passed",
                       "test_only": "synthetic ledger; not GPU execution"} for stage in subject.EXPECTED_STAGES)
    monkeypatch.setattr(subject, "_family", synthetic)
    assert subject.run_selftest(max_vram_mib=512)["status"] == "passed"
    assert calls == [(family, "cuda:0", 512 * 1024**2) for family in ("gpt2", "t5", "distilbert")]


@pytest.mark.parametrize("family", ["gpt2", "t5", "distilbert"])
def test_synthetic_local_assets_are_safe_and_deterministic(tmp_path, family):
    from looking_glass.app_runtime import LocalModelRuntime
    one, two = tmp_path / "one", tmp_path / "two"
    before = torch.random.get_rng_state().clone()
    subject._checkpoint(one, family); subject._checkpoint(two, family)
    assert torch.equal(before, torch.random.get_rng_state())
    assert (one / "model.safetensors").read_bytes() == (two / "model.safetensors").read_bytes()
    assert not list(one.glob("*.bin"))
    runtime = LocalModelRuntime()
    runtime.load(str(one), device="cpu")
    prompt = "hello [MASK]" if family == "distilbert" else "hello world"
    result = runtime.run(prompt, {"max_new_tokens": 2})
    assert result["model"]["family"] == family
    assert result["coverage"]["forward_count"] == (1 if family == "distilbert" else 2)
    runtime.unload()


def test_cpu_cli_subprocess_is_machine_readable():
    if torch.cuda.is_available(): pytest.skip("CPU prerequisite behavior only")
    result = subprocess.run([sys.executable, "-B", "-m", "looking_glass.app_selftest"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "skipped"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device/build required; CPU policy tests are not GPU validation")
def test_real_cuda_end_to_end_selftest():
    report = subject.run_selftest()
    if report["status"] == "skipped": pytest.skip(report["reason"])
    assert report["status"] == "passed", report
    assert len(report["stages"]) == 15
    assert all(stage["status"] == "passed" for stage in report["stages"])


@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "failed", "skipped", "reversed"])
def test_incomplete_or_unsuccessful_ledger_never_passes(monkeypatch, mutation):
    _synthetic_prerequisite(monkeypatch)
    def synthetic(root, family, device, budget, stages):
        rows = [{"family": family, "stage": stage, "status": "passed"} for stage in subject.EXPECTED_STAGES]
        if mutation == "empty": rows = []
        elif mutation == "missing": rows.pop()
        elif mutation == "duplicate": rows.append(dict(rows[-1]))
        elif mutation in ("failed", "skipped"): rows[-1]["status"] = mutation
        elif mutation == "reversed": rows.reverse()
        stages.extend(rows)
    monkeypatch.setattr(subject, "_family", synthetic)
    report = subject.run_selftest()
    assert report["status"] == "failed"
    assert report["stages"][-1]["status"] == "failed"
