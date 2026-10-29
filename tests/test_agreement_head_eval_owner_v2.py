"""Small source, asset and launcher custody checks; no checkpoint or GPU."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from looking_glass import agreement_head_eval_launcher_v2 as launcher
from looking_glass import agreement_head_eval_owner_v2 as owner


def test_asset_hashes_and_exact_file_set(tmp_path):
    asset = tmp_path / "weights"
    asset.write_bytes(b"abc")
    pins = {"weights": (3, hashlib.sha256(b"abc").hexdigest())}
    assert owner._verify_assets(tmp_path, pins)["weights"]["bytes"] == 3
    asset.write_bytes(b"abd")
    with pytest.raises(ValueError, match="differs"):
        owner._verify_assets(tmp_path, pins)
    (tmp_path / "extra").write_text("x")
    with pytest.raises(ValueError, match="exact pinned"):
        owner._verify_assets(tmp_path, pins)


def test_committed_source_bytes_are_bound_and_dirty_files_reject(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "source.py").write_bytes(b"x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "source"], check=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"]).decode().strip()
    monkeypatch.setattr(owner, "SOURCE_FILES", ("source.py",))
    assert owner._source_binding(tmp_path, commit)["source.py"] == hashlib.sha256(b"x = 1\n").hexdigest()
    (tmp_path / "source.py").write_bytes(b"x = 2\n")
    with pytest.raises(ValueError, match="clean"):
        owner._source_binding(tmp_path, commit)


def test_interrupted_launcher_wait_attempts_stop_and_retains_one_shot_claim(tmp_path, monkeypatch):
    work = tmp_path / "work"
    root = work / "teams" / "checkout"
    root.mkdir(parents=True)
    cache = work / "cache"
    cache.mkdir()
    inventory = work / "inventory.json"
    inventory.write_bytes(b"inventory")
    output = tmp_path / "separate-volume-work"
    output.mkdir()
    run = output / "new-run"
    monkeypatch.setattr(launcher, "_source_binding", lambda *_: {"source": "bound"})
    monkeypatch.setattr(launcher, "_committed_bytes", lambda *_: b"fixed")
    monkeypatch.setattr(launcher, "_fixed_inputs", lambda *_: [])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(work / "unused-prefix"))
    class Disk:
        free = 4 * 1024 ** 3
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda *_: Disk())
    calls = []
    class Child:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt("caller interrupted")
            return 1
        def terminate(self):
            calls.append(("terminate", None))
            raise OSError("synthetic termination failure")
        def kill(self):
            calls.append(("kill", None))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_a, **_k: Child())
    with pytest.raises(KeyboardInterrupt, match="caller interrupted"):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=run,
                        inventory_snapshot=inventory)
    receipt = json.loads((run / "supervisor-exit.json").read_bytes())
    assert receipt["child_start"] == "observed" and receipt["exit_code"] == 1
    assert receipt["launcher_error_type"] == "KeyboardInterrupt"
    assert ("terminate", None) in calls and ("wait", 5) in calls
    assert (output / "agreement-head-v2-evaluation.claim.json").is_file()
    with pytest.raises(FileExistsError):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=output / "second-run",
                        inventory_snapshot=inventory)


def test_revised_launcher_checks_source_and_output_volumes_before_claim(tmp_path, monkeypatch):
    work = tmp_path / "work"
    root = work / "teams" / "checkout"
    root.mkdir(parents=True)
    cache = work / "cache"
    cache.mkdir()
    inventory = work / "inventory.json"
    inventory.write_bytes(b"inventory")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(output / "unused-prefix"))
    class Disk:
        def __init__(self, free):
            self.free = free
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: Disk(500_000_000 if Path(path) == root else 9 * 1024 ** 3))
    with pytest.raises(ValueError, match="source volume"):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=output / "run",
                        inventory_snapshot=inventory)
    assert not (output / "agreement-head-v2-evaluation.claim.json").exists()
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: Disk(9 * 1024 ** 3 if Path(path) == root else 500_000_000))
    with pytest.raises(ValueError, match="output volume"):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=output / "run",
                        inventory_snapshot=inventory)
    assert not (output / "agreement-head-v2-evaluation.claim.json").exists()


def test_fixed_case_selection_and_planned_reservation_must_agree(monkeypatch):
    def encoded(value):
        return json.dumps(value, sort_keys=True).encode()
    selection = encoded({"status": "frozen_for_prospective_evaluation", "candidate": {"head": 4},
                         "comparator": {"head": 8}})
    selection_sha = hashlib.sha256(selection).hexdigest()
    cases = []
    records = []
    for i in range(16):
        row = [1, i + 2, 30]
        cases.append({"text": f"case {i}", "token_ids": row,
                      "continuations": [{"token_id": token, "tokens": row + [token]}
                                        for token in (318, 389, 373, 547)]})
        records.append({"record_id": f"planned-agreement-head-v2-{i + 1:02d}", "category": "planned",
                        "public_text": f"case {i}", "token_ids": row, "candidate_token_ids": [318, 389, 373, 547],
                        "reference_commit": "4d8b93ad4bf601cb3f01f0a9c53aeb01fc43adfe"})
    case_raw = encoded({"selection_sha256": selection_sha, "candidate": {"layer": 3, "head": 4},
                        "comparator": {"layer": 3, "head": 8}, "cases": cases})
    reservation = {"version": "distilgpt2-development-exposures-v39", "records": records}
    inventory_raw = encoded(reservation)
    monkeypatch.setattr(owner, "SELECTION_SHA256", selection_sha)
    monkeypatch.setattr(owner, "CASE_SHA256", hashlib.sha256(case_raw).hexdigest())
    monkeypatch.setattr(owner, "INVENTORY_SHA256", hashlib.sha256(inventory_raw).hexdigest())
    assert owner._fixed_inputs(case_raw, selection, inventory_raw) == cases
    reservation["records"][0]["category"] = "observed-saved"
    changed = encoded(reservation)
    monkeypatch.setattr(owner, "INVENTORY_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="planned reservation"):
        owner._fixed_inputs(case_raw, selection, changed)
