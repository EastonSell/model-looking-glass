"""Exercise the direct-child owner without loading a checkpoint."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import flan_weight_edit_owner as owner
import flan_weight_edit_driver as driver
import pytest


def test_handoff_rejects_inconsistent_child_exit():
    phase = {"pid": 100, "parent_pid": 90}
    record = {"status": "complete_candidate", "exit_code": 0, "reaped": True, "drained": True,
              "timed_out": False, "error": None, "identity_matches": True,
              "phase_sha256": "a" * 64, "direct_child_pid": 100, "operator_pid": 90,
              "phase_consistent": True, "within_evidence_cap": True, "cumulative_evidence_bytes_before_exit_record": 10,
              "logs": {name: {"finished": True, "truncated": False, "error": None, "bytes": 0, "sha256": "b" * 64}
                       for name in ("stdout", "stderr")}}
    driver._check_exit(phase, record, "a" * 64)
    for field, bad in (("exit_code", False), ("timed_out", True), ("error", "lost"),
                       ("identity_matches", False), ("direct_child_pid", 101), ("operator_pid", 91)):
        changed = {**record, field: bad}
        with pytest.raises(ValueError, match="process observation"):
            driver._check_exit(phase, changed, "a" * 64)
    changed = dict(record)
    changed.pop("error")
    with pytest.raises(ValueError, match="process observation"):
        driver._check_exit(phase, changed, "a" * 64)
    changed = {**record, "logs": {**record["logs"], "stdout": {**record["logs"]["stdout"], "finished": False}}}
    with pytest.raises(ValueError, match="child logs"):
        driver._check_exit(phase, changed, "a" * 64)


def test_handoff_rejects_contradictory_stdout_identity(tmp_path):
    phase = {"pid": 100, "parent_pid": 90}
    log = tmp_path / "stdout.log"
    log.write_text('{"pid":100,"parent_pid":90}\n', encoding="utf8")
    driver._check_stdout_identity(log, phase)
    log.write_text('{"pid":777,"parent_pid":888}\n', encoding="utf8")
    with pytest.raises(ValueError, match="identity differs"):
        driver._check_stdout_identity(log, phase)


def test_independent_review_binds_process_and_restoration():
    names = ("result", "events", "logits", "phase", "exit", "stdout", "stderr", "scope")
    artifacts = {name: {"sha256": str(index) * 64} for index, name in enumerate(names)}
    review = {"status": "passed", "source_revision": "a" * 40,
              "review_scope": ["saved-logits", "saved-process", "restoration"],
              **{name + "_sha256": artifacts[name]["sha256"] for name in names}}
    driver._check_review(review, "a" * 40, artifacts)
    for name in ("exit", "stdout", "stderr", "scope"):
        altered = {**review, name + "_sha256": "f" * 64}
        with pytest.raises(ValueError, match="review artifact"):
            driver._check_review(altered, "a" * 40, artifacts)
    with pytest.raises(ValueError, match="review scope"):
        driver._check_review({**review, "review_scope": ["saved-logits"]}, "a" * 40, artifacts)


def test_published_binary_parts_reassemble_on_e_volume(tmp_path):
    repo = tmp_path / "publication-git"
    repo.mkdir()
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo).decode().strip()
    git("init")
    git("config", "user.name", "Synthetic test")
    git("config", "user.email", "synthetic@example.invalid")
    parts = []
    full = b"first synthetic bytes\x00second synthetic bytes\xff"
    for index, raw in enumerate((full[:22], full[22:])):
        name = f"part-{index:03}.bin"
        (repo / name).write_bytes(raw)
        parts.append({"index": index, "bytes": len(raw), "git_path": name,
                      "sha256": hashlib.sha256(raw).hexdigest()})
    local = tmp_path / "complete.bin"
    local.write_bytes(full)
    git("add", "part-000.bin", "part-001.bin")
    git("commit", "-m", "synthetic parts")
    revision = git("rev-parse", "HEAD")
    descriptor = {"local_path": str(local), "sha256": hashlib.sha256(full).hexdigest(), "parts": parts}
    assert driver._published_artifact(repo, revision, descriptor)
    descriptor["parts"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="part bytes"):
        driver._published_artifact(repo, revision, descriptor)
    descriptor["parts"][1]["sha256"] = hashlib.sha256(full[22:]).hexdigest()
    descriptor["parts"][0]["index"] = False
    with pytest.raises(ValueError, match="part identity"):
        driver._published_artifact(repo, revision, descriptor)


def test_source_bound_direct_child_no_model_rehearsal(tmp_path):
    tests_dir = Path(__file__).resolve().parent
    source = tests_dir if (tests_dir / "flan_weight_edit_owner.py").is_file() else tests_dir.parents[1] / "scripts"
    workspace = source.parent
    if source == tests_dir:
        default_work = workspace / "work"
        reference = default_work / "teams/cpu_trace_output_v1"
    else:
        supplied_work = os.environ.get("FLAN_WEIGHT_TEST_WORK")
        if not supplied_work:
            pytest.skip("set FLAN_WEIGHT_TEST_WORK to the existing shared work directory for the direct-child rehearsal")
        default_work = Path(supplied_work)
        if not default_work.is_dir():
            pytest.skip("FLAN_WEIGHT_TEST_WORK is not an existing shared work directory")
        reference = default_work / "teams/cpu_trace_output_v1"
    execution_source = Path(os.environ.get("FLAN_WEIGHT_TEST_EXEC_SOURCE", str(source))).resolve()
    if execution_source.drive.upper() != "C:":
        pytest.skip("direct-child rehearsal requires FLAN_WEIGHT_TEST_EXEC_SOURCE on the bound C: source volume")
    def source_file(filename):
        flat = source / filename
        if flat.is_file():
            return flat
        return tests_dir / ("conftest.py" if filename == "flan_weight_edit_conftest.py" else filename)
    repo = tmp_path / "synthetic-git"
    repo.mkdir()
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo).decode().strip()
    git("init")
    git("config", "user.name", "Synthetic test")
    git("config", "user.email", "synthetic@example.invalid")
    git("config", "core.autocrlf", "false")
    sources = {}
    for filename in owner.RUNTIME_FILES + owner.TEST_FILES:
        location = owner.source_git_path(filename)
        destination = repo / location
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file(filename), destination)
        sources[filename] = {"git_path": location, "sha256": owner.digest(source_file(filename))}
    for location in owner.PUBLIC_INPUTS:
        destination = repo / location
        destination.parent.mkdir(parents=True, exist_ok=True)
        if location.endswith("STORAGE_V1.md"):
            destination.write_bytes((default_work / "flan_weight_edit_volume_plan.md").read_bytes())
        elif location.endswith("STORAGE_AMENDMENT_V1.md"):
            destination.write_bytes((default_work / "flan_weight_edit_storage_amendment.md").read_bytes())
        elif location.endswith("PUBLICATION_V1.md"):
            destination.write_bytes((default_work / "flan_weight_edit_publication_plan.md").read_bytes())
        else:
            revision = "a96bba67ca8067aa7bd6408455bfda81d9bb03b6" if location.startswith("experiments/") else "bc18fda362a102adda8ad1d94839089a83019d3e"
            destination.write_bytes(subprocess.check_output(["git", "--no-replace-objects", "show", revision + ":" + location], cwd=reference))
    git("add", "scripts", "tests", "docs", "experiments")
    git("commit", "-m", "synthetic source")
    source_revision = git("rev-parse", "HEAD")
    manifest = {"schema": "flan-weight-edit-manifest-v1", "source_revision": source_revision,
                "case_sha256": owner.CASE_SHA, "protocol_sha256": owner.PROTOCOL_SHA,
                "asset_manifest_sha256": owner.ASSET_SHA, "runtime": owner.RUNTIME,
                "contract": owner.CONTRACT, "sources": sources}
    manifest_path = repo / owner.MANIFEST_PATH
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf8")
    git("add", owner.MANIFEST_PATH)
    git("commit", "-m", "synthetic manifest")
    manifest_revision = git("rev-parse", "HEAD")
    scope = {"schema": "flan-weight-edit-scope-v1", "case_sha256": owner.CASE_SHA,
             "observed_utc": datetime.now(timezone.utc).isoformat(), "source_revision": source_revision,
             "manifest_sha256": owner.digest(manifest_path), "known_public_comparisons_clear": True,
             "no_unreserved_relevant_inputs_reported": True, "source_and_manifest_reviewed": True,
             "no_additional_relevant_protected_pool_reported": True,
             "known_injection_cases_sha256": owner.KNOWN_INJECTION_SHA,
             "sentiment_pool": {"status": "unformed", "custodian": "review_broker",
                                "proposal_sha256": owner.CASE_SHA}}
    scope_path = tmp_path / "synthetic-scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf8")
    prefix = tmp_path / "outer-prefix"
    prefix.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    attempt = run / "development"
    args = [sys._base_executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
            str(execution_source / "flan_weight_edit_owner.py"), "--phase", "development", "--rehearsal",
            "--manifest-revision", manifest_revision, "--repo", str(repo), "--source-dir", str(execution_source),
            "--manifest", str(manifest_path), "--scope", str(scope_path),
            "--cases", str(default_work / "flan_weight_edit_cases_v1.json"),
            "--assets", str(default_work / "t5-cache/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"),
            "--venv", str(default_work / ".venv"), "--attempt", str(attempt)]
    finished = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    assert finished.returncode == 0, finished.stderr.decode("utf8", "replace")
    phase = owner.read_json(attempt / "phase.json")
    observed = owner.read_json(attempt / "operator-exit.json")
    assert phase["execution_status"] == "rehearsal" and phase["model_loaded"] is False
    assert observed["status"] == "complete_candidate" and observed["reaped"] and observed["drained"]
    assert observed["direct_child_pid"] == phase["pid"]
    stale_owner = tmp_path / "stale-owner.py"
    shutil.copyfile(source / "flan_weight_edit_owner.py", stale_owner)
    other_prefix = tmp_path / "stale-prefix"
    other_prefix.mkdir()
    stale_args = list(args)
    stale_args[5] = "pycache_prefix=" + str(other_prefix)
    stale_args[6] = str(stale_owner)
    stale_attempt = run / "stale-attempt"
    stale_args[stale_args.index("--attempt") + 1] = str(stale_attempt)
    stale = subprocess.run(stale_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert stale.returncode != 0 and b"executing owner differs" in stale.stderr
    assert not stale_attempt.exists()
