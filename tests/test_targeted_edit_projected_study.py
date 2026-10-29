"""Cooperative owner gate tests, with fabricated files and no model."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "targeted_edit_projected_study.py"
SPEC = importlib.util.spec_from_file_location("projected_study_test", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def observation(path, when=None):
    timestamp = (when or datetime.now(timezone.utc)).isoformat()
    path.write_bytes(study.canonical({"observations": [
        {"custodian": "natural", "reported_by": "natural custodian", "observed_utc": timestamp,
         "status": "no_protected_rows_reported"},
        {"custodian": "injection", "reported_by": "injection custodian", "observed_utc": timestamp,
         "status": "no_protected_rows_reported"}]}))
    return path


def test_custody_requires_recent_scoped_no_input_reports(tmp_path):
    path = observation(tmp_path / "custody.json")
    assert len(study.custody_observation(path)["observations"]) == 2
    observation(path, datetime.now(timezone.utc) - timedelta(hours=1))
    with pytest.raises(study.StudyError, match="stale"):
        study.custody_observation(path)
    observation(path)
    data = json.loads(path.read_bytes())
    data["observations"][0]["status"] = "unknown"
    path.write_bytes(study.canonical(data))
    with pytest.raises(study.StudyError, match="explicit no-input"):
        study.custody_observation(path)


def test_recorded_chronology_cannot_run_backwards():
    study.check_reported_order("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00")
    with pytest.raises(study.StudyError, match="contradictory"):
        study.check_reported_order("2026-01-01T00:00:01+00:00", "2026-01-01T00:00:00+00:00")
    with pytest.raises(study.StudyError, match="contradictory"):
        study.check_reported_order("2026-01-01T00:00:00", "2026-01-01T00:00:01+00:00")


def fake_manifest():
    modules = ("__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
               "pretrained", "targeted_edit_cpu_v1", "targeted_edit_v2_edits", "targeted_edit_v2_owner",
               "targeted_edit_projected_capture", "targeted_edit_projected_development",
               "targeted_edit_projected_math", "targeted_edit_projected_execution", "targeted_edit_projected_score")
    return {"source_commit": "source", "checkpoint": {"asset_sha256": {"asset": "sha"}},
            "source_sha256": {"looking_glass/" + name + ".py": "digest" for name in modules}}


def good_record():
    return {"status": "complete", "phase": "development", "source_commit": "source",
            "manifest_sha256": "manifest", "job_drained": True, "owner_reaped": True,
            "child": {"candidate_sha256": "candidate", "model_disposed": True,
                      "phase": "development", "manifest_sha256": "manifest",
                      "post_work_checkpoint_assets": {"asset": "sha"},
                      "imported_sources": fake_manifest()["source_sha256"],
                      "owner_pid": 9, "owner_run_id": "a" * 32},
            "supervisor": {"pid": 9, "run_id": "a" * 32, "clean": True, "exit_code": 0, "timed_out": False}}


@pytest.mark.parametrize("field,value", [("clean", False), ("exit_code", 1), ("exit_code", False),
                                          ("timed_out", True), ("pid", 10), ("pid", True),
                                          ("run_id", "other")])
def test_unverified_development_cannot_admit_evaluation(field, value):
    record = good_record()
    study.check_development_record(record, fake_manifest(), "manifest", "candidate")
    record["supervisor"][field] = value
    with pytest.raises(study.StudyError, match="supervision"):
        study.check_development_record(record, fake_manifest(), "manifest", "candidate")


def test_rehashed_asset_or_source_mutation_fails_development_gate():
    record = good_record()
    record["child"]["post_work_checkpoint_assets"]["asset"] = "other"
    with pytest.raises(study.StudyError, match="asset inventory"):
        study.check_development_record(record, fake_manifest(), "manifest", "candidate")
    record = good_record()
    record["child"]["imported_sources"]["looking_glass/pretrained.py"] = "other"
    with pytest.raises(study.StudyError, match="import inventory"):
        study.check_development_record(record, fake_manifest(), "manifest", "candidate")


def test_failure_record_does_not_swallow_new_cancellation(tmp_path, monkeypatch):
    def interrupted(*_args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(study, "write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study.record_failure(tmp_path, "development", ValueError("original error"))


def test_phase_inventory_binds_all_artifacts_and_caps_size(tmp_path):
    for name in ("started.json", "child-result.json", "manifest-review-attestation.json", "public-exposure.json",
                 "public-exposure-attestation.json"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "candidate").mkdir()
    (tmp_path / "candidate" / "a.f32").write_bytes(b"abcd")
    inventory = study.inventory_attempt(tmp_path, "development")
    assert inventory["candidate/a.f32"] == {"bytes": 4, "sha256": study.digest(b"abcd")}
    with (tmp_path / "candidate" / "oversize").open("wb") as stream:
        stream.truncate(10_000_001)
    with pytest.raises(study.StudyError, match="limit"):
        study.inventory_attempt(tmp_path, "development")


def test_empty_or_incomplete_candidate_never_completes(tmp_path):
    for name in ("started.json", "child-result.json", "manifest-review-attestation.json", "public-exposure.json",
                 "public-exposure-attestation.json"):
        (tmp_path / name).write_bytes(b"x")
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    with pytest.raises(study.StudyError, match="empty"):
        study.inventory_attempt(tmp_path, "development")
    raw = study.canonical({"status": "private-candidate", "artifacts": {}})
    (candidate_dir / "pre-evaluation-candidate.json").write_bytes(raw)
    inventory = study.inventory_attempt(tmp_path, "development")
    with pytest.raises(study.StudyError, match="artifact list is incomplete"):
        study.validate_phase_output(tmp_path, "development", {"candidate_sha256": study.digest(raw)}, inventory)


def test_fabricated_complete_candidate_matches_parent_schema(tmp_path):
    from test_targeted_edit_projected_development import guarded_model
    from looking_glass import targeted_edit_projected_development as development
    for name in ("started.json", "child-result.json", "manifest-review-attestation.json", "public-exposure.json",
                 "public-exposure-attestation.json"):
        (tmp_path / name).write_bytes(b"x")
    _model, guard = guarded_model()
    development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, tmp_path / "candidate")
    raw = (tmp_path / "candidate" / "pre-evaluation-candidate.json").read_bytes()
    inventory = study.inventory_attempt(tmp_path, "development")
    study.validate_phase_output(tmp_path, "development", {"candidate_sha256": study.digest(raw)}, inventory)
    (tmp_path / "candidate" / "full-gradient.f32").unlink()
    inventory = study.inventory_attempt(tmp_path, "development")
    with pytest.raises(study.StudyError, match="output set differs"):
        study.validate_phase_output(tmp_path, "development", {"candidate_sha256": study.digest(raw)}, inventory)


def test_tokenizer_requires_frozen_append_boundaries():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"prompt": [1, 2], " token": [3], "prompt token": [1, 2, 3]}[text]
    rows = [{"prompt": "prompt", "token_ids": [1, 2],
             "continuations": [{"text": " token", "ids": [3]}]}]
    study.check_projected_tokenizer(Tokenizer(), rows)
    rows[0]["continuations"][0]["ids"] = [4]
    with pytest.raises(study.StudyError, match="continuation"):
        study.check_projected_tokenizer(Tokenizer(), rows)


def test_current_inventory_rejects_either_prefix_direction():
    rows = [{"case_id": f"projected-{i}", "token_ids": [100 + i, 10]} for i in range(13)]
    records = [{"record_id": "planned-" + row["case_id"], "token_ids": row["token_ids"],
                "category": "planned"} for row in rows]
    payload = {"version": "synthetic", "records": records + [
        {"record_id": "other", "token_ids": [500, 10], "category": "observed"}]}
    cases = {"tokenizer_preflight": {"cases": rows}}
    def payload_sha():
        return study.digest(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                       allow_nan=False).encode("utf-8"))
    assert study.check_public_payload(payload, payload_sha(), cases)["known_relations"] == 0
    with pytest.raises(study.StudyError, match="omits a frozen"):
        study.check_public_payload(payload, payload_sha(), cases, [[777, 1]])
    for collision in ([100], [100, 10, 11]):
        payload["records"][-1]["token_ids"] = collision
        with pytest.raises(study.StudyError, match="prefix relation"):
            study.check_public_payload(payload, payload_sha(), cases)


def test_post_creation_exception_retains_incomplete_attempt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(study, "_ACTIVE_ATTEMPT", tmp_path)
    monkeypatch.setattr(study, "_ACTIVE_PHASE", "development")
    monkeypatch.setattr(study, "_ATTEMPT_OWNED", True)
    def fail(_argv=None):
        raise ValueError("platform setup failed")
    monkeypatch.setattr(study, "main", fail)
    with pytest.raises(ValueError):
        study.entry()
    assert json.loads((tmp_path / "failure.json").read_bytes())["status"] == "incomplete"
    assert json.loads(capsys.readouterr().err)["attempt"] == str(tmp_path)


def test_unowned_collision_does_not_modify_existing_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(study, "_ACTIVE_ATTEMPT", tmp_path)
    monkeypatch.setattr(study, "_ACTIVE_PHASE", "development")
    monkeypatch.setattr(study, "_ATTEMPT_OWNED", False)
    def fail(_argv=None):
        raise FileExistsError("collision or interrupted mkdir")
    monkeypatch.setattr(study, "main", fail)
    with pytest.raises(FileExistsError):
        study.entry()
    assert list(tmp_path.iterdir()) == []
    assert json.loads(capsys.readouterr().err)["attempt_owned"] is False


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout.decode().strip()


def repository(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "synthetic reviewer")
    git(tmp_path, "config", "user.email", "synthetic@example.invalid")
    (tmp_path / "tracked.txt").write_text("source", encoding="utf-8")
    git(tmp_path, "add", "tracked.txt")
    git(tmp_path, "commit", "-m", "source")
    return tmp_path


def test_unrelated_history_does_not_satisfy_phase_chain(tmp_path):
    repo = repository(tmp_path)
    launch = study.launcher(SCRIPT.parents[1])
    source = git(repo, "rev-parse", "HEAD")
    (repo / "manifest.txt").write_text("manifest", encoding="utf-8")
    git(repo, "add", "manifest.txt")
    git(repo, "commit", "-m", "manifest")
    manifest = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "show", "-s", "--format=%T", "HEAD")
    orphan = git(repo, "commit-tree", tree, "-m", "unrelated")
    merge = git(repo, "commit-tree", tree, "-p", manifest, "-p", orphan, "-m", "merge")
    study.check_phase_ancestry(launch, repo, source, manifest, merge, manifest)
    with pytest.raises(Exception):
        study.check_phase_ancestry(launch, repo, source, orphan, merge)
    with pytest.raises(Exception):
        study.check_phase_ancestry(launch, repo, source, manifest, merge, orphan)


def test_extra_failure_marker_rejects_published_phase(tmp_path):
    repo = repository(tmp_path)
    launch = study.launcher(SCRIPT.parents[1])
    directory = repo / "results" / "development"
    (directory / "candidate").mkdir(parents=True)
    (directory / "started.json").write_bytes(b"start")
    (directory / "candidate" / "a.f32").write_bytes(b"value")
    (directory / "phase-record.json").write_bytes(b"record")
    git(repo, "add", "results")
    git(repo, "commit", "-m", "successful phase")
    revision = git(repo, "rev-parse", "HEAD")
    inventory = {"started.json": {}, "candidate/a.f32": {}}
    study.check_exact_phase_files(launch, repo, revision, directory, inventory)
    marker = directory / "failure.json"
    marker.write_bytes(b"incomplete")
    with pytest.raises(study.StudyError, match="unlisted failure"):
        study.check_exact_phase_files(launch, repo, revision, directory, inventory)
    git(repo, "add", "results/development/failure.json")
    git(repo, "commit", "-m", "lost acknowledgement")
    with pytest.raises(study.StudyError, match="unlisted failure"):
        study.check_exact_phase_files(launch, repo, git(repo, "rev-parse", "HEAD"), directory, inventory)


def test_candidate_must_be_in_the_recorded_phase_directory(tmp_path):
    launch = study.launcher(SCRIPT.parents[1])
    phase = tmp_path / "development"
    (phase / "candidate").mkdir(parents=True)
    external = tmp_path / "other-candidate"
    external.mkdir()
    (external / "failure.json").write_text("incomplete", encoding="utf-8")
    record = phase / "phase-record.json"
    record.write_text("record", encoding="utf-8")
    document = {"artifacts": {"a.f32": {}}}
    inventory = {"candidate/a.f32": {}, "candidate/pre-evaluation-candidate.json": {}}
    study.check_candidate_binding(launch, record, phase / "candidate", document, inventory)
    with pytest.raises(study.StudyError, match="candidate directory"):
        study.check_candidate_binding(launch, record, external, document, inventory)
    with pytest.raises(study.StudyError, match="inventories differ"):
        study.check_candidate_binding(launch, record, phase / "candidate", document, {})


def test_redirected_saved_artifact_is_rejected_before_read(tmp_path, monkeypatch):
    class Redirected:
        def _checked_path(self, _path):
            raise study.StudyError("redirected")
    reads = []
    monkeypatch.setattr(study, "read", lambda *_args: reads.append(True))
    with pytest.raises(study.StudyError, match="redirected"):
        study.committed_candidate(Redirected(), tmp_path, "commit", tmp_path, "sha")
    with pytest.raises(study.StudyError, match="redirected"):
        study.check_committed_inventory(Redirected(), tmp_path, "commit", tmp_path,
                                        {"candidate/a": {"bytes": 1, "sha256": "x"}})
    assert reads == []


def test_parent_rejects_redirected_manifest_before_read(tmp_path, monkeypatch):
    original = study.ordinary
    reads = []
    def path_check(path):
        if str(path).endswith("targeted_edit_projected_run_v1.json"):
            raise study.StudyError("redirected manifest")
        return original(path)
    monkeypatch.setattr(study, "ordinary", path_check)
    monkeypatch.setattr(study, "read", lambda *_args: reads.append(True))
    args = ["run", "--checkout", str(tmp_path), "--venv", str(tmp_path), "--dependency", str(tmp_path),
            "--work", str(tmp_path), "--phase", "rehearsal", "--manifest-sha", "sha", "--cache", str(tmp_path),
            "--study-root", str(tmp_path), "--custody-observation", "x", "--manifest-review-attestation", "x",
            "--current-public-inventory", "x", "--current-inventory-sha", "sha", "--current-inventory-attestation", "x"]
    with pytest.raises(study.StudyError, match="redirected manifest"):
        study.main(args)
    assert reads == []


def test_child_rejects_redirected_manifest_before_read(tmp_path, monkeypatch):
    cache = tmp_path / "pycache"
    cache.mkdir()
    fake_sys = SimpleNamespace(flags=SimpleNamespace(isolated=True, no_site=True),
                               dont_write_bytecode=True, pycache_prefix=str(cache), modules={})
    monkeypatch.setattr(study, "sys", fake_sys)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    original = study.ordinary
    reads = []
    def path_check(path):
        if str(path).endswith("targeted_edit_projected_run_v1.json"):
            raise study.StudyError("redirected manifest")
        return original(path)
    monkeypatch.setattr(study, "ordinary", path_check)
    monkeypatch.setattr(study, "read", lambda *_args: reads.append(True))
    with pytest.raises(study.StudyError, match="redirected manifest"):
        study.child(SimpleNamespace(checkout=str(tmp_path), attempt=str(tmp_path)))
    assert reads == []


def test_windows_case_alias_and_missing_tree_fail_closed(tmp_path):
    repo = repository(tmp_path)
    launch = study.launcher(SCRIPT.parents[1])
    directory = repo / "results" / "Development"
    directory.mkdir(parents=True)
    (directory / "partial").write_text("partial", encoding="utf-8")
    git(repo, "add", "results")
    git(repo, "commit", "-m", "prior phase")
    revision = git(repo, "rev-parse", "HEAD")
    assert not study.windows_path_absent(launch, repo, revision, "results/development")
    assert study.windows_path_absent(launch, repo, revision, "results/elsewhere")
    missing = "f" * 40
    tree = subprocess.run(["git", "-C", str(repo), "mktree", "--missing"],
        input=f"040000 tree {missing}\tbroken\n".encode(), check=True, capture_output=True).stdout.decode().strip()
    commit = git(repo, "commit-tree", tree, "-m", "missing tree")
    with pytest.raises(Exception):
        study.windows_path_absent(launch, repo, commit, "broken/child")


def test_annotated_tag_is_not_a_commit_identity(tmp_path):
    repo = repository(tmp_path)
    launch = study.launcher(SCRIPT.parents[1])
    git(repo, "tag", "-a", "synthetic", "-m", "tag")
    tag = git(repo, "rev-parse", "refs/tags/synthetic")
    with pytest.raises(study.StudyError, match="commit object"):
        study.commit_object(launch, repo, tag)


def test_real_no_model_child_claims_the_same_persistent_root(tmp_path):
    from looking_glass.cpu_study_supervision import run_cpu_study
    root = tmp_path / "persistent"
    root.mkdir()
    completed = tmp_path / "child-completed"
    code = ("import sys\nfrom pathlib import Path\n"
            "from looking_glass.cpu_study_supervision import claim_cpu_study\n"
            "assert sys.argv[1] == '--study-root'\n"
            "session, lease = claim_cpu_study(Path(sys.argv[2]))\n"
            "Path(sys.argv[3]).write_text('claimed')\n"
            "session.attest_clean(lease)\n")
    command = [sys._base_executable, "-B", "-c", code, "--study-root", str(root), str(completed)]
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP") if key in os.environ}
    env["PYTHONPATH"] = str(SCRIPT.parents[1])
    result = study.supervise_child(run_cpu_study, command, root, cwd=tmp_path, env=env, job=None)
    assert result.clean and result.exit_code == 0 and completed.read_text() == "claimed"
    assert not (root / "cpu-study.unsafe.json").exists()
    command[command.index("--study-root") + 1] = str(tmp_path / "different")
    with pytest.raises(study.StudyError, match="roots differ"):
        study.supervise_child(run_cpu_study, command, root, cwd=tmp_path, env=env, job=None)
