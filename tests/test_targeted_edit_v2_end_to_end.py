"""Synthetic real-Git handoff integration; never loads a pretrained model."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_custody as custody
from looking_glass import targeted_edit_v2_evidence as evidence
from looking_glass import targeted_edit_v2_exposure as exposure
from looking_glass import targeted_edit_v2_gate as gate
from looking_glass import targeted_edit_v2_manifest as manifest_module


ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout.decode("ascii").strip()


def write_json(path, value):
    path.write_bytes(v1._canonical(value))


def attestation(dose, weight_norm, digit):
    planned = dose * weight_norm
    return {"planned_norm": planned, "realized_norm": planned,
            "realized_relative_norm": planned / weight_norm,
            "intended_delta_sha256": digit * 64, "realized_weight_sha256": digit * 64}


def fake_development(directory, *, source, manifest_sha, model, environment, public, custody_result, prior):
    directory.mkdir()
    run_id = "a" * 32
    common = {"version": v2.VERSION, "phase": "development", "run_id": run_id,
              "manifest_sha256": manifest_sha, "source_commit": source}
    write_json(directory / "started.json", {**common, "started_utc": "2026-10-29T03:30:00+00:00"})
    baseline = np.zeros((len(v1.CASES), v1.VOCAB_SIZE), dtype="<f4")
    baseline[:5, v1.LONDON] = -1
    with open(ROOT / "results/targeted-edit-cpu-v1-20261028/reference/original.f32", "rb") as handle:
        baseline[0] = np.frombuffer(handle.read(v1.VOCAB_SIZE * 4), dtype="<f4")
    arrays = {name: baseline.copy() for name in evidence.CONDITIONS}
    arrays["dose-0001"][:5, v1.LONDON] = 2
    arrays["dose-0001"][0, v1.LONDON] = baseline[0, v1.PARIS] + 2
    selection = v2.select_development(arrays)
    assert selection["selected_condition"] == "dose-0001"
    weight_norm = prior["discovery"]["weight_norm"]
    grid = {name: attestation(dose, weight_norm, str(index + 1))
            for index, (name, dose) in enumerate(v2.GRID)}
    controls = {"selected": grid["dose-0001"],
                "random": attestation(0.001, weight_norm, "a"),
                "opposite": attestation(0.001, weight_norm, "b")}
    compatibility = {"status": "matching_recorded_v1", "reference_precommit_sha256": prior["precommit_sha256"],
                     "model_state_sha256": prior["model_state_sha256"], "weight_sha256": prior["weight_sha256"],
                     "discovery_logit_sha256": prior["discovery"]["discovery_logit_sha256"],
                     "gradient_delta_sha256": prior["candidates"]["gradient"]["intended_delta_sha256"],
                     "v1_random_delta_sha256": prior["candidates"]["random"]["intended_delta_sha256"]}
    write_json(directory / "metadata.json", {**common, "model": model, "environment": environment,
                "exposure": public, "custody": custody_result, "compatibility": compatibility,
                "grid": grid, "controls": controls})
    write_json(directory / "selection.json", {**common, "selection": selection})
    row_size = v1.VOCAB_SIZE * 4
    journal = bytearray()
    state_journal = bytearray()
    for name in evidence.CONDITIONS:
        applied = prior["weight_sha256"] if name == "original" else grid[name]["realized_weight_sha256"]
        state_journal.extend(v1._canonical({"version": v2.VERSION, "phase": "development", "run_id": run_id,
                            "condition": name, "model_state_before_sha256": prior["model_state_sha256"],
                            "weight_before_sha256": prior["weight_sha256"], "applied_weight_sha256": applied,
                            "model_state_after_sha256": prior["model_state_sha256"],
                            "weight_after_sha256": prior["weight_sha256"], "flags_restored": True,
                            "grads_cleared": True, "hooks_restored": True}))
        raw = arrays[name].tobytes()
        (directory / f"{name}.f32").write_bytes(raw)
        for index, case in enumerate(v1.CASES):
            offset = index * row_size
            journal.extend(v1._canonical({"version": v2.VERSION, "phase": "development", "run_id": run_id,
                                         "condition": name, "case_id": case[0], "offset": offset,
                                         "bytes": row_size, "sha256": v1._sha(raw[offset:offset + row_size]),
                                         "denormal_probe_before": evidence.DENORMAL_PROBE,
                                         "denormal_probe_after": evidence.DENORMAL_PROBE}))
    (directory / "observations.jsonl").write_bytes(journal)
    (directory / "condition_states.jsonl").write_bytes(state_journal)
    artifacts = {*evidence.DOCUMENTS, *(f"{name}.f32" for name in evidence.CONDITIONS)}
    terminal = {**common, "status": "complete", "finished_utc": "2026-10-29T03:31:00+00:00",
                "cleanup": {key: True for key in evidence.CLEANUP_KEYS},
                "artifacts": {name: v1._sha((directory / name).read_bytes()) for name in artifacts}}
    write_json(directory / "terminal.json", terminal)
    return v1._sha((directory / "terminal.json").read_bytes())


def test_real_source_manifest_custody_selection_git_handoff(tmp_path, monkeypatch):
    assert torch.set_flush_denormal(False)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "anchor.txt").write_text("synthetic prior evidence", encoding="utf-8")
    git(repo, "add", "anchor.txt")
    git(repo, "commit", "-m", "anchor")
    anchor = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(manifest_module, "V1_EVIDENCE_COMMIT", anchor)
    monkeypatch.setattr(exposure, "INVENTORY_COMMIT", anchor)
    for name in manifest_module.SOURCE_PATHS:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, target)
        else:
            target.write_text("# synthetic owner placeholder\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "source")
    source = git(repo, "rev-parse", "HEAD")
    frozen = manifest_module.make_manifest(repo, source)
    write_json(repo / "frozen.json", frozen)
    manifest_sha = v1._sha((repo / "frozen.json").read_bytes())
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "manifest")
    manifest_commit = git(repo, "rev-parse", "HEAD")
    observations = [{"lane": lane, "observed_utc": "2026-10-29T03:29:00+00:00",
                     "result": "no_protected_inputs_reported"} for lane in custody.LANES]
    public_check = {"status": "checked_no_collision", "observed_utc": "2026-10-29T03:29:00+00:00",
                    "commit_label": anchor, "payload_sha256": exposure.INVENTORY_SHA256,
                    "additional_rows_checked": 0, "collision_count": 0}
    write_json(repo / "custody.json", custody.receipt(manifest_sha, source, observations, public_check))
    custody_sha = v1._sha((repo / "custody.json").read_bytes())
    git(repo, "add", "custody.json")
    git(repo, "commit", "-m", "custody")
    custody_commit = git(repo, "rev-parse", "HEAD")
    custody_args = {"manifest_path": "frozen.json", "manifest_commit": manifest_commit,
                    "manifest_sha256": manifest_sha, "receipt_path": "custody.json",
                    "receipt_commit": custody_commit, "receipt_sha256": custody_sha}
    custody_result = custody.verify_committed_receipt(repo, **custody_args)
    public = {"known_row_collisions": 0, "private_split_check_performed": False}
    monkeypatch.setattr(gate.exposure, "verify_public_reservations", lambda *_a, **_kw: public)
    prior = frozen["v1_compatibility"]
    terminal_sha = fake_development(repo / "development", source=source, manifest_sha=manifest_sha,
                        model=prior["model"], environment=prior["environment"], public=public,
                        custody_result=custody_result, prior=prior)
    git(repo, "add", "development")
    git(repo, "commit", "-m", "selection")
    selection_commit = git(repo, "rev-parse", "HEAD")
    snapshot = gate.prepare_committed_selection(repo, manifest_path="frozen.json", manifest_commit=manifest_commit,
                    manifest_sha256=manifest_sha, custody_path="custody.json", custody_commit=custody_commit,
                    custody_sha256=custody_sha, development_dir="development", selection_commit=selection_commit,
                    terminal_sha256=terminal_sha)
    assert v1._decode_document(snapshot.selection_bytes)["selected_condition"] == "dose-0001"
    assert snapshot.selection_commit == selection_commit


def test_real_git_gate_rejects_development_directory_at_receipt_commit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "source.txt").write_text("source", encoding="utf-8")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "source")
    (repo / "frozen.json").write_text("manifest", encoding="utf-8")
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "manifest")
    manifest_commit = git(repo, "rev-parse", "HEAD")
    (repo / "development").mkdir()
    (repo / "development/partial.f32").write_bytes(b"already exists")
    git(repo, "add", "development")
    git(repo, "commit", "-m", "early development")
    (repo / "custody.json").write_text("custody", encoding="utf-8")
    git(repo, "add", "custody.json")
    git(repo, "commit", "-m", "custody")
    custody_commit = git(repo, "rev-parse", "HEAD")
    (repo / "selection.txt").write_text("later selection", encoding="utf-8")
    git(repo, "add", "selection.txt")
    git(repo, "commit", "-m", "selection")
    selection_commit = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(gate.manifest_module, "verify_manifest", lambda *a, **kw: {"source_commit": "a" * 40})
    monkeypatch.setattr(gate.exposure, "verify_public_reservations", lambda *a, **kw: {})
    monkeypatch.setattr(gate.custody, "verify_committed_receipt", lambda *a, **kw: {})
    with pytest.raises(v1.StudyError):
        gate.prepare_committed_selection(repo, manifest_path="frozen.json", manifest_commit=manifest_commit,
                manifest_sha256="b" * 64, custody_path="custody.json", custody_commit=custody_commit,
                custody_sha256="c" * 64, development_dir="development", selection_commit=selection_commit,
                terminal_sha256="d" * 64)


def test_literal_absence_and_missing_historical_tree_fail_closed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    git(repo, "config", "gc.auto", "0")
    (repo / "source.txt").write_text("source", encoding="utf-8")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "source")
    (repo / "frozen.json").write_text("manifest", encoding="utf-8")
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "manifest")
    manifest_commit = git(repo, "rev-parse", "HEAD")
    early = repo / "early" / "[selection]"
    early.mkdir(parents=True)
    (early / "value.txt").write_text("preexisting development", encoding="utf-8")
    (repo / "custody.json").write_text("custody", encoding="utf-8")
    git(repo, "add", "early", "custody.json")
    git(repo, "commit", "-m", "custody including early development")
    custody_commit = git(repo, "rev-parse", "HEAD")
    assert not evidence._path_absent(repo, custody_commit, "early/[selection]")
    assert not evidence._path_absent(repo, custody_commit, "early/[selection]/value.txt")
    assert evidence._path_absent(repo, custody_commit, "early/s/value.txt")
    assert evidence._path_absent(repo, custody_commit, "early/[missing]/value.txt")
    old_subtree = git(repo, "rev-parse", f"{custody_commit}:early")
    (early / "value.txt").unlink()
    (early / "later.txt").write_text("later development", encoding="utf-8")
    git(repo, "add", "early")
    git(repo, "commit", "-m", "selection")
    selection_commit = git(repo, "rev-parse", "HEAD")
    old_object = repo / ".git" / "objects" / old_subtree[:2] / old_subtree[2:]
    assert old_object.is_file() and repo.resolve() in old_object.resolve().parents
    os.chmod(old_object, stat.S_IWRITE | stat.S_IREAD)
    old_object.unlink()
    with pytest.raises(v1.StudyError, match="Git verification failed"):
        evidence._path_absent(repo, custody_commit, "early/[selection]")
    monkeypatch.setattr(gate.manifest_module, "verify_manifest", lambda *a, **kw: {"source_commit": "a" * 40})
    monkeypatch.setattr(gate.exposure, "verify_public_reservations", lambda *a, **kw: {})
    monkeypatch.setattr(gate.custody, "verify_committed_receipt", lambda *a, **kw: {})
    with pytest.raises(v1.StudyError, match="Git verification failed"):
        gate.prepare_committed_selection(repo, manifest_path="frozen.json", manifest_commit=manifest_commit,
                manifest_sha256="b" * 64, custody_path="custody.json", custody_commit=custody_commit,
                custody_sha256="c" * 64, development_dir="early/[selection]", selection_commit=selection_commit,
                terminal_sha256="d" * 64)
