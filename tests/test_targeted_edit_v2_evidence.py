"""Adversarial saved-artifact checks; no pretrained model is loaded."""

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_evidence as evidence


@pytest.fixture
def tiny_vocab(monkeypatch):
    assert torch.set_flush_denormal(False)
    monkeypatch.setattr(v1, "VOCAB_SIZE", 16)
    monkeypatch.setattr(v1, "LONDON", 1)
    monkeypatch.setattr(v1, "PARIS", 2)
    cases = tuple((*case[:4], (" label", 3) if case[1] == "locality" else None)
                  for case in v1.CASES)
    monkeypatch.setattr(v1, "CASES", cases)


def write_json(path, data):
    path.write_bytes(v1._canonical(data))


def refresh_terminal(directory):
    path = directory / "terminal.json"
    terminal = v1._decode_document(path.read_bytes())
    terminal["artifacts"] = {name: v1._sha((directory / name).read_bytes())
                             for name in terminal["artifacts"]}
    write_json(path, terminal)
    return v1._sha(path.read_bytes())


def make_development(directory, manifest_sha="1" * 64, source="2" * 40):
    directory.mkdir(parents=True)
    run_id = "a" * 32
    common = {"version": v2.VERSION, "phase": evidence.PHASE, "run_id": run_id,
              "manifest_sha256": manifest_sha, "source_commit": source}
    write_json(directory / "started.json", {**common, "started_utc": "2026-10-29T01:00:00+00:00"})
    owner = {"run_id": "b" * 32, "pid": 123}
    supervisor = {**owner, "exit_code": 0, "clean": True, "timed_out": False, "reaped": True}
    write_json(directory / "owner.json", {**common, "supervisor_run_id": owner["run_id"], "pid": owner["pid"]})
    baseline = np.zeros((len(v1.CASES), v1.VOCAB_SIZE), dtype="<f4")
    baseline[:5, 1] = -1
    arrays = {name: baseline.copy() for name in evidence.CONDITIONS}
    arrays["dose-0001"][:5, 1] = 2
    compatibility = {"model_state_sha256": "e" * 64, "weight_sha256": "f" * 64,
                     "discovery_logit_sha256": v1._sha(baseline[0].tobytes())}
    grid = {name: {"realized_weight_sha256": str(index + 1) * 64}
            for index, name in enumerate(evidence.CONDITIONS[1:])}
    environment = {"intraop_threads": 1, "interop_threads": 1, "deterministic_algorithms": True,
                   "mkldnn_enabled": True, "cuda_initialized": False, "omp_num_threads": None,
                   "mkl_num_threads": None}
    settings = {**environment, "float32_matmul_precision": "highest", "mkldnn_deterministic": False,
                "deterministic_warn_only": False, "cpu_autocast_enabled": False,
                "inference_mode_enabled": False, "cuda_visibility": "", "hf_hub_offline": "1",
                "transformers_offline": "1"}
    write_json(directory / "metadata.json", {**common, "description": "synthetic-only",
                                             "compatibility": compatibility, "grid": grid, "buffer_state_sha256": "c" * 64,
                                             "environment": environment, "runtime_settings": settings,
                                             "analysis_denormal_probe_before": copy.deepcopy(evidence.DENORMAL_PROBE),
                                             "analysis_denormal_probe_after": copy.deepcopy(evidence.DENORMAL_PROBE),
                                             "supervision": {"owner": owner, "supervisor": supervisor}})
    row_size = v1.VOCAB_SIZE * 4
    journal = bytearray()
    state_journal = bytearray()
    for name in evidence.CONDITIONS:
        applied = compatibility["weight_sha256"] if name == "original" else grid[name]["realized_weight_sha256"]
        state_journal.extend(v1._canonical({"version": v2.VERSION, "phase": evidence.PHASE, "run_id": run_id,
                            "condition": name, "model_state_before_sha256": compatibility["model_state_sha256"],
                            "weight_before_sha256": compatibility["weight_sha256"], "applied_weight_sha256": applied,
                            "model_state_after_sha256": compatibility["model_state_sha256"],
                            "weight_after_sha256": compatibility["weight_sha256"],
                            "buffer_state_before_sha256": "c" * 64, "buffer_state_after_sha256": "c" * 64,
                            "flags_restored": True,
                            "grads_cleared": True, "hooks_restored": True}))
        raw = arrays[name].tobytes()
        (directory / f"{name}.f32").write_bytes(raw)
        for index, case in enumerate(v1.CASES):
            offset = index * row_size
            journal.extend(v1._canonical({"version": v2.VERSION, "phase": evidence.PHASE,
                                         "run_id": run_id, "condition": name, "case_id": case[0],
                                         "offset": offset, "bytes": row_size,
                                         "sha256": v1._sha(raw[offset:offset + row_size]),
                                         "denormal_probe_before": evidence.DENORMAL_PROBE,
                                         "denormal_probe_after": evidence.DENORMAL_PROBE}))
    (directory / "observations.jsonl").write_bytes(journal)
    (directory / "condition_states.jsonl").write_bytes(state_journal)
    write_json(directory / "selection.json", {**common, "selection": v2.select_development(arrays)})
    artifacts = {*evidence.DOCUMENTS, *(f"{name}.f32" for name in evidence.CONDITIONS)}
    terminal = {**common, "status": "complete", "finished_utc": "2026-10-29T01:01:00+00:00",
                "cleanup": {name: True for name in evidence.CLEANUP_KEYS},
                "artifacts": {name: v1._sha((directory / name).read_bytes()) for name in artifacts}}
    write_json(directory / "terminal.json", terminal)
    return manifest_sha, v1._sha((directory / "terminal.json").read_bytes())


def test_valid_saved_selection_is_recomputed(tiny_vocab, tmp_path):
    directory = tmp_path / "development"
    manifest, terminal = make_development(directory)
    result = evidence.audit_development(directory, manifest_sha256=manifest, terminal_sha256=terminal)
    assert result["status"] == "matching_saved_selection"
    assert result["selection"]["selected_condition"] == "dose-0001"
    assert result["selection"]["development_qualified"] is True
    assert result["metadata"]["description"] == "synthetic-only"


@pytest.mark.parametrize("mutation", ["owner_run", "owner_pid", "parent_run", "parent_pid",
                                      "exit_code", "clean", "timeout", "not_reaped", "bool_pid", "missing"])
def test_supervision_must_bind_the_child_to_a_clean_reaped_result(tiny_vocab, tmp_path, mutation):
    directory = tmp_path / "development"
    manifest, _ = make_development(directory)
    path = directory / ("owner.json" if mutation.startswith("owner_") else "metadata.json")
    record = v1._decode_document(path.read_bytes())
    if mutation == "owner_run":
        record["supervisor_run_id"] = "c" * 32
    elif mutation == "owner_pid":
        record["pid"] = 124
    elif mutation == "missing":
        del record["supervision"]
    else:
        sup = record["supervision"]["supervisor"]
        if mutation == "parent_run":
            sup["run_id"] = "c" * 32
        elif mutation == "parent_pid":
            sup["pid"] = 124
        elif mutation == "bool_pid":
            sup["pid"] = True
        elif mutation == "exit_code":
            sup["exit_code"] = 1
        elif mutation == "clean":
            sup["clean"] = False
        elif mutation == "not_reaped":
            sup["reaped"] = False
        else:
            sup["timed_out"] = True
    write_json(path, record)
    pin = refresh_terminal(directory)
    with pytest.raises(v1.StudyError):
        evidence.audit_development(directory, manifest_sha256=manifest, terminal_sha256=pin)


@pytest.mark.parametrize("field", ["analysis_denormal_probe_before", "analysis_denormal_probe_after"])
def test_analysis_probe_change_is_rejected_even_after_rehash(tiny_vocab, tmp_path, field):
    directory = tmp_path / "development"
    manifest, _ = make_development(directory)
    path = directory / "metadata.json"
    metadata = v1._decode_document(path.read_bytes())
    metadata[field]["subnormal_times_one_bits"] = 0
    write_json(path, metadata)
    pin = refresh_terminal(directory)
    with pytest.raises(v1.StudyError, match="denormal"):
        evidence.audit_development(directory, manifest_sha256=manifest, terminal_sha256=pin)


@pytest.mark.parametrize("field", ["float32_matmul_precision", "deterministic_warn_only", "intraop_threads",
                                    "mkldnn_enabled", "cpu_autocast_enabled", "cuda_visibility", "hf_hub_offline"])
def test_inconsistent_runtime_settings_are_rejected_after_rehash(tiny_vocab, tmp_path, field):
    directory = tmp_path / "development"
    manifest, _ = make_development(directory)
    path = directory / "metadata.json"
    metadata = v1._decode_document(path.read_bytes())
    metadata["runtime_settings"][field] = {"float32_matmul_precision": "unknown", "deterministic_warn_only": True,
        "intraop_threads": True, "mkldnn_enabled": False, "cpu_autocast_enabled": True,
        "cuda_visibility": "0", "hf_hub_offline": "0"}[field]
    write_json(path, metadata)
    pin = refresh_terminal(directory)
    with pytest.raises(v1.StudyError, match="runtime"):
        evidence.audit_development(directory, manifest_sha256=manifest, terminal_sha256=pin)


@pytest.mark.parametrize("field", ["development_qualified", "point_prediction", "selected_condition"])
def test_rehashed_forged_selection_is_not_authoritative(tiny_vocab, tmp_path, field):
    directory = tmp_path / "development"
    manifest, _ = make_development(directory)
    path = directory / "selection.json"
    value = v1._decode_document(path.read_bytes())
    if field == "development_qualified":
        value["selection"][field] = False
    elif field == "point_prediction":
        value["selection"][field] = 100.0
    else:
        value["selection"][field] = "dose-0002"
    write_json(path, value)
    terminal = refresh_terminal(directory)
    with pytest.raises(v1.StudyError):
        evidence.audit_development(directory, manifest_sha256=manifest, terminal_sha256=terminal)


@pytest.mark.parametrize("mutation", ["truncated", "extra", "wrong_case", "bad_probe", "raw_change",
                                      "discovery_mismatch", "wrong_state", "wrong_buffer", "state_order",
                                      "bad_cleanup", "extra_file"])
def test_complete_terminal_rejects_mutated_evidence(tiny_vocab, tmp_path, mutation):
    directory = tmp_path / "development"
    manifest, original_terminal = make_development(directory)
    journal = directory / "observations.jsonl"
    if mutation == "truncated":
        journal.write_bytes(journal.read_bytes()[:-1])
    elif mutation == "extra":
        journal.write_bytes(journal.read_bytes() + b"{}\n")
    elif mutation in ("wrong_case", "bad_probe"):
        lines = journal.read_bytes().splitlines(keepends=True)
        first = v1._decode_document(lines[0])
        if mutation == "wrong_case":
            first["case_id"] = "not-the-discovery"
        else:
            first["denormal_probe_before"]["subnormal_times_one_bits"] = 0
        journal.write_bytes(v1._canonical(first) + b"".join(lines[1:]))
    elif mutation == "raw_change":
        path = directory / "dose-0001.f32"
        raw = bytearray(path.read_bytes())
        raw[8] ^= 1
        path.write_bytes(raw)
    elif mutation == "discovery_mismatch":
        path = directory / "original.f32"
        raw = bytearray(path.read_bytes())
        raw[:4] = np.float32(0.1).tobytes()
        path.write_bytes(raw)
        lines = journal.read_bytes().splitlines(keepends=True)
        first = v1._decode_document(lines[0])
        first["sha256"] = v1._sha(bytes(raw[:v1.VOCAB_SIZE * 4]))
        journal.write_bytes(v1._canonical(first) + b"".join(lines[1:]))
    elif mutation in ("wrong_state", "wrong_buffer", "state_order"):
        path = directory / "condition_states.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        if mutation in ("wrong_state", "wrong_buffer"):
            first = v1._decode_document(lines[0])
            field = "model_state_after_sha256" if mutation == "wrong_state" else "buffer_state_after_sha256"
            first[field] = "0" * 64
            lines[0] = v1._canonical(first)
        else:
            lines[0], lines[1] = lines[1], lines[0]
        path.write_bytes(b"".join(lines))
    elif mutation == "bad_cleanup":
        path = directory / "terminal.json"
        terminal = v1._decode_document(path.read_bytes())
        terminal["cleanup"]["state_restored"] = False
        write_json(path, terminal)
    else:
        (directory / "unexpected.txt").write_text("extra", encoding="utf-8")
    pin = refresh_terminal(directory) if mutation != "extra_file" else original_terminal
    with pytest.raises(v1.StudyError):
        evidence.audit_development(directory, manifest_sha256=manifest, terminal_sha256=pin)


def test_external_terminal_pin_is_required(tiny_vocab, tmp_path):
    directory = tmp_path / "development"
    manifest, _ = make_development(directory)
    with pytest.raises(v1.StudyError):
        evidence.audit_development(directory, manifest_sha256=manifest, terminal_sha256="f" * 64)


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout.decode("ascii").strip()


def test_committed_selection_checks_ancestry_and_exact_blobs(tiny_vocab, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "source.txt").write_text("source", encoding="utf-8")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "source")
    source = git(repo, "rev-parse", "HEAD")
    manifest_path = repo / "frozen.json"
    write_json(manifest_path, {"version": v2.VERSION, "source_commit": source})
    manifest_sha = v1._sha(manifest_path.read_bytes())
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "freeze")
    manifest_commit = git(repo, "rev-parse", "HEAD")
    directory = repo / "outer" / "development"
    _, terminal = make_development(directory, manifest_sha, source)
    git(repo, "add", "outer")
    git(repo, "commit", "-m", "selection")
    selection_commit = git(repo, "rev-parse", "HEAD")
    args = {"directory": "outer/development", "manifest_path": "frozen.json",
            "manifest_commit": manifest_commit, "selection_commit": selection_commit,
            "manifest_sha256": manifest_sha, "terminal_sha256": terminal}
    report = evidence.verify_committed_development(repo, **args)
    assert report["status"] == "matching_committed_selection"
    assert report["selection_commit"] == selection_commit
    real_lstat = Path.lstat

    def parent_reparse(path, *positional, **named):
        result = real_lstat(path, *positional, **named)
        if path == repo / "outer":
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", parent_reparse)
        with pytest.raises(v1.StudyError):
            evidence.verify_committed_development(repo, **args)
    selected_file = "outer/development/selection.json"
    blob = git(repo, "hash-object", selected_file)
    git(repo, "update-index", "--cacheinfo", "120000", blob, selected_file)
    git(repo, "commit", "-m", "record selection as symlink")
    args["selection_commit"] = git(repo, "rev-parse", "HEAD")
    assert (repo / selected_file).is_file()  # working bytes remain an ordinary file
    with pytest.raises(v1.StudyError):
        evidence.verify_committed_development(repo, **args)
    git(repo, "update-index", "--cacheinfo", "100644", blob, selected_file)
    git(repo, "commit", "-m", "restore regular selection")
    args["selection_commit"] = selection_commit
    metadata = v1._decode_document((directory / "metadata.json").read_bytes())
    metadata["description"] = "different but internally rehashed"
    write_json(directory / "metadata.json", metadata)
    args["terminal_sha256"] = refresh_terminal(directory)
    with pytest.raises(v1.StudyError):
        evidence.verify_committed_development(repo, **args)


def test_committed_blob_must_have_a_regular_tree_mode(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "x").write_bytes(b"same bytes")
    git(repo, "add", "x")
    git(repo, "commit", "-m", "regular")
    regular_commit = git(repo, "rev-parse", "HEAD")
    assert evidence._blob(repo, regular_commit, "x", 64) == b"same bytes"
    blob = git(repo, "hash-object", "x")
    git(repo, "update-index", "--cacheinfo", "120000", blob, "x")
    git(repo, "commit", "-m", "symlink")
    assert (repo / "x").is_file()
    with pytest.raises(v1.StudyError):
        evidence._blob(repo, git(repo, "rev-parse", "HEAD"), "x", 64)


def test_selection_directory_must_be_absent_from_manifest_commit(tiny_vocab, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "source.txt").write_text("source", encoding="utf-8")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "source")
    source = git(repo, "rev-parse", "HEAD")
    manifest_path = repo / "frozen.json"
    write_json(manifest_path, {"version": v2.VERSION, "source_commit": source})
    manifest_sha = v1._sha(manifest_path.read_bytes())
    directory = repo / "development"
    directory.mkdir()
    (directory / "selection.json").write_text("already present", encoding="utf-8")
    git(repo, "add", "frozen.json", "development")
    git(repo, "commit", "-m", "invalid early selection")
    manifest_commit = git(repo, "rev-parse", "HEAD")
    (directory / "selection.json").unlink()
    directory.rmdir()
    _, terminal = make_development(directory, manifest_sha, source)
    git(repo, "add", "development")
    git(repo, "commit", "-m", "terminal")
    selection_commit = git(repo, "rev-parse", "HEAD")
    with pytest.raises(v1.StudyError):
        evidence.verify_committed_development(repo, directory="development", manifest_path="frozen.json",
                                            manifest_commit=manifest_commit, selection_commit=selection_commit,
                                            manifest_sha256=manifest_sha, terminal_sha256=terminal)


def test_git_absence_rejects_windows_case_aliases_at_each_component(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    directory = repo / "Runs" / "Development"
    directory.mkdir(parents=True)
    (directory / "Partial.json").write_bytes(b"already present")
    git(repo, "add", "Runs")
    git(repo, "commit", "-m", "early artifact")
    commit = git(repo, "rev-parse", "HEAD")
    assert not evidence._path_absent(repo, commit, "runs/development")
    assert not evidence._path_absent(repo, commit, "runs/development/partial.json")
    assert not evidence._path_absent(repo, commit, "runs/development/partial.json/child")
    assert evidence._path_absent(repo, commit, "runs/other")


@pytest.mark.parametrize("suffix", [".", " "])
def test_git_absence_rejects_windows_trimmed_directory_alias(tmp_path, suffix):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "content").write_bytes(b"tracked")
    blob = git(repo, "hash-object", "-w", "content")
    git(repo, "-c", "core.protectNTFS=false", "update-index", "--add", "--cacheinfo",
        f"100644,{blob},Runs/Development{suffix}/partial")
    tree = git(repo, "write-tree")
    commit = git(repo, "commit-tree", tree, "-m", "synthetic alias")
    assert not evidence._path_absent(repo, commit, "runs/development")


@pytest.mark.parametrize("path", ["runs/évaluation", "runs/SHORT~1", "runs/name.",
                                  "runs/name ", "runs/COM1.json", "runs/name?x"])
def test_windows_chronology_paths_reject_ambiguous_components(path):
    with pytest.raises(v1.StudyError):
        evidence._windows_parts(path)


def test_windows_overlap_is_case_insensitive_and_component_based():
    assert evidence._paths_overlap("runs/Development", "RUNS/development/evaluation")
    assert evidence._paths_overlap("runs/development/evaluation", "RUNS/DEVELOPMENT")
    assert not evidence._paths_overlap("runs/development", "runs/development-extra")


def test_git_absence_fails_closed_on_non_ascii_tree_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "é.txt").write_bytes(b"tracked")
    git(repo, "add", "--all")
    git(repo, "commit", "-m", "unicode entry")
    with pytest.raises(v1.StudyError, match="non-ASCII"):
        evidence._path_absent(repo, git(repo, "rev-parse", "HEAD"), "frozen.json")


def test_git_identity_rejects_annotated_tag_objects(tiny_vocab, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "source.txt").write_text("source", encoding="utf-8")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "source")
    git(repo, "tag", "-a", "source-tag", "-m", "tag")
    source_tag = git(repo, "rev-parse", "refs/tags/source-tag")
    with pytest.raises(v1.StudyError):
        evidence._require_commit_object(repo, source_tag)
    manifest_path = repo / "frozen.json"
    write_json(manifest_path, {"version": v2.VERSION, "source_commit": source_tag})
    manifest_sha = v1._sha(manifest_path.read_bytes())
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "manifest")
    git(repo, "tag", "-a", "manifest-tag", "-m", "tag")
    manifest_tag = git(repo, "rev-parse", "refs/tags/manifest-tag")
    directory = repo / "development"
    _, terminal = make_development(directory, manifest_sha, source_tag)
    git(repo, "add", "development")
    git(repo, "commit", "-m", "selection")
    git(repo, "tag", "-a", "selection-tag", "-m", "tag")
    selection_tag = git(repo, "rev-parse", "refs/tags/selection-tag")
    for tag in (source_tag, manifest_tag, selection_tag):
        with pytest.raises(v1.StudyError):
            evidence._require_commit_object(repo, tag)
    with pytest.raises(v1.StudyError):
        evidence.verify_committed_development(repo, directory="development", manifest_path="frozen.json",
                                            manifest_commit=manifest_tag, selection_commit=selection_tag,
                                            manifest_sha256=manifest_sha, terminal_sha256=terminal)


def test_git_ancestry_ignores_replace_refs_and_legacy_grafts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "source.txt").write_text("source", encoding="utf-8")
    git(repo, "add", "source.txt")
    tree = git(repo, "write-tree")
    source = git(repo, "commit-tree", tree, "-m", "source")
    orphan = git(repo, "commit-tree", tree, "-m", "orphan")
    git(repo, "replace", "--graft", orphan, source)
    assert subprocess.run(["git", "merge-base", "--is-ancestor", source, orphan],
                          cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    assert evidence._git(repo, "merge-base", "--is-ancestor", source, orphan, check=False).returncode == 1
    git(repo, "replace", "-d", orphan)
    (repo / ".git" / "info" / "grafts").write_text(f"{orphan} {source}\n", encoding="ascii")
    assert subprocess.run(["git", "--no-replace-objects", "merge-base", "--is-ancestor", source, orphan],
                          cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    assert evidence._git(repo, "merge-base", "--is-ancestor", source, orphan, check=False).returncode == 1


@pytest.mark.parametrize("path", ["../development", "a/../development", "C:/x", "x\\y", "x\x00y", ""])
def test_repository_paths_reject_traversal_and_ambiguous_forms(path):
    with pytest.raises(v1.StudyError):
        evidence._relative(path)
