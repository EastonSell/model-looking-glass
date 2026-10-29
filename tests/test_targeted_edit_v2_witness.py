"""Synthetic local-Git checks for reported remote-fetch attestations."""

import copy
import subprocess
from pathlib import Path

import pytest

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_v2_witness as witness


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout.decode("ascii").strip()


@pytest.fixture
def history(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "review@example.invalid")
    git(repo, "config", "user.name", "Synthetic Witness")
    source_path = repo / witness.SOURCE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"synthetic witness source\n")
    (repo / "source.txt").write_bytes(b"synthetic study source\n")
    git(repo, "add", "looking_glass", "source.txt")
    git(repo, "commit", "-m", "source")
    source = git(repo, "rev-parse", "HEAD")
    hashes = {path: v1._sha((repo / path).read_bytes()) for path in (witness.SOURCE_PATH, "source.txt")}
    manifest = {"source_commit": source, "source_sha256": hashes}
    (repo / "manifest.json").write_bytes(v1._canonical(manifest))
    manifest_sha = v1._sha((repo / "manifest.json").read_bytes())
    git(repo, "add", "manifest.json")
    git(repo, "commit", "-m", "manifest")
    manifest_commit = git(repo, "rev-parse", "HEAD")
    (repo / "custody.json").write_bytes(v1._canonical({"test": "receipt"}))
    custody_sha = v1._sha((repo / "custody.json").read_bytes())
    git(repo, "add", "custody.json")
    git(repo, "commit", "-m", "custody")
    custody_commit = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(witness.manifest_module, "verify_manifest", lambda *a, **kw: manifest)
    monkeypatch.setattr(witness.custody, "verify_committed_receipt", lambda *a, **kw: {})
    context = {"phase": "development", "manifest_path": "manifest.json", "manifest_commit": manifest_commit,
               "manifest_sha256": manifest_sha, "custody_path": "custody.json", "custody_commit": custody_commit,
               "custody_sha256": custody_sha, "development_dir": "runs/development"}
    return repo, context


def write_witness(repo, context, name="witness.json", observed="2026-10-29T05:00:00+00:00"):
    items = witness.expected_fetch_items(repo, **context)
    record = witness.make_witness(repo, reviewer="independent-review", observed_utc=observed,
                                  fetched_items=items, next_directory=("runs/development" if context["phase"] == "development"
                                                                       else "runs/evaluation"), **context)
    path = repo / name
    path.write_bytes(v1._canonical(record))
    pin = v1._sha(path.read_bytes())
    git(repo, "add", name)
    git(repo, "commit", "-m", "witness")
    return git(repo, "rev-parse", "HEAD"), pin


def test_development_witness_binds_all_source_and_receipt_blobs(history):
    repo, context = history
    items = witness.expected_fetch_items(repo, **context)
    assert {(item["commit"], item["path"]) for item in items} == {
        (git(repo, "rev-parse", context["manifest_commit"] + "^"), witness.SOURCE_PATH),
        (git(repo, "rev-parse", context["manifest_commit"] + "^"), "source.txt"),
        (context["manifest_commit"], "manifest.json"), (context["custody_commit"], "custody.json")}
    witness_commit, pin = write_witness(repo, context)
    result = witness.verify_committed_witness(repo, witness_path="witness.json", witness_commit=witness_commit,
                 witness_sha256=pin, next_directory="runs/development", started_utc="2026-10-29T05:01:00+00:00",
                 **context)
    assert result["status"] == "matching_committed_remote_fetch_attestation"
    assert "not proof" in result["scope"]


@pytest.mark.parametrize("field", ["commit", "path", "git_blob_sha1", "sha256"])
def test_reported_fetch_must_match_each_locally_verified_object(history, field):
    repo, context = history
    items = witness.expected_fetch_items(repo, **context)
    items[0][field] = "0" * (40 if field in ("commit", "git_blob_sha1") else 64)
    with pytest.raises(v1.StudyError, match="remote objects differ"):
        witness.make_witness(repo, reviewer="reviewer", observed_utc="2026-10-29T05:00:00+00:00",
                             fetched_items=items, next_directory="runs/development", **context)


def test_rehashed_receipt_with_wrong_repository_or_late_clock_is_rejected(history):
    repo, context = history
    commit, pin = write_witness(repo, context)
    with pytest.raises(v1.StudyError, match="follows"):
        witness.verify_committed_witness(repo, witness_path="witness.json", witness_commit=commit,
            witness_sha256=pin, next_directory="runs/development", started_utc="2026-10-29T04:59:59+00:00",
            **context)
    record = v1._decode_document((repo / "witness.json").read_bytes())
    record["repository"] = "another/repository"
    (repo / "other.json").write_bytes(v1._canonical(record))
    other_pin = v1._sha((repo / "other.json").read_bytes())
    git(repo, "add", "other.json")
    git(repo, "commit", "-m", "forged witness")
    forged = git(repo, "rev-parse", "HEAD")
    with pytest.raises(v1.StudyError, match="fixed schema"):
        witness.verify_committed_witness(repo, witness_path="other.json", witness_commit=forged,
            witness_sha256=other_pin, next_directory="runs/development", **context)


def test_phase_directory_cannot_preexist_at_witness_commit(history):
    repo, context = history
    early = repo / "runs" / "development"
    early.mkdir(parents=True)
    (early / "partial.f32").write_bytes(b"already present")
    git(repo, "add", "runs")
    git(repo, "commit", "-m", "early development")
    commit, pin = write_witness(repo, context)
    with pytest.raises(v1.StudyError, match="precede"):
        witness.verify_committed_witness(repo, witness_path="witness.json", witness_commit=commit,
            witness_sha256=pin, next_directory="runs/development", **context)


def test_evaluation_witness_includes_terminal_and_every_inventoried_blob(history, monkeypatch):
    repo, context = history
    dev_witness_commit, dev_witness_sha = write_witness(repo, context, name="dev-witness.json")
    directory = repo / "runs" / "development"
    directory.mkdir(parents=True)
    (directory / "observations.jsonl").write_bytes(b"synthetic journal\n")
    (directory / "original.f32").write_bytes(b"synthetic raw bytes")
    artifacts = {name: v1._sha((directory / name).read_bytes())
                 for name in ("observations.jsonl", "original.f32")}
    (directory / "terminal.json").write_bytes(v1._canonical({"artifacts": artifacts}))
    terminal_sha = v1._sha((directory / "terminal.json").read_bytes())
    git(repo, "add", "runs")
    git(repo, "commit", "-m", "selection")
    selection = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(witness.evidence, "verify_committed_development",
                        lambda *a, **kw: {"status": "matching_committed_selection",
                                         "started_utc": "2026-10-29T05:01:00+00:00"})
    context = {**context, "phase": "evaluation", "selection_commit": selection,
               "selection_terminal_sha256": terminal_sha, "development_witness_path": "dev-witness.json",
               "development_witness_commit": dev_witness_commit, "development_witness_sha256": dev_witness_sha}
    items = witness.expected_fetch_items(repo, **context)
    selected_paths = {item["path"] for item in items if item["commit"] == selection}
    assert selected_paths == {"runs/development/terminal.json", "runs/development/observations.jsonl",
                              "runs/development/original.f32"}
    commit, pin = write_witness(repo, context, name="eval-witness.json")
    result = witness.verify_committed_witness(repo, witness_path="eval-witness.json", witness_commit=commit,
                 witness_sha256=pin, next_directory="runs/evaluation", **context)
    assert result["status"] == "matching_committed_remote_fetch_attestation"
    assert any(item["path"] == "dev-witness.json" and item["commit"] == dev_witness_commit for item in items)
    with pytest.raises(v1.StudyError, match="fixed schema"):
        witness.verify_committed_witness(repo, witness_path="eval-witness.json", witness_commit=commit,
                 witness_sha256=pin, next_directory="runs/other-evaluation", **context)
    with pytest.raises(v1.StudyError, match="overlap|precede"):
        witness.verify_committed_witness(repo, witness_path="eval-witness.json", witness_commit=commit,
                 witness_sha256=pin, next_directory="runs/development/evaluation", **context)
    with pytest.raises(v1.StudyError, match="overlap"):
        witness.verify_committed_witness(repo, witness_path="eval-witness.json", witness_commit=commit,
                 witness_sha256=pin, next_directory="RUNS/DEVELOPMENT/evaluation", **context)
    for changed in ({"development_witness_commit": None}, {"development_witness_commit": selection}):
        with pytest.raises(v1.StudyError):
            witness.expected_fetch_items(repo, **{**context, **changed})
    tree = git(repo, "show", "-s", "--format=%T", dev_witness_commit)
    unrelated = git(repo, "commit-tree", tree, "-m", "unrelated witness history")
    with pytest.raises(v1.StudyError, match="selection must follow"):
        witness.expected_fetch_items(repo, **{**context, "development_witness_commit": unrelated})
