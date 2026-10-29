"""Fabricated source/binding checks for the prospective CPU owner."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import looking_glass.flan_multitoken_owner_v2 as owner
from looking_glass.flan_multitoken_owner_v2 import source_snapshot, verify_binding, required_free_bytes


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()


def repository(path):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Synthetic reviewer")
    git(path, "config", "user.email", "synthetic@example.invalid")
    (path / "source.py").write_bytes(b"# synthetic source\n")
    (path / "discovery.json").write_bytes(b'{"fixture":"discovery"}\n')
    (path / "painting.json").write_bytes(b'{"fixture":"painting"}\n')
    discovery = hashlib.sha256((path / "discovery.json").read_bytes()).hexdigest()
    painting = hashlib.sha256((path / "painting.json").read_bytes()).hexdigest()
    review = {"schema": "flan-multisubtoken-selection-review-v1", "status": "consistent",
              "discovery_sha256": discovery, "painting_sha256": painting,
              "scope": "synthetic only"}
    (path / "review.json").write_bytes(canonical(review))
    (path / "reader.py").write_bytes(b"# synthetic independent reader\n")
    git(path, "add", "source.py", "discovery.json", "painting.json", "review.json", "reader.py")
    git(path, "commit", "-qm", "synthetic")
    revision = git(path, "rev-parse", "HEAD")
    def descriptor(name):
        return {"path": name, "revision": revision,
                "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest()}
    binding = {"schema": "flan-multisubtoken-owner-binding-v1", "source_revision": revision,
               "protocol_sha256": "bdb249673dc9045b8e83f4e5f525cc04e530f5f78f7a68fe72d26447e01fdb12",
               "asset_manifest_sha256": "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083",
               "discovery": descriptor("discovery.json"), "painting": descriptor("painting.json"),
               "selection_review": descriptor("review.json"), "reader_sources": [descriptor("reader.py")]}
    return revision, binding


def test_binding_reads_exact_manifest_review_and_reader_git_blobs(tmp_path):
    repo = tmp_path / "repo"
    revision, binding = repository(repo)
    result = verify_binding(repo, revision, canonical(binding),
                            (repo / "discovery.json").read_bytes(),
                            (repo / "painting.json").read_bytes(), (repo / "review.json").read_bytes())
    assert result["source_revision"] == revision
    assert result["discovery_sha256"] == binding["discovery"]["sha256"]
    assert result["reader_sources"] == binding["reader_sources"]


def test_independent_reader_can_be_finalized_after_launch(tmp_path):
    repo = tmp_path / "repo"
    revision, binding = repository(repo)
    binding["reader_sources"] = []
    result = verify_binding(repo, revision, canonical(binding),
                            (repo / "discovery.json").read_bytes(),
                            (repo / "painting.json").read_bytes(), (repo / "review.json").read_bytes())
    assert result["reader_sources"] == []


@pytest.mark.parametrize("changed", ["manifest", "review", "reader", "revision"])
def test_binding_rejects_rehashed_or_misbound_files(tmp_path, changed):
    repo = tmp_path / "repo"
    revision, binding = repository(repo)
    discovery = (repo / "discovery.json").read_bytes()
    painting = (repo / "painting.json").read_bytes()
    review = (repo / "review.json").read_bytes()
    if changed == "manifest":
        discovery = b'{"fixture":"different"}\n'
    elif changed == "review":
        review = canonical({"schema": "flan-multisubtoken-selection-review-v1", "status": "consistent",
                            "discovery_sha256": "0" * 64, "painting_sha256": binding["painting"]["sha256"],
                            "scope": "synthetic"})
    elif changed == "reader":
        binding["reader_sources"][0]["sha256"] = "0" * 64
    else:
        binding["source_revision"] = "0" * 40
    with pytest.raises(ValueError):
        verify_binding(repo, revision, canonical(binding), discovery, painting, review)


def test_source_snapshot_ignores_inherited_git_redirect_and_checks_worktree(tmp_path, monkeypatch):
    repo, other = tmp_path / "repo", tmp_path / "other"
    revision, _ = repository(repo)
    other_revision, _ = repository(other)
    # Ensure different commit identity even if timestamps happen to match.
    (other / "source.py").write_bytes(b"# different\n")
    git(other, "add", "source.py")
    git(other, "commit", "-qm", "different")
    other_revision = git(other, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(repo))
    with pytest.raises(ValueError):
        source_snapshot(repo, other_revision, paths=("source.py",))
    assert source_snapshot(repo, revision, paths=("source.py",))["revision"] == revision
    (repo / "source.py").write_bytes(b"# changed locally\n")
    with pytest.raises(ValueError):
        source_snapshot(repo, revision, paths=("source.py",))


def test_source_snapshot_rejects_a_changed_protocol_even_when_committed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _, _ = repository(repo)
    path = repo / "docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md"
    path.parent.mkdir()
    path.write_bytes(b"a different committed protocol\n")
    git(repo, "add", str(path.relative_to(repo)))
    git(repo, "commit", "-qm", "different protocol")
    revision = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(owner, "PROTOCOL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="protocol"):
        source_snapshot(repo, revision, paths=("docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md",))
    monkeypatch.setattr(owner, "PROTOCOL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert source_snapshot(repo, revision, paths=("docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md",))["revision"] == revision


def test_remaining_pass_capacity_includes_the_future_caps_and_headroom():
    assert required_free_bytes("story") == 9 * 1024**3 // 2
    assert required_free_bytes("concert") == 13 * 1024**3 // 4
    assert required_free_bytes("painting") == 2 * 1024**3
    with pytest.raises(ValueError):
        required_free_bytes("other")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_output_volume_rejects_a_junction_to_the_c_drive(tmp_path):
    junction = tmp_path / "c-drive-junction"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(Path(sys.prefix))],
                   check=True, capture_output=True)
    try:
        assert junction.drive.upper() == "E:"
        assert junction.resolve(strict=True).drive.upper() == "C:"
        with pytest.raises(ValueError, match="output volume"):
            owner.ensure_output_directory(junction)
    finally:
        os.rmdir(junction)


def test_empty_selection_writes_no_model_outcome_before_checkpoint_loading(tmp_path, monkeypatch):
    import torch
    import transformers
    import looking_glass.flan_t5_feasibility as feasibility
    import looking_glass.flan_multitoken_manifest_v2 as manifest

    repo, assets, attempt = tmp_path / "repo", tmp_path / "assets", tmp_path / "attempt"
    repo.mkdir()
    assets.mkdir()
    attempt.mkdir()
    paths = {}
    for name in ("binding", "discovery", "painting", "selection_review"):
        path = tmp_path / (name + ".json")
        path.write_bytes((name + "\n").encode())
        paths[name] = path
    revision = "a" * 40
    args = SimpleNamespace(repo=repo, asset_dir=assets, attempt=attempt, venv=Path(sys.prefix),
                           subject="story", revision=revision,
                           binding_sha256=hashlib.sha256(paths["binding"].read_bytes()).hexdigest(), **paths)
    for key, value in {"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(owner, "source_snapshot", lambda *a, **k: {"revision": revision})
    monkeypatch.setattr(owner, "verify_binding", lambda *a, **k: {"source_revision": revision})
    monkeypatch.setattr(feasibility, "verify_assets", lambda path: {"files": {}})
    monkeypatch.setattr(manifest, "project_manifests", lambda *a, **k: {"story": [], "concert": [], "painting": []})
    for name in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms", "set_float32_matmul_precision"):
        monkeypatch.setattr(torch, name, lambda *a, **k: None)
    monkeypatch.setattr(torch, "set_flush_denormal", lambda value: True)
    fake = type("T5TokenizerFast", (), {"pad_token_id": 0, "eos_token_id": 1, "all_special_ids": [0, 1, 2]})()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: fake)
    monkeypatch.setattr(transformers.T5ForConditionalGeneration, "from_pretrained",
                        lambda *a, **k: pytest.fail("checkpoint must not load"))
    result = owner.run(args)
    assert result["status"] == "no-selected-groups" and result["model_loaded"] is False
    assert not (attempt / "child").exists()
    assert json.loads((attempt / "owner-receipt.json").read_text())["status"] == "no-selected-groups"
