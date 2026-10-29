"""Synthetic Git/source binding tests; no checkpoint/model is loaded."""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_exposure as exposure
from looking_glass import targeted_edit_v2_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PRETRAINED = ROOT / "tests/fixtures/targeted-edit-v1-pretrained/pretrained.py"


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout.decode("ascii").strip()


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "anchor.txt").write_text("prior synthetic evidence", encoding="utf-8")
    git(repo, "add", "anchor.txt")
    git(repo, "commit", "-m", "prior evidence")
    anchor = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(manifest, "V1_EVIDENCE_COMMIT", anchor)
    monkeypatch.setattr(exposure, "INVENTORY_COMMIT", anchor)
    for name in manifest.SOURCE_PATHS:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = HISTORICAL_PRETRAINED if name == "looking_glass/pretrained.py" else ROOT / name
        if source.exists():
            shutil.copyfile(source, target)
        else:
            # The production owner is deliberately unfinished, so this object
            # exercises only manifest mechanics inside an isolated fake repo.
            target.write_text("# synthetic placeholder\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "source")
    source_commit = git(repo, "rev-parse", "HEAD")
    value = manifest.make_manifest(repo, source_commit)
    assert value["version"] == v2.VERSION
    path = repo / "frozen.json"
    path.write_bytes(v1._canonical(value))
    sha = v1._sha(path.read_bytes())
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "freeze")
    return repo, source_commit, git(repo, "rev-parse", "HEAD"), sha


def test_historical_pretrained_fixture_matches_published_v1_manifest():
    raw_manifest = (ROOT / manifest.V1_MANIFEST).read_bytes()
    assert v1._sha(raw_manifest) == manifest.V1_MANIFEST_SHA256
    published = v1._decode_document(raw_manifest)
    assert v1._sha(HISTORICAL_PRETRAINED.read_bytes()) == published["source_sha256"]["looking_glass/pretrained.py"]


def test_fixed_manifest_binds_all_cases_doses_and_source_bytes(frozen_repo):
    repo, source, frozen_commit, sha = frozen_repo
    value = manifest.verify_manifest(repo, "frozen.json", manifest_commit=frozen_commit, expected_sha256=sha)
    assert value["source_commit"] == source
    assert len(value["development_cases"]) == 13
    assert len(value["evaluation_cases"]) == 13
    assert value["random_seed"] == 20261030
    assert value["relative_doses"] == [{"condition": name, "dose": dose} for name, dose in v2.GRID]
    assert value["exposure"]["private_split_check_performed"] is False


def test_listed_project_sources_cover_static_local_imports():
    # This catches direct relative imports only; dynamic imports and the future
    # controller/bootstrap still require an independent source-origin gate.
    paths = set(manifest.SOURCE_PATHS)
    assert len(paths) == len(manifest.SOURCE_PATHS)
    for relative in manifest.SOURCE_PATHS:
        if not relative.startswith("looking_glass/") or not relative.endswith(".py"):
            continue
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            names = ([node.module.split(".", 1)[0]] if node.module else
                     [alias.name.split(".", 1)[0] for alias in node.names])
            for name in names:
                candidate = "looking_glass/" + name + ".py"
                if (ROOT / candidate).is_file():
                    assert candidate in paths, f"{relative} imports unbound {candidate}"


@pytest.mark.parametrize("relative", ["looking_glass/targeted_edit_cpu_v2.py",
                                       "looking_glass/targeted_edit_v2_custody.py",
                                       "looking_glass/targeted_edit_v2_witness.py",
                                       "looking_glass/targeted_edit_v2_evaluation_evidence.py",
                                       "looking_glass/targeted_edit_v2_evaluation_owner.py",
                                       "scripts/targeted_edit_v2_source_rehearsal.py",
                                       "scripts/targeted_edit_v2_source_child.py",
                                       "scripts/v2_source_bootstrap.py"])
def test_working_source_edit_is_rejected(frozen_repo, relative):
    repo, _source, frozen_commit, sha = frozen_repo
    path = repo / relative
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises(v1.StudyError):
        manifest.verify_manifest(repo, "frozen.json", manifest_commit=frozen_commit, expected_sha256=sha)


def test_rehashed_changed_rule_does_not_match_fixed_protocol(frozen_repo):
    repo, _source, _frozen_commit, _sha = frozen_repo
    path = repo / "frozen.json"
    value = v1._decode_document(path.read_bytes())
    value["rules"]["evaluation"]["minimum_target_reversals"] = 1
    path.write_bytes(v1._canonical(value))
    sha = v1._sha(path.read_bytes())
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "forged rules")
    forged_commit = git(repo, "rev-parse", "HEAD")
    with pytest.raises(v1.StudyError):
        manifest.verify_manifest(repo, "frozen.json", manifest_commit=forged_commit, expected_sha256=sha)


def test_mutating_a_returned_manifest_cannot_change_fixed_rules(frozen_repo):
    repo, source, _frozen_commit, _sha = frozen_repo
    returned = manifest.make_manifest(repo, source)
    returned["rules"]["evaluation"]["minimum_target_reversals"] = 1
    returned["v1_compatibility"]["model"]["model_id"] = "changed"
    returned["exposure"]["custodian_observations"]["reports"][0]["report"] = "changed"
    fresh = manifest.make_manifest(repo, source)
    assert fresh["rules"]["evaluation"]["minimum_target_reversals"] == 2
    assert fresh["v1_compatibility"]["model"]["model_id"] != "changed"
    assert fresh["exposure"]["custodian_observations"]["reports"][0]["report"] != "changed"
    path = repo / "frozen.json"
    path.write_bytes(v1._canonical(returned))
    sha = v1._sha(path.read_bytes())
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "forged returned object")
    with pytest.raises(v1.StudyError):
        manifest.verify_manifest(repo, "frozen.json", manifest_commit=git(repo, "rev-parse", "HEAD"),
                                 expected_sha256=sha)


def test_source_commit_cannot_also_be_manifest_commit(frozen_repo):
    repo, source, _frozen_commit, sha = frozen_repo
    with pytest.raises(v1.StudyError):
        manifest.verify_manifest(repo, "frozen.json", manifest_commit=source, expected_sha256=sha)


def test_wrong_external_manifest_pin_is_rejected(frozen_repo):
    repo, _source, frozen_commit, _sha = frozen_repo
    with pytest.raises(v1.StudyError):
        manifest.verify_manifest(repo, "frozen.json", manifest_commit=frozen_commit, expected_sha256="f" * 64)


@pytest.mark.parametrize("historical_source", ["looking_glass/targeted_edit_cpu_v1.py",
                                                "looking_glass/pretrained.py"])
def test_v1_reconstruction_source_cannot_be_changed_before_freeze(frozen_repo, historical_source):
    repo, _source, _frozen_commit, _sha = frozen_repo
    path = repo / historical_source
    path.write_bytes(path.read_bytes() + b"\n# changed after v1\n")
    git(repo, "add", historical_source)
    git(repo, "commit", "-m", "changed historical reconstruction")
    with pytest.raises(v1.StudyError):
        manifest.make_manifest(repo, git(repo, "rev-parse", "HEAD"))
