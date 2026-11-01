"""Lean bundles contain committed regular source blobs, never working-tree payloads."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("app_bundle_builder", Path(__file__).parents[1] / "tools/build_app_bundle.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

ROOT_FILES = ("README.md", "LICENSE", "install.py", "start.py", "pyproject.toml", "PROTOCOL.md",
              "requirements-tested.txt", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "ACCESSIBILITY.md")


def git(repo, *args, data=None):
    return subprocess.run(["git", "-C", str(repo), *args], input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo with spaces"; root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    files = {name: (name + "\n").encode() for name in ROOT_FILES}
    files["pyproject.toml"] = b'[project]\nname = "model-looking-glass"\nversion = "1.2.3"\n'
    files.update({"looking_glass/__init__.py": b"# committed package\n", "looking_glass/README.md": b"map\n",
        "looking_glass/module.py": b"raise RuntimeError('builder must never execute source')\n",
        "looking_glass/app_static/vendor/font-LICENSE.txt": b"license bytes\n",
        "looking_glass/app_static/vendor/font.woff2": b"\x00\x01asset\xff",
        "docs/LOCAL_APP.md": b"guide\n", "docs/README.md": b"docs index\n", "docs/nested/study.json": b"{}\n",
        "RESULTS.md": b"small navigation\n", "archives/README.md": b"archive navigation\n",
        "results/huge.json": b"omitted research", "tests/test_private.py": b"omitted tests",
        "checkpoints/model.safetensors": b"omitted checkpoint", "archives/big.zip": b"omitted archive",
        ".venv/private": b"omitted environment"})
    for name, content in files.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    git(root, "add", "."); git(root, "commit", "-qm", "fixture")
    return root, git(root, "rev-parse", "HEAD").decode().strip(), files


def test_deterministic_committed_identity_allowlist_and_manifest(repo, tmp_path):
    root, commit, files = repo
    (root / "looking_glass/module.py").write_text("private workspace edit")
    (root / "looking_glass/secret.key").write_text("untracked private")
    one, two = tmp_path / "one.zip", tmp_path / "two.zip"
    result = builder.build_bundle(root, commit, one)
    builder.build_bundle(root, commit, two)
    assert one.read_bytes() == two.read_bytes()
    assert result["source_commit"] == commit
    assert result["sha256"] == hashlib.sha256(one.read_bytes()).hexdigest()
    expected = {name for name in files if name in ROOT_FILES or name.startswith(("looking_glass/", "docs/")) or name in ("RESULTS.md", "archives/README.md")}
    with zipfile.ZipFile(one) as archive:
        prefix = "model-looking-glass-app-1.2.3/"
        assert set(archive.namelist()) == {prefix + name for name in expected | {"APP_BUNDLE_MANIFEST.json"}}
        assert archive.namelist() == sorted(archive.namelist())
        manifest = json.loads(archive.read(prefix + "APP_BUNDLE_MANIFEST.json"))
        assert manifest["source_commit"] == commit
        assert manifest["compression"]["method"] == "deflate" and manifest["compression"]["level"] == 9
        assert manifest["compression"]["zlib_runtime"]
        assert {row["path"] for row in manifest["files"]} == expected
        for row in manifest["files"]:
            data = archive.read(prefix + row["path"])
            assert data == files[row["path"]]
            assert row["bytes"] == len(data) and row["sha256"] == hashlib.sha256(data).hexdigest()
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
    assert not list(tmp_path.glob(".app-bundle-*.tmp"))


@pytest.mark.parametrize("revision", ["HEAD", "main", "--help", "abcd", "0" * 40])
def test_requires_existing_full_commit_object(repo, tmp_path, revision):
    root, _, _ = repo
    with pytest.raises((ValueError, RuntimeError)): builder.build_bundle(root, revision, tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_missing_required_file_fails_without_output(repo, tmp_path):
    root, _, _ = repo
    git(root, "rm", "LICENSE"); git(root, "commit", "-qm", "missing")
    with pytest.raises(ValueError, match="required"): builder.build_bundle(root, git(root, "rev-parse", "HEAD").decode().strip(), tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()
    assert not list(tmp_path.glob(".app-bundle-*.tmp"))


def test_symlink_selected_file_rejected_not_dereferenced(repo, tmp_path):
    root, _, _ = repo
    (root / "looking_glass/link").symlink_to("../../private")
    git(root, "add", "looking_glass/link"); git(root, "commit", "-qm", "link")
    with pytest.raises(ValueError, match="regular"): builder.build_bundle(root, git(root, "rev-parse", "HEAD").decode().strip(), tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


@pytest.mark.parametrize("name", ["looking_glass/bad\\name", "looking_glass/CON", "looking_glass/bad:stream", "looking_glass/trailing."])
def test_unsafe_portable_names_rejected(repo, tmp_path, name):
    root, _, _ = repo
    (root / name).write_text("unsafe")
    git(root, "add", "--", name); git(root, "commit", "-qm", "unsafe")
    with pytest.raises(ValueError, match="path"): builder.build_bundle(root, git(root, "rev-parse", "HEAD").decode().strip(), tmp_path / "bad.zip")


def test_existing_output_and_publication_race_never_overwrite(repo, tmp_path, monkeypatch):
    root, commit, _ = repo; output = tmp_path / "exists.zip"; output.write_bytes(b"keep")
    with pytest.raises(FileExistsError): builder.build_bundle(root, commit, output)
    assert output.read_bytes() == b"keep"
    output.unlink()
    original = os.link
    def race(src, dst, **kwargs):
        Path(dst).write_bytes(b"racing writer")
        return original(src, dst, **kwargs)
    monkeypatch.setattr(builder.os, "link", race)
    with pytest.raises(FileExistsError): builder.build_bundle(root, commit, output)
    assert output.read_bytes() == b"racing writer"
    assert not list(tmp_path.glob(".app-bundle-*.tmp"))


def test_failed_zip_write_cleans_temp_and_publishes_nothing(repo, tmp_path, monkeypatch):
    root, commit, _ = repo; output = tmp_path / "bad.zip"
    def fail(*args, **kwargs): raise OSError("injected write failure")
    monkeypatch.setattr(builder.zipfile.ZipFile, "writestr", fail)
    with pytest.raises(OSError): builder.build_bundle(root, commit, output)
    assert not output.exists() and not list(tmp_path.glob(".app-bundle-*.tmp"))


@pytest.mark.parametrize("name", ["/absolute", "looking_glass/../escape", "looking_glass/./file", "looking_glass//file", "looking_glass/bad\x01name", "looking_glass/checkpoints/model.safetensors"])
def test_path_validation_rejects_traversal_controls_and_checkpoint_payload(name):
    with pytest.raises(ValueError, match="path"): builder._safe_path(name)


def test_immutable_object_identity_ignores_replace_refs_and_workspace(repo, tmp_path):
    root, commit, files = repo
    (root / "README.md").write_text("replacement commit")
    git(root, "add", "README.md"); git(root, "commit", "-qm", "later")
    later = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "replace", commit, later)
    output = tmp_path / "original.zip"; builder.build_bundle(root, commit, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("model-looking-glass-app-1.2.3/README.md") == files["README.md"]


def test_no_fetch_git_configuration_and_inherited_repo_override_removed(repo, tmp_path, monkeypatch):
    root, commit, _ = repo
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated-private-repository"))
    original = subprocess.run; calls = []
    def checked(command, **kwargs):
        calls.append(command)
        assert "protocol.allow=never" in command and "--no-replace-objects" in command
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        assert kwargs["env"]["GIT_ALLOW_PROTOCOL"] == ""
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert "GIT_DIR" not in kwargs["env"]
        return original(command, **kwargs)
    monkeypatch.setattr(builder.subprocess, "run", checked)
    builder.build_bundle(root, commit, tmp_path / "offline.zip")
    assert calls and all("fetch" not in command for command in calls)


def test_noncommit_revision_and_selected_gitlink_rejected(repo, tmp_path):
    root, commit, _ = repo
    blob = git(root, "rev-parse", commit + ":README.md").decode().strip()
    with pytest.raises(ValueError, match="commit"): builder.build_bundle(root, blob, tmp_path / "blob.zip")
    git(root, "update-index", "--add", "--cacheinfo", "160000," + commit + ",looking_glass/submodule")
    git(root, "commit", "-qm", "gitlink")
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="regular"): builder.build_bundle(root, revision, tmp_path / "gitlink.zip")


@pytest.mark.parametrize("name", ['looking_glass/bad*name', 'looking_glass/bad?name', 'looking_glass/bad<name', 'looking_glass/bad|name', 'looking_glass/bad"name'])
def test_windows_reserved_characters_rejected(name):
    with pytest.raises(ValueError, match="path"): builder._safe_path(name)


def test_casefolded_file_directory_collision_rejected(repo, tmp_path):
    root, _, _ = repo
    (root / "looking_glass/FOlder").write_text("file")
    (root / "looking_glass/folder").mkdir()
    (root / "looking_glass/folder/module.py").write_text("nested")
    git(root, "add", "looking_glass"); git(root, "commit", "-qm", "collision")
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="Colliding"): builder.build_bundle(root, revision, tmp_path / "collision.zip")
