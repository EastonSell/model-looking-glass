#!/usr/bin/env python3
"""Build a lean app-source ZIP from an existing immutable local Git commit.

Standard library only. No fetch, checkout, source execution, dependency install,
working-tree payload inclusion, or history rewriting. Repeatable with the same
Python/zlib versions; those compressor versions are recorded in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
import unicodedata
import zipfile
import zlib

ROOT_FILES = (
    "README.md", "LICENSE", "install.py", "start.py", "pyproject.toml", "PROTOCOL.md",
    "requirements-tested.txt", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "ACCESSIBILITY.md",
)
OPTIONAL_FILES = ("RESULTS.md", "archives/README.md")
PREFIXES = ("looking_glass", "docs")
REQUIRED_FILES = ROOT_FILES + ("looking_glass/__init__.py", "looking_glass/README.md", "docs/LOCAL_APP.md", "docs/README.md")
MANIFEST = "APP_BUNDLE_MANIFEST.json"
MAX_FILES = 10_000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
STAMP = (1980, 1, 1, 0, 0, 0)


def _git(repo, *args):
    env = os.environ.copy()
    for key in tuple(env):
        if key in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                   "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_CONFIG_PARAMETERS",
                   "GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_REPLACE_REF_BASE"} or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            env.pop(key, None)
    env.update(GIT_NO_LAZY_FETCH="1", GIT_ALLOW_PROTOCOL="", GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0",
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    command = ["git", "--no-replace-objects", "-c", "protocol.allow=never", "-c", "core.fsmonitor=false",
               "-C", str(repo), *args]
    completed = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError("Local Git object read failed (no network fetch is permitted): " + completed.stderr.decode("utf-8", "replace").strip())
    return completed.stdout


def _safe_path(name):
    if (not name or len(name) > 1024 or name.startswith("/") or "\\" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ValueError(f"Unsafe archive path: {name!r}")
    for part in name.split("/"):
        if (part in ("", ".", "..") or part.endswith((".", " ")) or any(char in part for char in '<>:"|?*')
                or re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)):
            raise ValueError(f"Unsafe archive path: {name!r}")
    if any(part.casefold() in {".git", ".venv", "__pycache__", "results", "tests", "checkpoints"} for part in name.split("/")):
        raise ValueError(f"Excluded payload in selected source path: {name!r}")
    if Path(name).suffix.casefold() in {".pyc", ".pyo", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".bin"}:
        raise ValueError(f"Checkpoint/cache payload in selected source path: {name!r}")


def _inventory(repo, revision):
    if type(revision) is not str or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision):
        raise ValueError("revision must be an explicit full immutable Git commit ID, not a branch or tag")
    revision = revision.lower()
    if _git(repo, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("revision must identify a commit object directly")
    raw = _git(repo, "ls-tree", "--full-tree", "-r", "-l", "-z", revision, "--", *PREFIXES, *ROOT_FILES, *OPTIONAL_FILES)
    records, seen, directories, total = [], set(), set(), 0
    for entry in raw.split(b"\0"):
        if not entry: continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, oid, size = header.split()
        name = raw_name.decode("utf-8", "strict")
        _safe_path(name)
        if not (name in ROOT_FILES + OPTIONAL_FILES or any(name.startswith(prefix + "/") for prefix in PREFIXES)):
            raise ValueError(f"Unexpected selected source path: {name!r}")
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise ValueError(f"Selected source must contain regular blobs only: {name!r}")
        folded = unicodedata.normalize("NFC", name).casefold()
        parents = {"/".join(folded.split("/")[:index]) for index in range(1, len(folded.split("/")))}
        if folded in seen or folded in directories or parents & seen:
            raise ValueError(f"Colliding portable archive path: {name!r}")
        seen.add(folded)
        directories.update(parents)
        byte_count = int(size)
        total += byte_count
        if byte_count > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES or len(records) >= MAX_FILES:
            raise ValueError("Selected source exceeds lean bundle size/file limits")
        records.append({"path": name, "git_blob": oid.decode("ascii"), "mode": mode.decode("ascii"), "bytes": byte_count})
    missing = set(REQUIRED_FILES) - {row["path"] for row in records}
    if missing: raise ValueError("Missing required source files: " + ", ".join(sorted(missing)))
    return revision, sorted(records, key=lambda row: row["path"])


def _zip_entry(archive, name, data, mode="100644"):
    info = zipfile.ZipInfo(name, date_time=STAMP)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | (0o755 if mode == "100755" else 0o644)) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_bundle(repo, revision, output):
    """Publish a complete ZIP atomically without replacing any existing path."""
    repo, output = Path(repo).expanduser().absolute(), Path(output).expanduser().absolute()
    if os.path.lexists(output): raise FileExistsError(f"Output already exists: {output}")
    revision, records = _inventory(repo, revision)
    # Read only selected immutable blobs. Never filters/smudges or workspace files.
    payloads = {}
    for row in records:
        data = _git(repo, "cat-file", "blob", row["git_blob"])
        if len(data) != row["bytes"]: raise ValueError("Git blob byte count changed or is inconsistent")
        row["sha256"] = hashlib.sha256(data).hexdigest()
        payloads[row["path"]] = (data, row["mode"])
    project = tomllib.loads(payloads["pyproject.toml"][0].decode("utf-8")).get("project", {})
    version = project.get("version") if type(project) is dict else None
    if type(version) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", version):
        raise ValueError("Committed pyproject.toml must contain a safe explicit project.version")
    bundle_root = "model-looking-glass-app-" + version
    _safe_path(bundle_root)
    manifest = {
        "schema": "model-looking-glass-app-bundle-v1", "source_commit": revision,
        "project_version": version, "bundle_root": bundle_root,
        "git_object_format": "sha1" if len(revision) == 40 else "sha256",
        "compression": {"method": "deflate", "level": 9, "python": sys.version.split()[0],
                        "zlib_build": zlib.ZLIB_VERSION, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION},
        "manifest_excludes_self": True, "files": records,
    }
    payloads[MANIFEST] = ((json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), "100644")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".app-bundle-", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for name in sorted(payloads):
                    data, mode = payloads[name]
                    _zip_entry(archive, bundle_root + "/" + name, data, mode)
            stream.flush(); os.fsync(stream.fileno())
            byte_count = stream.tell()
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
        result = {"path": str(output), "source_commit": revision, "files": len(records),
                  "project_version": version, "bundle_root": bundle_root,
                  "bytes": byte_count, "sha256": digest.hexdigest()}
        # Same-filesystem hard-link publication is atomic and fails if *any* path
        # already occupies output. Unsupported filesystems fail; never overwrite.
        os.link(temporary, output)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Existing local Git repository")
    parser.add_argument("--revision", required=True, help="Full immutable commit object ID")
    parser.add_argument("--output", required=True, help="New ZIP path; parent directory must exist")
    args = parser.parse_args(argv)
    try:
        result = build_bundle(args.repo, args.revision, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"App bundle not published: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
