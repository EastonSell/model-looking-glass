#!/usr/bin/env python3
"""List or restore archived result bytes from a pinned Git commit.

No model code is run. Git may lazily obtain missing objects in a partial clone;
this tool never configures credentials or invokes fetch. Restore into a NEW
directory, then point historical auditors at that directory as documented.
This is an integrity check, not a sandbox against another same-account process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{40}\Z")


def safe_path(value: str) -> str:
    """Require an unambiguous relative POSIX result path on all platforms."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("invalid archive path")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts) or parts[0] != "results":
        raise ValueError(f"unsafe archive path: {value!r}")
    if len(parts) < 2 or any(any(ord(c) < 32 for c in p) for p in parts):
        raise ValueError("invalid archive path components")
    # Windows trims these names or interprets them as devices.
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if any(p.endswith((".", " ")) or p.split(".")[0].upper() in reserved for p in parts):
        raise ValueError("nonportable archive path")
    return value


def load_manifest(path: Path) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or not isinstance(data.get("base_commit"), str):
        raise ValueError("manifest must declare an immutable base_commit")
    if not SHA.fullmatch(data["base_commit"]):
        raise ValueError("base_commit must be a full lowercase SHA-1")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a nonempty list")
    seen = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "blob_sha", "size"}:
            raise ValueError("file entries require exactly path, blob_sha, size")
        name = safe_path(row["path"])
        if name in seen:
            raise ValueError(f"duplicate archive path: {name}")
        seen.add(name)
        if not isinstance(row["blob_sha"], str) or not SHA.fullmatch(row["blob_sha"]):
            raise ValueError("invalid blob SHA-1")
        if type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("invalid blob size")
    for name in seen:
        if any(str(p) in seen for p in PurePosixPath(name).parents):
            raise ValueError("archive file/directory collision")
    for key, expected in (("count", len(files)), ("total_bytes", sum(r["size"] for r in files))):
        if key in data and (type(data[key]) is not int or data[key] != expected):
            raise ValueError(f"incorrect manifest {key}")
    return data


def select_files(data: dict, paths=(), prefixes=(), all_files=False) -> list:
    if not all_files and not paths and not prefixes:
        raise ValueError("select --path, --prefix, or --all explicitly")
    if all_files and (paths or prefixes):
        raise ValueError("--all cannot be combined with selectors")
    names = {r["path"] for r in data["files"]}
    for path in paths:
        safe_path(path)
        if path not in names:
            raise ValueError(f"path absent from archive: {path}")
    clean_prefixes = []
    for prefix in prefixes:
        prefix = prefix.rstrip("/")
        if prefix != "results":
            safe_path(prefix)
        if not any(n == prefix or n.startswith(prefix + "/") for n in names):
            raise ValueError(f"prefix matches no archive files: {prefix}")
        clean_prefixes.append(prefix)
    return [r for r in data["files"] if all_files or r["path"] in paths or
            any(r["path"] == p or r["path"].startswith(p + "/") for p in clean_prefixes)]


def verify_tree(repo: Path, commit: str, rows: list) -> None:
    subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", commit + "^{commit}"],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # No -l: querying sizes can hydrate every blob in a partial clone.
    proc = subprocess.run(["git", "-C", str(repo), "ls-tree", "-rz", commit],
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tree = {}
    for entry in proc.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, kind, sha = metadata.decode("ascii").split()
        tree[path.decode("utf-8")] = (mode, kind, sha)
    for row in rows:
        mode, kind, sha = tree.get(row["path"], (None, None, None))
        if kind != "blob" or mode not in {"100644", "100755"} or sha != row["blob_sha"]:
            raise ValueError(f"manifest does not match pinned tree: {row['path']}")


def check_no_symlinks(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink output path is forbidden: {part}")


def _write_git_blob(repo: Path, sha: str, destination: Path, expected_size: int) -> None:
    with tempfile.TemporaryFile() as errors, destination.open("xb") as output:
        proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "blob", sha],
                                stdout=subprocess.PIPE, stderr=errors)
        try:
            count = 0
            while chunk := proc.stdout.read(1024 * 1024):
                count += len(chunk)
                if count > expected_size:
                    raise ValueError("blob exceeds manifest size")
                output.write(chunk)
            if proc.wait() != 0:
                errors.seek(0)
                raise ValueError("git cat-file failed: " + errors.read(4096).decode(errors="replace"))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()


def verify_blob(row: dict, path: Path) -> None:
    check_no_symlinks(path)
    if not path.is_file():
        raise ValueError(f"missing restored file: {row['path']}")
    digest = hashlib.sha1(f"blob {row['size']}\0".encode("ascii"))
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
    if count != row["size"] or digest.hexdigest() != row["blob_sha"]:
        raise ValueError(f"restored blob failed integrity check: {row['path']}")


def verify_local(rows: list, output: Path) -> None:
    """Verify selected local bytes against the manifest, without invoking Git."""
    if ".." in output.parts:
        raise ValueError("output must not contain parent traversal")
    check_no_symlinks(output)
    if not output.is_dir():
        raise ValueError("verification output must be an existing directory")
    for row in rows:
        verify_blob(row, output.joinpath(*PurePosixPath(safe_path(row["path"])).parts))


def copy_blob(repo: Path, row: dict, destination: Path) -> None:
    # Only expose the final filename after checking the bytes actually on disk.
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=destination.parent) as staging:
        pending = Path(staging) / "blob"
        _write_git_blob(repo, row["blob_sha"], pending, row["size"])
        verify_blob(row, pending)
        # Atomic no-overwrite publication. Same-volume hard-link support is
        # required; failure is preferable to a rename that overwrites a racer.
        os.link(pending, destination)


def restore(data: dict, rows: list, repo: Path, output: Path) -> dict:
    if ".." in output.parts:
        raise ValueError("output must not contain parent traversal")
    output = Path(os.path.abspath(output))
    check_no_symlinks(output)
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    verify_tree(repo, data["base_commit"], rows)
    output.mkdir(mode=0o700)  # Exclusive creation; never merge with an old run.
    status = {"status": "incomplete", "base_commit": data["base_commit"],
              "selected_count": len(rows), "verified_paths": []}
    status_path = output / "archive-restore-status.json"
    def save_status():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    save_status()
    try:
        for row in rows:
            target = output.joinpath(*PurePosixPath(row["path"]).parts)
            check_no_symlinks(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            check_no_symlinks(target)
            copy_blob(repo, row, target)
            status["verified_paths"].append(row["path"])
        status["status"] = "complete"
        save_status()
        return status
    except BaseException as error:
        status["status"] = "incomplete"
        status["error"] = f"{type(error).__name__}: {error}"
        try:
            save_status()
        except BaseException as status_error:
            raise error from status_error
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "restore", "verify"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "archives/results-manifest.json")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        data = load_manifest(args.manifest)
        rows = select_files(data, args.path, args.prefix,
                            args.all or (args.command == "list" and not args.path and not args.prefix))
        if args.command == "list":
            for row in rows:
                print(f"{row['size']}\t{row['blob_sha']}\t{row['path']}")
            print(f"{len(rows)} files, {sum(r['size'] for r in rows)} bytes", file=sys.stderr)
        else:
            if args.output is None:
                raise ValueError("restore/verify requires --output")
            if args.command == "verify":
                verify_local(rows, args.output)
                print(f"Verified {len(rows)} local files against the manifest (no Git/network access)")
            else:
                status = restore(data, rows, args.repo, args.output)
                print(f"Restored and verified {len(status['verified_paths'])} files to {args.output}")
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Archive operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
