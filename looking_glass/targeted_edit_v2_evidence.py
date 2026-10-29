"""Read and verify prospective v2 development-selection artifacts.

This checks saved bytes, selection arithmetic and optional Git object links.
It does not attest that a model ran, that claimed edits were applied, that a
remote commit was published before a forward, or that execution is safe.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2


PHASE = "development"
CONDITIONS = ("original", *(name for name, _ in v2.GRID))
DOCUMENTS = ("started.json", "metadata.json", "selection.json", "observations.jsonl")
CLEANUP_KEYS = ("attempted", "weight_restored", "state_restored", "flags_restored",
                "grads_cleared", "hooks_restored")
MAX_JSON = 1_000_000
MAX_JOURNAL = 1_000_000


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _commit(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _run_id(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value) is not None


def _utc(value):
    if type(value) is not str:
        raise v1.StudyError("invalid recorded UTC time")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise v1.StudyError("invalid recorded UTC time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise v1.StudyError("recorded time must be UTC")
    return parsed


def _regular(path, *, directory=False):
    try:
        info = path.lstat()
    except (OSError, ValueError) as exc:
        raise v1.StudyError("artifact path is unavailable") from exc
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400 or
            (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode))):
        raise v1.StudyError("artifact is not an ordinary file or directory")


def _read(path, limit):
    _regular(path)
    try:
        return v1._bounded_read(path, limit)
    except (OSError, ValueError) as exc:
        raise v1.StudyError("artifact could not be read") from exc


def _identity(record, run_id, manifest_sha, source_commit):
    return (type(record) is dict and record.get("version") == v2.VERSION and
            record.get("phase") == PHASE and record.get("run_id") == run_id and
            record.get("manifest_sha256") == manifest_sha and
            record.get("source_commit") == source_commit)


def _inspect(directory, manifest_sha, terminal_sha):
    if not _digest(manifest_sha) or not _digest(terminal_sha):
        raise v1.StudyError("an externally supplied manifest and terminal digest are required")
    root = Path(directory)
    _regular(root, directory=True)
    try:
        names = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise v1.StudyError("artifact directory could not be listed") from exc
    artifact_names = {*DOCUMENTS, *(f"{name}.f32" for name in CONDITIONS)}
    if names != artifact_names | {"terminal.json"}:
        raise v1.StudyError("development inventory is incomplete or has extra files")
    files = {"terminal.json": _read(root / "terminal.json", MAX_JSON)}
    if v1._sha(files["terminal.json"]) != terminal_sha:
        raise v1.StudyError("terminal digest differs from its external pin")
    terminal = v1._decode_document(files["terminal.json"])
    expected_keys = {"version", "phase", "run_id", "manifest_sha256", "source_commit",
                     "status", "cleanup", "finished_utc", "artifacts"}
    run_id, source_commit = terminal.get("run_id"), terminal.get("source_commit")
    cleanup = terminal.get("cleanup")
    if (set(terminal) != expected_keys or not _run_id(run_id) or not _commit(source_commit) or
            not _identity(terminal, run_id, manifest_sha, source_commit) or
            terminal.get("status") != "complete" or type(cleanup) is not dict or
            set(cleanup) != set(CLEANUP_KEYS) or any(cleanup[key] is not True for key in CLEANUP_KEYS) or
            type(terminal.get("artifacts")) is not dict or set(terminal["artifacts"]) != artifact_names or
            any(not _digest(value) for value in terminal["artifacts"].values())):
        raise v1.StudyError("development terminal is incomplete or inconsistent")
    finished = _utc(terminal["finished_utc"])
    row_bytes = v1.VOCAB_SIZE * 4
    matrix_bytes = len(v1.CASES) * row_bytes
    for name in sorted(artifact_names):
        limit = (matrix_bytes if name.endswith(".f32") else
                 (MAX_JOURNAL if name == "observations.jsonl" else MAX_JSON))
        files[name] = _read(root / name, limit)
        if v1._sha(files[name]) != terminal["artifacts"][name]:
            raise v1.StudyError("development artifact differs from terminal inventory")
        if name.endswith(".f32") and len(files[name]) != matrix_bytes:
            raise v1.StudyError("development matrix has the wrong byte length")
    started = v1._decode_document(files["started.json"])
    selection = v1._decode_document(files["selection.json"])
    metadata = v1._decode_document(files["metadata.json"])
    if (not _identity(started, run_id, manifest_sha, source_commit) or
            set(started) != {"version", "phase", "run_id", "manifest_sha256", "source_commit", "started_utc"} or
            _utc(started["started_utc"]) > finished or
            not _identity(selection, run_id, manifest_sha, source_commit) or
            set(selection) != {"version", "phase", "run_id", "manifest_sha256", "source_commit", "selection"} or
            not _identity(metadata, run_id, manifest_sha, source_commit)):
        raise v1.StudyError("development record identities or times differ")
    lines = files["observations.jsonl"].splitlines(keepends=True)
    if len(lines) != len(CONDITIONS) * len(v1.CASES):
        raise v1.StudyError("development journal has missing or extra rows")
    arrays = {}
    for condition_index, condition in enumerate(CONDITIONS):
        raw = files[f"{condition}.f32"]
        arrays[condition] = np.frombuffer(raw, dtype="<f4").reshape(len(v1.CASES), v1.VOCAB_SIZE)
        for case_index, case in enumerate(v1.CASES):
            offset = case_index * row_bytes
            expected = {"version": v2.VERSION, "phase": PHASE, "run_id": run_id,
                        "condition": condition, "case_id": case[0], "offset": offset,
                        "bytes": row_bytes, "sha256": v1._sha(raw[offset:offset + row_bytes])}
            observed = v1._decode_document(lines[condition_index * len(v1.CASES) + case_index])
            if v1._canonical(observed) != v1._canonical(expected):
                raise v1.StudyError("development journal does not bind ordered output rows")
    derived = v2.verify_development_selection(arrays, selection["selection"])
    return ({"status": "matching_saved_selection", "run_id": run_id,
             "manifest_sha256": manifest_sha, "source_commit": source_commit,
             "terminal_sha256": terminal_sha, "selection": derived, "metadata": metadata}, files)


def audit_development(directory, *, manifest_sha256, terminal_sha256):
    """Check a complete run's saved bytes and selected-dose arithmetic.

    A passing result does not independently verify metadata or Git ancestry and
    must not itself authorize a new evaluation forward.
    """
    result, _files = _inspect(directory, manifest_sha256, terminal_sha256)
    return result


def _relative(value):
    if type(value) is not str or not value or "\\" in value or ":" in value or "\x00" in value:
        raise v1.StudyError("invalid repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise v1.StudyError("invalid repository-relative path")
    return value


def _ordinary_path(root, relative, *, final_directory=False):
    """Reject redirected components below the trusted repository root.

    The caller must keep the checkout quiescent: lstat followed by later reads
    is not an atomic defense against another process replacing a component.
    """
    root = Path(root)
    _regular(root, directory=True)
    parts = PurePosixPath(_relative(relative)).parts
    current = root
    for index, part in enumerate(parts):
        current = current / part
        _regular(current, directory=index < len(parts) - 1 or final_directory)
    return current


def _git(repo, *args, check=True):
    # Replacement refs and legacy grafts can make unrelated commits appear to
    # have a common history. Ignore both, including inherited Git overrides.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env["GIT_GRAFT_FILE"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        process = subprocess.run(["git", "--no-replace-objects", *args], cwd=repo,
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=30, check=check)
    except (OSError, subprocess.SubprocessError) as exc:
        raise v1.StudyError("Git verification failed") from exc
    return process


def _blob(repo, commit, path, limit):
    listing = _git(repo, "ls-tree", "-z", commit, "--", path).stdout.split(b"\x00")
    if len(listing) != 2 or listing[1] != b"":
        raise v1.StudyError("committed path is not a single ordinary file")
    try:
        header, recorded_path = listing[0].split(b"\t", 1)
        mode, kind, _object_id = header.split(b" ")
    except ValueError as exc:
        raise v1.StudyError("invalid committed tree entry") from exc
    if (mode not in (b"100644", b"100755") or kind != b"blob" or
            recorded_path != path.encode("utf-8")):
        raise v1.StudyError("committed path is not a regular blob")
    spec = f"{commit}:{path}"
    try:
        size = int(_git(repo, "cat-file", "-s", spec).stdout)
    except (ValueError, OverflowError) as exc:
        raise v1.StudyError("invalid committed blob size") from exc
    if size < 0 or size > limit:
        raise v1.StudyError("committed blob exceeds its bound")
    result = _git(repo, "cat-file", "blob", spec).stdout
    if len(result) != size:
        raise v1.StudyError("committed blob length changed")
    return result


def _require_commit_object(repo, identity):
    if not _commit(identity) or _git(repo, "cat-file", "-t", identity).stdout.strip() != b"commit":
        raise v1.StudyError("Git identity is not a commit object")


def verify_committed_development(repo, *, directory, manifest_path, manifest_commit,
                                 selection_commit, manifest_sha256, terminal_sha256):
    """Also compare exact saved bytes with Git blobs and check commit ancestry.

    This trusts the local Git executable and repository and does not replace
    frozen source/runtime/checkpoint checks or an independent remote witness.
    """
    if not _commit(manifest_commit) or not _commit(selection_commit) or manifest_commit == selection_commit:
        raise v1.StudyError("invalid selection and manifest commits")
    relative_dir, relative_manifest = _relative(directory), _relative(manifest_path)
    root = Path(repo)
    development_dir = _ordinary_path(root, relative_dir, final_directory=True)
    checked_manifest = _ordinary_path(root, relative_manifest)
    result, files = _inspect(development_dir, manifest_sha256, terminal_sha256)
    source = result["source_commit"]
    head = _git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    for identity in (source, manifest_commit, selection_commit, head):
        _require_commit_object(root, identity)
    if (source == manifest_commit or
            _git(root, "merge-base", "--is-ancestor", source, manifest_commit, check=False).returncode != 0 or
            _git(root, "merge-base", "--is-ancestor", manifest_commit, selection_commit, check=False).returncode != 0 or
            _git(root, "merge-base", "--is-ancestor", selection_commit, head, check=False).returncode != 0):
        raise v1.StudyError("source, manifest and selection ancestry differs")
    if (_git(root, "cat-file", "-e", f"{source}:{relative_manifest}", check=False).returncode == 0 or
            _git(root, "cat-file", "-e", f"{manifest_commit}:{relative_dir}", check=False).returncode == 0):
        raise v1.StudyError("a later artifact was already present at an earlier commit")
    manifest_raw = _blob(root, manifest_commit, relative_manifest, MAX_JSON)
    if (v1._sha(manifest_raw) != manifest_sha256 or
            _blob(root, selection_commit, relative_manifest, MAX_JSON) != manifest_raw or
            _read(checked_manifest, MAX_JSON) != manifest_raw):
        raise v1.StudyError("committed manifest bytes differ")
    manifest = v1._decode_document(manifest_raw)
    if manifest.get("version") != v2.VERSION or manifest.get("source_commit") != source:
        raise v1.StudyError("manifest source identity differs")
    listing = _git(root, "ls-tree", "-z", "--name-only", f"{selection_commit}:{relative_dir}").stdout
    try:
        tracked_names = {name.decode("utf-8") for name in listing.split(b"\x00") if name}
    except UnicodeError as exc:
        raise v1.StudyError("invalid committed inventory") from exc
    if tracked_names != set(files):
        raise v1.StudyError("committed development inventory differs")
    matrix_limit = len(v1.CASES) * v1.VOCAB_SIZE * 4
    for name, data in files.items():
        limit = matrix_limit if name.endswith(".f32") else MAX_JSON
        if _blob(root, selection_commit, f"{relative_dir}/{name}", limit) != data:
            raise v1.StudyError("committed development bytes differ")
    result["status"] = "matching_committed_selection"
    result["selection_commit"] = selection_commit
    return result
