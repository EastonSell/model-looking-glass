"""Check an operator-attested remote fetch before a targeted-edit phase.

The reviewer must independently fetch the named Git objects from the stated
remote before producing a receipt. These functions check the reported hashes
and local Git ordering; they cannot prove the fetch or the time it happened.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_v2_custody as custody
from . import targeted_edit_v2_evidence as evidence
from . import targeted_edit_v2_manifest as manifest_module


VERSION = "targeted-edit-v2-remote-witness-v1"
REPOSITORY = "EastonSell/model-looking-glass"
SOURCE_PATH = "looking_glass/targeted_edit_v2_witness.py"
SCOPE = "reviewer-reported remote fetch; not proof of chronology or execution"


def _item(repo, commit, path, limit):
    path = evidence._relative(path)
    raw = evidence._blob(repo, commit, path, limit)
    git_sha1 = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    return {"commit": commit, "path": path, "git_blob_sha1": git_sha1, "sha256": v1._sha(raw)}


def _context(repo, *, phase, manifest_path, manifest_commit, manifest_sha256,
             custody_path, custody_commit, custody_sha256, development_dir,
             selection_commit=None, selection_terminal_sha256=None,
             development_witness_path=None, development_witness_commit=None,
             development_witness_sha256=None):
    if phase not in ("development", "evaluation"):
        raise v1.StudyError("invalid witnessed phase")
    root = Path(repo)
    development_dir = evidence._relative(development_dir)
    manifest = manifest_module.verify_manifest(root, manifest_path, manifest_commit=manifest_commit,
                                               expected_sha256=manifest_sha256)
    custody.verify_committed_receipt(root, manifest_path=manifest_path, manifest_commit=manifest_commit,
                                    manifest_sha256=manifest_sha256, receipt_path=custody_path,
                                    receipt_commit=custody_commit, receipt_sha256=custody_sha256)
    source = manifest.get("source_commit")
    hashes = manifest.get("source_sha256")
    if type(hashes) is not dict or SOURCE_PATH not in hashes:
        raise v1.StudyError("the witness source is not bound by the manifest")
    items = []
    for path, expected in sorted(hashes.items()):
        item = _item(root, source, path, evidence.MAX_JSON)
        if not evidence._digest(expected) or item["sha256"] != expected:
            raise v1.StudyError("witness source differs from the manifest")
        items.append(item)
    items.append(_item(root, manifest_commit, manifest_path, evidence.MAX_JSON))
    items.append(_item(root, custody_commit, custody_path, evidence.MAX_JSON))
    if items[-2]["sha256"] != manifest_sha256 or items[-1]["sha256"] != custody_sha256:
        raise v1.StudyError("witness manifest or custody identity differs")
    if phase == "development":
        if any(value is not None for value in (selection_commit, selection_terminal_sha256,
                development_witness_path, development_witness_commit, development_witness_sha256)):
            raise v1.StudyError("development witness cannot contain a selection")
        prior = custody_commit
    else:
        if (not evidence._commit(selection_commit) or not evidence._digest(selection_terminal_sha256) or
                not evidence._commit(development_witness_commit) or
                not evidence._digest(development_witness_sha256)):
            raise v1.StudyError("evaluation witness requires a pinned selection and development witness")
        development_witness_path = evidence._relative(development_witness_path)
        result = evidence.verify_committed_development(
            root, directory=development_dir, manifest_path=manifest_path, manifest_commit=manifest_commit,
            selection_commit=selection_commit, manifest_sha256=manifest_sha256,
            terminal_sha256=selection_terminal_sha256)
        if result.get("status") != "matching_committed_selection":
            raise v1.StudyError("development selection is not complete")
        if (custody_commit == selection_commit or
                evidence._git(root, "merge-base", "--is-ancestor", custody_commit,
                              selection_commit, check=False).returncode):
            raise v1.StudyError("selection must follow custody")
        if (development_witness_commit == selection_commit or
                evidence._git(root, "merge-base", "--is-ancestor", development_witness_commit,
                              selection_commit, check=False).returncode):
            raise v1.StudyError("selection must follow the development witness")
        evidence._utc(result.get("started_utc"))
        verify_committed_witness(root, witness_path=development_witness_path,
            witness_commit=development_witness_commit, witness_sha256=development_witness_sha256,
            next_directory=development_dir, started_utc=result["started_utc"], phase="development",
            manifest_path=manifest_path, manifest_commit=manifest_commit, manifest_sha256=manifest_sha256,
            custody_path=custody_path, custody_commit=custody_commit, custody_sha256=custody_sha256,
            development_dir=development_dir)
        development_witness = _item(root, development_witness_commit, development_witness_path, evidence.MAX_JSON)
        if development_witness["sha256"] != development_witness_sha256:
            raise v1.StudyError("development witness differs from its pin")
        items.append(development_witness)
        terminal_path = f"{development_dir}/terminal.json"
        terminal = _item(root, selection_commit, terminal_path, evidence.MAX_JSON)
        if terminal["sha256"] != selection_terminal_sha256:
            raise v1.StudyError("selection terminal differs from the witness pin")
        terminal_value = v1._decode_document(evidence._blob(root, selection_commit, terminal_path,
                                                           evidence.MAX_JSON))
        items.append(terminal)
        matrix_limit = len(v1.CASES) * v1.VOCAB_SIZE * 4
        for name in sorted(terminal_value["artifacts"]):
            limit = matrix_limit if name.endswith(".f32") else evidence.MAX_JSON
            item = _item(root, selection_commit, f"{development_dir}/{name}", limit)
            if item["sha256"] != terminal_value["artifacts"][name]:
                raise v1.StudyError("selection inventory differs from the terminal")
            items.append(item)
        prior = selection_commit
    items.sort(key=lambda item: (item["commit"], item["path"]))
    if len({(item["commit"], item["path"]) for item in items}) != len(items):
        raise v1.StudyError("witness paths are not unique")
    return manifest, prior, items


def expected_fetch_items(repo, **context):
    """Return locally computed identities for a reviewer to check remotely."""
    _manifest, _prior, items = _context(repo, **context)
    return [dict(item) for item in items]


def make_witness(repo, *, reviewer, observed_utc, fetched_items, next_directory, **context):
    """Build a receipt only after the reviewer has checked every remote object."""
    if type(reviewer) is not str or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", reviewer) is None:
        raise v1.StudyError("invalid reviewer label")
    evidence._utc(observed_utc)
    next_directory = evidence._relative(next_directory)
    manifest, _prior, expected = _context(repo, **context)
    try:
        matches = v1._canonical(fetched_items) == v1._canonical(expected)
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise v1.StudyError("invalid reported fetch items") from exc
    if not matches:
        raise v1.StudyError("reported remote objects differ from the frozen local objects")
    return {"version": VERSION, "repository": REPOSITORY, "phase": context["phase"],
            "source_commit": manifest["source_commit"], "manifest_commit": context["manifest_commit"],
            "manifest_sha256": context["manifest_sha256"], "custody_commit": context["custody_commit"],
            "custody_sha256": context["custody_sha256"], "development_dir": context["development_dir"],
            "selection_commit": context.get("selection_commit"),
            "selection_terminal_sha256": context.get("selection_terminal_sha256"),
            "development_witness_path": context.get("development_witness_path"),
            "development_witness_commit": context.get("development_witness_commit"),
            "development_witness_sha256": context.get("development_witness_sha256"),
            "next_directory": next_directory, "reviewer": reviewer, "observed_utc": observed_utc,
            "items": expected, "scope": SCOPE}


def verify_committed_witness(repo, *, witness_path, witness_commit, witness_sha256,
                             next_directory, started_utc=None, **context):
    """Validate pinned receipt bytes and local order, not the remote fetch itself."""
    root = Path(repo)
    path = evidence._relative(witness_path)
    next_directory = evidence._relative(next_directory)
    if not evidence._digest(witness_sha256):
        raise v1.StudyError("an external witness digest is required")
    manifest, prior, _items = _context(root, **context)
    evidence._require_commit_object(root, witness_commit)
    head = evidence._git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    evidence._require_commit_object(root, head)
    if (prior == witness_commit or
            evidence._git(root, "merge-base", "--is-ancestor", prior, witness_commit, check=False).returncode or
            evidence._git(root, "merge-base", "--is-ancestor", witness_commit, head, check=False).returncode or
            not evidence._path_absent(root, prior, path) or
            not evidence._path_absent(root, witness_commit, next_directory)):
        raise v1.StudyError("witness must precede the next phase and follow its inputs")
    if context["phase"] == "development" and next_directory != context["development_dir"]:
        raise v1.StudyError("development witness must precede its designated directory")
    if context["phase"] == "evaluation":
        if evidence._paths_overlap(context["development_dir"], next_directory):
            raise v1.StudyError("evaluation and development directories overlap")
    local = evidence._read(evidence._ordinary_path(root, path), evidence.MAX_JSON)
    if (v1._sha(local) != witness_sha256 or
            evidence._blob(root, witness_commit, path, evidence.MAX_JSON) != local or
            evidence._blob(root, head, path, evidence.MAX_JSON) != local):
        raise v1.StudyError("committed witness differs from its external pin")
    record = v1._decode_document(local)
    expected = make_witness(root, reviewer=record.get("reviewer"), observed_utc=record.get("observed_utc"),
                            fetched_items=record.get("items"), next_directory=next_directory, **context)
    if v1._canonical(record) != v1._canonical(expected) or record["source_commit"] != manifest["source_commit"]:
        raise v1.StudyError("witness receipt differs from the fixed schema")
    if started_utc is not None and evidence._utc(record["observed_utc"]) > evidence._utc(started_utc):
        raise v1.StudyError("reported witness time follows the recorded phase start")
    return {"status": "matching_committed_remote_fetch_attestation", "witness_commit": witness_commit,
            "witness_sha256": witness_sha256, "observed_utc": record["observed_utc"],
            "reviewer": record["reviewer"], "scope": SCOPE}
