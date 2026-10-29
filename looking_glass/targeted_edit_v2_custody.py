"""Bind a public, operator-attested custody recheck to the v2 manifest.

No protected rows are loaded or stored here. Git/time records and reports do
not prove the absence of undisclosed private inputs or historical exposure.
"""

from __future__ import annotations

from pathlib import Path

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2
from . import targeted_edit_v2_evidence as evidence
from . import targeted_edit_v2_exposure as exposure
from . import targeted_edit_v2_manifest as manifest_module


VERSION = "targeted-edit-v2-custody-v1"
LANES = ("exposure inventory", "pretrained trace adapter", "pretrained loader", "number-study coordinator")
RESULTS = ("no_protected_inputs_reported", "private_exact_and_prefix_check_passed",
           "private_exact_or_prefix_collision", "unreachable_or_unknown")
CLEAR_RESULTS = frozenset(RESULTS[:2])


def public_rows_sha256():
    rows = [{"case_id": case[0], "token_ids": list(case[3])} for case in v2.EVALUATION_CASES]
    return v1._sha(v1._canonical(rows))


def receipt(manifest_sha256, source_commit, observations, public_inventory):
    """Build a fixed-format report from per-lane, public attestations only."""
    if not evidence._digest(manifest_sha256) or not evidence._commit(source_commit):
        raise v1.StudyError("invalid custody manifest identity")
    if type(observations) is not list or len(observations) != len(LANES):
        raise v1.StudyError("custody observations are incomplete")
    public_keys = {"status", "observed_utc", "commit_label", "payload_sha256",
                   "additional_rows_checked", "collision_count"}
    if type(public_inventory) is not dict or set(public_inventory) != public_keys:
        raise v1.StudyError("latest-known public inventory report is incomplete")
    status = public_inventory["status"]
    evidence._utc(public_inventory["observed_utc"])
    if status not in ("checked_no_collision", "checked_collision", "unavailable_or_unknown"):
        raise v1.StudyError("invalid public inventory status")
    if status != "unavailable_or_unknown":
        if (not evidence._commit(public_inventory["commit_label"]) or
                not evidence._digest(public_inventory["payload_sha256"]) or
                type(public_inventory["additional_rows_checked"]) is not int or
                public_inventory["additional_rows_checked"] < 0 or
                type(public_inventory["collision_count"]) is not int or
                public_inventory["collision_count"] < 0 or
                (status == "checked_no_collision" and public_inventory["collision_count"] != 0) or
                (status == "checked_collision" and public_inventory["collision_count"] == 0)):
            raise v1.StudyError("inconsistent public inventory check")
    elif any(public_inventory[key] is not None for key in public_keys - {"status", "observed_utc"}):
        raise v1.StudyError("an unknown public inventory cannot assert a checked result")
    saved = []
    for lane, row in zip(LANES, observations):
        if (type(row) is not dict or set(row) != {"lane", "observed_utc", "result"} or
                row["lane"] != lane or row["result"] not in RESULTS):
            raise v1.StudyError("invalid or unordered custody observation")
        evidence._utc(row["observed_utc"])
        saved.append({"lane": lane, "observed_utc": row["observed_utc"], "result": row["result"]})
    return {"version": VERSION, "manifest_sha256": manifest_sha256, "source_commit": source_commit,
            "inventory_sha256": exposure.INVENTORY_SHA256, "evaluation_rows_sha256": public_rows_sha256(),
            "public_inventory": dict(public_inventory), "observations": saved,
            "scope": "operator-relayed reports from these lanes only",
            "complete_history_certified": False}


def verify_committed_receipt(repo, *, manifest_path, manifest_commit, manifest_sha256,
                             receipt_path, receipt_commit, receipt_sha256):
    """Verify receipt bytes and local commit order, without claiming truth or freshness."""
    root = Path(repo)
    manifest = manifest_module.verify_manifest(root, manifest_path, manifest_commit=manifest_commit,
                                               expected_sha256=manifest_sha256)
    path = evidence._relative(receipt_path)
    if not evidence._digest(receipt_sha256):
        raise v1.StudyError("an external custody receipt digest is required")
    evidence._require_commit_object(root, receipt_commit)
    head = evidence._git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    evidence._require_commit_object(root, head)
    if (manifest_commit == receipt_commit or
            evidence._git(root, "merge-base", "--is-ancestor", manifest_commit, receipt_commit, check=False).returncode or
            evidence._git(root, "merge-base", "--is-ancestor", receipt_commit, head, check=False).returncode or
            not evidence._path_absent(root, manifest_commit, path)):
        raise v1.StudyError("custody receipt was not committed after the manifest")
    local = evidence._read(evidence._ordinary_path(root, path), evidence.MAX_JSON)
    if (v1._sha(local) != receipt_sha256 or
            evidence._blob(root, receipt_commit, path, evidence.MAX_JSON) != local or
            evidence._blob(root, head, path, evidence.MAX_JSON) != local):
        raise v1.StudyError("committed custody receipt differs from its pin")
    value = v1._decode_document(local)
    observations = value.get("observations")
    expected = receipt(manifest_sha256, manifest["source_commit"], observations, value.get("public_inventory"))
    if v1._canonical(value) != v1._canonical(expected):
        raise v1.StudyError("custody receipt does not match the public fixed schema")
    if (expected["public_inventory"]["status"] != "checked_no_collision" or
            any(row["result"] not in CLEAR_RESULTS for row in expected["observations"])):
        raise v1.StudyError("custody receipt records a collision or unresolved inventory or lane")
    return {"status": "matching_committed_custody_attestation", "receipt_sha256": receipt_sha256,
            "receipt_commit": receipt_commit, "public_inventory": expected["public_inventory"],
            "observations": expected["observations"],
            "complete_history_certified": False}
