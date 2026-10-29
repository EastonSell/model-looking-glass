"""Public custody-attestation schema checks; no protected rows are accessed."""

import subprocess

import pytest

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_v2_custody as custody


def observations():
    return [{"lane": lane, "observed_utc": "2026-10-29T03:30:00+00:00",
             "result": "no_protected_inputs_reported"} for lane in custody.LANES]


def public_inventory():
    return {"status": "checked_no_collision", "observed_utc": "2026-10-29T03:30:00+00:00",
            "commit_label": "a" * 40, "payload_sha256": "b" * 64,
            "additional_rows_checked": 0, "collision_count": 0}


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout.decode("ascii").strip()


def test_receipt_distinguishes_no_reported_inputs_from_a_private_check():
    rows = observations()
    rows[-1]["result"] = "private_exact_and_prefix_check_passed"
    value = custody.receipt("1" * 64, "2" * 40, rows, public_inventory())
    assert value["observations"][0]["result"] == "no_protected_inputs_reported"
    assert value["observations"][-1]["result"] == "private_exact_and_prefix_check_passed"
    assert value["evaluation_rows_sha256"] == custody.public_rows_sha256()
    assert value["complete_history_certified"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_result", "private_data", "bad_time"])
def test_receipt_rejects_incomplete_or_nonpublic_observations(mutation):
    rows = observations()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["lane"] = rows[0]["lane"]
    elif mutation == "unknown_result":
        rows[0]["result"] = "no_collisions_assumed"
    elif mutation == "private_data":
        rows[0]["private_rows"] = [[1, 2, 3]]
    else:
        rows[0]["observed_utc"] = "not a UTC time"
    with pytest.raises(v1.StudyError):
        custody.receipt("1" * 64, "2" * 40, rows, public_inventory())


def test_public_inventory_counts_are_type_strict_and_consistent():
    check = public_inventory()
    check["additional_rows_checked"] = True
    with pytest.raises(v1.StudyError):
        custody.receipt("1" * 64, "2" * 40, observations(), check)
    check = public_inventory()
    check["status"] = "checked_collision"
    with pytest.raises(v1.StudyError):
        custody.receipt("1" * 64, "2" * 40, observations(), check)


def test_receipt_bytes_and_commit_order_are_bound(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "study@example.invalid")
    git(repo, "config", "user.name", "Synthetic Study")
    (repo / "source.txt").write_text("source", encoding="utf-8")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "source")
    source = git(repo, "rev-parse", "HEAD")
    (repo / "frozen.json").write_text("manifest", encoding="utf-8")
    git(repo, "add", "frozen.json")
    git(repo, "commit", "-m", "manifest")
    manifest_commit = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(custody.manifest_module, "verify_manifest", lambda *a, **kw: {"source_commit": source})
    manifest_sha = "1" * 64
    path = repo / "custody.json"
    value = custody.receipt(manifest_sha, source, observations(), public_inventory())
    path.write_bytes(v1._canonical(value))
    sha = v1._sha(path.read_bytes())
    git(repo, "add", "custody.json")
    git(repo, "commit", "-m", "custody")
    receipt_commit = git(repo, "rev-parse", "HEAD")
    args = {"manifest_path": "frozen.json", "manifest_commit": manifest_commit,
            "manifest_sha256": manifest_sha, "receipt_path": "custody.json",
            "receipt_commit": receipt_commit, "receipt_sha256": sha}
    result = custody.verify_committed_receipt(repo, **args)
    assert result["status"] == "matching_committed_custody_attestation"
    assert result["complete_history_certified"] is False
    bad = dict(args, receipt_commit=manifest_commit)
    with pytest.raises(v1.StudyError):
        custody.verify_committed_receipt(repo, **bad)
    for blocked_status in ("private_exact_or_prefix_collision", "unreachable_or_unknown"):
        blocked = custody.receipt(manifest_sha, source, observations(), public_inventory())
        blocked["observations"][0]["result"] = blocked_status
        path.write_bytes(v1._canonical(blocked))
        blocked_sha = v1._sha(path.read_bytes())
        git(repo, "add", "custody.json")
        git(repo, "commit", "-m", "preserve blocked custody")
        blocked_args = dict(args, receipt_commit=git(repo, "rev-parse", "HEAD"), receipt_sha256=blocked_sha)
        with pytest.raises(v1.StudyError):
            custody.verify_committed_receipt(repo, **blocked_args)
    for status in ("checked_collision", "unavailable_or_unknown"):
        check = public_inventory()
        check["status"] = status
        if status == "checked_collision":
            check["collision_count"] = 1
        else:
            for key in ("commit_label", "payload_sha256", "additional_rows_checked", "collision_count"):
                check[key] = None
        blocked = custody.receipt(manifest_sha, source, observations(), check)
        path.write_bytes(v1._canonical(blocked))
        blocked_sha = v1._sha(path.read_bytes())
        git(repo, "add", "custody.json")
        git(repo, "commit", "-m", "preserve blocked public inventory")
        blocked_args = dict(args, receipt_commit=git(repo, "rev-parse", "HEAD"), receipt_sha256=blocked_sha)
        with pytest.raises(v1.StudyError):
            custody.verify_committed_receipt(repo, **blocked_args)
    value["evaluation_rows_sha256"] = "0" * 64
    path.write_bytes(v1._canonical(value))
    sha = v1._sha(path.read_bytes())
    git(repo, "add", "custody.json")
    git(repo, "commit", "-m", "forged public rows")
    bad = dict(args, receipt_commit=git(repo, "rev-parse", "HEAD"), receipt_sha256=sha)
    with pytest.raises(v1.StudyError):
        custody.verify_committed_receipt(repo, **bad)
