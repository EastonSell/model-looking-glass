import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def owner():
    return module("continuation_owner", "scripts/bert_cue_dose_continuation_v1.py")


def comparator():
    return module("continuation_comparator", "scripts/compare_bert_cue_dose_continuation_v1.py")


def test_fixed_case_order_requires_all_original_ids():
    subject = owner()
    ids = ("exposed-calibration-prior", "exposed-calibration-new", "fresh-01-prior", "fresh-01-new",
           *subject.CASE_IDS)
    manifest = {"cases": [{"id": ident} for ident in ids]}
    assert [case["id"] for case in subject.select_cases(manifest)] == list(subject.CASE_IDS)
    for changed in (ids[:-1], ids + (ids[-1],), ids[:-2] + (ids[-1], ids[-2]), ids[:-1] + (ids[-2],)):
        with pytest.raises(ValueError):
            subject.select_cases({"cases": [{"id": ident} for ident in changed]})


def test_both_disk_reserves_and_event_guard(tmp_path, monkeypatch):
    subject = owner()
    scratch = tmp_path / "scratch"
    checkout = tmp_path / "checkout"
    scratch.mkdir()
    checkout.mkdir()
    output = scratch / "result"
    monkeypatch.setattr(subject, "volume", lambda path: "c" if Path(path) == checkout else "e")
    available = {scratch: 2 * subject.GIB, checkout: subject.GIB}
    monkeypatch.setattr(subject.shutil, "disk_usage", lambda path: SimpleNamespace(free=available[Path(path)]))
    subject.require_space(output, checkout, scratch, initial=True)
    available[scratch] -= 1
    with pytest.raises(RuntimeError, match="output-volume"):
        subject.require_space(output, checkout, scratch, initial=True)
    available[scratch] = subject.GIB
    available[checkout] -= 1
    writes = []
    prior = SimpleNamespace(write_event=lambda *args: writes.append(args))
    with pytest.raises(RuntimeError, match="checkout-volume"):
        subject.emit_with_headroom(prior, None, None, output, checkout, scratch, {})
    assert writes == []
    available[checkout] = subject.GIB
    subject.emit_with_headroom(prior, None, None, output, checkout, scratch, {})
    assert len(writes) == 1
    monkeypatch.setattr(subject, "volume", lambda path: "same")
    with pytest.raises(RuntimeError, match="distinct"):
        subject.require_space(output, checkout, scratch, initial=True)


def test_original_evidence_hash_and_failure_binding(tmp_path, monkeypatch):
    import hashlib
    subject = owner()
    objects = {
        "started": {"source_commit": subject.ORIGINAL_SOURCE, "maximum_forwards": 960,
                    "source_sha256": {str(i): "x" for i in range(17)}},
        "failure": {"source_commit": subject.ORIGINAL_SOURCE, "status": "incomplete", "phase": "forwards",
                    "error_type": "RuntimeError", "message": "disk reserve exhausted before the next evidence event"},
        "receipt": {"source_commit": subject.ORIGINAL_SOURCE, "process_waited": True, "exit_code": 1},
    }
    paths = {}
    for name in ("events", "started", "failure", "receipt"):
        path = tmp_path / name
        path.write_bytes((json.dumps(objects[name]) if name in objects else "synthetic journal").encode())
        paths[name] = path
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    monkeypatch.setattr(subject, "ORIGINAL_HASHES", {name: digest(path) for name, path in paths.items()})
    assert subject.original_records(SimpleNamespace(file_hash=digest), paths)["maximum_forwards"] == 960
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpectedly has a result"):
        subject.original_records(SimpleNamespace(file_hash=digest), paths)
    (tmp_path / "result.json").unlink()
    paths["events"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hashes"):
        subject.original_records(SimpleNamespace(file_hash=digest), paths)


def test_exact_prefix_comparison_rejects_missing_extra_reordered_and_signed_zero():
    subject = comparator()
    events = [{"case_id": subject.CASE, **event, "witness": "same", "dose_value": 0.0}
              for event in subject.expected()]
    encode = lambda event: (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
    original = [encode(event) for event in events]
    assert subject.compare(original, original + [b'{}\n'])["matched"]
    assert not subject.compare(original, original[:-1])["matched"]
    reordered = original.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert not subject.compare(original, reordered)["matched"]
    inserted = original[:5] + [b'{}\n'] + original[5:]
    assert not subject.compare(original, inserted)["matched"]
    changed = [dict(event) for event in events]
    changed[0]["dose_value"] = -0.0
    assert not subject.compare(original, [encode(event) for event in changed])["matched"]


def test_foreign_event_inside_prefix_is_not_filtered_out(tmp_path):
    import gzip
    subject = comparator()
    encode = lambda event: (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
    rows = [encode({"case_id": subject.CASE, **event}) for event in subject.expected()]
    other = encode({"case_id": "foreign-case", "kind": "clean"})
    old = tmp_path / "original.gz"
    new = tmp_path / "continuation.gz"
    with gzip.open(old, "wb") as stream:
        stream.writelines([other, *rows])
    with gzip.open(new, "wb") as stream:
        stream.writelines([rows[0], other, *rows[1:]])
    original = subject.farmer_records(old, original=True)
    continuation = subject.farmer_records(new, original=False)
    assert not subject.compare(original, continuation)["matched"]
    with gzip.open(old, "wb") as stream:
        stream.writelines([other, rows[0], other, *rows[1:]])
    assert not subject.compare(subject.farmer_records(old, original=True), rows)["matched"]
