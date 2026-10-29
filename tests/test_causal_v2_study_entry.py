"""Fabricated owner-composition tests; no study cases or forwards are run."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from looking_glass import causal_v2_study_entry as entry


@pytest.fixture
def owner(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source_root = Path(__file__).resolve().parents[1]
    sources = {}
    for relative in sorted(entry._REQUIRED_SOURCE_FILES | {entry._ENTRY}):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ((source_root / relative).read_bytes()
                if relative in entry._EXPECTED else relative.encode("ascii"))
        path.write_bytes(data)
        sources[relative] = hashlib.sha256(data).hexdigest()
    output = tmp_path / "fresh-run"
    finder = Mock(return_value={entry._ENTRY: sources[entry._ENTRY]})
    policy = object()
    backend = {"fixed": "backend"}
    runtime = {"fixed": "runtime"}
    states = [{"seed": seed, "sha256": "a" * 64} for seed in (31201, 31202, 31203, 31204, 31205, 31206)]
    executor = Mock()
    monkeypatch.setattr(entry, "inspect_backend_settings", Mock(return_value=backend))
    monkeypatch.setattr(entry, "inspect_runtime_and_models", Mock(return_value=(policy, runtime, states)))
    factory = Mock(return_value=executor)
    monkeypatch.setattr(entry, "StudyExecutor", factory)
    prepare = Mock(return_value=object())
    monkeypatch.setattr(entry, "prepare_study", prepare)
    calls = []

    def custodian(path, *, source_inventory, **kwargs):
        calls.append((path, source_inventory(), kwargs))
        return {"status": "fabricated"}

    monkeypatch.setattr(entry, "run_study_once", custodian)
    return {"checkout": checkout, "output": output, "sources": sources, "finder": finder,
            "policy": policy, "backend": backend, "runtime": runtime, "states": states,
            "executor": executor, "factory": factory, "prepare": prepare, "calls": calls}


def _run(owner):
    return entry.run_source_bound_study(owner["checkout"], owner["output"], revision="1" * 40,
        source_sha256=owner["sources"], source_inventory=owner["finder"])


def test_rehearsal_initializes_owner_without_preparing_cases(owner):
    report = entry.rehearse_source_bound_owner(owner["checkout"], owner["output"],
        revision="1" * 40, source_sha256=owner["sources"], source_inventory=owner["finder"])
    assert report == {"status": "rehearsal", "model_states_checked": 6,
                      "root_forwards": 0, "study_ready": False}
    owner["prepare"].assert_not_called()
    owner["executor"].close.assert_called_once()
    assert not owner["output"].exists()
    assert not entry._marker_path(owner["output"]).exists()


def test_checked_inputs_bind_actual_scorer_backend_and_source_callback(owner):
    assert _run(owner) == {"status": "fabricated"}
    owner["factory"].assert_called_once_with(owner["policy"], backend_settings=owner["backend"])
    path, inventory, args = owner["calls"][0]
    assert path == owner["output"]
    assert inventory == {entry._ENTRY: owner["sources"][entry._ENTRY]}
    assert args["source_sha256"] == owner["sources"]
    assert args["score"] is entry.score_heldout
    assert args["classify_exposure"]()["classification"] == "unknown"
    assert args["runtime"] == {**owner["runtime"], **owner["backend"]}
    assert args["expected_context"]["model_state_hashes"] == owner["states"]
    assert args["expected_context"]["source_revision"] == "1" * 40
    assert args["clarification_sha256"] == {
        field: owner["sources"][path] for field, path in entry._CLARIFICATION_FILES.items()}
    assert len(args["historical_cases"]) == 24
    assert not owner["output"].exists()
    assert json.loads(entry._marker_path(owner["output"]).read_bytes()) == {
        "schema": "synthetic-v2-attempt-v1", "state": "started",
        "source_revision": "1" * 40,
        "protocol_sha256": owner["sources"]["docs/CAUSAL_CONFIRMATION_V2.md"]}


def test_missing_inventory_hook_stops_before_private_preparation(owner, monkeypatch):
    monkeypatch.setattr(entry, "run_study_once", lambda **kwargs: None)
    with pytest.raises(entry.StudyEntryError, match="inventory recording"):
        _run(owner)
    owner["prepare"].assert_not_called()
    owner["factory"].assert_not_called()
    with pytest.raises(entry.StudyEntryError, match="inventory recording"):
        entry.rehearse_source_bound_owner(owner["checkout"], owner["output"], revision="1" * 40,
            source_sha256=owner["sources"], source_inventory=owner["finder"])


@pytest.mark.parametrize("relative", ["docs/CAUSAL_CONFIRMATION_V2.md", entry._EXCLUSIONS])
def test_changed_source_stops_before_private_preparation(owner, relative):
    (owner["checkout"] / relative).write_bytes(b"changed")
    with pytest.raises(entry.StudyEntryError, match="source bytes"):
        _run(owner)
    owner["prepare"].assert_not_called()


def test_backend_drift_and_unrecognized_loader_fail_before_preparation(owner, monkeypatch):
    monkeypatch.setattr(entry, "inspect_backend_settings", Mock(side_effect=[{"x": 1}, {"x": 2}]))
    with pytest.raises(entry.StudyEntryError, match="backend changed"):
        _run(owner)
    owner["prepare"].assert_not_called()
    owner["finder"].return_value = {entry._ENTRY: "0" * 64}
    with pytest.raises(entry.StudyEntryError, match="source-loader"):
        _run(owner)
    owner["prepare"].assert_not_called()


def test_nonfresh_or_checkout_output_is_rejected(owner):
    owner["output"].mkdir()
    with pytest.raises(entry.StudyEntryError, match="fresh"):
        _run(owner)
    owner["output"] = owner["checkout"] / "new-run"
    with pytest.raises(entry.StudyEntryError, match="outside"):
        _run(owner)
    owner["prepare"].assert_not_called()


def test_preparation_failure_closes_executor_and_preserves_primary(owner):
    owner["prepare"].side_effect = ValueError("preparation failed")
    owner["executor"].close.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(ValueError, match="preparation failed") as error:
        _run(owner)
    assert isinstance(error.value.__cause__, RuntimeError)
    owner["executor"].close.assert_called_once()
    assert owner["calls"] == []
    assert entry._marker_path(owner["output"]).exists()
    with pytest.raises(entry.StudyEntryError, match="attempt marker"):
        _run(owner)


def test_marker_synchronization_failure_prevents_preparation_and_reuse(owner, monkeypatch):
    monkeypatch.setattr(entry.os, "fsync", Mock(side_effect=OSError("cannot synchronize")))
    with pytest.raises(OSError, match="synchronize"):
        _run(owner)
    owner["prepare"].assert_not_called()
    owner["executor"].close.assert_called_once()
    assert entry._marker_path(owner["output"]).exists()
    with pytest.raises(entry.StudyEntryError, match="attempt marker"):
        _run(owner)


@pytest.mark.parametrize("write_error, close_error, expected, cause", [
    (KeyboardInterrupt(), OSError("close failed"), KeyboardInterrupt, OSError),
    (ValueError("write failed"), KeyboardInterrupt(), KeyboardInterrupt, ValueError),
])
def test_marker_cancellation_precedence_retains_partial_attempt(
        owner, monkeypatch, write_error, close_error, expected, cause):
    marker = entry._marker_path(owner["output"])
    original = Path.open
    fake = Mock()
    fake.write.side_effect = write_error
    fake.close.side_effect = close_error

    def open_with_failure(path, mode="r", *args, **kwargs):
        if path == marker and mode == "xb":
            marker.touch(exist_ok=False)
            return fake
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_failure)
    with pytest.raises(expected) as error:
        _run(owner)
    assert isinstance(error.value.__cause__, cause)
    assert marker.exists()
    owner["prepare"].assert_not_called()
    owner["executor"].close.assert_called_once()


def test_early_custodian_failure_closes_executor_and_preserves_cancellation(owner, monkeypatch):
    def fail_before_ownership(path, *, source_inventory, **kwargs):
        raise ValueError("custodian failed before cleanup")
    monkeypatch.setattr(entry, "run_study_once", fail_before_ownership)
    owner["executor"].close.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as error:
        _run(owner)
    assert isinstance(error.value.__cause__, ValueError)
    owner["executor"].close.assert_called_once()
