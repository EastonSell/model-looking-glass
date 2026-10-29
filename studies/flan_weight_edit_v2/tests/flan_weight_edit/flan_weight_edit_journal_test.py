"""Bounded raw-logit journaling without loading a model."""

import hashlib
import json

import numpy as np
import pytest

import flan_weight_edit_journal as evidence


def test_full_logit_rows_are_recoverable_with_exact_offsets_and_hashes(tmp_path):
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=4, max_calls=3, max_bytes=4096, reserve=256)
    logits = np.array([[0, -0.0, 3, -4], [1, 2, 3, 4]], dtype=np.float32)
    returned = journal.call("case-a", "label", lambda: logits, save_logits=True,
                            metadata={"decoder_prefix": [0, 7], "arm": "baseline"})
    assert np.array_equal(returned.view(np.uint32), logits.view(np.uint32))
    journal.close()
    records = [json.loads(line) for line in (tmp_path / "attempt" / "events.jsonl").read_text().splitlines()]
    completed = records[-1]
    raw = (tmp_path / "attempt" / "logits.f32").read_bytes()
    assert completed["offset"] == 0 and completed["bytes"] == len(raw)
    assert completed["sha256"] == hashlib.sha256(raw).hexdigest()
    assert completed["metadata"] == {"decoder_prefix": [0, 7], "arm": "baseline"}
    assert raw == logits.astype("<f4", copy=False).tobytes()
    assert records[0]["kind"] == "forward_started" and completed["kind"] == "forward_complete"


def test_exhausted_call_budget_refuses_to_enter_callback(tmp_path):
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=4, max_calls=1, max_bytes=4096, reserve=256)
    calls = []
    journal.call("case-a", "encoder", lambda: calls.append(1), save_logits=False)
    with pytest.raises(evidence.EvidenceLimit):
        journal.call("case-b", "encoder", lambda: calls.append(2), save_logits=False)
    journal.close()
    assert calls == [1]


def test_nonfinite_logits_leave_started_call_and_no_success_event(tmp_path):
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=4, max_calls=2, max_bytes=4096, reserve=256)
    with pytest.raises(ValueError, match="finite"):
        journal.call("case-a", "generation", lambda: np.array([[0, np.nan, 1, 2]], dtype=np.float32),
                     save_logits=True)
    called = []
    with pytest.raises(evidence.JournalFenced):
        journal.call("case-b", "encoder", lambda: called.append(True), save_logits=False)
    assert called == []
    journal.close()
    records = [json.loads(line) for line in (tmp_path / "attempt" / "events.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in records] == ["forward_started"]
    assert (tmp_path / "attempt" / "logits.f32").read_bytes() == b""


def test_external_attempt_usage_is_included_before_starting_a_call(tmp_path):
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=4, max_calls=2, max_bytes=4096,
                               reserve=256, external_bytes=lambda: 3900)
    called = []
    with pytest.raises(evidence.EvidenceLimit):
        journal.call("case", "encoder", lambda: called.append(True), save_logits=False)
    assert called == [] and journal.poisoned is True
    journal.close()
