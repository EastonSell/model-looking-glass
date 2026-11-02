"""Synthetic model verifies frozen rows and exact restoration."""

from types import SimpleNamespace

import torch
import pytest

import flan_weight_edit_guard as guard
import flan_weight_edit_journal as evidence
import flan_weight_edit_session as session_module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Embedding(7, 512)
        self.encoder = torch.nn.Module()
        self.encoder.embed_tokens = self.shared
        self.decoder = torch.nn.Module()
        self.decoder.embed_tokens = self.shared
        self.lm_head = torch.nn.Linear(512, 7, bias=False)
        self.eval()

    def forward(self, input_ids, decoder_input_ids, **kwargs):
        value = self.lm_head(self.shared(decoder_input_ids))
        return SimpleNamespace(logits=value)


def test_primary_is_exact_donor_and_each_arm_restores_state(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    case = {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]}
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal, case)
    before = guard.state_hash(model)
    donor = model.shared.weight[3].detach().clone()
    session.prepare()
    with session.arm("primary", 1.0):
        assert torch.equal(model.shared.weight[2], donor)
    assert guard.state_hash(model) == before
    assert journal.calls == 2
    journal.close()


def test_candidate_handoff_mismatch_is_rejected_before_first_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    with pytest.raises(ValueError, match="candidate receipt"):
        session.prepare(expected_receipt="0" * 64)
    assert journal.calls == 0 and journal.poisoned is True
    journal.close()


def test_last_boundary_rejection_occurs_before_reference_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(TinyModel(), None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    def reject():
        raise ValueError("scope expired")
    with pytest.raises(ValueError, match="scope expired"):
        session.prepare(before_probe=reject)
    assert journal.calls == 0 and journal.poisoned
    journal.close()


def test_failed_arm_restores_without_a_new_probe_and_fences_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    session.prepare()
    original = guard.state_hash(model)
    with pytest.raises(guard.RestorationUnprobed):
        with session.arm("primary", 1.0):
            raise RuntimeError("uncertain forward")
    assert guard.state_hash(model) == original
    assert journal.calls == 1 and journal.poisoned is True
    journal.close()


def test_lost_outer_checked_return_is_fenced_by_session(tmp_path, monkeypatch):
    import sys
    import flan_weight_edit_worker as worker

    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    session.prepare()
    def fake_worker(model, tokenizer, case, journal, arm, *, labels, validate):
        return worker._checked(journal, "lost-outer", lambda:
                               journal.call(case["id"], "fake-logits", lambda: torch.zeros(7),
                                            save_logits=True))
    monkeypatch.setattr(worker, "evaluate_case", fake_worker)
    def fault(frame, event, arg):
        if frame.f_code is worker._checked.__code__ and event == "return" and frame.f_locals.get("stage") == "lost-outer":
            raise RuntimeError("outer lost return")
        return fault
    try:
        sys.settrace(fault)
        with pytest.raises(RuntimeError, match="outer lost return"):
            session.evaluate("baseline", {"id": "fake"})
    finally:
        sys.settrace(None)
    assert journal.calls == 2 and journal.poisoned is True
    with pytest.raises(evidence.JournalFenced):
        journal.call("later", "encoder", lambda: None, save_logits=False)
    journal.close()


def test_mutated_candidate_is_rejected_before_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    session.prepare()
    original = guard.state_hash(model)
    session.rows[("opposite", 0.25)][0] += 1
    with pytest.raises(ValueError, match="candidate inventory"):
        with session.arm("primary", 1.0):
            pytest.fail("mutated inventory must not start an edit")
    assert guard.state_hash(model) == original
    assert journal.calls == 1 and journal.poisoned
    journal.close()


def test_frozen_receipt_and_arm_records_bind_source_and_restore(tmp_path, monkeypatch):
    import hashlib
    import json
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    original = model.shared.weight[2].detach().numpy().tobytes()
    donor = model.shared.weight[3].detach().numpy().tobytes()
    session.prepare()
    primary = next(item for item in session.receipt["entries"] if item["arm"] == "primary")
    assert primary["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert primary["donor_sha256"] == hashlib.sha256(donor).hexdigest()
    assert primary["donor_token"] == 3
    with session.arm("primary", 1.0):
        pass
    journal.close()
    events = [json.loads(line) for line in (tmp_path / "attempt" / "events.jsonl").read_text().splitlines()]
    completed = next(event["metadata"] for event in events if event["kind"] == "arm_completed")
    assert completed["installed_sha256"] == hashlib.sha256(donor).hexdigest()
    assert completed["restored_row_sha256"] == hashlib.sha256(original).hexdigest()
    assert completed["post_restore_state_sha256"] == session.base_state
    assert completed["post_probe_buffer_sha256"] == completed["pre_buffer_sha256"]
    assert completed["probe_span"]["sha256"] == hashlib.sha256(session.reference_probe).hexdigest()


def test_zero_direction_is_scientific_ineligible_without_model_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    with torch.no_grad():
        model.shared.weight[3].copy_(model.shared.weight[2])
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    outcome = session.prepare()
    assert outcome["status"] == "ineligible" and outcome["reason"] == "invalid-direction"
    assert journal.calls == 0
    with pytest.raises(ValueError, match="not prepared"):
        session.evaluate("baseline", {"id": "fake"})
    assert journal.calls == 0
    journal.close()


def test_lost_score_return_fences_session(tmp_path, monkeypatch):
    import sys
    import flan_weight_edit_worker as worker
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(TinyModel(), None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    session.prepare()
    def fault(frame, event, arg):
        if frame.f_code is worker._checked.__code__ and event == "return" and frame.f_locals.get("stage") == "score-fault":
            raise RuntimeError("lost score return")
        return fault
    try:
        sys.settrace(fault)
        with pytest.raises(RuntimeError, match="lost score return"):
            session.score("score-fault", lambda: 1.0)
    finally:
        sys.settrace(None)
    assert journal.poisoned and journal.calls == 1
    with pytest.raises(evidence.JournalFenced):
        journal.call("later", "encoder", lambda: None, save_logits=False)
    journal.close()


def test_equal_byte_alias_replacement_fences_before_edited_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    session.prepare()
    clone = torch.nn.Embedding(7, 512)
    with torch.no_grad():
        clone.weight.copy_(model.shared.weight)
    model.encoder.embed_tokens = clone
    assert guard.state_hash(model) == session.base_state
    with pytest.raises(ValueError, match="topology"):
        with session.arm("primary", 1.0):
            pytest.fail("an altered alias must never enter the edit arm")
    assert journal.poisoned and journal.calls == 1
    journal.close()


def test_submodule_training_mode_change_stops_before_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_IDS", {"wonderful": 2, "awful": 3, "excellent": 4,
                                                       "positive": 5, "negative": 6})
    model = TinyModel()
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=7, max_calls=10,
                               max_bytes=500000, reserve=1024)
    session = session_module.Session(model, None, journal,
                                     {"id": "probe", "input_ids": [5, 1], "attention_mask": [1, 1]})
    session.prepare()
    model.encoder.train()
    with pytest.raises(ValueError, match="training mode"):
        session.evaluate("baseline", {"id": "fake"})
    assert journal.poisoned and journal.calls == 1
    journal.close()
