"""A fake sequence model exercises greedy/EOS and full-logit recording."""

import json
from types import SimpleNamespace

import numpy as np
import torch

import flan_weight_edit_journal as evidence
import flan_weight_edit_worker as worker


class FakeTokenizer:
    all_special_ids = [0, 1]

    def decode(self, ids, **kwargs):
        return {1465: "positive", 2841: "negative"}.get(ids[0], "other") if ids else ""


class FakeModel:
    def __init__(self, tie=False):
        self.tie = tie
        self.encoder = self

    def __call__(self, **kwargs):
        if "input_ids" in kwargs:
            return SimpleNamespace(last_hidden_state=torch.zeros((1, kwargs["input_ids"].shape[1], 3)))
        prefix = kwargs["decoder_input_ids"][0].tolist()
        rows = []
        for i in range(len(prefix)):
            row = np.zeros(32128, dtype=np.float32)
            if i == 0:
                row[1465] = 3.0
                row[2841] = 3.0 if self.tie else 2.0
            else:
                row[1] = 4.0
            rows.append(row)
        return SimpleNamespace(logits=torch.from_numpy(np.array([rows], dtype=np.float32)))


def test_complete_greedy_label_and_teacher_sequences_are_saved(tmp_path):
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=8, max_bytes=2000000, reserve=256)
    case = {"id": "case", "input_ids": [4, 1], "attention_mask": [1, 1],
            "expected": "positive", "expected_ids": [1465, 1]}
    outcome = worker.evaluate_case(FakeModel(), FakeTokenizer(), case, journal, "baseline",
                                   labels={"positive": [1465, 1], "negative": [2841, 1]})
    journal.close()
    assert outcome["greedy_ids"] == [1465, 1]
    assert outcome["strict_correct"] is True
    assert outcome["strict_label"] == "positive"
    assert outcome["label_terms"]["positive"] and outcome["label_terms"]["negative"]
    records = [json.loads(line) for line in (tmp_path / "attempt" / "events.jsonl").read_text().splitlines()]
    assert len([r for r in records if r["kind"] == "forward_complete"]) == 5
    assert all("arm" in r["metadata"] for r in records)
    assert [0, 1465] in [r["metadata"].get("decoder_prefix") for r in records]


def test_tied_greedy_top_is_visible_and_ineligible(tmp_path):
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=8, max_bytes=2000000, reserve=256)
    case = {"id": "case", "input_ids": [4, 1], "attention_mask": [1, 1],
            "expected": "positive", "expected_ids": [1465, 1]}
    outcome = worker.evaluate_case(FakeModel(tie=True), FakeTokenizer(), case, journal, "baseline",
                                   labels={"positive": [1465, 1], "negative": [2841, 1]})
    journal.close()
    assert outcome["greedy_ids"] == [1465, 1]
    assert outcome["tops"][0]["tied_ids"] == [1465, 2841]
    assert outcome["strict_correct"] is False
    assert outcome["strict_label"] is None


def test_wrong_label_ids_are_rejected_before_any_encoder_call(tmp_path):
    import pytest

    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=8, max_bytes=2000000, reserve=256)
    case = {"id": "case", "input_ids": [4, 1], "attention_mask": [1, 1],
            "expected": "positive", "expected_ids": [1465, 1]}
    with pytest.raises(ValueError, match="fixed labels"):
        worker.evaluate_case(FakeModel(), FakeTokenizer(), case, journal, "baseline",
                             labels={"positive": [2, 1], "negative": [2841, 1]})
    assert journal.calls == 0
    journal.close()


def test_numeric_mode_drift_fences_the_journal_before_more_calls(tmp_path, monkeypatch):
    import pytest

    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=8, max_bytes=2000000, reserve=256)
    case = {"id": "case", "input_ids": [4, 1], "attention_mask": [1, 1],
            "expected": "positive", "expected_ids": [1465, 1]}
    probes = []
    def probe():
        probes.append(True)
        if len(probes) == 2:
            raise ValueError("numeric mode drift")
        return {"subnormal_bits": 1}
    monkeypatch.setattr(worker, "numeric_probe", probe)
    with pytest.raises(ValueError, match="numeric mode drift"):
        worker.evaluate_case(FakeModel(), FakeTokenizer(), case, journal, "baseline",
                             labels={"positive": [1465, 1], "negative": [2841, 1]})
    assert journal.calls == 1
    with pytest.raises(evidence.JournalFenced):
        journal.call("other", "encoder", lambda: None, save_logits=False)
    journal.close()


def test_lost_return_after_durable_logits_fences_later_calls(tmp_path):
    import sys
    import pytest

    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=8, max_bytes=2000000, reserve=256)
    case = {"id": "case", "input_ids": [4, 1], "attention_mask": [1, 1],
            "expected": "positive", "expected_ids": [1465, 1]}
    def fault(frame, event, arg):
        if frame.f_code is evidence.Journal.call.__code__ and event == "return" and frame.f_locals.get("save_logits"):
            raise RuntimeError("lost return")
        return fault
    try:
        sys.settrace(fault)
        with pytest.raises(RuntimeError, match="lost return"):
            worker.evaluate_case(FakeModel(), FakeTokenizer(), case, journal, "baseline",
                                 labels={"positive": [1465, 1], "negative": [2841, 1]})
    finally:
        sys.settrace(None)
    assert journal.calls == 2
    with pytest.raises(evidence.JournalFenced):
        journal.call("other", "encoder", lambda: None, save_logits=False)
    journal.close()
    records = [json.loads(line) for line in (tmp_path / "attempt" / "events.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "forward_complete" and r["stage"] == "greedy-0" for r in records)


def test_nonfinite_earlier_decoder_row_fences_before_any_more_calls(tmp_path):
    import pytest
    class NonfinitePrefix(FakeModel):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if "decoder_input_ids" in kwargs and kwargs["decoder_input_ids"].shape[1] > 1:
                result.logits[0, 0, 7] = float("nan")
            return result
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=8, max_bytes=2000000, reserve=256)
    case = {"id": "case", "input_ids": [4, 1], "attention_mask": [1, 1],
            "expected": "positive", "expected_ids": [1465, 1]}
    with pytest.raises(ValueError, match="nonfinite or invalid"):
        worker.evaluate_case(NonfinitePrefix(), FakeTokenizer(), case, journal, "baseline",
                             labels={"positive": [1465, 1], "negative": [2841, 1]})
    assert journal.calls == 3 and journal.poisoned
    with pytest.raises(evidence.JournalFenced):
        journal.call("later", "encoder", lambda: None, save_logits=False)
    journal.close()
