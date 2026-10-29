"""Fabricated CPU checks; no pretrained model or reserved inputs are used."""
import hashlib
import json
import math

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from scripts import bert_taste_matched_dose_v1 as study


def tensor(values):
    return torch.tensor(values, dtype=torch.float32)


def test_common_radius_uses_minimum_and_never_extrapolates():
    source = study.matched_sources(tensor([0, 0]), tensor([3, 4]),
                                   tensor([0, 0]), tensor([0, 2]))
    assert source["radius"] == 2
    assert source["distances"] == {"ingredient": 5, "control": 2}
    assert source["fractions"] == {"ingredient": 0.4, "control": 1}
    for direction in ("sugar_to_lemon", "lemon_to_sugar"):
        for site in ("ingredient", "control"):
            item = source["actions"][direction][site]
            assert math.isclose(item["delivered_norm"], 2, abs_tol=1e-6)
            assert 0 <= item["fraction"] <= 1
            assert item["relative_error"] <= 1e-6


def test_zero_small_and_nonfinite_vectors_fail_without_redraw():
    for vectors in ((tensor([0, 0]), tensor([0, 0]), tensor([1, 0]), tensor([2, 0])),
                    (tensor([0, 0]), tensor([1, 0]), tensor([0, 0]), tensor([1e-6, 0])),
                    (tensor([float("nan"), 0]), tensor([1, 0]), tensor([0, 0]), tensor([1, 0]))):
        with pytest.raises(ValueError):
            study.matched_sources(*vectors)


def test_float32_rounding_mismatch_or_zero_delivery_fails():
    for control_delta in (5.0, 1e-5):
        with pytest.raises(ValueError, match="dose"):
            study.matched_sources(tensor([1e8]), tensor([1e8 + 8]), tensor([0]), tensor([control_delta]))


def test_baseline_eligibility_requires_signs_and_top_five_for_both_rows():
    sugar = {"sweet_minus_sour": 1.0, "top_five_ids": [1, 3, 4, 5, 6]}
    lemon = {"sweet_minus_sour": -0.5, "top_five_ids": [2, 3, 4, 5, 6]}
    assert study.eligible(sugar, lemon, 1, 2)
    assert not study.eligible({**sugar, "sweet_minus_sour": 0.0}, lemon, 1, 2)
    assert not study.eligible(sugar, {**lemon, "top_five_ids": [3, 4, 5, 6, 7]}, 1, 2)


def test_top_five_ties_prefer_smaller_vocabulary_ids():
    assert study.top_five(tensor([1, 3, 3, 2, 2, 2, 2])) == [1, 2, 3, 4, 5]


def test_actual_entry_ids_mask_and_method_cleanup():
    torch.random.default_generator.manual_seed(1701)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32,
                              n_layers=2, dim=16, hidden_dim=32, n_heads=2,
                              dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = DistilBertForMaskedLM(config).eval()
    ids = tuple(range(10))
    observed, entry = study.observe_entry(model, ids, layer=0, position=7, mask_index=4)
    assert entry == {"input_ids": list(ids), "attention_mask": [1] * 10}
    assert observed.before.shape == (16,)
    assert "forward" not in model.__dict__


def test_actual_entry_wrapper_is_restored_on_failure(monkeypatch):
    model = torch.nn.Linear(2, 2)
    def fail(*_args, **_kwargs):
        raise RuntimeError("fabricated observation failure")
    monkeypatch.setattr(study.previous, "observe", fail)
    with pytest.raises(RuntimeError, match="fabricated"):
        study.observe_entry(model, tuple(range(10)), layer=0, position=7, mask_index=4)
    assert "forward" not in model.__dict__


def test_parent_completion_requires_all_six_journal_hashes(tmp_path):
    from scripts import bert_taste_matched_parent_v1 as parent
    source = {"commit": "a" * 40, "selected_source_sha256": {}}
    journals = {}
    for frame in study.FRAMES:
        data = frame.encode()
        (tmp_path / (frame + ".jsonl")).write_bytes(data)
        journals[frame + ".jsonl"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    result = {"status": "complete", "completed_forwards": 516, "source": source, "journals": journals}
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "complete"
    assert parent.assess_result(tmp_path, 1, source, source, True)["status"] == "incomplete"
    (tmp_path / "coffee.jsonl").write_bytes(b"tampered")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"


def test_parent_preserves_fresh_cancellation_on_receipt_failure(tmp_path, monkeypatch):
    from scripts import bert_taste_matched_parent_v1 as parent
    original, cancellation = OSError("child failure"), KeyboardInterrupt("receipt interrupted")
    monkeypatch.setattr(parent, "source_snapshot", lambda: {"commit": "a" * 40})
    def fail_child(*_args, **_kwargs):
        raise original
    def fail_receipt(*_args, **_kwargs):
        raise cancellation
    monkeypatch.setattr(parent.subprocess, "run", fail_child)
    monkeypatch.setattr(parent, "_write_receipt", fail_receipt)
    with pytest.raises(KeyboardInterrupt) as caught:
        parent.main(["--cache-dir", str(tmp_path), "--output", str(tmp_path / "run")])
    assert caught.value is cancellation and caught.value.__cause__ is original


def test_complete_fabricated_schedule_binds_donors_doses_and_self_controls():
    torch.random.default_generator.manual_seed(1777)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32,
                              n_layers=2, dim=16, hidden_dim=32, n_heads=2,
                              dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = DistilBertForMaskedLM(config).eval()
    ids = list(range(10))
    second = ids.copy()
    second[7] = 15
    rows = [{"id": "toy-sugar", "input_ids": ids}, {"id": "toy-lemon", "input_ids": second}]

    class Tokenizer:
        def convert_ids_to_tokens(self, values):
            return [str(value) for value in values]

    events = []
    result = study.run_cases(model, rows, sweet_id=1, sour_id=2, tokenizer=Tokenizer(), emit=events.append)
    assert result["completed_forwards"] == 30
    assert len([e for e in events if e["event"] == "forward_start"]) == 30
    assert len([e for e in events if e["event"] == "forward_complete"]) == 30
    assert len([e for e in events if e["event"] == "dose_plan"]) == 2
    assert len(result["effects"]) == 20
    for event in events:
        if event["event"] == "forward_complete":
            assert event["model_entry"]["input_ids"] == (ids if event["row_id"] == "toy-sugar" else second)
            assert event["model_entry"]["attention_mask"] == [1] * 10
            if event["mode"].endswith("_self"):
                assert event["before_float32_le_base64"] == event["after_float32_le_base64"]
            if event["mode"].endswith("_matched"):
                assert event["actual_pre_post_norm"] == event["dose"]["delivered_norm"]
                assert event["donor_clean_float32_le_base64"] is not None
    assert "forward" not in model.__dict__ and not study.previous.has_hooks(model)


def test_unavailable_dose_records_reason_and_stops_without_retry(monkeypatch):
    torch.random.default_generator.manual_seed(1777)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32,
                              n_layers=2, dim=16, hidden_dim=32, n_heads=2,
                              dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = DistilBertForMaskedLM(config).eval()
    ids = list(range(10))
    second = ids.copy()
    second[7] = 15
    rows = [{"id": "toy-sugar", "input_ids": ids}, {"id": "toy-lemon", "input_ids": second}]

    class Tokenizer:
        def convert_ids_to_tokens(self, values):
            return [str(value) for value in values]

    def fail(*_args):
        raise ValueError("fabricated unconstructible dose")

    monkeypatch.setattr(study, "matched_sources", fail)
    events = []
    with pytest.raises(ValueError, match="unconstructible"):
        study.run_cases(model, rows, sweet_id=1, sour_id=2, tokenizer=Tokenizer(), emit=events.append)
    assert [e["event"] for e in events].count("forward_start") == 4
    assert [e["event"] for e in events].count("forward_complete") == 4
    assert events[-1]["event"] == "dose_unavailable"
    assert events[-1]["reason"] == "fabricated unconstructible dose"
    assert not study.previous.has_hooks(model)
