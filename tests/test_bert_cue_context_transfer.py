import importlib
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn


def subject():
    return importlib.import_module("looking_glass.bert_cue_context_transfer")


class Block(nn.Module):
    def forward(self, hidden):
        return (hidden + hidden.mean(dim=1, keepdim=True) * 0.1,)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.distilbert = nn.Module()
        self.distilbert.transformer = nn.Module()
        self.distilbert.transformer.layer = nn.ModuleList([Block() for _ in range(6)])
        self.head = nn.Linear(8, 7)

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.embedding(input_ids)
        for block in self.distilbert.transformer.layer:
            hidden = block(hidden)[0]
        return type("Output", (), {"logits": self.head(hidden)})()


def pair(identical_contexts=False):
    study = subject()
    contexts = []
    for ci in range(2):
        row_context = 0 if identical_contexts else ci
        rows = {cue: {"input_ids": [1, 2 + row_context, 3, 5 + i, 4, 9, 10,
                                    11 + row_context, 12, 13 + row_context, 14],
                      "attention_mask": [1] * 11} for i, cue in enumerate(study.CUES)}
        contexts.append({"id": "c" + str(ci), "indices": {"cue": 3, "mask": 5, "object": 8}, "rows": rows})
    return {"id": "discovery-01", "split": "discovery", "contexts": contexts}


def effect_graph(same, control):
    study = subject()
    graph = {}
    for ci in range(2):
        for cue in study.CUES:
            graph["cue2", ci, cue, 1-ci, cue] = {"delta": same}
            graph["cue2", ci, cue, 1-ci, study.same_class_other(cue)] = {"delta": control}
    return graph


def test_half_size_floor_and_baseline_boundaries():
    study = subject()
    tied = study.summarize_pair(effect_graph(-0.005, 0.01), True, True)
    assert tied["informative"] and tied["status"] == "at_least_half" and tied["half_difference"] == 0
    assert study.summarize_pair(effect_graph(0.004, 0.01), True, True)["status"] == "below_half"
    tiny = math.nextafter(0.01, 0.0)
    assert study.summarize_pair(effect_graph(0.02, tiny), True, True)["status"] == "uninformative"
    assert study.summarize_pair(effect_graph(0.02, 0.02), False, True)["status"] == "baseline_ineligible"
    assert study.summarize_pair({}, True, False)["status"] == "dose_ineligible"


def test_common_norm_inclusive_boundary_and_nonfinite_rejection():
    study = subject()
    assert study.choose_q([0.05] * 224) == (0.05, 0.05, True)
    below = math.nextafter(0.05, 0.0)
    assert study.choose_q([below] * 224) == (below, below, False)
    assert study.choose_q([0.7] * 224) == (0.5, 0.7, True)
    for bad in (float("nan"), float("inf"), -1.0, True):
        with pytest.raises(ValueError):
            study.choose_q([0.5] * 223 + [bad])


def test_evaluation_requires_both_pairs_and_uses_all_recipients():
    study = subject()
    outcomes = [study.summarize_pair(effect_graph(0.02, 0.02), True, True),
                study.summarize_pair(effect_graph(0.005, 0.02), True, True),
                study.summarize_pair(effect_graph(0.04, 0.02), True, True)]
    pairs = [{"id": pair_id, "split": "discovery" if i == 0 else "evaluation", "comparison": outcomes[i]}
             for i, pair_id in enumerate(study.PAIR_IDS)]
    result = study.evaluate(pairs)
    assert result["status"] == "at_least_half" and result["informative_pairs"] == 2
    assert result["recipients"] == 16 and result["same_cue_mean"] == pytest.approx(0.0225)
    pairs[1]["comparison"] = study.summarize_pair(effect_graph(0.02, 0.02), False, True)
    result = study.evaluate(pairs)
    assert result["status"] == "unevaluable" and result["informative_pairs"] == 1
    assert result["half_difference"] is None


def test_stable_full_and_excluded_total_variation():
    study = subject()
    clean, patched = [1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1001.0, 1000.0]
    assert study.total_variation(clean, clean) == 0
    full = study.total_variation(clean, patched)
    conditional = study.total_variation(clean, patched, (0, 1))
    assert 0 < full < 1 and 0 < conditional < 1
    assert conditional == pytest.approx((math.e / (math.e + 1)) - 0.5)
    with pytest.raises(ValueError):
        study.stable_probabilities([float("nan"), 0, 1])


def test_log_probability_stays_stable_for_large_equal_logits():
    summary = subject()._baseline(torch.full((7,), 1e30, dtype=torch.float32), 2, 3)
    assert summary["any_log_probability"] == pytest.approx(-math.log(7))
    assert summary["some_log_probability"] == pytest.approx(-math.log(7))
    assert summary["target_pair_probability"] == pytest.approx(2 / 7)


def test_toy_pair_checks_all_matched_transfers_and_full_controls():
    study = subject()
    torch.manual_seed(22)
    model, events = ToyModel().eval(), []
    with torch.no_grad():
        model.embedding.weight.mul_(100)
    outcome = study.run_pair(model, pair(), events.append, any_id=2, some_id=3,
                             expected_vocab=7, expected_width=8)
    assert outcome["dose_eligible"] and outcome["forwards"] == 272
    assert sum(event["kind"] == "clean" for event in events) == 8
    assert sum(event["kind"] == "self" for event in events) == 32
    assert sum(event["kind"] == "transfer" for event in events) == 224
    assert sum(event["kind"] == "repeat" for event in events) == 8
    assert len(outcome["distances"]) == len(outcome["effects"]) == 224
    assert len(outcome["tv_summary"]) == 8
    assert all(0 < event["requested_fraction"] <= 1 and event["actual_norm"] > 0
               for event in events if event["kind"] == "transfer")
    assert all(item["delta"] == item["full_tv"] == item["excluded_tv"] == 0
               for item in outcome["effects"] if item["site"] == "cue5")
    assert all(0 <= item["clean_pair_probability"] <= 1 and 0 <= item["patched_pair_probability"] <= 1
               for item in outcome["effects"])


def test_zero_distance_pair_retains_clean_and_repeats_without_patches():
    study = subject()
    model, events = ToyModel().eval(), []
    outcome = study.run_pair(model, pair(identical_contexts=True), events.append,
                             any_id=2, some_id=3, expected_vocab=7, expected_width=8)
    assert outcome["minimum_distance"] == 0 and not outcome["dose_eligible"]
    assert outcome["forwards"] == 16 and not outcome["effects"]
    assert [event["kind"] for event in events].count("dose_ineligible") == 1
    assert not any(event["kind"] in ("transfer", "self") for event in events)
    assert outcome["comparison"]["status"] == "dose_ineligible"


def test_owner_refuses_oversized_event_before_writing(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/bert_cue_context_transfer_v1.py"
    spec = importlib.util.spec_from_file_location("context_owner_test", path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    monkeypatch.setattr(owner, "require_space", lambda *_args, **_kwargs: None)
    class Sink:
        def __init__(self): self.wrote = False
        def tell(self): return 0
        def write(self, _): self.wrote = True
    raw, zipped = Sink(), Sink()
    with pytest.raises(RuntimeError, match="size bound"):
        owner.write_event(zipped, raw, ".", ".", {"payload": "x" * owner.EVENT_MAX})
    assert not zipped.wrote


@pytest.mark.parametrize("patched", [False, True])
def test_returned_logits_survive_incomplete_plan_exit(patched):
    from looking_glass.bert_cue_class import CompletedObservationFailure, _observed
    from looking_glass.bert_negation_cue import _bits, _has_hooks

    class SkippingModel(ToyModel):
        def forward(self, input_ids, attention_mask, **_kwargs):
            return type("Output", (), {"logits": torch.zeros((1, input_ids.shape[1], 7))})()

    model, events = SkippingModel().eval(), []
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    kwargs = {"site": "cue2", "source": torch.ones(8)} if patched else {}
    with pytest.raises(CompletedObservationFailure) as failure:
        _observed(model, ids, {"cue": 3, "mask": 5, "object": 8}, events.append,
                  {"pair_id": "fake"}, expected_vocab=7, expected_width=8, **kwargs)
    assert failure.value.phase == "plan_exit" and failure.value.__cause__ is not None
    assert len(events) == 1 and events[0]["kind"] == "forward_failed"
    assert events[0]["mask_logits_f32le"] == _bits(torch.zeros(7))
    assert events[0]["omitted_components"] == ["plan_captures_unavailable_or_cleared"]
    assert ("submitted_f32le" in events[0]) is patched
    assert not _has_hooks(model)


def test_unavailable_returned_logits_are_explicitly_omitted_on_plan_exit():
    from looking_glass.bert_cue_class import CompletedObservationFailure, _observed

    class MalformedModel(ToyModel):
        def forward(self, input_ids, attention_mask, **_kwargs):
            return type("Output", (), {"logits": None})()

    events = []
    with pytest.raises(CompletedObservationFailure):
        _observed(MalformedModel().eval(), torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
                  {"cue": 3, "mask": 5, "object": 8}, events.append, {"pair_id": "fake"},
                  expected_vocab=7, expected_width=8)
    assert events[0]["omitted_components"] == ["mask_logits_unavailable_or_unbounded",
                                               "plan_captures_unavailable_or_cleared"]
