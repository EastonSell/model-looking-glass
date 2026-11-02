import copy
import importlib

import pytest
import torch
from torch import nn


def subject():
    try:
        return importlib.import_module("looking_glass.bert_cue_heldout_frames")
    except ModuleNotFoundError as exc:
        if exc.name == "looking_glass.bert_cue_heldout_frames":
            pytest.fail("held-out-frame study module is not implemented")
        raise


class Block(nn.Module):
    def forward(self, hidden):
        return (hidden + hidden.mean(dim=1, keepdim=True) * 0.1,)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(24, 8)
        self.distilbert = nn.Module()
        self.distilbert.transformer = nn.Module()
        self.distilbert.transformer.layer = nn.ModuleList([Block() for _ in range(6)])
        self.head = nn.Linear(8, 7)

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.embedding(input_ids)
        for block in self.distilbert.transformer.layer:
            hidden = block(hidden)[0]
        return type("Output", (), {"logits": self.head(hidden)})()


def case(ident="toy", frame="subject_relative"):
    rows = {}
    for cue, token in (("not", 5), ("hardly", 6), ("really", 7), ("certainly", 8)):
        ids = [1, 2, 3, token, 4, 9, 10, 11, 12, 13, 14]
        rows[cue] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    return {"id": ident, "split": "evaluation", "frame": frame,
            "indices": {"cue": 3, "mask": 5, "object": 8}, "rows": rows}


def effects():
    return {("hardly", "not"): 0.04, ("really", "not"): -0.2, ("certainly", "not"): -0.3,
            ("not", "hardly"): 0.04, ("really", "hardly"): -0.2, ("certainly", "hardly"): -0.3,
            ("not", "really"): 0.2, ("hardly", "really"): 0.3, ("certainly", "really"): -0.04,
            ("not", "certainly"): 0.2, ("hardly", "certainly"): 0.3, ("really", "certainly"): -0.04}


def test_baseline_screen_keeps_a_strict_half_logit_boundary():
    study = subject()
    assert not study.baseline_screen({"not": 1.0, "hardly": 0.5, "really": 0.0, "certainly": 0.0})["eligible"]
    result = study.baseline_screen({"not": 1.0, "hardly": 0.5001, "really": 0.0, "certainly": 0.0})
    assert result["eligible"] and len(result["gaps"]) == 4


def test_each_opposite_donor_must_beat_the_same_recipient_control():
    study = subject()
    values = effects()
    assert study.context_specificity(values)["passed"]
    values[("really", "not")] = -0.05
    values[("hardly", "not")] = 0.0
    decision = study.context_specificity(values)
    assert decision["targets"]["not"]["margin"] == 0.05
    assert not decision["targets"]["not"]["passed"]
    values = effects()
    values[("hardly", "not")] = 0.21
    assert not study.context_specificity(values)["targets"]["not"]["passed"]


def test_frame_eligibility_precedes_the_two_of_three_pass_rule():
    study = subject()
    rows = [{"id": f"{frame}-{i}", "split": "evaluation", "frame": frame,
             "baseline_eligible": True, "dose_eligible": True, "specificity_passed": i < 2}
            for frame in ("subject_relative", "causal_adjunct") for i in range(3)]
    assert study.evaluate(rows)["status"] == "passed"
    failed = copy.deepcopy(rows)
    failed[0]["specificity_passed"] = False
    assert study.evaluate(failed)["status"] == "failed"
    failed[0]["baseline_eligible"] = False
    failed[1]["dose_eligible"] = False
    outcome = study.evaluate(failed)
    assert outcome["status"] == "unevaluable"
    assert outcome["frames"]["subject_relative"]["eligible"] == 1


def test_real_toy_interventions_retain_72_forward_schedule():
    study = subject()
    torch.manual_seed(22)
    model, journal = ToyModel().eval(), []
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    result = study.run_case(model, case(), journal.append, any_id=2, some_id=3,
                            expected_vocab=7, expected_width=8)
    assert not result["baseline_eligible"]
    assert result["dose_eligible"] and result["forwards"] == 72
    assert [event["kind"] for event in journal[:6]] == ["clean"] * 4 + ["baseline", "radius"]
    assert sum(event["kind"] == "self" for event in journal) == 16
    assert sum(event["kind"] == "transfer" for event in journal) == 48
    assert [event["kind"] for event in journal[-4:]] == ["repeat"] * 4


def test_genuine_zero_radius_keeps_all_clean_and_repeat_records():
    study = subject()
    torch.manual_seed(22)
    model, journal = ToyModel().eval(), []
    with torch.no_grad():
        model.embedding.weight[6:9].copy_(model.embedding.weight[5].expand(3, -1))
    result = study.run_case(model, case(), journal.append, any_id=2, some_id=3,
                            expected_vocab=7, expected_width=8)
    assert not result["dose_eligible"] and result["forwards"] == 8
    assert sum(event["kind"] == "dose_ineligible" for event in journal) == 1
    assert sum(event["kind"] == "transfer" for event in journal) == 0
    assert [event["kind"] for event in journal[-4:]] == ["repeat"] * 4


def test_nonfinite_distance_fails_even_when_other_pairs_are_finite():
    study = subject()
    captures = {cue: {site: torch.tensor([float(i), 0.0]) for site in study.SITES}
                for i, cue in enumerate(study.CUES)}
    captures["certainly"]["cue5"] = torch.tensor([float("nan"), 0.0])
    with pytest.raises(ValueError, match="finite"):
        study.choose_radius(captures)


def test_completed_transfer_is_retained_if_its_dose_measurement_fails(monkeypatch):
    study = subject()
    torch.manual_seed(22)
    model, journal = ToyModel().eval(), []
    original_observe, original_distance = study._observed, study.distance
    state = {"transfer_returned": False}

    def observed(*args, **kwargs):
        value = original_observe(*args, **kwargs)
        if "source" in kwargs:
            state["transfer_returned"] = True
        return value

    def distance(*args):
        if state["transfer_returned"]:
            raise RuntimeError("fabricated dose failure")
        return original_distance(*args)

    monkeypatch.setattr(study, "_observed", observed)
    monkeypatch.setattr(study, "distance", distance)
    with pytest.raises(RuntimeError, match="fabricated dose failure"):
        study.run_case(model, case(), journal.append, any_id=2, some_id=3,
                       expected_vocab=7, expected_width=8)
    last = journal[-1]
    assert (last["kind"], last["phase"]) == ("forward_failed", "dose_measurement")
    assert {"mask_logits_f32le", "submitted_f32le", "before_f32le", "after_f32le",
            "full_before_f32le", "full_after_f32le"}.issubset(last)


def test_six_fixed_toy_contexts_are_retained_in_manifest_order():
    study = subject()
    torch.manual_seed(22)
    cases = [case(f"{frame}-{i}", frame) for frame in ("subject_relative", "causal_adjunct") for i in range(3)]
    journal = []
    manifest = {"cases": cases, "scored_tokens": {"any": 2, "some": 3}}
    result = study.run_study(ToyModel().eval(), manifest, journal.append, expected_vocab=7, expected_width=8)
    assert [row["id"] for row in result["cases"]] == [row["id"] for row in cases]
    assert result["forwards"] == 432
    assert sum(event["kind"] == "transfer" for event in journal) == 288
    assert sum(event["kind"] == "clean" for event in journal) == 24
