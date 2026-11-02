import copy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def subject():
    return importlib.import_module("looking_glass.bert_cue_dose_response")


class Block(nn.Module):
    def forward(self, hidden):
        return (hidden + hidden.mean(dim=1, keepdim=True) * 0.1,)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.distilbert = nn.Module()
        self.distilbert.transformer = nn.Module()
        self.distilbert.transformer.layer = nn.ModuleList([Block() for _ in range(6)])
        self.head = nn.Linear(8, 7)

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.embedding(input_ids)
        for block in self.distilbert.transformer.layer:
            hidden = block(hidden)[0]
        return type("Output", (), {"logits": self.head(hidden)})()


def case(group="new", triple_id="fresh-01", split="evaluation"):
    groups = {"prior": ("never", "rarely", "always", "often"),
              "new": ("not", "hardly", "really", "certainly")}
    rows = {}
    for token, cue in enumerate(groups[group], 5):
        ids = [1, 2, 3, token, 4, 9, 10, 11, 12, 13, 14]
        rows[cue] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    return {"id": triple_id + "-" + group, "split": split, "triple_id": triple_id,
            "frame": "main", "cue_set": group,
            "indices": {"cue": 3, "mask": 5, "object": 8}, "rows": rows}


def synthetic_rows():
    rows = []
    for triple, split in (("exposed-calibration", "calibration"), ("fresh-01", "evaluation"),
                          ("fresh-02", "evaluation"), ("fresh-03", "evaluation")):
        for group in ("prior", "new"):
            low, high = ((0.02, 0.10) if triple.endswith("01") else
                         (0.08, 0.02) if triple.endswith("02") else (0.01, 0.03))
            rows.append({"id": triple + "-" + group, "triple_id": triple, "split": split,
                         "cue_set": group, "frame": "main", "baseline_eligible": True,
                         "doses": {label: {"eligible": True, "cue_min_margin": score,
                                           "specificity_passed": score > 0.05}
                                   for label, score in (("low", low), ("high", high))}})
    return rows


def test_fixed_dose_boundary_and_nonfinite_distances():
    study = subject()
    assert study.dose_eligibility(0.7) == {"low": True, "high": True}
    assert study.dose_eligibility(0.5) == {"low": True, "high": False}
    assert study.dose_eligibility(0.499) == {"low": False, "high": False}
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            study.dose_eligibility(value)


def test_paired_transitions_exclude_exposed_calibration():
    result = subject().evaluate(synthetic_rows())
    for group in ("prior", "new"):
        assert result["cells"][group]["low"] == {"eligible": 3, "passing": 1, "total": 3, "status": "failed"}
        assert result["cells"][group]["high"] == {"eligible": 3, "passing": 1, "total": 3, "status": "failed"}
        pair = result["paired"][group]
        assert pair["eligible"] == 3 and pair["status"] == "changed"
        assert pair["transitions"] == {"fail_to_pass": 1, "pass_to_fail": 1,
                                       "pass_to_pass": 0, "fail_to_fail": 1}
        assert pair["mean"] == pytest.approx((0.08 - 0.06 + 0.02) / 3)


def test_ineligible_cells_not_negative_and_pairs_not_imputed():
    rows = synthetic_rows()
    next(r for r in rows if r["id"] == "fresh-01-prior")["baseline_eligible"] = False
    high = next(r for r in rows if r["id"] == "fresh-02-prior")["doses"]["high"]
    high.update(eligible=False, cue_min_margin=None, specificity_passed=None)
    result = subject().evaluate(rows)
    assert result["cells"]["prior"]["low"]["eligible"] == 2
    assert result["cells"]["prior"]["high"]["eligible"] == 1
    assert result["cells"]["prior"]["high"]["status"] == "unevaluable"
    assert result["paired"]["prior"]["eligible"] == 1
    assert result["paired"]["prior"]["status"] == "unevaluable"
    assert result["paired"]["prior"]["mean"] is None
    invalid = copy.deepcopy(rows)
    invalid[-1]["triple_id"] = invalid[-3]["triple_id"]
    with pytest.raises(ValueError):
        subject().evaluate(invalid)


@pytest.mark.parametrize("group", ["prior", "new"])
def test_both_doses_run_even_when_baseline_fails(group):
    study = subject()
    torch.manual_seed(22)
    model, events = ToyModel().eval(), []
    with torch.no_grad():
        model.embedding.weight.mul_(100)
        model.head.weight.zero_()
        model.head.bias.zero_()
    result = study.run_case(model, case(group), events.append, any_id=2, some_id=3,
                            expected_vocab=7, expected_width=8)
    assert not result["baseline_eligible"] and result["forwards"] == 120
    for label, radius in (("low", 0.5), ("high", 0.7)):
        assert result["doses"][label]["eligible"]
        transfers = [e for e in events if e["kind"] == "transfer" and e["dose"] == label]
        assert len(transfers) == 48 and all(e["requested_radius"] == radius for e in transfers)
        assert all(e["requested_fraction"] <= 1.0 for e in transfers)
    assert sum(e["kind"] == "self" for e in events) == 16
    assert [e["kind"] for e in events[-4:]] == ["repeat"] * 4


def test_genuine_low_only_and_zero_distance_paths():
    study = subject()
    torch.manual_seed(22)
    model, item, events = ToyModel().eval(), case(), []
    with torch.no_grad():
        model.embedding.weight.mul_(100)
    captures = {}
    for cue, row in item["rows"].items():
        observed = study._observed(model, torch.tensor([row["input_ids"]]), item["indices"], lambda _: None,
                                   {"case_id": "toy", "cue": cue}, expected_vocab=7, expected_width=8)
        captures[cue] = observed["captures"]
    minimum, _ = study.choose_distances(captures, "new")
    with torch.no_grad():
        model.embedding.weight.mul_(0.6 / minimum)
    low_only = study.run_case(model, item, events.append, any_id=2, some_id=3, expected_vocab=7, expected_width=8)
    assert 0.5 <= low_only["minimum_distance"] < 0.7
    assert low_only["doses"]["low"]["eligible"] and not low_only["doses"]["high"]["eligible"]
    assert low_only["forwards"] == 72
    assert sum(e["kind"] == "transfer" and e["dose"] == "low" for e in events) == 48
    assert not any(e["kind"] == "transfer" and e["dose"] == "high" for e in events)
    with torch.no_grad():
        model.embedding.weight[6:9].copy_(model.embedding.weight[5].expand(3, -1))
    events = []
    neither = study.run_case(model, item, events.append, any_id=2, some_id=3, expected_vocab=7, expected_width=8)
    assert neither["forwards"] == 8 and not neither["doses"]["low"]["eligible"]
    assert [e["kind"] for e in events[-4:]] == ["repeat"] * 4


def test_completed_high_transfer_witness_survives_dose_measurement_error(monkeypatch):
    study = subject()
    torch.manual_seed(22)
    model, events, state = ToyModel().eval(), [], {"armed": False}
    with torch.no_grad():
        model.embedding.weight.mul_(100)
    original_observe, original_distance = study._observed, study.distance

    def observed(*args, **kwargs):
        value = original_observe(*args, **kwargs)
        if kwargs.get("source") is not None and args[4].get("dose") == "high":
            state["armed"] = True
        return value

    def bad_distance(*args):
        if state["armed"]:
            raise RuntimeError("fabricated high-dose failure")
        return original_distance(*args)

    monkeypatch.setattr(study, "_observed", observed)
    monkeypatch.setattr(study, "distance", bad_distance)
    with pytest.raises(RuntimeError, match="fabricated high-dose failure"):
        study.run_case(model, case(), events.append, any_id=2, some_id=3, expected_vocab=7, expected_width=8)
    last = events[-1]
    assert (last["kind"], last["phase"], last["dose"]) == ("forward_failed", "dose_measurement", "high")
    assert {"mask_logits_f32le", "submitted_f32le", "before_f32le", "after_f32le",
            "full_before_f32le", "full_after_f32le"}.issubset(last)


def test_full_toy_schedule_retains_calibration_and_checks_order_before_forward():
    study = subject()
    torch.manual_seed(22)
    model, cases, events = ToyModel().eval(), [], []
    with torch.no_grad():
        model.embedding.weight.mul_(100)
        model.head.weight.zero_()
        model.head.bias.zero_()
    for triple in (study.CALIBRATION, *study.TRIPLES):
        for group in study.GROUPS:
            cases.append(case(group, triple, "calibration" if triple == study.CALIBRATION else "evaluation"))
    manifest = {"cases": cases, "scored_tokens": {"any": 2, "some": 3}}
    wrong = copy.deepcopy(manifest)
    wrong["cases"][0], wrong["cases"][1] = wrong["cases"][1], wrong["cases"][0]
    with pytest.raises(ValueError, match="order"):
        study.run_study(model, wrong, events.append, expected_vocab=7, expected_width=8)
    assert not events
    result = study.run_study(model, manifest, events.append, expected_vocab=7, expected_width=8)
    assert result["forwards"] == 960
    assert [row["id"] for row in result["cases"]] == [row["id"] for row in cases]
    assert sum(event["kind"] == "self" for event in events) == 128
    assert sum(event["kind"] == "transfer" for event in events) == 768


def test_event_disk_reserve_stops_before_writing(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "bert_cue_dose_response_v1.py"
    spec = importlib.util.spec_from_file_location("dose_owner_test", path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    calls = []
    prior = SimpleNamespace(write_event=lambda *args: calls.append(args))
    monkeypatch.setattr(owner.shutil, "disk_usage", lambda _: SimpleNamespace(free=owner.EVENT_DISK_RESERVE - 1))
    with pytest.raises(RuntimeError, match="disk reserve"):
        owner.emit_with_headroom(prior, None, None, Path("."), {"kind": "test"})
    assert not calls
    monkeypatch.setattr(owner.shutil, "disk_usage", lambda _: SimpleNamespace(free=owner.EVENT_DISK_RESERVE))
    owner.emit_with_headroom(prior, None, None, Path("."), {"kind": "test"})
    assert len(calls) == 1
    monkeypatch.setattr(owner.shutil, "disk_usage", lambda _: SimpleNamespace(free=owner.PRELAUNCH_DISK_RESERVE - 1))
    with pytest.raises(RuntimeError, match="disk reserve"):
        owner.require_prelaunch_space(Path("run"))
    monkeypatch.setattr(owner.shutil, "disk_usage", lambda _: SimpleNamespace(free=owner.PRELAUNCH_DISK_RESERVE))
    owner.require_prelaunch_space(Path("run"))
