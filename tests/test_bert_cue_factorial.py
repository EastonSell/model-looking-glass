import copy
import importlib

import pytest
import torch
from torch import nn


def subject():
    return importlib.import_module("looking_glass.bert_cue_factorial")


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


def case(cue_set="new"):
    groups = {"prior": ("never", "rarely", "always", "often"),
              "new": ("not", "hardly", "really", "certainly")}
    rows = {}
    for token, cue in enumerate(groups[cue_set], 5):
        ids = [1, 2, 3, token, 4, 9, 10, 11, 12, 13, 14]
        rows[cue] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    return {"id": "toy", "split": "evaluation", "triple_id": "triple-01", "frame": "main",
            "cue_set": cue_set, "indices": {"cue": 3, "mask": 5, "object": 8}, "rows": rows}


def synthetic_rows():
    rows = []
    for i in range(3):
        for frame in ("main", "subject_relative"):
            for group in ("prior", "new"):
                base = {("prior", "main"): 0.1, ("new", "main"): 0.2,
                        ("prior", "subject_relative"): 0.3, ("new", "subject_relative"): 0.6}[group, frame]
                score = base + i * 0.01
                rows.append({"id": f"{i}-{frame}-{group}", "split": "evaluation", "triple_id": f"triple-{i+1:02d}",
                             "frame": frame, "cue_set": group, "baseline_eligible": True,
                             "dose_eligible": True, "specificity_passed": score > 0.05, "cue_min_margin": score})
    return rows


@pytest.mark.parametrize("group,negative,affirmative", [
    ("prior", ("never", "rarely"), ("always", "often")),
    ("new", ("not", "hardly"), ("really", "certainly")),
])
def test_strict_screen_and_same_target_control(group, negative, affirmative):
    study = subject()
    margins = {negative[0]: 1.0, negative[1]: 0.5, affirmative[0]: 0.0, affirmative[1]: 0.0}
    assert not study.baseline_screen(margins, group)["eligible"]
    margins[negative[1]] = 0.5001
    assert study.baseline_screen(margins, group)["eligible"]
    cues = negative + affirmative
    effects = {(donor, target): ((-1 if target in negative else 1) *
                                (0.2 if (donor in negative) != (target in negative) else 0.04))
               for target in cues for donor in cues if donor != target}
    assert study.context_specificity(effects, group)["passed"]
    effects[affirmative[0], negative[0]] = -0.05
    effects[negative[1], negative[0]] = 0.0
    result = study.context_specificity(effects, group)
    assert result["targets"][negative[0]]["margin"] == 0.05
    assert not result["targets"][negative[0]]["passed"]


def test_paired_contrasts_interaction_and_cell_outcomes():
    study = subject()
    outcome = study.evaluate(synthetic_rows())
    assert all(c["status"] == "passed" for c in outcome["cells"].values())
    assert outcome["cue_contrasts"]["main"]["mean"] == pytest.approx(0.1)
    assert outcome["cue_contrasts"]["subject_relative"]["mean"] == pytest.approx(0.3)
    assert outcome["frame_contrasts"]["prior"]["mean"] == pytest.approx(0.2)
    assert outcome["frame_contrasts"]["new"]["mean"] == pytest.approx(0.4)
    assert outcome["interaction"]["mean"] == pytest.approx(0.2)
    assert outcome["interaction"]["eligible"] == 3


def test_contrast_eligibility_is_paired_and_never_imputed():
    study, rows = subject(), synthetic_rows()
    rows[0]["baseline_eligible"] = False
    rows[7]["dose_eligible"] = False
    rows[7]["specificity_passed"], rows[7]["cue_min_margin"] = None, None
    outcome = study.evaluate(rows)
    assert outcome["cue_contrasts"]["main"]["eligible"] == 2
    assert outcome["interaction"]["eligible"] == 1
    assert outcome["interaction"]["status"] == "unevaluable"
    assert outcome["interaction"]["mean"] is None
    rows[4]["baseline_eligible"] = False
    outcome = study.evaluate(rows)
    assert outcome["cells"]["prior/main"]["status"] == "unevaluable"
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["triple_id"] = duplicate[-5]["triple_id"]
    with pytest.raises(ValueError):
        study.evaluate(duplicate)


def test_interaction_uses_the_declared_single_four_term_sum():
    study, rows = subject(), synthetic_rows()
    values = {("new", "subject_relative"): 1.0, ("prior", "subject_relative"): 1e-16,
              ("new", "main"): 1.0, ("prior", "main"): 2e-16}
    for row in rows:
        row["cue_min_margin"] = values[row["cue_set"], row["frame"]]
        row["specificity_passed"] = row["cue_min_margin"] > 0.05
    assert study.evaluate(rows)["interaction"]["values"][0]["value"] == 1e-16


def test_fixed_dose_tie_and_nonfinite_distance():
    study = subject()
    assert study.dose_eligible(0.5)
    assert not study.dose_eligible(0.499999)
    with pytest.raises(ValueError):
        study.dose_eligible(float("nan"))


@pytest.mark.parametrize("group", ["prior", "new"])
def test_real_toy_interventions_use_half_norm_even_with_failed_baseline(group):
    study = subject()
    torch.manual_seed(22)
    model, journal = ToyModel().eval(), []
    with torch.no_grad():
        model.embedding.weight.mul_(100)
        model.head.weight.zero_()
        model.head.bias.zero_()
    result = study.run_case(model, case(group), journal.append, any_id=2, some_id=3,
                            expected_vocab=7, expected_width=8)
    assert not result["baseline_eligible"] and result["dose_eligible"]
    assert result["forwards"] == 72
    expected_limiting = [cue for cue, detail in result["specificity"]["cue2"]["targets"].items()
                         if detail["margin"] == result["cue_min_margin"]]
    assert result["limiting_targets"] == expected_limiting
    transfers = [event for event in journal if event["kind"] == "transfer"]
    assert len(transfers) == 48 and all(event["requested_radius"] == 0.5 for event in transfers)
    assert all(event["requested_fraction"] <= 1.0 for event in transfers)
    assert [event["kind"] for event in journal[-4:]] == ["repeat"] * 4


def test_zero_distance_retains_baselines_and_repeats_without_patches():
    study = subject()
    torch.manual_seed(22)
    model, journal = ToyModel().eval(), []
    with torch.no_grad():
        model.embedding.weight[6:9].copy_(model.embedding.weight[5].expand(3, -1))
    result = study.run_case(model, case(), journal.append, any_id=2, some_id=3,
                            expected_vocab=7, expected_width=8)
    assert not result["dose_eligible"] and result["forwards"] == 8
    assert result["limiting_targets"] == [] and result["cue_min_margin"] is None
    assert sum(event["kind"] == "dose_ineligible" for event in journal) == 1
    assert not any(event["kind"] == "transfer" for event in journal)
    assert [event["kind"] for event in journal[-4:]] == ["repeat"] * 4


def test_nonfinite_nonbinding_distance_never_becomes_eligible():
    study = subject()
    cues = ("not", "hardly", "really", "certainly")
    captures = {cue: {site: torch.tensor([float(i), 0.0]) for site in study.SITES}
                for i, cue in enumerate(cues)}
    captures["certainly"]["cue5"] = torch.tensor([float("nan"), 0.0])
    with pytest.raises(ValueError, match="finite"):
        study.choose_distances(captures, "new")


def test_completed_transfer_witness_survives_dose_measurement_error(monkeypatch):
    study = subject()
    torch.manual_seed(22)
    model, journal, state = ToyModel().eval(), [], {"armed": False}
    with torch.no_grad():
        model.embedding.weight.mul_(100)
    original_observe, original_distance = study._observed, study.distance

    def observed(*args, **kwargs):
        value = original_observe(*args, **kwargs)
        if "source" in kwargs:
            state["armed"] = True
        return value

    def bad_distance(*args):
        if state["armed"]:
            raise RuntimeError("fabricated dose failure")
        return original_distance(*args)

    monkeypatch.setattr(study, "_observed", observed)
    monkeypatch.setattr(study, "distance", bad_distance)
    with pytest.raises(RuntimeError, match="fabricated dose failure"):
        study.run_case(model, case(), journal.append, any_id=2, some_id=3,
                       expected_vocab=7, expected_width=8)
    last = journal[-1]
    assert (last["kind"], last["phase"]) == ("forward_failed", "dose_measurement")
    assert {"mask_logits_f32le", "submitted_f32le", "before_f32le", "after_f32le",
            "full_before_f32le", "full_after_f32le"}.issubset(last)


def test_full_fake_study_checks_order_before_running_and_retains_all_cells():
    study = subject()
    torch.manual_seed(22)
    model, cases, events = ToyModel().eval(), [], []
    with torch.no_grad():
        model.embedding.weight.mul_(100)
    for triple in study.TRIPLES:
        for frame in study.FRAMES:
            for group in study.GROUPS:
                item = case(group)
                item.update(id=f"{triple}-{frame}-{group}", triple_id=triple, frame=frame)
                cases.append(item)
    manifest = {"cases": cases, "scored_tokens": {"any": 2, "some": 3}}
    wrong = copy.deepcopy(manifest)
    wrong["cases"][0], wrong["cases"][1] = wrong["cases"][1], wrong["cases"][0]
    with pytest.raises(ValueError, match="order"):
        study.run_study(model, wrong, events.append, expected_vocab=7, expected_width=8)
    assert not events
    result = study.run_study(model, manifest, events.append, expected_vocab=7, expected_width=8)
    assert result["forwards"] == 864
    assert [row["id"] for row in result["cases"]] == [row["id"] for row in cases]
    assert sum(item["kind"] == "transfer" for item in events) == 576
