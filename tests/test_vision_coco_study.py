"""No-checkpoint, synthetic-state checks for the vision runner."""
from types import SimpleNamespace
import hashlib
import json

import pytest
import torch

from looking_glass.vision_coco_study import (
    Evidence, aggregate, baseline_status, check_denormals, check_landing, forward, prepare_dose, run_split, score,
)


class Layer(torch.nn.Module):
    def forward(self, x):
        return x + .01 * x.mean(dim=1, keepdim=True)


class FakeVit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.nn.Module()
        self.vit.encoder = torch.nn.Module()
        self.vit.encoder.layer = torch.nn.ModuleList([Layer() for _ in range(12)])

    def forward(self, pixel_values):
        x = pixel_values
        for layer in self.vit.encoder.layer:
            x = layer(x)
        value = x[:, 0, 0]
        logits = torch.zeros((1, 1000), dtype=torch.float32)
        logits[:, 151:269] = 2 + value[:, None]
        logits[:, 281:286] = -value[:, None]
        return SimpleNamespace(logits=logits)


def test_unsupported_pair_still_runs_descriptive_interventions(tmp_path):
    model = FakeVit().eval()
    torch.manual_seed(4)
    pixels = {1: torch.randn(1, 197, 192), 2: torch.randn(1, 197, 192)}
    pair = {"pair_id": "pair-00", "split": "development", "cat": {"image_id": 1},
            "dog": {"image_id": 2}, "box_tokens": [1, 2, 15, 16],
            "control_tokens": [40, 41, 54, 55]}
    donors = {"rows": [{"pair_id": "pair-00", "receiver_side": side, "donor": None}
                        for side in ("cat", "dog")]}
    evidence = Evidence(tmp_path / "run", 30)
    rows = run_split(model, [pair], donors, pixels, evidence)
    assert rows[0]["status"] == "baseline-ineligible"
    for side in ("cat", "dog"):
        info = rows[0]["sides"][side]
        assert info["dose"]["status"] == "matchable"
        assert {"matched_inside", "matched_outside", "matched_random", "full_swap",
                "null_inside", "null_outside", "self_inside", "self_outside", "post_clean"} <= set(info["conditions"])
    assert evidence.submissions == 28
    with pytest.raises(ValueError, match="eight"):
        aggregate(rows)


def test_same_species_never_extrapolates_and_random_is_repeatable():
    own = torch.zeros(1, 197, 192)
    other = torch.ones_like(own)
    same = torch.full_like(own, .1)
    inside, outside = [1, 2, 15, 16], [40, 41, 54, 55]
    status, requested = prepare_dose(own, other, inside, outside, "pair-00", "cat", same)
    assert status["status"] == "matchable"
    assert status["same_species_status"] == "distance-too-small"
    assert "same_species" not in requested
    status2, requested2 = prepare_dose(own, other, inside, outside, "pair-00", "cat", same)
    assert status == status2
    assert torch.equal(requested["random"], requested2["random"])


def test_nonfinite_is_failure_not_dose_ineligibility():
    own = torch.zeros(1, 197, 192)
    other = torch.ones_like(own)
    other[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        prepare_dose(own, other, [1, 2, 15, 16], [40, 41, 54, 55], "pair-00", "cat")


def test_landing_checks_unchanged_and_selected_bits():
    own = torch.zeros(1, 197, 192)
    after = own.clone()
    after[0, 3, 0] = 1
    captures = {"before_all": own, "after_all": after}
    with pytest.raises(ValueError, match="untouched"):
        check_landing(captures, own, [1, 2, 15, 16], own[0, [1, 2, 15, 16]])


def test_primary_denominator_excludes_ineligible_and_uses_at_least_four():
    def rows(statuses):
        return [{"status": s, "pair_id": f"pair-{i + 4:02d}", "split": "heldout"} for i, s in enumerate(statuses)]
    excluded = ["baseline-ineligible"] * 4
    assert aggregate(rows(["positive", "positive", "negative", "baseline-ineligible"] + excluded))["status"] == "inconclusive"
    assert aggregate(rows(["positive", "positive", "positive", "negative"] + excluded))["status"] == "positive"
    assert aggregate(rows(["positive", "positive", "negative", "negative"] + excluded))["status"] == "negative"
    status = baseline_status({"margin": -1.0, "cat_mass": .3}, {"margin": 1.0, "dog_mass": .3})
    assert status["eligible"] and status["gap"] == 2


def test_group_score_stable_when_global_softmax_underflows():
    logits = torch.zeros(1, 1000)
    logits[0, 0] = 1000
    measured, _ = score(logits)
    assert measured["margin"] == pytest.approx(__import__("math").log(118 / 5))
    assert measured["dog_mass"] == 0 and measured["cat_mass"] == 0


def test_group_score_preserves_large_common_offset():
    logits = torch.full((1, 1000), 1e20)
    measured, conditional = score(logits)
    assert measured["margin"] == pytest.approx(__import__("math").log(118 / 5))
    assert measured["dog_mass"] == pytest.approx(.118)
    assert measured["cat_mass"] == pytest.approx(.005)
    assert measured["off_mass"] == pytest.approx(.877)
    assert sum(conditional) == pytest.approx(1)


def test_returned_bits_survive_metadata_failure(tmp_path, monkeypatch):
    from looking_glass.intervention_plan import ActivationPlan
    def fail(_self):
        raise RuntimeError("metadata failure")
    monkeypatch.setattr(ActivationPlan, "metadata", property(fail))
    evidence = Evidence(tmp_path / "failed", 1)
    with pytest.raises(RuntimeError, match="metadata failure"):
        forward(FakeVit(), torch.ones(1, 197, 192), evidence, "metadata-test")
    assert evidence.submissions == 1
    assert evidence.active["return_observed"] is True
    assert "logits" in evidence.active["tensor_layout"]
    assert (evidence.root / "forward-001.bin").exists()


def test_denormal_mode_is_checked():
    assert torch.set_flush_denormal(False)
    check_denormals()
    try:
        assert torch.set_flush_denormal(True)
        with pytest.raises(ValueError, match="denormal"):
            check_denormals()
    finally:
        torch.set_flush_denormal(False)


def test_development_gate_rejects_missing_or_inconsistent_evidence(tmp_path):
    from tools.run_vision_coco_study import CASES_SHA, DONORS_SHA, completed_development
    data = tmp_path / "data"
    data.mkdir()
    def put(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")
    revision = "a" * 40
    rows = [{"pair_id": f"pair-{i:02d}", "split": "development", "status": "baseline-ineligible"}
            for i in range(4)]
    started = {"split": "development", "case_sha256": CASES_SHA, "donor_sha256": DONORS_SHA,
               "checkpoint_preflight_only": False, "source": {"revision": revision}}
    put(data / "started.json", started)
    for row in rows:
        put(data / f"{row['pair_id']}.json", row)
    for i in range(1, 9):
        name = f"forward-{i:03d}.bin"
        payload = bytes([i])
        (data / name).write_bytes(payload)
        put(data / f"forward-{i:03d}.json", {"submission": i, "file": name,
                                              "sha256": hashlib.sha256(payload).hexdigest()})
    terminal = {"status": "complete", "split": "development", "source_revision": revision,
                "case_sha256": CASES_SHA, "donor_sha256": DONORS_SHA, "submissions": 8,
                "saved_forwards": 8, "pairs": rows, "checkpoint_preflight_only": False,
                "synthetic_rehearsal": False}
    put(data / "terminal.json", terminal)
    supervisor = {"status": "complete", "clean": True, "exit_code": 0, "timed_out": False,
                  "terminal_sha256": hashlib.sha256((data / "terminal.json").read_bytes()).hexdigest()}
    put(tmp_path / "supervisor.json", supervisor)
    assert completed_development(tmp_path, revision) == supervisor["terminal_sha256"]
    for bad_field, bad_value in (("timed_out", True), ("exit_code", False)):
        bad = dict(supervisor, **{bad_field: bad_value})
        put(tmp_path / "supervisor.json", bad)
        with pytest.raises(ValueError):
            completed_development(tmp_path, revision)
    put(tmp_path / "supervisor.json", supervisor)
    (data / "forward-003.bin").unlink()
    with pytest.raises(ValueError):
        completed_development(tmp_path, revision)
