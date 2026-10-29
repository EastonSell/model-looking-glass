"""Small synthetic checks for the public taste-context intervention."""

import importlib
import base64
import hashlib
import json
import math

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM


def _study():
    try:
        return importlib.import_module("scripts.bert_taste_patch_v1")
    except ModuleNotFoundError as exc:
        if exc.name == "scripts.bert_taste_patch_v1":
            pytest.fail("taste-context intervention is not implemented yet")
        raise


@pytest.fixture
def model():
    torch.random.default_generator.manual_seed(1701)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32,
                              n_layers=2, dim=16, hidden_dim=32, n_heads=2,
                              dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    return DistilBertForMaskedLM(config).eval()


def _bits(tensor):
    return tensor.contiguous().view(torch.int32)


def test_patch_lands_on_selected_token_and_self_patch_restores_logits(model):
    study = _study()
    left = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    right = (1, 2, 3, 4, 5, 6, 7, 11, 9, 10)
    baseline = study.observe(model, left, layer=0, position=7, mask_index=4)
    donor = study.observe(model, right, layer=0, position=7, mask_index=4)
    patched = study.observe(model, left, layer=0, position=7, mask_index=4, source=donor.before)
    self_patch = study.observe(model, left, layer=0, position=7, mask_index=4, self_patch=True)
    assert torch.equal(_bits(patched.before), _bits(baseline.before))
    assert torch.equal(_bits(patched.after), _bits(donor.before))
    assert patched.untouched_positions_equal is True
    assert torch.equal(_bits(self_patch.mask_logits), _bits(baseline.mask_logits))
    assert self_patch.untouched_positions_equal is True
    assert not study.has_hooks(model)


def test_final_block_other_position_patch_leaves_mask_logits_unchanged(model):
    study = _study()
    left = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    right = (1, 2, 3, 4, 5, 6, 7, 11, 9, 10)
    baseline = study.observe(model, left, layer=1, position=7, mask_index=4)
    donor = study.observe(model, right, layer=1, position=7, mask_index=4)
    patched = study.observe(model, left, layer=1, position=7, mask_index=4, source=donor.before)
    assert torch.equal(_bits(patched.mask_logits), _bits(baseline.mask_logits))
    assert patched.untouched_positions_equal is True
    assert not study.has_hooks(model)


def test_forward_failure_releases_installed_hooks(model, monkeypatch):
    study = _study()
    def fail(*_args, **_kwargs):
        raise OSError("synthetic forward failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(OSError, match="synthetic forward failure"):
        study.observe(model, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                      layer=0, position=7, mask_index=4)
    assert not study.has_hooks(model)


def test_logit_summary_keeps_raw_margin_and_full_vocab_probabilities():
    study = _study()
    logits = torch.tensor([0.0, 2.0, -1.0, 1.0, -2.0, 0.5], dtype=torch.float32)
    result = study.summarize_logits(logits, sweet_id=1, sour_id=2)
    denominator = sum(math.exp(value) for value in (0.0, 2.0, -1.0, 1.0, -2.0, 0.5))
    assert result["sweet_logit"] == 2.0
    assert result["sour_logit"] == -1.0
    assert result["sweet_minus_sour"] == 3.0
    assert result["sweet_probability"] == pytest.approx(math.exp(2.0) / denominator)
    assert result["sour_probability"] == pytest.approx(math.exp(-1.0) / denominator)
    assert result["top_five_ids"][0] == 1


def test_public_exposure_comparison_detects_exact_and_prefix_collisions():
    study = _study()
    planned = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    assert study.compare_exposures(planned, [[{"input_ids": [1, 9], "attention_mask": [1, 1]}]]) == 0
    for prior in ([1, 2, 3], [1, 2], [1, 2, 3, 4]):
        with pytest.raises(ValueError, match="exposure collision"):
            study.compare_exposures(planned, [[{"input_ids": prior, "attention_mask": [1] * len(prior)}]])


def test_complete_synthetic_pair_records_both_directions_and_final_control(model):
    study = _study()
    rows = [{"id": "toy-sugar", "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            {"id": "toy-lemon", "input_ids": [1, 2, 3, 4, 5, 6, 7, 11, 9, 10]}]
    records = []
    result = study.run_cases(model, rows, sweet_id=1, sour_id=2, emit=records.append)
    assert result["completed_forwards"] == 22  # 2 rows × 2 layers × (2 captures + 3 patches) + 2 repeats
    assert len(result["effects"]) == 12  # 2 directions × 2 layers × 3 patch modes
    assert {record["recipient"] for record in result["effects"]} == {"toy-sugar", "toy-lemon"}
    assert {record["mode"] for record in result["effects"]} == {"ingredient_swap", "ingredient_self", "control_swap"}
    assert all(record["effect"] == pytest.approx(record["patched_margin"] - record["baseline_margin"])
               for record in result["effects"])
    assert all(record["effect"] == 0 for record in result["effects"] if record["layer"] == 1)
    assert len([r for r in records if r["event"] == "forward_start"]) == 22
    assert len([r for r in records if r["event"] == "forward_complete"]) == 22
    for record in records:
        if record["event"] == "forward_complete" and record["mode"] in ("ingredient_swap", "control_swap"):
            assert base64.b64decode(record["after_float32_le_base64"], validate=True) == base64.b64decode(
                record["source_float32_le_base64"], validate=True)
        elif record["event"] == "forward_complete" and record["mode"] == "ingredient_self":
            assert base64.b64decode(record["after_float32_le_base64"], validate=True) == base64.b64decode(
                record["before_float32_le_base64"], validate=True)
    assert not study.has_hooks(model)


def test_writer_handles_short_writes_and_rejects_zero_progress():
    study = _study()

    class ShortWriter:
        def __init__(self, maximum):
            self.maximum, self.data = maximum, bytearray()

        def write(self, value):
            written = min(self.maximum, len(value))
            self.data.extend(value[:written])
            return written

    writer = ShortWriter(3)
    study.write_all(writer, b"abcdefg")
    assert bytes(writer.data) == b"abcdefg"
    with pytest.raises(OSError, match="write"):
        study.write_all(ShortWriter(0), b"data")


def test_self_patch_summary_uses_self_labels():
    study = _study()
    effects = []
    for mode in ("ingredient_swap", "ingredient_self", "control_swap"):
        for family in ("tea", "drink", "sauce", "water"):
            for ingredient in ("sugar", "lemon"):
                effects.append({"layer": 0, "mode": mode, "recipient": family + "-" + ingredient,
                                "effect": 0.0})
    summary = study._layer_summary(effects, 1)[0]
    assert set(summary["ingredient_self"]) == {"sugar_self", "lemon_self"}
    assert set(summary["ingredient_swap"]) == {"sugar_to_lemon", "lemon_to_sugar"}


def test_denormal_probe_rejects_flush_to_zero():
    study = _study()
    assert torch.set_flush_denormal(False)
    assert study.denormal_probe() == {"smallest_times_one_bits": 1, "normal_half_bits": 0x00400000}
    assert torch.set_flush_denormal(True)
    try:
        with pytest.raises(RuntimeError, match="denormal"):
            study.denormal_probe()
    finally:
        torch.set_flush_denormal(False)


def test_missing_cited_inventories_cannot_report_clear_exposure(tmp_path, monkeypatch):
    study = _study()
    manifest = json.loads((study.ROOT / study.CASES_PATH).read_bytes())
    manifest["prior_public_inventories"] = {}
    destination = tmp_path / study.CASES_PATH
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(study, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="inventor"):
        study.verify_cases(None)


def test_fresh_cancellation_while_recording_incomplete_stays_primary(tmp_path, monkeypatch):
    study = _study()
    original = OSError("original failure")
    cancellation = KeyboardInterrupt("interrupted terminal write")

    def interrupt(*_args, **_kwargs):
        raise cancellation

    monkeypatch.setattr(study, "_write_json", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        study.record_incomplete(tmp_path, "test", {"started": 0, "completed": 0}, original)
    assert caught.value is cancellation and caught.value.__cause__ is original


def test_parent_requires_clean_exit_complete_result_and_matching_journals(tmp_path):
    try:
        parent = importlib.import_module("scripts.bert_taste_parent_v1")
    except ModuleNotFoundError as exc:
        if exc.name == "scripts.bert_taste_parent_v1":
            pytest.fail("taste-context parent observer is not implemented yet")
        raise
    source = {"commit": "a" * 40, "selected_source_sha256": {}}
    journals = {}
    for name in ("tea", "drink", "sauce", "water"):
        data = name.encode()
        (tmp_path / (name + ".jsonl")).write_bytes(data)
        journals[name + ".jsonl"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    result = {"status": "complete", "completed_forwards": 248, "source": source, "journals": journals}
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "complete"
    assert parent.assess_result(tmp_path, 1, source, source, True)["status"] == "incomplete"
    (tmp_path / "incomplete.json").write_text("{}", encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
    (tmp_path / "incomplete.json").unlink()
    (tmp_path / "tea.jsonl").write_bytes(b"tampered")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
