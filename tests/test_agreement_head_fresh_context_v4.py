"""Fabricated CPU checks for the fixed fresh-context comparison."""

import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.agreement_head_audit_v2 import audit
from looking_glass.agreement_head_study_v2 import DoseIneligible, run_fresh_context


def _fixture():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(402)
        config = GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=4,
                            n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    rows = [[1, 4 * group + index + 2, 30] for group in range(4) for index in range(4)]
    return model, rows


def test_fresh_mapping_control_dose_and_fixed_rule_replay(tmp_path):
    model, rows = _fixture()
    summary = run_fresh_context(model, rows, tmp_path / "fresh", candidate=1, comparator=0,
                                present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True})
    result = audit(tmp_path / "fresh", rows, phase="fresh_context", heads=(1, 0), present_ids=(1, 2),
                   expected_width=16, expected_nheads=2, expected_vocab=64)
    assert summary["forward_count"] == result["forward_count"] == 128
    assert summary["evaluation"] == result["evaluation"]
    assert result["evaluation"]["status"] in ("baseline_limited", "passes_fixed_rule", "does_not_pass_fixed_rule")
    assert summary["details"] == result["details"]
    for head in summary["dose_plan"]["heads"]:
        for cell in head["targets"]:
            target = cell["target"]
            group, local = target // 4 * 4, target % 4
            donors = {item["kind"]: item["donor"] for item in cell["donors"]}
            assert donors == {"same": group + (local ^ 2), "opposite": group + (local ^ 3)}
    for detail in result["details"]:
        assert set(detail["donor_distances"]) == set(detail["interpolation_fractions"]) == {"same", "opposite"}
        for name, distance in detail["donor_distances"].items():
            assert 0 < detail["interpolation_fractions"][name] == detail["requested_q"] / distance <= 1
    with pytest.raises(ValueError, match="not a bounded complete candidate"):
        audit(tmp_path / "fresh", rows, phase="evaluation", heads=(1, 0), present_ids=(1, 2),
              expected_width=16, expected_nheads=2, expected_vocab=64)


def test_ineligible_fresh_head_stops_before_any_patch(tmp_path):
    model, rows = _fixture()
    with torch.no_grad():
        model.transformer.h[3].attn.c_attn.weight[:, 40:48] = 0
        model.transformer.h[3].attn.c_attn.bias[40:48] = 0
    with pytest.raises(DoseIneligible):
        run_fresh_context(model, rows, tmp_path / "ineligible", candidate=0, comparator=1,
                          present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True})
    records = [json.loads(line) for line in (tmp_path / "ineligible" / "events.jsonl").read_bytes().splitlines()]
    completed = [r for r in records if r["event"]["type"] == "forward_complete"]
    assert len(completed) == 16 and all(r["event"]["kind"] == "observe" for r in completed)
    replay = audit(tmp_path / "ineligible", rows, phase="fresh_context", heads=(0, 1), present_ids=(1, 2),
                   expected_width=16, expected_nheads=2, expected_vocab=64)
    assert replay["outcome"] == "dose_ineligible" and replay["evaluation"] is None
    assert (replay["candidate"], replay["comparator"]) == (0, 1)


def test_fresh_accepts_deliverable_positive_doses_below_legacy_floor(tmp_path):
    model, rows = _fixture()
    with torch.no_grad():
        model.transformer.h[3].attn.c_attn.weight[:, 32:48] *= 1e-4
        model.transformer.h[3].attn.c_attn.bias[32:48] *= 1e-4
    summary = run_fresh_context(model, rows, tmp_path / "small", candidate=1, comparator=0,
                                present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True})
    assert all(0 < cell["requested_q"] < 2.0 ** -20
               for head in summary["dose_plan"]["heads"] for cell in head["targets"])
    replay = audit(tmp_path / "small", rows, phase="fresh_context", heads=(1, 0), present_ids=(1, 2),
                   expected_width=16, expected_nheads=2, expected_vocab=64)
    assert replay["forward_count"] == 128 and replay["evaluation"] == summary["evaluation"]
