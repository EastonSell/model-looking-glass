"""Fabricated CPU checks for the exposed fixed-donor follow-up."""

import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.agreement_head_study_v2 import DoseIneligible, _bits, run_fixed_donor_exploration
from looking_glass.agreement_head_audit_v2 import audit
from looking_glass.gpt2_head_slice import GPT2HeadSlicePoint, run_gpt2_head_slice


def _model():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(402)
        config = GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=4,
                            n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        return GPT2LMHeadModel(config).eval()


def _rows():
    return [[1, 4 * group + index + 2, 30] for group in range(4) for index in range(4)]


def _references(model, rows):
    witnesses = []
    for index, row in enumerate(rows):
        value = run_gpt2_head_slice(model, torch.tensor([row]),
                                    point=GPT2HeadSlicePoint(layer=3, token=len(row) - 1, head=0),
                                    run_id=f"reference-{index}")
        witnesses.append((_bits(value.before), _bits(value.logits[0, -1].detach().contiguous())))
    return tuple(witnesses)


def test_same_other_donor_lemma_mapping_and_saved_replay(tmp_path):
    rows, model = _rows(), _model()
    references = _references(model, rows)
    out = run_fixed_donor_exploration(model, rows, tmp_path / "exploratory", candidate=1, comparator=0,
                                      present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True},
                                      reference_observations=references)
    replay = audit(tmp_path / "exploratory", rows, phase="exploratory", heads=(1, 0), present_ids=(1, 2),
                   expected_width=16, expected_nheads=2, expected_vocab=64, reference_observations=references)
    assert out["forward_count"] == replay["forward_count"] == 128
    assert out["candidate"] == replay["candidate"] == 1
    assert out["comparison"] == replay["comparison"]
    assert "status" not in out["comparison"] and "criterion" not in out["comparison"]
    for head in out["dose_plan"]["heads"]:
        for cell in head["targets"]:
            target = cell["target"]
            group, local = target // 4 * 4, target % 4
            donors = {item["kind"]: item["donor"] for item in cell["donors"]}
            assert donors == {"same": group + (local ^ 2), "opposite": group + (local ^ 3)}
            assert donors["same"] // 2 == donors["opposite"] // 2
            assert donors["same"] % 2 != donors["opposite"] % 2
    with pytest.raises(ValueError, match="not a bounded complete candidate"):
        audit(tmp_path / "exploratory", rows, phase="evaluation", heads=(1, 0), present_ids=(1, 2),
              expected_width=16, expected_nheads=2, expected_vocab=64)
    with pytest.raises(ValueError, match="requires the 16 prior clean witnesses"):
        audit(tmp_path / "exploratory", rows, phase="exploratory", heads=(1, 0), present_ids=(1, 2),
              expected_width=16, expected_nheads=2, expected_vocab=64)
    wrong = ((references[0][0], b"wrong"),) + references[1:]
    with pytest.raises(ValueError, match="prior saved witnesses"):
        audit(tmp_path / "exploratory", rows, phase="exploratory", heads=(1, 0), present_ids=(1, 2),
              expected_width=16, expected_nheads=2, expected_vocab=64, reference_observations=wrong)


def test_ineligible_fixed_donor_head_stops_before_any_patch(tmp_path):
    model, rows = _model(), _rows()
    with torch.no_grad():
        model.transformer.h[3].attn.c_attn.weight[:, 40:48] = 0
        model.transformer.h[3].attn.c_attn.bias[40:48] = 0
    references = _references(model, rows)
    with pytest.raises(DoseIneligible):
        run_fixed_donor_exploration(model, rows, tmp_path / "ineligible", candidate=0, comparator=1,
                                    present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True},
                                    reference_observations=references)
    records = [json.loads(line) for line in (tmp_path / "ineligible" / "events.jsonl").read_bytes().splitlines()]
    completed = [r for r in records if r["event"]["type"] == "forward_complete"]
    assert len(completed) == 16 and all(r["event"]["kind"] == "observe" for r in completed)
    replay = audit(tmp_path / "ineligible", rows, phase="exploratory", heads=(0, 1), present_ids=(1, 2),
                   expected_width=16, expected_nheads=2, expected_vocab=64, reference_observations=references)
    assert replay["outcome"] == "dose_ineligible" and replay["comparison"] is None
    assert (replay["candidate"], replay["comparator"]) == (0, 1)


def test_clean_mismatch_stops_before_any_patch(tmp_path):
    model, rows = _model(), _rows()
    references = _references(model, rows)
    wrong = ((references[0][0], b"wrong"),) + references[1:]
    with pytest.raises(RuntimeError, match="clean observation differs"):
        run_fixed_donor_exploration(model, rows, tmp_path / "mismatch", candidate=0, comparator=1,
                                    present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True},
                                    reference_observations=wrong)
    records = [json.loads(line) for line in (tmp_path / "mismatch" / "events.jsonl").read_bytes().splitlines()]
    completed = [r for r in records if r["event"]["type"] == "forward_complete"]
    assert len(completed) == 1 and completed[0]["event"]["kind"] == "observe"
    assert not any(r["event"].get("kind") in ("self", "same", "opposite") for r in records)
