"""Fabricated CPU checks for the agreement head-selection mechanics."""

import math
import hashlib
import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.agreement_head_study_v1 import (
    _distance, _evaluation_rule, _replacement, _rank_heads, run_development, run_evaluation,
)
from looking_glass.agreement_head_audit_v1 import audit


def test_matched_replacement_is_non_amplifying_and_matches_actual_dose():
    target = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32)
    donor = torch.tensor([2.0, 4.0, -2.0], dtype=torch.float32)
    distance = math.sqrt(6)
    q = distance / 2
    result, before_distance, actual = _replacement(target, donor, q)
    assert before_distance == pytest.approx(distance)
    assert actual == pytest.approx(q, rel=1e-5)
    assert result.tolist() == [1.5, 3.0, -2.5]
    with pytest.raises(ValueError, match="dose"):
        _replacement(target, donor, distance * 2)
    with pytest.raises(ValueError, match="dose"):
        _replacement(target, target, q)


def test_ranking_preserves_negative_scores_and_resolves_ties_by_head():
    values = {2: [-0.2, 0.0], 0: [-0.1, -0.1], 1: [-0.3, 0.0]}
    ranked = _rank_heads(values)
    assert [item["head"] for item in ranked] == [0, 2, 1]
    assert [item["mean_D"] for item in ranked] == pytest.approx([-0.1, -0.1, -0.15])


def test_complete_tiny_model_scan_records_all_calls_and_self_controls(tmp_path):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(401)
        config = GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=4,
                            n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    rows = [[1, 4 * group + index + 2, 30] for group in range(4) for index in range(4)]
    output = run_development(model, rows, tmp_path / "attempt", heads=(0, 1),
                             present_ids=(1, 2), secondary_ids=None, metadata={"fixture": "tiny CPU GPT-2"})
    assert output["forward_count"] == 66
    assert len(output["ranking"]) == 2
    assert output["candidate"] != output["comparator"]
    assert output["q"] > 0
    assert all(len(item["values"]) == 8 for item in output["ranking"])
    assert (tmp_path / "attempt" / "terminal.json").is_file()
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    replay = audit(tmp_path / "attempt", rows, phase="development", heads=(0, 1), present_ids=(1, 2))
    assert replay["ranking"] == output["ranking"]
    assert replay["forward_count"] == 66
    # Rehash a malicious change to an untouched coordinate: hashes alone still
    # match, but independent before/after replay must reject the modification.
    event_path = tmp_path / "attempt" / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    patch = next(item for item in events if item["event"].get("type") == "forward_complete"
                 and item["event"].get("kind") == "opposite")
    reference = patch["tensors"]["after"]
    data = bytearray((tmp_path / "attempt" / "tensors" / (reference["sha256"] + ".f32le")).read_bytes())
    data[0] ^= 1
    reference["sha256"] = hashlib.sha256(data).hexdigest()
    (tmp_path / "attempt" / "tensors" / (reference["sha256"] + ".f32le")).write_bytes(data)
    raw = b"".join((json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n").encode() for e in events)
    event_path.write_bytes(raw)
    terminal_path = tmp_path / "attempt" / "terminal.json"
    terminal = json.loads(terminal_path.read_bytes())
    terminal["event_log_bytes"] = len(raw)
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="untouched bits"):
        audit(tmp_path / "attempt", rows, phase="development", heads=(0, 1), present_ids=(1, 2))


def test_distance_rejects_nonfinite():
    with pytest.raises(ValueError):
        _distance(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_fixed_rule_retains_negative_and_baseline_limited_results():
    def details(correct=True, candidate=0.1):
        return [{"head": head, "target": i, "sign": 1 if i % 2 == 0 else -1,
                 "clean_margin": (1 if i % 2 == 0 else -1) if correct else 0,
                 "D": candidate if head == 3 else 0.02} for head in (3, 7) for i in range(16)]
    assert _evaluation_rule(details(), 3, 7)["status"] == "passes_fixed_rule"
    negative = _evaluation_rule(details(candidate=-0.1), 3, 7)
    assert negative["status"] == "does_not_pass_fixed_rule"
    assert negative["candidate_negative"] == 16
    assert _evaluation_rule(details(correct=False), 3, 7)["status"] == "baseline_limited"


def test_evaluation_replay_checks_fixed_candidate_and_rule(tmp_path):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(402)
        config = GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=4,
                            n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    rows = [[1, 4 * group + index + 2, 30] for group in range(4) for index in range(4)]
    output = run_evaluation(model, rows, tmp_path / "evaluation", candidate=1, comparator=0,
                            present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True})
    replay = audit(tmp_path / "evaluation", rows, phase="evaluation", heads=(1, 0), present_ids=(1, 2))
    assert output["forward_count"] == replay["forward_count"] == 128
    assert output["candidate"] == replay["candidate"] == 1
    assert output["evaluation"] == replay["evaluation"]
