"""Fabricated CPU checks for the agreement head-selection mechanics."""

import math
import hashlib
import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.agreement_head_study_v2 import (
    _distance, _dose_plan, _evaluation_rule, _replacement, _rank_heads, run_development, run_evaluation,
)
from looking_glass.agreement_head_audit_v2 import audit
from looking_glass import agreement_head_study_v2 as study


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
    assert output["dose_plan"]["relative_requested"] == 0.05
    assert [item["head"] for item in output["dose_plan"]["heads"] if item["eligible"]] == [0, 1]
    assert all(len(item["values"]) == 8 for item in output["ranking"])
    assert (tmp_path / "attempt" / "terminal.json").is_file()
    records = [json.loads(line) for line in (tmp_path / "attempt" / "events.jsonl").read_bytes().splitlines()]
    plan_record = next(record for record in records if record["event"]["type"] == "dose_plan")
    first_patch = next(record for record in records if record["event"].get("kind") == "self")
    assert plan_record["index"] < first_patch["index"]
    assert sum(len(head["targets"]) for head in plan_record["event"]["heads"]) == 16
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    audit_options = {"phase": "development", "heads": (0, 1), "present_ids": (1, 2),
                     "expected_width": 16, "expected_nheads": 2, "expected_vocab": 64}
    replay = audit(tmp_path / "attempt", rows, **audit_options)
    assert replay["ranking"] == output["ranking"]
    assert replay["dose_plan"] == output["dose_plan"]
    assert replay["forward_count"] == 66
    with pytest.raises(ValueError, match="coordinates or shapes"):
        audit(tmp_path / "attempt", rows, **{**audit_options, "expected_nheads": 4})
    with pytest.raises(ValueError, match="coordinates or shapes"):
        audit(tmp_path / "attempt", rows, **{**audit_options, "expected_vocab": 63})
    event_path = tmp_path / "attempt" / "events.jsonl"
    terminal_path = tmp_path / "attempt" / "terminal.json"
    original_events, original_terminal = event_path.read_bytes(), terminal_path.read_bytes()
    altered = [json.loads(line) for line in original_events.splitlines()]
    plan = next(item for item in altered if item["event"]["type"] == "dose_plan")
    plan["event"]["heads"][0]["eligible"] = False
    altered_bytes = b"".join((json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n").encode() for e in altered)
    event_path.write_bytes(altered_bytes)
    terminal = json.loads(original_terminal)
    terminal["event_log_bytes"] = len(altered_bytes)
    terminal_path.write_bytes(json.dumps(terminal).encode())
    with pytest.raises(ValueError, match="eligibility plan differs"):
        audit(tmp_path / "attempt", rows, **audit_options)
    event_path.write_bytes(original_events)
    terminal_path.write_bytes(original_terminal)
    # Rehash a malicious change to an untouched coordinate: hashes alone still
    # match, but independent before/after replay must reject the modification.
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
    terminal = json.loads(terminal_path.read_bytes())
    terminal["event_log_bytes"] = len(raw)
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="untouched bits"):
        audit(tmp_path / "attempt", rows, **audit_options)


def test_distance_rejects_nonfinite():
    with pytest.raises(ValueError):
        _distance(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_dose_plan_excludes_insufficient_donors_before_patch_and_never_amplifies():
    observations = []
    for row in range(16):
        # Head zero's subject-dependent donor differences are ample; head one
        # is identical across every row and must not receive any patch.
        vector = torch.tensor([[[1.0 + 0.2 * (row % 4), 2.0, 1.0, 2.0]]])
        observations.append((vector, None))
    plan, replacements = _dose_plan(observations, (0, 1),
                                    (0, 1, 4, 5, 8, 9, 12, 13), 2)
    assert [entry["eligible"] for entry in plan["heads"]] == [True, False]
    assert all(cell["status"] == "insufficient_donor_distance" for cell in plan["heads"][1]["targets"])
    assert all(key[0] == 0 for key in replacements)
    for cell in plan["heads"][0]["targets"]:
        assert cell["requested_q"] == pytest.approx(0.05 * cell["receiver_norm"])
        assert all(item["distance"] >= cell["requested_q"] for item in cell["donors"])
        assert max(item["realized_q"] for item in cell["donors"]) / min(
            item["realized_q"] for item in cell["donors"]) <= 1.00001


def test_scan_skips_a_real_ineligible_head_and_replays_remaining_heads(tmp_path):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(405)
        config = GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=18, n_layer=4,
                            n_head=3, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    # Zero the selected layer's value channels for head 2: its projection
    # input is identically zero, so it is genuinely ineligible without fakes.
    with torch.no_grad():
        model.transformer.h[3].attn.c_attn.weight[:, 48:54] = 0
        model.transformer.h[3].attn.c_attn.bias[48:54] = 0
    rows = [[1, 4 * group + index + 2, 30] for group in range(4) for index in range(4)]
    result = run_development(model, rows, tmp_path / "subset", heads=(0, 1, 2),
                             present_ids=(1, 2), secondary_ids=None, metadata={"fixture": "zero head"})
    assert result["forward_count"] == 66
    assert [item["eligible"] for item in result["dose_plan"]["heads"]] == [True, True, False]
    assert {entry["head"] for entry in result["ranking"]} == {0, 1}
    replay = audit(tmp_path / "subset", rows, phase="development", heads=(0, 1, 2), present_ids=(1, 2),
                   expected_width=18, expected_nheads=3, expected_vocab=64)
    assert replay["dose_plan"] == result["dose_plan"]
    assert replay["ranking"] == result["ranking"]


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
    replay = audit(tmp_path / "evaluation", rows, phase="evaluation", heads=(1, 0), present_ids=(1, 2),
                   expected_width=16, expected_nheads=2, expected_vocab=64)
    assert output["forward_count"] == replay["forward_count"] == 128
    assert output["candidate"] == replay["candidate"] == 1
    assert output["evaluation"] == replay["evaluation"]


def test_evaluation_ineligible_head_stops_before_all_patches(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(404)
        config = GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=4,
                            n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    rows = [[1, 4 * group + index + 2, 30] for group in range(4) for index in range(4)]
    original = study._dose_plan

    def fail_one(*args):
        plan, replacements = original(*args)
        plan["heads"][1]["eligible"] = False
        plan["heads"][1]["targets"][0]["status"] = "insufficient_donor_distance"
        return plan, replacements

    monkeypatch.setattr(study, "_dose_plan", fail_one)
    with pytest.raises(ValueError, match="dose_ineligible"):
        run_evaluation(model, rows, tmp_path / "ineligible", candidate=0, comparator=1,
                       present_ids=(1, 2), secondary_ids=None, metadata={"fixture": True})
    records = [json.loads(line) for line in (tmp_path / "ineligible" / "events.jsonl").read_bytes().splitlines()]
    completed = [r for r in records if r["event"]["type"] == "forward_complete"]
    assert len(completed) == 16 and all(r["event"]["kind"] == "observe" for r in completed)
    assert records[-1]["event"]["type"] == "dose_plan"
    assert json.loads((tmp_path / "ineligible" / "terminal.json").read_bytes())["status"] == "failed"


def test_postforward_rounding_failure_retains_result_and_stops(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(403)
        config = GPT2Config(vocab_size=64, n_positions=16, n_ctx=16, n_embd=16, n_layer=4,
                            n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    rows = [[1, 4 * group + index + 2, 30] for group in range(4) for index in range(4)]
    original, state = study.run_gpt2_head_slice, {"returned": False, "calls": 0}

    def forward(*args, **kwargs):
        state["calls"] += 1
        value = original(*args, **kwargs)
        state["returned"] = True
        return value

    def probe():
        if state["returned"]:
            raise RuntimeError("synthetic rounding drift")

    monkeypatch.setattr(study, "run_gpt2_head_slice", forward)
    monkeypatch.setattr(study, "binary64_rounding_probe", probe)
    with pytest.raises(RuntimeError, match="rounding drift"):
        run_development(model, rows, tmp_path / "drift", heads=(0, 1), present_ids=(1, 2),
                        secondary_ids=None, metadata={"fixture": True})
    events = [json.loads(line) for line in (tmp_path / "drift" / "events.jsonl").read_text().splitlines()]
    terminal = json.loads((tmp_path / "drift" / "terminal.json").read_bytes())
    assert [record["event"]["type"] for record in events] == ["forward_begin", "forward_complete"]
    assert state["calls"] == 1 and terminal["status"] == "failed"
    assert terminal["detail"]["completed_forwards"] == 1
    for reference in events[1]["tensors"].values():
        assert (tmp_path / "drift" / "tensors" / (reference["sha256"] + ".f32le")).is_file()
