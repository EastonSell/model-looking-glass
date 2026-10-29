"""Fabricated full-pass execution, failures and evidence ordering; no checkpoint."""

import copy
import hashlib
import json
import inspect
import struct
import sys
from types import SimpleNamespace

import pytest
import torch

from test_flan_multitoken_span_v2 import fake_group
import looking_glass.flan_multitoken_runner_v2 as runner
from looking_glass.flan_multitoken_journal_v2 import EvidenceJournalV2, LimitsV2


class FakeTokenizer:
    all_special_ids = [0, 1, 2]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"positive": [3], "negative": [4]}[text]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join({3: "positive", 4: "negative"}.get(i, "<special>") for i in ids)


class FakeModel(torch.nn.Module):
    def __init__(self, *, force_negative=False, tie=False, scenario="ordinary"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(d_model=2, vocab_size=256, decoder_start_token_id=0,
                                      eos_token_id=1, use_cache=False)
        self.encoder_calls = self.decoder_calls = 0
        self.force_negative, self.tie, self.scenario = force_negative, tie, scenario

    def get_encoder(self):
        return self.encoder

    def encoder(self, *, input_ids, attention_mask, return_dict):
        self.encoder_calls += 1
        index = int(input_ids[0, 4]) % 10
        if self.scenario == "zero-q":
            index %= 2
        if self.scenario == "low-only":
            x, y = {0: (0., 0.), 1: (8., 0.), 2: (0., 8.), 3: (8., 8.)}[index]
            value, word_value = (x, y), (1e8 + x, y)
        else:
            value = {0: (2., 1.), 1: (-2., -1.), 2: (3., 2.), 3: (-3., -2.)}[index]
            word_value = value
        state = torch.zeros((1, input_ids.shape[1], 2))
        for site in (0, 2):
            state[0, site] = torch.tensor(value)
        state[0, 4] = torch.tensor(word_value)
        if input_ids.shape[1] == 8:
            state[0, 5] = torch.tensor(word_value)
        return SimpleNamespace(last_hidden_state=state)

    def forward(self, *, encoder_outputs, attention_mask, decoder_input_ids,
                decoder_attention_mask, use_cache, return_dict):
        assert torch.equal(attention_mask, torch.ones_like(attention_mask))
        assert torch.equal(decoder_attention_mask, torch.ones_like(decoder_attention_mask))
        assert use_cache is False
        self.decoder_calls += 1
        state = encoder_outputs[0]
        position = 5 if state.shape[1] == 8 else 4
        value = -3.0 if self.force_negative else float(state[0, position, 0])
        length = decoder_input_ids.shape[1]
        logits = torch.full((1, length, 256), -4.0)
        for index in range(length):
            if index == 0:
                logits[0, index, 3] = value
                logits[0, index, 4] = -value
                if self.tie:
                    logits[0, index, 3] = logits[0, index, 4] = 3.0
            else:
                logits[0, index, 1] = 3.0
        return SimpleNamespace(logits=logits, past_key_values=None)


class MemoryJournal:
    """Exercise action ordering; disk durability has separate journal tests."""
    def __init__(self):
        self.events, self.blobs, self.component_count = [], {}, 0
        self.active_component = None
        self.remaining_component_bytes = 0
        self.terminal = None

    def record(self, event, detail):
        self.events.append({"event": event, "detail": detail})

    def begin_component(self, identity, *, reserve_bytes):
        assert self.active_component is None
        self.active_component = identity
        self.remaining_component_bytes = reserve_bytes
        self.record("component_start", identity)
        self.component_count += 1

    def _save(self, raw, shape, dtype):
        path = f"saved/{len(self.blobs):06d}"
        ref = {"path": path, "shape": list(shape), "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest(), "dtype": dtype}
        self.blobs[path] = raw
        return ref

    def tensor(self, shape, raw):
        if self.active_component is not None:
            self.remaining_component_bytes -= len(raw)
            assert self.remaining_component_bytes >= 0
        return self._save(raw, shape, "float32-le")

    def vector64(self, raw):
        return self._save(raw, (len(raw) // 8,), "float64-le")

    def read_tensor(self, ref):
        raw = self.blobs[ref["path"]]
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
        return raw

    def finish_component(self, returned):
        self.record("component_return", returned)
        self.active_component = None

    def finish(self, status, detail):
        assert self.terminal is None
        self.terminal = {"status": status, "detail": detail, "active": self.active_component}


def one_unit_groups():
    return [{"unit": 0, "instruction": i, "cases": fake_group(lexical_count=1)} for i in (0, 1)]


def full_groups():
    words = (("good", "evil", "rosy", "grim"), ("fine", "ugly", "nice", "poor"),
             ("able", "dire", "glad", "dull"), ("gold", "torn", "safe", "lost"))
    groups = []
    for unit in range(4):
        for instruction in (0, 1):
            cases = fake_group(lexical_count=1 if unit < 2 else 2)
            for side, word in zip(("P1", "N1", "P2", "N2"), words[unit]):
                case = cases[side]
                case["text"] = case["text"].replace(case["adjective"], word)
                case["adjective"] = word
                case["input_ids"][4] += 10 * unit
            groups.append({"unit": unit, "instruction": instruction, "cases": cases})
    return groups


def event_rows(writer, name):
    return [row["detail"] for row in writer.events if row["event"] == name]


def test_fabricated_pass_runs_all_145_actions_and_saves_both_anchors():
    model, writer = FakeModel().eval(), MemoryJournal()
    result = runner.run_fabricated(model, FakeTokenizer(), one_unit_groups(), writer,
                                   subject="story", width=2, vocab_size=256)
    rows = event_rows(writer, "evaluation_return")
    assert len(rows) == 145 and model.encoder_calls == 9
    assert len([row for row in rows if row["action"]["kind"] == "matched"]) == 80
    swaps = [row for row in rows if row["action"]["kind"] == "full_swap"]
    assert all(row["action"]["source_mask_origin"] == "donor" for row in swaps)
    assert len(event_rows(writer, "group_geometry")) == 2
    assert len(event_rows(writer, "natural_anchor_verified")) == 8
    assert writer.terminal["status"] == "complete_candidate"
    assert result["testing_only"] is True
    assert result["groups"][0]["doses"]["high"]["A"] != 0.0


def test_full_four_slot_fabricated_pass_preserves_original_ids_and_both_strata():
    model, writer = FakeModel().eval(), MemoryJournal()
    result = runner.run_fabricated(model, FakeTokenizer(), full_groups(), writer,
                                   subject="story", width=2, vocab_size=256)
    assert len(event_rows(writer, "evaluation_return")) == 580
    assert model.encoder_calls == 36 and result["slots"] == [0, 1, 2, 3]
    assert len(event_rows(writer, "natural_anchor_verified")) == 16


@pytest.mark.parametrize("model", [FakeModel(force_negative=True), FakeModel(tie=True)])
def test_unsupported_or_tied_baseline_still_runs_valid_interventions(model):
    writer = MemoryJournal()
    result = runner.run_fabricated(model.eval(), FakeTokenizer(), one_unit_groups(), writer,
                                   subject="story", width=2, vocab_size=256)
    assert len([row for row in event_rows(writer, "evaluation_return")
                if row["action"]["kind"] == "matched"]) == 80
    assert all(not row["baseline_supported"] for row in result["groups"])
    if model.tie:
        assert any(any(row["ties"]) for row in event_rows(writer, "greedy_diagnostics"))


def test_finite_low_miss_skips_only_that_action(monkeypatch):
    original = runner.plan_group_geometry
    def with_low_miss(*args, **kwargs):
        plan = original(*args, **kwargs)
        if kwargs["instruction"] == 0:
            item = plan["landings"][("P1", "low", "orthogonal")]
            item.update(status="unavailable", failure="fabricated low miss")
        return plan
    monkeypatch.setattr(runner, "plan_group_geometry", with_low_miss)
    writer = MemoryJournal()
    result = runner.run_fabricated(FakeModel().eval(), FakeTokenizer(), one_unit_groups(), writer,
                                   subject="story", width=2, vocab_size=256)
    matched = [row for row in event_rows(writer, "evaluation_return") if row["action"]["kind"] == "matched"]
    assert len(matched) == 79
    first_group = result["groups"][0]
    assert first_group["doses"]["low"]["B4"] is None
    assert first_group["doses"]["high"]["B4"] is not None


def test_physical_low_only_miss_does_not_suppress_a_valid_high_landing():
    writer = MemoryJournal()
    result = runner.run_fabricated(FakeModel(scenario="low-only").eval(), FakeTokenizer(), one_unit_groups(), writer,
                                   subject="story", width=2, vocab_size=256)
    geometry = event_rows(writer, "group_geometry")[0]
    low = next(item for item in geometry["landings"]
               if (item["receiver"], item["dose"], item["arm"]) == ("P1", "low", "opposite_adjective"))
    high = next(item for item in geometry["landings"]
                if (item["receiver"], item["dose"], item["arm"]) == ("P1", "high", "opposite_adjective"))
    assert geometry["q"] == 8.0 and low["status"] == "unavailable" and high["status"] == "matchable"
    assert result["groups"][0]["doses"]["low"]["B4"] is None


def test_physical_zero_radius_keeps_natural_and_plumbing_controls():
    writer = MemoryJournal()
    result = runner.run_fabricated(FakeModel(scenario="zero-q").eval(), FakeTokenizer(), one_unit_groups(), writer,
                                   subject="story", width=2, vocab_size=256)
    rows = event_rows(writer, "evaluation_return")
    assert len(rows) == 65 and not [row for row in rows if row["action"]["kind"] == "matched"]
    assert len([row for row in rows if row["action"]["kind"] in ("natural_final", "natural_span")]) == 16
    assert all(row["geometry_status"] == "dose-ineligible" for row in result["groups"])


def test_nonfinite_landing_is_saved_then_stops_before_any_control(monkeypatch):
    original = runner.plan_group_geometry
    infinity = struct.pack("<2I", 0x7f800000, 0)
    def with_nonfinite(*args, **kwargs):
        plan = original(*args, **kwargs)
        plan["status"] = "incomplete"
        item = plan["landings"][("P1", "low", "orthogonal")]
        item.update(status="incomplete", failure="nonfinite rounded landing", value=infinity)
        return plan
    monkeypatch.setattr(runner, "plan_group_geometry", with_nonfinite)
    writer, model = MemoryJournal(), FakeModel().eval()
    with pytest.raises(RuntimeError, match="nonfinite"):
        runner.run_fabricated(model, FakeTokenizer(), one_unit_groups(), writer,
                              subject="story", width=2, vocab_size=256)
    rows = event_rows(writer, "evaluation_return")
    assert len(rows) == 4 and all(row["action"]["kind"] == "clean" for row in rows)
    geometry = event_rows(writer, "group_geometry")[0]
    entry = next(item for item in geometry["landings"]
                 if (item["receiver"], item["dose"], item["arm"]) == ("P1", "low", "orthogonal"))
    assert writer.blobs[entry["value_ref"]["path"]] == infinity
    assert writer.terminal["status"] == "incomplete"


def test_empty_selection_causes_no_forward():
    writer, model = MemoryJournal(), FakeModel().eval()
    with pytest.raises(ValueError, match="selection"):
        runner.run_fabricated(model, FakeTokenizer(), [], writer,
                              subject="story", width=2, vocab_size=256)
    assert model.encoder_calls == model.decoder_calls == 0
    assert writer.terminal["status"] == "incomplete"


def test_plumbing_mismatch_immediately_stops_the_affected_pass():
    class NondeterministicFake(FakeModel):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            if self.decoder_calls == 17:
                output.logits[0, -1, 10] += 0.25
            return output
    writer, model = MemoryJournal(), NondeterministicFake().eval()
    with pytest.raises(RuntimeError, match="control did not match"):
        runner.run_fabricated(model, FakeTokenizer(), one_unit_groups(), writer,
                              subject="story", width=2, vocab_size=256)
    rows = event_rows(writer, "evaluation_return")
    assert len(rows) == 5 and rows[-1]["action"]["kind"] == "passthrough"
    assert model.decoder_calls == 20 and writer.terminal["status"] == "incomplete"


def interrupt_before_return(function):
    lines, start = inspect.getsourcelines(function)
    matches = [start + i for i, line in enumerate(lines) if line.strip() == "return result"]
    assert len(matches) == 1
    def trace(frame, event, arg):
        if event == "line" and frame.f_code is function.__code__ and frame.f_lineno == matches[0]:
            sys.settrace(None)
            raise KeyboardInterrupt("fabricated lost return")
        return trace
    return trace


def test_lost_evaluation_return_preserves_incomplete_and_stops():
    writer, model = MemoryJournal(), FakeModel().eval()
    try:
        sys.settrace(interrupt_before_return(runner._EvaluationRunner.evaluate))
        with pytest.raises(KeyboardInterrupt, match="lost return"):
            runner.run_fabricated(model, FakeTokenizer(), one_unit_groups(), writer,
                                  subject="story", width=2, vocab_size=256)
    finally:
        sys.settrace(None)
    assert len(event_rows(writer, "evaluation_return")) == 1
    assert model.encoder_calls == 1 and writer.terminal["status"] == "incomplete"


def test_lost_final_return_cannot_turn_provisional_terminal_into_success():
    writer, model = MemoryJournal(), FakeModel().eval()
    try:
        sys.settrace(interrupt_before_return(runner._run_pass))
        with pytest.raises(KeyboardInterrupt, match="lost return"):
            runner.run_fabricated(model, FakeTokenizer(), one_unit_groups(), writer,
                                  subject="story", width=2, vocab_size=256)
    finally:
        sys.settrace(None)
    assert len(event_rows(writer, "evaluation_return")) == 145
    assert writer.terminal["status"] == "complete_candidate"
    # A separately observed successful direct-child exit is still required.


def real_test_journal(directory):
    limits = LimitsV2(subject="story", total_bytes=16 * 1024**2, metadata_bytes=8 * 1024**2,
                      failure_bytes=1024**2, forward_bytes=4096, max_forwards=3516, free_bytes=0)
    return EvidenceJournalV2(directory, {"schema": "fabricated-multisubtoken-pass"}, limits=limits)


def test_full_fabricated_pass_roundtrips_a_real_file_journal(tmp_path):
    writer = real_test_journal(tmp_path / "attempt")
    result = runner.run_fabricated(FakeModel().eval(), FakeTokenizer(), one_unit_groups(), writer,
                                   subject="story", width=2, vocab_size=256)
    events = [json.loads(line) for line in (writer.directory / "events.jsonl").read_text().splitlines()]
    rows = [item["detail"] for item in events if item["event"] == "evaluation_return"]
    assert len(rows) == 145 and len(result["groups"]) == 2
    for row in rows:
        for ref in row["all_logit_refs"]:
            raw = (writer.directory / ref["path"]).read_bytes()
            assert len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
    geometry = next(item["detail"] for item in events if item["event"] == "group_geometry")
    for item in geometry["orthogonal"]:
        for ref in (item["source_ref"], item["residual_ref"]):
            raw = (writer.directory / ref["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == ref["sha256"] and len(raw) == ref["bytes"]
    assert json.loads((writer.directory / "terminal.json").read_text())["status"] == "complete_candidate"


def test_nonfinite_proposal_is_durable_before_the_real_journal_stops(tmp_path, monkeypatch):
    original = runner.plan_group_geometry
    infinity = struct.pack("<2I", 0x7f800000, 0)
    def with_nonfinite(*args, **kwargs):
        plan = original(*args, **kwargs)
        plan["status"] = "incomplete"
        plan["landings"][("P1", "low", "orthogonal")].update(
            status="incomplete", failure="nonfinite rounded landing", value=infinity)
        return plan
    monkeypatch.setattr(runner, "plan_group_geometry", with_nonfinite)
    writer, model = real_test_journal(tmp_path / "attempt"), FakeModel().eval()
    with pytest.raises(RuntimeError, match="nonfinite"):
        runner.run_fabricated(model, FakeTokenizer(), one_unit_groups(), writer,
                              subject="story", width=2, vocab_size=256)
    events = [json.loads(line) for line in (writer.directory / "events.jsonl").read_text().splitlines()]
    record = next(item["detail"] for item in events if item["event"] == "group_geometry")
    entry = next(item for item in record["landings"]
                 if (item["receiver"], item["dose"], item["arm"]) == ("P1", "low", "orthogonal"))
    assert (writer.directory / entry["value_ref"]["path"]).read_bytes() == infinity
    assert model.encoder_calls == 4
    assert json.loads((writer.directory / "terminal.json").read_text())["status"] == "incomplete"
