"""Saved-development replay prevents selection substitution before evaluation."""
from copy import deepcopy

import pytest

import flan_weight_edit_cases as cases
import flan_weight_edit_handoff as handoff
import flan_weight_edit_study as study
from flan_weight_edit_study_test import FakeSession


def frozen_setup(journal):
    import hashlib
    import json
    import torch
    import flan_weight_edit_worker as worker
    raw = bytes(512 * 4)
    candidate_sha = hashlib.sha256(raw).hexdigest()
    original_sha = hashlib.sha256(b"original").hexdigest()
    entries = []
    arms = [("primary", 1.0), ("same", 1.0)] + [(name, dose) for name in study.CONTROLS for dose in study.DOSES]
    for name, dose in arms:
        entries.append({"arm": name, "dose": dose, "sha256": candidate_sha, "bytes_hex": raw.hex(),
                        "original_sha256": original_sha, "donor_sha256": candidate_sha,
                        "donor_token": 1465 if name == "broad" else 1287 if name == "same" else 16242,
                        "donor_storage": "lm_head" if name == "broad" else "shared",
                        "intended_norm": 1.0, "realized_norm": 1.0})
    direction = bytes(512 * 8)
    receipt = {"state_sha256": "0" * 64, "entries": entries,
               "random_direction_sha256": hashlib.sha256(direction).hexdigest(),
               "random_direction_hex": direction.hex()}
    raw_receipt = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    worker._checked(journal, "candidate-construction", lambda: None)
    journal.record("frozen_candidates", {"sha256": hashlib.sha256(raw_receipt).hexdigest(), "receipt": receipt})
    worker._checked(journal, "clean-probe", lambda:
                    journal.call("discovery-1-target", "clean-probe", lambda: torch.zeros(32128), save_logits=True,
                                 metadata={"arm": "clean-probe", "decoder_prefix": [0]}))
    return {(item["arm"], item["dose"]): item for item in entries}


class CaptureSession(FakeSession):
    def __init__(self):
        super().__init__()
        self.outcomes = []

    def evaluate(self, arm, case):
        result = super().evaluate(arm, case)
        self.outcomes.append(deepcopy(result))
        return result


def test_complete_development_is_replayed_and_forged_selection_is_rejected():
    session = CaptureSession()
    fixed = cases.fixed_texts()
    saved = study.development(fixed, session)
    checked = []
    def verify(case, arm, outcome):
        assert outcome["case_id"] == case["id"] and outcome["arm"] == arm
        checked.append(case["id"])
        return deepcopy(outcome)
    assert handoff.replay_selection(fixed, saved, session.outcomes, verify) == saved
    assert len(checked) == 68
    altered = deepcopy(saved)
    altered["selection"]["broad"]["dose"] = 4.0
    with pytest.raises(ValueError, match="development replay"):
        handoff.replay_selection(fixed, altered, session.outcomes, verify)


def test_typed_json_equality_does_not_treat_boolean_as_number():
    assert handoff.typed_equal({"ok": True}, {"ok": True})
    assert not handoff.typed_equal({"ok": True}, {"ok": 1})
    with pytest.raises(ValueError, match="duplicate"):
        handoff.strict_json('{"x": true, "x": 1}')


def test_uneosed_greedy_sequence_must_reach_the_fixed_limit():
    with pytest.raises(ValueError, match="stopped early"):
        handoff.validate_greedy_termination([2841])
    handoff.validate_greedy_termination([2841, 1])
    handoff.validate_greedy_termination([2841, 2841, 2841, 2841])


def test_saved_fabricated_logits_replay_and_corruption_is_rejected(tmp_path):
    import json
    import flan_weight_edit_journal as evidence
    import flan_weight_edit_worker as worker
    from flan_weight_edit_worker_test import FakeModel, FakeTokenizer

    fixed = cases.fixed_texts()
    for case in fixed:
        case["input_ids"] = [4, 1]
        case["attention_mask"] = [1, 1]
        case["expected_ids"] = [1465, 1] if case["expected"] == "positive" else [2841, 1]
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=100,
                               max_bytes=20_000_000, reserve=1024)
    frozen_setup(journal)
    class Session:
        def score(self, name, function):
            return worker._checked(journal, name, function)
        def evaluate(self, arm, case):
            result = worker.evaluate_case(FakeModel(), FakeTokenizer(), case, journal, arm,
                                          labels={"positive": [1465, 1], "negative": [2841, 1]})
            journal.record("case_complete", {"arm": arm, "outcome": result})
            return result
    saved = study.development(fixed, Session())
    assert saved["status"] == "ineligible"
    journal.close()
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(saved), encoding="utf8")
    logit_path = tmp_path / "attempt" / "logits.f32"
    event_path = tmp_path / "attempt" / "events.jsonl"
    replay = handoff.verify_development(fixed, FakeTokenizer(), result_path, event_path, logit_path)
    assert replay["case_count"] == 8 and replay["result"]["status"] == "ineligible"
    original_size = logit_path.stat().st_size
    with logit_path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="unreferenced"):
        handoff.verify_development(fixed, FakeTokenizer(), result_path, event_path, logit_path)
    renamed = logit_path.with_name("renamed-logits.f32")
    logit_path.rename(renamed)
    renamed.rename(logit_path)
    with logit_path.open("r+b") as stream:
        stream.truncate(original_size)
    raw_events = event_path.read_text(encoding="utf8")
    events = [json.loads(line) for line in raw_events.splitlines()]
    first = next(i for i, event in enumerate(events) if event["kind"] == "forward_started" and event["call"] == 0)
    second = next(i for i, event in enumerate(events) if event["kind"] == "forward_started" and event["call"] == 1)
    events.insert(first + 1, events.pop(second))
    for i, event in enumerate(events):
        event["event_index"] = i
    event_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="forwards overlap"):
        handoff.verify_development(fixed, FakeTokenizer(), result_path, event_path, logit_path)
    event_path.write_text(raw_events, encoding="utf8")
    with logit_path.open("r+b") as stream:
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="logit bytes"):
        handoff.verify_development(fixed, FakeTokenizer(), result_path, event_path, logit_path)


def test_full_width_fabricated_complete_grid_replays_all_arms(tmp_path):
    from contextlib import contextmanager
    import json
    from types import SimpleNamespace
    import numpy as np
    import torch
    import flan_weight_edit_journal as evidence
    import flan_weight_edit_worker as worker
    from flan_weight_edit_worker_test import FakeTokenizer

    fixed = cases.fixed_texts()
    for case in fixed:
        case["input_ids"] = [4, 1]
        case["attention_mask"] = [1, 1]
        case["expected_ids"] = [1465, 1] if case["expected"] == "positive" else [2841, 1]
    journal = evidence.Journal(tmp_path / "attempt", vocabulary=32128, max_calls=1000,
                               max_bytes=80_000_000, reserve=1024)
    candidates = frozen_setup(journal)
    class Model:
        def __init__(self, negative):
            self.negative = negative
            self.encoder = self
        def __call__(self, **kwargs):
            if "input_ids" in kwargs:
                return SimpleNamespace(last_hidden_state=torch.zeros((1, kwargs["input_ids"].shape[1], 3)))
            prefix = kwargs["decoder_input_ids"][0].tolist()
            rows = np.zeros((len(prefix), 32128), dtype=np.float32)
            rows[0, 2841 if self.negative else 1465] = 4
            rows[0, 1465 if self.negative else 2841] = 2
            if len(prefix) > 1:
                rows[1:, 1] = 4
            return SimpleNamespace(logits=torch.from_numpy(rows[None, :, :]))
    class Session:
        @contextmanager
        def arm(self, name, dose):
            item = candidates[(name, dose)]
            journal.record("arm_started", {"arm": name, "dose": dose, "state_sha256": "0" * 64,
                                           "buffer_sha256": "1" * 64, "original_sha256": item["original_sha256"],
                                           "donor_sha256": item["donor_sha256"]})
            journal.record("arm_installed", {"arm": name, "dose": dose, "sha256": item["sha256"]})
            yield
            worker._checked(journal, "clean-probe", lambda:
                            journal.call("discovery-1-target", "clean-probe", lambda: torch.zeros(32128), save_logits=True,
                                         metadata={"arm": "clean-probe", "decoder_prefix": [0]}))
            journal.record("arm_completed", {"arm": name, "dose": dose,
                                              "probe_span": journal.last_record.copy(),
                                              "installed_sha256": item["sha256"],
                                              "restored_row_sha256": item["original_sha256"],
                                              "pre_buffer_sha256": "1" * 64,
                                              "post_probe_buffer_sha256": "1" * 64,
                                              "post_restore_state_sha256": "0" * 64,
                                              "topology_unchanged": True})
        def score(self, name, function):
            return worker._checked(journal, name, function)
        def evaluate(self, arm, case):
            negative = case["expected"] == "negative" or arm not in ("baseline", "same")
            result = worker.evaluate_case(Model(negative), FakeTokenizer(), case, journal, arm,
                                          labels={"positive": [1465, 1], "negative": [2841, 1]})
            journal.record("case_complete", {"arm": arm, "outcome": result})
            return result
    saved = study.development(fixed, Session())
    assert saved["status"] == "complete" and all(item["matched"] for item in saved["selection"].values())
    journal.close()
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(saved), encoding="utf8")
    replay = handoff.verify_development(fixed, FakeTokenizer(), result_path,
                                        tmp_path / "attempt" / "events.jsonl", tmp_path / "attempt" / "logits.f32")
    assert replay["case_count"] == 68
    event_path = tmp_path / "attempt" / "events.jsonl"
    original = event_path.read_text(encoding="utf8")
    events = [json.loads(line) for line in original.splitlines()]
    effects = [i for i, event in enumerate(events) if event["kind"] == "numeric_probe"
               and event["metadata"]["stage"] == "effect"][:2]
    assert len(effects) == 2
    events = [event for i, event in enumerate(events) if i not in effects]
    for i, event in enumerate(events):
        event["event_index"] = i
    event_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="development replay"):
        handoff.verify_development(fixed, FakeTokenizer(), result_path, event_path,
                                   tmp_path / "attempt" / "logits.f32")
    events = [json.loads(line) for line in original.splitlines()]
    installed = next(event for event in events if event["kind"] == "arm_installed")
    installed["metadata"]["sha256"] = "f" * 64
    event_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="installed row"):
        handoff.verify_development(fixed, FakeTokenizer(), result_path, event_path,
                                   tmp_path / "attempt" / "logits.f32")
    events = [json.loads(line) for line in original.splitlines()]
    restored = next(event for event in events if event["kind"] == "arm_completed")
    restored["metadata"]["post_restore_state_sha256"] = "f" * 64
    event_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="restoration record"):
        handoff.verify_development(fixed, FakeTokenizer(), result_path, event_path,
                                   tmp_path / "attempt" / "logits.f32")
