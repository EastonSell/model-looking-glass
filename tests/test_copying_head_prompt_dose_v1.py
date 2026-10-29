"""Synthetic schedule, bit-landing and recovery checks with no model forward."""
import json
import inspect
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from looking_glass.copy_head_prompt_dose import SHAPE_BYTES, plan_prompt
from scripts import copying_head_prompt_dose_v1 as owner
from scripts import copying_head_prompt_dose_bootstrap_v1 as bootstrap


def witness(zero=False):
    values = [0.0] * (SHAPE_BYTES // 4)
    for head, value in ((2, 0.0 if zero else 1.0), (10, 2.0), (7, 4.0)):
        values[10 * 768 + 64 * head] = value
    return struct.pack(f"<{len(values)}f", *values)


def test_fabricated_192_forward_schedule_and_floor_ineligibility():
    calls, plans = [], []
    def perform(case, kind, cache, head=None, delivery=None):
        calls.append((case["case_id"], kind))
        if kind == "noop":
            cache["before"] = witness()
        if head is not None:
            assert len(delivery) == 256
            cache.setdefault("comparisons", {})[kind] = {"target_log_probability_delta": 0.0, "target_logit_delta": 0.0}
    for group in range(1, 9):
        for variant in range(4):
            case = {"case_id": f"{group}-{variant}", "group": group, "variant": variant}
            result = owner.execute_case(case, perform, lambda _: None,
                                        lambda _case, plan: plans.append(plan))
            assert result["status"] == "eligible"
    assert len(calls) == 192 and len(plans) == 32
    for i in range(32):
        actions = [action for _, action in calls[6 * i:6 * (i + 1)]]
        assert actions[:2] == ["clean", "noop"]
        assert actions[-1] == "clean_after"
        assert set(actions[2:5]) == {"h2", "h10", "h7"}
    calls.clear()
    def floor_perform(case, kind, cache, head=None, delivery=None):
        calls.append(kind)
        if kind == "noop": cache["before"] = witness(zero=True)
    result = owner.execute_case({"case_id": "floor", "group": 1, "variant": 0},
                                floor_perform, lambda _: None, lambda _case, _plan: None)
    assert calls == ["clean", "noop", "clean_after"]
    assert result["status"] == "dose-ineligible"
    assert result["plan"]["plan_unavailable"] == "plan unavailable: dose floor"


def test_failed_arm_propagates_without_a_complete_result():
    calls = []
    def perform(case, kind, cache, head=None, delivery=None):
        calls.append(kind)
        if kind == "noop": cache["before"] = witness()
        if kind == "h10": raise RuntimeError("fabricated forward failure")
    with pytest.raises(RuntimeError, match="fabricated"):
        owner.execute_case({"case_id": "fail", "group": 1, "variant": 0},
                           perform, lambda _: None, lambda _case, _plan: None)
    assert "clean_after" not in calls


def test_complete_landing_and_untouched_coordinates():
    before = witness()
    plan = plan_prompt(before)
    delivery = plan["plans"][10]["delivered"]
    start = (10 * 768 + 10 * 64) * 4
    after = bytearray(before)
    after[start:start + 256] = delivery
    payload = {"logits": struct.pack("<f", 1.0), "cproj_before": before,
               "cproj_after": bytes(after), "cproj_output": before}
    owner.validate_returned_payload(payload, "h10", {"before": before}, 10, delivery)
    after[0] ^= 1
    payload["cproj_after"] = bytes(after)
    with pytest.raises(RuntimeError, match="untouched"):
        owner.validate_returned_payload(payload, "h10", {"before": before}, 10, delivery)


def test_partial_returned_payload_is_completed_once_and_journaled(tmp_path):
    path = tmp_path / "raw.f32"
    path.write_bytes(b"priorabc")
    journal = tmp_path / "attempt.jsonl"
    payload = {"logits": b"abcdef", "cproj_before": b"XYZ"}
    with journal.open("wb") as log:
        assert owner.recover_payload(log, path, "c1", "noop", payload, 5, True)
    assert path.read_bytes() == b"priorabcdefXYZ"
    record = json.loads(journal.read_bytes())
    assert record["event"] == "raw_recorded_recovered"
    assert record["raw"]["logits"]["offset"] == 5
    assert record["raw"]["cproj_before"]["offset"] == 11
    assert record["model_call_return_observed"] is True
    bad = tmp_path / "bad.f32"
    bad.write_bytes(b"priorwrong")
    with journal.open("ab") as log, pytest.raises(RuntimeError, match="differs"):
        owner.recover_payload(log, bad, "c2", "noop", payload, 5, True)
    assert bad.read_bytes() == b"priorwrong"


@pytest.mark.parametrize("next_line", [
    'observation["helper_return_observed"] = True',
    'counts["returned"] = before_returned + 1',
    'if counts["retained"] > MAX_RAW_BYTES:',
])
def test_interruption_after_return_or_admission_does_not_double_count(tmp_path, monkeypatch, next_line):
    raw = struct.pack(f"<{owner.base.VOCAB}f", *([0.0] * owner.base.VOCAB))
    def fake_forward(_model, _ids, _kind, *, expected_before, deliveries, observation):
        observation.update({"model_call_return_observed": True, "payload": {"logits": raw}})
        return observation["payload"]
    monkeypatch.setattr(owner.prior, "run_forward", fake_forward)
    monkeypatch.setattr(owner.prior, "runtime", lambda _model: "fixed")
    source, start = inspect.getsourcelines(owner.perform_action)
    matches = [start + i for i, line in enumerate(source) if line.strip() == next_line]
    assert len(matches) == 1
    target_line, fired = matches[0], []
    def trace(frame, event, arg):
        if frame.f_code is owner.perform_action.__code__ and event == "line" and frame.f_lineno == target_line:
            fired.append(True)
            sys.settrace(None)
            raise KeyboardInterrupt("injected post-return interruption")
        return trace
    counts = {"submitted": 10, "model_returns": 5, "returned": 4, "retained": 123}
    activity, completed = {}, []
    case = {"case_id": "synthetic", "token_ids": list(range(11)), "target_id": 5,
            "group": 1, "variant": 0}
    with (tmp_path / "attempt.jsonl").open("wb") as log:
        try:
            sys.settrace(trace)
            with pytest.raises(KeyboardInterrupt, match="injected"):
                owner.perform_action(case, "clean", {}, None, None, model=None,
                    runtime_before="fixed", output=tmp_path, log=log,
                    counts=counts, activity=activity, completed=completed)
        finally:
            sys.settrace(None)
    assert fired == [True]
    assert counts == {"submitted": 11, "model_returns": 6, "returned": 5,
                      "retained": 123 + len(raw)}
    assert activity["observation"]["model_call_return_observed"] is True
    assert activity["observation"]["helper_return_observed"] is True
    assert activity["marker"]["partial"] is True
    assert (tmp_path / "synthetic.f32").read_bytes() == raw
    events = [json.loads(line) for line in (tmp_path / "attempt.jsonl").read_bytes().splitlines()]
    assert [event["event"] for event in events] == ["start", "raw_recorded_recovered"]
    assert not completed


def test_startup_refuses_missing_or_dirty_bytecode_directory_before_torch(tmp_path):
    poison = tmp_path / "torch.py"
    marker = tmp_path / "torch-imported"
    poison.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="utf8")
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(tmp_path), "CUDA_VISIBLE_DEVICES": "-1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    env.pop("PYTHONPYCACHEPREFIX", None)
    script = Path(owner.__file__)
    missing = subprocess.run([sys.executable, "-B", str(script), "--help"], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert missing.returncode != 0 and not marker.exists()
    cache = tmp_path / "dirty-cache"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"x")
    dirty = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={cache}", str(script), "--help"],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert dirty.returncode != 0 and not marker.exists()


def test_bootstrap_checks_complete_source_before_any_project_import(tmp_path, monkeypatch):
    root = tmp_path
    manifest_path = root / bootstrap.MANIFEST
    manifest_path.parent.mkdir(parents=True)
    source_file = root / "src.py"
    original = b"trusted_source\n"
    source_file.write_bytes(original)
    manifest = {"schema": "distilgpt2-copy-head-prompt-dose-manifest-v1",
                "source_revision": "a" * 40, "source_tree": "b" * 40,
                "source_sha256": {"src.py": hashlib.sha256(original).hexdigest()}}
    manifest_bytes = json.dumps(manifest).encode()
    manifest_path.write_bytes(manifest_bytes)
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr(bootstrap, "REQUIRED", {bootstrap.MANIFEST, "src.py"})
    def fake_git(*args):
        if args[0] == "status": return b""
        if args[0] == "rev-parse": return ("c" * 40 + "\n").encode()
        if args[0] == "merge-base": return b""
        if args[0] == "diff": return (bootstrap.MANIFEST + "\n").encode()
        if args[:3] == ("show", "-s", "--format=%T"): return ("b" * 40 + "\n").encode()
        if args == ("show", "c" * 40 + ":" + bootstrap.MANIFEST): return manifest_bytes
        if args == ("show", "a" * 40 + ":src.py"): return original
        raise AssertionError(args)
    monkeypatch.setattr(bootstrap, "git", fake_git)
    assert bootstrap.verify_manifest(manifest_path) == hashlib.sha256(manifest_bytes).hexdigest()
    source_file.write_bytes(b"tampered_source\n")
    with pytest.raises(RuntimeError, match="before import"):
        bootstrap.verify_manifest(manifest_path)


def test_reservation_checks_own_ids_status_and_exact_predecessor(monkeypatch):
    from looking_glass import pretrained_exposures_v30 as predecessor, pretrained_exposures_v31 as reservation
    cases = [{"case_id": f"case-{i}", "token_ids": [i], "target_id": i + 100, "text": f"text-{i}"}
             for i in range(32)]
    old = {"records": [{"record_id": "old", "category": "planned", "token_ids": [400]}]}
    added = [{"record_id": "planned-" + case["case_id"], "category": "planned",
              "token_ids": case["token_ids"], "target_token_id": case["target_id"],
              "public_text": case["text"]} for case in cases]
    service = (("s1", (201, 202)), ("s2", (301, 302, 303)))
    added.extend({"record_id": name, "category": "planned", "token_ids": list(row),
                  "attention_mask": [1] * len(row)} for name, row in service)
    payload = {"records": [*old["records"], *added]}
    monkeypatch.setattr(predecessor, "_checked_inventory", lambda: (old, "old-digest"))
    monkeypatch.setattr(reservation, "_checked_inventory", lambda: (payload, "new-digest"))
    monkeypatch.setattr(reservation, "COPY_PROMPT_DOSE_ROWS", tuple(tuple(x["token_ids"]) for x in cases))
    monkeypatch.setattr(reservation, "SYNTHETIC_SERVICE_CASES", service)
    manifest = {"predecessor_inventory_sha256": "old-digest", "reservation_inventory_sha256": "new-digest"}
    assert owner.verify_reservation(cases, manifest)["reservation_rows"] == 35
    added[0]["category"] = "observed"
    with pytest.raises(RuntimeError, match="identity, input, label or status"):
        owner.verify_reservation(cases, manifest)
