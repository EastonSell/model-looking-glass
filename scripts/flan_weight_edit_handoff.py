"""Replay the fixed development selection from saved full-vocabulary rows."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import torch

import flan_weight_edit_numerical as science
import flan_weight_edit_study as study
import flan_weight_edit_worker as worker


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def typed_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_greedy_termination(ids):
    if not 1 <= len(ids) <= 4 or 1 in ids[:-1] or (ids[-1] != 1 and len(ids) != 4):
        raise ValueError("development greedy decoding stopped early or continued after EOS")


def replay_selection(cases, saved, outcomes, verify, expected_arms=None, expected_numerics=None):
    class Replay:
        def __init__(self):
            self.index = 0
            self.arms = []
            self.numerics = ["candidate-construction", "clean-probe"]
        @contextmanager
        def arm(self, name, dose):
            self.arms.append((name, dose))
            yield
            self.numerics.append("clean-probe")
        def score(self, name, function):
            self.numerics.append(name)
            return function()
        def evaluate(self, arm, case):
            if self.index >= len(outcomes):
                raise ValueError("development replay lacks a case")
            value = verify(case, arm, outcomes[self.index])
            self.index += 1
            return value
    replay = Replay()
    result = study.development(cases, replay)
    if (replay.index != len(outcomes) or not typed_equal(result, saved)
            or (expected_arms is not None and not typed_equal(replay.arms, expected_arms))
            or (expected_numerics is not None and not typed_equal(replay.numerics, expected_numerics))):
        raise ValueError("development replay differs from published result")
    return result


def verify_development(cases, tokenizer, result_path, event_path, logit_path):
    """Recompute each saved case and complete fixed grid; never loads a model."""
    saved = strict_json(Path(result_path).read_text(encoding="utf8"))
    events = [strict_json(line) for line in Path(event_path).read_text(encoding="utf8").splitlines()]
    if any(type(event.get("event_index")) is not int or event["event_index"] != i for i, event in enumerate(events)):
        raise ValueError("development journal ordering differs")
    starts = [event for event in events if event.get("kind") == "forward_started"]
    complete = [event for event in events if event.get("kind") == "forward_complete"]
    if (len(starts) != len(complete) or any(type(x.get("call")) is not int for x in starts + complete)
            or [x["call"] for x in starts] != list(range(len(starts)))
            or [x["call"] for x in complete] != list(range(len(starts)))):
        raise ValueError("development journal contains an incomplete or unordered call")
    by_call = {event["call"]: {k: v for k, v in event.items() if k != "event_index"} for event in complete}
    outcomes = [event["metadata"]["outcome"] for event in events if event.get("kind") == "case_complete"]
    used = set()
    logit_size = Path(logit_path).stat().st_size
    with Path(logit_path).open("rb") as stream:
        pending, arm, installed, offset, last_complete = None, None, False, 0, None
        candidates, base_state, baseline_probe, arm_start = None, None, None, None
        baseline_buffer, probe_count, arm_probe_count = None, 0, None
        active_numeric, numeric_calls, case_numeric, noncase_numeric = None, 0, [], []
        arms = []
        for event in events:
            kind = event.get("kind")
            if kind == "numeric_probe":
                metadata = event["metadata"]
                if (type(metadata.get("subnormal_bits")) is not int or metadata["subnormal_bits"] != 1
                        or type(metadata.get("midpoint_even")) is not float or metadata["midpoint_even"] != 1.0
                        or type(metadata.get("midpoint_odd")) is not float or metadata["midpoint_odd"] != 1.0 + 2.0 ** -22
                        or type(metadata.get("stage")) is not str or metadata.get("when") not in ("before", "after")):
                    raise ValueError("development numeric-mode observation differs")
                identity = {k: v for k, v in metadata.items() if k not in ("when", "subnormal_bits", "midpoint_even", "midpoint_odd")}
                if metadata["when"] == "before":
                    if active_numeric is not None:
                        raise ValueError("development numeric probes overlap")
                    active_numeric = identity
                    numeric_calls = 0
                else:
                    if active_numeric is None or not typed_equal(identity, active_numeric) or pending is not None:
                        raise ValueError("development numeric probes are unpaired")
                    stage = identity["stage"]
                    is_forward = stage in ("encoder", "clean-probe") or stage.startswith(("greedy-", "teacher-"))
                    if numeric_calls != (1 if is_forward else 0):
                        raise ValueError("development numeric pair has an unexpected model-call count")
                    if "case_id" in identity:
                        case_numeric.append(identity)
                    else:
                        noncase_numeric.append(stage)
                    active_numeric = None
            elif kind == "frozen_candidates":
                if candidates is not None or pending is not None or arm is not None:
                    raise ValueError("development candidate receipt ordering differs")
                metadata = event["metadata"]
                receipt = metadata["receipt"]
                encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
                if hashlib.sha256(encoded).hexdigest() != metadata["sha256"]:
                    raise ValueError("development candidate receipt differs")
                base_state = receipt["state_sha256"]
                if type(base_state) is not str or len(base_state) != 64:
                    raise ValueError("development base-state identity differs")
                candidates = {}
                for item in receipt["entries"]:
                    key = (item["arm"], item["dose"])
                    raw = bytes.fromhex(item["bytes_hex"])
                    if (key in candidates or len(raw) != 512 * 4 or not np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
                            or hashlib.sha256(raw).hexdigest() != item["sha256"]
                            or any(type(item.get(field)) is not str or len(item[field]) != 64
                                   for field in ("original_sha256", "donor_sha256"))):
                        raise ValueError("development frozen candidate differs")
                    candidates[key] = item
                required = {("primary", 1.0), ("same", 1.0)} | {(name, dose) for name in study.CONTROLS for dose in study.DOSES}
                if set(candidates) != required:
                    raise ValueError("development candidate grid differs")
                shared_original = candidates[("primary", 1.0)]["original_sha256"]
                head_original = candidates[("broad", study.DOSES[0])]["original_sha256"]
                for (name, dose), item in candidates.items():
                    if item["original_sha256"] != (head_original if name == "broad" else shared_original):
                        raise ValueError("development candidate original row differs")
                    expected_token = 1465 if name == "broad" else 1287 if name == "same" else 16242
                    if item["donor_token"] != expected_token or item["donor_storage"] != ("lm_head" if name == "broad" else "shared"):
                        raise ValueError("development candidate donor identity differs")
                    if name in ("primary", "same") and item["sha256"] != item["donor_sha256"]:
                        raise ValueError("development donor replacement differs")
                direction = bytes.fromhex(receipt["random_direction_hex"])
                if len(direction) != 512 * 8 or hashlib.sha256(direction).hexdigest() != receipt["random_direction_sha256"]:
                    raise ValueError("development random direction differs")
            elif kind == "forward_started":
                if pending is not None:
                    raise ValueError("development forwards overlap")
                if active_numeric is None or active_numeric.get("stage") != event.get("stage"):
                    raise ValueError("development forward lacks its numeric probe")
                numeric_calls += 1
                if "case_id" in active_numeric and (active_numeric["case_id"] != event["case_id"]
                        or active_numeric.get("arm") != event["metadata"].get("arm")):
                    raise ValueError("development forward numeric context differs")
                if candidates is None:
                    raise ValueError("development forward precedes candidate receipt")
                name = event["metadata"].get("arm")
                expected_arm = "baseline" if arm is None else (arm[0] if arm[0] in ("primary", "same") else f"{arm[0]}@{arm[1]}")
                if name not in (expected_arm, "clean-probe"):
                    raise ValueError("development forward is outside its arm")
                if name == "clean-probe" and (event["case_id"] != "discovery-1-target"
                        or event["stage"] != "clean-probe" or event["metadata"].get("decoder_prefix") != [0]):
                    raise ValueError("development clean-probe identity differs")
                pending = event
            elif kind == "forward_complete":
                if pending is None or any(not typed_equal(event.get(key), pending.get(key))
                                          for key in ("call", "case_id", "stage", "metadata")):
                    raise ValueError("development forward completion differs from start")
                pending = None
                if "offset" in event:
                    if (type(event["offset"]) is not int or event["offset"] != offset
                            or type(event.get("bytes")) is not int or event["bytes"] <= 0
                            or offset + event["bytes"] > logit_size):
                        raise ValueError("development logit spans are not contiguous")
                    stream.seek(offset)
                    raw = stream.read(event["bytes"])
                    if (hashlib.sha256(raw).hexdigest() != event.get("sha256")
                            or len(raw) % 4 or not np.isfinite(np.frombuffer(raw, dtype="<f4")).all()):
                        raise ValueError("development logit bytes differ")
                    offset += event["bytes"]
                last_complete = {k: v for k, v in event.items() if k != "event_index"}
                if event["stage"] == "clean-probe":
                    if event.get("shape") != [32128] or event.get("bytes") != 32128 * 4:
                        raise ValueError("development clean-probe shape differs")
                    if baseline_probe is None:
                        if arm is not None:
                            raise ValueError("development initial probe follows an edit")
                        baseline_probe = event["sha256"]
                    elif arm is None or not installed or event["sha256"] != baseline_probe:
                        raise ValueError("development clean-probe bytes differ")
                    probe_count += 1
            elif kind == "arm_started":
                if pending is not None or arm is not None or candidates is None or baseline_probe is None:
                    raise ValueError("development arm overlaps another operation")
                arm = (event["metadata"]["arm"], event["metadata"]["dose"])
                if arm not in candidates:
                    raise ValueError("development arm is not a frozen candidate")
                arm_start = event["metadata"]
                item = candidates[arm]
                if (arm_start.get("state_sha256") != base_state
                        or arm_start.get("original_sha256") != item["original_sha256"]
                        or arm_start.get("donor_sha256") != item["donor_sha256"]
                        or type(arm_start.get("buffer_sha256")) is not str or len(arm_start["buffer_sha256"]) != 64):
                    raise ValueError("development arm original-state receipt differs")
                if baseline_buffer is None:
                    baseline_buffer = arm_start["buffer_sha256"]
                elif arm_start["buffer_sha256"] != baseline_buffer:
                    raise ValueError("development baseline buffer identity differs")
                arm_probe_count = probe_count
                installed = False
                arms.append(arm)
            elif kind == "arm_installed":
                if pending is not None or arm is None or installed or not typed_equal(
                        (event["metadata"]["arm"], event["metadata"]["dose"]), arm):
                    raise ValueError("development arm install differs")
                if event["metadata"].get("sha256") != candidates[arm]["sha256"]:
                    raise ValueError("development installed row differs from frozen candidate")
                installed = True
            elif kind == "arm_completed":
                if (pending is not None or arm is None or not installed or not typed_equal(
                        (event["metadata"]["arm"], event["metadata"]["dose"]), arm)
                        or not typed_equal(event["metadata"].get("probe_span"), last_complete)
                        or last_complete is None or last_complete.get("stage") != "clean-probe"
                        or event["metadata"].get("installed_sha256") != candidates[arm]["sha256"]
                        or event["metadata"].get("restored_row_sha256") != candidates[arm]["original_sha256"]
                        or event["metadata"].get("post_restore_state_sha256") != base_state
                        or event["metadata"].get("pre_buffer_sha256") != arm_start["buffer_sha256"]
                        or event["metadata"].get("post_probe_buffer_sha256") != arm_start["buffer_sha256"]
                        or probe_count != arm_probe_count + 1
                        or event["metadata"].get("topology_unchanged") is not True):
                    raise ValueError("development arm restoration record differs")
                arm, installed = None, False
            elif kind == "case_complete":
                expected_arm = "baseline" if arm is None else (arm[0] if arm[0] in ("primary", "same") else f"{arm[0]}@{arm[1]}")
                if (pending is not None or (arm is not None and not installed)
                        or event["metadata"].get("arm") != expected_arm
                        or event["metadata"]["outcome"].get("arm") != expected_arm):
                    raise ValueError("development case completion is out of order")
                outcome = event["metadata"]["outcome"]
                stages = (["encoder"] + [f"greedy-{i}" for i in range(len(outcome["greedy_spans"]))]
                          + ["teacher-positive", "score-positive", "teacher-negative", "score-negative", "score-summary"])
                required = [{"arm": expected_arm, "case_id": outcome["case_id"], "stage": stage} for stage in stages]
                if active_numeric is not None or not typed_equal(case_numeric, required):
                    raise ValueError("development case numeric observations differ")
                case_numeric = []
            else:
                raise ValueError("development journal contains an unknown or failure event")
        if (pending is not None or arm is not None or active_numeric is not None or case_numeric
                or offset != logit_size or candidates is None or baseline_probe is None or probe_count != 1 + len(arms)):
            raise ValueError("development journal has incomplete work or unreferenced logit bytes")
        def read(span, case_id, arm, stage, prefix, rows):
            call = span.get("call")
            if (type(call) is not int or call in used or not typed_equal(by_call.get(call), span)
                    or span.get("case_id") != case_id or span.get("stage") != stage
                    or not typed_equal(span.get("metadata"), {"arm": arm, "decoder_prefix": prefix})
                    or not typed_equal(span.get("shape"), ([32128] if rows == 1 else [rows, 32128]))
                    or type(span.get("offset")) is not int or span["offset"] < 0
                    or span.get("bytes") != rows * 32128 * 4
                    or span["offset"] + span["bytes"] > logit_size):
                raise ValueError("development logit span or call identity differs")
            used.add(call)
            stream.seek(span["offset"])
            raw = stream.read(span["bytes"])
            if len(raw) != span["bytes"] or hashlib.sha256(raw).hexdigest() != span.get("sha256"):
                raise ValueError("development logit bytes differ")
            values = np.frombuffer(raw, dtype="<f4").reshape((rows, 32128)).copy()
            if not np.isfinite(values).all():
                raise ValueError("development logits are nonfinite")
            return torch.from_numpy(values)
        def verify(case, arm, outcome):
            if outcome.get("case_id") != case["id"] or outcome.get("arm") != arm:
                raise ValueError("development case identity differs")
            prefix, ids, tops = [0], [], []
            spans = outcome.get("greedy_spans")
            if not isinstance(spans, list) or not 1 <= len(spans) <= 4:
                raise ValueError("development greedy record differs")
            for index, span in enumerate(spans):
                row = read(span, case["id"], arm, f"greedy-{index}", prefix, 1).numpy()[0]
                ties = np.flatnonzero(row == np.max(row)).tolist()
                token = int(ties[0])
                tops.append({"representative": token, "tied_ids": ties, "unique": len(ties) == 1})
                ids.append(token)
                prefix = [*prefix, token]
                if token == 1 and index != len(spans) - 1:
                    raise ValueError("development greedy continues after EOS")
            terminated = ids[-1] == 1
            validate_greedy_termination(ids)
            content = ids[:-1] if terminated else ids
            decoded = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            normalize = lambda text: unicodedata.normalize("NFC", text).strip().casefold()
            strict = terminated and all(top["unique"] for top in tops) and not any(x in tokenizer.all_special_ids for x in content)
            correct = bool(strict and normalize(decoded) == normalize(case["expected"]))
            label = normalize(decoded) if strict and normalize(decoded) in ("positive", "negative") else None
            terms = {}
            for name, target in (("positive", [1465, 1]), ("negative", [2841, 1])):
                logits = read(outcome["label_spans"][name], case["id"], arm, "teacher-" + name,
                              [0, target[0]], 2)
                terms[name] = worker._score_terms(logits, target)
            checks = {"greedy_ids": ids, "tops": tops, "terminated": terminated, "decoded": decoded,
                      "strict_correct": correct, "strict_label": label, "label_terms": terms,
                      "candidate_terms": None, "candidate_span": None,
                      "margin": science.sequence_effect(terms["negative"], terms["positive"]),
                      "negative_score": science.sequence_score(terms["negative"])}
            if any(not typed_equal(outcome.get(key), value) for key, value in checks.items()):
                raise ValueError("development outcome differs from saved logits")
            return outcome
        result = replay_selection(cases, saved, outcomes, verify, expected_arms=arms,
                                  expected_numerics=noncase_numeric)
        expected_logits = {item["call"] for item in complete if "offset" in item and item["stage"] != "clean-probe"}
        if used != expected_logits:
            raise ValueError("development contains an unreferenced case logit call")
        return {"result": result, "calls": len(starts), "case_count": len(outcomes),
                "referenced_case_logit_calls": len(used)}
