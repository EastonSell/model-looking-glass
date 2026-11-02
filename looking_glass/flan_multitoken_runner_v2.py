"""Per-subject multi-subtoken CPU core, admitted by a separate owner.

The owner must independently bind and retokenize the published selection,
check source and assets, enforce the combined disk preflight, and observe
the direct child's exit. A complete terminal is only a saved candidate; a
separate scientific replay is required before any result classification.
"""

import copy
import math

from .flan_multitoken_geometry_v2 import plan_group_geometry
from .flan_multitoken_journal_v2 import EvidenceJournalV2, require_protocol_allocation
from .flan_multitoken_schedule_v2 import ARMS, OPPOSITE, RECEIVERS, potential_schedule
from .flan_multitoken_scoring_v2 import group_decision
from .flan_multitoken_span_v2 import validate_aligned_group
from .flan_t5_sentiment_numeric import _norm, runtime_numeric_probe
from .flan_t5_sentiment_runner import (
    _EvaluationRunner, _encode_case, _label_targets, _natural_geometry, _patched, _raw,
    _record_comparison, _same_output, _values,
)


def _validated_pass(groups, *, subject, vocab_size, special_ids):
    if type(groups) not in (list, tuple) or not groups or len(groups) > 8:
        raise ValueError("selection must provide nonempty bounded groups")
    groups = copy.deepcopy(groups)
    selected, sites = {}, {}
    for group in groups:
        if (type(group) is not dict or set(group) != {"unit", "instruction", "cases"}
                or type(group["unit"]) is not int or not 0 <= group["unit"] < 4
                or type(group["instruction"]) is not int or group["instruction"] not in (0, 1)
                or (subject == "painting" and group["unit"] not in (0, 2))
                or type(group["cases"]) is not dict or set(group["cases"]) != set(RECEIVERS)):
            raise ValueError("invalid selected group")
        key = (group["unit"], group["instruction"])
        if key in selected:
            raise ValueError("duplicate group")
        for case in group["cases"].values():
            if (type(case) is not dict or case.get("subject") != subject
                    or type(case.get("text")) is not str or len(case["text"].encode("utf8")) > 256
                    or type(case.get("adjective")) is not str
                    or not 1 <= len(case["adjective"].encode("utf8")) <= 128):
                raise ValueError("invalid bounded case")
        count = 1 if group["unit"] < 2 else 2
        sites[key] = validate_aligned_group(group["cases"], lexical_count=count, boundary_count=count,
                                            vocab_size=vocab_size, special_ids=special_ids)
        selected[key] = group
    slots = tuple(sorted({unit for unit, _ in selected}))
    if set(selected) != {(unit, instruction) for unit in slots for instruction in (0, 1)}:
        raise ValueError("selected slots require both instruction groups")
    if tuple((g["unit"], g["instruction"]) for g in groups) != tuple(sorted(selected)):
        raise ValueError("selected groups are not in fixed order")
    for unit in slots:
        first, second = (selected[(unit, i)]["cases"] for i in (0, 1))
        if any(first[side]["adjective"] != second[side]["adjective"] for side in RECEIVERS):
            raise ValueError("instruction groups do not share the fixed words")
    potential_schedule(subject, slots)
    return selected, sites, slots


def _persist_geometry(journal, plan, *, unit, subject, instruction, width, before, after):
    orthogonal = []
    for receiver, item in plan["orthogonal"].items():
        row = {key: value for key, value in item.items() if key not in ("source", "residual")}
        row.update(receiver=receiver, source_ref=journal.vector64(item["source"]),
                   residual_ref=None if item["residual"] is None else journal.vector64(item["residual"]))
        orthogonal.append(row)
    landings = []
    for (receiver, dose, arm), item in plan["landings"].items():
        row = {key: value for key, value in item.items() if key != "value"}
        row.update(receiver=receiver, dose=dose, arm=arm,
                   value_ref=None if item["value"] is None else journal.tensor((width,), item["value"]))
        landings.append(row)
    record = {"unit": unit, "subject": subject, "instruction": instruction,
              "status": plan["status"], "q": plan["q"],
              "distances": [{"receiver": receiver, "arm": arm, "distance": value}
                            for (receiver, arm), value in plan["distances"].items()],
              "orthogonal": orthogonal, "landings": landings,
              "numeric_before": before, "numeric_after": after}
    journal.record("group_geometry", record)
    return {(item["receiver"], item["dose"], item["arm"]): item for item in landings}


def _diagnostics(journal, tokenizer, result):
    ties = []
    for step in result["greedy"]:
        values = _values(journal.read_tensor(step["logits"]))
        maximum = max(values)
        ties.append(sum(value == maximum for value in values) != 1)
    ids = result["generated_ids"]
    special = set(tokenizer.all_special_ids)
    detail = {"action": result["action"], "side": result["side"], "ties": ties,
              "special_positions": [i for i, value in enumerate(ids) if value in special],
              "terminated": bool(ids and ids[-1] == 1),
              "truncated": not bool(ids and ids[-1] == 1)}
    journal.record("greedy_diagnostics", detail)
    return detail


def _evaluate(evaluator, journal, tokenizer, case, before, after, action, targets):
    result = evaluator.evaluate(case, before, after, action, targets)
    diagnostics = _diagnostics(journal, tokenizer, result)
    return result, diagnostics


def _dose_outcome(clean, measured, landings, unit, instruction, dose):
    arms = {}
    for arm in ARMS:
        if all((unit, instruction, receiver, arm, dose) in measured
               and landings[(receiver, dose, arm)]["status"] == "matchable"
               and landings[(receiver, dose, arm)].get("relative_status")
                   == ("reference" if arm == "opposite_adjective" else "matched")
               for receiver in RECEIVERS):
            arms[arm] = {receiver: measured[(unit, instruction, receiver, arm, dose)]["margin"]
                         for receiver in RECEIVERS}
    if not all(arm in arms for arm in ARMS[:-1]):
        return {"status": "unmeasured", "B3": None, "B4": None,
                "passes_B3": None, "passes_B4": None}
    result = group_decision(clean, arms)
    return {"status": "measured" if result["B4"] is not None else "B3-only", **result}


def _run_pass(model, tokenizer, groups, journal, *, subject, width, vocab_size,
              test_only, custody_check):
    step = None
    try:
        if not test_only:
            if type(journal) is not EvidenceJournalV2 or journal.limits.subject != subject:
                raise ValueError("production requires the matching per-pass journal")
            require_protocol_allocation(journal.limits)
            journal._check()
            if journal.component_count != 0 or journal.active_component is not None:
                raise ValueError("production journal is not fresh")
            if not callable(custody_check):
                raise ValueError("production requires custody checks")
        if custody_check is not None:
            custody_check()
        selected, sites, slots = _validated_pass(groups, subject=subject, vocab_size=vocab_size,
                                                  special_ids=set(tokenizer.all_special_ids))
        targets = _label_targets(tokenizer, vocab_size=vocab_size)
        if not test_only and targets != {"positive": [1465, 1], "negative": [2841, 1]}:
            raise ValueError("pinned label tokenization differs")
        evaluator = _EvaluationRunner(model, tokenizer, journal, width=width, vocab_size=vocab_size,
                                      test_only=test_only, custody_check=custody_check)
        evaluator._check_runtime()
        journal.record("runtime_observation", {"testing_only": test_only, "subject": subject,
                                               "settings": evaluator.runtime,
                                               "numeric": runtime_numeric_probe()})
        journal.record("label_targets", {"targets": targets, "decoder_start_id": 0, "eos_id": 1})
        states, baseline, diagnostics, measured = {}, {}, {}, {}
        geometries, support = {}, {}
        for step in potential_schedule(subject, slots):
            kind, unit, instruction, side = step["kind"], step["unit"], step["instruction"], step["receiver"]
            key, group_key = (unit, instruction, side), (unit, instruction)
            group = selected[group_key]
            case = group["cases"][side]
            position = {"instruction": 0, "subject": case["subject_index"], "adjective": case["adjective_index"]}
            if kind == "clean":
                state, encoder_ref = _encode_case(model, case, journal, width, evaluator._check_runtime)
                states[key] = state
                action = dict(step, encoder_state=encoder_ref)
                baseline[key], diagnostics[key] = _evaluate(evaluator, journal, tokenizer, case, state,
                                                           state.clone(), action, targets)
                if side == RECEIVERS[-1]:
                    rows = {receiver: {site: _raw(states[(unit, instruction, receiver)][0, index])
                                       for site, index in position.items()} for receiver in RECEIVERS}
                    numeric_before = runtime_numeric_probe()
                    plan = plan_group_geometry(rows, unit=unit, subject=subject, instruction=instruction)
                    numeric_after = runtime_numeric_probe()
                    landings = _persist_geometry(journal, plan, unit=unit, subject=subject,
                                                 instruction=instruction, width=width,
                                                 before=numeric_before, after=numeric_after)
                    geometries[group_key] = {"status": plan["status"], "landings": landings}
                    if plan["status"] == "incomplete":
                        raise RuntimeError("nonfinite geometry landing")
                    support[group_key] = all(
                        baseline[(unit, instruction, receiver)]["strict_correct"]
                        and not any(diagnostics[(unit, instruction, receiver)]["ties"])
                        and (baseline[(unit, instruction, receiver)]["margin"] > 0
                             if receiver.startswith("P") else baseline[(unit, instruction, receiver)]["margin"] < 0)
                        for receiver in RECEIVERS)
                    journal.record("baseline_support", {"subject": subject, "unit": unit,
                                                        "instruction": instruction,
                                                        "supported": support[group_key]})
                continue
            original = states[key]
            action = dict(step, source_state=baseline[key]["action"]["encoder_state"])
            if kind in ("passthrough", "self", "full_swap"):
                evaluation_case = case
                if kind == "self":
                    raw = _raw(original[0, position[step["site"]]])
                    after = _patched(original, position[step["site"]], raw)
                    action["position"] = position[step["site"]]
                elif kind == "full_swap":
                    donor_key = (unit, instruction, step["donor"])
                    donor_case = group["cases"][step["donor"]]
                    if donor_case["attention_mask"] != case["attention_mask"]:
                        raise ValueError("full-state donor mask differs")
                    evaluation_case = dict(case, attention_mask=list(donor_case["attention_mask"]))
                    after = states[donor_key].clone()
                    action["donor_state"] = baseline[donor_key]["action"]["encoder_state"]
                    action["source_mask_origin"] = "donor"
                else:
                    after = original.clone()
                outcome, _ = _evaluate(evaluator, journal, tokenizer, evaluation_case, original, after, action, targets)
                expected = (unit, instruction, step["donor"]) if kind == "full_swap" else key
                if not _same_output(journal, outcome, baseline[expected]):
                    raise RuntimeError("identity or full-state donor control did not match")
                journal.record("control_verified", {"action": step, "expected": list(expected)})
                _record_comparison(journal, action, baseline[key], outcome)
                continue
            if kind == "repeat":
                repeated, encoder_ref = _encode_case(model, case, journal, width, evaluator._check_runtime)
                action["encoder_state"] = encoder_ref
                outcome, _ = _evaluate(evaluator, journal, tokenizer, case, repeated, repeated.clone(), action, targets)
                if not _same_output(journal, outcome, baseline[key]):
                    raise RuntimeError("repeat did not match clean output")
                journal.record("control_verified", {"action": step, "expected": list(key)})
                _record_comparison(journal, action, baseline[key], outcome)
                continue
            if kind == "matched":
                plan = geometries[group_key]
                landing = plan["landings"].get((side, step["dose"], step["action"]))
                if landing is None or landing["status"] != "matchable":
                    journal.record("action_skipped", {"action": step, "reason":
                                   plan["status"] if landing is None else landing["failure"]})
                    continue
                raw = journal.read_tensor(landing["value_ref"])
                after = _patched(original, position[landing["site"]], raw)
                action.update({key: value for key, value in landing.items()
                               if key not in ("receiver", "dose", "arm")})
                action["position"] = position[landing["site"]]
                if landing["donor"] is not None:
                    action["donor_state"] = baseline[(unit, instruction, landing["donor"])]["action"]["encoder_state"]
            else:
                donor = states[(unit, instruction, OPPOSITE[side])]
                positions = (position["adjective"],) if kind == "natural_final" else sites[group_key]["lexical_positions"]
                after = original.clone()
                differences = []
                for index in positions:
                    own, target = _values(_raw(original[0, index])), _values(_raw(donor[0, index]))
                    differences.extend(b - a for a, b in zip(own, target))
                    after = _patched(after, index, _raw(donor[0, index]))
                action.update(positions=list(positions), composite_dose=_norm(differences),
                              donor_state=baseline[(unit, instruction, OPPOSITE[side])]["action"]["encoder_state"])
                if kind == "natural_final":
                    action.update(_natural_geometry(original, donor, position["adjective"]))
            outcome, _ = _evaluate(evaluator, journal, tokenizer, case, original, after, action, targets)
            measured[(unit, instruction, side, step.get("action", kind), step.get("dose"))] = outcome
            _record_comparison(journal, action, baseline[key], outcome)
            if kind == "natural_span" and len(sites[group_key]["lexical_positions"]) == 1:
                single = measured[(unit, instruction, side, "natural_final", None)]
                if (journal.read_tensor(single["after"]) != journal.read_tensor(outcome["after"])
                        or not _same_output(journal, single, outcome)):
                    raise RuntimeError("single-token natural anchors differ")
                journal.record("natural_anchor_verified", {"action": step})
        groups_out = []
        for unit in slots:
            for instruction in (0, 1):
                group_key = (unit, instruction)
                clean = {receiver: baseline[(unit, instruction, receiver)]["margin"] for receiver in RECEIVERS}
                doses = {dose: _dose_outcome(clean, measured, geometries[group_key]["landings"],
                                             unit, instruction, dose) for dose in ("low", "high")}
                row = {"unit": unit, "subject": subject, "instruction": instruction,
                       "baseline_supported": support[group_key], "geometry_status": geometries[group_key]["status"],
                       "doses": doses}
                groups_out.append(row)
                journal.record("group_outcome", row)
        evaluator._check_runtime()
        result = {"subject": subject, "slots": list(slots), "groups": groups_out, "testing_only": test_only}
        journal.finish("complete_candidate", result)
        return result
    except BaseException as primary:
        try:
            journal.finish("incomplete", {"error_type": type(primary).__name__, "action": step,
                                          "subject": subject, "testing_only": test_only})
        except BaseException as terminal_error:
            if isinstance(primary, Exception) and not isinstance(terminal_error, Exception):
                raise terminal_error from primary
        raise


def run_fabricated(model, tokenizer, groups, journal, *, subject, width, vocab_size, custody_check=None):
    """Run only fabricated model/cases in a test-owned journal."""
    return _run_pass(model, tokenizer, groups, journal, subject=subject, width=width,
                     vocab_size=vocab_size, test_only=True, custody_check=custody_check)


def run_cpu_pass(model, tokenizer, groups, journal, *, subject, custody_check):
    """Run one separately admitted pass; this function does not load a model."""
    return _run_pass(model, tokenizer, groups, journal, subject=subject, width=512,
                     vocab_size=32128, test_only=False, custody_check=custody_check)
