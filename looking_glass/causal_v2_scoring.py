"""Score complete, revealed synthetic-v2 evidence without running a model.

The caller must independently establish execution, phase order, source identity,
plan metadata and durable custody. Hashes and attestations do not prove those
facts, blinding, or random sampling. No partial run is converted to a result.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import struct

from . import registry
from .causal_trace_v2 import _binary32_value
from .causal_v2_custody import (_float32_bits, canonical_json, validate_nonce_set,
                                verify_commitment)
from .causal_v2_evidence_adapter import (CELL_CHUNK_LIMIT, MAX_RECORD_BYTES, _FIELDS,
                                         _logits, _plans, _private, _trace)
from .causal_v2_executor import _check_capture_metadata, _check_plan_metadata, _target_spec
from .causal_v2_preparation import _context
from .pretrained_number_injection import add_float32_bits


_SEEDS = (31201, 31202, 31203, 31204, 31205, 31206)
_SITES = (1, 2, 3)
_STRENGTHS = (0.25, 0.75)
_NODES = tuple(f"block.{block}.{node}" for block in _SITES
               for node in ("branch.0", "activation", "merge"))
_MAX_HEADER = 64 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PREDICTION_FIELDS = {"model_seed", "cell_id", "case_id", "strength", "site", "polarity",
                      "polarity_prediction", "constant_prediction", "zero_prediction"}
_THRESHOLD = 1e-7


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json(raw, limit, encoder):
    if type(raw) is not bytes or len(raw) > limit:
        raise ValueError("expected bounded immutable JSON bytes")
    try:
        value = json.loads(raw)
        if type(value) is not dict or encoder(value) != raw:
            raise ValueError("JSON is not canonical")
        return value
    except (TypeError, UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("invalid canonical JSON") from error


def _digest(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


def _name(value):
    return type(value) is str and 0 < len(value) <= 256


def _number(value):
    # A correctly formed prediction is a binary64 mean of float32 contrasts.
    # Binary64 subnormals cannot arise here and can be flushed by ambient DAZ.
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("prediction must be finite binary64")
    b = struct.unpack(">Q", struct.pack(">d", value))[0]
    if b & 0x7ff0000000000000 == 0 and b & 0x000fffffffffffff:
        raise ValueError("subnormal binary64 prediction is unsupported")
    return value


def _same_number(a, b):
    a, b = _number(a), _number(b)
    x, y = struct.unpack(">Q", struct.pack(">d", a))[0], struct.unpack(">Q", struct.pack(">d", b))[0]
    return x == y or (x & 0x7fffffffffffffff == y & 0x7fffffffffffffff == 0)


def _same_bits(a, b):
    return len(a) == len(b) and all(x == y or (x & 0x7fffffff == y & 0x7fffffff == 0)
                                    for x, y in zip(a, b))


def _canonical_bits(bits):
    return tuple(0 if value & 0x7fffffff == 0 else value for value in bits)


def _metric(tensor):
    left, right = tensor["bits"]
    result = add_float32_bits(right, left ^ 0x80000000)
    if result & 0x7f800000 == 0x7f800000:
        raise ValueError("nonfinite float32 logit contrast")
    return _binary32_value(result)


def _comparison(clean, changed):
    differences = [not _same_bits(a["value"]["bits"], b["value"]["bits"])
                   for a, b in zip(clean, changed)]
    norms = []
    for a, b in zip(clean, changed):
        norms.append(math.sqrt(math.fsum((_binary32_value(x) - _binary32_value(y)) ** 2
                                        for x, y in zip(a["value"]["bits"], b["value"]["bits"]))))
    first = next((i for i, changed in enumerate(differences) if changed), None)
    return {"status": ("no_difference_observed" if first is None else
                       "upstream_difference" if first == 0 else "difference_observed"),
            "first_unequal": None if first is None else f"block.{first}.output",
            "last_equal": "block.3.output" if first is None else
                          None if first == 0 else f"block.{first - 1}.output",
            "l2_by_block": norms}


def _match_comparison(actual, expected):
    if (type(actual) is not dict or set(actual) != set(expected) or
            any(actual[key] != expected[key] for key in ("status", "first_unequal", "last_equal")) or
            type(actual["l2_by_block"]) is not list or len(actual["l2_by_block"]) != 4 or
            not all(_same_number(a, b) for a, b in zip(actual["l2_by_block"], expected["l2_by_block"]))):
        raise ValueError("saved sparse comparison differs from its trace")


def validate_saved_controls(record, *, target, strength, polarity):
    """Validate one saved v2 condition, without executing or authenticating it.

    Caller owns an unchanged decoded record and supplies the already revealed
    target and independently verified strength/polarity. Phase order, global
    identities, same-input source ownership and provenance remain external.
    This accepts both discovery and held-out records; target-bearing evidence
    must stay private until the schedule is permitted to be revealed.
    """
    try:
        return _validate_saved_controls(record, target, strength, polarity)
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("invalid saved control evidence") from error


def _validate_saved_controls(record, target, strength, polarity):
    if (type(record) is not dict or set(record) != _FIELDS or
            record["schema"] != "causal-v2-condition-record-v2" or
            record["phase"] not in ("discovery", "heldout") or
            type(strength) is not float or strength not in _STRENGTHS or
            type(record["strength"]) is not float or record["strength"] != strength or
            type(polarity) is not int or polarity not in (-1, 1)):
        raise ValueError("invalid saved condition schema or supplied policy")
    block, target_path = _target_spec(target)
    traces, logits = record["traces"], record["logits"]
    if type(traces) is not dict or set(traces) != {"clean", "repeat_clean", "perturbed"}:
        raise ValueError("trace inventory differs")
    for trace in traces.values():
        _trace(trace)
    if len({trace[0]["metadata"]["run_id"] for trace in traces.values()}) != 3:
        raise ValueError("sparse forward identities are not distinct")
    _logits(logits); _plans(record["plans"], record["schema"]); _private(record["private_integrity"], strength)
    clean_trace, repeat_trace, changed_trace = traces["clean"], traces["repeat_clean"], traces["perturbed"]
    compare = _comparison(clean_trace, changed_trace)
    _match_comparison(record["comparison"], compare)
    if (compare["status"] == "upstream_difference" or
            any(not _same_bits(clean_trace[j]["value"]["bits"], changed_trace[j]["value"]["bits"])
                for j in range(block)) or
            any(not _same_bits(a["value"]["bits"], b["value"]["bits"])
                for a, b in zip(clean_trace, repeat_trace)) or
            not _same_bits(logits["clean"]["bits"], logits["repeat_clean"]["bits"])):
        raise ValueError("upstream or repeated-clean control failed")
    if (_same_bits(clean_trace[3]["value"]["bits"], changed_trace[3]["value"]["bits"]) and
            not _same_bits(logits["clean"]["bits"], logits["perturbed"]["bits"])):
        raise ValueError("equal final state produced unequal logits")
    clean_key = (tuple(_canonical_bits(event["value"]["bits"]) for event in clean_trace),
                 _canonical_bits(logits["clean"]["bits"]))
    private = record["private_integrity"]
    if private["target"] != target or private["delta"] != polarity * strength:
        raise ValueError("injection target or delta differs")
    _check_capture_metadata(private["clean_capture_metadata"], target_path)
    _check_capture_metadata(private["perturbed_capture_metadata"], target_path)
    clean_target, pert_target = private["clean_target"]["bits"], private["perturbed_target"]["bits"]
    expected_target = add_float32_bits(clean_target[0], _float32_bits(private["delta"]))
    if (expected_target & 0x7f800000 == 0x7f800000 or
            not _same_bits([expected_target], [pert_target[0]]) or
            _same_bits([clean_target[0]], [pert_target[0]]) or
            not _same_bits(clean_target[1:], pert_target[1:])):
        raise ValueError("private target integrity failed")
    references = {}
    for plan in record["plans"]:
        site, kind, captures = plan["site"], plan["kind"], plan["captures"]
        path = f"blocks.{site}.output_tap"
        if kind in ("forward", "reverse", "upstream"):
            names, operations, capture_bytes, source_bytes, source_capture = (
                ("patch", "post_action"), ("patch", "capture"), 32, 32, None)
        elif kind == "null":
            names, operations, capture_bytes, source_bytes, source_capture = (
                ("null", "post_action"), ("null", "capture"), 32, 0, None)
        else:
            names, operations, capture_bytes, source_bytes, source_capture = (
                ("self_source", "self_patch", "post_action"), ("capture", "patch", "capture"), 64, 0, "self_source")
        _check_plan_metadata(plan["metadata"], path, names, operations, source_capture, capture_bytes, source_bytes)
        role = plan["comparison_role"]
        suffix = f"/block.{site}.output"
        reference = plan["comparison_id"][:-len(suffix)]
        for key, value in ((role, reference), (plan["base_role"], plan["base_id"])):
            if not _name(value) or (key in references and references[key] != value):
                raise ValueError("plan source or base references differ within a condition")
            references[key] = value
        trace = changed_trace if kind in ("reverse", "null", "self") else clean_trace
        if not _same_bits(captures["post_action"]["bits"], trace[site]["value"]["bits"]):
            raise ValueError("patch or null did not deliver its source")
        if kind == "self" and not _same_bits(captures["self_source"]["bits"], captures["post_action"]["bits"]):
            raise ValueError("self-patch source differs")
    clean_run, changed_run = clean_trace[0]["metadata"]["run_id"], changed_trace[0]["metadata"]["run_id"]
    expected_references = {"clean_sparse_trace": clean_run, "perturbed_sparse_trace": changed_run,
                           "clean": clean_run + "/logits", "perturbed": changed_run + "/logits"}
    if references != expected_references or len(set(references.values())) != 4:
        raise ValueError("plan source and base references differ from saved trace identities")
    clean_bits, changed_bits = logits["clean"]["bits"], logits["perturbed"]["bits"]
    if (not _same_bits(logits["upstream"]["bits"], changed_bits) or
            not _same_bits(logits["post_hook_clean"]["bits"], clean_bits) or
            any(not _same_bits(item["bits"], changed_bits) for item in logits["null"] + logits["self"])):
        raise ValueError("logit controls failed")
    clean_metric, perturbed_metric = _metric(logits["clean"]), _metric(logits["perturbed"])
    forward_effects, reverse_effects = [], []
    for site in _SITES:
        forward, reverse = logits["forward"][site - 1], logits["reverse"][site - 1]
        if (not _same_bits(forward["bits"], clean_bits if site >= block else changed_bits) or
                not _same_bits(reverse["bits"], changed_bits if site >= block else clean_bits)):
            raise ValueError("forward or reverse step-map control failed")
        forward_effects.append(_metric(forward) - perturbed_metric)
        reverse_effects.append(_metric(reverse) - clean_metric)
    return {"comparison": compare, "clean_baseline": clean_key, "clean_target": tuple(clean_target),
            "target_repair": clean_metric - perturbed_metric,
            "forward_effects": tuple(forward_effects), "reverse_effects": tuple(reverse_effects)}


def _metric_summary(rows):
    if not rows:
        return None
    return {key + "_mae": math.fsum(abs(row[key + "_prediction"] - row["effect"]) for row in rows) / len(rows)
            for key in ("polarity", "constant", "zero")}


def _sign_summary(rows):
    eligible = [row for row in rows if abs(row["effect"]) > _THRESHOLD]
    positive = sum(row["effect"] > 0 for row in eligible)
    negative = len(eligible) - positive
    def accuracy(key):
        if not eligible:
            return None
        return sum(abs(row[key + "_prediction"]) > _THRESHOLD and
                   (row[key + "_prediction"] > 0) == (row["effect"] > 0)
                   for row in eligible) / len(eligible)
    return {"eligible": len(eligible), "excluded": len(rows) - len(eligible),
            "positive": positive, "negative": negative,
            "sign_variation_tested": positive > 0 and negative > 0,
            "polarity_accuracy": accuracy("polarity"), "constant_accuracy": accuracy("constant")}


def _mean_or_none(values):
    return None if not values or any(value is None for value in values) else statistics.fmean(values)


def _selection_summary(cells, policy):
    selected = [cell for cell in cells if cell["selected"][policy] is not None]
    summaries = [cell["site_metrics"][cell["selected"][policy] - 1] for cell in selected]
    result = {"selected_cells": len(selected), "total_cells": 9,
              "coverage": len(selected) / 9, "conditional_on_selection": len(selected) != 9,
              "localization_hits": sum(cell["selected"][policy] == cell["true_block"] for cell in cells),
              "block_hit_rate": sum(cell["selected"][policy] == cell["true_block"] for cell in cells) / 9}
    for predictor in ("polarity", "constant", "zero"):
        result[predictor + "_mae"] = _mean_or_none([item[predictor + "_mae"] for item in summaries])
    result["constant_improvement"] = (None if not summaries else
        statistics.fmean(item["constant_mae"] - item["polarity_mae"] for item in summaries))
    result["zero_improvement"] = (None if not summaries else
        statistics.fmean(item["zero_mae"] - item["polarity_mae"] for item in summaries))
    rows = [row for cell in selected for row in cell["site_rows"][cell["selected"][policy] - 1]]
    result["sign"] = _sign_summary(rows)
    result["by_strength"] = {str(q): _metric_summary([row for row in rows if row["strength"] == q])
                             for q in _STRENGTHS}
    result["by_polarity"] = {str(p): _metric_summary([row for row in rows if row["polarity"] == p])
                             for p in (1, -1)}
    return result


def score_heldout(pre_manifest_bytes, freeze_bytes, prediction_record_bytes,
                  prediction_commitment_bytes, heldout_bytes, schedule_bytes,
                  heldout_root, site_nonce, prediction_nonce, heldout_chunks, *,
                  phase_order_verified=False, input_model_pairing_verified=False,
                  private_plan_metadata_verified=False, durable_evidence_verified=False,
                  case_generation_verified=False, discovery_freeze_verified=False,
                  exposure_classification=None):
    """Return canonical descriptive results from a complete revealed record.

    ``heldout_chunks`` is a one-pass iterable of 54 persisted cell chunks in
    freeze order. All six flags are caller attestations, not verified proofs.
    Phase-order attestation also requires no invalidating failure,
    cancellation, failed control, cap violation or retry before scoring,
    and that the attempt obeyed global fail-stop. The function consumes no
    model and never writes files or opens a GPU.
    """
    if not all(flag is True for flag in (phase_order_verified, input_model_pairing_verified,
                                        private_plan_metadata_verified, durable_evidence_verified,
                                        case_generation_verified, discovery_freeze_verified)):
        raise ValueError("independent execution and custody attestations are required")
    if type(exposure_classification) is not str or exposure_classification not in ("unexposed", "unblinded", "unknown"):
        raise ValueError("a caller-reported exposure classification is required")
    try:
        return _score(pre_manifest_bytes, freeze_bytes, prediction_record_bytes,
                      prediction_commitment_bytes, heldout_bytes, schedule_bytes,
                      heldout_root, site_nonce, prediction_nonce, heldout_chunks, exposure_classification)
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError, ZeroDivisionError) as error:
        raise ValueError("invalid or inconsistent v2 evidence") from error


def _score(manifest_raw, freeze_raw, prediction_raw, commitment_raw, heldout_raw, schedule_raw,
           heldout_root, site_nonce, prediction_nonce, chunks, exposure_classification):
    manifest, freeze, prediction, public_commitment, heldout, schedule = [
        _json(raw, _MAX_HEADER, canonical_json) for raw in
        (manifest_raw, freeze_raw, prediction_raw, commitment_raw, heldout_raw, schedule_raw)]
    validate_nonce_set(heldout_root, site_nonce, prediction_nonce)
    context = manifest["context"]
    if (manifest["schema"] != "synthetic-v2-pre-manifest-v1" or
            manifest["mode"] not in ("development", "study") or
            context["protocol_id"] != "synthetic-causal-v2" or
            manifest["context_sha256"] != _sha(canonical_json(context)) or
            heldout["context_sha256"] != manifest["context_sha256"] or
            schedule["context_sha256"] != manifest["context_sha256"] or
            not verify_commitment("heldout", heldout_root, heldout, manifest["heldout_commitment"]) or
            not verify_commitment("schedule", site_nonce, schedule, manifest["schedule_commitment"])):
        raise ValueError("pre-manifest or revealed commitments differ")
    states = context["model_state_hashes"]
    if (type(states) is not list or len(states) != 6 or
            any(type(item) is not dict or set(item) != {"seed", "sha256"} or
                type(item["seed"]) is not int or item["seed"] < 0 or not _digest(item["sha256"])
                for item in states)):
        raise ValueError("invalid model identity inventory")
    seeds = [item["seed"] for item in states]
    if (len(set(seeds)) != 6 or
            (manifest["mode"] == "study" and tuple(seeds) != _SEEDS) or
            (manifest["mode"] == "development" and set(seeds) & set(_SEEDS)) or
            _context(context, seeds) != manifest["context_sha256"]):
        raise ValueError("invalid model seed order")
    if (freeze["schema"] != "synthetic-v2-prediction-freeze-v1" or
            freeze["pre_manifest_sha256"] != _sha(manifest_raw) or
            type(freeze["cells"]) is not list or len(freeze["cells"]) != 54 or
            set(prediction) != {"pre_manifest_sha256", "freeze_sha256", "predictions"} or
            prediction["pre_manifest_sha256"] != _sha(manifest_raw) or
            prediction["freeze_sha256"] != _sha(freeze_raw) or
            set(public_commitment) != {"schema", "pre_manifest_sha256", "freeze_sha256", "commitment"} or
            public_commitment["schema"] != "synthetic-v2-prediction-commitment-v1" or
            public_commitment["pre_manifest_sha256"] != _sha(manifest_raw) or
            public_commitment["freeze_sha256"] != _sha(freeze_raw) or
            not verify_commitment("predictions", prediction_nonce, prediction, public_commitment["commitment"])):
        raise ValueError("prediction freeze or commitment differs")
    cases, discovery = heldout["cases"], manifest["discovery_cases"]
    assignments, preds = schedule["assignments"], prediction["predictions"]
    if (type(cases) is not list or len(cases) != 96 or type(assignments) is not list or
            len(assignments) != 54 or type(preds) is not list or len(preds) != 5184 or
            type(discovery) is not list or len(discovery) != 96):
        raise ValueError("incomplete cases, schedule, or prediction inventory")
    case_info, ids, vectors = {}, set(), set()
    for i, case in enumerate(discovery + cases):
        split = "discovery" if i < 96 else "heldout"
        i %= 96
        seed, index = seeds[i // 16], i % 16
        if (type(case) is not dict or set(case) != {"case_id", "model_seed", "split", "index", "values"} or
                type(case["model_seed"]) is not int or case["model_seed"] != seed or
                type(case["index"]) is not int or case["index"] != index or case["split"] != split or
                not _name(case["case_id"]) or case["case_id"] in ids or
                type(case["values"]) is not list or len(case["values"]) != 8):
            raise ValueError("invalid held-out case order")
        case_bits = tuple(_float32_bits(value) for value in case["values"])
        key = tuple(0 if value & 0x7fffffff == 0 else value for value in case_bits)
        if (key in vectors or case_bits[0] & 0x7fffffff == 0 or
                bool(case_bits[0] & 0x80000000) != bool(index % 2)):
            raise ValueError("duplicate case or incorrect polarity")
        if split == "heldout":
            case_info[(seed, index)] = (case["case_id"], 1 if index % 2 == 0 else -1,
                _sha(b"MLG-v2-input-f32be-v1\0" + struct.pack(">II", 1, 8) +
                     b"".join(struct.pack(">I", bit) for bit in case_bits)))
        ids.add(case["case_id"]); vectors.add(key)
    cell_ids = set()
    for mi, seed in enumerate(seeds):
        nodes = []
        for ci in range(9):
            index = mi * 9 + ci
            cell, assignment = freeze["cells"][index], assignments[index]
            if (type(cell) is not dict or set(cell) != {"cell_id", "model_seed", "selection"} or
                    type(cell["model_seed"]) is not int or cell["model_seed"] != seed or
                    not _name(cell["cell_id"]) or cell["cell_id"] in cell_ids or
                    type(assignment) is not dict or set(assignment) != {"cell_id", "model_seed", "injection_node"} or
                    type(assignment["model_seed"]) is not int or assignment["model_seed"] != seed or
                    assignment["cell_id"] != cell["cell_id"] or assignment["injection_node"] not in _NODES):
                raise ValueError("cell or revealed schedule identity differs")
            selection = cell["selection"]
            expected_negative = {"cases": 16, "false_detections": 0,
                                 "first_changed": None, "magnitude": None}
            if canonical_json(selection["no_injection"]) != canonical_json(expected_negative):
                raise ValueError("discovery no-injection summary is incomplete")
            for name in ("first_changed", "magnitude"):
                chosen = selection[name]
                if chosen is not None and (type(chosen) is not int or chosen not in _SITES):
                    raise ValueError("invalid frozen selection")
            nodes.append(assignment["injection_node"]); cell_ids.add(cell["cell_id"])
        if set(nodes) != set(_NODES):
            raise ValueError("schedule is not a permutation")
    output_cells, output_conditions, baseline, targets = [], [], {}, {}
    conditions_seen, chunk_names, run_ids = set(), set(), {}
    chunk_iterator = iter(chunks)
    prediction_hash = _sha(prediction_raw)
    for cell_index, cell in enumerate(freeze["cells"]):
        try:
            raw = next(chunk_iterator)
        except StopIteration as error:
            raise ValueError("missing held-out cell chunk") from error
        chunk = _json(raw, CELL_CHUNK_LIMIT, registry._json_bytes)
        if (set(chunk) != {"name", "records", "schema", "state"} or
                chunk["schema"] != "causal-v2-chunk-v1" or chunk["state"] != "complete" or
                type(chunk["name"]) is not str or chunk["name"].lower() in chunk_names or
                type(chunk["records"]) is not list or len(chunk["records"]) != 32):
            raise ValueError("incomplete or duplicate held-out chunk")
        registry._validate_name(chunk["name"])
        chunk_names.add(chunk["name"].lower())
        seed, cid = cell["model_seed"], cell["cell_id"]
        target = assignments[cell_index]["injection_node"]
        block = int(target.split(".")[1])
        rows_by_site = [[], [], []]
        misses = 0
        for position, row in enumerate(chunk["records"]):
            case_index, strength_index = divmod(position, 2)
            strength = _STRENGTHS[strength_index]
            case_id, polarity, input_sha = case_info[(seed, case_index)]
            first = cell_index * 96 + position * 3
            if (type(row) is not dict or set(row) != _FIELDS or
                    len(registry._json_bytes(row)) > MAX_RECORD_BYTES or
                    row["schema"] != "causal-v2-condition-record-v2" or row["phase"] != "heldout" or
                    row["cell_id"] != cid or type(row["model_seed"]) is not int or row["model_seed"] != seed or
                    row["model_state_sha256"] != states[cell_index // 9]["sha256"] or
                    row["case_id"] != case_id or type(row["case_index"]) is not int or
                    row["case_index"] != case_index or row["input_sha256"] != input_sha or
                    type(row["strength"]) is not float or row["strength"] != strength or
                    row["prediction_ref"] != f"sha256:{prediction_hash}:{first}:3" or
                    not _name(row["condition_id"]) or row["condition_id"] in conditions_seen):
                raise ValueError("held-out condition identity or prediction reference differs")
            conditions_seen.add(row["condition_id"])
            frozen_predictions = []
            for site in _SITES:
                pred = preds[first + site - 1]
                if (type(pred) is not dict or set(pred) != _PREDICTION_FIELDS or
                        type(pred["model_seed"]) is not int or pred["model_seed"] != seed or
                        pred["cell_id"] != cid or pred["case_id"] != case_id or
                        type(pred["site"]) is not int or pred["site"] != site or
                        type(pred["strength"]) is not float or pred["strength"] != strength or
                        type(pred["polarity"]) is not int or pred["polarity"] != polarity):
                    raise ValueError("prediction row identity differs")
                expected = cell["selection"]["predictions"][str(site)][str(strength)]
                if (set(expected) != {"positive", "negative", "constant", "zero"} or
                        not _same_number(pred["polarity_prediction"], expected["positive" if polarity > 0 else "negative"]) or
                        not _same_number(pred["constant_prediction"], expected["constant"]) or
                        not _same_number(pred["zero_prediction"], expected["zero"]) or
                        not _same_number(pred["zero_prediction"], 0.0)):
                    raise ValueError("prediction differs from frozen functions")
                frozen_predictions.append(pred)
            controls = validate_saved_controls(row, target=target, strength=strength, polarity=polarity)
            compare, logits = controls["comparison"], row["logits"]
            for role in ("clean", "repeat_clean", "perturbed"):
                run_id = row["traces"][role][0]["metadata"]["run_id"]
                identity = (role, seed, case_index)
                if run_id in run_ids and (role == "perturbed" or run_ids[run_id] != identity):
                    raise ValueError("trace run ID aliases a different recorded forward")
                run_ids[run_id] = identity
            if compare["first_unequal"] is None:
                misses += 1
            baseline_key = (seed, case_index)
            if baseline_key in baseline and baseline[baseline_key] != controls["clean_baseline"]:
                raise ValueError("reused clean model/input baseline differs")
            baseline[baseline_key] = controls["clean_baseline"]
            target_key = (cid, case_index)
            if target_key in targets and not _same_bits(targets[target_key], controls["clean_target"]):
                raise ValueError("reused private clean target differs")
            targets[target_key] = controls["clean_target"]
            target_repair = controls["target_repair"]
            site_results = []
            for site in _SITES:
                effect = controls["forward_effects"][site - 1]
                pred = frozen_predictions[site - 1]
                result = {"site": site, "strength": strength, "polarity": polarity, "effect": effect,
                          "reverse_effect": controls["reverse_effects"][site - 1],
                          "polarity_prediction": pred["polarity_prediction"],
                          "constant_prediction": pred["constant_prediction"],
                          "zero_prediction": pred["zero_prediction"],
                          "residual": pred["polarity_prediction"] - effect,
                          "constant_residual": pred["constant_prediction"] - effect,
                          "zero_residual": pred["zero_prediction"] - effect,
                          "repair_absolute_residual": abs(effect - target_repair)}
                rows_by_site[site - 1].append(result); site_results.append(dict(result))
            output_conditions.append({"model_seed": seed, "cell_id": cid, "case_id": case_id,
                "condition_id": row["condition_id"], "case_index": case_index, "strength": strength,
                "polarity": polarity, "input_sha256": input_sha, "true_target": target,
                "comparison": compare, "logits": logits, "target_repair": target_repair,
                "sites": site_results})
        metrics = [_metric_summary(rows) for rows in rows_by_site]
        selected = {name: cell["selection"][name] for name in ("first_changed", "magnitude")}
        output_cells.append({"model_seed": seed, "cell_id": cid, "true_block": block,
                             "true_target": target, "selected": selected, "site_rows": rows_by_site,
                             "site_metrics": metrics, "observation_misses": misses,
                             "chunk_name": chunk["name"], "chunk_sha256": _sha(raw)})
    try:
        next(chunk_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("unexpected extra held-out chunk")
    models = []
    for mi, seed in enumerate(seeds):
        cells = output_cells[mi * 9:(mi + 1) * 9]
        selection = {name: _selection_summary(cells, name) for name in ("first_changed", "magnitude")}
        random = {key + "_mae": statistics.fmean(statistics.fmean(
                    site[key + "_mae"] for site in cell["site_metrics"]) for cell in cells)
                  for key in ("polarity", "constant", "zero")}
        random["localization_hit_probability"] = 1 / 3
        random["constant_improvement"] = statistics.fmean(statistics.fmean(
            site["constant_mae"] - site["polarity_mae"] for site in cell["site_metrics"]) for cell in cells)
        random["zero_improvement"] = statistics.fmean(statistics.fmean(
            site["zero_mae"] - site["polarity_mae"] for site in cell["site_metrics"]) for cell in cells)
        models.append({"model_seed": seed, "selection": selection, "random": random,
                       "no_injection": {"discovery": dict(freeze["cells"][mi * 9]["selection"]["no_injection"]),
                                        "heldout": {"cases": 16, "false_detections": 0,
                                        "first_changed": None, "magnitude": None}},
                       "observation_misses": sum(cell["observation_misses"] for cell in cells)})
    equal = {"selection": {}, "random": {}}
    for policy in ("first_changed", "magnitude"):
        equal["selection"][policy] = {key: _mean_or_none([model["selection"][policy][key] for model in models])
             for key in ("polarity_mae", "constant_mae", "zero_mae", "constant_improvement", "zero_improvement",
                         "coverage", "block_hit_rate")}
        equal["selection"][policy]["conditional_on_selection"] = any(
             model["selection"][policy]["conditional_on_selection"] for model in models)
    equal["random"] = {key: statistics.fmean(model["random"][key] for model in models)
                       for key in ("polarity_mae", "constant_mae", "zero_mae", "constant_improvement",
                                   "zero_improvement", "localization_hit_probability")}
    for cell in output_cells:
        del cell["site_rows"]
        cell["localization"] = {name: {"selected": site, "abstained": site is None,
                                        "hit": site == cell["true_block"]}
                                for name, site in cell["selected"].items()}
    repair_groups = []
    for mi, seed in enumerate(seeds):
        for cell in output_cells[mi * 9:(mi + 1) * 9]:
            for strength in _STRENGTHS:
                for polarity in (1, -1):
                    conditions = [row for row in output_conditions if row["cell_id"] == cell["cell_id"]
                                  and row["strength"] == strength and row["polarity"] == polarity]
                    for site in _SITES:
                        residuals = [row["sites"][site - 1]["repair_absolute_residual"] for row in conditions]
                        if len(residuals) != 8:
                            raise ValueError("incomplete repair grouping")
                        metrics = _metric_summary([row["sites"][site - 1] for row in conditions])
                        repair_groups.append({"model_seed": seed, "cell_id": cell["cell_id"],
                            "true_target": cell["true_target"], "true_block": cell["true_block"],
                            "injection_operation": cell["true_target"].split(".", 2)[2],
                            "strength": strength, "polarity": polarity, "site": site,
                            "count": len(residuals), "mean_absolute_residual": statistics.fmean(residuals),
                            "maximum_absolute_residual": max(residuals), "prediction_mae": metrics,
                            "constant_improvement": metrics["constant_mae"] - metrics["polarity_mae"],
                            "zero_improvement": metrics["zero_mae"] - metrics["polarity_mae"]})
    interpretation = ("exploratory_unblinded" if exposure_classification == "unblinded" else
                      "development_only" if manifest["mode"] == "development" else
                      "unconfirmed_exposure_unknown" if exposure_classification == "unknown" else
                      "confirmation_conditional_on_caller_attestations")
    return canonical_json({"schema": "synthetic-v2-heldout-summary-v1", "status": "complete",
                           "mode": manifest["mode"], "interpretation": interpretation,
                           "exposure_classification": exposure_classification,
                           "pre_manifest_sha256": _sha(manifest_raw), "freeze_sha256": _sha(freeze_raw),
                           "prediction_sha256": prediction_hash, "models": models,
                           "equal_model_means": equal, "cells": output_cells,
                           "conditions_count": len(output_conditions), "conditions": output_conditions,
                           "repair_groups": repair_groups})
