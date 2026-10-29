"""Independent saved-byte checks for the synthetic-v2 study.

The auditor neither executes a model nor establishes execution provenance,
publication chronology, or secrecy. Float32 arithmetic below uses integer
bits so its results do not depend on the host's denormal mode.
"""
from __future__ import annotations

import math
import hashlib
import json
import re
import statistics
import struct


def _parts(bits):
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("invalid binary32 bits")
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("nonfinite binary32 value")
    sign = -1 if bits >> 31 else 1
    if exponent == 0:
        return sign * fraction, -149
    return sign * ((1 << 23) | fraction), exponent - 150


def _rounded_shift(value, shift):
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return quotient


def _add_bits(left, right):
    a, ae = _parts(left)
    b, be = _parts(right)
    exponent = min(ae, be)
    total = (a << (ae - exponent)) + (b << (be - exponent))
    if total == 0:
        return 0
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    top = magnitude.bit_length() - 1 + exponent
    if top < -126:
        mantissa = _rounded_shift(magnitude, -(exponent + 149))
        if mantissa >= 1 << 23:
            return sign | 0x00800000
        return sign | mantissa
    shift = magnitude.bit_length() - 24
    mantissa = _rounded_shift(magnitude, shift)
    stored_exponent = exponent + shift + 150
    if mantissa == 1 << 24:
        mantissa >>= 1
        stored_exponent += 1
    if not 1 <= stored_exponent < 255:
        raise ValueError("binary32 result overflows")
    return sign | (stored_exponent << 23) | (mantissa & 0x7FFFFF)


def _value(bits):
    mantissa, exponent = _parts(bits)
    return math.ldexp(mantissa, exponent)


def binary32_contrast(bits):
    """Return binary32-rounded ``logit[1] - logit[0]`` as a binary64 float."""
    if type(bits) not in (list, tuple) or len(bits) != 2:
        raise ValueError("two binary32 logit bit patterns are required")
    _parts(bits[0])
    _parts(bits[1])
    return _value(_add_bits(bits[1], bits[0] ^ 0x80000000))


def same_float32_bits(left, right):
    """Compare finite float32 vectors while treating both signed zeros equally."""
    if type(left) not in (list, tuple) or type(right) not in (list, tuple):
        raise ValueError("binary32 vectors are required")
    for bits in (*left, *right):
        _parts(bits)
    return len(left) == len(right) and all(
        a == b or (a & 0x7FFFFFFF == 0 and b & 0x7FFFFFFF == 0)
        for a, b in zip(left, right)
    )


def _tensor_bits(tensor, width):
    if (type(tensor) is not dict or set(tensor) != {"dtype", "device", "shape", "bits"}
            or tensor["dtype"] != "torch.float32" or tensor["device"] != "cpu"
            or tensor["shape"] != [1, width] or type(tensor["bits"]) is not list
            or len(tensor["bits"]) != width):
        raise ValueError("invalid saved tensor")
    for value in tensor["bits"]:
        _parts(value)
    return tensor["bits"]


def _trace_bits(trace):
    if type(trace) is not list or len(trace) != 4:
        raise ValueError("incomplete sparse trace")
    result = []
    run_id = None
    for index, event in enumerate(trace):
        if type(event) is not dict or set(event) != {"metadata", "value"}:
            raise ValueError("invalid sparse event")
        meta = event["metadata"]
        if (type(meta) is not dict or meta.get("node_id") != f"block.{index}.output"
                or type(meta.get("sequence")) is not int or meta["sequence"] != index
                or type(meta.get("occurrence")) is not int or meta["occurrence"] != 0
                or type(meta.get("run_id")) is not str or not meta["run_id"]):
            raise ValueError("invalid sparse coordinates")
        if run_id is None:
            run_id = meta["run_id"]
        elif run_id != meta["run_id"]:
            raise ValueError("sparse run identity differs")
        result.append(_tensor_bits(event["value"], 8))
    return run_id, result


def inspect_condition(record, *, target, strength, polarity):
    """Reconstruct one saved condition and independently check its controls.

    The existing metadata validator is an additional structural check. Effects,
    comparisons, target arithmetic, and control values are recalculated here
    from the saved float32 bits. This does not establish that a forward ran.
    """
    from .causal_v2_scoring import validate_saved_controls

    try:
        if (type(target) is not str or target not in {
                f"block.{block}.{site}" for block in (1, 2, 3)
                for site in ("branch.0", "activation", "merge")}
                or type(strength) is not float or strength not in (0.25, 0.75)
                or type(polarity) is not int or polarity not in (-1, 1)):
            raise ValueError("invalid condition policy")
        structural = validate_saved_controls(record, target=target, strength=strength, polarity=polarity)
        block = int(target.split(".")[1])
        traces = record["traces"]
        clean_id, clean = _trace_bits(traces["clean"])
        repeat_id, repeat = _trace_bits(traces["repeat_clean"])
        changed_id, changed = _trace_bits(traces["perturbed"])
        if len({clean_id, repeat_id, changed_id}) != 3:
            raise ValueError("sparse run identities overlap")
        if any(not same_float32_bits(a, b) for a, b in zip(clean, repeat)):
            raise ValueError("clean trace repeat differs")
        differences = [not same_float32_bits(a, b) for a, b in zip(clean, changed)]
        if any(differences[:block]):
            raise ValueError("upstream trace changed")
        first = next((index for index, differs in enumerate(differences) if differs), None)
        norms = [math.sqrt(math.fsum((_value(a) - _value(b)) ** 2 for a, b in zip(x, y)))
                 for x, y in zip(clean, changed)]
        comparison = {"status": "no_difference_observed" if first is None else "difference_observed",
                      "first_unequal": None if first is None else f"block.{first}.output",
                      "last_equal": f"block.{3 if first is None else first - 1}.output",
                      "l2_by_block": norms}
        if record["comparison"] != comparison:
            raise ValueError("saved comparison differs from traces")

        private = record["private_integrity"]
        source = _tensor_bits(private["clean_target"], 8)
        injected = _tensor_bits(private["perturbed_target"], 8)
        delta = (0x3E800000 if strength == 0.25 else 0x3F400000) | (0x80000000 if polarity < 0 else 0)
        if (private["target"] != target or type(private["delta"]) is not float
                or private["delta"] != strength * polarity
                or not same_float32_bits([_add_bits(source[0], delta)], [injected[0]])
                or same_float32_bits([source[0]], [injected[0]])
                or not same_float32_bits(source[1:], injected[1:])):
            raise ValueError("saved injection landing differs")

        logits = record["logits"]
        base = _tensor_bits(logits["clean"], 2)
        repeated = _tensor_bits(logits["repeat_clean"], 2)
        perturbed = _tensor_bits(logits["perturbed"], 2)
        if (not same_float32_bits(base, repeated)
                or not same_float32_bits(_tensor_bits(logits["upstream"], 2), perturbed)
                or not same_float32_bits(_tensor_bits(logits["post_hook_clean"], 2), base)
                or (same_float32_bits(clean[3], changed[3]) and not same_float32_bits(base, perturbed))):
            raise ValueError("clean or upstream control differs")
        for kind in ("forward", "reverse", "null", "self"):
            if type(logits[kind]) is not list or len(logits[kind]) != 3:
                raise ValueError("incomplete intervention outputs")
        plan_order = [(kind, site) for kind in ("forward", "reverse", "null", "self")
                      for site in (1, 2, 3)] + [("upstream", 0)]
        if (type(record["plans"]) is not list or len(record["plans"]) != len(plan_order)
                or [(plan["kind"], plan["site"]) for plan in record["plans"]] != plan_order):
            raise ValueError("saved intervention plan order differs")
        for plan, (kind, site) in zip(record["plans"], plan_order):
            expected = changed[site] if kind in ("reverse", "null", "self") else clean[site]
            captures = plan["captures"]
            if not same_float32_bits(_tensor_bits(captures["post_action"], 8), expected):
                raise ValueError("intervention source did not land")
            if kind == "self" and not same_float32_bits(_tensor_bits(captures["self_source"], 8), expected):
                raise ValueError("self-patch source differs")
        forward, reverse = [], []
        clean_metric, changed_metric = binary32_contrast(base), binary32_contrast(perturbed)
        for index, site in enumerate((1, 2, 3)):
            restored = _tensor_bits(logits["forward"][index], 2)
            inverted = _tensor_bits(logits["reverse"][index], 2)
            if (not same_float32_bits(restored, base if site >= block else perturbed)
                    or not same_float32_bits(inverted, perturbed if site >= block else base)
                    or not same_float32_bits(_tensor_bits(logits["null"][index], 2), perturbed)
                    or not same_float32_bits(_tensor_bits(logits["self"][index], 2), perturbed)):
                raise ValueError("intervention output control differs")
            forward.append(binary32_contrast(restored) - changed_metric)
            reverse.append(binary32_contrast(inverted) - clean_metric)
        if (structural["comparison"] != comparison or structural["forward_effects"] != tuple(forward)
                or structural["reverse_effects"] != tuple(reverse)
                or structural["target_repair"] != clean_metric - changed_metric):
            raise ValueError("independent condition reconstruction disagrees")
        normalized = lambda bits: tuple(0 if value & 0x7FFFFFFF == 0 else value for value in bits)
        return {"comparison": comparison, "first_unequal": comparison["first_unequal"],
                "trace_ids": (clean_id, repeat_id, changed_id),
                "clean_baseline": (tuple(normalized(item) for item in clean), normalized(base)),
                "clean_target": normalized(source), "forward_effects": tuple(forward),
                "reverse_effects": tuple(reverse), "target_repair": clean_metric - changed_metric}
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("invalid saved condition") from error


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value, *, ascii_only=True):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=ascii_only, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as error:
        raise ValueError("invalid saved JSON") from error


def _object(raw, limit, *, ascii_only=True):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError("invalid saved JSON size")
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("invalid saved JSON") from error
    if type(value) is not dict or _canonical(value, ascii_only=ascii_only) != raw:
        raise ValueError("saved JSON is not canonical")
    return value


def _same_value(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (set(left) == set(right)
                and all(_same_value(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


def _require_same(actual, expected, description):
    if not _same_value(actual, expected):
        raise ValueError(description + " differs from saved evidence")


def _input_sha(values):
    if type(values) is not list or len(values) != 8:
        raise ValueError("invalid saved input")
    try:
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("invalid saved input")
        raw = b"".join(struct.pack(">f", value) for value in values)
    except (OverflowError, struct.error) as error:
        raise ValueError("invalid saved input") from error
    return _sha(b"MLG-v2-input-f32be-v1\0" + struct.pack(">II", 1, 8) + raw)


def _site_metrics(rows):
    if not rows:
        return None
    return {name + "_mae": math.fsum(abs(row[name + "_prediction"] - row["effect"])
                                      for row in rows) / len(rows)
            for name in ("polarity", "constant", "zero")}


def _sign_metrics(rows):
    eligible = [row for row in rows if abs(row["effect"]) > 1e-7]
    positive = sum(row["effect"] > 0 for row in eligible)
    negative = len(eligible) - positive

    def accuracy(name):
        if not eligible:
            return None
        return sum(abs(row[name + "_prediction"]) > 1e-7
                   and (row[name + "_prediction"] > 0) == (row["effect"] > 0)
                   for row in eligible) / len(eligible)

    return {"eligible": len(eligible), "excluded": len(rows) - len(eligible),
            "positive": positive, "negative": negative,
            "sign_variation_tested": positive > 0 and negative > 0,
            "polarity_accuracy": accuracy("polarity"), "constant_accuracy": accuracy("constant")}


def _mean_or_none(values):
    return None if not values or any(value is None for value in values) else statistics.fmean(values)


def _policy_metrics(cells, policy):
    selected = [cell for cell in cells if cell["selected"][policy] is not None]
    metrics = [cell["site_metrics"][cell["selected"][policy] - 1] for cell in selected]
    rows = [row for cell in selected for row in cell["site_rows"][cell["selected"][policy] - 1]]
    hits = sum(cell["selected"][policy] == cell["true_block"] for cell in cells)
    out = {"selected_cells": len(selected), "total_cells": 9, "coverage": len(selected) / 9,
           "conditional_on_selection": len(selected) != 9, "localization_hits": hits,
           "block_hit_rate": hits / 9}
    for name in ("polarity", "constant", "zero"):
        out[name + "_mae"] = _mean_or_none([item[name + "_mae"] for item in metrics])
    out["constant_improvement"] = (None if not metrics else statistics.fmean(
        item["constant_mae"] - item["polarity_mae"] for item in metrics))
    out["zero_improvement"] = (None if not metrics else statistics.fmean(
        item["zero_mae"] - item["polarity_mae"] for item in metrics))
    out["sign"] = _sign_metrics(rows)
    out["by_strength"] = {str(value): _site_metrics([row for row in rows if row["strength"] == value])
                          for value in (0.25, 0.75)}
    out["by_polarity"] = {str(value): _site_metrics([row for row in rows if row["polarity"] == value])
                          for value in (1, -1)}
    return out


def _summaries(cells, freeze):
    models = []
    for model_index in range(6):
        segment = cells[model_index * 9:(model_index + 1) * 9]
        seed = segment[0]["model_seed"]
        selection = {policy: _policy_metrics(segment, policy) for policy in ("first_changed", "magnitude")}
        random = {name + "_mae": statistics.fmean(statistics.fmean(
            site[name + "_mae"] for site in cell["site_metrics"]) for cell in segment)
                  for name in ("polarity", "constant", "zero")}
        random["localization_hit_probability"] = 1 / 3
        for name in ("constant", "zero"):
            random[name + "_improvement"] = statistics.fmean(statistics.fmean(
                site[name + "_mae"] - site["polarity_mae"] for site in cell["site_metrics"])
                for cell in segment)
        models.append({"model_seed": seed, "selection": selection, "random": random,
            "no_injection": {"discovery": freeze["cells"][model_index * 9]["selection"]["no_injection"],
                             "heldout": {"cases": 16, "false_detections": 0,
                                         "first_changed": None, "magnitude": None}},
            "observation_misses": sum(cell["observation_misses"] for cell in segment)})
    equal = {"selection": {}, "random": {}}
    for policy in ("first_changed", "magnitude"):
        equal["selection"][policy] = {name: _mean_or_none(
            [model["selection"][policy][name] for model in models]) for name in
            ("polarity_mae", "constant_mae", "zero_mae", "constant_improvement",
             "zero_improvement", "coverage", "block_hit_rate")}
        equal["selection"][policy]["conditional_on_selection"] = any(
            model["selection"][policy]["conditional_on_selection"] for model in models)
    for name in ("polarity_mae", "constant_mae", "zero_mae", "constant_improvement",
                 "zero_improvement", "localization_hit_probability"):
        equal["random"][name] = statistics.fmean(model["random"][name] for model in models)
    return models, equal


def audit_evidence(*, manifest, freeze, prediction, commitment, reveal, exposure,
                   private_material, results, discovery_chunks, heldout_chunks):
    """Replay the complete revealed, saved synthetic-v2 evidence without a model.

    Chunk iterables must yield immutable bytes from a quiescent retained attempt
    in cell order. The caller is responsible for filesystem inventory and outcome
    checks. The result reports consistency, not execution, timing, secrecy,
    case-generation provenance, or exclusion-inventory completeness.
    """
    from .causal_v2_evidence_adapter import CELL_CHUNK_LIMIT
    from .causal_v2_freeze import build_freeze, verify_discovery_evidence
    from .causal_v2_preparation import PreparedSnapshot

    try:
        pre = _object(manifest, 512 * 1024)
        frozen = _object(freeze, 32 * 1024 * 1024)
        predicted = _object(prediction, 16 * 1024 * 1024)
        prediction_sha256 = _sha(prediction)
        _object(commitment, 16 * 1024)
        shown = _object(reveal, 16 * 1024 * 1024)
        private = _object(private_material, 4 * 1024 * 1024)
        exposed = _object(exposure, 16 * 1024)
        reported = _object(results, 32 * 1024 * 1024)
        if (set(shown) != {"schema", "heldout", "schedule", "heldout_root", "site_nonce",
                           "prediction_nonce", "prediction_record", "exposure"}
                or shown["schema"] != "synthetic-v2-reveal-v1"
                or set(private) != {"schema", "heldout", "schedule", "heldout_root", "site_nonce", "prediction_nonce"}
                or private["schema"] != "synthetic-v2-private-material-v1"
                or any(not _same_value(private[key], shown[key]) for key in
                       ("heldout", "schedule", "heldout_root", "site_nonce", "prediction_nonce"))
                or not _same_value(shown["prediction_record"], predicted)
                or set(exposed) != {"schema", "classification", "basis"}
                or exposed["schema"] != "synthetic-v2-exposure-v1"
                or exposed["classification"] not in ("unexposed", "unblinded", "unknown")
                or type(exposed["basis"]) is not str or not 0 < len(exposed["basis"]) <= 4096
                or shown["exposure"] != {"schema": "synthetic-v2-exposure-reveal-v1",
                    "classification": exposed["classification"], "private_record_sha256": _sha(exposure)}):
            raise ValueError("revealed private records differ")
        nonce_keys = ("heldout_root", "site_nonce", "prediction_nonce")
        if any(type(shown[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", shown[key]) is None
               for key in nonce_keys):
            raise ValueError("invalid revealed nonce")
        prep_fields = ("mode", "context", "context_sha256", "pilot_exclusions_sha256",
                       "development_exclusions_sha256", "historical_exclusions_sha256",
                       "historical_exclusions_count", "discovery_cases", "heldout_commitment",
                       "schedule_commitment")
        prepared = {key: pre[key] for key in prep_fields}
        snapshot = PreparedSnapshot(_canonical(prepared), _canonical(shown["heldout"]),
                                    _canonical(shown["schedule"]),
                                    *(bytes.fromhex(shown[key]) for key in nonce_keys))
        cells = frozen["cells"]
        if type(cells) is not list or len(cells) != 54:
            raise ValueError("the freeze must contain 54 ordered cells")
        rebuilt = build_freeze(snapshot, manifest, cells)
        if rebuilt != (freeze, prediction, commitment):
            raise ValueError("frozen functions or commitments differ")
        assignments = shown["schedule"]["assignments"]
        heldout_cases, discovery_cases = shown["heldout"]["cases"], pre["discovery_cases"]
        states = pre["context"]["model_state_hashes"]
        if (type(assignments) is not list or len(assignments) != 54 or type(states) is not list
                or len(states) != 6 or type(discovery_cases) is not list or len(discovery_cases) != 96
                or type(heldout_cases) is not list or len(heldout_cases) != 96):
            raise ValueError("incomplete saved study inventory")
        prediction_rows = predicted["predictions"]
        if type(prediction_rows) is not list or len(prediction_rows) != 5184:
            raise ValueError("incomplete concrete predictions")
        condition_ids, run_owners, clean_owners, target_owners = set(), {}, {}, {}
        discovery_hashes, heldout_hashes = [], []

        def check_record(row, cell_index, position, phase):
            model_index = cell_index // 9
            seed = states[model_index]["seed"]
            assignment = assignments[cell_index]
            case_index, strength_index = divmod(position, 2)
            strength = (0.25, 0.75)[strength_index]
            polarity = 1 if case_index % 2 == 0 else -1
            cases = discovery_cases if phase == "discovery" else heldout_cases
            case = cases[model_index * 16 + case_index]
            expected_ref = (None if phase == "discovery" else
                            f"sha256:{prediction_sha256}:{cell_index * 96 + position * 3}:3")
            if (type(row) is not dict or len(_canonical(row, ascii_only=False)) > 64 * 1024
                    or row.get("phase") != phase or row.get("cell_id") != assignment["cell_id"]
                    or type(row.get("model_seed")) is not int or row["model_seed"] != seed
                    or row.get("model_state_sha256") != states[model_index]["sha256"]
                    or row.get("case_id") != case["case_id"]
                    or type(row.get("case_index")) is not int or row["case_index"] != case_index
                    or type(row.get("strength")) is not float or row["strength"] != strength
                    or row.get("input_sha256") != _input_sha(case["values"])
                    or row.get("prediction_ref") != expected_ref
                    or type(row.get("condition_id")) is not str or not row["condition_id"]
                    or row["condition_id"] in condition_ids):
                raise ValueError("saved condition identity differs")
            condition_ids.add(row["condition_id"])
            checked = inspect_condition(row, target=assignment["injection_node"],
                                        strength=strength, polarity=polarity)
            clean_id, repeat_id, changed_id = checked["trace_ids"]
            for role, run_id in zip(("clean", "repeat_clean", "perturbed"), checked["trace_ids"]):
                owner = (phase, role, seed, case_index)
                if run_id in run_owners and (role == "perturbed" or run_owners[run_id] != owner):
                    raise ValueError("trace identity is reused across forwards")
                run_owners[run_id] = owner
            clean_key = (phase, seed, case_index)
            clean_value = (clean_id, repeat_id, checked["clean_baseline"])
            if clean_key in clean_owners and clean_owners[clean_key] != clean_value:
                raise ValueError("reused clean evidence differs")
            clean_owners[clean_key] = clean_value
            target_key = (phase, cell_index, case_index)
            if target_key in target_owners and target_owners[target_key] != checked["clean_target"]:
                raise ValueError("reused target capture differs")
            target_owners[target_key] = checked["clean_target"]
            return checked, strength, polarity

        def check_chunk(raw, cell_index, phase):
            chunk = _object(raw, CELL_CHUNK_LIMIT, ascii_only=False)
            expected_name = f"cell-{cell_index:02d}-{'d' if phase == 'discovery' else 'h'}.json"
            if (set(chunk) != {"name", "records", "schema", "state"}
                    or chunk["name"] != expected_name or chunk["schema"] != "causal-v2-chunk-v1"
                    or chunk["state"] != "complete" or type(chunk["records"]) is not list
                    or len(chunk["records"]) != 32):
                raise ValueError("saved chunk is missing or incomplete")
            return chunk

        def verify_selection(selection, checked_rows):
            votes = {str(site): 0 for site in (1, 2, 3)}
            norms = {site: [] for site in (1, 2, 3)}
            effects = {site: {strength: {1: [], -1: []} for strength in (0.25, 0.75)}
                       for site in (1, 2, 3)}
            misses = 0
            for checked, strength, polarity in checked_rows:
                first = checked["first_unequal"]
                if first is None:
                    misses += 1
                else:
                    votes[first.split(".")[1]] += 1
                for site in (1, 2, 3):
                    norms[site].append(checked["comparison"]["l2_by_block"][site])
                    effects[site][strength][polarity].append(checked["forward_effects"][site - 1])
            means = {str(site): statistics.fmean(norms[site]) for site in (1, 2, 3)}
            first = max((1, 2, 3), key=lambda site: (votes[str(site)], -site)) if sum(votes.values()) else None
            magnitude = max((1, 2, 3), key=lambda site: (means[str(site)], -site)) if any(means.values()) else None
            functions = {}
            for site in (1, 2, 3):
                functions[str(site)] = {}
                for strength in (0.25, 0.75):
                    positive, negative = effects[site][strength][1], effects[site][strength][-1]
                    ordered = [value for pair in zip(positive, negative) for value in pair]
                    functions[str(site)][str(strength)] = {"positive": statistics.fmean(positive),
                        "negative": statistics.fmean(negative), "constant": statistics.fmean(ordered), "zero": 0.0}
            expected = {"first_changed": first, "magnitude": magnitude, "first_votes": votes,
                "mean_l2": means, "observation_misses": misses,
                "no_injection": {"cases": 16, "false_detections": 0, "first_changed": None, "magnitude": None},
                "random": {"sites": [1, 2, 3], "each_probability": "1/3"}, "predictions": functions}
            _require_same({key: selection[key] for key in expected}, expected, "discovery selection")

        d_iter = iter(discovery_chunks)
        def checked_discovery():
            for cell_index in range(54):
                try:
                    raw = next(d_iter)
                except StopIteration as error:
                    raise ValueError("a discovery chunk is missing") from error
                chunk = check_chunk(raw, cell_index, "discovery")
                observations = [check_record(row, cell_index, position, "discovery")
                                for position, row in enumerate(chunk["records"])]
                verify_selection(cells[cell_index]["selection"], observations)
                discovery_hashes.append({"name": chunk["name"], "sha256": _sha(raw)})
                yield raw
            try:
                next(d_iter)
            except StopIteration:
                return
            raise ValueError("unexpected discovery chunk")

        # This separately reviewed binder checks the full private metadata and
        # all sanitized measurement-to-chunk links; arithmetic is also rebuilt above.
        verify_discovery_evidence(snapshot, cells, checked_discovery())

        h_iter = iter(heldout_chunks)
        reconstructed_cells, reconstructed_conditions, condition_groups = [], [], {}
        for cell_index, cell in enumerate(cells):
            try:
                raw = next(h_iter)
            except StopIteration as error:
                raise ValueError("a held-out chunk is missing") from error
            chunk = check_chunk(raw, cell_index, "heldout")
            heldout_hashes.append({"name": chunk["name"], "sha256": _sha(raw)})
            assignment = assignments[cell_index]
            block = int(assignment["injection_node"].split(".")[1])
            by_site, outputs = [[], [], []], []
            misses = 0
            for position, row in enumerate(chunk["records"]):
                checked, strength, polarity = check_record(row, cell_index, position, "heldout")
                misses += int(checked["first_unequal"] is None)
                site_results = []
                for site in (1, 2, 3):
                    pred = prediction_rows[cell_index * 96 + position * 3 + site - 1]
                    effect = checked["forward_effects"][site - 1]
                    value = {"site": site, "strength": strength, "polarity": polarity, "effect": effect,
                        "reverse_effect": checked["reverse_effects"][site - 1],
                        "polarity_prediction": pred["polarity_prediction"],
                        "constant_prediction": pred["constant_prediction"], "zero_prediction": pred["zero_prediction"],
                        "residual": pred["polarity_prediction"] - effect,
                        "constant_residual": pred["constant_prediction"] - effect,
                        "zero_residual": pred["zero_prediction"] - effect,
                        "repair_absolute_residual": abs(effect - checked["target_repair"])}
                    by_site[site - 1].append(value)
                    site_results.append(dict(value))
                out = {"model_seed": row["model_seed"], "cell_id": row["cell_id"], "case_id": row["case_id"],
                    "condition_id": row["condition_id"], "case_index": row["case_index"], "strength": strength,
                    "polarity": polarity, "input_sha256": row["input_sha256"],
                    "true_target": assignment["injection_node"], "comparison": checked["comparison"],
                    "logits": row["logits"], "target_repair": checked["target_repair"], "sites": site_results}
                outputs.append(out)
                reconstructed_conditions.append(out)
            selected = {name: cell["selection"][name] for name in ("first_changed", "magnitude")}
            reconstructed_cells.append({"model_seed": cell["model_seed"], "cell_id": cell["cell_id"],
                "true_block": block, "true_target": assignment["injection_node"], "selected": selected,
                "site_rows": by_site, "site_metrics": [_site_metrics(rows) for rows in by_site],
                "observation_misses": misses, "chunk_name": chunk["name"], "chunk_sha256": _sha(raw)})
            condition_groups[cell["cell_id"]] = outputs
        try:
            next(h_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("unexpected held-out chunk")
        if len(clean_owners) != 192 or len(condition_ids) != 3456:
            raise ValueError("paired model/input inventory is incomplete")
        models, equal = _summaries(reconstructed_cells, frozen)
        repair = []
        for cell in reconstructed_cells:
            for strength in (0.25, 0.75):
                for polarity in (1, -1):
                    rows = [row for row in condition_groups[cell["cell_id"]]
                            if row["strength"] == strength and row["polarity"] == polarity]
                    for site in (1, 2, 3):
                        values = [row["sites"][site - 1] for row in rows]
                        residuals = [row["repair_absolute_residual"] for row in values]
                        if len(residuals) != 8:
                            raise ValueError("repair group is incomplete")
                        metrics = _site_metrics(values)
                        repair.append({"model_seed": cell["model_seed"], "cell_id": cell["cell_id"],
                            "true_target": cell["true_target"], "true_block": cell["true_block"],
                            "injection_operation": cell["true_target"].split(".", 2)[2], "strength": strength,
                            "polarity": polarity, "site": site, "count": len(residuals),
                            "mean_absolute_residual": statistics.fmean(residuals),
                            "maximum_absolute_residual": max(residuals), "prediction_mae": metrics,
                            "constant_improvement": metrics["constant_mae"] - metrics["polarity_mae"],
                            "zero_improvement": metrics["zero_mae"] - metrics["polarity_mae"]})
        public_cells = []
        for cell in reconstructed_cells:
            public_cell = {key: value for key, value in cell.items() if key != "site_rows"}
            public_cell["localization"] = {name: {"selected": site, "abstained": site is None,
                                                   "hit": site == cell["true_block"]}
                                           for name, site in cell["selected"].items()}
            public_cells.append(public_cell)
        classification = exposed["classification"]
        interpretation = ("exploratory_unblinded" if classification == "unblinded" else
                          "development_only" if pre["mode"] == "development" else
                          "unconfirmed_exposure_unknown" if classification == "unknown" else
                          "confirmation_conditional_on_caller_attestations")
        expected_results = {"schema": "synthetic-v2-heldout-summary-v1", "status": "complete",
            "mode": pre["mode"], "interpretation": interpretation, "exposure_classification": classification,
            "pre_manifest_sha256": _sha(manifest), "freeze_sha256": _sha(freeze),
            "prediction_sha256": prediction_sha256, "models": models, "equal_model_means": equal,
            "cells": public_cells, "conditions_count": len(reconstructed_conditions),
            "conditions": reconstructed_conditions, "repair_groups": repair}
        _require_same(reported, expected_results, "held-out result")
        return {"schema": "synthetic-v2-independent-saved-audit-v1", "status": "saved_data_consistent",
                "discovery_chunks": len(discovery_hashes), "heldout_chunks": len(heldout_hashes),
                "conditions_per_phase": 1728, "reported_exposure": classification,
                "execution_or_chronology_proven": False, "secrecy_proven": False,
                "case_generation_or_exclusions_proven": False, "manifest_sha256": _sha(manifest),
                "freeze_sha256": _sha(freeze), "prediction_sha256": prediction_sha256,
                "prediction_commitment_sha256": _sha(commitment), "reveal_sha256": _sha(reveal),
                "results_sha256": _sha(results), "discovery_artifacts": discovery_hashes,
                "heldout_artifacts": heldout_hashes}
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError, ZeroDivisionError) as error:
        raise ValueError("invalid or inconsistent saved study evidence") from error


def audit_run(path):
    """Read and independently audit a completed, externally quiescent saved run.

    Both registries and every artifact are bounded and checked again after the
    replay. The caller must keep the files private and unchanged throughout;
    hashes and terminal records do not establish execution or publication time.
    """
    import os
    from pathlib import Path
    import stat
    from . import registry
    from .causal_v2_evidence_adapter import CELL_CHUNK_LIMIT

    root = Path(path)
    if not root.is_absolute():
        raise ValueError("saved run path must be absolute")
    stages = ("before_discovery", "after_discovery", "after_heldout", "before_success")
    public_limits = {"pre_manifest.json": 512 * 1024, "prediction_freeze.json": 32 * 1024 ** 2,
                     "prediction_commitment.json": 16 * 1024, "results.json": 32 * 1024 ** 2,
                     "reveal.json": 16 * 1024 ** 2}
    public_limits.update({f"source_inventory_{stage}.json": 1024 ** 2 for stage in stages})
    private_limits = {"private_material.json": 4 * 1024 ** 2, "prediction_record.json": 16 * 1024 ** 2,
                      "exposure_record.json": 16 * 1024}
    chunk_names = [f"cell-{index:02d}-{suffix}.json" for index in range(54) for suffix in ("d", "h")]
    private_limits.update({name: CELL_CHUNK_LIMIT for name in chunk_names})

    def checked_stat(item, *, directory=False, maximum=None):
        info = item.lstat()
        if not directory and info.st_nlink != 1:
            raise ValueError("saved registry contains a hard link")
        if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                or (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode))
                or (maximum is not None and info.st_size > maximum)):
            raise ValueError("saved run contains an invalid file or directory")
        return info

    def read_file(item, limit):
        before = checked_stat(item, maximum=limit)
        with item.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(limit + 1)
            finished = os.fstat(stream.fileno())
        after = checked_stat(item, maximum=limit)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_nlink")
        if (len(raw) > limit or len(raw) != before.st_size
                or any(tuple(getattr(info, field) for field in fields) !=
                       tuple(getattr(before, field) for field in fields)
                       for info in (opened, finished, after))):
            raise ValueError("saved file changed while reading")
        return raw

    def check_layout(directory, limits):
        checked_stat(directory, directory=True)
        if {item.name for item in directory.iterdir()} != {".lock", "manifest.json", "outcome.json", "artifacts"}:
            raise ValueError("saved registry has missing or unexpected entries")
        checked_stat(directory / "artifacts", directory=True)
        checked_stat(directory / ".lock", maximum=registry.MAX_METADATA_BYTES)
        for name in ("manifest.json", "outcome.json"):
            checked_stat(directory / name, maximum=registry.MAX_METADATA_BYTES)
        if {item.name for item in (directory / "artifacts").iterdir()} != set(limits):
            raise ValueError("saved registry has missing or unexpected artifacts")
        total = 0
        for name, maximum in limits.items():
            total += checked_stat(directory / "artifacts" / name, maximum=maximum).st_size
        if total > registry.MAX_TOTAL_BYTES:
            raise ValueError("saved registry exceeds the artifact budget")

    def registry_state(directory, limits):
        check_layout(directory, limits)
        try:
            state = registry.verify_run(directory)
        except registry.RegistryIntegrityError as error:
            raise ValueError("saved registry integrity failed") from error
        if (state.get("state") != "succeeded" or state.get("pending_write") is not False
                or set(state["outcome"]["artifacts"]) != set(limits)):
            raise ValueError("saved registry is incomplete")
        if (set(state["manifest"]["required_artifacts"]) != set(limits)
                or set(state["manifest"]["artifact_limits"]) != set(limits) | {"failure.json"}):
            raise ValueError("saved registry declarations differ")
        return state

    checked_stat(root, directory=True)
    if {item.name for item in root.iterdir()} != {"public", "private"}:
        raise ValueError("saved run root has missing or unexpected entries")
    public_dir, private_dir = root / "public", root / "private"
    public_state = registry_state(public_dir, public_limits)
    private_state = registry_state(private_dir, private_limits)

    def artifact(directory, state, name, limit):
        raw = read_file(directory / "artifacts" / name, limit)
        if state["outcome"]["artifacts"].get(name) != {"bytes": len(raw), "sha256": _sha(raw)}:
            raise ValueError("saved artifact differs from its terminal inventory")
        return raw

    public_bytes = {name: artifact(public_dir, public_state, name, limit)
                    for name, limit in public_limits.items()}
    private_outcome = read_file(private_dir / "outcome.json", registry.MAX_METADATA_BYTES)
    public_outcome = read_file(public_dir / "outcome.json", registry.MAX_METADATA_BYTES)
    expected_private_details = {"scope": "private evidence", "submissions": 57408, "root_forwards": 57408}
    expected_public_details = {"scope": "complete study", "submissions": 57408, "root_forwards": 57408,
                               "private_outcome_sha256": _sha(private_outcome)}
    _require_same(private_state["outcome"]["details"], expected_private_details, "private terminal")
    _require_same(public_state["outcome"]["details"], expected_public_details, "public terminal")
    _require_same(_object(private_outcome, registry.MAX_METADATA_BYTES, ascii_only=False),
                  private_state["outcome"], "private terminal bytes")
    _require_same(_object(public_outcome, registry.MAX_METADATA_BYTES, ascii_only=False),
                  public_state["outcome"], "public terminal bytes")

    manifest = public_bytes["pre_manifest.json"]
    pre = _object(manifest, public_limits["pre_manifest.json"])
    prep_fields = ("mode", "context", "context_sha256", "pilot_exclusions_sha256",
                   "development_exclusions_sha256", "historical_exclusions_sha256",
                   "historical_exclusions_count", "discovery_cases", "heldout_commitment",
                   "schedule_commitment")
    try:
        prep = _canonical({key: pre[key] for key in prep_fields})
        expected_identities = {"inputs": {"public_preparation_sha256": _sha(prep)},
                               "model": {"states": pre["context"]["model_state_hashes"]},
                               "code": {"source_sha256": pre["source_sha256"]}}
        protocol = {"id": "synthetic-causal-v2", "pre_manifest_sha256": _sha(manifest)}
        for state, is_private in ((public_state, False), (private_state, True)):
            identities = state["manifest"]["identities"]
            for name, expected in expected_identities.items():
                _require_same(identities[name]["value"], expected, "registry identity")
            actual_protocol = identities["protocol"]["value"]
            if is_private:
                if set(actual_protocol) != {"id", "pre_manifest_sha256", "v2_evidence"}:
                    raise ValueError("private registry identity differs")
                config = actual_protocol["v2_evidence"]
                if (type(config) is not dict or set(config) != {"schema", "chunks", "external_reserve_bytes", "disk_limit_bytes"}
                        or config["schema"] != "causal-v2-evidence-layout-v1"
                        or config["chunks"] != sorted(chunk_names)
                        or type(config["external_reserve_bytes"]) is not int or config["external_reserve_bytes"] < 0
                        or type(config["disk_limit_bytes"]) is not int or not 0 < config["disk_limit_bytes"] <= 1024 ** 3):
                    raise ValueError("private evidence layout differs")
                actual_protocol = {key: actual_protocol[key] for key in protocol}
            _require_same(actual_protocol, protocol, "registry protocol")
            _require_same(state["manifest"]["metadata"], {"scope": "one-shot cooperative custodian"},
                          "registry metadata")
        previous = {}
        inventory_hashes = {}
        for stage in stages:
            name = f"source_inventory_{stage}.json"
            saved = _object(public_bytes[name], public_limits[name])
            source = saved.get("source_sha256")
            if (set(saved) != {"schema", "stage", "pre_manifest_sha256", "source_sha256"}
                    or saved["schema"] != "synthetic-v2-source-inventory-v1" or saved["stage"] != stage
                    or saved["pre_manifest_sha256"] != _sha(manifest) or type(source) is not dict
                    or not 0 < len(source) <= 4096
                    or any(type(key) is not str or type(value) is not str
                           or pre["source_sha256"].get(key) != value for key, value in source.items())
                    or any(source.get(key) != value for key, value in previous.items())):
                raise ValueError("source inventory differs")
            previous = source
            inventory_hashes[name] = _sha(public_bytes[name])
        private_bytes = {name: artifact(private_dir, private_state, name, private_limits[name])
                         for name in ("private_material.json", "prediction_record.json", "exposure_record.json")}
        def chunks(suffix):
            for index in range(54):
                name = f"cell-{index:02d}-{suffix}.json"
                yield artifact(private_dir, private_state, name, private_limits[name])

        result = audit_evidence(manifest=manifest, freeze=public_bytes["prediction_freeze.json"],
            prediction=private_bytes["prediction_record.json"], commitment=public_bytes["prediction_commitment.json"],
            reveal=public_bytes["reveal.json"], exposure=private_bytes["exposure_record.json"],
            private_material=private_bytes["private_material.json"], results=public_bytes["results.json"],
            discovery_chunks=chunks("d"), heldout_chunks=chunks("h"))
        _require_same(registry_state(public_dir, public_limits), public_state, "public registry after replay")
        _require_same(registry_state(private_dir, private_limits), private_state, "private registry after replay")
        if {item.name for item in root.iterdir()} != {"public", "private"}:
            raise ValueError("saved run root changed during replay")
        if (read_file(public_dir / "outcome.json", registry.MAX_METADATA_BYTES) != public_outcome
                or read_file(private_dir / "outcome.json", registry.MAX_METADATA_BYTES) != private_outcome):
            raise ValueError("terminal record changed during replay")
        result.update({"public_outcome_sha256": _sha(public_outcome), "private_outcome_sha256": _sha(private_outcome),
                       "source_inventory_sha256": inventory_hashes})
        return result
    except (KeyError, TypeError, IndexError, RecursionError) as error:
        raise ValueError("invalid saved registry bindings") from error


def main(argv=None):
    """Print a bounded canonical report without disclosing failure tracebacks."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Audit a completed synthetic-v2 saved run")
    parser.add_argument("run", help="absolute path to the quiescent saved run")
    arguments = parser.parse_args(argv)
    try:
        report = audit_run(arguments.run)
    except Exception:
        print("Saved audit incomplete or inconsistent; retain the evidence for private review.", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(_canonical(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
