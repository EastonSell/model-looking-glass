"""Audit the saved synthetic causal v1 evidence without running a model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct

from .registry import RegistryIntegrityError, verify_run
from .research_cases import (DISCOVERY_CASES, HELDOUT_CASES, SYNTHETIC_CASES,
                             synthetic_content_sha256, synthetic_suite_sha256)


PUBLISHED_MANIFEST = "869dc68521f149cf34de918de7db45908ceb36804f65f126269aa1c380412190"
PUBLISHED_PRECOMMIT = "f5a3719b0b95b1436f3831956af25d4a14723d7ffd9d186453689ad616f288f0"
PUBLISHED_RESULTS = "298e35840226e21cb36beff2f0092c64ed9502c88f3c758bc5f9b3336c77d4a2"
PUBLISHED_SUITE = "883b28112ccee7605125959a4c8209c9e20bb438b73938b323e3fc80994f1e8e"
PUBLISHED_PROTOCOL = "0631d7543e2230c33c66ae41d1ddc420caf234b89762553c9c992cc93a196549"
MODEL_STATE = "efe24b13a22bb86112420db9f2e466736d487d1d3cb652ac89f7a5c60c97342d"
KINDS = ("first_observed", "magnitude", "random")
ZERO_TOLERANCE = 1e-7


class EvidenceError(ValueError):
    """The evidence is incomplete or contradicts the recorded protocol."""


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(actual) == set(expected)
                and all(_exact(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise EvidenceError(f"{label} must be a finite number")
    return float(value)


def _same_number(actual, expected, label):
    value = _number(actual, label)
    if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise EvidenceError(f"{label} contradicts recomputed value")


def _f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _logit_metric(logits, label):
    _require(isinstance(logits, list) and len(logits) == 1
             and isinstance(logits[0], list) and len(logits[0]) == 2,
             f"{label} must contain one pair of logits")
    a = _number(logits[0][0], label)
    b = _number(logits[0][1], label)
    _require(_f32(a) == a and _f32(b) == b, f"{label} is not exactly float32")
    return _f32(b - a)


def _case_ids_and_hashes(cases):
    return [{"case_id": case.case_id, "content_sha256": synthetic_content_sha256(case.values)}
            for case in cases]


def _expected_metrics(pairs):
    errors = [abs(predicted - observed) for _, predicted, observed in pairs]
    null_errors = [abs(observed) for _, _, observed in pairs]
    mae = statistics.fmean(errors)
    rmse = math.sqrt(statistics.fmean(x * x for x in errors))
    null_mae = statistics.fmean(null_errors)
    null_rmse = math.sqrt(statistics.fmean(x * x for x in null_errors))
    eligible = [(predicted, observed) for _, predicted, observed in pairs
                if abs(observed) > ZERO_TOLERANCE]
    def sign(x):
        return 1 if x > ZERO_TOLERANCE else -1 if x < -ZERO_TOLERANCE else 0
    accuracy = (sum(sign(predicted) == sign(observed) for predicted, observed in eligible) / len(eligible)
                if eligible else None)
    null_accuracy = 0.0 if eligible else None
    point = {"mae": mae, "rmse": rmse, "null_mae": null_mae, "null_rmse": null_rmse,
             "mae_improvement_over_null": null_mae - mae,
             "rmse_improvement_over_null": null_rmse - rmse,
             "sign_accuracy": accuracy, "null_sign_accuracy": null_accuracy,
             "sign_accuracy_improvement_over_null": accuracy if eligible else None,
             "prediction_interval_coverage": None, "prediction_interval_coverage_minus_nominal": None}
    return point, len(eligible)


def _check_summary(summary, pairs, label):
    expected, eligible = _expected_metrics(pairs)
    _require(_exact(summary["n_cases"], len(pairs)) and _exact(summary["zero_tolerance"], ZERO_TOLERANCE),
             f"{label} has wrong count or tolerance")
    _require(summary["prediction_interval_level"] is None, f"{label} reports a nominal interval")
    _require(_exact(summary["directional_cases"], {"included": eligible, "excluded": len(pairs) - eligible}),
             f"{label} has wrong directional counts")
    _require(set(summary["point"]) == set(expected), f"{label} point metric fields differ")
    for name, expected_value in expected.items():
        actual = summary["point"][name]
        if expected_value is None:
            _require(actual is None, f"{label}.{name} must be unavailable")
        else:
            _same_number(actual, expected_value, f"{label}.{name}")
    errors = summary["case_errors"]
    _require(len(errors) == len(pairs), f"{label} case error count differs")
    for actual, (case_id, predicted, observed) in zip(errors, sorted(pairs)):
        _require(actual["case_id"] == case_id, f"{label} case error ID differs")
        _same_number(actual["absolute_error"], abs(predicted - observed), f"{label} absolute error")
        _same_number(actual["null_absolute_error"], abs(observed), f"{label} null error")
    bootstrap = summary["bootstrap"]
    expected_bootstrap = {"unit": "independent_case", "method": "none", "samples": 0,
                          "confidence": 0.95, "seed": 0,
                          "intervals": {name: None for name in expected},
                          "defined_samples": {name: 0 for name in expected},
                          "undefined_samples": {name: 0 for name in expected}}
    _require(_exact(bootstrap, expected_bootstrap),
             f"{label} must not report bootstrap intervals")
    return expected


def _audit_payload(manifest, frozen, result):
    identities = manifest["identities"]
    protocol = identities["protocol"]["value"]
    inputs = identities["inputs"]["value"]
    model = identities["model"]["value"]
    code = identities["code"]["value"]
    _require(protocol["id"] == "synthetic-causal-v1" and protocol["sha256"] == PUBLISHED_PROTOCOL,
             "wrong protocol identity")
    _require(_exact(manifest["metadata"], {"mode": "recorded-v1", "evidence_status": "exploratory"}),
             "wrong run classification")
    _require(_exact(model, {"kind": "TinyResidual", "width": 8, "depth": 3,
                            "seed": 2301, "state_sha256": MODEL_STATE}),
             "wrong model identity")
    _require(code["revision"] == "6d0131ee3eb6c2bb7dbef9433f0a8733c759331f"
             and code["dirty"] is False
             and code["clarification_sha256"] == "b44dd544de80021b01049be6201239591f7399f0304679fa9d2b0730f5404918",
             "wrong source revision or dirty source")
    settings = protocol["settings"]
    expected_settings = {"model_seed": 2301, "random_seed": 918, "route": 0,
                         "injection_node": "block.1.activation", "injection_element": [0, 0],
                         "injection_delta": 1.0, "upstream_control": "block.0.output", "invocation": 0,
                         "source_policy": "same-input clean activation", "operation": "replace whole output",
                         "capture": "all active dense taps, direct", "max_trace_events": 12,
                         "max_trace_tensor_bytes": 384, "zero_tolerance": ZERO_TOLERANCE,
                         "metric": "output[0,1] - output[0,0]", "threads": 1}
    _require(_exact(settings, expected_settings), "protocol settings differ")
    _require(synthetic_suite_sha256(SYNTHETIC_CASES) == PUBLISHED_SUITE
             and inputs["suite_sha256"] == PUBLISHED_SUITE, "wrong fixture suite")
    expected_input_cases = [_case_ids_and_hashes([case])[0] | {"split": case.split}
                            for case in SYNTHETIC_CASES]
    _require(_exact(inputs["cases"], expected_input_cases), "manifest case identities differ")
    discovery_cases, heldout_cases = _case_ids_and_hashes(DISCOVERY_CASES), _case_ids_and_hashes(HELDOUT_CASES)
    _require(len(discovery_cases) == 8 and len(heldout_cases) == 16,
             "unexpected discovery or held-out count")
    _require(_exact(frozen["discovery_cases"], discovery_cases) and _exact(frozen["heldout_cases"], heldout_cases),
             "frozen cases differ from suite")
    _require(frozen["suite_sha256"] == result["suite_sha256"] == PUBLISHED_SUITE,
             "suite hashes differ")
    _require(frozen["protocol"] == result["protocol"] == "synthetic-causal-v1"
             and frozen["mode"] == result["mode"] == "recorded-v1"
             and frozen["evidence_status"] == result["evidence_status"] == "exploratory",
             "recorded/exploratory status differs")
    _require(_exact(frozen["model_seed"], 2301) and _exact(frozen["random_seed"], 918)
             and _exact(frozen["zero_tolerance"], ZERO_TOLERANCE) and _exact(frozen["zero_prediction"], 0.0),
             "frozen seed or zero baseline differs")
    _require(_exact(frozen["injection"], {"node_id": "block.1.activation", "element": [0, 0], "delta": 1.0})
             and frozen["upstream_control"] == "block.0.output"
             and frozen["metric"] == settings["metric"]
             and frozen["source"] == "same-input clean activation at selected tap, invocation 0"
             and frozen["base"] == "same-input corrupt run"
             and frozen["operation"] == "replace complete tap output", "intervention definition differs")
    expected_controls = {"null": "corrupt base at first-observed site, null operation",
                         "identity_patch": "corrupt source and corrupt base at first-observed site",
                         "upstream_patch": "clean source and corrupt base at block.0.output",
                         "reverse_patch": "corrupt source and clean base at first-observed site",
                         "clean_null": "clean base with null operation at first-observed site",
                         "clean_trace": "two clean dense traces on the same input"}
    _require(_exact(frozen["controls"], expected_controls), "frozen controls differ")
    expected_comparisons = ["prediction versus own-site patch effect and same-site zero prediction",
                            "prediction versus clean-minus-corrupt target correction",
                            "patch effect versus target correction and zero correction"]
    _require(_exact(frozen["comparisons"], expected_comparisons), "frozen comparisons differ")
    frozen_bytes = json.dumps(frozen, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _require(result["precommit_sha256"] == hashlib.sha256(frozen_bytes).hexdigest(),
             "results do not bind the frozen precommit")
    sites = frozen["sites"]
    _require(set(sites) == set(KINDS) and sites == result["sites"], "site identities differ")
    active = [f"block.{i}.{suffix}" for i in range(3)
              for suffix in ("branch.0", "activation", "merge", "output")]
    _require(all(site in active for site in sites.values()), "selected site is not active")
    _require(sites["random"] == random.Random(918).choice(active), "random-site draw differs")
    _require(result["known_injection"] == "block.1.activation"
             and result["first_observed_site_matches_injection"] == (sites["first_observed"] == "block.1.activation"),
             "injection comparison differs")
    _require(set(frozen["predictions"]) == set(KINDS), "prediction methods differ")
    predictions = {}
    for kind in KINDS:
        prediction = frozen["predictions"][kind]
        site = sites[kind]
        block, suffix = site.split(".")[1], ".".join(site.split(".")[2:])
        module = f"blocks.{block}.branch_taps.0" if suffix == "branch.0" else f"blocks.{block}.{suffix}_tap"
        _require(prediction["site"] == site and prediction["module_path"] == module
                 and _exact(prediction["invocation"], 0), f"{kind} coordinate differs")
        samples = prediction["discovery_effects"]
        _require([row["case_id"] for row in samples] == [case["case_id"] for case in discovery_cases],
                 f"{kind} discovery IDs differ")
        values = [_number(row["effect"], f"{kind} discovery effect") for row in samples]
        predictions[kind] = statistics.fmean(values)
        _same_number(prediction["predicted_effect"], predictions[kind], f"{kind} frozen mean")
        _same_number(prediction["descriptive_min"], min(values), f"{kind} minimum")
        _same_number(prediction["descriptive_max"], max(values), f"{kind} maximum")
    coincident_pairs = [(left, right) for index, left in enumerate(KINDS) for right in KINDS[index + 1:]
                        if sites[left] == sites[right]]
    discovery_delta = max((abs(a["effect"] - b["effect"])
                           for left, right in coincident_pairs
                           for a, b in zip(frozen["predictions"][left]["discovery_effects"],
                                           frozen["predictions"][right]["discovery_effects"])), default=0.0)
    rows = result["cases"]
    _require([row["case_id"] for row in rows] == [case["case_id"] for case in heldout_cases],
             "held-out case IDs or ordering differ")
    comparisons = {kind: {"intervention_prediction": [], "target_prediction": [], "repair_residual": []}
                   for kind in KINDS}
    reverse_errors, coincident_effect_deltas, coincident_metric_deltas, all_site_effect_deltas = [], [], [], []
    for row in rows:
        case_id = row["case_id"]
        clean = _logit_metric(row["clean_logits"], f"{case_id} clean logits")
        corrupt = _logit_metric(row["corrupt_logits"], f"{case_id} corrupt logits")
        _same_number(row["clean_metric"], clean, f"{case_id} clean metric")
        _same_number(row["corrupt_metric"], corrupt, f"{case_id} corrupt metric")
        target = clean - corrupt
        _same_number(row["target_correction"], target, f"{case_id} correction")
        controls = row["controls"]
        for name in ("null_effect", "identity_patch_effect", "upstream_patch_effect", "clean_null_effect"):
            _require(_number(controls[name], f"{case_id} {name}") == 0.0,
                     f"{case_id} scalar control differs")
        _number(controls["reverse_patch_effect"], f"{case_id} reverse patch effect")
        _require(controls["clean_trace_difference"] is None, f"{case_id} clean trace control differs")
        _require(set(row["interventions"]) == set(KINDS), f"{case_id} intervention methods differ")
        for kind in KINDS:
            intervention = row["interventions"][kind]
            _require(intervention["site"] == sites[kind], f"{case_id} {kind} site differs")
            _same_number(intervention["predicted_effect"], predictions[kind], f"{case_id} {kind} prediction")
            effect = _number(intervention["effect"], f"{case_id} {kind} effect")
            patched = _number(intervention["patched_metric"], f"{case_id} {kind} patched metric")
            _same_number(effect, patched - corrupt, f"{case_id} {kind} effect")
            comparisons[kind]["intervention_prediction"].append((case_id, predictions[kind], effect))
            comparisons[kind]["target_prediction"].append((case_id, predictions[kind], target))
            comparisons[kind]["repair_residual"].append((case_id, effect, target))
        interventions = row["interventions"]
        reverse_errors.append(abs(controls["reverse_patch_effect"] + interventions["first_observed"]["effect"]))
        for left, right in coincident_pairs:
            coincident_effect_deltas.append(abs(interventions[left]["effect"] - interventions[right]["effect"]))
            coincident_metric_deltas.append(abs(interventions[left]["patched_metric"] - interventions[right]["patched_metric"]))
        all_site_effect_deltas.append(max(interventions[kind]["effect"] for kind in KINDS)
                                      - min(interventions[kind]["effect"] for kind in KINDS))
    _require(set(result["summaries"]) == set(KINDS), "summary methods differ")
    recomputed = {}
    for kind in KINDS:
        _require(set(result["summaries"][kind]) == set(comparisons[kind]), f"{kind} summary types differ")
        checked = {name: _check_summary(result["summaries"][kind][name], pairs, f"{kind}.{name}")
                   for name, pairs in comparisons[kind].items()}
        recomputed[kind] = {"site": sites[kind],
                            "intervention_prediction_mae": checked["intervention_prediction"]["mae"],
                            "same_site_zero_mae": checked["intervention_prediction"]["null_mae"],
                            "repair_residual_mae": checked["repair_residual"]["mae"]}
    return {"status": "consistent", "protocol": "synthetic-causal-v1", "evidence_status": "exploratory",
            "heldout_cases": len(rows), "summaries": recomputed,
            "reverse_opposition_max_error": max(reverse_errors),
            "coincident_site_discovery_max_delta": discovery_delta,
            "coincident_site_effect_max_delta": max(coincident_effect_deltas, default=0.0),
            "coincident_site_metric_max_delta": max(coincident_metric_deltas, default=0.0),
            "all_site_effect_max_delta": max(all_site_effect_deltas),
            "not_verified": ["discovery trace selection", "full control logits", "prior blindness and ordering",
                             "truth of saved model executions", "unique causal mechanism"]}


def audit_payload(manifest, frozen, result):
    """Recompute recorded arithmetic; this does not authenticate JSON payloads."""
    try:
        return _audit_payload(manifest, frozen, result)
    except EvidenceError:
        raise
    except (KeyError, IndexError, AttributeError, TypeError, ValueError, OverflowError, struct.error) as error:
        raise EvidenceError(f"malformed evidence: {error}") from error


def audit_run(path):
    """Verify the published v1 artifact identity and semantic consistency."""
    path = Path(path)
    try:
        verified = verify_run(path)
        _require(verified["state"] == "succeeded", "run did not succeed")
        outcome = verified["outcome"]
        _require(outcome["manifest_sha256"] == PUBLISHED_MANIFEST, "published manifest digest differs")
        precommit = (path / "artifacts" / "precommit.json").read_bytes()
        results = (path / "artifacts" / "results.json").read_bytes()
        _require(hashlib.sha256(precommit).hexdigest() == PUBLISHED_PRECOMMIT,
                 "published precommit digest differs")
        _require(hashlib.sha256(results).hexdigest() == PUBLISHED_RESULTS,
                 "published results digest differs")
        frozen = json.loads(precommit)
        result = json.loads(results)
        _require(result["precommit_sha256"] == PUBLISHED_PRECOMMIT, "result does not bind precommit")
        report = audit_payload(verified["manifest"], frozen, result)
        _require(_exact(outcome["details"], {"heldout_cases": report["heldout_cases"]}),
                 "terminal outcome case count differs")
        _require(all(report[name] == 0.0 for name in ("reverse_opposition_max_error",
                     "coincident_site_discovery_max_delta", "coincident_site_effect_max_delta",
                     "coincident_site_metric_max_delta", "all_site_effect_max_delta")),
                 "published report's observed equality claims differ")
        report["artifact_identity"] = "published-v1-digests"
        return report
    except EvidenceError:
        raise
    except (RegistryIntegrityError, OSError, KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"unable to verify run: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="saved synthetic causal v1 run directory")
    args = parser.parse_args()
    print(json.dumps(audit_run(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
