"""Prespecified CPU-only causal intervention experiment."""

from collections import Counter
import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from . import causal_metrics, interventions, model as model_module, probes, registry, research_cases
from .causal_metrics import EffectCase, evaluate_effects
from .interventions import ActivationIntervention
from .model import TinyResidual
from .probes import TapPoint, TraceSession
from .registry import create_run, sha256_file
from .research_cases import (SYNTHETIC_CASES, float32_values,
                             synthetic_content_sha256, synthetic_suite_sha256,
                             validate_synthetic_cases)


PROTOCOL_ID = "synthetic-causal-v1"
MODEL_SEED = 2301
RANDOM_SEED = 918
ZERO_TOLERANCE = 1e-7
INJECTION = ("block.1.activation", 1.0)
UPSTREAM_CONTROL = "block.0.output"
TENSOR_BYTE_LIMIT = 32


def _validate_trace_pair(clean, corrupt, expected_nodes):
    if not clean or len(clean) != len(corrupt):
        raise ValueError("paired traces must have identical nonempty coverage")
    if [event.node_id for event in clean] != expected_nodes:
        raise ValueError("trace is not complete for the expected dense tap policy")
    first, magnitudes = None, []
    for i, (a, b) in enumerate(zip(clean, corrupt)):
        if (a.node_id != b.node_id or a.sequence != i or b.sequence != i
                or a.occurrence != 0 or b.occurrence != 0
                or a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
                or not isinstance(a.value, torch.Tensor)
                or not isinstance(b.value, torch.Tensor)
                or tuple(a.value.shape) != a.shape or tuple(b.value.shape) != b.shape
                or str(a.value.dtype) != a.dtype or str(b.value.dtype) != b.dtype
                or str(a.value.device) != a.device or str(b.value.device) != b.device
                or not a.value.is_floating_point() or not b.value.is_floating_point()
                or not bool(torch.isfinite(a.value).all())
                or not bool(torch.isfinite(b.value).all())):
            raise ValueError("paired trace metadata or values do not match")
        if first is None and not torch.equal(a.value, b.value):
            first = a.node_id
        magnitudes.append(float(torch.linalg.vector_norm(
            a.value.to(torch.float64) - b.value.to(torch.float64))))
    return first, magnitudes


def select_sites(pairs, *, random_seed, expected_nodes):
    """Select sites without access to the injection label or held-out outcomes."""
    if type(random_seed) is not int:
        raise ValueError("random_seed must be an integer")
    labels = list(expected_nodes)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("expected_nodes must be a nonempty unique sequence")
    pairs = list(pairs)
    if not pairs:
        raise ValueError("discovery traces are required")
    votes = Counter()
    magnitudes = [0.0] * len(labels)
    for clean, corrupt in pairs:
        first, pair_magnitudes = _validate_trace_pair(clean, corrupt, labels)
        if first is None:
            raise ValueError("no difference observed on a discovery case")
        votes[first] += 1
        magnitudes = [total + value for total, value in zip(magnitudes, pair_magnitudes)]
    first_observed = max(labels, key=lambda label: votes[label])
    magnitude = labels[max(range(len(labels)), key=lambda i: magnitudes[i])]
    return {"first_observed": first_observed, "magnitude": magnitude,
            "random": random.Random(random_seed).choice(labels)}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def _make_model(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return TinyResidual(width=8, depth=3).eval()


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        header = _json_bytes({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        data = bytes(value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _source_identity(recorded):
    root = Path(__file__).resolve().parents[1]
    modules = (sys.modules[__name__], model_module, probes, interventions,
               causal_metrics, registry, research_cases, sys.modules[__package__])
    source_paths = [Path(module.__file__).resolve() for module in modules]
    sources = {module.__name__: sha256_file(module.__file__) for module in modules}
    protocol_path = root / "docs" / "CAUSAL_PROTOCOL_V1.md"
    protocol_hash = sha256_file(protocol_path)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, "git status unavailable"
    if recorded:
        if revision is None or status:
            raise RuntimeError("recorded protocol-v1 runs require a clean committed worktree")
        if any(not source.is_relative_to(root) for source in source_paths):
            raise RuntimeError("recorded protocol-v1 dependencies must come from this worktree")
        for source in source_paths + [protocol_path]:
            relative = source.relative_to(root).as_posix()
            try:
                committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root,
                                                    stderr=subprocess.DEVNULL)
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError(f"recorded source is not tracked at HEAD: {relative}") from error
            if hashlib.sha256(committed).hexdigest() != sha256_file(source):
                raise RuntimeError(f"recorded source does not match committed bytes: {relative}")
    runtime = {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
               "torch": str(torch.__version__), "torch_build": torch.__config__.show(),
               "platform": platform.platform(), "machine": platform.machine(),
               "processor": platform.processor(), "device": "cpu", "dtype": "float32", "threads": 1,
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "float32_matmul_precision": torch.get_float32_matmul_precision()}
    return root, protocol_hash, {"revision": revision, "dirty": bool(status),
                                 "source_sha256": sources, "runtime": runtime}


def _case_identity(case):
    return {"case_id": case.case_id, "content_sha256": synthetic_content_sha256(case.values)}


def _metric(output):
    return float((output[0, 1] - output[0, 0]).item())


def _trace(model, x, *, corrupt):
    with TraceSession(model, mode="direct", placement="dense", max_events=12,
                      max_tensor_bytes=12 * TENSOR_BYTE_LIMIT) as trace:
        output = model(x, route=0, perturb=INJECTION if corrupt else None)
    return output.detach().clone(), trace.events


def _run_patch_output(model, x, module_path, *, source=None, corrupt, operation="patch"):
    with ActivationIntervention(model, module_path, operation=operation, value=source,
                                invocation=0, max_tensor_bytes=TENSOR_BYTE_LIMIT):
        return model(x, route=0, perturb=INJECTION if corrupt else None).detach().clone()


def _run_patch(model, x, module_path, *, source=None, corrupt, operation="patch"):
    return _metric(_run_patch_output(model, x, module_path, source=source,
                                     corrupt=corrupt, operation=operation))


def _tensor(case):
    return torch.tensor([float32_values(case.values)], dtype=torch.float32, device="cpu")


def _site_modules(model):
    return {module.node_id: path for path, module in model.named_modules()
            if isinstance(module, TapPoint)}


def _events_by_node(events):
    return {event.node_id: event for event in events}


def _discovery(model, cases, random_seed):
    records = []
    for case in cases:
        x = _tensor(case)
        clean_output, clean = _trace(model, x, corrupt=False)
        corrupt_output, corrupt = _trace(model, x, corrupt=True)
        clean_metric, corrupt_metric = _metric(clean_output), _metric(corrupt_output)
        records.append((case, x, clean_metric, corrupt_metric, clean, corrupt))
    sites = select_sites([(item[4], item[5]) for item in records],
                         random_seed=random_seed, expected_nodes=model.active_nodes(0))
    paths = _site_modules(model)
    predictions = {}
    for kind, site in sites.items():
        effects = []
        for case, x, _, corrupt_metric, clean, _ in records:
            value = _events_by_node(clean)[site].value
            patched_metric = _run_patch(model, x, paths[site], source=value, corrupt=True)
            effects.append({"case_id": case.case_id, "effect": patched_metric - corrupt_metric})
        values = [item["effect"] for item in effects]
        predictions[kind] = {"site": site, "module_path": paths[site], "invocation": 0,
                             "predicted_effect": statistics.fmean(values),
                             "descriptive_min": min(values), "descriptive_max": max(values),
                             "discovery_effects": effects}
    return sites, predictions


def _heldout_case(model, case, sites, predictions):
    x = _tensor(case)
    clean_output, clean_events = _trace(model, x, corrupt=False)
    corrupt_output, corrupt_events = _trace(model, x, corrupt=True)
    repeated_clean_output, repeated_clean_events = _trace(model, x, corrupt=False)
    clean_metric, corrupt_metric = _metric(clean_output), _metric(corrupt_output)
    expected = model.active_nodes(0)
    _validate_trace_pair(clean_events, corrupt_events, expected)
    clean_difference, _ = _validate_trace_pair(clean_events, repeated_clean_events, expected)
    if clean_difference is not None or not torch.equal(clean_output, repeated_clean_output):
        raise RuntimeError("clean-versus-clean trace control differs")
    clean, corrupt = _events_by_node(clean_events), _events_by_node(corrupt_events)
    paths = _site_modules(model)
    site_effects = {}
    for kind, site in sites.items():
        patched = _run_patch(model, x, paths[site], source=clean[site].value, corrupt=True)
        site_effects[kind] = {"site": site, "predicted_effect": predictions[kind]["predicted_effect"],
                              "patched_metric": patched, "effect": patched - corrupt_metric}
    selected = sites["first_observed"]
    null_output = _run_patch_output(model, x, paths[selected], corrupt=True, operation="null")
    identity_output = _run_patch_output(model, x, paths[selected], source=corrupt[selected].value, corrupt=True)
    upstream_output = _run_patch_output(model, x, paths[UPSTREAM_CONTROL],
                                        source=clean[UPSTREAM_CONTROL].value, corrupt=True)
    clean_null_output = _run_patch_output(model, x, paths[selected], corrupt=False, operation="null")
    for name, actual, baseline in (("null", null_output, corrupt_output),
                                   ("identity_patch", identity_output, corrupt_output),
                                   ("upstream_patch", upstream_output, corrupt_output),
                                   ("clean_null", clean_null_output, clean_output)):
        if not torch.equal(actual, baseline):
            raise RuntimeError(f"invalid {name} control: full output changed")
    controls = {"null_effect": _metric(null_output) - corrupt_metric,
                "identity_patch_effect": _metric(identity_output) - corrupt_metric,
                "upstream_patch_effect": _metric(upstream_output) - corrupt_metric,
                "reverse_patch_effect": _run_patch(model, x, paths[selected], source=corrupt[selected].value,
                                                    corrupt=False) - clean_metric,
                "clean_null_effect": _metric(clean_null_output) - clean_metric,
                "clean_trace_difference": clean_difference}
    return {"case_id": case.case_id, "clean_metric": clean_metric, "corrupt_metric": corrupt_metric,
            "clean_logits": clean_output.tolist(), "corrupt_logits": corrupt_output.tolist(),
            "target_correction": clean_metric - corrupt_metric, "interventions": site_effects,
            "controls": controls}


def _summarize(rows, sites):
    summaries = {}
    for kind in sites:
        effect_prediction = []
        target_prediction = []
        repair = []
        for row in rows:
            intervention = row["interventions"][kind]
            effect_prediction.append(EffectCase(row["case_id"], intervention["predicted_effect"],
                                                intervention["effect"]))
            target_prediction.append(EffectCase(row["case_id"], intervention["predicted_effect"],
                                                row["target_correction"]))
            repair.append(EffectCase(row["case_id"], intervention["effect"], row["target_correction"]))
        kwargs = {"zero_tolerance": ZERO_TOLERANCE, "bootstrap_samples": 0}
        summaries[kind] = {"intervention_prediction": evaluate_effects(effect_prediction, **kwargs),
                           "target_prediction": evaluate_effects(target_prediction, **kwargs),
                           "repair_residual": evaluate_effects(repair, **kwargs)}
    return summaries


def run_experiment(path, *, cases=None, model_seed=MODEL_SEED, random_seed=RANDOM_SEED):
    """Run frozen v1, or an explicitly labeled development run on other inputs."""
    if type(model_seed) is not int or model_seed < 0 or type(random_seed) is not int:
        raise ValueError("seeds must be integers and the model seed must be nonnegative")
    recorded = cases is None and model_seed == MODEL_SEED and random_seed == RANDOM_SEED
    if cases is None and not recorded:
        raise ValueError("custom seeds require a distinct development case suite")
    cases = tuple(SYNTHETIC_CASES if cases is None else cases)
    validate_synthetic_cases(cases)
    suite_hash = synthetic_suite_sha256(cases)
    reserved = {synthetic_content_sha256(case.values) for case in SYNTHETIC_CASES}
    if not recorded and any(synthetic_content_sha256(case.values) in reserved for case in cases):
        raise ValueError("development inputs must not overlap the reserved protocol-v1 suite")
    if torch.is_autocast_enabled("cpu"):
        raise RuntimeError("CPU autocast is incompatible with the float32 protocol")
    _, protocol_hash, code_identity = _source_identity(recorded)
    model = _make_model(model_seed)
    discovery = [case for case in cases if case.split == "discovery"]
    heldout = [case for case in cases if case.split == "heldout"]
    identities = {"suite_sha256": suite_hash, "cases": [_case_identity(case) | {"split": case.split}
                                                           for case in cases]}
    settings = {"model_seed": model_seed, "random_seed": random_seed, "route": 0,
                "injection_node": INJECTION[0], "injection_element": [0, 0], "injection_delta": INJECTION[1],
                "upstream_control": UPSTREAM_CONTROL, "invocation": 0,
                "source_policy": "same-input clean activation", "operation": "replace whole output",
                "capture": "all active dense taps, direct", "max_trace_events": 12,
                "max_trace_tensor_bytes": 12 * TENSOR_BYTE_LIMIT, "zero_tolerance": ZERO_TOLERANCE,
                "metric": "output[0,1] - output[0,0]", "threads": 1}
    run = create_run(path, protocol={"id": PROTOCOL_ID, "sha256": protocol_hash, "settings": settings},
                     inputs=identities, model={"kind": "TinyResidual", "width": 8, "depth": 3,
                                               "seed": model_seed, "state_sha256": _state_sha256(model)},
                     code=code_identity, metadata={"mode": "recorded-v1" if recorded else "development"},
                     artifact_limits={"precommit.json": 1024 * 1024, "results.json": 1024 * 1024,
                                      "partial_results.json": 1024 * 1024},
                     required_artifacts=["precommit.json", "results.json"], max_total_bytes=3 * 1024 * 1024)
    previous_threads = torch.get_num_threads()
    threads_restored = False
    rows = []
    current_case_id = None
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            sites, predictions = _discovery(model, discovery, random_seed)
            frozen = {"protocol": PROTOCOL_ID, "mode": "recorded-v1" if recorded else "development",
                      "suite_sha256": suite_hash, "discovery_cases": [_case_identity(c) for c in discovery],
                      "heldout_cases": [_case_identity(c) for c in heldout], "sites": sites,
                      "predictions": predictions, "zero_prediction": 0.0, "zero_tolerance": ZERO_TOLERANCE,
                      "model_seed": model_seed, "random_seed": random_seed,
                      "injection": {"node_id": INJECTION[0], "element": [0, 0], "delta": INJECTION[1]},
                      "source": "same-input clean activation at selected tap, invocation 0",
                      "base": "same-input corrupt run", "operation": "replace complete tap output",
                      "metric": "output[0,1] - output[0,0]", "upstream_control": UPSTREAM_CONTROL,
                      "controls": {
                          "null": "corrupt base at first-observed site, null operation",
                          "identity_patch": "corrupt source and corrupt base at first-observed site",
                          "upstream_patch": "clean source and corrupt base at block.0.output",
                          "reverse_patch": "corrupt source and clean base at first-observed site",
                          "clean_null": "clean base with null operation at first-observed site",
                          "clean_trace": "two clean dense traces on the same input"},
                      "comparisons": ["prediction versus own-site patch effect and same-site zero prediction",
                                      "prediction versus clean-minus-corrupt target correction",
                                      "patch effect versus target correction and zero correction"]}
            frozen_bytes = _json_bytes(frozen)
            run.write_artifact("precommit.json", frozen_bytes)
            # No model forward on a held-out input may occur before this publication.
            for case in heldout:
                current_case_id = case.case_id
                rows.append(_heldout_case(model, case, sites, predictions))
            current_case_id = None
            result = {"protocol": PROTOCOL_ID, "mode": frozen["mode"], "suite_sha256": suite_hash,
                      "precommit_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "sites": sites,
                      "known_injection": INJECTION[0],
                      "first_observed_site_matches_injection": sites["first_observed"] == INJECTION[0],
                      "cases": rows, "summaries": _summarize(rows, sites),
                      "limitations": ["one untrained CPU model and one known injection",
                                      "fixed public inputs are not a blind or population sample",
                                      "same-input patches and first differences do not identify a unique mechanism",
                                      "single random-site draw; final-state patch can restore by construction"]}
        # A successful outcome means the caller's process state was restored too.
        torch.set_num_threads(previous_threads)
        threads_restored = True
        run.write_artifact("results.json", _json_bytes(result))
        run.finalize("succeeded", {"heldout_cases": len(rows)})
        return result
    except BaseException as exc:
        restore_error = None
        if not threads_restored:
            try:
                torch.set_num_threads(previous_threads)
                threads_restored = True
            except BaseException as error:
                restore_error = error
        partial_error = None
        if rows:
            try:
                run.write_artifact("partial_results.json", _json_bytes({"state": "incomplete",
                                   "protocol": PROTOCOL_ID, "completed_cases": len(rows),
                                   "failed_case_id": current_case_id, "cases": rows}))
            except Exception as error:
                partial_error = error
        try:
            run.finalize("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                         {"error_type": type(exc).__name__, "message": str(exc),
                          "partial_write_error": str(partial_error) if partial_error else None,
                          "restore_error": str(restore_error) if restore_error else None})
        except Exception as failure:
            raise exc from failure
        if partial_error is not None:
            raise exc from partial_error
        if restore_error is not None:
            raise exc from restore_error
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new output directory for this one v1 run")
    args = parser.parse_args()
    result = run_experiment(args.output)
    print(json.dumps({"protocol": result["protocol"], "sites": result["sites"],
                      "heldout_cases": len(result["cases"])}, sort_keys=True))


if __name__ == "__main__":
    main()
