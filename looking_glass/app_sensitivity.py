"""Small, fixed-prefix zero-vector sweeps built on the app's existing controls.

This report is an observation about one input, not a concept detector. Each
site uses its own captured vector; missing or ambiguous captures are not zeros.
"""
from __future__ import annotations


SITES = ("attention", "mlp", "block")
MAX_CELLS = 18
DEFAULT_LAYERS = 6


def report_model(model):
    """Keep bounded demo provenance without exporting local checkpoint paths."""
    result = {"family": model.get("family"), "device": model.get("device")}
    architecture = model.get("architecture")
    if type(architecture) is str and 0 < len(architecture) <= 128:
        result["architecture"] = architecture
    summary = model.get("config_summary")
    if type(summary) is dict:
        keys = ("vocab_size", "n_positions", "n_embd", "n_layer", "n_head", "max_position_embeddings",
                "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim")
        result["config_summary"] = {key: summary[key] for key in keys
                                    if type(summary.get(key)) is int and 0 <= summary[key] <= 2**31}
    note = model.get("identity_note")
    if type(note) is str and note:
        result["identity_note"] = note[:512]
    provenance = model.get("provenance")
    if type(provenance) is dict and provenance.get("kind") == "synthetic_untrained_demo":
        result["provenance"] = {"kind": "synthetic_untrained_demo"}
        if type(provenance.get("seed")) is int:
            result["provenance"]["seed"] = provenance["seed"]
    warning = model.get("warning")
    if type(warning) is str and warning:
        result["warning"] = warning[:512]
    return result


def _path(family, layer, site):
    prefix = f"transformer.h.{layer}" if family == "gpt2" else f"model.layers.{layer}"
    if site == "attention":
        return prefix + (".attn" if family == "gpt2" else ".self_attn")
    return prefix if site == "block" else prefix + ".mlp"


def _selection(data, model, capture):
    if set(data) - {"capture_id", "event", "vector", "layers", "sites"}:
        raise ValueError("Unknown sensitivity experiment option")
    possible = model.get("layers")
    if type(possible) is not list or not possible or any(type(x) is not int or x < 0 for x in possible):
        raise ValueError("Model layer inventory is unavailable")
    layers = data.get("layers")
    if layers is None:
        covered = capture.get("coverage", {}).get("layers")
        if type(covered) is not list:
            raise ValueError("Captured layer inventory is unavailable")
        layers = sorted(set(x for x in covered if type(x) is int and x in possible))[:DEFAULT_LAYERS]
    if (type(layers) is not list or not layers or any(type(x) is not int or x not in possible for x in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("layers must be distinct valid captured-model layer indices")
    sites = data.get("sites", list(SITES))
    if (type(sites) is not list or not sites or any(type(x) is not str or x not in SITES for x in sites)
            or len(set(sites)) != len(sites)):
        raise ValueError("sites must be a nonempty distinct selection of attention, mlp and block")
    layers = sorted(layers)
    sites = [site for site in SITES if site in sites]
    if len(layers) * len(sites) > MAX_CELLS:
        raise ValueError("Sensitivity experiments are limited to 18 layer/site cells")
    return layers, sites


def run_sensitivity(state, data):
    """Measure <=18 separately controlled zeroings; retain no capture tensors."""
    state._require_capture_id(data.get("capture_id"))
    available, reason = state._intervention_status()
    if not available: raise ValueError(reason)
    capture = state.store.summary()
    model = state.runtime.info()
    layers, sites = _selection(data, model, capture)
    anchor = state.store.intervention_source(data.get("event"), data.get("vector"))
    coordinate, step = anchor["coordinate"], anchor["step"]
    family = model["family"]
    cells = [(layer, site, _path(family, layer, site)) for layer in layers for site in sites]
    candidates = state.store.intervention_candidates(coordinate["app_forward_index"],
        coordinate["token_position"], [path for _, _, path in cells])
    retained = (capture["total_bytes"] + state.reference.total_bytes
                + (state.experiment_store.summary() or {}).get("total_bytes", 0))
    allowance = min(128 * 1024**2, state.max_trace_bytes - retained)
    if allowance <= 0: raise ValueError("No remaining trace allowance for a sensitivity experiment")

    rows, replay_count, failed, baseline = [], 0, False, None
    for layer, site, path in cells:
        row = {"layer": layer, "site": site, "module_path": path}
        if failed:
            row.update(status="skipped", reason="Stopped after an earlier failed or inconsistent trial.")
            rows.append(row)
            continue
        matches = candidates[path]
        if len(matches) != 1:
            row.update(status="unsupported", reason=(
                "No matching captured vector for this forward and token. Capture this site and try again."
                if not matches else "Multiple matching captured vectors; the source is ambiguous."))
            rows.append(row)
            continue
        try:
            source = state.store.intervention_source(*matches[0])
            result = state.runtime.intervene_fixed_forward(state._live_binding, source, "zero",
                max_pair_bytes=allowance, _include_baseline_digest=True)
            replay_count += 1
            baseline_digest = result.pop("_baseline_digest", None)
            if type(baseline_digest) is not str or len(baseline_digest) != 64:
                raise ValueError("Full baseline identity is unavailable; trials cannot be compared reliably")
            experiment = result["experiment"]
            distribution = experiment["next_token_distribution"]
            token_id = experiment["comparison"]["baseline_next_token_id"]
            token = next(item for item in distribution["rows"] if item["token_id"] == token_id)
            baseline_key = baseline_digest
            if baseline is None:
                baseline = baseline_key
            elif baseline != baseline_key:
                row.update(status="failed", failure_kind="baseline_mismatch",
                           reason="Full baseline logits differed across trials; results cannot be compared reliably.")
                failed = True
                rows.append(row)
                continue
            row.update(status="ok", reason=None, activation_changed=experiment["activation_changed"],
                controls=experiment["controls"], baseline_token={key: token[key] for key in (
                    "token_id", "text", "text_unavailable", "text_truncated",
                    "baseline_probability", "intervention_probability", "probability_delta",
                    "baseline_logit", "intervention_logit")}, total_variation=distribution["total_variation"],
                last_position_max_abs_delta=experiment["comparison"]["last_position_max_abs_delta"],
                next_token_distribution=distribution)
        except Exception as error:
            # An execution error quarantines the runtime. A preflight error can
            # leave it ready; either way, stop rather than publish zero effects.
            detail = str(error)[:512] or type(error).__name__
            control = any(part in detail.lower() for part in (
                "null control", "restoration", "original source", "other token positions", "did not land exactly"))
            row.update(status="failed", failure_kind="control_failure" if control else "execution_failure",
                       reason=f"{type(error).__name__}: {detail}")
            failed = True
        rows.append(row)
    completed = sum(row["status"] == "ok" for row in rows)
    status = "inconclusive" if failed or completed == 0 else "complete" if completed == len(rows) else "partial"
    return {"preset": "component_sensitivity", "capture_id": data["capture_id"], "status": status,
        "anchor": {"forward_index": coordinate["app_forward_index"], "token_position": coordinate["token_position"],
                   "token_id": coordinate["token_id"], "readout_position": len(step["input_token_ids"]) - 1,
                   "input_token_ids": list(step["input_token_ids"]), "input_attention_mask": list(step["input_attention_mask"])},
        "model": report_model(model), "layers": layers, "sites": sites,
        "planned_runs": len(cells), "completed_runs": completed, "completed_model_forwards": replay_count * 4,
        "model_forwards_upper_bound": len(cells) * 4, "results": rows,
        "scope": "One recorded uncached input prefix, one selected input token, and the next-token readout. Each supported cell zeros its own whole output vector temporarily; no checkpoint weights are edited.",
        "limitations": ["These are local intervention observations, not explanations of concepts or general model behavior.",
            "Only selected captured boundaries are tested; unsupported and skipped cells are not zero effects.",
            "Probabilities use the full vocabulary; the rows show only the union of the baseline and edited top five tokens.",
            "Four model forwards are used per completed replay. Failed trial forward counts can be unknown; memory allowances are planning estimates, not peak-memory guarantees.",
            "No generated continuations, cross-prompt activation patching, robustness study, or statistical significance test is performed."]}
