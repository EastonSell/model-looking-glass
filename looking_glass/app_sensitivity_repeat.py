"""Bounded same-owner sensitivity sweeps over separately supplied prompts."""
from __future__ import annotations

from types import SimpleNamespace

from .app_sensitivity import SITES, _selection, report_model, run_sensitivity
from .app_storage import TraceStore

MAX_PROMPTS = 4
MAX_TOTAL_CELLS = 12
MAX_TEMP_CAPTURE_BYTES = 32 * 1024**2


def run_repeated_sensitivity(state, data, *, cancel_event=None):
    """Run fresh temporary captures without replacing the user's retained run."""
    if set(data) - {"capture_id", "layers", "sites", "prompts"}:
        raise ValueError("Unknown repeated sensitivity experiment option")
    state._require_capture_id(data.get("capture_id"))
    available, reason = state._intervention_status()
    if not available:
        raise ValueError(reason)
    prompts = data.get("prompts")
    if (type(prompts) is not list or not 2 <= len(prompts) <= MAX_PROMPTS
            or any(type(prompt) is not str or not prompt.strip() or len(prompt) > 512 for prompt in prompts)):
        raise ValueError("Enter two to four nonempty prompts of at most 512 characters each")
    capture = state.store.summary()
    model = state.runtime.info()
    possible = model.get("layers")
    if type(possible) is not list or not possible or any(type(x) is not int or x < 0 for x in possible):
        raise ValueError("Model layer inventory is unavailable")
    _, sites = _selection({"layers": [possible[0]], "sites": data.get("sites", list(SITES))}, model, capture)
    chosen = data.get("layers")
    if chosen is None:
        covered = capture.get("coverage", {}).get("layers")
        if type(covered) is not list:
            raise ValueError("Captured layer inventory is unavailable")
        max_layers = min(6, MAX_TOTAL_CELLS // (len(prompts) * len(sites)))
        chosen = sorted(set(x for x in covered if type(x) is int and x in possible))[:max_layers]
    layers, sites = _selection({"layers": chosen, "sites": sites}, model, capture)
    per_prompt = len(layers) * len(sites)
    if per_prompt * len(prompts) > MAX_TOTAL_CELLS:
        raise ValueError("Repeated experiments are limited to 12 prompt/layer/site cells")

    retained = (capture["total_bytes"] + state.reference.total_bytes
                + (state.experiment_store.summary() or {}).get("total_bytes", 0))
    remaining = state.max_trace_bytes - retained
    if remaining < 2:
        raise ValueError("No remaining trace allowance for temporary captures and controls")
    capture_budget = min(MAX_TEMP_CAPTURE_BYTES, remaining // 2)
    targets = ["attn" if site == "attention" and model["family"] == "gpt2" else site for site in sites]
    settings = {"layers": layers, "targets": targets, "max_input_tokens": 64,
                "max_new_tokens": 1, "max_events": per_prompt,
                "max_trace_bytes": capture_budget, "max_tensor_bytes": capture_budget,
                "max_logits_bytes": 16 * 1024**2}
    rows, capture_count, completed_runs, completed_forwards = [], 0, 0, 0
    stopped_failure = stopped_cancel = False
    for index, prompt in enumerate(prompts):
        row = {"prompt_index": index, "prompt": prompt}
        if stopped_failure:
            row.update(status="skipped", reason="Stopped after an earlier failed or inconclusive prompt.")
            rows.append(row)
            continue
        if stopped_cancel or (cancel_event is not None and cancel_event.is_set()):
            stopped_cancel = True
            row.update(status="skipped", reason="Stopped by request before this prompt.")
            rows.append(row)
            continue
        scratch = TraceStore(state.store.export_root, max_bytes=capture_budget, max_events=per_prompt)
        try:
            result = state.runtime.run(prompt, settings)
            capture_count += 1
            scratch.publish(result)
            del result
            step = scratch.summary()["steps"][0]
            position = len(step["input_token_ids"]) - 1
            # The original capture, retained reference and manual comparison
            # count against the same server allowance as the temporary capture.
            proxy = SimpleNamespace(runtime=state.runtime, store=scratch,
                experiment_store=state.experiment_store, reference=state.reference,
                max_trace_bytes=state.max_trace_bytes - capture["total_bytes"],
                _live_binding=state._live_binding, _require_capture_id=state._require_capture_id,
                _intervention_status=state._intervention_status)
            report = run_sensitivity(proxy, {"capture_id": data["capture_id"], "event": 0,
                "vector": position, "layers": layers, "sites": sites}, cancel_event=cancel_event)
            coordinate = scratch._coordinate(0, position)
            report["anchor"]["token_text"] = coordinate.get("token_text")
            report["anchor"]["token_piece"] = coordinate.get("token_piece")
            report["anchor"]["token_label_truncated"] = coordinate.get("token_label_truncated", False)
            # A fresh temporary input must not be presented as the original capture.
            report.pop("capture_id", None)
            report["source"] = "temporary capture for this prompt"
            completed_runs += report["completed_runs"]
            completed_forwards += report["completed_model_forwards"]
            row.update(status=report["status"], report=report)
            if report["status"] == "inconclusive":
                stopped_failure = True
            elif report["status"] == "cancelled":
                stopped_cancel = True
        except Exception as error:
            row.update(status="failed", reason=f"{type(error).__name__}: {str(error)[:512]}")
            stopped_failure = True
        finally:
            scratch.clear()
        rows.append(row)
    complete_prompts = sum(row["status"] == "complete" for row in rows)
    overall = ("inconclusive" if stopped_failure else "cancelled" if stopped_cancel
               else "complete" if complete_prompts == len(prompts) else "partial")
    return {"preset": "multi_prompt_sensitivity", "capture_id": data["capture_id"],
        "status": overall, "model": report_model(model),
        "layers": layers, "sites": sites, "planned_prompts": len(prompts),
        "completed_prompts": complete_prompts, "captured_prompts": capture_count,
        "planned_runs": per_prompt * len(prompts), "completed_runs": completed_runs,
        "completed_model_forwards": capture_count + completed_forwards,
        "model_forwards_upper_bound": len(prompts) * (1 + 4 * per_prompt), "results": rows,
        "scope": "Each prompt is freshly captured on the same live model owner. Tests zero the final input token's output vector at selected sites, then measure that prompt's next-token distribution.",
        "limitations": ["Final token positions need not represent corresponding words or concepts across prompts.",
            "Each prompt can have a different baseline top token; its probability change is not a same-token comparison across prompts.",
            "Results are observations for supplied inputs, not statistical significance or proof of general robustness.",
            "No activation is transferred between prompts; cross-prompt patching is not performed.",
            "A failed forward can have an unknown actual count; capture and memory limits are not peak-memory guarantees."]}
