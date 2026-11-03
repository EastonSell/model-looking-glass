"""Explicit, same-owner donor-to-receiver activation patching."""
import torch
from .app_sensitivity import _selection, report_model
from .app_storage import TraceStore


def _display_label(get_value):
    """Bound token labels while keeping missing labels distinct from empty ones."""
    try:
        value = get_value()
        if type(value) is not str:
            raise ValueError("tokenizer label must be text")
        return value[:512], False, len(value) > 512
    except Exception as error:
        if isinstance(error, (MemoryError, torch.OutOfMemoryError)) or (
                isinstance(error, RuntimeError) and "out of memory" in str(error).lower()):
            raise
        return "", True, False


def _inputs(state, data):
    state._require_capture_id(data.get("capture_id"))
    available, reason = state._intervention_status()
    if not available: raise ValueError(reason)
    prompts = [data.get("donor_prompt"), data.get("receiver_prompt")]
    if any(type(p) is not str or not p.strip() or len(p) > 512 for p in prompts):
        raise ValueError("Enter two nonempty prompts of at most 512 characters each")
    answer = []
    with state.runtime._lock:
        opts = state.runtime._settings({"max_input_tokens": 64})
        for prompt in prompts:
            ids, mask = state.runtime._text_inputs(prompt, opts)
            token_ids, masks = ids[0].tolist(), mask[0].tolist()
            tokens = []
            for index, token in enumerate(token_ids):
                text, text_unavailable, text_truncated = _display_label(
                    lambda: state.runtime.tokenizer.decode([token]))
                piece, piece_unavailable, piece_truncated = _display_label(
                    lambda: state.runtime.tokenizer.convert_ids_to_tokens(token))
                tokens.append({"index": index, "token_id": token, "text": text, "piece": piece,
                               "text_unavailable": text_unavailable, "text_truncated": text_truncated,
                               "piece_unavailable": piece_unavailable, "piece_truncated": piece_truncated})
            answer.append({"token_ids": token_ids, "attention_mask": masks, "tokens": tokens})
    return answer


def preview(state, data):
    if set(data) != {"capture_id", "donor_prompt", "receiver_prompt"}:
        raise ValueError("Provide the current capture and exactly two prompts")
    donor, receiver = _inputs(state, data)
    return {"capture_id": data["capture_id"], "donor": donor, "receiver": receiver}


def patch(state, data):
    required = {"capture_id", "donor_prompt", "receiver_prompt", "donor_position", "receiver_position",
                "donor_token_id", "receiver_token_id", "donor_token_ids", "receiver_token_ids", "layer", "site"}
    if set(data) != required: raise ValueError("Cross-prompt patch requires both previewed prompts and explicit positions")
    donor, receiver = _inputs(state, data)
    for name, side in (("donor", donor), ("receiver", receiver)):
        position, token = data[name + "_position"], data[name + "_token_id"]
        supplied = data[name + "_token_ids"]
        if (type(supplied) is not list or not 1 <= len(supplied) <= 64 or any(type(x) is not int for x in supplied)
                or supplied != side["token_ids"] or type(position) is not int or not 0 <= position < len(supplied)
                or type(token) is not int or supplied[position] != token):
            raise ValueError("Token preview is stale or selected position/ID does not match; preview again")
    capture = state.store.summary()
    model = state.runtime.info()
    layers, sites = _selection({"layers": [data["layer"]], "sites": [data["site"]]}, model, capture)
    retained = capture["total_bytes"] + state.reference.total_bytes + (state.experiment_store.summary() or {}).get("total_bytes", 0)
    remaining = state.max_trace_bytes - retained
    if remaining < 4: raise ValueError("No remaining allowance for temporary captures and controls")
    budget = min(16 * 1024**2, remaining // 4)
    target = "attn" if sites[0] == "attention" and model["family"] == "gpt2" else sites[0]
    settings = {"layers": layers, "targets": [target], "max_input_tokens": 64, "max_new_tokens": 1,
                "max_events": 1, "max_trace_bytes": budget, "max_tensor_bytes": budget,
                "max_logits_bytes": 16 * 1024**2}
    stores = []
    try:
        for name, side in (("donor", donor), ("receiver", receiver)):
            store = TraceStore(state.store.export_root, max_bytes=budget, max_events=1)
            stores.append(store)
            result = state.runtime.run(data[name + "_prompt"], settings)
            store.publish(result)
            del result
            step = store.summary()["steps"][0]
            if step["input_token_ids"] != side["token_ids"] or step["input_attention_mask"] != side["attention_mask"]:
                raise ValueError("Fresh captured input differs from preview")
        source = stores[0].intervention_source(0, data["donor_position"])
        destination = stores[1].intervention_source(0, data["receiver_position"])
        allowance = min(128 * 1024**2, remaining - sum(s.summary()["total_bytes"] for s in stores))
        clean, patched = state.runtime.intervene_cross_prompt(state._live_binding, source, destination, allowance)
        c, p = clean["experiment"], patched["experiment"]
        clean_id = c["comparison"]["baseline_next_token_id"]
        clean_token = next(row for row in c["next_token_distribution"]["rows"] if row["token_id"] == clean_id)
        watched = p["next_token_distribution"]["watched_tokens"][0]
        controls = {"donor_null_full_logits_bitwise_equal": c["controls"]["null_full_logits_bitwise_equal"],
            "donor_restoration_full_logits_bitwise_equal": c["controls"]["restoration_full_logits_bitwise_equal"],
            "donor_source_vector_bitwise_equal": c["controls"]["source_vector_bitwise_equal"],
            "donor_other_positions_bitwise_equal": c["controls"]["other_token_positions_at_selected_site_bitwise_equal"],
            "donor_same_vector_full_logits_bitwise_equal": c["full_logits"]["bitwise_equal"],
            "receiver_null_full_logits_bitwise_equal": p["controls"]["null_full_logits_bitwise_equal"],
            "receiver_restoration_full_logits_bitwise_equal": p["controls"]["restoration_full_logits_bitwise_equal"],
            "receiver_source_vector_bitwise_equal": p["controls"]["source_vector_bitwise_equal"],
            "receiver_other_positions_bitwise_equal": p["controls"]["other_token_positions_at_selected_site_bitwise_equal"],
            "patched_vector_matches_donor_bitwise": True,
            "fixed_input_ids_masks_unchanged": True}
        if model["family"] == "llama":
            controls["donor_static_rotary_state_unchanged"] = c["controls"]["static_rotary_state_unchanged"]
            controls["receiver_static_rotary_state_unchanged"] = p["controls"]["static_rotary_state_unchanged"]
        return {"preset": "cross_prompt_patch", "capture_id": data["capture_id"], "status": "complete",
            "model": report_model(model), "layer": layers[0], "site": sites[0],
            "donor": {"prompt": data["donor_prompt"], "position": data["donor_position"],
                      **donor["tokens"][data["donor_position"]], "token_ids": donor["token_ids"]},
            "receiver": {"prompt": data["receiver_prompt"], "position": data["receiver_position"],
                         **receiver["tokens"][data["receiver_position"]], "token_ids": receiver["token_ids"]},
            "clean_baseline": {"token_id": clean_id, "text": clean_token["text"],
                               "text_unavailable": clean_token["text_unavailable"],
                               "text_truncated": clean_token["text_truncated"],
                               "probability": clean_token["baseline_probability"], "logit": clean_token["baseline_logit"]},
            "corrupted_baseline": {"token_id": clean_id, "probability": watched["baseline_probability"],
                                   "logit": watched["baseline_logit"], "top_token_id": p["comparison"]["baseline_next_token_id"]},
            "patched": {"token_id": clean_id, "probability": watched["intervention_probability"],
                        "logit": watched["intervention_logit"], "top_token_id": p["comparison"]["intervention_next_token_id"]},
            "controls": controls, "distribution": p["next_token_distribution"],
            "metrics": {"clean_token_probability_delta": watched["probability_delta"],
                        "total_variation": p["next_token_distribution"]["total_variation"],
                        "activation_changed": p["activation_changed"]},
            "total_model_forwards": 10,
            "scope": "One explicitly selected donor vector replaces one receiver vector at the same boundary on the same live model. The receiver readout is compared on its unchanged prefix.",
            "limitations": ["Token positions are user-selected, not semantically aligned automatically.",
                "The clean top token is watched under both receiver conditions using full-vocabulary probabilities.",
                "This single intervention does not establish a general mechanism, statistical significance, or correctness."]}
    finally:
        for store in stores: store.clear()
