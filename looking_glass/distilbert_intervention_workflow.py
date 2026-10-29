"""Bounded DistilBERT intervention workflow built from existing primitives.

This tests a known artificial edit; it does not find or authenticate a natural
model mechanism. The caller must exclusively own the model across all forwards.
"""

from dataclasses import dataclass, field
import math

import torch

from .distilbert_trace import trace_distilbert_masked_lm
from .intervention_plan import ActivationPlan


@dataclass(frozen=True)
class DistilBertInterventionResult:
    """JSON-compatible report plus bounded, mutable CPU tensor witnesses."""

    report: dict
    witnesses: dict = field(repr=False)


def _same_bits(left, right):
    return (left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.int32),
                            right.contiguous().view(torch.int32)))


def _numerically_equal(left, right):
    """Compare finite float32 values exactly, treating signed zeros as equal."""
    if left.dtype != torch.float32 or right.dtype != torch.float32 or left.shape != right.shape:
        return False
    a, b = left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)
    both_zero = ((a & 0x7fffffff) == 0) & ((b & 0x7fffffff) == 0)
    return bool(((a == b) | both_zero).all())


def _l2(left, right):
    return float(torch.linalg.vector_norm(left.double() - right.double()).item())


def _rows(tensor):
    return tuple(tuple(int(value) for value in row) for row in tensor.tolist())


def run_known_distilbert_intervention(model, input_ids, attention_mask, *, injection_layer,
                                     restoration_layer, target_token, control_token,
                                     readout_position, readout_token_id, coordinate, amount,
                                     run_id="distilbert-intervention"):
    """Inject a known delta and compare clean-state patches at two positions.

    Supports one CPU int64 row of 2--16 tokens, an eager evaluation-mode CPU
    float32 DistilBertForMaskedLM, and a later restoration block. The target,
    control and readout positions must be active in the supplied binary mask;
    target and control are distinct. The caller chooses the masked-LM readout
    position and vocabulary ID, and must verify their intended meaning.

    Returns all bounded full-output intervention witnesses and readout vectors
    alongside JSON-compatible measurements. Session limits exclude model state,
    full MLM logits, input tuples, temporaries and peak memory. A zero delivered
    dose or unchanged readout is reported, never interpreted as a successful
    restoration. Model identity, semantic alignment and exclusive ownership are
    caller responsibilities.
    """
    from transformers import DistilBertForMaskedLM

    if type(model) is not DistilBertForMaskedLM:
        raise TypeError("model must be an unmodified DistilBertForMaskedLM")
    if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.int64
            or input_ids.device.type != "cpu" or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or not 2 <= input_ids.shape[1] <= 16):
        raise ValueError("input_ids must be one CPU int64 row of 2--16 tokens")
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.int64
            or attention_mask.device.type != "cpu" or attention_mask.shape != input_ids.shape
            or bool(((attention_mask != 0) & (attention_mask != 1)).any())):
        raise ValueError("attention_mask must be a matching binary CPU int64 row")
    ids, mask = input_ids.detach().clone(), attention_mask.detach().clone()
    length = ids.shape[1]
    hidden, vocab = model.config.dim, model.config.vocab_size
    if (type(hidden) is not int or not 1 <= hidden <= 1024
            or type(vocab) is not int or not 1 <= vocab <= 65536):
        raise ValueError("hidden size or vocabulary exceeds workflow bounds")
    if bool((ids < 0).any()) or bool((ids >= vocab).any()):
        raise ValueError("input IDs must be within the vocabulary")
    layers = len(model.distilbert.transformer.layer)
    if (type(injection_layer) is not int or type(restoration_layer) is not int
            or not 0 <= injection_layer < restoration_layer < layers):
        raise ValueError("injection_layer must precede the existing restoration_layer")
    for position in (target_token, control_token, readout_position):
        if type(position) is not int or not 0 <= position < length or mask[0, position].item() != 1:
            raise ValueError("target, control and readout positions must be active input tokens")
    if target_token == control_token:
        raise ValueError("target and control positions must differ")
    if type(readout_token_id) is not int or not 0 <= readout_token_id < vocab:
        raise ValueError("readout_token_id must be a vocabulary ID")
    if type(coordinate) is not int or not 0 <= coordinate < hidden:
        raise ValueError("coordinate must be a hidden-vector index")
    if type(amount) not in (int, float):
        raise ValueError("amount must be a finite nonzero number")
    try:
        amount = float(amount)
    except (OverflowError, ValueError):
        raise ValueError("amount must be a finite nonzero number") from None
    if not math.isfinite(amount) or amount == 0:
        raise ValueError("amount must be a finite nonzero number")
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    delta = torch.zeros(hidden, dtype=torch.float32)
    try:
        delta[coordinate] = amount
    except (RuntimeError, OverflowError):
        raise ValueError("amount is outside float32 range") from None
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("amount is outside float32 range")

    id_rows, mask_rows = _rows(ids), _rows(mask)
    dimensions = (layers, hidden, vocab, model.config.max_position_embeddings)
    full_bytes, vector_bytes = length * hidden * 4, hidden * 4
    limits = {"max_tensor_bytes": full_bytes, "max_capture_bytes": 4 * full_bytes + vector_bytes,
              "max_source_bytes": 2 * vector_bytes, "max_steps": 7, "max_invocations": 1}
    injection_path = f"distilbert.transformer.layer.{injection_layer}"
    restoration_path = f"distilbert.transformer.layer.{restoration_layer}"

    def forward():
        output = model(input_ids=ids, attention_mask=mask, output_attentions=False,
                       output_hidden_states=False, return_dict=True)
        if (ids.dtype != torch.int64 or mask.dtype != torch.int64
                or ids.device.type != "cpu" or mask.device.type != "cpu"
                or _rows(ids) != id_rows or _rows(mask) != mask_rows
                or (len(model.distilbert.transformer.layer), model.config.dim,
                    model.config.vocab_size, model.config.max_position_embeddings) != dimensions):
            raise RuntimeError("inputs or model dimensions changed during the forward")
        logits = output.logits
        if (type(logits) is not torch.Tensor or logits.device.type != "cpu"
                or logits.dtype != torch.float32 or tuple(logits.shape) != (1, length, vocab)
                or not bool(torch.isfinite(logits).all())):
            raise RuntimeError("masked-LM logits must be finite CPU float32 at every input position")
        return logits.detach()

    with torch.no_grad():
        clean_trace = trace_distilbert_masked_lm(
            model, ids, mask, run_id=f"{run_id}/clean", layers=(injection_layer, restoration_layer),
            modules=("block",), samples=(0,), tokens=(target_token, control_token),
            max_events=4, max_trace_bytes=4 * vector_bytes, max_tensor_bytes=full_bytes)
        clean_logits = forward()
        if not _same_bits(clean_trace.output.logits, clean_logits):
            raise RuntimeError("clean trace changed the unhooked logits")
        clean_values = {(item.event.layer_index, item.event.token_positions[0]): item.value[0, 0]
                        for item in clean_trace.events}
        if len(clean_values) != 4 or any(not bool(torch.isfinite(x).all()) for x in clean_values.values()):
            raise RuntimeError("clean trace did not retain finite selected states")

        def run_plan(kind):
            early = {"module_path": injection_path, "invocation": 0}
            early_selected = dict(early, batch_index=0, token_index=target_token)
            late = {"module_path": restoration_path, "invocation": 0}
            steps = [dict(early, name="injection_before", operation="capture")]
            if kind == "undo":
                steps.append(dict(early_selected, name="injection_source", operation="capture"))
            steps.extend((dict(early_selected, name="inject", operation="add", value=delta),
                          dict(early, name="injection_after", operation="capture")))
            if kind == "undo":
                steps.extend((dict(early_selected, name="undo", operation="patch",
                                   source_capture="injection_source"),
                              dict(early, name="undo_after", operation="capture")))
            else:
                steps.append(dict(late, name="restoration_before", operation="capture"))
                if kind in ("target", "control", "self"):
                    selected_token = control_token if kind == "control" else target_token
                    selected = dict(late, batch_index=0, token_index=selected_token)
                    if kind == "self":
                        steps.extend((dict(selected, name="self_source", operation="capture"),
                                      dict(selected, name="patch", operation="patch", source_capture="self_source")))
                    else:
                        steps.append(dict(selected, name="patch", operation="patch",
                                          value=clean_values[(restoration_layer, selected_token)]))
                    steps.append(dict(late, name="restoration_after", operation="capture"))
            plan = ActivationPlan(model, steps, **limits)
            with plan:
                logits = forward()
            metadata, captures = plan.metadata, plan.captures
            observed = metadata["steps"]
            capture_names = {step["name"] for step in steps if step["operation"] == "capture"}
            source_count = sum(step["operation"] in ("add", "patch") and "value" in step
                               for step in steps)
            if (len(observed) != len(steps) or set(captures) != capture_names
                    or metadata["capture_bytes"] != sum(value.numel() * value.element_size()
                                                         for value in captures.values())
                    or metadata["source_bytes"] != source_count * vector_bytes):
                raise RuntimeError("plan returned an incomplete action sequence")
            for index, (requested, event) in enumerate(zip(steps, observed)):
                selected = requested.get("token_index") is not None
                if (event["name"] != requested["name"] or event["operation"] != requested["operation"]
                        or event["order"] != index or event["module_path"] != requested["module_path"]
                        or event["module_aliases"] != [requested["module_path"]]
                        or event["forward_index"] != 0 or event["invocation"] != 0
                        or event["batch_index"] != requested.get("batch_index")
                        or event["token_index"] != requested.get("token_index")
                        or event["source_capture"] != requested.get("source_capture")
                        or event["output_selector"] != [0] or event["output_shape"] != [1, length, hidden]
                        or event["selected_shape"] != ([hidden] if selected else [1, length, hidden])
                        or event["dtype"] != "torch.float32" or event["device"] != "cpu"):
                    raise RuntimeError("plan metadata does not match the requested ordered sites")
            if any(not bool(torch.isfinite(value).all()) for value in captures.values()):
                raise RuntimeError("plan captured nonfinite activation values")
            return logits, captures, metadata

        injected_logits, injected, injected_meta = run_plan("injected")
        self_logits, self_state, self_meta = run_plan("self")
        target_logits, target_state, target_meta = run_plan("target")
        control_logits, control_state, control_meta = run_plan("control")
        undo_logits, undo_state, undo_meta = run_plan("undo")
        repeat_clean = forward()
        repeat_injected_logits, repeat_injected, _ = run_plan("injected")

        plans = {"injected": injected, "self": self_state, "target": target_state,
                 "off_target": control_state, "undo": undo_state}
        plan_metadata = {"injected": injected_meta, "self": self_meta, "target": target_meta,
                         "off_target": control_meta, "undo": undo_meta}
        before = injected["injection_before"]
        expected = before.clone()
        expected[0, target_token] = before[0, target_token] + delta
        if (not bool(torch.isfinite(expected).all())
                or not _same_bits(before[0, target_token], clean_values[(injection_layer, target_token)])):
            raise RuntimeError("injection receiver differs from the clean trace or is nonfinite")
        unchanged = expected.clone()
        unchanged[0, target_token, coordinate] = before[0, target_token, coordinate]
        if not _same_bits(unchanged, before):
            raise RuntimeError("the requested addition changed an unintended coordinate")
        for captures in (*plans.values(), repeat_injected):
            if (not _same_bits(captures["injection_before"], before)
                    or not _same_bits(captures["injection_after"], expected)):
                raise RuntimeError("injection landing or same-forward pre-state differs")
        for captures in (self_state, target_state, control_state, repeat_injected):
            if not _same_bits(captures["restoration_before"], injected["restoration_before"]):
                raise RuntimeError("later receiver does not match the injected baseline")
        for captures, token in ((target_state, target_token), (control_state, control_token),
                                (self_state, target_token)):
            expected_after = captures["restoration_before"].clone()
            source = (captures["self_source"] if captures is self_state
                      else clean_values[(restoration_layer, token)])
            expected_after[0, token] = source
            if not _same_bits(captures["restoration_after"], expected_after):
                raise RuntimeError("later patch did not land exactly or changed another position")
        controls = {
            "clean_trace_matches_unhooked": _same_bits(clean_trace.output.logits, clean_logits),
            "self_patch_matches_injected_logits": _same_bits(self_logits, injected_logits),
            "immediate_undo_matches_clean_activation": _same_bits(undo_state["undo_after"], before),
            "immediate_undo_matches_clean_logits": _same_bits(undo_logits, clean_logits),
            "clean_repeat_matches": _same_bits(repeat_clean, clean_logits),
            "injected_repeat_matches": _same_bits(repeat_injected_logits, injected_logits),
        }
        if not all(controls.values()):
            raise RuntimeError(f"workflow control failed: {controls}")

        all_logits = {"clean": clean_logits, "injected": injected_logits, "self": self_logits,
                      "target": target_logits, "off_target": control_logits, "undo": undo_logits}
        readout = {name: logits[0, readout_position].detach().clone() for name, logits in all_logits.items()}
        measurements = {name: {"selected_logit": float(values[readout_token_id].item()),
                               "selected_logit_float32_bits": int(values[readout_token_id:readout_token_id + 1].view(torch.int32)[0].item()) & 0xffffffff,
                               "max_absolute_distance_from_clean": float((values.double() - readout["clean"].double()).abs().max().item()),
                               "max_absolute_distance_from_injected": float((values.double() - readout["injected"].double()).abs().max().item())}
                        for name, values in readout.items()}
        injection_dose = _l2(injected["injection_after"][0, target_token], before[0, target_token])
        target_dose = _l2(target_state["restoration_after"][0, target_token], target_state["restoration_before"][0, target_token])
        control_dose = _l2(control_state["restoration_after"][0, control_token], control_state["restoration_before"][0, control_token])
        readout_changed = not _numerically_equal(readout["clean"], readout["injected"])
        selected_changed = not _numerically_equal(readout["clean"][readout_token_id:readout_token_id + 1],
                                                 readout["injected"][readout_token_id:readout_token_id + 1])
        unavailable = []
        if injection_dose == 0:
            unavailable.append("injection delivered zero displacement")
        if target_dose == 0:
            unavailable.append("target patch delivered zero displacement")
        if not readout_changed:
            unavailable.append("injection did not change the selected readout vector")
        elif not selected_changed:
            unavailable.append("injection did not change the selected vocabulary logit")
        control_status = ("unavailable_zero_injection" if injection_dose == 0 else
                          "unavailable_zero_displacement" if control_dose == 0 else "active")
        if control_status != "active":
            unavailable.append("off-target patch is unavailable")
        clean_snapshots = {f"layer{item.event.layer_index}-token{item.event.token_positions[0]}": item.value
                           for item in clean_trace.events}
        retained = sum(x.numel() * x.element_size() for x in clean_snapshots.values())
        retained += sum(x.numel() * x.element_size() for captures in plans.values() for x in captures.values())
        retained += sum(x.numel() * x.element_size() for x in readout.values())
        retained_bound = 18 * full_bytes + 6 * vector_bytes + 6 * vocab * 4
        if retained > retained_bound:
            raise RuntimeError("retained tensor evidence exceeds the fixed workflow bound")
        report = {
            "schema": "distilbert-known-intervention-v1", "run_id": run_id,
            "input_ids": list(id_rows[0]), "attention_mask": list(mask_rows[0]),
            "injection": {"module_path": injection_path, "layer": injection_layer, "invocation": 0,
                          "batch_index": 0, "token_index": target_token, "coordinate": coordinate,
                          "requested_addition": amount, "realized_l2": injection_dose,
                          "float32_addition": float(delta[coordinate].item()),
                          "before": float(before[0, target_token, coordinate].item()),
                          "after": float(injected["injection_after"][0, target_token, coordinate].item()),
                          "status": "nonzero" if injection_dose > 0 else "zero_displacement"},
            "restoration": {"module_path": restoration_path, "layer": restoration_layer,
                            "invocation": 0, "batch_index": 0, "token_index": target_token,
                            "realized_l2": target_dose,
                            "status": "nonzero" if target_dose > 0 else "zero_displacement"},
            "off_target": {"module_path": restoration_path, "layer": restoration_layer,
                           "invocation": 0, "batch_index": 0, "token_index": control_token,
                           "realized_l2": control_dose, "status": control_status},
            "readout": {"input_position": readout_position, "vocabulary_id": readout_token_id,
                        "injected_vector_changed": readout_changed,
                        "injected_selected_logit_changed": selected_changed, "measurements": measurements},
            "restoration_comparison": {"status": "unavailable" if unavailable else "descriptive",
                                       "reasons": unavailable},
            "controls": controls, "plan_limits": limits, "plan_metadata": plan_metadata,
            "clean_trace_events": [{"metadata": item.event.to_dict(),
                                    "witness_key": f"layer{item.event.layer_index}-token{item.event.token_positions[0]}",
                                    "original_saved_shape": list(item.original_saved_shape),
                                    "comparison_transform": item.comparison_transform}
                                   for item in clean_trace.events],
            "retained_tensor_bytes": retained, "retained_tensor_byte_bound": retained_bound,
            "interpretation": "An artificial intervention and selected readout are measured; the off-target patch is descriptive and not assumed dose-matched. Immediate undo is a by-construction mechanics check, not a unique causal localization.",
        }
        return DistilBertInterventionResult(report, {"clean_trace": clean_snapshots,
                                                     "plans": plans, "readout_logits": readout})
