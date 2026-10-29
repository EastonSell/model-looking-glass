"""Bounded donor-state patches in one uncached, teacher-forced T5 forward.

The caller owns model/runtime pairing and semantic alignment across all runs.
The comparison is an intervention on supplied examples, not localization proof.
"""

from dataclasses import dataclass, field
import math
import sys

import torch

from .agreement_numeric import decode_f32_le
from .flan_t5_trace import T5TracePoint, trace_t5_forward
from .intervention_plan import ActivationPlan


@dataclass(frozen=True)
class T5InterventionResult:
    """JSON-compatible report and bounded, detached CPU tensor witnesses."""

    report: dict
    witnesses: dict = field(repr=False)


def _same_bits(a, b):
    return (a.dtype == b.dtype == torch.float32 and a.shape == b.shape
            and torch.equal(a.view(torch.int32).contiguous(), b.view(torch.int32).contiguous()))


def _decoded(value):
    if value.dtype != torch.float32 or value.device.type != "cpu" or sys.byteorder != "little":
        raise ValueError("finite little-endian CPU float32 tensors are required")
    # Copy integer encodings, avoiding flush-sensitive float32 casts/copies.
    raw = value.detach().view(torch.int32).contiguous().numpy().tobytes()
    return decode_f32_le(raw)


def _l2(a, b):
    if a.shape != b.shape:
        raise ValueError("displacement vectors have different shapes")
    return math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(_decoded(a), _decoded(b))))


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _rows(value):
    return tuple(tuple(int(x) for x in row) for row in value.tolist())


def compare_t5_donor_patch(model, *, receiver_input_ids, donor_input_ids, attention_mask,
                           receiver_decoder_input_ids, donor_decoder_input_ids, decoder_attention_mask,
                           target, control, readout_position, readout_token_id, run_id="t5-donor-patch",
                           max_input_tokens=4096, max_tensor_bytes=16 * 1024 * 1024,
                           max_logits_bytes=128 * 1024 * 1024, max_retained_bytes=16 * 1024 * 1024,
                           synchronous_cpu=False):
    """Compare same-site donor and off-target patches on an explicit input pair.

    Each input is one CPU int64 row; the two runs use the exact same masks and
    shapes. Target/control are distinct active tokens at the same stack, layer,
    module and invocation. A same-site self patch and a receiver repeat check
    mechanics. The control patch can be nonzero and its dose need not match.
    Readout is a teacher-forced decoder position, not generated text.
    ``synchronous_cpu=True`` asserts exclusive, quiescent ownership and a
    synchronous CPU model with no deferred native or device work. It only
    supports Python hook/lease cleanup, not native or CUDA cleanup.
    """
    if type(synchronous_cpu) is not bool:
        raise TypeError("synchronous_cpu must be a boolean")
    if torch.compiler.is_compiling():
        raise ValueError("compiled execution is unsupported")
    if sys.byteorder != "little":
        raise ValueError("only little-endian CPU execution is supported")
    for name, value in (("max_input_tokens", max_input_tokens), ("max_tensor_bytes", max_tensor_bytes),
                        ("max_logits_bytes", max_logits_bytes), ("max_retained_bytes", max_retained_bytes)):
        _positive(value, name)
    if (type(target) is not T5TracePoint or type(control) is not T5TracePoint
            or (target.stack, target.layer, target.module, target.batch, target.invocation) !=
               (control.stack, control.layer, control.module, control.batch, control.invocation)
            or target.batch != 0 or target.token == control.token):
        raise ValueError("target and control must be distinct tokens at one batch-zero boundary")
    if type(run_id) is not str or not run_id.strip() or len(run_id) > 100:
        raise ValueError("run_id must be a nonempty string of at most 100 characters")
    tensors = (receiver_input_ids, donor_input_ids, attention_mask, receiver_decoder_input_ids,
               donor_decoder_input_ids, decoder_attention_mask)
    if any(type(t) is not torch.Tensor or t.dtype != torch.long or t.device.type != "cpu"
           or t.ndim != 2 or t.shape[0] != 1 or not t.numel() for t in tensors):
        raise ValueError("inputs and masks must be nonempty single-row CPU int64 tensors")
    receiver_ids, donor_ids, source_mask, receiver_dec, donor_dec, dec_mask = (t.detach().clone() for t in tensors)
    if (receiver_ids.shape != donor_ids.shape or receiver_ids.shape != source_mask.shape
            or receiver_dec.shape != donor_dec.shape or receiver_dec.shape != dec_mask.shape):
        raise ValueError("receiver and donor IDs and shared masks must have matching shapes")
    if receiver_ids.numel() + receiver_dec.numel() > max_input_tokens:
        raise ValueError("input token budget exceeded")
    if any(not bool(((mask == 0) | (mask == 1)).all()) for mask in (source_mask, dec_mask)):
        raise ValueError("masks must contain only zero or one")
    selected_mask = source_mask if target.stack == "encoder" else dec_mask
    if any(point.token >= selected_mask.shape[1] or not bool(selected_mask[0, point.token])
           for point in (target, control)):
        raise ValueError("target and control must select active positions in their own stack")
    if (type(readout_position) is not int or readout_position < 0 or readout_position >= dec_mask.shape[1]
            or not bool(dec_mask[0, readout_position])):
        raise ValueError("readout position must be an active decoder token")
    from transformers import T5ForConditionalGeneration
    if type(model) is not T5ForConditionalGeneration:
        raise TypeError("model must be a standard T5ForConditionalGeneration")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("only CPU models are supported")
    hidden = int(model.config.d_model)
    vocab = int(model.get_output_embeddings().weight.shape[0])
    if (hidden <= 0 or vocab <= 0 or max(hidden, vocab) * 4 > 1024 * 1024
            or type(readout_token_id) is not int or not 0 <= readout_token_id < vocab):
        raise ValueError("hidden/vocabulary size or readout token is unsupported")
    input_vocab = int(model.get_input_embeddings().weight.shape[0])
    if any(bool(((ids < 0) | (ids >= input_vocab)).any())
           for ids in (receiver_ids, donor_ids, receiver_dec, donor_dec)):
        raise ValueError("input token ID outside model vocabulary")
    length = selected_mask.shape[1]
    full_bytes, vector_bytes, readout_bytes = length * hidden * 4, hidden * 4, vocab * 4
    logits_bytes = receiver_dec.shape[1] * readout_bytes
    retained_bound = 6 * full_bytes + 5 * vector_bytes + 5 * readout_bytes
    if full_bytes > max_tensor_bytes or logits_bytes > max_logits_bytes or retained_bound > max_retained_bytes:
        raise ValueError("tensor, logits or retained-evidence budget exceeded")
    snapshots = tuple(_rows(t) for t in (receiver_ids, donor_ids, source_mask, receiver_dec, donor_dec, dec_mask))
    input_shapes = tuple(tuple(t.shape) for t in (receiver_ids, donor_ids, source_mask, receiver_dec, donor_dec, dec_mask))
    dimensions = (hidden, vocab, input_vocab, len(model.encoder.block), len(model.decoder.block))

    def check_inputs():
        current = (receiver_ids, donor_ids, source_mask, receiver_dec, donor_dec, dec_mask)
        if (any(t.dtype != torch.long or t.device.type != "cpu" or tuple(t.shape) != shape
                for t, shape in zip(current, input_shapes))
                or tuple(_rows(t) for t in current)
                != snapshots or (int(model.config.d_model), int(model.get_output_embeddings().weight.shape[0]),
                                  int(model.get_input_embeddings().weight.shape[0]), len(model.encoder.block),
                                  len(model.decoder.block)) != dimensions):
            raise RuntimeError("inputs or model dimensions changed during the workflow")

    def check_logits(output):
        check_inputs()
        logits = output.logits
        if (output.past_key_values is not None or type(logits) is not torch.Tensor
                or logits.dtype != torch.float32 or logits.device.type != "cpu"
                or tuple(logits.shape) != (1, receiver_dec.shape[1], vocab)
                or not bool(torch.isfinite(logits).all())):
            raise RuntimeError("expected finite CPU float32 logits from an uncached forward")
        return logits.detach()

    def trace(ids, dec, name):
        check_inputs()
        output, records = trace_t5_forward(model, input_ids=ids, attention_mask=source_mask,
            decoder_input_ids=dec, decoder_attention_mask=dec_mask, points=[target, control],
            run_id=f"{run_id}/{name}", max_events=2, max_trace_bytes=2 * vector_bytes,
            max_tensor_bytes=max_tensor_bytes, max_input_tokens=max_input_tokens, max_logits_bytes=max_logits_bytes,
            synchronous_cpu=synchronous_cpu)
        logits = check_logits(output)
        if len(records) != 2:
            raise RuntimeError("trace omitted a selected site")
        by_token = {record.event.token_positions[0]: record for record in records}
        if set(by_token) != {target.token, control.token}:
            raise RuntimeError("trace coordinates differ from the requested sites")
        for point in (target, control):
            record = by_token[point.token]
            if (record.event.invocation != 0 or record.event.batch_positions != (0,)
                    or record.event.layer_index != point.layer or tuple(record.event.shape) != (1, length, hidden)
                    or record.value.shape != (hidden,) or record.value.dtype != torch.float32
                    or record.value.device.type != "cpu"):
                raise RuntimeError("trace metadata or selected shape differs")
            _decoded(record.value)
        return logits, by_token

    receiver_logits, receiver = trace(receiver_ids, receiver_dec, "receiver")
    donor_logits, donor = trace(donor_ids, donor_dec, "donor")
    receiver_events = {str(token): record.event.to_dict() for token, record in receiver.items()}
    donor_events = {str(token): record.event.to_dict() for token, record in donor.items()}
    for point in (target, control):
        a, b = receiver[point.token].event, donor[point.token].event
        if (a.module_path != b.module_path or a.output_path != b.output_path or a.shape != b.shape
                or a.module_kind != b.module_kind or a.token_positions != b.token_positions):
            raise RuntimeError("receiver and donor capture policies differ")
    path = receiver[target.token].event.module_path
    selector = list(receiver[target.token].event.output_path)
    if receiver[control.token].event.module_path != path or list(receiver[control.token].event.output_path) != selector:
        raise RuntimeError("target and control do not share one boundary")
    limits = {"max_tensor_bytes": max_tensor_bytes, "max_capture_bytes": 2 * full_bytes + vector_bytes,
              "max_source_bytes": vector_bytes, "max_steps": 4, "max_invocations": 1}

    def arm(kind, point):
        base = {"module_path": path, "invocation": 0}
        selected = dict(base, batch_index=0, token_index=point.token)
        steps = [dict(base, name="before", operation="capture")]
        if kind == "self":
            steps.append(dict(selected, name="self_source", operation="capture"))
            steps.append(dict(selected, name="patch", operation="patch", source_capture="self_source"))
        else:
            steps.append(dict(selected, name="patch", operation="patch", value=donor[point.token].value))
        steps.append(dict(base, name="after", operation="capture"))
        check_inputs()
        plan = ActivationPlan(model, steps, **limits)
        with torch.inference_mode(), plan:
            forward = plan.run_synchronous_cpu if synchronous_cpu else model
            output = forward(input_ids=receiver_ids, attention_mask=source_mask, decoder_input_ids=receiver_dec,
                decoder_attention_mask=dec_mask, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=True)
        logits = check_logits(output)
        captures, metadata = plan.captures, plan.metadata
        if (len(metadata["steps"]) != len(steps)
                or set(captures) != {step["name"] for step in steps if step["operation"] == "capture"}
                or metadata["source_bytes"] != (0 if kind == "self" else vector_bytes)
                or metadata["capture_bytes"] != sum(x.numel() * x.element_size() for x in captures.values())):
            raise RuntimeError("intervention did not retain the complete requested actions")
        for index, (expected, actual) in enumerate(zip(steps, metadata["steps"], strict=True)):
            selected_step = "token_index" in expected
            if (actual["name"] != expected["name"] or actual["operation"] != expected["operation"]
                    or actual["order"] != index or actual["module_path"] != path
                    or actual["module_aliases"] != [path] or actual["invocation"] != 0
                    or actual["forward_index"] != 0 or actual["output_selector"] != selector
                    or actual["batch_index"] != expected.get("batch_index")
                    or actual["token_index"] != expected.get("token_index")
                    or actual["source_capture"] != expected.get("source_capture")
                    or actual["output_shape"] != [1, length, hidden]
                    or actual["selected_shape"] != ([hidden] if selected_step else [1, length, hidden])
                    or actual["dtype"] != "torch.float32" or actual["device"] != "cpu"):
                raise RuntimeError("intervention action metadata differs from its requested site")
        before, after = captures["before"], captures["after"]
        source = captures["self_source"] if kind == "self" else donor[point.token].value
        expected = before.view(torch.int32).clone()
        expected[0, point.token] = source.view(torch.int32)
        if (not _same_bits(before[0, point.token], receiver[point.token].value)
                or (kind == "self" and not _same_bits(source, before[0, point.token]))
                or not torch.equal(after.view(torch.int32), expected)):
            raise RuntimeError("patch failed exact landing, receiver matching or untouched-position checks")
        return logits, captures, metadata

    self_logits, self_state, self_meta = arm("self", target)
    target_logits, target_state, target_meta = arm("target", target)
    control_logits, control_state, control_meta = arm("control", control)
    repeat_logits, repeat = trace(receiver_ids, receiver_dec, "receiver-repeat")
    controls = {"self_patch_matches_receiver_logits": _same_bits(self_logits, receiver_logits),
                "repeat_matches_receiver_logits": _same_bits(repeat_logits, receiver_logits),
                "repeat_matches_receiver_sites": all(_same_bits(repeat[p.token].value, receiver[p.token].value)
                                                     for p in (target, control)),
                "all_patch_pre_states_match": (_same_bits(self_state["before"], target_state["before"])
                                                and _same_bits(self_state["before"], control_state["before"]))}
    if not all(controls.values()):
        raise RuntimeError(f"workflow control failed: {controls}")
    plans = {"self": self_state, "target": target_state, "off_target": control_state}
    readouts = {name: logits[0, readout_position].clone() for name, logits in
                (("receiver", receiver_logits), ("donor", donor_logits), ("self", self_logits),
                 ("target", target_logits), ("off_target", control_logits))}
    values = {name: _decoded(tensor) for name, tensor in readouts.items()}
    measurements = {name: {"selected_logit": float(row[readout_token_id]),
                            "max_absolute_difference_from_receiver": max(abs(float(a) - float(b))
                                for a, b in zip(row, values["receiver"]))} for name, row in values.items()}
    doses = {"target": _l2(target_state["before"][0, target.token], target_state["after"][0, target.token]),
             "off_target": _l2(control_state["before"][0, control.token], control_state["after"][0, control.token])}
    retained = (sum(x.numel() * x.element_size() for r in (receiver, donor) for x in (v.value for v in r.values()))
                + sum(x.numel() * x.element_size() for arm_values in plans.values() for x in arm_values.values())
                + sum(x.numel() * x.element_size() for x in readouts.values()))
    if retained > retained_bound:
        raise RuntimeError("retained tensor evidence exceeds its preflight bound")
    report = {"schema": "flan-t5-donor-patch-v1", "run_id": run_id, "forwards": 6,
              "inputs": {"receiver_encoder_ids": list(snapshots[0][0]), "donor_encoder_ids": list(snapshots[1][0]),
                         "encoder_mask": list(snapshots[2][0]), "receiver_decoder_ids": list(snapshots[3][0]),
                         "donor_decoder_ids": list(snapshots[4][0]), "decoder_mask": list(snapshots[5][0])},
              "sites": {"target": {"stack": target.stack, "layer": target.layer, "module": target.module,
                                   "batch": target.batch, "token": target.token, "invocation": target.invocation},
                        "off_target": {"stack": control.stack, "layer": control.layer, "module": control.module,
                                       "batch": control.batch, "token": control.token, "invocation": control.invocation}},
              "trace_events": {"receiver": receiver_events, "donor": donor_events},
              "plan_metadata": {"self": self_meta, "target": target_meta, "off_target": control_meta},
              "realized_l2": doses,
              "off_target_status": "active" if doses["off_target"] > 0 else "unavailable_zero_displacement",
              "target_status": "active" if doses["target"] > 0 else "zero_displacement",
              "readout": {"decoder_position": readout_position, "vocabulary_id": readout_token_id,
                          "measurements": measurements}, "controls": controls,
              "limits": {"max_input_tokens": max_input_tokens, "max_tensor_bytes": max_tensor_bytes,
                         "max_logits_bytes": max_logits_bytes, "max_retained_bytes": max_retained_bytes},
              "retained_tensor_bytes": retained, "retained_tensor_byte_bound": retained_bound,
              "interpretation": "Donor-state intervention with an off-target comparison; doses may differ. "
                                "Logits are teacher-forced, and neither site uniqueness nor generation is established."}
    return T5InterventionResult(report, {"receiver": {str(k): v.value for k, v in receiver.items()},
                                         "donor": {str(k): v.value for k, v in donor.items()},
                                         "plans": plans, "readout_logits": readouts})
