"""Compose bounded GPT-2 traces, comparison and same-site interventions.

This is a CPU development workflow for a known input-embedding perturbation,
not a causal-discovery algorithm. The caller must exclusively own the model.
"""

import math

import torch

from .intervention_plan import ActivationPlan
from .trace_schema import GenerationContext
from .transformer_localization import TensorObservation, compare_transformer_traces
from .transformer_trace import GPT2TraceSession


def _same_bits(left, right):
    return (left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.int32),
                            right.contiguous().view(torch.int32)))


def _site_path(site):
    layer, kind = site
    return f"transformer.h.{layer}" + ("" if kind == "block" else f".{kind}")


def _dose_matches(q, tolerance, target_dose, comparison_dose):
    """Check both positive actual doses against q and against each other."""
    return (all(math.isfinite(value) and value > 0 for value in (q, target_dose, comparison_dose))
            and math.isfinite(tolerance) and tolerance >= 0
            and abs(target_dose - q) <= tolerance and abs(comparison_dose - q) <= tolerance
            and abs(target_dose - comparison_dose) <= 0.01 * target_dose)


def _plan_matched_patches(target_clean, target_corrupt, comparison_clean, comparison_corrupt):
    """Interpolate two float32 vectors toward clean without extrapolating."""
    values = (target_clean, target_corrupt, comparison_clean, comparison_corrupt)
    if (any(type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.ndim != 1 or not value.numel() for value in values)
            or len({tuple(value.shape) for value in values}) != 1
            or any(not bool(torch.isfinite(value).all()) for value in values)):
        raise RuntimeError("matched directions require finite CPU float32 vectors of equal shape")
    target_delta = target_clean.double() - target_corrupt.double()
    comparison_delta = comparison_clean.double() - comparison_corrupt.double()
    target_distance = target_delta.norm().item()
    comparison_distance = comparison_delta.norm().item()
    if not math.isfinite(target_distance) or not math.isfinite(comparison_distance):
        raise RuntimeError("matched direction has nonfinite L2 distance")
    q = min(target_distance, comparison_distance)
    result = {"status": "dose_ineligible", "natural_target_l2": target_distance,
              "natural_comparison_l2": comparison_distance, "common_dose_l2": q}
    if q <= 1e-6:
        return result, None
    tolerance = min(max(1e-5, 1e-5 * q), 0.01 * q)
    target_fraction, comparison_fraction = q / target_distance, q / comparison_distance
    target_value = (target_corrupt.double() + target_fraction * target_delta).float()
    comparison_value = (comparison_corrupt.double() + comparison_fraction * comparison_delta).float()
    planned_target = (target_value.double() - target_corrupt.double()).norm().item()
    planned_comparison = (comparison_value.double() - comparison_corrupt.double()).norm().item()
    result.update({"absolute_dose_tolerance": tolerance, "target_fraction": target_fraction,
                   "comparison_fraction": comparison_fraction, "planned_target_l2": planned_target,
                   "planned_comparison_l2": planned_comparison})
    if not _dose_matches(q, tolerance, planned_target, planned_comparison):
        result["status"] = "dose_unmatchable"
        return result, None
    result["status"] = "ready"
    return result, (target_value, comparison_value)


def compare_known_corruption(model, input_ids, *, coordinate, amount, readout_token,
                             layers=(0, 1), target_site=(0, "block"),
                             comparison_site=(0, "attn"), matched_control=False):
    """Trace a known embedding edit and test two clean-state patches.

    ``input_ids`` is one unpadded CPU int64 row of 2--16 tokens. The final
    token's embedding is copied and changed at one coordinate by ``amount``.
    Each site is ``(layer, 'attn'|'mlp'|'block')`` and both must be selected,
    distinct, and in the same layer. The model must be an unmodified,
    evaluation-mode CPU float32 GPT2LMHeadModel. No cache or sampling is used.
    With ``matched_control=True``, two further partial patches match the
    smaller clean-to-corrupt L2 dose at both sites when float32 permits it.

    Returns JSON-compatible metadata, exact comparison statuses, input edit,
    raw selected next-token logits, logit distances, plan witnesses and
    controls. Retained snapshots are limited per trace and plan; these limits
    do not bound model weights, logits, temporary tensors or peak memory.
    The result depends on a cooperative single-owner model and is not an
    independently authenticated execution record.
    """
    from transformers import GPT2LMHeadModel

    if type(matched_control) is not bool:
        raise ValueError("matched_control must be a boolean")
    if type(model) is not GPT2LMHeadModel:
        raise TypeError("model must be an unmodified GPT2LMHeadModel")
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("model and all submodules must be in evaluation mode")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("model parameters must be CPU float32")
    module_hooks = torch.nn.modules.module
    if (module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks
            or any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())):
        raise ValueError("model must have no existing local or global forward hooks")
    if (not isinstance(input_ids, torch.Tensor) or input_ids.device.type != "cpu"
            or input_ids.dtype != torch.int64 or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or not 2 <= input_ids.shape[1] <= 16):
        raise ValueError("input_ids must be one CPU int64 row of 2--16 tokens")
    ids = input_ids.detach().clone()
    hidden = model.config.n_embd
    vocab = model.config.vocab_size
    if (type(hidden) is not int or not 1 <= hidden <= 1024
            or type(vocab) is not int or not 1 <= vocab <= 65536):
        raise ValueError("hidden size or vocabulary exceeds this workflow's limits")
    if bool((ids < 0).any() or (ids >= vocab).any()):
        raise ValueError("input token ID outside vocabulary")
    if (type(coordinate) is not int or not 0 <= coordinate < hidden
            or isinstance(amount, bool) or not isinstance(amount, (int, float))
            or not math.isfinite(amount) or amount == 0):
        raise ValueError("provide a valid coordinate and finite nonzero amount")
    if type(readout_token) is not int or not 0 <= readout_token < vocab:
        raise ValueError("readout_token must be a vocabulary ID")
    if (not isinstance(layers, (tuple, list)) or not 1 <= len(layers) <= 6
            or any(type(x) is not int or x < 0 or x >= model.config.n_layer for x in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("select one to six distinct existing layers")
    for site in (target_site, comparison_site):
        if (not isinstance(site, tuple) or len(site) != 2 or type(site[0]) is not int
                or site[0] not in layers or site[1] not in ("attn", "mlp", "block")):
            raise ValueError("each site must be a selected (layer, kind) tuple")
    if target_site == comparison_site or target_site[0] != comparison_site[0]:
        raise ValueError("target and comparison must be distinct sites in the same layer")

    token = ids.shape[1] - 1
    mask = torch.ones_like(ids)
    trace_limit = 128 * 1024
    event_limit = 18
    full_bytes = ids.shape[1] * hidden * 4
    vector_bytes = hidden * 4
    plan_limits = {"max_tensor_bytes": full_bytes,
                   "max_capture_bytes": 2 * full_bytes + vector_bytes,
                   "max_source_bytes": vector_bytes, "max_steps": 4,
                   "max_invocations": 1}

    def forward(**kwargs):
        logits = model(attention_mask=mask, use_cache=False, **kwargs).logits.detach()
        if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("expected finite float32 logits")
        return logits

    def trace(run_id, embeds):
        session = GPT2TraceSession(
            model, run_id=run_id, layers=tuple(layers), targets=("attn", "mlp", "block"),
            samples=(0,), tokens=(token,), max_events=event_limit,
            max_trace_bytes=trace_limit, capture_device="cpu",
            forward_context=GenerationContext(stage="forward", use_cache=False),
        )
        with session:
            logits = forward(inputs_embeds=embeds)
        if not session.successful:
            raise RuntimeError("trace did not complete")
        return session, logits

    with torch.no_grad():
        clean_embeds = model.get_input_embeddings()(ids).detach().clone()
        if clean_embeds.dtype != torch.float32 or not bool(torch.isfinite(clean_embeds).all()):
            raise RuntimeError("expected finite float32 input embeddings")
        corrupted_embeds = clean_embeds.clone()
        corrupted_embeds[0, token, coordinate] += amount
        if not bool(torch.isfinite(corrupted_embeds).all()):
            raise ValueError("corruption produced a nonfinite embedding")
        before_coordinate = float(clean_embeds[0, token, coordinate])
        after_coordinate = float(corrupted_embeds[0, token, coordinate])
        if before_coordinate == after_coordinate:
            raise ValueError("corruption rounded to an unchanged coordinate")
        other_bits = corrupted_embeds.clone()
        other_bits[0, token, coordinate] = clean_embeds[0, token, coordinate]
        if not _same_bits(clean_embeds, other_bits):
            raise RuntimeError("corruption changed another embedding coordinate")

        ordinary_logits = forward(input_ids=ids)
        clean_trace, clean_logits = trace("workflow-clean", clean_embeds)
        corrupt_trace, corrupt_logits = trace("workflow-corrupt", corrupted_embeds)
        if not _same_bits(ordinary_logits, clean_logits):
            raise RuntimeError("input-ID and clean-embedding forwards differ")

        expected_sites = [(layer, kind) for layer in sorted(layers)
                          for kind in ("attn", "mlp", "block")]
        for session in (clean_trace, corrupt_trace):
            if len(session.events) != len(expected_sites):
                raise RuntimeError("trace did not retain every selected site")
            for index, (event, (layer, kind)) in enumerate(zip(session.events, expected_sites)):
                if (event.sequence != index or event.forward_index != 0 or event.invocation != 0
                        or event.module_path != _site_path((layer, kind))
                        or event.module_kind != kind or event.layer_index != layer
                        or event.output_path != ((0,) if kind in ("attn", "block") else ())
                        or event.batch_positions != (0,) or event.token_positions != (token,)
                        or event.shape != (1, ids.shape[1], hidden)
                        or event.saved_shape != (1, 1, hidden)
                        or event.dtype != "torch.float32" or event.device != "cpu"
                        or event.generation != GenerationContext(stage="forward", use_cache=False)
                        or event.value.dtype != torch.float32 or event.value.device.type != "cpu"):
                    raise RuntimeError("trace coordinates or execution order do not match the selected sites")

        clean_obs = [TensorObservation(event.to_schema(), event.value)
                     for event in clean_trace.events]
        corrupt_obs = [TensorObservation(event.to_schema(), event.value)
                       for event in corrupt_trace.events]
        # The same model, input IDs, masks, ordered taps and uncached forward
        # are used above; only the explicit embedding edit differs. The trace
        # sessions' successful exit attests to their declared captures.
        comparison = compare_transformer_traces(
            clean_obs, corrupt_obs, reference_complete=clean_trace.successful,
            candidate_complete=corrupt_trace.successful, pairing_verified=True,
            execution_order_verified=True,
        )
        def measure(logits):
            value = logits[0, -1, readout_token].item()
            return {"readout_logit": value,
                    "max_absolute_logit_distance_to_clean": (logits.double() - clean_logits.double()).abs().max().item(),
                    "max_absolute_logit_distance_to_corrupt": (logits.double() - corrupt_logits.double()).abs().max().item()}

        candidates = [row["coordinate"] for row in comparison["comparisons"]
                      if row["result"] == "different"]
        target_path, other_path = _site_path(target_site), _site_path(comparison_site)
        result = {
            "schema": ("gpt2-known-corruption-workflow-v2" if matched_control
                       else "gpt2-known-corruption-workflow-v1"), "input_ids": ids[0].tolist(),
            "attention_mask": mask[0].tolist(), "readout_token_id": readout_token,
            "corruption": {"token_index": token, "coordinate": coordinate,
                           "requested_addition": amount, "before": before_coordinate,
                           "after": after_coordinate,
                           "realized_addition": after_coordinate - before_coordinate},
            "trace_limits": {"max_events_per_run": event_limit,
                             "max_tensor_bytes_per_run": trace_limit,
                             "clean_tensor_bytes": clean_trace.tensor_bytes,
                             "corrupt_tensor_bytes": corrupt_trace.tensor_bytes},
            "localization": comparison, "candidate_sites": candidates,
            "target_site": target_path, "comparison_site": other_path,
            "baselines": {"clean": measure(clean_logits), "corrupt": measure(corrupt_logits),
                          "input_ids_equal_clean_embeds": _same_bits(ordinary_logits, clean_logits)},
            "interpretation": (
                "Differences describe only selected observations after an artificial input "
                "edit. Testing a clean-state patch at an observed site does not establish "
                "the causal origin or a natural feature; the comparison-site patch is not "
                "assumed to be ineffective or matched in dose. In the example, replacing "
                "the only changed token's full layer-0 block output restores that whole "
                "block output by construction, so downstream restoration is a mechanics "
                "check rather than evidence of a unique location."
            ) + (" The original full comparison-site patch is distinct from the optional "
                 "matched pair of partial clean-direction patches, when available. Equal "
                 "raw L2 doses do not match module scales, directions, computation "
                 "positions or sensitivities."
                 if matched_control else ""),
        }
        if (comparison["status"] != "difference_observed"
                or not any(item["module_path"] == target_path for item in candidates)):
            reason = ("trace comparison inconclusive" if comparison["status"] == "inconclusive"
                      else "target is not an observed changed site")
            result["interventions"] = {"status": "not_run", "reason": reason}
            if matched_control:
                result["interventions"]["matched_control"] = {"status": "not_run", "reason": reason}
            return result

        clean_by_path = {event.module_path: event.value[0, 0] for event in clean_trace.events}
        corrupt_by_path = {event.module_path: event.value[0, 0] for event in corrupt_trace.events}

        def patch(path, *, self_patch=False, replacement=None):
            full = {"module_path": path, "invocation": 0}
            selected = dict(full, batch_index=0, token_index=token)
            steps = [dict(full, name="before_all", operation="capture")]
            if self_patch:
                steps.extend([dict(selected, name="self_source", operation="capture"),
                              dict(selected, name="patch", operation="patch", source_capture="self_source")])
            else:
                value = clean_by_path[path] if replacement is None else replacement
                steps.append(dict(selected, name="patch", operation="patch", value=value))
            steps.append(dict(full, name="after_all", operation="capture"))
            plan = ActivationPlan(model, steps, **plan_limits)
            with plan:
                logits = forward(inputs_embeds=corrupted_embeds)
            before, after = plan.captures["before_all"], plan.captures["after_all"]
            expected = (before[0, token] if self_patch else
                        clean_by_path[path] if replacement is None else replacement)
            unaffected_before = torch.cat((before[:, :token], before[:, token + 1:]), dim=1)
            unaffected_after = torch.cat((after[:, :token], after[:, token + 1:]), dim=1)
            checks = {
                "receiver_matches_corrupt_trace": _same_bits(before[0, token], corrupt_by_path[path]),
                "selected_output_matches_source": _same_bits(after[0, token], expected),
                "other_positions_unchanged": _same_bits(unaffected_before, unaffected_after),
            }
            if not all(checks.values()):
                raise RuntimeError(f"intervention landing check failed: {checks}")
            displacement = (after[0, token].double() - before[0, token].double()).norm().item()
            return logits, {"checks": checks, "realized_displacement_l2": displacement,
                            "steps": plan.metadata["steps"]}

        self_logits, self_witness = patch(target_path, self_patch=True)
        restored_logits, target_witness = patch(target_path)
        other_logits, other_witness = patch(other_path)
        matched = None
        if matched_control:
            matched, values = _plan_matched_patches(clean_by_path[target_path], corrupt_by_path[target_path],
                                                    clean_by_path[other_path], corrupt_by_path[other_path])
            if values is not None:
                target_matched_logits, target_matched_witness = patch(target_path, replacement=values[0])
                other_matched_logits, other_matched_witness = patch(other_path, replacement=values[1])
                target_dose = target_matched_witness["realized_displacement_l2"]
                other_dose = other_matched_witness["realized_displacement_l2"]
                if not _dose_matches(matched["common_dose_l2"], matched["absolute_dose_tolerance"],
                                     target_dose, other_dose):
                    raise RuntimeError("actual same-forward matched doses failed their tolerance")
                matched.update({"status": "complete",
                    "target": {"interpolation_fraction": matched["target_fraction"],
                               "measurement": measure(target_matched_logits), "witness": target_matched_witness},
                    "comparison": {"interpolation_fraction": matched["comparison_fraction"],
                                   "measurement": measure(other_matched_logits), "witness": other_matched_witness}})
        clean_repeat = forward(inputs_embeds=clean_embeds)
        corrupt_repeat = forward(inputs_embeds=corrupted_embeds)
        controls = {
            "input_ids_equal_clean_embeds": _same_bits(ordinary_logits, clean_logits),
            "target_self_patch_equal_corrupt": _same_bits(self_logits, corrupt_logits),
            "post_session_clean_repeat_equal": _same_bits(clean_repeat, clean_logits),
            "post_session_corrupt_repeat_equal": _same_bits(corrupt_repeat, corrupt_logits),
        }
        if not all(controls.values()):
            raise RuntimeError(f"workflow control failed: {controls}")

        result["interventions"] = {
            "status": "complete", "plan_limits": plan_limits, "controls": controls,
            "target_self_patch": {"measurement": measure(self_logits), "witness": self_witness},
            "target_patch": {"measurement": measure(restored_logits), "witness": target_witness},
            "comparison_patch": {"measurement": measure(other_logits), "witness": other_witness},
        }
        if matched_control:
            result["interventions"]["matched_control"] = matched
        return result
