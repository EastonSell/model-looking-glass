"""Maintenance-only v1-compatible candidate derivation with explicit cleanup.

This does not replace the frozen v1 study entry point or its source binding.
Any caller must discard the isolated model/process after an uncertain failure.
"""

import math

import torch

from . import targeted_edit_cpu_v1 as v1


def _is_cancellation(error):
    return isinstance(error, (KeyboardInterrupt, SystemExit))


def _prefer_error(previous, current):
    if previous is None or (_is_cancellation(current) and not _is_cancellation(previous)):
        return current
    return previous


def _cleanup_candidate(handles, original_flags, weight):
    """Try every recorded cleanup operation, retaining the preferred error."""
    caught = None
    for handle in handles:
        for _ in range(2):
            try:
                handle.remove()
                break
            except BaseException as exc:
                caught = _prefer_error(caught, exc)
    for _ in range(2):
        try:
            weight.grad = None
            break
        except BaseException as exc:
            caught = _prefer_error(caught, exc)
    for parameter, flag in original_flags:
        for _ in range(2):
            try:
                parameter.requires_grad_(flag)
                break
            except BaseException as exc:
                caught = _prefer_error(caught, exc)
    return caught


def derive_candidate(model, module, ids, target, competitor, *, expected_shape=None):
    """Derive v1's two updates; a cleanup failure never produces success."""
    weight = module.weight
    if expected_shape is not None and tuple(weight.shape) != tuple(expected_shape):
        raise v1.StudyError("unexpected projection shape")
    captured = {}

    def before(_module, inputs):
        if "input" in captured or len(inputs) != 1 or inputs[0].ndim != 3:
            raise v1.StudyError("unexpected projection input")
        captured["input"] = inputs[0][0, -1].detach().clone()

    def after(_module, _inputs, output):
        if "output" in captured or not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise v1.StudyError("unexpected projection output")
        output.retain_grad()
        captured["output"] = output

    handles = []
    original_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    body_error = None
    try:
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
        for parameter, _ in original_flags:
            parameter.requires_grad_(False)
        weight.requires_grad_(True)
        weight.grad = None
        tokens = torch.tensor([ids], dtype=torch.int64)
        logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        if logits.shape[:2] != tokens.shape or not torch.isfinite(logits).all():
            raise v1.StudyError("invalid discovery logits")
        margin = logits[0, -1, target] - logits[0, -1, competitor]
        discovery_row = logits[0, -1].detach().contiguous()
        discovery_sha = v1._tensor_sha(discovery_row)
        margin_float64 = float((discovery_row[target].to(torch.float64) -
                                discovery_row[competitor].to(torch.float64)).item())
        margin.backward()
        if set(captured) != {"input", "output"} or captured["output"].grad is None or weight.grad is None:
            raise v1.StudyError("missing projection gradient")
        left, right = captured["input"], captured["output"].grad[0, -1].detach().clone()
        outer = torch.outer(left, right)
        if not torch.isfinite(weight.grad).all() or not torch.allclose(weight.grad, outer, rtol=1e-5, atol=1e-7):
            raise v1.StudyError("projection gradient is not the specified outer product")
        gradient_norm = v1._norm(weight.grad)
        if not math.isfinite(gradient_norm) or gradient_norm == 0:
            raise v1.StudyError("projection gradient is zero or nonfinite")
        residual = v1._norm(weight.grad.to(torch.float64) - outer.to(torch.float64)) / gradient_norm
        if not math.isfinite(residual) or residual > 1e-5:
            raise v1.StudyError("projection gradient has excessive rank-one residual")
        original_norm = v1._norm(weight)
        planned = v1.RELATIVE_NORM * original_norm
        gradient = v1._scale_outer(left, right, planned)
        generator = torch.Generator(device="cpu").manual_seed(v1.SEED)
        random_left = torch.randn(weight.shape[0], generator=generator, dtype=torch.float32)
        random_right = torch.randn(weight.shape[1], generator=generator, dtype=torch.float32)
        random = v1._scale_outer(random_left, random_right, planned)
        return gradient, random, {
            "discovery_objective_float32": float(margin.detach().item()),
            "discovery_margin_float64": margin_float64,
            "discovery_logit_sha256": discovery_sha,
            "weight_norm": original_norm, "planned_delta_norm": planned,
            "gradient_norm": gradient_norm, "input_factor_norm": v1._norm(left),
            "output_factor_norm": v1._norm(right), "gradient_outer_relative_residual": residual,
        }
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            cleanup_error = _cleanup_candidate(handles, original_flags, weight)
        except BaseException as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            if body_error is None or cleanup_error is body_error:
                raise cleanup_error
            if _is_cancellation(cleanup_error) and not _is_cancellation(body_error):
                raise cleanup_error from body_error
            raise body_error from cleanup_error
