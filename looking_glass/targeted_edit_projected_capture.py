"""Private activation capture primitives for the prospective projected study.

These functions do not load a model, verify provenance, publish evidence, or
authorize a forward. The caller must exclusively own the model, enforce the
frozen inputs and resource bounds, and terminate the isolated owner on any
failure or interruption. The caller must separately check complete state and
buffer restoration and the frozen runtime and denormal settings.
The cloned full gradient overlaps the model's weight gradient in memory;
calibration concatenation overlaps per-row buffers, and exception frames can
retain tensor aliases. Saved artifact sizes do not bound the process peak.
"""
import torch


class CaptureError(ValueError):
    """A model invocation did not match the capture contract."""


def _tokens(ids):
    if (not isinstance(ids, (tuple, list)) or not ids
            or any(type(value) is not int or value < 0 for value in ids)):
        raise CaptureError("token IDs must be a nonempty sequence of nonnegative integers")
    return torch.tensor([ids], dtype=torch.int64, device="cpu")


def _tensor(value, shape):
    if (type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.layout != torch.strided
            or tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all())):
        raise CaptureError("a captured tensor has the wrong shape, type, or values")
    return value.detach().contiguous().clone()


def _finish(actions, primary):
    """Attempt each cleanup, preserving the primary error or fresh cancellation."""
    failure = None
    for action in actions:
        try:
            action()
        except BaseException as error:
            if failure is None or (
                isinstance(error, (KeyboardInterrupt, SystemExit))
                and not isinstance(failure, (KeyboardInterrupt, SystemExit))
            ):
                failure = error
    if failure is not None and (
        primary is None or (
            isinstance(failure, (KeyboardInterrupt, SystemExit))
            and not isinstance(primary, (KeyboardInterrupt, SystemExit))
        )
    ):
        raise failure


def capture_discovery(model, module, ids, target, competitor):
    """Capture the final-position input, output gradient, full gradient and row.

    The site weight must be an unshared CPU float32 matrix with shape
    (input width, output width); the caller binds it to the real named
    parameter and checks the original model and inputs before invocation.
    Existing parameter gradients are rejected rather than discarded.
    """
    weight = module.weight
    if (type(weight) is not torch.nn.Parameter or weight.device.type != "cpu"
            or weight.dtype != torch.float32 or weight.ndim != 2):
        raise CaptureError("the site is not a CPU float32 matrix parameter")
    if type(target) is not int or type(competitor) is not int or min(target, competitor) < 0:
        raise CaptureError("target IDs are invalid")
    tokens = _tokens(ids)
    flags = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    if not any(parameter is weight for parameter, _ in flags):
        raise CaptureError("the site parameter is not part of the model")
    if any(parameter.grad is not None for parameter, _ in flags):
        raise CaptureError("the model has a preexisting gradient")
    captured = {}
    handles = []

    def before(_module, inputs):
        if "input" in captured or len(inputs) != 1:
            raise CaptureError("the site was invoked unexpectedly")
        captured["input"] = _tensor(inputs[0], (1, tokens.shape[1], weight.shape[0]))

    def after(_module, _inputs, output):
        if "output" in captured or type(output) is not torch.Tensor:
            raise CaptureError("the site output is unexpected")
        _tensor(output, (1, tokens.shape[1], weight.shape[1]))
        output.retain_grad()
        captured["output"] = output

    primary = None
    try:
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
        for parameter, _flag in flags:
            parameter.requires_grad_(False)
        weight.requires_grad_(True)
        logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        if (type(logits) is not torch.Tensor or logits.device.type != "cpu"
                or logits.dtype != torch.float32 or logits.layout != torch.strided or logits.ndim != 3
                or tuple(logits.shape[:2]) != tuple(tokens.shape)
                or max(target, competitor) >= logits.shape[2]
                or not bool(torch.isfinite(logits).all())):
            raise CaptureError("the discovery logits have an invalid shape")
        row = _tensor(logits[0, -1], (logits.shape[2],))
        objective = logits[0, -1, target] - logits[0, -1, competitor]
        if not bool(torch.isfinite(objective)):
            raise CaptureError("the float32 discovery objective is nonfinite")
        objective.backward()
        if set(captured) != {"input", "output"} or captured["output"].grad is None or weight.grad is None:
            raise CaptureError("the discovery gradient is missing")
        h = _tensor(captured["input"][0, -1], (weight.shape[0],))
        g = _tensor(captured["output"].grad[0, -1], (weight.shape[1],))
        gradient = _tensor(weight.grad, weight.shape)
        margin = float(row[target].to(torch.float64) - row[competitor].to(torch.float64))
        return {"input_factor": h, "output_factor": g, "gradient": gradient,
                "logits": row, "margin": margin}
    except BaseException as error:
        primary = error
        raise
    finally:
        actions = [handle.remove for handle in handles]
        actions.append(lambda: setattr(weight, "grad", None))
        actions.extend(lambda parameter=parameter, flag=flag: parameter.requires_grad_(flag)
                       for parameter, flag in flags)
        _finish(actions, primary)


def capture_calibration(model, module, token_rows):
    """Capture every site input/output token and final logit for each row.

    A real-study caller must require the four frozen rows and the resulting
    18-position matrix before constructing the projection.
    """
    weight = module.weight
    if (type(weight) is not torch.nn.Parameter or weight.device.type != "cpu"
            or weight.dtype != torch.float32 or weight.ndim != 2):
        raise CaptureError("the site is not a CPU float32 matrix parameter")
    if not isinstance(token_rows, (tuple, list)) or not token_rows:
        raise CaptureError("calibration rows are empty")
    inputs, outputs, logits_rows = [], [], []
    for ids in token_rows:
        tokens = _tokens(ids)
        captured, handles = {}, []

        def before(_module, values):
            if "input" in captured or len(values) != 1:
                raise CaptureError("the site was invoked unexpectedly")
            captured["input"] = _tensor(values[0], (1, tokens.shape[1], weight.shape[0]))

        def after(_module, _values, output):
            if "output" in captured:
                raise CaptureError("the site was invoked unexpectedly")
            captured["output"] = _tensor(output, (1, tokens.shape[1], weight.shape[1]))

        primary = None
        try:
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
            with torch.no_grad():
                result = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
            if (set(captured) != {"input", "output"} or type(result) is not torch.Tensor
                    or result.device.type != "cpu" or result.dtype != torch.float32
                    or result.layout != torch.strided
                    or result.ndim != 3 or tuple(result.shape[:2]) != tuple(tokens.shape)
                    or not bool(torch.isfinite(result).all())):
                raise CaptureError("the calibration invocation is incomplete")
            inputs.append(captured["input"][0])
            outputs.append(captured["output"][0])
            logits_rows.append(_tensor(result[0, -1], (result.shape[2],)))
        except BaseException as error:
            primary = error
            raise
        finally:
            _finish([handle.remove for handle in handles], primary)
    return {"inputs": torch.cat(inputs, dim=0), "outputs": torch.cat(outputs, dim=0),
            "logits": torch.stack(logits_rows, dim=0)}
