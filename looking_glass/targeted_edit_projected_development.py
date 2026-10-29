"""Private projected-edit development work; no model loader or live entry.

The caller must bind the source, checkpoint, exact public discovery and
calibration rows, runtime, exposure/custody and a supervised process before
invoking this helper. The returned record is a candidate, not an audited or
published result. The owner must dispose the model, reap the process and
publish the complete pre-evaluation record before any evaluation-row forward.
The caller must freeze and exclusively own the discovery and calibration
token rows for the entire call. The owner must bind the private candidate and
every artifact in its inventory before any evaluation-row forward.
Calibration is captured one row at a time; its per-row buffers overlap the
concatenated buffers, so the output-size bound does not bound peak memory.
Gradient predictions describe the grad-enabled execution. Matching site
activations and baseline logits across modes does not prove equal Jacobians.
On any error or interrupted return, terminate the isolated owner and retain
the partial directory; do not replay a condition.
"""
from pathlib import Path

import torch

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_projected_capture as capture
from . import targeted_edit_projected_math as projected
from . import targeted_edit_v2_owner as owner


class DevelopmentError(ValueError):
    """The prospective development stage is incomplete or incompatible."""


def _bytes(tensor):
    if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
            or tensor.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(tensor).all())):
        raise DevelopmentError("a saved tensor is not finite CPU float32 or float64")
    dtype = "<f4" if tensor.dtype == torch.float32 else "<f8"
    return tensor.detach().contiguous().numpy().astype(dtype, copy=False).tobytes()


def _save(output, name, tensor, inventory):
    raw = _bytes(tensor)
    owner._write_new(output / name, raw)
    inventory[name] = {"sha256": v1._sha(raw), "bytes": len(raw),
                       "shape": list(tensor.shape),
                       "dtype": "float32-le" if tensor.dtype == torch.float32 else "float64-le"}


def _save_json(output, name, value, inventory):
    raw = v1._canonical(value)
    owner._write_new(output / name, raw)
    inventory[name] = {"sha256": v1._sha(raw), "bytes": len(raw), "dtype": "canonical-json"}


def _state(guard, condition, applied, add_before, add_after, forward_probes):
    return {"condition": condition, "model_state_sha256": guard.state_sha,
            "weight_sha256": guard.weight_sha, "buffer_state_sha256": guard.buffer_sha,
            "applied_weight_sha256": applied, "cleanup": guard.last_cleanup,
            "addition_denormal_probe_before": add_before, "addition_denormal_probe_after": add_after,
            "forward_denormal_probes": forward_probes}


def _calibration_rows(guard, token_rows):
    """Probe the calling thread around every individual calibration forward."""
    captures, probes = [], []
    for ids in token_rows:
        before = v1._denormal_probe()
        item = capture.capture_calibration(guard.model, guard.module, (ids,))
        after = v1._denormal_probe()
        captures.append(item)
        probes.append({"before": before, "after": after})
    if not captures:
        raise DevelopmentError("calibration rows are empty")
    return {name: torch.cat([item[name] for item in captures], dim=0)
            for name in ("inputs", "outputs", "logits")}, probes


def _no_grad_discovery(guard, ids, expected_size):
    before = v1._denormal_probe()
    captured = capture.capture_calibration(guard.model, guard.module, (ids,))
    if tuple(captured["logits"].shape) != (1, expected_size):
        raise DevelopmentError("a discovery logit row is invalid")
    after = v1._denormal_probe()
    return {"logits": captured["logits"][0], "inputs": captured["inputs"],
            "outputs": captured["outputs"]}, {"before": before, "after": after}


def prepare_development(guard, discovery_ids, calibration_rows, target, competitor, output):
    """Write a bounded-input development candidate without any evaluation row.

    The caller must require the real 3072 by 768 named site, the four frozen
    calibration rows and 18 total positions, and reserve peak memory before
    this call. The guard must be an exclusively held baseline model guard;
    the token rows must remain exclusively owned and unchanged during the
    call. This helper does not establish provenance or process supervision.
    """
    try:
        return _prepare_development(guard, discovery_ids, calibration_rows, target, competitor, output)
    except BaseException:
        # A failure before a condition or after its acknowledgement must fence
        # the model just as a failure inside a condition does. The isolated
        # owner still has to exit; this flag is not proof of cleanup.
        guard.poisoned = True
        raise


def _prepare_development(guard, discovery_ids, calibration_rows, target, competitor, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    inventory = {}
    guard.verify()
    discovery_probe_before = v1._denormal_probe()
    discovery = capture.capture_discovery(guard.model, guard.module, discovery_ids, target, competitor)
    discovery_probe_after = v1._denormal_probe()
    guard.verify()
    for name, tensor in (("discovery-logits.f32", discovery["logits"]),
                         ("discovery-grad-inputs.f32", discovery["site_inputs"]),
                         ("discovery-grad-outputs.f32", discovery["site_outputs"]),
                         ("input-factor.f32", discovery["input_factor"]),
                         ("output-factor.f32", discovery["output_factor"]),
                         ("full-gradient.f32", discovery["gradient"])):
        _save(output, name, tensor, inventory)

    observed, baseline_probes = {}, {}
    baseline_probes["before"] = v1._denormal_probe()

    def baseline_rows():
        baseline_probes["after"] = v1._denormal_probe()
        calibration, calibration_probes = _calibration_rows(guard, calibration_rows)
        observed.update(calibration)
        observed["discovery"], discovery_probes = _no_grad_discovery(
            guard, discovery_ids, discovery["logits"].numel()
        )
        observed["forward_probes"] = {"calibration": calibration_probes, "discovery": discovery_probes}
        for name, tensor in (("discovery-original-logits.f32", observed["discovery"]["logits"]),
                             ("discovery-original-inputs.f32", observed["discovery"]["inputs"]),
                             ("discovery-original-outputs.f32", observed["discovery"]["outputs"])):
            _save(output, name, tensor, inventory)
        if v1._tensor_sha(observed["discovery"]["logits"]) != v1._tensor_sha(discovery["logits"]):
            raise DevelopmentError("gradient and no-grad discovery rows differ")
        if (v1._tensor_sha(observed["discovery"]["inputs"]) != v1._tensor_sha(discovery["site_inputs"])
                or v1._tensor_sha(observed["discovery"]["outputs"]) != v1._tensor_sha(discovery["site_outputs"])):
            raise DevelopmentError("gradient and no-grad discovery site activations differ")

    guard.condition(None, guard.weight_sha, baseline_rows)
    baseline = observed
    for name, tensor in (("calibration-inputs.f32", baseline["inputs"]),
                         ("calibration-original-outputs.f32", baseline["outputs"]),
                         ("calibration-original-logits.f32", baseline["logits"])):
        _save(output, name, tensor, inventory)
    _save_json(output, "state-original.json",
               _state(guard, "original", guard.weight_sha,
                      baseline_probes["before"], baseline_probes["after"], baseline["forward_probes"]), inventory)
    guard.commit_condition()

    built = projected.prepare_projected_edits(
        guard.original, baseline["inputs"], discovery["input_factor"],
        discovery["output_factor"], discovery["gradient"],
    )
    for name, tensor in built["factors"].items():
        _save(output, "factor-" + name + ".f64", tensor, inventory)
    base_input_sha = v1._tensor_sha(baseline["inputs"])
    shifts, discovery_changes = {}, {}
    for condition in projected.CONDITIONS:
        actual, addition_probes = {}, {}
        attestation = built["attestation"]["conditions"][condition]
        addition_probes["before"] = v1._denormal_probe()

        def condition_rows():
            addition_probes["after"] = v1._denormal_probe()
            calibration, calibration_probes = _calibration_rows(guard, calibration_rows)
            actual.update(calibration)
            actual["discovery"], discovery_probes = _no_grad_discovery(
                guard, discovery_ids, discovery["logits"].numel()
            )
            actual["forward_probes"] = {"calibration": calibration_probes, "discovery": discovery_probes}
            actual["algebraic_probe_before"] = v1._denormal_probe()
            realized = guard.weight.detach().to(torch.float64) - guard.original.to(torch.float64)
            algebraic = baseline["inputs"].to(torch.float64) @ realized
            actual["algebraic_max_by_token"] = torch.max(torch.abs(algebraic), dim=1).values.tolist()
            actual["algebraic_probe_after"] = v1._denormal_probe()

        guard.condition(built["updates"][condition], attestation["realized_weight_sha256"], condition_rows)
        if (actual["inputs"].shape != baseline["inputs"].shape
                or actual["outputs"].shape != baseline["outputs"].shape
                or actual["logits"].shape != baseline["logits"].shape
                or v1._tensor_sha(actual["inputs"]) != base_input_sha
                or v1._tensor_sha(actual["discovery"]["inputs"]) !=
                   v1._tensor_sha(baseline["discovery"]["inputs"])):
            raise DevelopmentError("calibration or discovery inputs changed under an edit")
        _save(output, "calibration-" + condition + "-outputs.f32", actual["outputs"], inventory)
        _save(output, "calibration-" + condition + "-logits.f32", actual["logits"], inventory)
        _save(output, "discovery-" + condition + "-logits.f32", actual["discovery"]["logits"], inventory)
        _save(output, "discovery-" + condition + "-inputs.f32", actual["discovery"]["inputs"], inventory)
        _save(output, "discovery-" + condition + "-outputs.f32", actual["discovery"]["outputs"], inventory)
        before = v1._denormal_probe()
        difference = actual["outputs"].to(torch.float64) - baseline["outputs"].to(torch.float64)
        per_token = torch.max(torch.abs(difference), dim=1).values.tolist()
        margin = float(actual["discovery"]["logits"][target].to(torch.float64)
                       - actual["discovery"]["logits"][competitor].to(torch.float64))
        after = v1._denormal_probe()
        shifts[condition] = {
            "max_output_abs": max(per_token), "max_output_abs_by_token": per_token,
            "max_algebraic_abs": max(actual["algebraic_max_by_token"]),
            "max_algebraic_abs_by_token": actual["algebraic_max_by_token"],
            "algebraic_probe_before": actual["algebraic_probe_before"],
            "algebraic_probe_after": actual["algebraic_probe_after"],
            "denormal_probe_before": before, "denormal_probe_after": after,
        }
        if shifts[condition]["max_algebraic_abs"] != attestation["calibration_algebraic_max_abs"]:
            raise DevelopmentError("the realized calibration product differs from construction")
        discovery_changes[condition] = {"margin": margin, "change": margin - discovery["margin"]}
        _save_json(output, "state-" + condition + ".json",
                   _state(guard, condition, attestation["realized_weight_sha256"],
                          addition_probes["before"], addition_probes["after"], actual["forward_probes"]), inventory)
        guard.commit_condition()
    guard.verify()
    candidate = {"version": "targeted-edit-projected-development-v1",
                 "status": "private-candidate", "discovery_margin": discovery["margin"],
                 "discovery_denormal_probe_before": discovery_probe_before,
                 "discovery_denormal_probe_after": discovery_probe_after,
                 "discovery_token_ids": list(discovery_ids),
                 "calibration_token_ids": [list(row) for row in calibration_rows],
                 "target_token_id": target, "competitor_token_id": competitor,
                 "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                 "buffer_state_sha256": guard.buffer_sha, "runtime_settings": dict(guard.runtime_settings),
                 "construction": built["attestation"], "calibration_shifts": shifts,
                 "discovery_changes": discovery_changes,
                 "gradient_prediction_scope": "grad-enabled derivative; no-grad outcome comparison, not proven Jacobian equivalence",
                 "artifacts": inventory,
                 "scope": "recorded computations; model provenance and cleanup require the owner"}
    owner._json_new(output / "pre-evaluation-candidate.json", candidate)
    return candidate
