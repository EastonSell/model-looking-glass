"""Deterministic CPU edit construction for the prospective v2 study.

These helpers do not load or forward a model. Matching recorded v1 values is
an integrity check, not proof of checkpoint provenance or an evaluation gate.
The caller must exclusively own all input tensors and keep them quiescent
through validation and construction; these checks are not atomic against a
concurrent writer.
"""

from __future__ import annotations

import math

import torch

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2


V1_PRECOMMIT_SHA256 = "cb666aa911db020b8904dab6f08f989e506d85e875d7cac65e72d3b94cef2fa5"
ENVIRONMENT_FIELDS = (
    "python_version", "platform", "machine", "processor", "torch_version", "numpy_version",
    "transformers_version", "torch_build_sha256", "torch_cpu_capability", "gpt2_source_sha256",
    "intraop_threads", "interop_threads", "deterministic_algorithms", "mkldnn_enabled",
    "omp_num_threads", "mkl_num_threads", "cuda_visibility", "cuda_initialized",
    "bytecode_attested", "denormal_probe",
)


def _tensor(value, *, shape=None):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
            value.dtype != torch.float32 or value.ndim != 2 or value.requires_grad or
            (shape is not None and tuple(value.shape) != tuple(shape)) or
            not bool(torch.isfinite(value).all())):
        raise v1.StudyError("invalid finite CPU float32 edit tensor")


def _positive_norm(value):
    result = v1._norm(value)
    if not math.isfinite(result) or result <= 0:
        raise v1.StudyError("edit tensor norm must be positive and finite")
    return result


def _dose_value(value):
    if type(value) is not float or value not in {dose for _, dose in v2.GRID}:
        raise v1.StudyError("dose differs from the fixed grid")
    return value


def construct_dose(original, v1_gradient, dose):
    """Rescale the rounded intended v1 update to a fixed planned v2 norm."""
    v1._denormal_probe()
    _tensor(original)
    _tensor(v1_gradient, shape=original.shape)
    dose = _dose_value(dose)
    desired = dose * _positive_norm(original)
    direction_norm = _positive_norm(v1_gradient)
    result = (v1_gradient.to(torch.float64) * (desired / direction_norm)).to(torch.float32)
    _tensor(result, shape=original.shape)
    _positive_norm(result)
    return result


def attest(original, delta, dose):
    """Measure float32 post-addition bytes and float64 realized-difference norm."""
    v1._denormal_probe()
    _tensor(original)
    _tensor(delta, shape=original.shape)
    dose = _dose_value(dose)
    original_norm = _positive_norm(original)
    planned = dose * original_norm
    changed = original.clone()
    changed.add_(delta)
    _tensor(changed, shape=original.shape)
    realized = _positive_norm(changed.to(torch.float64) - original.to(torch.float64))
    if not 0.999 <= realized / planned <= 1.001:
        raise v1.StudyError("realized edit does not match the planned dose")
    return {"planned_norm": planned, "intended_delta_sha256": v1._tensor_sha(delta),
            "realized_weight_sha256": v1._tensor_sha(changed), "realized_norm": realized,
            "realized_relative_norm": realized / original_norm}


def construct_controls(original, v1_gradient, dose):
    """Construct selected, one fresh seeded random and opposite edit."""
    selected = construct_dose(original, v1_gradient, dose)
    generator = torch.Generator(device="cpu").manual_seed(v2.RANDOM_SEED)
    left = torch.randn(original.shape[0], generator=generator, dtype=torch.float32, device="cpu")
    right = torch.randn(original.shape[1], generator=generator, dtype=torch.float32, device="cpu")
    random = v1._scale_outer(left, right, dose * _positive_norm(original))
    opposite = torch.neg(selected)
    updates = {"selected": selected, "random": random, "opposite": opposite}
    facts = {name: attest(original, delta, dose) for name, delta in updates.items()}
    selected_norm = facts["selected"]["realized_norm"]
    for name in ("random", "opposite"):
        if not 0.999 <= facts[name]["realized_norm"] / selected_norm <= 1.001:
            raise v1.StudyError("realized control norm differs from the selected edit")
    return updates, facts


def verify_v1_compatibility(reference_bytes, *, environment, model_metadata, model_state_sha256,
                            original, v1_gradient, v1_random, discovery):
    """Compare current reconstruction with the independently pinned v1 precommit.

    The caller separately verifies source files, checkpoint assets, imported
    runtime and model ownership before trusting these recorded comparisons.
    """
    v1._denormal_probe()
    if type(reference_bytes) is not bytes or v1._sha(reference_bytes) != V1_PRECOMMIT_SHA256:
        raise v1.StudyError("v1 reference precommit digest differs")
    reference = v1._decode_document(reference_bytes)
    _tensor(original)
    _tensor(v1_gradient, shape=original.shape)
    _tensor(v1_random, shape=original.shape)
    if (type(environment) is not dict or type(reference.get("environment")) is not dict or
            any(field not in environment or field not in reference["environment"] for field in ENVIRONMENT_FIELDS)):
        raise v1.StudyError("v1 execution environment fields are incomplete")
    try:
        current_env = {field: environment[field] for field in ENVIRONMENT_FIELDS}
        prior_env = {field: reference["environment"][field] for field in ENVIRONMENT_FIELDS}
        compatible = (v1._canonical(current_env) == v1._canonical(prior_env) and
                      v1._canonical(model_metadata) == v1._canonical(reference.get("model")) and
                      v1._canonical(discovery) == v1._canonical(reference.get("discovery")) and
                      v1._canonical(v1._candidate_attestation(original,
                          {"gradient": v1_gradient, "random": v1_random})) ==
                      v1._canonical(reference.get("candidates")))
    except (TypeError, ValueError, RecursionError, OverflowError, UnicodeError) as exc:
        raise v1.StudyError("invalid v1 comparison metadata") from exc
    if (reference.get("version") != v1.VERSION or reference.get("site") != v1.SITE or
            v1._canonical(reference.get("site_shape")) != v1._canonical(list(v1.WEIGHT_SHAPE)) or
            tuple(original.shape) != v1.WEIGHT_SHAPE or not compatible or
            model_state_sha256 != reference.get("model_state_sha256") or
            v1._tensor_sha(original) != reference.get("weight_sha256")):
        raise v1.StudyError("v1 reconstruction or execution environment is incompatible")
    return {"status": "matching_recorded_v1", "reference_precommit_sha256": V1_PRECOMMIT_SHA256,
            "model_state_sha256": model_state_sha256,
            "weight_sha256": reference["weight_sha256"],
            "discovery_logit_sha256": reference["discovery"]["discovery_logit_sha256"],
            "gradient_delta_sha256": reference["candidates"]["gradient"]["intended_delta_sha256"],
            "v1_random_delta_sha256": reference["candidates"]["random"]["intended_delta_sha256"]}
