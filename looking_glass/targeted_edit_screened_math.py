"""Prospective screened-study edit construction; does not load or run a model.

The caller must own the input tensors exclusively while they are copied. This
numerical helper does not authorize a study, verify model/source provenance,
publish a pre-evaluation record, or establish that any update was applied.
"""
import hashlib
import math

import torch

from . import targeted_edit_cpu_v1 as v1


CONDITIONS = ("projected", "unprojected", "projected_random", "opposite", "nuisance")
DOSE = 0.001
RANK_CUTOFF = 1e-10
MIN_RETAINED_FRACTION = 1e-6
MAX_NULL_RESIDUAL = 1e-8
MAX_GRADIENT_RESIDUAL = 1e-5
RANDOM_SEED = 20261103
NUISANCE_SEED = 20261104
DOSE_GRID = (0.0005, 0.000625, 0.00075, 0.000875, 0.001, 0.001125, 0.00125)


class ProjectionError(ValueError):
    """A fixed construction constraint was not met."""


def _norm(value):
    result = float(torch.linalg.vector_norm(value).item())
    if not math.isfinite(result) or result <= 0:
        raise ProjectionError("a required norm is nonfinite or zero")
    return result


def _finite_tensor(value, dimensions):
    if (type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.layout != torch.strided
            or value.ndim != dimensions or any(size <= 0 for size in value.shape)
            or not bool(torch.isfinite(value).all())):
        raise ProjectionError("expected a finite, nonempty CPU float32 tensor")
    return value.detach().contiguous().clone()


def _project(value, rows):
    return value - rows.T @ (rows @ value)


def _fraction(projected, original_norm):
    retained = _norm(projected) / original_norm
    if not math.isfinite(retained) or retained < MIN_RETAINED_FRACTION:
        raise ProjectionError("a projected factor collapsed")
    return retained


def _planned_outer(left, right, planned_norm):
    outer = torch.outer(left, right)
    if not bool(torch.isfinite(outer).all()):
        raise ProjectionError("a planned direction is nonfinite")
    direction = outer * (planned_norm / _norm(outer))
    if not bool(torch.isfinite(direction).all()):
        raise ProjectionError("a scaled direction is nonfinite")
    return direction


def _prediction(full_gradient, rank_one, residual_norm, direction):
    """Report a theoretical real-arithmetic bound beside floating reductions.

    A reader that checks the bound numerically must declare a rounding
    tolerance; the direct full-gradient dot is the primary prediction.
    """
    before = v1._denormal_probe()
    norm = _norm(direction)
    direct = float(torch.sum(full_gradient * direction).item())
    ideal = float(torch.sum(rank_one * direction).item())
    bound = residual_norm * norm
    if not all(math.isfinite(value) for value in (direct, ideal, bound)):
        raise ProjectionError("a first-order prediction is nonfinite")
    after = v1._denormal_probe()
    return {"full_gradient_dot": direct, "rank_one_dot": ideal,
            "residual_bound": bound, "denormal_probe_before": before,
            "denormal_probe_after": after}


def _planned_sha(value):
    """Hash a contiguous little-endian float64 planned direction."""
    data = value.detach().contiguous().numpy().astype("<f8", copy=False).tobytes()
    return hashlib.sha256(data).hexdigest()


def prepare_projected_edits(weight, calibration, input_factor, output_factor, gradient):
    """Construct fixed-dose directions and finite-precision attestations.

    The tensors are CPU float32 values with shapes (d, m), (n, d), (d,),
    (m,), and (d, m) respectively. Smaller shapes are supported for synthetic
    tests; the actual owner must separately require the protocol's 18 by 3072
    calibration matrix and 3072 by 768 site weight, and reserve sufficient
    memory before calling. This helper is not a memory-admission guard. The
    result contains mutable updates and factors; the owner must retain
    exclusive custody or recheck their digests before using them. The caller
    must already have enabled denormal preservation; this helper probes it
    without resetting an unexpected mode change.
    """
    initial_probe = v1._denormal_probe()
    with torch.no_grad():
        w = _finite_tensor(weight, 2)
        x = _finite_tensor(calibration, 2)
        h = _finite_tensor(input_factor, 1)
        g = _finite_tensor(output_factor, 1)
        full = _finite_tensor(gradient, 2)
        if (x.shape[1] != w.shape[0] or h.shape != (w.shape[0],)
                or g.shape != (w.shape[1],) or full.shape != w.shape):
            raise ProjectionError("projection inputs have incompatible shapes")
        w64, x64, h64, g64, full64 = (item.to(torch.float64) for item in (w, x, h, g, full))
        weight_norm, x_norm, h_norm, g_norm, gradient_norm = (
            _norm(item) for item in (w64, x64, h64, g64, full64)
        )
        rank_one = torch.outer(h64, g64)
        residual = full64 - rank_one
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if not math.isfinite(residual_norm) or residual_norm / gradient_norm > MAX_GRADIENT_RESIDUAL:
            raise ProjectionError("the full gradient is not sufficiently rank one")

        _u, singular, vh = torch.linalg.svd(x64, full_matrices=False)
        largest = float(singular[0].item())
        if not bool(torch.isfinite(singular).all()) or not math.isfinite(largest) or largest <= 0:
            raise ProjectionError("calibration singular values are invalid")
        rows = vh[singular > RANK_CUTOFF * largest]
        rank = int(rows.shape[0])
        if not rank:
            raise ProjectionError("calibration span has zero rank")
        projected_h = _project(h64, rows)
        retained = _fraction(projected_h, h_norm)
        discarded = float(torch.linalg.vector_norm(h64 - projected_h).item()) / h_norm
        null_residual = float(torch.linalg.vector_norm(x64 @ projected_h).item()) / (x_norm * h_norm)
        if not math.isfinite(null_residual) or null_residual > MAX_NULL_RESIDUAL:
            raise ProjectionError("calibration projection residual exceeds its bound")

        random_generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
        random_left = torch.randn(w.shape[0], generator=random_generator, dtype=torch.float64)
        random_right = torch.randn(w.shape[1], generator=random_generator, dtype=torch.float64)
        projected_random = _project(random_left, rows)
        random_retained = _fraction(projected_random, _norm(random_left))
        _norm(random_right)

        nuisance_generator = torch.Generator(device="cpu").manual_seed(NUISANCE_SEED)
        draw = torch.randn((w.shape[0], rank), generator=nuisance_generator, dtype=torch.float64)
        q, r = torch.linalg.qr(draw, mode="reduced")
        diagonal = torch.abs(torch.diagonal(r))
        maximum = float(diagonal.max().item())
        if (not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(r).all())
                or not math.isfinite(maximum) or maximum <= 0
                or not bool((diagonal > 1e-12 * maximum).all())):
            raise ProjectionError("nuisance QR is rank deficient or nonfinite")
        nuisance_h = h64 - q @ (q.T @ h64)
        nuisance_retained = _fraction(nuisance_h, h_norm)

        planned_norm = DOSE * weight_norm
        if not math.isfinite(planned_norm) or planned_norm <= 0:
            raise ProjectionError("the planned norm is invalid")
        plans = {
            "projected": _planned_outer(projected_h, g64, planned_norm),
            "unprojected": _planned_outer(h64, g64, planned_norm),
            "projected_random": _planned_outer(projected_random, random_right, planned_norm),
            "nuisance": _planned_outer(nuisance_h, g64, planned_norm),
        }
        plans["opposite"] = -plans["projected"]
        updates = {}
        attestations = {}
        for name in CONDITIONS:
            planned = plans[name]
            intended = (-updates["projected"] if name == "opposite"
                        else planned.to(torch.float32))
            if not bool(torch.isfinite(intended).all()):
                raise ProjectionError("an intended update is nonfinite")
            add_before = v1._denormal_probe()
            changed = w + intended
            add_after = v1._denormal_probe()
            if not bool(torch.isfinite(changed).all()):
                raise ProjectionError("an edited weight is nonfinite")
            actual = changed.to(torch.float64) - w64
            actual_norm = _norm(actual)
            if not 0.999 <= actual_norm / planned_norm <= 1.001:
                raise ProjectionError("a realized norm differs from its planned dose")
            updates[name] = intended.contiguous()
            attestations[name] = {
                "planned_float64_sha256": _planned_sha(planned),
                "intended_delta_sha256": v1._tensor_sha(updates[name]),
                "realized_weight_sha256": v1._tensor_sha(changed),
                "realized_delta_float64_sha256": _planned_sha(actual),
                "realized_delta_norm": actual_norm,
                "realized_to_planned": actual_norm / planned_norm,
                "addition_denormal_probe_before": add_before,
                "addition_denormal_probe_after": add_after,
                "planned_prediction": _prediction(full64, rank_one, residual_norm, planned),
                "intended_prediction": _prediction(full64, rank_one, residual_norm, intended.to(torch.float64)),
                "realized_prediction": _prediction(full64, rank_one, residual_norm, actual),
                "calibration_algebraic_max_abs": float(torch.max(torch.abs(x64 @ actual)).item()),
            }
        projected_norm = attestations["projected"]["realized_delta_norm"]
        for name in CONDITIONS:
            ratio = attestations[name]["realized_delta_norm"] / projected_norm
            if not 0.999 <= ratio <= 1.001:
                raise ProjectionError("a control does not match the projected realized norm")
            attestations[name]["realized_to_projected"] = ratio
        final_probe = v1._denormal_probe()
        denominator = attestations["unprojected"]["realized_prediction"]["full_gradient_dot"]
        numerator = attestations["projected"]["realized_prediction"]["full_gradient_dot"]
        return {
            "updates": updates,
            "factors": {"calibration_basis": rows, "projected_input": projected_h,
                        "random_input": random_left, "random_output": random_right,
                        "projected_random_input": projected_random, "nuisance_basis": q,
                        "nuisance_input": nuisance_h},
            "attestation": {
                "denormal_probe_before": initial_probe, "denormal_probe_after": final_probe,
                "planned_hash_format": "contiguous C-order little-endian float64",
                "random_seed": RANDOM_SEED, "nuisance_seed": NUISANCE_SEED,
                "relative_dose": DOSE, "rank_relative_cutoff": RANK_CUTOFF,
                "minimum_retained_fraction": MIN_RETAINED_FRACTION,
                "weight_sha256": v1._tensor_sha(w), "calibration_sha256": v1._tensor_sha(x),
                "input_factor_sha256": v1._tensor_sha(h), "output_factor_sha256": v1._tensor_sha(g),
                "full_gradient_sha256": v1._tensor_sha(full),
                "weight_shape": list(w.shape), "calibration_shape": list(x.shape),
                "singular_values": [float(value) for value in singular.tolist()],
                "rank": rank, "weight_norm": weight_norm, "calibration_norm": x_norm,
                "input_factor_norm": h_norm, "output_factor_norm": g_norm,
                "full_gradient_norm": gradient_norm, "gradient_residual_norm": residual_norm,
                "gradient_relative_residual": residual_norm / gradient_norm,
                "retained_fraction": retained, "discarded_fraction": discarded,
                "calibration_relative_residual": null_residual,
                "random_retained_fraction": random_retained,
                "nuisance_retained_fraction": nuisance_retained,
                "nuisance_qr_diagonal": [float(value) for value in diagonal.tolist()],
                "planned_delta_norm": planned_norm,
                "realized_prediction_ratio": None if denominator == 0 else numerator / denominator,
                "conditions": attestations,
            },
        }


def prepare_unprojected_grid(weight, calibration, input_factor, output_factor, gradient, base):
    """Prepare every fixed-dose unprojected candidate before any screen.

    Inputs and the prior construction remain caller-owned and quiescent. The
    owner must bind all returned updates, authorize the checkpoint and source,
    reserve memory and dispose its isolated process; this is a pure helper.
    """
    before = v1._denormal_probe()
    with torch.no_grad():
        w = _finite_tensor(weight, 2)
        x = _finite_tensor(calibration, 2)
        h = _finite_tensor(input_factor, 1)
        g = _finite_tensor(output_factor, 1)
        full = _finite_tensor(gradient, 2)
        if (x.shape[1] != w.shape[0] or h.shape != (w.shape[0],)
                or g.shape != (w.shape[1],) or full.shape != w.shape):
            raise ProjectionError("grid inputs have incompatible shapes")
        if not isinstance(base, dict) or set(base) != {"updates", "factors", "attestation"}:
            raise ProjectionError("base construction differs")
        info = base["attestation"]
        for name, tensor in (("weight", w), ("calibration", x), ("input_factor", h),
                             ("output_factor", g), ("full_gradient", full)):
            if info.get(name + "_sha256") != v1._tensor_sha(tensor):
                raise ProjectionError("grid inputs differ from the base construction")
        if info.get("random_seed") != RANDOM_SEED or info.get("nuisance_seed") != NUISANCE_SEED:
            raise ProjectionError("base control seeds differ")
        w64, x64, h64, g64, full64 = (item.to(torch.float64) for item in (w, x, h, g, full))
        weight_norm = _norm(w64)
        rank_one = torch.outer(h64, g64)
        residual_norm = float(torch.linalg.vector_norm(full64 - rank_one).item())
        if not math.isfinite(residual_norm) or residual_norm / _norm(full64) > MAX_GRADIENT_RESIDUAL:
            raise ProjectionError("grid full gradient differs from the rank-one factor")
        updates, attestations = {}, []
        for dose in DOSE_GRID:
            planned_norm = dose * weight_norm
            plan = _planned_outer(h64, g64, planned_norm)
            intended = plan.to(torch.float32).contiguous()
            if not bool(torch.isfinite(intended).all()):
                raise ProjectionError("grid intended update is nonfinite")
            add_before = v1._denormal_probe()
            changed = w + intended
            add_after = v1._denormal_probe()
            if not bool(torch.isfinite(changed).all()):
                raise ProjectionError("grid edited weight is nonfinite")
            actual = changed.to(torch.float64) - w64
            realized_norm = _norm(actual)
            if not 0.999 <= realized_norm / planned_norm <= 1.001:
                raise ProjectionError("grid realized norm differs from its planned dose")
            record = {"dose": dose, "planned_norm": planned_norm,
                      "planned_float64_sha256": _planned_sha(plan),
                      "intended_delta_sha256": v1._tensor_sha(intended),
                      "realized_weight_sha256": v1._tensor_sha(changed),
                      "realized_delta_float64_sha256": _planned_sha(actual),
                      "realized_delta_norm": realized_norm,
                      "realized_to_planned": realized_norm / planned_norm,
                      "addition_denormal_probe_before": add_before,
                      "addition_denormal_probe_after": add_after,
                      "planned_prediction": _prediction(full64, rank_one, residual_norm, plan),
                      "intended_prediction": _prediction(full64, rank_one, residual_norm, intended.to(torch.float64)),
                      "realized_prediction": _prediction(full64, rank_one, residual_norm, actual),
                      "calibration_algebraic_max_abs": float(torch.max(torch.abs(x64 @ actual)).item())}
            if dose == DOSE:
                old = info["conditions"]["unprojected"]
                if (record["planned_float64_sha256"] != old["planned_float64_sha256"]
                        or record["intended_delta_sha256"] != old["intended_delta_sha256"]
                        or record["realized_weight_sha256"] != old["realized_weight_sha256"]
                        or record["realized_delta_float64_sha256"] != old["realized_delta_float64_sha256"]
                        or not torch.equal(intended, base["updates"]["unprojected"])):
                    raise ProjectionError("the equal-norm grid candidate differs")
            updates[dose] = intended
            attestations.append(record)
        after = v1._denormal_probe()
    return {"updates": updates, "attestation": {"denormal_probe_before": before,
            "denormal_probe_after": after, "candidates": attestations}}


def select_discovery_match(projected_gain, grid):
    """Apply the frozen floor, nearest-candidate and tolerance rules.

    Every planned dose must have one finite measurement in the frozen order;
    caller-owned records must remain quiescent during this call. This function
    cannot establish that the gains came from actual model forwards.
    """
    def number(value):
        return type(value) in (float, int) and math.isfinite(value)

    if not number(projected_gain) or type(grid) is not list or len(grid) != len(DOSE_GRID):
        raise ProjectionError("projected gain or grid shape differs")
    candidates = []
    for expected, item in zip(DOSE_GRID, grid):
        if (type(item) is not dict or set(item) != {"dose", "gain"}
                or not number(item["dose"]) or item["dose"] != expected
                or not number(item["gain"])):
            raise ProjectionError("grid gain or order differs")
        if item["gain"] > 0.25:
            candidates.append((abs(float(item["gain"]) - float(projected_gain)), expected, float(item["gain"])))
    nearest = min(candidates, default=None)
    tolerance = max(0.05, 0.10 * abs(float(projected_gain)))
    if projected_gain <= 0.25:
        reason = "projected_gain_not_above_floor"
    elif nearest is None:
        reason = "no_unprojected_gain_above_floor"
    elif nearest[0] > tolerance:
        reason = "nearest_discrepancy_exceeds_tolerance"
    else:
        reason = None
    return {"available": reason is None, "selected_dose": nearest[1] if reason is None else None,
            "nearest_dose": None if nearest is None else nearest[1],
            "nearest_gain": None if nearest is None else nearest[2],
            "nearest_discrepancy": None if nearest is None else nearest[0],
            "tolerance": tolerance, "reason": reason,
            "shared_equal_norm_arm": bool(reason is None and nearest[1] == DOSE)}
