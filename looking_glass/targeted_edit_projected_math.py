"""CPU-only projected rank-one edit construction; does not load or run a model.

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
RANDOM_SEED = 20261102
NUISANCE_SEED = 20261101


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
    exclusive custody or recheck their digests before using them. This call
    sets the calling thread's Torch denormal mode to preservation and does
    not restore its previous mode; the owner must account for that change.
    """
    if not torch.set_flush_denormal(False):
        raise ProjectionError("the calling thread cannot preserve denormals")
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
