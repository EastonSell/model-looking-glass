"""Fixed-grid CPU edit construction for the proposed France top study.

This numerical helper does not load a model, authorize execution, establish
source or case custody, or prove an update was applied or restored. Inputs
must remain quiescent while copied, and results must remain exclusively owned
until their hashes are checked and they are applied by a supervised owner.
"""
import math

import torch

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_france_top_v2 as scoring
from . import targeted_edit_full_top as protocol
from . import targeted_edit_screened_math as prior


def _probe():
    """Fail closed on altered CPU subnormal handling without resetting it."""
    binary64 = scoring._probe()
    binary32 = v1._denormal_probe()
    words = torch.tensor([1, -2147483647], dtype=torch.int32)
    observed = (words.view(torch.float32) * torch.ones(2, dtype=torch.float32)).view(torch.int32)
    if observed.tolist() != [1, -2147483647]:
        raise prior.ProjectionError("signed binary32 denormals are not preserved")
    return {"binary64_bits": binary64, "binary32": binary32,
            "signed_binary32_bits": observed.tolist()}


def _dose(value, allowed):
    if type(value) is not float or not math.isfinite(value) or value not in allowed:
        raise prior.ProjectionError("dose is not a member of the fixed grid")
    return value


class _Prepared:
    def __init__(self, weight, calibration, input_factor, output_factor, gradient):
        self.initial_probe = _probe()
        self._inputs = tuple(prior._finite_tensor(tensor, dimensions) for tensor, dimensions in (
            (weight, 2), (calibration, 2), (input_factor, 1), (output_factor, 1), (gradient, 2)))
        self._input_hashes = tuple(v1._tensor_sha(item) for item in self._inputs)
        # The inherited helper fixes the SVD, seeded draw order and QR. Its
        # 0.001 updates are used only as a construction cross-check.
        self._base = prior.prepare_projected_edits(*self._inputs)
        self._factor_hashes = {name: prior._planned_sha(value)
                               for name, value in self._base["factors"].items()}
        self._update_hashes = {name: v1._tensor_sha(value)
                               for name, value in self._base["updates"].items()}
        self._attestation = v1._canonical(self._base["attestation"])
        self.final_probe = _probe()

    def _verify(self):
        _probe()
        if tuple(v1._tensor_sha(item) for item in self._inputs) != self._input_hashes:
            raise prior.ProjectionError("prepared input tensors changed")
        if ({name: prior._planned_sha(value) for name, value in self._base["factors"].items()}
                != self._factor_hashes):
            raise prior.ProjectionError("prepared factor tensors changed")
        if ({name: v1._tensor_sha(value) for name, value in self._base["updates"].items()}
                != self._update_hashes or v1._canonical(self._base["attestation"]) != self._attestation):
            raise prior.ProjectionError("prepared construction changed")

    def _construct(self, dose, names):
        self._verify()
        before = _probe()
        with torch.no_grad():
            w, x, h, g, gradient = self._inputs
            w64, x64, h64, g64, full64 = (item.to(torch.float64) for item in self._inputs)
            rank_one = torch.outer(h64, g64)
            residual_norm = float(torch.linalg.vector_norm(full64 - rank_one).item())
            planned_norm = dose * prior._norm(w64)
            if not math.isfinite(planned_norm) or planned_norm <= 0:
                raise prior.ProjectionError("the planned norm is invalid")
            factors = self._base["factors"]
            sources = {"projected": (factors["projected_input"], g64),
                       "unprojected": (h64, g64),
                       "projected_random": (factors["projected_random_input"], factors["random_output"]),
                       "nuisance": (factors["nuisance_input"], g64)}
            plans, updates, facts = {}, {}, {}
            for name in names:
                plan = (-plans["projected"] if name == "opposite" else
                        prior._planned_outer(*sources[name], planned_norm))
                intended = (-updates["projected"] if name == "opposite" else
                            plan.to(torch.float32).contiguous())
                if not bool(torch.isfinite(intended).all()):
                    raise prior.ProjectionError("an intended update is nonfinite")
                add_before = _probe()
                changed = w + intended
                add_after = _probe()
                if not bool(torch.isfinite(changed).all()):
                    raise prior.ProjectionError("an edited weight is nonfinite")
                actual = changed.to(torch.float64) - w64
                actual_norm = prior._norm(actual)
                if not 0.999 <= actual_norm / planned_norm <= 1.001:
                    raise prior.ProjectionError("a realized norm differs from its planned dose")
                plans[name], updates[name] = plan, intended
                facts[name] = {
                    "planned_float64_sha256": prior._planned_sha(plan),
                    "intended_delta_sha256": v1._tensor_sha(intended),
                    "realized_weight_sha256": v1._tensor_sha(changed),
                    "realized_delta_float64_sha256": prior._planned_sha(actual),
                    "realized_delta_norm": actual_norm,
                    "realized_to_planned": actual_norm / planned_norm,
                    "addition_denormal_probe_before": add_before,
                    "addition_denormal_probe_after": add_after,
                    "planned_prediction": prior._prediction(full64, rank_one, residual_norm, plan),
                    "intended_prediction": prior._prediction(full64, rank_one, residual_norm, intended.to(torch.float64)),
                    "realized_prediction": prior._prediction(full64, rank_one, residual_norm, actual),
                    "calibration_algebraic_max_abs": float(torch.max(torch.abs(x64 @ actual)).item()),
                }
                if dose == prior.DOSE and name in self._base["updates"]:
                    old = self._base["attestation"]["conditions"][name]
                    keys = ("planned_float64_sha256", "intended_delta_sha256",
                            "realized_weight_sha256", "realized_delta_float64_sha256")
                    if (not torch.equal(intended, self._base["updates"][name]) or
                            any(facts[name][key] != old[key] for key in keys)):
                        raise prior.ProjectionError("the inherited fixed-dose construction differs")
            if "projected" in facts:
                reference = facts["projected"]["realized_delta_norm"]
                for name in names:
                    ratio = facts[name]["realized_delta_norm"] / reference
                    if not 0.999 <= ratio <= 1.001:
                        raise prior.ProjectionError("a control does not match the projected realized norm")
                    facts[name]["realized_to_projected"] = ratio
        self._verify()
        return {"dose": dose, "planned_norm": planned_norm, "updates": updates,
                "attestations": facts, "denormal_probe_before": before,
                "denormal_probe_after": _probe()}

    def projected(self, dose):
        return self._construct(_dose(dose, protocol.PROJECTED_GRID), ("projected",))

    def unprojected(self, dose):
        return self._construct(_dose(dose, protocol.UNPROJECTED_GRID), ("unprojected",))

    def conditions(self, dose):
        return self._construct(_dose(dose, protocol.PROJECTED_GRID), prior.CONDITIONS)


def prepare(weight, calibration, input_factor, output_factor, gradient):
    return _Prepared(weight, calibration, input_factor, output_factor, gradient)
