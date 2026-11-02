"""Plan all matched arm landings from saved finite encoder-row bytes.

The result is model-free and must be durably journaled before any planned
action. A separate owner validates row provenance, numeric mode and custody.
Nominally valid controls with an unavailable reference remain descriptive.
"""

import math
import struct

from .bert_matched_numeric import f32_bits_to_f64, f64_to_f32_bits
from .flan_multitoken_orthogonal_v2 import (
    _cosine, _norm, construct_orthogonal, gaussian_source, land_orthogonal, orthogonal_seed,
)
from .flan_multitoken_schedule_v2 import RECEIVERS, OPPOSITE, SAME


_DONOR_ARMS = ("opposite_adjective", "same_adjective", "opposite_subject", "opposite_instruction")
_ARM_SITE = {"opposite_adjective": "adjective", "same_adjective": "adjective",
             "opposite_subject": "subject", "opposite_instruction": "instruction"}
_SUBJECT_INDEX = {"story": 0, "concert": 1, "painting": 2}


def _decode(raw):
    if type(raw) is not bytes or not 4 <= len(raw) <= 512 * 4 or len(raw) % 4:
        raise ValueError("expected bounded float32 row bytes")
    words = tuple(word for (word,) in struct.iter_unpack("<I", raw))
    if any(word & 0x7f800000 == 0x7f800000 for word in words):
        raise ValueError("nonfinite encoder row")
    return tuple(f32_bits_to_f64(word) for word in words)


def _donor_landing(base, donor, radius):
    own, other = _decode(base), _decode(donor)
    if len(own) != len(other) or type(radius) is not float or not math.isfinite(radius) or radius <= 0:
        raise ValueError("invalid donor landing")
    difference = tuple(b - a for a, b in zip(own, other))
    distance = _norm(difference)
    if distance == 0 or radius > distance:
        raise ValueError("invalid donor interpolation")
    fraction = radius / distance
    words = tuple(f64_to_f32_bits(float(a + fraction * d)) for a, d in zip(own, difference))
    value = struct.pack("<" + str(len(words)) + "I", *words)
    tolerance = min(max(1e-5, 1e-5 * radius), 0.01 * radius)
    if any(word & 0x7f800000 == 0x7f800000 for word in words):
        return {"status": "incomplete", "failure": "nonfinite rounded landing", "value": value,
                "radius": radius, "achieved_norm": None, "intended_cosine": None,
                "tolerance": tolerance, "fraction": fraction}
    landed = tuple(f32_bits_to_f64(word) for word in words)
    displacement = tuple(a - b for a, b in zip(landed, own))
    achieved = _norm(displacement)
    cosine = _cosine(displacement, difference, achieved, distance)
    failure = None
    if achieved == 0:
        failure = "zero achieved displacement"
    elif abs(achieved - radius) > tolerance:
        failure = "landing norm outside tolerance"
    elif cosine is None or cosine < 0.99:
        failure = "landing cosine below threshold"
    return {"status": "matchable" if failure is None else "unavailable", "failure": failure,
            "value": value, "radius": radius, "achieved_norm": achieved,
            "intended_cosine": cosine, "tolerance": tolerance, "fraction": fraction}


def plan_group_geometry(site_rows, *, unit, subject, instruction):
    """Return all 16 distances and at most 40 per-dose proposed landings.

    Keys of ``landings`` are ``(receiver, dose, arm)``. Orthogonal source and
    residual bytes are returned for independent reconstruction. The caller
    must persist even an incomplete landing's exact returned bits.
    """
    if (type(site_rows) is not dict or set(site_rows) != set(RECEIVERS)
            or type(subject) is not str or subject not in _SUBJECT_INDEX
            or type(unit) is not int or not 0 <= unit < 4
            or (subject == "painting" and unit not in (0, 2))
            or type(instruction) is not int or instruction not in (0, 1)):
        raise ValueError("invalid group identity or rows")
    vectors, width = {}, None
    for receiver in RECEIVERS:
        rows = site_rows[receiver]
        if type(rows) is not dict or set(rows) != {"adjective", "subject", "instruction"}:
            raise ValueError("expected all three sites")
        vectors[receiver] = {}
        for site in ("adjective", "subject", "instruction"):
            values = _decode(rows[site])
            if width is not None and len(values) != width:
                raise ValueError("encoder row widths differ")
            width = len(values)
            vectors[receiver][site] = values
    distances = {}
    for receiver in RECEIVERS:
        for arm in _DONOR_ARMS:
            site = _ARM_SITE[arm]
            donor = SAME[receiver] if arm == "same_adjective" else OPPOSITE[receiver]
            delta = tuple(b - a for a, b in zip(vectors[receiver][site], vectors[donor][site]))
            distances[(receiver, arm)] = _norm(delta)
    q = min(distances.values())
    result = {"status": "dose-ineligible" if q / 2 <= 1e-6 else "planned",
              "q": q, "distances": distances, "orthogonal": {}, "landings": {}}
    if result["status"] == "dose-ineligible":
        return result
    for j, receiver in enumerate(RECEIVERS):
        donor = OPPOSITE[receiver]
        seed = orthogonal_seed(unit, _SUBJECT_INDEX[subject], instruction, j)
        source = gaussian_source(seed, width)
        constructed = construct_orthogonal(site_rows[receiver]["adjective"],
                                           site_rows[donor]["adjective"], source)
        direction = constructed["direction"]
        result["orthogonal"][receiver] = {
            "seed": seed, "status": constructed["status"], "failure": constructed["failure"],
            "source": source, "residual": None if direction is None else direction.residual,
            "coefficient": None if direction is None else direction.coefficient,
            "residual_norm": None if direction is None else direction.residual_norm,
            "donor_norm": None if direction is None else direction.donor_norm,
        }
        for dose, radius in (("low", q / 2), ("high", q)):
            for arm in _DONOR_ARMS:
                site = _ARM_SITE[arm]
                target = SAME[receiver] if arm == "same_adjective" else donor
                landing = _donor_landing(site_rows[receiver][site], site_rows[target][site], radius)
                landing["site"], landing["donor"] = site, target
                result["landings"][(receiver, dose, arm)] = landing
                if landing["status"] == "incomplete":
                    result["status"] = "incomplete"
                    return result
            if direction is None:
                orthogonal = {"status": "unavailable", "failure": constructed["failure"],
                              "value": None, "radius": radius, "achieved_norm": None,
                              "intended_cosine": None, "donor_cosine": None}
            else:
                orthogonal = land_orthogonal(direction, radius)
            orthogonal["site"], orthogonal["donor"] = "adjective", None
            result["landings"][(receiver, dose, "orthogonal")] = orthogonal
            if orthogonal["status"] == "incomplete":
                result["status"] = "incomplete"
                return result
            reference = result["landings"][(receiver, dose, "opposite_adjective")]
            reference_valid = (reference["status"] == "matchable"
                               and reference["achieved_norm"] is not None
                               and reference["achieved_norm"] > 0)
            reference["relative_status"] = "reference" if reference_valid else "reference-unavailable"
            for arm in (*_DONOR_ARMS[1:], "orthogonal"):
                control = result["landings"][(receiver, dose, arm)]
                if not reference_valid:
                    control["relative_status"] = "reference-unavailable"
                elif control["status"] != "matchable":
                    control["relative_status"] = "not-applicable"
                elif abs(control["achieved_norm"] - reference["achieved_norm"]) > 0.01 * reference["achieved_norm"]:
                    control["status"], control["failure"] = "unavailable", "control norm differs from reference"
                    control["relative_status"] = "failed"
                else:
                    control["relative_status"] = "matched"
    return result
