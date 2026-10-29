"""Matched-dose planning checks on fabricated encoder rows only."""

import struct

import pytest

import looking_glass.flan_multitoken_geometry_v2 as geometry


def vector(x, y):
    return struct.pack("<2f", x, y)


def fabricated_sites():
    return {side: {site: vector(*xy) for site in ("adjective", "subject", "instruction")}
            for side, xy in {"P1": (0, 0), "N1": (2, 0),
                             "P2": (0, 2), "N2": (2, 2)}.items()}


def test_all_doses_and_five_arms_are_precomputed_per_receiver():
    result = geometry.plan_group_geometry(fabricated_sites(), unit=2, subject="painting", instruction=1)
    assert result["status"] == "planned" and result["q"] == 2.0
    assert len(result["distances"]) == 16
    assert len(result["landings"]) == 40
    assert set(result["orthogonal"]) == {"P1", "N1", "P2", "N2"}
    assert len({row["seed"] for row in result["orthogonal"].values()}) == 4
    for value in result["orthogonal"].values():
        assert len(value["source"]) == 16 and len(value["residual"]) == 16
    for (side, dose, arm), landing in result["landings"].items():
        assert landing["status"] == "matchable"
        assert landing["radius"] == (1.0 if dose == "low" else 2.0)
        assert landing["relative_status"] == ("reference" if arm == "opposite_adjective" else "matched")
        assert len(landing["value"]) == 8


def test_zero_radius_does_not_draw_or_project_a_direction(monkeypatch):
    sites = fabricated_sites()
    sites["P2"]["adjective"] = sites["P1"]["adjective"]
    monkeypatch.setattr(geometry, "gaussian_source", lambda *args: pytest.fail("must not draw"))
    result = geometry.plan_group_geometry(sites, unit=0, subject="story", instruction=0)
    assert result["status"] == "dose-ineligible" and result["q"] == 0
    assert result["landings"] == {} and result["orthogonal"] == {}


def test_low_dose_failure_does_not_suppress_high_or_other_actions(monkeypatch):
    original = geometry.land_orthogonal
    def one_failure(direction, radius):
        result = original(direction, radius)
        if radius == 1.0:
            return dict(result, status="unavailable", failure="fabricated low miss")
        return result
    monkeypatch.setattr(geometry, "land_orthogonal", one_failure)
    result = geometry.plan_group_geometry(fabricated_sites(), unit=0, subject="story", instruction=0)
    assert result["landings"][("P1", "low", "orthogonal")]["status"] == "unavailable"
    assert result["landings"][("P1", "high", "orthogonal")]["status"] == "matchable"
    assert result["landings"][("P1", "low", "same_adjective")]["status"] == "matchable"


def test_unavailable_reference_leaves_nominal_controls_descriptive(monkeypatch):
    original = geometry._donor_landing
    def one_failure(base, donor, radius):
        result = original(base, donor, radius)
        if base == vector(0, 0) and donor == vector(2, 0) and radius == 1.0:
            return dict(result, status="unavailable", failure="fabricated reference miss")
        return result
    monkeypatch.setattr(geometry, "_donor_landing", one_failure)
    result = geometry.plan_group_geometry(fabricated_sites(), unit=0, subject="story", instruction=0)
    control = result["landings"][("P1", "low", "same_adjective")]
    assert control["status"] == "matchable"
    assert control["relative_status"] == "reference-unavailable"
    assert result["landings"][("P1", "high", "same_adjective")]["relative_status"] == "matched"


def test_nonfinite_landing_bits_remain_available_for_incomplete_evidence(monkeypatch):
    raw = struct.pack("<2I", 0x7f800000, 0)
    monkeypatch.setattr(geometry, "land_orthogonal", lambda direction, radius: {
        "status": "incomplete", "failure": "nonfinite rounded landing", "value": raw,
        "radius": radius, "achieved_norm": None, "intended_cosine": None,
        "donor_cosine": None, "tolerance": 1e-5,
    })
    result = geometry.plan_group_geometry(fabricated_sites(), unit=0, subject="story", instruction=0)
    assert result["status"] == "incomplete"
    assert result["landings"][("P1", "low", "orthogonal")]["value"] == raw


def test_nonfinite_encoder_values_reject_before_planning():
    sites = fabricated_sites()
    sites["P1"]["subject"] = struct.pack("<2I", 0x7fc00000, 0)
    with pytest.raises(ValueError, match="nonfinite"):
        geometry.plan_group_geometry(sites, unit=0, subject="story", instruction=0)
