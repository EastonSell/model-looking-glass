"""Fabricated numeric checks for the proposed orthogonal control."""

import math
import struct

import pytest

from looking_glass.flan_multitoken_orthogonal_v2 import (
    construct_orthogonal, gaussian_source, land_orthogonal, orthogonal_seed, plan_orthogonal,
)


def f32(*values):
    return struct.pack("<" + "f" * len(values), *values)


def f64(*values):
    return struct.pack("<" + "d" * len(values), *values)


@pytest.mark.parametrize("radius", [1.0, 2.0])
def test_landing_orthogonal_to_a_known_donor_direction(radius):
    result = plan_orthogonal(f32(0.0, 0.0), f32(3.0, 4.0), f64(-4.0, 3.0), radius)
    assert result["status"] == "matchable"
    assert result["value"] == f32(-0.8 * radius, 0.6 * radius)
    assert result["achieved_norm"] == pytest.approx(radius, rel=1e-6)
    assert abs(result["donor_cosine"]) < 1e-4
    assert result["intended_cosine"] >= 0.99


def test_degenerate_residual_does_not_trigger_resampling():
    result = plan_orthogonal(f32(0.0, 0.0), f32(3.0, 4.0), f64(3.0, 4.0), 1.0)
    assert result["status"] == "unavailable"
    assert result["failure"] == "zero orthogonal residual"
    assert result["value"] is None


def test_finite_zero_float32_landing_stays_unavailable():
    result = plan_orthogonal(f32(1e8, 0.0), f32(1e8, 1.0), f64(1.0, 0.0), 1.0)
    assert result["status"] == "unavailable"
    assert result["failure"] == "zero achieved displacement"
    assert result["value"] == f32(1e8, 0.0)


def test_nonfinite_rounded_landing_retains_bits_and_marks_incomplete():
    result = plan_orthogonal(f32(3e38, 0.0), f32(3e38, 3e38), f64(1.0, 0.0), 1e38)
    assert result["status"] == "incomplete"
    assert result["failure"] == "nonfinite rounded landing"
    assert math.isinf(struct.unpack("<2f", result["value"])[0])
    assert result["achieved_norm"] is None


def test_source_is_reproducible_and_preserves_all_binary64_draws():
    source = gaussian_source(20261029, 512)
    assert len(source) == 512 * 8
    assert source == gaussian_source(20261029, 512)
    assert source != gaussian_source(20261030, 512)
    assert all(math.isfinite(v[0]) for v in struct.iter_unpack("<d", source))


def test_seed_slots_are_unique_across_all_three_subject_indices():
    seeds = {orthogonal_seed(u, s, i, j)
             for u in range(4) for s in range(3) for i in range(2) for j in range(4)}
    assert len(seeds) == 96
    assert orthogonal_seed(0, 0, 0, 0) == 20261029
    assert orthogonal_seed(2, 2, 1, 3) == 20261029 + 24 * 2 + 8 * 2 + 4 + 3


def test_one_saved_direction_is_reused_for_both_doses():
    base, donor, source = f32(0.0, 0.0), f32(3.0, 4.0), f64(-4.0, 3.0)
    constructed = construct_orthogonal(base, donor, source)
    assert constructed["status"] == "ready"
    direction = constructed["direction"]
    assert direction.source == source
    assert direction.residual == f64(-4.0, 3.0)
    assert land_orthogonal(direction, 1.0) == plan_orthogonal(base, donor, source, 1.0)
    assert land_orthogonal(direction, 2.0) == plan_orthogonal(base, donor, source, 2.0)


@pytest.mark.parametrize("base,donor,source,radius", [
    (f32(0.0), f32(1.0), b"bad", 1.0),
    (f32(0.0), f32(1.0), f64(float("nan")), 1.0),
    (f32(0.0), f32(float("inf")), f64(1.0), 1.0),
    (f32(0.0), f32(1.0), f64(1.0), 0.0),
])
def test_invalid_inputs_do_not_create_a_landing(base, donor, source, radius):
    with pytest.raises(ValueError):
        plan_orthogonal(base, donor, source, radius)
