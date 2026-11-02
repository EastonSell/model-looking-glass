"""Synthetic checks of the common radius and signed control geometry."""
import importlib.util
import math
from pathlib import Path
import struct

import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'scripts/bert_taste_template_geometry_v1.py'
spec = importlib.util.spec_from_file_location('taste_geometry_v1', path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def captures():
    return {'after': {'sugar': {'ingredient': (4., 0., 0.), 'the': (0., 2., 0.)},
                      'lemon': {'ingredient': (0., 0., 0.), 'the': (0., 0., 0.)}},
            'with': {'sugar': {'ingredient': (3., 0., 0.), 'the': (0., 1., 0.)},
                     'lemon': {'ingredient': (0., 0., 0.), 'the': (0., 0., 0.)}}}


def test_one_radius_across_both_templates_positions_and_signs():
    plan = geometry.plan('fabricated', captures(), width=3)
    assert plan['radius'] == 1.
    assert plan['distances'] == {'after': {'ingredient': 4., 'the': 2.},
                                 'with': {'ingredient': 3., 'the': 1.}}
    assert plan['fractions'] == {'after': {'ingredient': .25, 'the': .5},
                                  'with': {'ingredient': 1/3, 'the': 1.}}
    for template in ('after', 'with'):
        for ingredient in ('sugar', 'lemon'):
            row = plan['actions'][template][ingredient]
            for action in row.values():
                assert action['requested_norm'] == 1.
                assert math.isclose(action['delivered_norm'], 1., abs_tol=1e-6)
            assert row['ingredient_toward']['cosine_with_donor_direction'] >= .99
            assert row['ingredient_away']['cosine_with_donor_direction'] <= -.99
            assert abs(row['orth_plus']['cosine_with_donor_direction']) <= .01
            assert abs(row['orth_minus']['cosine_with_donor_direction']) <= .01
    lemon = plan['actions']['after']['lemon']
    assert struct.unpack('<3f', lemon['ingredient_toward']['raw']) == (1., 0., 0.)
    assert struct.unpack('<3f', lemon['ingredient_away']['raw']) == (-1., 0., 0.)
    assert lemon['ingredient_toward']['recipient_norm_status'] == 'zero'
    assert lemon['ingredient_toward']['dose_to_recipient_norm'] is None


def test_orthogonal_prf_is_deterministic_and_truly_projected():
    axis = (2., 3., 4.)
    first = geometry.orthogonal('fabricated', 'after', axis)
    assert first == geometry.orthogonal('fabricated', 'after', axis)
    assert first != geometry.orthogonal('fabricated', 'with', axis)
    assert math.isclose(geometry.norm(first), 1., abs_tol=1e-15)
    assert abs(math.fsum(a*b for a, b in zip(first, axis))) < 1e-14


def test_invalid_or_rounded_away_doses_fail_without_retry():
    changed = captures()
    changed['with']['sugar']['the'] = (0., 0., 0.)
    with pytest.raises(ValueError, match='radius'):
        geometry.plan('fabricated', changed, width=3)
    with pytest.raises(ValueError, match='degenerate'):
        geometry.orthogonal('fabricated', 'after', (1.,))
    with pytest.raises(ValueError, match='delivered norm'):
        geometry.source((1e8, 0., 0.), (1., 0., 0.), 1, 5., (8., 0., 0.))
    with pytest.raises(ValueError, match='finite'):
        geometry.values((float('nan'), 0.))


def test_signed_subnormal_source_bits_are_independent_of_flush_mode():
    for flush in (False, True):
        assert torch.set_flush_denormal(flush)
        try:
            for tiny, expected in ((math.ldexp(1., -149), 1), (-math.ldexp(1., -149), 0x80000001)):
                action = geometry.source((tiny, 0.), (0., 1.), 1, 1., (0., 2.))
                assert struct.unpack('<I', action['raw'][:4])[0] == expected
                assert geometry.f32_bits(tiny) == expected
                assert geometry.f32_value(expected) == tiny
        finally:
            assert torch.set_flush_denormal(False)


def test_integer_rounding_matches_independent_fraction_reference():
    from scripts import bert_taste_matched_independent as reference
    import random
    rng = random.Random(7251)
    for _ in range(5000):
        bits = rng.getrandbits(32)
        if (bits >> 23) & 255 == 255:
            continue
        exact = reference.f32(bits)
        assert geometry.f32_bits(exact) == bits
        assert geometry.f32_bits(math.nextafter(exact, math.inf)) == reference.f32_round(math.nextafter(exact, math.inf))
