"""No-model controls for the fixed shared-radius propagation geometry."""
import math

import pytest

from scripts import bert_taste_propagation_geometry_v1 as geometry


def fabricated(width=16):
    data = {}
    for template in ('after', 'with'):
        data[template] = {}
        for ingredient in ('sugar', 'lemon'):
            values = {}
            for index, site in enumerate(geometry.SITES):
                vector = [float(i + index) for i in range(width)]
                vector[0] += 4.0 if ingredient == 'sugar' else 0.0
                vector[1] += 1.0 if template == 'after' else 0.0
                values[site] = tuple(vector)
            values['late_mask'] = tuple(float(i) for i in range(width))
            data[template][ingredient] = values
    return data


def test_common_radius_and_all_signed_controls():
    planned = geometry.plan('toy', fabricated(), width=16)
    assert planned['radius'] == 1.0
    for site in geometry.SITES:
        assert planned['distances'][site] == {'lex_after': 4.0, 'lex_with': 4.0,
                                               'context_sugar': 1.0, 'context_lemon': 1.0}
        for template in ('after', 'with'):
            for ingredient in ('sugar', 'lemon'):
                actions = planned['actions'][template][ingredient][site]
                assert set(actions) == {'lex_plus', 'lex_minus', 'context_plus', 'context_minus',
                                        'orth1_plus', 'orth1_minus', 'orth2_plus', 'orth2_minus'}
                for name, action in actions.items():
                    assert math.isclose(action['delivered_norm'], 1., abs_tol=1e-6)
                    if name.startswith('lex_'):
                        assert abs(action['lexical_cosine']) >= .99
                    if name.startswith('context_'):
                        assert abs(action['context_cosine']) >= .99
                    if name.startswith('orth'):
                        assert abs(action['lexical_cosine']) <= .01
                assert all(abs(x) <= .01 for x in planned['mutual_cosines'][template][ingredient][site].values())


def test_zero_distance_and_degenerate_orthogonal_fail_without_redraw(monkeypatch):
    data = fabricated()
    data['after']['lemon']['late_the'] = data['after']['sugar']['late_the']
    with pytest.raises(ValueError, match='radius|distance'):
        geometry.plan('toy', data, width=16)
    monkeypatch.setattr(geometry, 'signs', lambda *_args: (1.,) * 16)
    with pytest.raises(ValueError, match='orthogonal'):
        geometry.plan('toy', fabricated(), width=16)


def test_rounded_displacement_can_fail_instead_of_being_clamped():
    with pytest.raises(ValueError, match='delivered norm'):
        geometry.base.source((1e8, 0.), (1., 0.), 1, 5., (8., 0.))
