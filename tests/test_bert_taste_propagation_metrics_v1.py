"""Hand-computed, no-model checks of propagation and output metrics."""
import math

import pytest

from scripts import bert_taste_propagation_metrics_v1 as metrics


def test_odd_even_and_projection_with_zero_denominators():
    odd, even = metrics.odd_even((5., 2.), (1., 4.), (2., 3.))
    assert odd == (2., -1.)
    assert even == (1., 0.)
    aligned = metrics.projection((2., -1.), (4., -2.))
    assert aligned['coefficient'] == .5
    assert aligned['residual_norm'] == 0.
    assert math.isclose(aligned['cosine'], 1.)
    empty = metrics.projection((1., 0.), (0., 0.))
    assert empty['coefficient'] is None and empty['cosine'] is None
    zero = metrics.projection((0., 0.), (1., 0.))
    assert zero['coefficient'] == 0. and zero['cosine'] is None


def test_centering_and_total_variation_are_shift_invariant():
    assert metrics.center((11., 13., 15.)) == (-2., 0., 2.)
    assert metrics.center((1., 3., 5.)) == (-2., 0., 2.)
    assert metrics.total_variation((0., 0.), (0., 0.)) == 0.
    distance = metrics.total_variation((0., 0.), (math.log(3.), 0.))
    assert math.isclose(distance, .25, abs_tol=1e-15)
    assert math.isclose(distance, metrics.total_variation((9., 9.), (10. + math.log(3.), 10.)), abs_tol=1e-15)


def test_null_aggregation_never_silently_drops_cases():
    assert metrics.complete_mean([1., 2.]) == {'value': 1.5, 'missing_count': 0, 'reason': None}
    assert metrics.complete_mean([1., None]) == {'value': None, 'missing_count': 1,
                                                  'reason': 'required value unavailable'}
    with pytest.raises(ValueError, match='nonempty'):
        metrics.complete_mean([])
    with pytest.raises(ValueError, match='finite'):
        metrics.complete_mean([None, float('nan')])
