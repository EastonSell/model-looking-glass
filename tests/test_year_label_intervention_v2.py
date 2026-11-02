import math
import struct

import pytest
import torch

from looking_glass import year_label_intervention_v2 as v2


def _raw(first, second=0.0):
    return struct.pack('<768f', first, second, *([0.0] * 766))


def _delta(first, second=0.0):
    return (float(first), float(second), *([0.0] * 766))


def test_exact_certificate_accepts_both_inclusive_limits():
    before = _raw(0.0)
    q = 5000.0
    for after in (_raw(4999.0), _raw(5001.0)):
        result = v2.delivery_certificate(before, after, _delta(q), q)
        assert result['D2'] == {'numerator': '25000000', 'denominator': '1'}
        assert result['E2'] == {'numerator': '1', 'denominator': '1'}
        assert result['vector_error'] == 1.0


def test_exact_certificate_rejects_one_float32_step_beyond_bound():
    with pytest.raises(ValueError, match='fidelity'):
        v2.delivery_certificate(_raw(0.0), _raw(5001.00048828125), _delta(5000.0), 5000.0)


def test_same_norm_wrong_direction_fails_vector_check():
    with pytest.raises(ValueError, match='fidelity'):
        v2.delivery_certificate(_raw(0.0), _raw(0.0, 1.0), _delta(1.0), 1.0)


def test_ideal_direction_and_nonfinite_or_small_dose_are_rejected():
    with pytest.raises(ValueError, match='fidelity'):
        v2.delivery_certificate(_raw(0.0), _raw(1.0), _delta(1.000001), 1.0)
    for dose in (0.0, 2.0**-20, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            v2.delivery_certificate(_raw(0.0), _raw(1.0), _delta(1.0), dose)
    with pytest.raises(ValueError):
        v2.delivery_certificate(_raw(float('inf')), _raw(1.0), _delta(1.0), 1.0)


def test_binary64_flush_to_zero_is_detected():
    if not torch.set_flush_denormal(True):
        pytest.skip('host cannot change denormal mode')
    try:
        with pytest.raises(RuntimeError, match='binary64 subnormal'):
            v2.delivery_certificate(_raw(0.0), _raw(1.0), _delta(1.0), 1.0)
    finally:
        torch.set_flush_denormal(False)


def test_subnormal_error_is_not_reported_as_exact_zero():
    tiny = struct.unpack('<d', struct.pack('<Q', 1))[0]
    result = v2.delivery_certificate(_raw(0.0), _raw(1.0), _delta(1.0, tiny), 1.0)
    assert result['E2']['numerator'] == '1'
    assert result['vector_error'] is None
    assert result['vector_error_decimal'] != '0'


def test_hook_recomputes_fidelity_from_actual_bits_and_preserves_other_positions():
    class Block(torch.nn.Module):
        def forward(self, value):
            return (value,)

    block = Block()
    tensor = torch.zeros((1, 2, 768), dtype=torch.float32)
    original = tensor.numpy().tobytes()
    displacement = _delta(1.0)
    before, after = _raw(0.0), _raw(1.0)
    certificate = v2.delivery_certificate(before, after, displacement, 1.0)
    plan = {'position': 1, 'before': before, 'after': after, 'requested': 1.0,
            'displacement': displacement, 'certificate': certificate, 'changed_bits': 1}
    with v2.BlockOutputWitness(block, sequence_length=2, expected_before=original, plan=plan) as w:
        result = block(tensor)[0]
    w.verify_complete()
    assert result[0, 1, 0] == 1 and result[0, 0, 0] == 0
    assert w.before[:3072] == w.after[:3072] and w.observed['certificate'] == certificate
    assert not block._forward_hooks


def test_same_forward_certificate_rejects_tampered_expected_metrics():
    class Block(torch.nn.Module):
        def forward(self, value):
            return (value,)
    block = Block()
    tensor = torch.zeros((1, 2, 768), dtype=torch.float32)
    before, after = _raw(0.0), _raw(1.0)
    cert = v2.delivery_certificate(before, after, _delta(1.0), 1.0)
    cert['E2'] = {'numerator': '1', 'denominator': '1'}
    plan = {'position': 1, 'before': before, 'after': after, 'requested': 1.0,
            'displacement': _delta(1.0), 'certificate': cert, 'changed_bits': 1}
    with v2.BlockOutputWitness(block, sequence_length=2,
                               expected_before=tensor.numpy().tobytes(), plan=plan) as w:
        block(tensor)
    with pytest.raises(ValueError, match='certificate'):
        w.verify_complete()


def test_self_is_bit_identity_not_numerical_equality():
    class Block(torch.nn.Module):
        def forward(self, value):
            return (value,)
    block = Block()
    tensor = torch.zeros((1, 2, 768), dtype=torch.float32)
    raw = tensor.numpy().tobytes()
    before = _raw(0.0)
    minus_zero = _raw(-0.0)
    plan = {'position': 1, 'before': before, 'after': minus_zero, 'requested': 0.0,
            'displacement': None, 'certificate': None, 'changed_bits': 1}
    with v2.BlockOutputWitness(block, sequence_length=2, expected_before=raw, plan=plan) as w:
        block(tensor)
    with pytest.raises(ValueError, match='self'):
        w.verify_complete()


def test_all_crossed_donor_actions_get_the_same_certified_dose():
    labels = ('red', 'blue')
    noops = {}
    for condition in ('after', 'before', 'near'):
        noops[condition] = {}
        for li, label in enumerate(labels):
            noops[condition][label] = {}
            for yi, role in enumerate(('low', 'high')):
                noops[condition][label][role] = _raw(0.0) + _raw(10.0*yi, 5.0*li)
    family = v2.plan_family(noops, labels)
    assert family['dose'] == 2.5 and len(family['norms']) == 12
    for condition in noops:
        for label in labels:
            for role in ('low', 'high'):
                plans = family['plans'][condition][label][role]
                assert plans['self']['before'] == plans['self']['after']
                for name in ('year_final', 'label_final', 'year_at_zero', 'rotated_year_final'):
                    plan = plans[name]
                    assert plan['requested'] == 2.5
                    assert v2.delivery_certificate(plan['before'], plan['after'],
                                                   plan['displacement'], 2.5) == plan['certificate']
                    assert plan['certificate']['vector_error'] == 0.0
