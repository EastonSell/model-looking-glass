import math
import struct

import pytest
import torch

from looking_glass import year_label_intervention as inter


def _raw(year, label, condition):
    values = [0.0] * (15 * 768)
    values[-768 + 5] = year * (10 + condition)
    values[-768 + 9] = label * (5 + condition)
    values[-768 + 12] = 1.0
    return struct.pack(f'<{len(values)}f', *values)


def _family():
    return {condition: {label: {role: _raw(role == 'high', label == 'beta', i)
                               for role in ('low', 'high')}
                        for label in ('alpha', 'beta')}
            for i, condition in enumerate(('after', 'before', 'near'))}


def test_all_twelve_edges_set_one_common_dose_and_keep_both_directions():
    family = _family()
    result = inter.plan_family(family, ('alpha', 'beta'))
    assert result['dose'] == 2.5
    assert len(result['norms']) == 12
    assert min(result['norms'].values()) == 5.0
    for condition in family:
        for label in family[condition]:
            for role in ('low', 'high'):
                plans = result['plans'][condition][label][role]
                assert set(plans) == {'self', 'year_final', 'label_final', 'year_at_zero', 'rotated_year_final'}
                assert plans['self']['before'] == plans['self']['after']
                for action in ('year_final', 'label_final', 'year_at_zero', 'rotated_year_final'):
                    assert plans[action]['achieved'] == pytest.approx(2.5)
                before = struct.unpack('<768f', plans['year_final']['before'])
                year = struct.unpack('<768f', plans['year_final']['after'])
                lexical = struct.unpack('<768f', plans['label_final']['after'])
                assert year[5] - before[5] == (2.5 if role == 'low' else -2.5)
                assert lexical[9] - before[9] == (2.5 if label == 'alpha' else -2.5)
                assert year[9] == before[9] and lexical[5] == before[5]
                assert plans['year_at_zero']['position'] == 0
                rotated = struct.unpack('<768f', plans['rotated_year_final']['after'])
                assert rotated[6] - before[6] == (2.5 if role == 'low' else -2.5)


def test_minimum_label_edge_and_position_zero_mismatch_fail_closed():
    family = _family()
    family['near']['beta']['low'] = family['near']['alpha']['low']
    with pytest.raises(ValueError, match='dose'):
        inter.plan_family(family, ('alpha', 'beta'))
    family = _family()
    raw = family['after']['beta']['high']
    family['after']['beta']['high'] = struct.pack('<f', 1.0) + raw[4:]
    with pytest.raises(ValueError, match='position-zero'):
        inter.plan_family(family, ('alpha', 'beta'))


def test_small_label_edge_and_nonfinite_witness_stop_the_family():
    family = _family()
    raw = bytearray(family['after']['beta']['low'])
    struct.pack_into('<f', raw, len(raw) - 768 * 4 + 9 * 4, 2.0**-21)
    family['after']['beta']['low'] = bytes(raw)
    with pytest.raises(ValueError, match='dose'):
        inter.plan_family(family, ('alpha', 'beta'))
    family = _family()
    raw = bytearray(family['before']['alpha']['high'])
    struct.pack_into('<I', raw, len(raw) - 4, 0x7f800000)
    family['before']['alpha']['high'] = bytes(raw)
    with pytest.raises(ValueError):
        inter.plan_family(family, ('alpha', 'beta'))


def test_planned_label_delivery_lands_exactly_and_leaves_other_bits_untouched():
    family = _family()
    raw = family['before']['beta']['high']
    plan = inter.plan_family(family, ('alpha', 'beta'))['plans']['before']['beta']['high']['label_final']

    class Block(torch.nn.Module):
        def forward(self, value):
            return (value,)

    block = Block()
    data = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(1, 15, 768)
    witness = inter.BlockOutputWitness(block, sequence_length=15, expected_before=raw, plan=plan)
    with witness:
        actual = block(data)[0]
    witness.verify_complete()
    expected = raw[:-768*4] + plan['after']
    assert actual.numpy().tobytes() == expected == witness.after
    assert not block._forward_hooks
