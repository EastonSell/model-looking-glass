import math
import struct

import pytest
import torch

from looking_glass import year_transfer_intervention as inter


def _raw(length, final_scale):
    values = [0.25] * (length * 768)
    for i in range(768):
        values[(length-1)*768+i] = final_scale * (i + 1) / 1024
    return struct.pack(f'<{len(values)}f', *values)


def _family():
    return {name: {'low': _raw(length, 0.0), 'high': _raw(length, scale)}
            for name, length, scale in (('forward', 12, 1.0), ('reverse', 15, 2.0), ('neutral', 13, 3.0))}


def test_family_dose_is_common_and_both_directions_are_exactly_planned():
    family = _family()
    result = inter.plan_family(family)
    expected = 0.5 * math.sqrt(math.fsum(((i+1)/1024)**2 for i in range(768)))
    assert result['dose'] == expected
    for condition, pair in result['plans'].items():
        for role, actions in pair.items():
            assert set(actions) == {'self', 'year_final', 'year_at_zero', 'rotated_final'}
            assert actions['self']['before'] == actions['self']['after']
            assert actions['self']['achieved'] == 0.0
            for name in ('year_final', 'year_at_zero', 'rotated_final'):
                assert abs(actions[name]['achieved'] - expected) <= expected * 1e-5
            assert actions['year_final']['position'] == actions['self']['position']
            assert actions['year_at_zero']['position'] == 0
            assert actions['rotated_final']['after'] != actions['year_final']['after']


def test_wrong_position_zero_or_zero_dose_rejects_entire_family():
    family = _family()
    family['reverse']['high'] = struct.pack('<f', 0.5) + family['reverse']['high'][4:]
    with pytest.raises(ValueError):
        inter.plan_family(family)
    family = _family()
    family['neutral']['high'] = family['neutral']['low']
    with pytest.raises(ValueError):
        inter.plan_family(family)


def test_nonfinite_bits_and_oversize_witness_rejected():
    family = _family()
    family['forward']['low'] = struct.pack('<I', 0x7f800000) + family['forward']['low'][4:]
    with pytest.raises(ValueError):
        inter.plan_family(family)
    with pytest.raises(ValueError):
        inter.plan_family({'forward': {'low': b'\0' * (33 * 768 * 4), 'high': b'\0' * (33 * 768 * 4)},
                           'reverse': _family()['reverse'], 'neutral': _family()['neutral']})


def test_block_witness_changes_only_exact_planned_bits_and_removes_hook():
    family = _family()
    plan = inter.plan_family(family)['plans']['forward']['low']['year_final']
    before = family['forward']['low']
    class Block(torch.nn.Module):
        def forward(self, x):
            return (x,)
    block = Block()
    data = torch.frombuffer(bytearray(before), dtype=torch.float32).reshape(1, 12, 768)
    witness = inter.BlockOutputWitness(block, sequence_length=12, expected_before=before, plan=plan)
    with witness:
        out = block(data)[0]
    witness.verify_complete()
    expected = bytearray(before)
    start = plan['position'] * 768 * 4
    expected[start:start+768*4] = plan['after']
    assert witness.before == before and witness.after == bytes(expected)
    assert witness.observed['achieved'] == plan['achieved']
    assert witness.observed['changed_bits'] == plan['changed_bits']
    assert out.numpy().tobytes() == bytes(expected)
    assert not block._forward_hooks


def test_self_and_noop_are_exact_and_stale_capture_fails_closed():
    family = _family()
    raw = family['neutral']['low']
    plan = inter.plan_family(family)['plans']['neutral']['low']['self']
    class Block(torch.nn.Module):
        def forward(self, x):
            return (x,)
    block = Block()
    data = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(1, 13, 768)
    with inter.BlockOutputWitness(block, sequence_length=13, expected_before=raw, plan=plan) as witness:
        assert block(data)[0].numpy().tobytes() == raw
    witness.verify_complete()
    assert witness.before == witness.after == raw
    assert witness.observed == {'achieved': 0.0, 'changed_bits': 0}
    bad = b'\0\0\0\0' + raw[4:]
    witness = inter.BlockOutputWitness(block, sequence_length=13, expected_before=bad, plan=plan)
    with pytest.raises(ValueError):
        with witness:
            block(data)
    assert witness.before == raw
    assert not block._forward_hooks


def test_real_eager_gpt2_block_self_replacement_is_identical():
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(203)
        block = GPT2Block(GPT2Config(n_embd=768, n_head=12, n_layer=1, n_positions=16,
                                     attn_implementation='eager'), layer_idx=0).eval()
        values = torch.zeros((1, 3, 768), dtype=torch.float32)
        with torch.inference_mode(), inter.BlockOutputWitness(block, sequence_length=3) as passive:
            clean = block(values)[0]
        passive.verify_complete()
        raw = passive.before
        plan = {'position': 2, 'before': raw[-inter.ROW_BYTES:], 'after': raw[-inter.ROW_BYTES:],
                'requested': 0.0, 'achieved': 0.0, 'changed_bits': 0}
        with torch.inference_mode(), inter.BlockOutputWitness(block, sequence_length=3,
                                                              expected_before=raw, plan=plan) as self_hook:
            patched = block(values)[0]
        self_hook.verify_complete()
        assert clean.numpy().tobytes() == patched.numpy().tobytes()
        assert not block._forward_hooks
