import csv
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from looking_glass import agreement_science_v1 as science
from tools import agreement_direction_context_posthoc as posthoc


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results/distilgpt2-agreement-direction-study-v1/posthoc-context'


def test_binary32_bits_include_subnormals_and_signed_zero():
    bits = (0, 0x80000000, 1, 0x807fffff, 0x00800000, 0x3f800000, 0xc0200000)
    raw = struct.pack('<7I', *bits)
    actual, subnormal_count = posthoc.decode_f32(raw, len(bits))
    expected = science.decode_float32(raw, expected_count=len(bits))
    assert subnormal_count == 2
    assert all(struct.pack('<d', float(a)) == struct.pack('<d', b) for a, b in zip(actual, expected))
    for bad in (struct.pack('<I', 0x7f800000), struct.pack('<I', 0x7fc00001), b'bad'):
        with pytest.raises(ValueError):
            posthoc.decode_f32(bad, 1)


def test_dose_normalization_and_even_component_are_not_assumed_linear():
    clean, plus, minus = np.array([1.0, 3.0]), np.array([5.0, 7.0]), np.array([0.0, 1.0])
    odd, even = posthoc.odd_and_even(clean, plus, minus, 2.0)
    np.testing.assert_array_equal(odd, [1.25, 1.5])
    np.testing.assert_array_equal(even, [1.5, 1.0])
    with pytest.raises(ValueError):
        posthoc.odd_and_even(clean, plus, minus, 0.0)


def test_centered_vocabulary_cosine_ignores_only_a_common_logit_offset():
    a = posthoc.centered_unit([1.0, 2.0, 5.0])
    b = posthoc.centered_unit([11.0, 12.0, 15.0])
    assert float(np.dot(a, b)) == pytest.approx(1.0, abs=1e-15)
    assert float(np.dot(a, posthoc.centered_unit([5.0, 2.0, 1.0]))) < 0
    with pytest.raises(ValueError):
        posthoc.centered_unit([2.0, 2.0, 2.0])


def test_probability_l1_share_and_tv_use_entire_vocabulary():
    clean = np.zeros(548, dtype=np.float64)
    arm = clean.copy()
    arm[318] = math.log(2)
    arm[9] = math.log(3)
    result = posthoc.distribution_change(clean, arm)
    # The baseline is uniform; the new normalizer is 551.
    p = 1 / 548
    expected = [1 / 551 - p] * 548
    expected[318], expected[9] = 2 / 551 - p, 3 / 551 - p
    l1 = math.fsum(abs(x) for x in expected)
    assert result['tv'] == pytest.approx(l1 / 2, abs=1e-15)
    assert result['is_are_l1_share'] == pytest.approx((abs(expected[318]) + abs(expected[389])) / l1,
                                                       abs=1e-14)
    assert result['top_increase_id'] == 9
    assert result['top_decrease_id'] == 0
    shifted = posthoc.distribution_change(clean + 100, arm - 200)
    assert shifted['tv'] == pytest.approx(result['tv'], abs=1e-15)
    assert shifted['kl_clean_to_arm'] == pytest.approx(result['kl_clean_to_arm'], abs=1e-15)


def test_every_published_case_matches_original_independent_science_replay():
    with (RESULT / 'per-case.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    original = json.loads((RESULT.parent / 'science-replay.json').read_text(encoding='utf-8'))
    by_id = {r['case_id']: r for r in original['targets'] + original['contexts']}
    assert len(rows) == 120
    assert len({(r['case_id'], r['direction']) for r in rows}) == 120
    names = {'is_minus_are': 'present', 'was_minus_were': 'past',
             'the_minus_a': 'the_vs_a', 'in_minus_on': 'in_vs_on'}
    for row in rows:
        ref = by_id[row['case_id']]
        assert float(row['q']) == ref['dose']
        for new, old in names.items():
            response = ref['responses'][row['direction']][old]
            for field, original_field in (('odd', 'odd'), ('even', 'even'), ('odd_per_q', 'odd_per_unit')):
                assert float(row[new + '_' + field]) == pytest.approx(response[original_field], abs=2e-14)
        for sign in ('plus', 'minus'):
            old = original['vocabulary'][row['case_id']][row['direction'] + '_' + sign]
            assert float(row[sign + '_kl_clean_to_arm']) == pytest.approx(old['kl_clean_to_arm'], abs=2e-14)
            assert int(row[sign + '_top_arm_id']) == old['top_arm'][0]


def test_published_pair_inventory_and_summary_are_complete():
    with (RESULT / 'pairwise-vocabulary-cosines.csv').open(encoding='utf-8', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    summary = json.loads((RESULT / 'summary.json').read_text(encoding='utf-8'))
    assert len(pairs) == 2340
    for direction in posthoc.DIRECTIONS:
        for group, count in (('context-context', 28), ('target-target', 496), ('target-context', 256)):
            group_values = [float(r['cosine']) for r in pairs if r['direction'] == direction and r['group'] == group]
            assert len(group_values) == count
            assert all(-1 <= value <= 1 for value in group_values)
            assert summary['centered_vocabulary_cosines'][direction][group]['mean'] == pytest.approx(
                math.fsum(group_values) / count, abs=1e-15)
    assert summary['same_lemma_target_pairs'] == {'n': 16, 'singular_greater_count': 16}
    assert summary['evidence'] == {'read_logit_events': 400, 'signed_intervention_witnesses': 240,
                                   'subnormal_logit_values': 0}


def test_context_table_uses_published_rows_without_selection():
    with (RESULT / 'per-case.csv').open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['role'] == 'context' and r['direction'] == 'u']
    note = (RESULT / 'README.md').read_text(encoding='utf-8')
    assert len(rows) == 8
    for row in rows:
        expected = (f"| {row['text']} | {float(row['q']):.4f} | "
                    f"{float(row['is_minus_are_odd_per_q']):.4f} | "
                    f"{float(row['was_minus_were_odd_per_q']):.4f} | "
                    f"{100 * float(row['plus_tv']):.2f}% | "
                    f"{100 * float(row['plus_is_are_l1_share']):.2f}% |")
        assert expected in note


def test_changed_journal_is_rejected_before_any_tensor_is_read(tmp_path, monkeypatch):
    (tmp_path / 'events.jsonl').write_bytes(b'{}\n')

    def unexpected_tensor_load(*args, **kwargs):
        raise AssertionError('direction tensors must not load before the journal pin')

    monkeypatch.setattr(posthoc.saved, 'load_frozen_inputs', unexpected_tensor_load)
    with pytest.raises(ValueError, match='independent pin'):
        posthoc.load_evidence(ROOT, tmp_path)
