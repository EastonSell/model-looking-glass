"""Small fabricated-only saved-evidence tests; no model or private inputs."""
import hashlib
import json
import math
from pathlib import Path
import random
import struct

import pytest

try:
    from looking_glass import bert_matched_evidence as audit
except ImportError:
    audit = None


def vector(position=None, value=0.0):
    data = [0.0] * 768
    if position is not None:
        data[position] = value
    return struct.pack('<768f', *data)


def fabricated_witness():
    base = vector()
    captures = {
        'block0': {'target': base, 'A_opposite': vector(0, 1), 'B_same': vector(1, 2),
                   'B_opposite': vector(2, 4)},
        'block2': {'target': base, 'A_opposite': vector(3, .5), 'B_same': vector(4, 1),
                   'B_opposite': vector(5, 2)}}
    patches = {}
    for site, offset in (('block0', 0), ('block2', 3)):
        prefix = b'bert-matched-v1\0fake-row\0' + site.encode('ascii') + b'\0'
        digest = b''.join(hashlib.sha256(prefix + i.to_bytes(4, 'big')).digest() for i in range(3))
        signs = [1 if byte & (1 << bit) else -1 for byte in digest for bit in range(7, -1, -1)]
        artificial = struct.pack('<768f', *[sign * (.5 / math.sqrt(768.0)) for sign in signs])
        submitted = {'self': base, 'A_opposite': vector(offset, .5), 'B_same': vector(offset + 1, .5),
                     'B_opposite': vector(offset + 2, .5), 'artificial': artificial}
        patches[site] = {name: {'submitted': payload, 'pre': base, 'post': payload}
                         for name, payload in submitted.items()}
    return captures, patches


def test_saved_witness_recomputes_six_distances_and_every_landing():
    assert audit is not None, 'the independent saved-evidence auditor is missing'
    captures, patches = fabricated_witness()
    result = audit.audit_witness('fake-row', captures, patches)
    assert result['radius'] == .5
    assert result['sites']['block0']['B_same']['distance'] == 2.0
    assert result['sites']['block0']['B_same']['fraction'] == .25
    assert result['sites']['block2']['A_opposite']['fraction'] == 1.0


def test_previously_exposed_public_rehearsal_manifest_is_structurally_accepted():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'experiments' / 'bert_agreement_public_cases_v1.json').read_text(encoding='utf-8'))
    assert len(manifest['rows']) == 96
    rows, targets, by_id = audit._family_rows(manifest, 'development', 1, 103)
    assert (len(rows), len(targets), len(by_id)) == (8, 4, 8)


def test_random_fabricated_plans_match_an_independent_witness_replay():
    from looking_glass import bert_matched_numeric as numeric
    rng = random.Random(68871)
    for trial in range(12):
        raw = lambda: struct.pack('<768f', *(rng.uniform(-2, 2) for _ in range(768)))
        captures = {site: {'target': raw(), **{name: raw() for name in
                    ('A_opposite', 'B_same', 'B_opposite')}} for site in ('block0', 'block2')}
        bases = {site: captures[site]['target'] for site in captures}
        donors = {site: {name: captures[site][name] for name in
                  ('A_opposite', 'B_same', 'B_opposite')} for site in captures}
        row_id = f'fabricated-{trial}'
        plan = numeric.plan_case(row_id, bases, donors)
        patches = {}
        for site in captures:
            requested = {'self': bases[site], **{name: action['value'] for name, action in plan['sites'][site].items()}}
            patches[site] = {name: {'pre': bases[site], 'submitted': value, 'post': value}
                             for name, value in requested.items()}
        assert audit.audit_witness(row_id, captures, patches)['radius'] == plan['radius']


@pytest.mark.parametrize('failure', ('pre', 'post', 'dose', 'floor', 'nan', 'wrong_signs', 'mutable'))
def test_corrupt_or_ineffective_saved_witness_is_rejected(failure):
    assert audit is not None, 'the independent saved-evidence auditor is missing'
    captures, patches = fabricated_witness()
    if failure in ('pre', 'post', 'dose'):
        field = 'submitted' if failure == 'dose' else failure
        patches['block0']['B_same'][field] = vector(1, .25)
    elif failure == 'floor':
        captures['block0']['B_same'] = vector(1, 0.00001)
    elif failure == 'nan':
        captures['block0']['target'] = struct.pack('<768I', 0x7fc00000, *([0] * 767))
    elif failure == 'wrong_signs':
        patches['block0']['artificial']['post'] = patches['block2']['artificial']['post']
    else:
        patches['block0']['self']['submitted'] = bytearray(patches['block0']['self']['submitted'])
    with pytest.raises(ValueError):
        audit.audit_witness('fake-row', captures, patches)


def _bits(value):
    return int.from_bytes(struct.pack('<f', value), 'little')


def _pair(margin):
    return [_bits(margin), _bits(0.0)]


def fabricated_records():
    records = []
    cells = (('singular', 'singular'), ('singular', 'plural'),
             ('plural', 'singular'), ('plural', 'plural'))
    for family in range(1, 9):
        for index, (subject, distractor) in enumerate(cells):
            r = 1 if subject == 'singular' else -1
            clean = {'bits': _pair(r), 'mask_sha256': 'a' * 64}
            sites = {}
            for site, primary in (('block0', .25), ('block2', .125)):
                # A-opposite deliberately differs from B-opposite: the primary
                # must compare B-opposite to B-same, not A to identity.
                margins = {'self': r, 'A_opposite': r - r * .75,
                           'B_same': r + .5, 'B_opposite': r + .5 - r * primary,
                           'artificial': r + .125}
                sites[site] = {key: {'bits': _pair(value),
                    'mask_sha256': ('a' if key == 'self' else 'b') * 64}
                    for key, value in margins.items()}
            records.append({'family': family, 'row_id': f'fake-{family}-{index}',
                'subject_number': subject, 'distractor_number': distractor, 'clean': clean,
                'final_mask_sha256': 'a' * 64, 'sites': sites})
    return records


def test_family_reduction_uses_b_lexical_control_and_paired_decision():
    assert audit is not None, 'the independent saved-evidence auditor is missing'
    summary = audit.summarize(fabricated_records())
    assert summary['block0_mean'] == .25
    assert summary['block2_mean'] == .125
    assert summary['paired_mean'] == .125
    assert summary['positive_families'] == 8
    assert summary['criterion_met'] is True
    assert summary['baseline_accuracy'] == 1.0
    assert summary['families'][0]['block0'] == .25
    assert summary['effects_by_cell']['plural/singular']['paired_mean'] == .125
    assert summary['effects_by_cell']['singular/plural']['sites']['block0']['primary_mean'] == .25


def test_family_sum_retains_small_terms_and_tied_baselines_are_incorrect():
    records = fabricated_records()
    for family in range(1, 9):
        for index, effect in enumerate((2.0**50, .125, -(2.0**50), .125)):
            row = records[(family - 1) * 4 + index]
            r = 1 if row['subject_number'] == 'singular' else -1
            for site in ('block0', 'block2'):
                row['sites'][site]['B_same']['bits'] = _pair(0.0)
                row['sites'][site]['B_opposite']['bits'] = _pair(-r * effect)
    records[0]['clean']['bits'] = _pair(0.0)
    for site in ('block0', 'block2'):
        records[0]['sites'][site]['self']['bits'] = _pair(0.0)
    summary = audit.summarize(records)
    assert summary['families'][0]['block0'] == .0625
    assert summary['block0_mean'] == .0625
    assert summary['paired_mean'] == 0.0
    assert summary['criterion_met'] is False
    assert summary['baseline_accuracy'] == 31 / 32


def test_opposite_donor_reversals_include_incorrect_to_correct_and_exclude_ties():
    records = fabricated_records()
    # Singular target begins wrong and then becomes correct; it is retained.
    records[0]['clean']['bits'] = _pair(-1.0)
    for site in ('block0', 'block2'):
        records[0]['sites'][site]['self']['bits'] = _pair(-1.0)
        records[0]['sites'][site]['A_opposite']['bits'] = _pair(1.0)
        records[0]['sites'][site]['B_opposite']['bits'] = _pair(1.0)
    # A tied clean margin does not change strict sign.
    records[1]['clean']['bits'] = _pair(0.0)
    for site in ('block0', 'block2'):
        records[1]['sites'][site]['self']['bits'] = _pair(0.0)
        records[1]['sites'][site]['B_opposite']['bits'] = _pair(-1.0)
    result = audit.summarize(records)
    first, tied = result['cases'][:2]
    assert first['baseline_correct'] is False
    assert first['sites']['block0']['b_opposite_sign_reversal'] is True
    assert first['sites']['block0']['b_opposite_wrong_to_correct'] is True
    assert first['sites']['block0']['b_opposite_correct_to_wrong'] is False
    assert tied['sites']['block0']['b_opposite_sign_reversal'] is False
    assert result['effects_by_cell']['singular/singular']['sites']['block0']['b_opposite_wrong_to_correct'] == 1


def test_criterion_does_not_round_a_below_threshold_paired_mean_up():
    records = fabricated_records()
    for row in records:
        r = 1 if row['subject_number'] == 'singular' else -1
        row['sites']['block2']['B_opposite']['bits'] = _pair(r + .5 - r * .1875)
    summary = audit.summarize(records)
    assert summary['block0_mean'] == .25
    assert summary['paired_mean'] == .0625
    assert summary['positive_families'] == 8
    assert summary['criterion_met'] is False


@pytest.mark.parametrize('positive_families, expected', ((5, False), (6, True)))
def test_criterion_requires_six_positive_family_differences(positive_families, expected):
    records = fabricated_records()
    for row in records:
        sign = 1 if row['subject_number'] == 'singular' else -1
        for site, effect in (('block0', .5 if row['family'] <= positive_families else .125),
                             ('block2', .25)):
            row['sites'][site]['B_same']['bits'] = _pair(0)
            row['sites'][site]['B_opposite']['bits'] = _pair(-sign * effect)
    summary = audit.summarize(records)
    assert summary['block0_mean'] > .1
    assert summary['paired_mean'] > .1
    assert summary['positive_families'] == positive_families
    assert summary['criterion_met'] is expected


@pytest.mark.parametrize('failure', ('missing', 'order', 'self', 'final', 'nonfinite'))
def test_missing_misordered_or_invalid_analysis_record_fails(failure):
    assert audit is not None, 'the independent saved-evidence auditor is missing'
    records = fabricated_records()
    if failure == 'missing':
        records.pop()
    elif failure == 'order':
        records[0], records[1] = records[1], records[0]
    elif failure == 'self':
        records[0]['sites']['block0']['self']['mask_sha256'] = 'b' * 64
    elif failure == 'final':
        records[0]['final_mask_sha256'] = 'b' * 64
    else:
        records[0]['sites']['block0']['B_opposite']['bits'][0] = 0x7f800000
    with pytest.raises(ValueError):
        audit.summarize(records)
