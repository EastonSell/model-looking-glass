"""Fabricated-only successor execution and fail-closed boundary checks."""
import hashlib
import json

import pytest

from looking_glass import agreement_heldout, natural_agreement_successor as successor
from looking_glass import pretrained_exposures_v4 as exposures
from looking_glass import pretrained_number_commitments as commitments
from looking_glass.agreement_journal import AgreementJournal
from looking_glass.agreement_patch_runner import NaturalAgreementRunner
from test_agreement_heldout import cpu_threads, cpu_toy, discovery_prediction, freeze_prediction
from test_natural_successor_sampler import FakeTokenizer, pool

try:
    from looking_glass import natural_successor_execution as execution
except ImportError:
    execution = None


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')


@pytest.fixture
def fabricated(monkeypatch):
    prediction, freeze = freeze_prediction(discovery_prediction())
    monkeypatch.setattr(successor, 'PREDICTION_FREEZE_SHA256', freeze.sha256)
    values = pool()
    pool_bytes = canonical(values)
    class ToyTokenizer(FakeTokenizer):
        def encode(self, text, *, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            return [400 if i == 0 and token == 1 else token for i, token in enumerate(ids)]
    tokenizer = ToyTokenizer(values)
    rows = successor.tokenize_candidate_pool(values, tokenizer)
    input_bytes = canonical({'schema': 'mlg-natural-successor-candidate-inputs-v1',
                             'pool_sha256': hashlib.sha256(pool_bytes).hexdigest(),
                             'tokenizer_sha256': 'a' * 64,
                             'cases': [{'input_ids': list(row)} for row in rows]})
    inventory = commitments.load_inventory_snapshot(exposures.canonical_inventory_bytes(),
                                                     exposures.INVENTORY_SHA256)
    old_commitment = 'b' * 64
    receipt = canonical({'schema': 'mlg-natural-external-input-comparison-v1',
                         'status': 'clear', 'comparison': 'bidirectional-complete-row-prefix',
                         'natural_heldout_commitment': old_commitment,
                         'natural_prediction_freeze_sha256': freeze.sha256,
                         'public_input_file_sha256': hashlib.sha256(input_bytes).hexdigest(),
                         'natural_case_count': 16, 'public_case_count': 256,
                         'comparator_source_sha256': 'c' * 64, 'observed_utc': 'fabricated'})
    pins = {'pool_sha256': hashlib.sha256(pool_bytes).hexdigest(),
            'protocol_sha256': 'd' * 64, 'exposure_sha256': inventory.sha256,
            'old_commitment': old_commitment, 'pool_clear_sha256': hashlib.sha256(receipt).hexdigest()}
    nonce = bytes(range(32))
    commitment, record = successor.seal_record(pool_bytes, nonce, tokenizer, **pins)
    return {'freeze': freeze, 'prediction_bytes': prediction, 'pool_bytes': pool_bytes,
            'input_bytes': input_bytes, 'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
            'tokenizer_sha256': 'a' * 64, 'comparator_sha256': 'c' * 64,
            'pool_clearance_bytes': receipt, 'split_nonce': nonce, 'record_bytes': record,
            'heldout_commitment': commitment, 'sealed_snapshot': inventory,
            'current_snapshot': inventory, 'observed_current_sha256': inventory.sha256,
            'pins': pins, 'tokenizer': tokenizer}


def test_new_record_drives_exact_frozen_runner_without_reranking(fabricated, tmp_path, cpu_threads):
    assert execution is not None
    model = cpu_toy(123)
    runner = NaturalAgreementRunner(model, AgreementJournal(tmp_path / 'attempt', {}), expected_quartets=4)
    payload = json.loads(fabricated['current_snapshot'].record_bytes)
    payload['records'].append({'record_id': 'fabricated-nonoverlap', 'category': 'observed',
                               'token_ids': [399, 4]})
    current_raw = canonical(payload)
    current_sha = hashlib.sha256(current_raw).hexdigest()
    current = commitments.load_inventory_snapshot(current_raw, current_sha)
    result = execution.run_successor_heldout(runner, **dict(fabricated, current_snapshot=current,
                                                            observed_current_sha256=current_sha))
    assert model.calls == result['forwards'] == 608
    assert len(result['heldout_evaluation']['clean_cases']) == 16
    assert len(result['heldout_evaluation']['sites']) == 8
    assert result['heldout_evaluation']['heldout_commitment'] == fabricated['heldout_commitment']
    assert result['heldout_evaluation']['exposure_inventory_sha256'] == fabricated['pins']['exposure_sha256']
    events = [json.loads(line)['event'] for line in (tmp_path / 'attempt' / 'events.jsonl').read_text().splitlines()]
    for family in json.loads(fabricated['record_bytes'])['families']:
        for index, item in enumerate(family['cases']):
            actual = next(row for row in events[0]['cases']
                          if row['case_id'] == f"heldout-{family['family_index']}-{index}")
            assert actual['input_ids'] == item['input_ids']
            assert actual['subject_index'] == item['subject_index']


@pytest.mark.parametrize('change', [
    {'heldout_commitment': '0' * 64}, {'observed_current_sha256': '0' * 64},
    {'prediction_bytes': b'{}'}, {'pool_clearance_bytes': b'{}'},
    {'input_bytes': b'{}'}, {'comparator_sha256': '0' * 64},
])
def test_invalid_binding_never_calls_the_model(fabricated, tmp_path, cpu_threads, change):
    assert execution is not None
    model = cpu_toy(456)
    runner = NaturalAgreementRunner(model, AgreementJournal(tmp_path / 'attempt', {}), expected_quartets=4)
    with pytest.raises((ValueError, RuntimeError)):
        execution.run_successor_heldout(runner, **dict(fabricated, **change))
    assert model.calls == 0


@pytest.mark.parametrize('relation', ['exact', 'newer-prefix', 'selected-prefix'])
def test_newer_inventory_collision_stops_before_a_forward(fabricated, tmp_path, cpu_threads, relation):
    assert execution is not None
    selected = json.loads(fabricated['record_bytes'])['families'][0]['cases'][0]['input_ids']
    collision = (selected if relation == 'exact' else selected[:-1] if relation == 'newer-prefix'
                 else selected + [77])
    payload = json.loads(fabricated['current_snapshot'].record_bytes)
    payload['records'].append({'record_id': 'fabricated-newer', 'category': 'observed', 'token_ids': collision})
    raw = canonical(payload)
    digest = hashlib.sha256(raw).hexdigest()
    newer = commitments.load_inventory_snapshot(raw, digest)
    model = cpu_toy(789)
    runner = NaturalAgreementRunner(model, AgreementJournal(tmp_path / 'attempt', {}), expected_quartets=4)
    with pytest.raises((ValueError, RuntimeError)):
        execution.run_successor_heldout(runner, **dict(fabricated, current_snapshot=newer,
                                                        observed_current_sha256=digest))
    assert model.calls == 0
