"""Fabricated number inputs only; no pretrained or private study cases."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from looking_glass import agreement_heldout as heldout
from looking_glass import agreement_patch_runner as runner_module
from looking_glass import pretrained_exposures_v4 as exposures
from looking_glass import pretrained_number_commitments as commitments
from looking_glass.agreement_journal import AgreementJournal
from looking_glass.agreement_metrics import score_factorial
from looking_glass.agreement_patch_runner import NaturalAgreementRunner, StudyFailure


class Attention(nn.Module):
    def forward(self, value):
        return (torch.cumsum(value, dim=1) * 0.02,)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.mlp = nn.Linear(8, 8)

    def forward(self, value):
        value = value + self.attn(value)[0]
        return (value + self.mlp(value) * 0.03,)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.embedding = nn.Embedding(600, 8)
        self.transformer.h = nn.ModuleList(Block() for _ in range(6))
        self.head = nn.Linear(8, 600)
        self.config = SimpleNamespace(vocab_size=600, n_embd=8, use_cache=False,
                                      _attn_implementation='eager')
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.calls += 1
        if self.calls == 1 and hasattr(self, 'on_first_forward'):
            self.on_first_forward()
        value = self.transformer.embedding(input_ids)
        for block in self.transformer.h:
            value = block(value)[0]
        return SimpleNamespace(logits=self.head(value), past_key_values=None)


def cpu_toy(seed):
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ToyModel().eval()


def discovery_prediction(effect_sign=1, magnitude_tie=False):
    """Make a complete fake discovery file from the independent scorer."""
    cases, patches, magnitudes = [], [], []
    sites = [item['id'] for item in runner_module._CANDIDATES]
    for family in range(4):
        members = []
        for lemma in range(2):
            for number in ('singular', 'plural'):
                row = {'id': f'q{family}-{lemma}-{number}', 'quartet': f'q{family}',
                       'lemma': f'l{lemma}', 'number': number,
                       'clean_is': 4.0 if number == 'singular' else 0.0,
                       'clean_are': 0.0 if number == 'singular' else 4.0}
                cases.append(row)
                members.append(row)
        for site_index, site in enumerate(sites):
            for lemma in range(2):
                magnitudes.append({'site': site, 'quartet': f'q{family}', 'lemma': f'l{lemma}',
                                   'singular': f'q{family}-{lemma}-singular',
                                   'plural': f'q{family}-{lemma}-plural',
                                   'l2': 1.0 if magnitude_tie else float(site_index + 1)})
            for target in members:
                for donor in members:
                    shift = (effect_sign * 0.125 * (site_index + 1)
                             if donor['number'] != target['number'] else 0.0)
                    weight = 1 if target['number'] == 'singular' else -1
                    patches.append({'site': site, 'target': target['id'], 'donor': donor['id'],
                                    'patched_is': target['clean_is'] - weight * shift,
                                    'patched_are': target['clean_are']})
    metrics = score_factorial(cases, patches, sites)
    selection = runner_module._discovery_selection(metrics, magnitudes, 4)
    def events(kind):
        result = []
        contrast = ({'first_token': ' is', 'first_id': 318, 'second_token': ' are', 'second_id': 389}
                    if kind == 'present_site_metrics' else
                    {'first_token': ' was', 'first_id': 373, 'second_token': ' were', 'second_id': 547})
        for site in sites:
            pairs = [runner_module._pair_with_signs(row) for row in metrics['pairs'] if row['site'] == site]
            if kind != 'present_site_metrics':
                pairs = [runner_module._label_past_pair(row) for row in pairs]
            result.append({'kind': kind, 'site': site, 'contrast': contrast,
                           'metrics': {'baseline': metrics['baseline'],
                                       'site_summary': next(row for row in metrics['site_summaries']
                                                            if row['site'] == site),
                                       'quartets': [row for row in metrics['quartets'] if row['site'] == site],
                                       'pairs': pairs}})
        return result
    record = {'schema': 'mlg-number-natural-predictions-v1',
              'protocol_sha256': commitments.PROTOCOL_SHA256, 'source_commit': 'a' * 40,
              'checkpoint_revision': commitments.CHECKPOINT_REVISION, 'selection': selection,
              'present_site_metrics': events('present_site_metrics'),
              'secondary_past_site_metrics': events('secondary_past_site_metrics')}
    return record


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('utf-8')


def freeze_prediction(record):
    data = canonical(record)
    freeze = commitments.make_prediction_freeze({'natural': {
        'source_commit': 'a' * 40, 'runtime_sha256': 'b' * 64,
        'prediction_sha256': hashlib.sha256(data).hexdigest()}})
    return data, freeze


@pytest.fixture
def fake_study():
    nouns = ('alpha/alphas', 'beta/betas', 'gamma/gammas', 'delta/deltas',
             'epsilon/epsilons', 'zeta/zetas', 'eta/etas', 'theta/thetas')
    plural = ('near the boxes', 'behind the cars', 'above the lamps', 'below the desks')
    singular = ('near the box', 'behind the car', 'above the lamp', 'below the desk')
    with mock.patch.multiple(commitments, NOUNS=nouns, PLURAL_CONTEXTS=plural, SINGULAR_CONTEXTS=singular):
        prediction = canonical(discovery_prediction())
        freeze = commitments.make_prediction_freeze({'natural': {
            'source_commit': 'a' * 40, 'runtime_sha256': 'b' * 64,
            'prediction_sha256': hashlib.sha256(prediction).hexdigest()}})
        inventory = commitments.load_inventory_snapshot(exposures.canonical_inventory_bytes(),
                                                        exposures.INVENTORY_SHA256)
        nonce = bytes(range(32))
        allocation = commitments.sample_families(freeze, nonce, inventory_snapshot=inventory,
                                                  current_inventory_sha256=inventory.sha256)
        tokenized = []
        for family in allocation.families:
            tokenized.append([{'text': text, 'input_ids': [10, 100 + 4 * family.family_index + index,
                                                          30 + family.family_index],
                               'attention_mask': [1, 1, 1], 'subject_index': 1}
                              for index, text in enumerate(family.texts)])
        sealed = commitments.seal_heldout(freeze, nonce, tokenized, allocation=allocation,
                                         inventory_snapshot=inventory, current_inventory_sha256=inventory.sha256)
        yield {'freeze': freeze, 'prediction_bytes': prediction, 'split_nonce': nonce,
               'record_bytes': sealed.record_bytes, 'heldout_commitment': sealed.commitment_sha256,
               'inventory_snapshot': inventory, 'current_inventory_sha256': inventory.sha256}


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_score_uses_frozen_site_predictions_and_zero_without_selection():
    selection = discovery_prediction()['selection']
    site = selection['ranking'][0]
    selection['predictions'][site] = 2.0
    rows = [{'site': item['id'], 'quartet': f'q{i}', 'number_main_contrast': value}
            for item in runner_module._CANDIDATES for i, value in enumerate((1.0, 2.0, -1.0, 0.0))]
    scored = heldout.score_heldout({'quartets': rows}, selection)
    item = next(row for row in scored['sites'] if row['site'] == site)
    assert [row['prediction_error'] for row in item['families']] == [-1.0, 0.0, -3.0, -2.0]
    assert item['frozen_predictor'] == {'mae': 1.5, 'rmse': math.sqrt(3.5)}
    assert item['zero_predictor'] == {'mae': 1.0, 'rmse': math.sqrt(1.5)}
    assert scored['selected_site'] == selection['selected_site']
    for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [dict(rows[-1], quartet='q0')]):
        with pytest.raises(StudyFailure):
            heldout.score_heldout({'quartets': bad}, selection)


def test_toy_model_is_cpu_even_under_non_cpu_default():
    with torch.device('meta'):
        model = cpu_toy(123)
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())


def test_canonical_prediction_file_is_owned_and_rejects_mismatches(fake_study):
    frozen = heldout.load_frozen_natural_predictions(fake_study['prediction_bytes'], fake_study['freeze'])
    decoded = json.loads(fake_study['prediction_bytes'])
    original = heldout.frozen_selection(frozen)[0]['predictions'].copy()
    decoded['selection']['predictions'].clear()
    assert heldout.frozen_selection(frozen)[0]['predictions'] == original
    with pytest.raises(StudyFailure):
        heldout.load_frozen_natural_predictions(fake_study['prediction_bytes'] + b' ', fake_study['freeze'])
    mismatch = commitments.make_prediction_freeze({'natural': {
        'source_commit': 'a' * 40, 'runtime_sha256': 'b' * 64, 'prediction_sha256': '0' * 64}})
    with pytest.raises(StudyFailure):
        heldout.load_frozen_natural_predictions(fake_study['prediction_bytes'], mismatch)


def test_abstention_and_duplicate_baselines_remain_frozen():
    record = discovery_prediction(effect_sign=-1, magnitude_tie=True)
    data, freeze = freeze_prediction(record)
    selection = heldout.frozen_selection(heldout.load_frozen_natural_predictions(data, freeze))[0]
    assert selection['selected_site'] is None
    assert selection['magnitude_site'] == selection['earliest_site']
    rows = [{'site': site['id'], 'quartet': f'q{i}', 'number_main_contrast': float(-i)}
            for site in runner_module._CANDIDATES for i in range(4)]
    scored = heldout.score_heldout({'quartets': rows}, selection)
    assert scored['selected_site'] is None
    assert scored['magnitude_site'] == scored['earliest_site']
    assert 'ranking' not in scored


@pytest.mark.parametrize('change', ['selected', 'mean', 'pair', 'missing_pair', 'past_label',
                                    'magnitude_family', 'boolean_count', 'magnitude_ids', 'past_case_ids',
                                    'duplicate_magnitude_ids'])
def test_frozen_discovery_corruption_rejected_even_when_digest_matches(change):
    record = discovery_prediction()
    event = record['present_site_metrics'][0]['metrics']
    if change == 'selected':
        record['selection']['selected_site'] = '1:attn:last'
    elif change == 'mean':
        event['quartets'][0]['number_main_contrast'] += 1.0
    elif change == 'pair':
        event['pairs'][0]['signed_target_shift'] += 1.0
    elif change == 'missing_pair':
        event['pairs'].pop()
    elif change == 'past_label':
        pair = record['secondary_past_site_metrics'][0]['metrics']['pairs'][0]
        pair['clean_is'] = pair.pop('clean_was')
    elif change == 'magnitude_family':
        record['selection']['magnitude_pairs'][0]['quartet'] = 'unrelated'
    elif change == 'magnitude_ids':
        for pair in record['selection']['magnitude_pairs']:
            if pair['quartet'] == 'q0' and pair['lemma'] == 'l0':
                pair['singular'], pair['plural'] = 'invented-singular', 'invented-plural'
    elif change == 'duplicate_magnitude_ids':
        for pair in record['selection']['magnitude_pairs']:
            if pair['quartet'] == 'q0' and pair['lemma'] == 'l1':
                pair['singular'], pair['plural'] = 'q0-0-singular', 'q0-0-plural'
    elif change == 'past_case_ids':
        for past_event in record['secondary_past_site_metrics']:
            for pair in past_event['metrics']['pairs']:
                pair['target'] += '-other'
                pair['donor'] += '-other'
    else:
        event['quartets'][0]['cell_counts']['same_lemma_same_number'] = True
    data, freeze = freeze_prediction(record)
    with pytest.raises(StudyFailure):
        heldout.load_frozen_natural_predictions(data, freeze)


def test_heldout_runner_uses_committed_cases_and_never_reranks(fake_study, tmp_path, cpu_threads):
    model = cpu_toy(123)
    journal = AgreementJournal(tmp_path / 'attempt', {'protocol': 'fabricated-only'})
    owner = NaturalAgreementRunner(model, journal, expected_quartets=4)
    original_prediction = fake_study['prediction_bytes']
    original_selection_record = json.loads(original_prediction)['selection']
    def mutate_caller_binding():
        altered = json.loads(original_prediction)
        altered['selection']['selected_site'] = '1:attn:last'
        altered['selection']['predictions']['1:attn:last'] = 999.0
        fake_study['prediction_bytes'] = canonical(altered)
    model.on_first_forward = mutate_caller_binding
    original_selection = runner_module._discovery_selection
    def check_discovery_only(metrics, magnitudes, families):
        assert model.calls == 0, 'selection must not inspect held-out outcomes'
        return original_selection(metrics, magnitudes, families)
    with mock.patch.object(heldout, '_discovery_selection', side_effect=check_discovery_only), \
         mock.patch.object(runner_module, '_vector_l2', side_effect=AssertionError('held-out magnitude ranking')):
        result = heldout.run_committed_heldout(owner, **fake_study)
    assert model.calls == 608 and result['forwards'] == 608
    assert 'selection' not in result
    evaluation = result['heldout_evaluation']
    assert len(evaluation['sites']) == 8 and len(evaluation['clean_cases']) == 16
    assert fake_study['prediction_bytes'] != original_prediction
    assert evaluation['selected_site'] == original_selection_record['selected_site']
    assert next(site for site in evaluation['sites'] if site['site'] == '1:attn:last')['frozen_prediction'] == original_selection_record['predictions']['1:attn:last']
    assert evaluation['prediction_freeze_sha256'] == fake_study['freeze'].sha256
    assert evaluation['heldout_commitment'] == fake_study['heldout_commitment']
    assert evaluation['exposure_inventory_sha256'] == fake_study['inventory_snapshot'].sha256
    assert all(len(site['families']) == 4 for site in evaluation['sites'])
    events = [json.loads(line)['event'] for line in (tmp_path / 'attempt' / 'events.jsonl').read_text().splitlines()]
    assert events[0]['phase'] == 'heldout'
    assert events[0]['heldout_commitment'] == fake_study['heldout_commitment']
    assert events[0]['exposure_inventory_sha256'] == fake_study['inventory_snapshot'].sha256
    assert len(events[0]['cases']) == 16
    committed = json.loads(fake_study['record_bytes'])['families']
    for family in committed:
        for index, item in enumerate(family['cases']):
            recorded = next(case for case in events[0]['cases']
                            if case['case_id'] == f"heldout-{family['family_index']}-{index}")
            assert recorded['input_ids'] == item['input_ids']
            assert recorded['subject_index'] == item['subject_index']
    assert all(case['number'] == ('singular' if int(case['case_id'][-1]) % 2 == 0 else 'plural')
               for case in events[0]['cases'])
    summary = next(event['result'] for event in events if event['kind'] == 'summary')
    assert 'selection' not in summary and 'heldout_evaluation' in summary
    past = next(event for event in events if event['kind'] == 'secondary_past_site_metrics')
    assert 'clean_was' in past['metrics']['pairs'][0] and 'clean_is' not in past['metrics']['pairs'][0]


def test_bad_commitment_and_stale_inventory_never_call_model(fake_study, tmp_path, cpu_threads):
    model = cpu_toy(123)
    for index, change in enumerate(({'heldout_commitment': '0' * 64},
                                    {'current_inventory_sha256': '0' * 64},
                                    {'prediction_bytes': fake_study['prediction_bytes'] + b' '})):
        owner = NaturalAgreementRunner(model, AgreementJournal(tmp_path / f'attempt-{index}', {}),
                                       expected_quartets=4)
        with pytest.raises((StudyFailure, ValueError)):
            heldout.run_committed_heldout(owner, **{**fake_study, **change})
    assert model.calls == 0


def test_exact_discovery_journal_events_form_a_compatible_prediction_file(tmp_path, cpu_threads):
    model = cpu_toy(321)
    cases = tuple(runner_module.AgreementCase(case_id=f'q{family}-{index}', quartet=f'q{family}',
                                              lemma=f'l{index // 2}',
                                              number='singular' if index % 2 == 0 else 'plural',
                                              input_ids=(10, 100 + 4 * family + index, 30 + family),
                                              subject_index=1)
                  for family in range(4) for index in range(4))
    path = tmp_path / 'discovery'
    result = NaturalAgreementRunner(model, AgreementJournal(path, {'protocol': 'fabricated-only'})).run(cases)
    events = [json.loads(line)['event'] for line in (path / 'events.jsonl').read_text().splitlines()]
    prediction = {'schema': 'mlg-number-natural-predictions-v1',
                  'protocol_sha256': commitments.PROTOCOL_SHA256, 'source_commit': 'a' * 40,
                  'checkpoint_revision': commitments.CHECKPOINT_REVISION, 'selection': result['selection'],
                  'present_site_metrics': [event for event in events if event['kind'] == 'present_site_metrics'],
                  'secondary_past_site_metrics': [event for event in events
                                                  if event['kind'] == 'secondary_past_site_metrics']}
    data, freeze = freeze_prediction(prediction)
    loaded = heldout.load_frozen_natural_predictions(data, freeze)
    assert heldout.frozen_selection(loaded)[0] == result['selection']
