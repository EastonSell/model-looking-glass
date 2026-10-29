"""No-checkpoint CPU execution tests using fabricated logits and states."""
import json
import hashlib
import copy
import re
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from looking_glass.flan_t5_sentiment_journal import EvidenceJournal, Limits
from looking_glass.flan_t5_sentiment_runner import (
    _EvaluationRunner, _encode_case, _label_targets, _run_fabricated, _require_live_runtime,
    run_cpu_core, selection_to_groups,
)


class ToyTokenizer:
    all_special_ids = [0, 1, 7]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {'positive': [2], 'negative': [3]}[text]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return ''.join({2: 'positive', 3: 'negative', 1: '</s>'}.get(i, '<special>') for i in ids)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(.5)
        self.config = SimpleNamespace(d_model=2, vocab_size=8, decoder_start_token_id=0,
                                      eos_token_id=1, use_cache=False)
        self.calls = 0
        self.encoder_calls = 0
        self.fail_mode = None
        self.fail_at = None
        self.force_negative = False

    def get_encoder(self):
        return self.encoder

    def encoder(self, *, input_ids, attention_mask, return_dict):
        self.encoder_calls += 1
        value = {2: 3., 3: -3., 4: 2., 5: -2.}[int(input_ids[0, 2])]
        state = torch.tensor([[[value, 0.], [2 * value, 0.], [3 * value, 0.], [0., 0.]]])
        return SimpleNamespace(last_hidden_state=state)

    def forward(self, *, encoder_outputs, attention_mask, decoder_input_ids,
                decoder_attention_mask, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        assert torch.equal(attention_mask, torch.ones_like(attention_mask))
        assert torch.equal(decoder_attention_mask, torch.ones_like(decoder_attention_mask))
        self.calls += 1
        if self.fail_at == self.calls:
            raise RuntimeError('synthetic decoder failure')
        state = encoder_outputs[0]
        length = decoder_input_ids.shape[1]
        logits = torch.full((1, length, 8), -4.0)
        positive = bool(state[0, 2, 0] > 0) and not self.force_negative
        for i in range(length):
            if i == 0:
                logits[0, i, 2] = 3.0 if positive else -3.0
                logits[0, i, 3] = -3.0 if positive else 3.0
            else:
                logits[0, i, 1] = 3.0
        if self.fail_mode == 'nonfinite':
            logits[0, 0, 7] = float('nan')
        elif self.fail_mode == 'mutate':
            state[0, 0, 0] += 1
        elif self.fail_mode == 'training':
            self.dropout.train()
        return SimpleNamespace(logits=logits, past_key_values=None)


def journal(tmp_path):
    return EvidenceJournal(tmp_path / 'attempt', {'mode': 'fabricated'},
                           limits=Limits(total_bytes=4_000_000, metadata_bytes=1_000_000,
                                         failure_bytes=64_000, forward_bytes=4096,
                                         max_forwards=20, free_bytes=0))


def test_generated_and_teacher_forced_outputs_save_full_logits_and_eos(tmp_path):
    model, tokenizer = ToyModel().eval(), ToyTokenizer()
    targets = _label_targets(tokenizer, vocab_size=8)
    assert targets == {'positive': [2, 1], 'negative': [3, 1]}
    writer = journal(tmp_path)
    runner = _EvaluationRunner(model, tokenizer, writer, width=2, vocab_size=8, test_only=True)
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    result = runner.evaluate(case, state, state.clone(), {'kind': 'clean'}, targets)
    assert result['generated_ids'] == [2, 1]
    assert result['strict_correct'] is True
    assert result['margin'] > 0
    assert result['teacher']['positive']['target_ids'] == [2, 1]
    assert result['teacher']['positive']['decoder_ids'] == [0, 2]
    assert result['teacher']['positive']['ranks'] == [1, 1]
    assert len(result['greedy']) == 2 and model.calls == 4
    assert all((writer.directory / ref['path']).is_file()
               for ref in result['all_logit_refs'])
    assert writer.component_count == 4


@pytest.mark.parametrize('failure', ['nonfinite', 'mutate', 'training'])
def test_rejected_forward_keeps_returned_bytes_and_stops(failure, tmp_path):
    model, tokenizer = ToyModel().eval(), ToyTokenizer()
    model.fail_mode = failure
    writer = journal(tmp_path)
    runner = _EvaluationRunner(model, tokenizer, writer, width=2, vocab_size=8, test_only=True)
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    with pytest.raises(RuntimeError):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'}, _label_targets(tokenizer, vocab_size=8))
    assert model.calls == 1
    assert writer.active_component is not None
    with pytest.raises(RuntimeError, match='incomplete'):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'}, _label_targets(tokenizer, vocab_size=8))
    assert model.calls == 1
    events = [json.loads(line) for line in (writer.directory / 'events.jsonl').read_text().splitlines()]
    assert any(item['event'] == 'component_rejected' for item in events)
    assert len(list((writer.directory / 'tensors').glob('*.f32le'))) >= 3


def test_runtime_drift_after_return_retains_logits_and_forbids_retry(tmp_path):
    model, tokenizer = ToyModel().eval(), ToyTokenizer()
    writer = journal(tmp_path)
    runner = _EvaluationRunner(model, tokenizer, writer, width=2, vocab_size=8, test_only=True)
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    with mock.patch('looking_glass.flan_t5_sentiment_runner.runtime_numeric_probe',
                    side_effect=[{}, RuntimeError('synthetic drift')]):
        with pytest.raises(RuntimeError, match='drift'):
            runner.evaluate(case, state, state.clone(), {'kind': 'clean'}, _label_targets(tokenizer, vocab_size=8))
    events = [json.loads(line) for line in (writer.directory / 'events.jsonl').read_text().splitlines()]
    rejected = next(item['detail'] for item in events if item['event'] == 'component_rejected')
    assert rejected['return_observed'] is True
    assert (writer.directory / rejected['logits']['path']).is_file()
    with pytest.raises(RuntimeError, match='incomplete'):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'}, _label_targets(tokenizer, vocab_size=8))
    assert model.calls == 1


def test_custody_check_brackets_each_decoder_call_and_fails_closed(tmp_path):
    events = []

    class ObservedModel(ToyModel):
        def forward(self, **kwargs):
            events.append(('enter', self.calls))
            result = super().forward(**kwargs)
            events.append(('return', self.calls))
            return result

    model, writer = ObservedModel().eval(), journal(tmp_path)
    def custody_check():
        events.append(('check', model.calls))

    runner = _EvaluationRunner(model, ToyTokenizer(), writer, width=2, vocab_size=8,
                               test_only=True, custody_check=custody_check)
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    runner.evaluate(case, state, state.clone(), {'kind': 'clean'},
                    _label_targets(ToyTokenizer(), vocab_size=8))
    for index, (kind, count) in enumerate(events):
        if kind == 'enter':
            assert events[index - 1] == ('check', count)
        if kind == 'return':
            assert events[index + 1] == ('check', count)
    assert model.calls == 4


def test_custody_failure_after_encoder_return_retains_state_and_stops():
    writer, model = MemoryJournal(), ToyModel().eval()
    def custody_check():
        if model.encoder_calls:
            raise RuntimeError('synthetic custody drift')

    with pytest.raises(RuntimeError, match='custody drift'):
        _run_fabricated(model, ToyTokenizer(), fabricated_groups(), writer,
                        width=2, vocab_size=8, custody_check=custody_check)
    assert model.encoder_calls == 1 and model.calls == 0
    assert writer.terminal['status'] == 'incomplete'
    rejected = next(row['detail'] for row in writer.events if row['event'] == 'component_rejected')
    assert rejected['return_observed'] is True
    assert rejected['retained_prefix']['bytes'] == 4 * 2 * 4


def test_custody_failure_after_decoder_return_retains_logits_and_stops(tmp_path):
    model, writer = ToyModel().eval(), journal(tmp_path)
    def custody_check():
        if model.calls:
            raise KeyboardInterrupt('synthetic custody interruption')

    runner = _EvaluationRunner(model, ToyTokenizer(), writer, width=2, vocab_size=8,
                               test_only=True, custody_check=custody_check)
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    with pytest.raises(KeyboardInterrupt, match='custody interruption'):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'},
                        _label_targets(ToyTokenizer(), vocab_size=8))
    events = [json.loads(line) for line in (writer.directory / 'events.jsonl').read_text().splitlines()]
    rejected = next(item['detail'] for item in events if item['event'] == 'component_rejected')
    assert rejected['return_observed'] is True
    assert (writer.directory / rejected['retained_prefix']['path']).read_bytes()
    with pytest.raises(RuntimeError, match='incomplete'):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'},
                        _label_targets(ToyTokenizer(), vocab_size=8))
    assert model.calls == 1


@pytest.mark.parametrize('changed', ['input_ids', 'attention_mask'])
def test_encoder_rejects_equal_values_with_mutated_input_dtype(changed, tmp_path):
    class MutatingModel(ToyModel):
        def encoder(self, **kwargs):
            result = super().encoder(**kwargs)
            kwargs[changed].data = kwargs[changed].to(dtype=torch.float32)
            return result

    model, writer = MutatingModel().eval(), journal(tmp_path)
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    with pytest.raises(RuntimeError, match='mutated'):
        _encode_case(model, case, writer, 2, lambda: None)
    rows = [json.loads(line) for line in (writer.directory / 'events.jsonl').read_text().splitlines()]
    rejected = next(row['detail'] for row in rows if row['event'] == 'component_rejected')
    assert rejected['return_observed'] is True
    assert (writer.directory / rejected['state']['path']).is_file()
    assert model.encoder_calls == 1 and writer.active_component is not None


@pytest.mark.parametrize('changed', ['attention_mask', 'decoder_input_ids', 'decoder_attention_mask'])
def test_decoder_rejects_equal_values_with_mutated_input_dtype(changed, tmp_path):
    class MutatingModel(ToyModel):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            kwargs[changed].data = kwargs[changed].to(dtype=torch.float32)
            return result

    model, writer = MutatingModel().eval(), journal(tmp_path)
    runner = _EvaluationRunner(model, ToyTokenizer(), writer, width=2, vocab_size=8, test_only=True)
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    with pytest.raises(RuntimeError, match='mutated'):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'},
                        _label_targets(ToyTokenizer(), vocab_size=8))
    rows = [json.loads(line) for line in (writer.directory / 'events.jsonl').read_text().splitlines()]
    rejected = next(row['detail'] for row in rows if row['event'] == 'component_rejected')
    assert (writer.directory / rejected['logits']['path']).is_file()
    assert model.calls == 1 and writer.active_component is not None


def test_decoder_rejects_encoder_state_reshaped_without_changing_bytes(tmp_path):
    class MutatingModel(ToyModel):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            value = kwargs['encoder_outputs'][0]
            value.data = value.reshape(-1)
            return result

    model, writer = MutatingModel().eval(), journal(tmp_path)
    runner = _EvaluationRunner(model, ToyTokenizer(), writer, width=2, vocab_size=8, test_only=True)
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [1, 1, 1, 1]}
    with pytest.raises(RuntimeError, match='encoder state mutated'):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'},
                        _label_targets(ToyTokenizer(), vocab_size=8))
    rows = [json.loads(line) for line in (writer.directory / 'events.jsonl').read_text().splitlines()]
    rejected = next(row['detail'] for row in rows if row['event'] == 'component_rejected')
    assert (writer.directory / rejected['logits']['path']).is_file()
    assert rejected['observed_state']['shape'] == [8]
    assert model.calls == 1 and writer.active_component is not None


def test_live_entry_rejects_small_journal_before_model_work(tmp_path):
    model, writer = ToyModel().eval(), journal(tmp_path)
    with pytest.raises(Exception, match='protocol allocation'):
        run_cpu_core(model, ToyTokenizer(), fabricated_groups(), writer, custody_check=lambda: None)
    assert model.encoder_calls == model.calls == 0
    assert json.loads((writer.directory / 'terminal.json').read_text())['status'] == 'incomplete'


def test_live_runtime_rejects_contradictory_or_incomplete_observations():
    runtime = {'device': 'cpu', 'num_threads': 1, 'num_interop_threads': 1,
               'deterministic': True, 'deterministic_warn_only': False,
               'default_dtype': 'torch.float32', 'float32_matmul_precision': 'highest',
               'cpu_autocast': False, 'attention_implementation': 'eager',
               'config_use_cache': False, 'hidden_width': 512, 'vocabulary': 32128,
               'decoder_start_id': 0, 'eos_id': 1,
               'environment': {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1',
                               'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1',
                               'TRANSFORMERS_OFFLINE': '1'}}
    _require_live_runtime(runtime)
    for name, value in [('num_threads', True), ('deterministic', False),
                        ('cpu_autocast', True), ('config_use_cache', True),
                        ('attention_implementation', None)]:
        altered = copy.deepcopy(runtime)
        altered[name] = value
        with pytest.raises(RuntimeError, match='runtime'):
            _require_live_runtime(altered)
    altered = copy.deepcopy(runtime)
    del altered['environment']['HF_HUB_OFFLINE']
    with pytest.raises(RuntimeError, match='runtime'):
        _require_live_runtime(altered)


class MemoryJournal:
    """Test-only sink; disk durability is checked independently above."""
    def __init__(self):
        self.events, self.blobs, self.component_count = [], {}, 0
        self.active_component = None
        self.remaining_component_bytes = 0
        self.terminal = None

    def record(self, event, detail):
        self.events.append({'event': event, 'detail': detail})

    def begin_component(self, identity, *, reserve_bytes):
        assert self.active_component is None
        self.active_component = identity
        self.remaining_component_bytes = reserve_bytes
        self.record('component_start', identity)
        self.component_count += 1

    def tensor(self, shape, raw):
        path = f'tensors/{len(self.blobs):06d}.f32le'
        ref = {'path': path, 'shape': list(shape), 'bytes': len(raw),
               'sha256': hashlib.sha256(raw).hexdigest(), 'dtype': 'float32-le'}
        self.blobs[path] = raw
        if self.active_component is not None:
            self.remaining_component_bytes -= len(raw)
            assert self.remaining_component_bytes >= 0
        return ref

    def read_tensor(self, ref):
        raw = self.blobs[ref['path']]
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
        return raw

    def finish_component(self, returned):
        self.record('component_return', returned)
        self.active_component = None

    def finish(self, status, detail):
        self.terminal = {'status': status, 'detail': detail, 'active': self.active_component}


def fabricated_groups():
    groups = []
    for unit in range(4):
        for instruction in range(2):
            cases = {}
            for side, token in [('P1', 2), ('N1', 3), ('P2', 4), ('N2', 5)]:
                cases[side] = {'side': side, 'input_ids': [6, 5, token, 1],
                               'attention_mask': [1, 1, 1, 1],
                               'special_tokens_mask': [0, 0, 0, 1],
                               'adjective_index': 2, 'subject_index': 1}
            groups.append({'unit': unit, 'instruction': instruction, 'cases': cases})
    return groups


def test_full_fabricated_schedule_preserves_controls_and_fixed_denominator():
    writer, model = MemoryJournal(), ToyModel().eval()
    result = _run_fabricated(model, ToyTokenizer(), fabricated_groups(), writer,
                             width=2, vocab_size=8)
    rows = [event['detail'] for event in writer.events if event['event'] == 'evaluation_return']
    kinds = [row['action']['kind'] for row in rows]
    assert len(rows) == 484
    assert [kinds.count(name) for name in ('clean', 'passthrough', 'self', 'full_swap',
                                            'matched', 'natural', 'repeat')] == [32, 32, 96, 32, 256, 32, 4]
    assert model.encoder_calls == 36
    assert model.calls == 484 * 4
    assert len([event for event in writer.events if event['event'] == 'group_geometry']) == 8
    assert writer.terminal['status'] == 'complete_candidate'
    assert result['primary']['total_units'] == 4


def test_unsupported_baseline_keeps_controls_and_skips_both_doses():
    writer, model = MemoryJournal(), ToyModel().eval()
    model.force_negative = True
    result = _run_fabricated(model, ToyTokenizer(), fabricated_groups(), writer,
                             width=2, vocab_size=8)
    rows = [event['detail'] for event in writer.events if event['event'] == 'evaluation_return']
    kinds = [row['action']['kind'] for row in rows]
    assert len(rows) == 196
    assert kinds.count('matched') == kinds.count('natural') == 0
    assert kinds.count('full_swap') == 32 and model.encoder_calls == 36
    assert result['primary'] == {'status': 'threshold-not-met', 'passed_units': 0, 'total_units': 4}
    assert all(group['status'] == 'baseline-unsupported' for pair in result['groups'] for group in pair)


def test_component_failure_leaves_incomplete_and_no_further_forwards():
    writer, model = MemoryJournal(), ToyModel().eval()
    model.fail_at = 10
    with pytest.raises(RuntimeError, match='synthetic decoder failure'):
        _run_fabricated(model, ToyTokenizer(), fabricated_groups(), writer, width=2, vocab_size=8)
    assert model.calls == 10
    assert writer.terminal['status'] == 'incomplete'
    assert writer.terminal['active']['kind'] == 'decoder'


def test_complete_fabricated_schedule_uses_the_real_file_journal(tmp_path):
    writer = EvidenceJournal(tmp_path / 'attempt', {'mode': 'fabricated'},
                             limits=Limits(total_bytes=32 * 1024**2, metadata_bytes=16 * 1024**2,
                                           failure_bytes=1024**2, forward_bytes=4096,
                                           max_forwards=2940, free_bytes=0))
    model = ToyModel().eval()
    result = _run_fabricated(model, ToyTokenizer(), fabricated_groups(), writer,
                             width=2, vocab_size=8)
    events = [json.loads(line) for line in (writer.directory / 'events.jsonl').read_text().splitlines()]
    evaluations = [event['detail'] for event in events if event['event'] == 'evaluation_return']
    assert len(evaluations) == 484 and model.encoder_calls == 36
    for event in evaluations:
        for ref in event['all_logit_refs']:
            raw = (writer.directory / ref['path']).read_bytes()
            assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256']
    terminal = json.loads((writer.directory / 'terminal.json').read_text())
    assert terminal['status'] == 'complete_candidate' and result['primary']['total_units'] == 4
    actual_bytes = sum(path.stat().st_size for path in writer.directory.rglob('*') if path.is_file())
    assert actual_bytes == terminal['retained_bytes'] + (writer.directory / 'terminal.json').stat().st_size


def test_boolean_attention_mask_is_rejected_before_any_forward():
    groups, writer, model = fabricated_groups(), MemoryJournal(), ToyModel().eval()
    groups[0]['cases']['P1']['attention_mask'][0] = True
    with pytest.raises(ValueError, match='selected case'):
        _run_fabricated(model, ToyTokenizer(), groups, writer, width=2, vocab_size=8)
    assert model.calls == model.encoder_calls == 0
    assert writer.terminal['status'] == 'incomplete'


def test_direct_evaluator_rejects_boolean_mask_before_any_forward(tmp_path):
    model, tokenizer = ToyModel().eval(), ToyTokenizer()
    runner = _EvaluationRunner(model, tokenizer, journal(tmp_path), width=2, vocab_size=8, test_only=True)
    state = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 0.]]])
    case = {'side': 'P1', 'input_ids': [4, 5, 2, 1], 'attention_mask': [True, 1, 1, 1]}
    with pytest.raises(ValueError, match='evaluation inputs'):
        runner.evaluate(case, state, state.clone(), {'kind': 'clean'}, _label_targets(tokenizer, vocab_size=8))
    assert model.calls == 0


def test_training_child_module_is_rejected_before_any_forward(tmp_path):
    model = ToyModel().eval()
    model.dropout.train()
    with pytest.raises(ValueError, match='evaluation configuration'):
        _EvaluationRunner(model, ToyTokenizer(), journal(tmp_path), width=2, vocab_size=8, test_only=True)
    assert model.calls == 0


def fabricated_selection():
    groups = fabricated_groups()
    screens = []
    for unit in range(12):
        accepted = unit < 4
        instructions = []
        for instruction in range(2):
            cases = []
            for case in groups[(unit % 4) * 2 + instruction]['cases'].values():
                cases.append(dict(case, text='abc', offset_mapping=[[0, 0]] * 4))
            instructions.append({'instruction_index': instruction, 'cases': cases,
                                 'source_comparisons': [{'source_index': 0, 'comparisons': 4,
                                                         'conflict': False}],
                                 'eligible': accepted,
                                 'failure_reasons': [] if accepted else ['semantic-rejected']})
        screens.append({'unit_id': f'u{unit:02d}', 'semantic_ok': accepted,
                        'lexical_exclusion_ok': True, 'instructions': instructions})
    opposite = {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
    same = {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}
    return {'schema': 'flan-sentiment-private-selection-v1', 'pool_commitment': 'b' * 64,
            'protocol_sha256': '83d666bc67af0951d21eceaa8efb8e7cb16f53a40aefdb72eff03f8b57bd7813',
            'runner_revision': 'a' * 40,
            'asset_manifest_sha256': 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083',
            'exposure_sources': [{'path': 'fixture/source.json', 'revision': 'c' * 40, 'sha256': 'd' * 64}],
            'unit_screens': screens, 'selected_units': ['u00', 'u01', 'u02', 'u03'],
            'donor_mapping': [{'unit_id': f'u{i:02d}', 'opposite': opposite.copy(), 'same': same.copy()}
                              for i in range(4)]}


def test_selection_adapter_binds_first_four_and_reciprocal_donors():
    selection = fabricated_selection()
    groups = selection_to_groups(selection, expected_runner_revision='a' * 40, vocab_size=8,
                                 special_ids={0, 1, 7})
    assert [(x['unit'], x['instruction']) for x in groups] == [(u, i) for u in range(4) for i in range(2)]
    selection['unit_screens'][0]['instructions'][0]['cases'][0]['input_ids'][0] = 7
    assert groups[0]['cases']['P1']['input_ids'][0] == 6


@pytest.mark.parametrize('mutation', ['wrong-order', 'wrong-donor', 'boolean-mask', 'wrong-revision'])
def test_selection_adapter_rejects_mismatches(mutation):
    selection = fabricated_selection()
    if mutation == 'wrong-order':
        selection['selected_units'][:2] = reversed(selection['selected_units'][:2])
    elif mutation == 'wrong-donor':
        selection['donor_mapping'][0]['same']['P1'] = 'N1'
    elif mutation == 'boolean-mask':
        selection['unit_screens'][0]['instructions'][0]['cases'][0]['attention_mask'][0] = True
    else:
        selection['runner_revision'] = 'e' * 40
    with pytest.raises(ValueError):
        selection_to_groups(selection, expected_runner_revision='a' * 40, vocab_size=8, special_ids={0, 1, 7})


def test_fewer_than_four_selected_yields_no_forward_groups():
    selection = fabricated_selection()
    screen = selection['unit_screens'][3]
    screen['semantic_ok'] = False
    for instruction in screen['instructions']:
        instruction['eligible'] = False
        instruction['failure_reasons'] = ['semantic-rejected']
    selection['selected_units'] = selection['selected_units'][:3]
    selection['donor_mapping'] = selection['donor_mapping'][:3]
    assert selection_to_groups(selection, expected_runner_revision='a' * 40, vocab_size=8,
                               special_ids={0, 1, 7}) == ()


def test_published_selector_output_adapts_without_a_model():
    from looking_glass import flan_t5_sentiment_selector_v1 as selector

    class FakeFastTokenizer:
        eos_token_id, pad_token_id = 1, 0
        all_special_ids = [0, 1, 2]

        def __call__(self, text, **kwargs):
            spans = list(re.finditer(r'[A-Za-z]+|[^\w\s]', text))
            ids, offsets = [], []
            for match in spans:
                ids.append(3 + int.from_bytes(hashlib.sha256(match.group().encode()).digest()[:4], 'big') % 30000)
                start = match.start()
                offsets.append([start - 1 if start and text[start - 1] == ' ' else start, match.end()])
            return {'input_ids': ids + [1], 'attention_mask': [1] * (len(ids) + 1),
                    'special_tokens_mask': [0] * len(ids) + [1], 'offset_mapping': offsets + [[0, 0]]}

    source = {'path': 'fixture/source.json', 'revision': 'c' * 40, 'sha256': 'd' * 64}
    units = [{'id': f'u{i:02d}', 'subject': f'subject{letter}',
              'adjectives': {'P1': f'brava{letter}', 'N1': f'grima{letter}',
                             'P2': f'cheera{letter}', 'N2': f'dreada{letter}'},
              'semantic_ok': i != 1, 'semantic_reason': 'ambiguous' if i == 1 else ''}
             for i, letter in enumerate('abcdefghijkl')]
    pool = {'schema': 'flan-sentiment-private-pool-v1',
            'protocol_sha256': selector.PROTOCOL_SHA256, 'runner_revision': 'a' * 40,
            'asset_manifest_sha256': selector.ASSET_MANIFEST_SHA256,
            'exposure_sources': [source], 'units': units}
    selection = selector.select(pool, FakeFastTokenizer(),
                                [{'descriptor': source, 'rows': [], 'words': []}], 'b' * 64)
    assert selection['selected_units'] == ['u00', 'u02', 'u03', 'u04']
    groups = selection_to_groups(selection, expected_runner_revision='a' * 40,
                                 vocab_size=32128, special_ids={0, 1, 2})
    assert len(groups) == 8
    assert groups[2]['cases']['P1'] == selection['unit_screens'][2]['instructions'][0]['cases'][0]
