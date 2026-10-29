"""Independent, no-checkpoint semantic probe for the fabricated FLAN runner.

The tiny decoder has an analytically known label-score margin. No prospective
case, pretrained checkpoint, tokenizer asset, or GPU is read by this test.
"""
import hashlib
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import torch


class Tokenizer:
    all_special_ids = [0, 1, 7]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {'positive': [2], 'negative': [3]}[text]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return ''.join({2: 'positive', 3: 'negative'}.get(x, '?') for x in ids)


VALUES = {'P1': 3.0, 'N1': -3.0, 'P2': 2.0, 'N2': -2.0}
TOKEN_VALUE = {2: 3.0, 3: -3.0, 4: 2.0, 5: -2.0}
OPPOSITE = {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
SAME = {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}


class LinearDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(d_model=2, vocab_size=8,
                                      decoder_start_token_id=0, eos_token_id=1)
        self.encodes = self.decodes = 0

    def get_encoder(self):
        return self.encoder

    def encoder(self, *, input_ids, attention_mask, return_dict):
        assert return_dict and attention_mask.tolist() == [[1, 1, 1, 1]]
        self.encodes += 1
        v = TOKEN_VALUE[int(input_ids[0, 2])]
        state = torch.tensor([[[v / 8, 0], [v / 4, 0], [v, 0], [0, 0]]], dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=state)

    def forward(self, *, encoder_outputs, attention_mask, decoder_input_ids,
                decoder_attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        assert attention_mask.tolist() == [[1, 1, 1, 1]]
        assert decoder_attention_mask.tolist() == [[1] * decoder_input_ids.shape[1]]
        self.decodes += 1
        state = encoder_outputs[0]
        z = state[0, 2, 0] + 0.25 * state[0, 0, 0] - 0.125 * state[0, 1, 0]
        logits = torch.full((1, decoder_input_ids.shape[1], 8), -10.0)
        logits[0, 0, 2], logits[0, 0, 3] = z / 2, -z / 2
        if decoder_input_ids.shape[1] > 1:
            logits[0, 1:, 1] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=None)


class SkipDecoder(LinearDecoder):
    """A supported clean baseline with either zero or unlandable dose."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def encoder(self, *, input_ids, attention_mask, return_dict):
        assert return_dict and attention_mask.tolist() == [[1, 1, 1, 1]]
        self.encodes += 1
        token = int(input_ids[0, 2])
        v = TOKEN_VALUE[token]
        instruction = {2: 0., 3: 1e-5, 4: 1., 5: 1.00001}[token] if self.mode == 'rounding' else 0.
        adjective = {2: 1e8, 3: 1e8 + 8, 4: 1e8 + 16, 5: 1e8 + 24}[token]
        state = torch.tensor([[[instruction, 0], [v, 0], [adjective, v], [0, 0]]],
                             dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=state)

    def forward(self, *, encoder_outputs, attention_mask, decoder_input_ids,
                decoder_attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        self.decodes += 1
        z = encoder_outputs[0][0, 2, 1]
        logits = torch.full((1, decoder_input_ids.shape[1], 8), -10.0)
        logits[0, 0, 2], logits[0, 0, 3] = z / 2, -z / 2
        if decoder_input_ids.shape[1] > 1:
            logits[0, 1:, 1] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=None)


class PassingDecoder(LinearDecoder):
    """Opposite donors vary the scoring coordinate; same donors vary another."""
    def encoder(self, *, input_ids, attention_mask, return_dict):
        assert return_dict and attention_mask.tolist() == [[1, 1, 1, 1]]
        self.encodes += 1
        side = {2: 'P1', 3: 'N1', 4: 'P2', 5: 'N2'}[int(input_ids[0, 2])]
        s = 1. if side.startswith('P') else -1.
        t = 1. if side.endswith('1') else -1.
        state = torch.tensor([[[s, 0], [2 * s, 0], [3 * s, t], [0, 0]]],
                             dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=state)


class MemoryJournal:
    def __init__(self):
        self.events, self.blobs, self.active = [], {}, None
        self.terminal = None
        self.remaining_component_bytes = 0
        self.component_count = 0

    def record(self, event, detail):
        self.events.append({'event': event, 'detail': detail})

    def begin_component(self, identity, *, reserve_bytes):
        assert self.active is None
        self.active = identity
        self.remaining_component_bytes = reserve_bytes
        self.component_count += 1
        self.record('component_start', identity)

    def tensor(self, shape, raw):
        path = f'tensors/{len(self.blobs):06}.f32le'
        ref = {'path': path, 'shape': list(shape), 'bytes': len(raw),
               'sha256': hashlib.sha256(raw).hexdigest(), 'dtype': 'float32-le'}
        self.blobs[path] = raw
        if self.active is not None:
            self.remaining_component_bytes -= len(raw)
            assert self.remaining_component_bytes >= 0
        return ref

    def read_tensor(self, ref):
        raw = self.blobs[ref['path']]
        assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256']
        return raw

    def finish_component(self, returned):
        assert self.active is not None
        self.record('component_return', returned)
        self.active = None
        self.remaining_component_bytes = 0

    def finish(self, status, detail):
        self.terminal = {'status': status, 'detail': detail, 'active': self.active}


def groups():
    return [
        {'unit': unit, 'instruction': instruction,
         'cases': {side: {'side': side, 'input_ids': [6, 5, token, 1],
                          'attention_mask': [1, 1, 1, 1],
                          'special_tokens_mask': [0, 0, 0, 1],
                          'adjective_index': 2, 'subject_index': 1}
                   for side, token in [('P1', 2), ('N1', 3), ('P2', 4), ('N2', 5)]}}
        for unit in range(4) for instruction in range(2)
    ]


def state(side):
    v = VALUES[side]
    return [v / 8, 0, v / 4, 0, v, 0, 0, 0]


def expected_state(action, side):
    original = state(side)
    kind = action['kind']
    if kind == 'full_swap':
        return state(OPPOSITE[side])
    if kind == 'natural':
        original[4] = VALUES[OPPOSITE[side]]
    if kind == 'matched':
        name, dose = action['action'], (0.25 if action['dose'] == 'low' else 0.5)
        donor = SAME[side] if name == 'same_adjective' else OPPOSITE[side]
        offset = {'opposite_adjective': 4, 'same_adjective': 4,
                  'opposite_instruction': 0, 'opposite_subject': 2}[name]
        target = state(donor)[offset]
        direction = 1 if target > original[offset] else -1
        original[offset] += direction * dose
    return original


def z_of(values):
    return values[4] + 0.25 * values[0] - 0.125 * values[2]


def passing_state(side):
    s = 1. if side.startswith('P') else -1.
    t = 1. if side.endswith('1') else -1.
    return [s, 0., 2 * s, 0., 3 * s, t, 0., 0.]


def passing_expected(action, side):
    values = passing_state(side)
    kind = action['kind']
    if kind == 'full_swap':
        return passing_state(OPPOSITE[side])
    if kind == 'natural':
        values[4:6] = passing_state(OPPOSITE[side])[4:6]
    if kind == 'matched':
        name = action['action']
        donor = SAME[side] if name == 'same_adjective' else OPPOSITE[side]
        offset = {'opposite_adjective': 4, 'same_adjective': 4,
                  'opposite_instruction': 0, 'opposite_subject': 2}[name]
        difference = [passing_state(donor)[offset + j] - values[offset + j] for j in (0, 1)]
        distance = math.sqrt(sum(x * x for x in difference))
        dose = 1. if action['dose'] == 'low' else 2.
        for j in (0, 1):
            values[offset + j] += dose * difference[j] / distance
    return values


def close(actual, expected, *, tolerance=2e-6):
    assert math.isfinite(actual) and abs(actual - expected) <= tolerance, (actual, expected)


def run(checkout):
    sys.path.insert(0, str(Path(checkout).resolve()))
    from looking_glass.flan_t5_sentiment_runner import _run_fabricated

    torch.set_num_threads(1)
    model, journal = LinearDecoder().eval(), MemoryJournal()
    result = _run_fabricated(model, Tokenizer(), groups(), journal, width=2, vocab_size=8)
    assert journal.terminal['status'] == 'complete_candidate' and journal.active is None
    assert model.encodes == 36 and model.decodes == 484 * 4
    evaluations = [e['detail'] for e in journal.events if e['event'] == 'evaluation_return']
    assert len(evaluations) == 484
    counts = {}
    for row in evaluations:
        action, side = row['action'], row['side']
        kind = action['kind']
        counts[kind] = counts.get(kind, 0) + 1
        expected = expected_state(action, side)
        landed = journal.read_tensor(row['after'])
        assert landed == struct.pack('<8f', *expected), (kind, side, expected)
        close(row['margin'], z_of(expected))
        assert row['generated_ids'][-1] == 1
        assert row['generated_ids'][0] == (2 if z_of(expected) > 0 else 3)
        if kind == 'matched':
            target = .25 if action['dose'] == 'low' else .5
            close(action['target_dose'], target, tolerance=0)
            close(action['achieved_norm'], target, tolerance=0)
            close(action['cosine'], 1, tolerance=0)
    assert counts == {'clean': 32, 'passthrough': 32, 'self': 96, 'full_swap': 32,
                      'matched': 256, 'natural': 32, 'repeat': 4}
    assert result['primary'] == {'status': 'threshold-not-met', 'passed_units': 0, 'total_units': 4}
    for pair in result['groups']:
        for group in pair:
            assert group['status'] == 'measured' and group['passes'] is False
            close(group['A'], .5)
            close(group['B'], .5)
            assert group['positive_count'] == 4
            close(group['control_means']['same_adjective'], .5)
            close(group['control_means']['opposite_instruction'], .125)
            close(group['control_means']['opposite_subject'], .0625)
            for side in VALUES:
                close(group['movements']['opposite_adjective'][side], .5)
                close(group['movements']['opposite_instruction'][side], .125)
                close(group['movements']['opposite_subject'][side], -.0625)
                close(group['movements']['same_adjective'][side],
                      .5 if side in ('P1', 'N1') else -.5)
                close(group['low_dose']['movements']['opposite_adjective'][side], .25)

    model, journal = PassingDecoder().eval(), MemoryJournal()
    passing = _run_fabricated(model, Tokenizer(), groups(), journal, width=2, vocab_size=8)
    evals = [e['detail'] for e in journal.events if e['event'] == 'evaluation_return']
    assert len(evals) == 484 and model.encodes == 36 and model.decodes == 484 * 4
    for row in evals:
        expected = passing_expected(row['action'], row['side'])
        assert journal.read_tensor(row['after']) == struct.pack('<8f', *expected)
        close(row['margin'], z_of(expected))
    assert passing['primary'] == {'status': 'supported', 'passed_units': 4, 'total_units': 4}
    for pair in passing['groups']:
        for group in pair:
            assert group['passes'] is True and group['positive_count'] == 4
            close(group['A'], 2.)
            close(group['B'], .5)
            close(group['control_means']['same_adjective'], 0.)
            close(group['control_means']['opposite_instruction'], .5)
            close(group['control_means']['opposite_subject'], .25)

    for mode, reason in [('zero', 'dose-ineligible'), ('rounding', 'dose-unmatchable')]:
        model, journal = SkipDecoder(mode).eval(), MemoryJournal()
        skipped_result = _run_fabricated(model, Tokenizer(), groups(), journal, width=2, vocab_size=8)
        evals = [e['detail'] for e in journal.events if e['event'] == 'evaluation_return']
        skips = [e['detail'] for e in journal.events if e['event'] == 'action_skipped']
        assert len(evals) == 196 and len(skips) == 288
        assert all(s['reason'] == reason for s in skips)
        assert sum(e['action']['kind'] == 'full_swap' for e in evals) == 32
        assert sum(e['action']['kind'] == 'self' for e in evals) == 96
        assert model.encodes == 36 and model.decodes == 196 * 4
        assert all(g['status'] == reason for pair in skipped_result['groups'] for g in pair)
        assert skipped_result['primary'] == {'status': 'threshold-not-met',
                                             'passed_units': 0, 'total_units': 4}
    print('PASS: two 484-evaluation fake runs; patched state bits, scores, donor signs, '
          'active-control absolute means, two doses, supported and failing group decisions; '
          'zero/unmatchable dose retains controls and skips both action classes')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: review_flan_sentiment_linear_toy.py CHECKOUT')
    run(sys.argv[1])
