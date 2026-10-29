import pytest
import torch
import json

from looking_glass import flan_t5_sentiment_screen as study


def test_screen_is_fixed_balanced_and_covers_all_crosses():
    cases = study.build_cases()
    assert len(cases) == 64
    assert [case['index'] for case in cases] == list(range(64))
    assert len({case['text'] for case in cases}) == 64
    assert sum(case['expected'] == 'positive' for case in cases) == 32
    assert sum(case['expected'] == 'negative' for case in cases) == 32
    assert cases[0]['text'] == ('Classify the sentiment of this review as positive or negative: '
                                'The book was wonderful.')
    assert cases[32]['text'] == 'Is this review positive or negative? The book was wonderful.'
    assert cases[-1]['text'] == 'Is this review positive or negative? The service was disappointing.'


def test_greedy_uses_raw_prefix_logits_lowest_id_ties_and_stops_only_on_eos():
    observed = []
    def forward(prefix):
        observed.append(prefix)
        if len(prefix) == 1:
            return torch.tensor([0., -1., 2., 2., -3.])
        return torch.tensor([0., 3., -1., -2., -3.])
    steps = study.greedy(forward, vocab_size=5)
    assert observed == [(0,), (0, 2)]
    assert [step['emitted'] for step in steps] == [2, 1]
    assert torch.equal(steps[0]['logits'], torch.tensor([0., -1., 2., 2., -3.]))


def test_greedy_truncation_and_nonfinite_are_not_silently_accepted():
    steps = study.greedy(lambda prefix: torch.tensor([0., -1., 1.]), vocab_size=3)
    assert [step['emitted'] for step in steps] == [2, 2, 2, 2]
    with pytest.raises(ValueError, match='finite'):
        study.greedy(lambda prefix: torch.tensor([0., float('nan'), 1.]), vocab_size=3)


class Decoder:
    all_special_ids = [0, 1, 9, 10]
    def decode(self, ids, **kwargs):
        assert kwargs == {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False}
        return {2: 'Positive', 3: 'negative', 4: 'positive.'}.get(ids[0], '')


@pytest.mark.parametrize('ids,expected,good', [
    ([2, 1], 'positive', True), ([3, 1], 'negative', True),
    ([4, 1], 'positive', False), ([2], 'positive', False),
    ([2, 9, 1], 'positive', False), ([2, 1, 1], 'positive', False),
    ([2, 0, 1], 'positive', False),
])
def test_strict_label_preserves_format_and_special_token_failures(ids, expected, good):
    result = study.classify_generation(ids, Decoder(), expected)
    assert result['correct'] is good


def test_truncation_is_not_misreported_as_a_special_token():
    result = study.classify_generation([2, 2, 2, 2], Decoder(), 'positive')
    assert result['correct'] is False
    assert result['terminated'] is False
    assert result['no_content_specials'] is True


def test_label_targets_require_distinct_non_special_lexical_sequences():
    class Tokenizer:
        all_special_ids = [0, 1, 9]
        def __init__(self, values):
            self.values = values
        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {'input_ids': self.values[text]}
    assert study.label_targets(Tokenizer({'positive': [2], 'negative': [3, 4]})) == {
        'positive': [2, 1], 'negative': [3, 4, 1]}
    for bad in ({'positive': [], 'negative': [3]},
                {'positive': [2], 'negative': [2]},
                {'positive': [2, 1], 'negative': [3]},
                {'positive': [2], 'negative': [9]}):
        with pytest.raises(ValueError, match='label'):
            study.label_targets(Tokenizer(bad))


def test_teacher_forced_sequence_scores_include_every_token_and_eos():
    seen = []
    def forward(prefix):
        seen.append(prefix)
        return torch.tensor([[0., 0., 2., 1.], [0., 3., 0., 0.]])
    scored = study.teacher_force(forward, [2, 1], vocab_size=4)
    expected = torch.log_softmax(torch.tensor([[0., 0., 2., 1.], [0., 3., 0., 0.]]), dim=-1)
    assert seen == [(0, 2)]
    assert scored['token_logprobs'] == [float(expected[0, 2]), float(expected[1, 1])]
    assert scored['score'] == sum(scored['token_logprobs'])


def test_feasibility_uses_all_pairs_and_subject_gate():
    cases = study.build_cases()
    rows = [{'index': case['index'], 'correct': True} for case in cases]
    report = study.summarize(rows)
    assert report['complete_pairs'] == 32
    assert report['feasible'] is True
    for row, case in zip(rows, cases):
        if case['subject'] == 'book':
            row['correct'] = False
    report = study.summarize(rows)
    assert report['complete_pairs'] == 24
    assert report['subject_pairs']['book'] == 0
    assert report['feasible'] is False
    with pytest.raises(ValueError, match='64'):
        study.summarize(rows[:-1])


def test_case_journal_keeps_prefix_logits_targets_and_actual_eos(tmp_path):
    journal = study.Journal(tmp_path / 'attempt', byte_cap=1_000_000, forward_cap=7)
    case = study.build_cases()[0]
    def encode():
        return torch.tensor([[[0., 1.]]])
    def decode(_encoded, prefix):
        rows = torch.zeros((len(prefix), 5), dtype=torch.float32)
        if len(prefix) == 1:
            rows[-1, 2] = 4
        else:
            rows[-1, 1] = 3
        return rows
    result = study.execute_case(case, Decoder(), {'positive': [2, 1], 'negative': [3, 1]},
                                encode, decode, journal, vocab_size=5, repeat=False)
    assert result['generated_ids'] == [2, 1]
    assert result['classification']['correct'] is True
    assert result['label_scores']['positive']['targets'] == [2, 1]
    assert len(result['generation_refs']) == 2
    events = [json.loads(line) for line in (tmp_path / 'attempt' / 'events.jsonl').read_text().splitlines()]
    assert sum(e['kind'] == 'forward_started' for e in events) == 5
    assert sum(e['kind'] == 'forward_completed' for e in events) == 5
    assert events[-1]['kind'] == 'case_completed'
    assert (tmp_path / 'attempt' / result['generation_refs'][0]['path']).is_file()
    with pytest.raises(FileExistsError):
        study.Journal(tmp_path / 'attempt', byte_cap=1_000_000, forward_cap=7)


def test_failed_case_preserves_completed_forward_and_open_start(tmp_path):
    journal = study.Journal(tmp_path / 'attempt', byte_cap=1_000_000, forward_cap=7)
    def encode():
        return torch.tensor([[[0., 1.]]])
    def decode(_encoded, prefix):
        raise RuntimeError('fabricated forward failure')
    with pytest.raises(RuntimeError, match='fabricated'):
        study.execute_case(study.build_cases()[0], Decoder(),
                           {'positive': [2, 1], 'negative': [3, 1]}, encode, decode,
                           journal, vocab_size=5, repeat=False)
    events = [json.loads(line) for line in (tmp_path / 'attempt' / 'events.jsonl').read_text().splitlines()]
    assert [e['kind'] for e in events] == ['forward_started', 'forward_completed', 'forward_started']
    assert len(list((tmp_path / 'attempt' / 'tensors').iterdir())) == 1


def test_alignment_requires_single_target_token_and_clean_polarity_control():
    cases = study.build_cases()
    token_rows = []
    for case in cases:
        text = case['text']
        begin = text.rfind(case['adjective'])
        token_rows.append({'input_ids': [101, case['pair_index'] * 2 + case['side'] + 10, 1],
                           'attention_mask': [1, 1, 1], 'special_tokens_mask': [0, 0, 1],
                           'offset_mapping': [[0, 1], [begin, begin + len(case['adjective'])], [0, 0]]})
    outcomes = [{'index': case['index'], 'correct': True} for case in cases]
    report = study.alignment_report(cases, token_rows, outcomes)
    assert report['aligned_opposite_pairs'] == 32
    assert report['behaviorally_usable_pairs'] == 32
    outcomes[2]['correct'] = False
    # Its absence is visible but other correctly classified same-label controls remain.
    report = study.alignment_report(cases, token_rows, outcomes)
    assert 2 not in report['cases'][0]['same_polarity_clean_donors']
    assert len(report['cases'][0]['same_polarity_clean_donors']) == 2
    token_rows[1]['offset_mapping'][1][1] -= 1
    report = study.alignment_report(cases, token_rows, outcomes)
    assert report['aligned_opposite_pairs'] == 31
    assert report['cases'][0]['opposite_aligned'] is False
    token_rows[1]['offset_mapping'][1][1] += 1
    token_rows[0]['attention_mask'][1] = 0
    report = study.alignment_report(cases, token_rows, outcomes)
    assert report['cases'][0]['word_position'] is None


def test_complete_66_case_schedule_with_fabricated_logits(tmp_path):
    class Tokenizer(Decoder):
        def __call__(self, text, **kwargs):
            if kwargs.get('add_special_tokens') is False:
                return {'input_ids': {'positive': [2], 'negative': [3]}[text]}
            case = next(case for case in study.build_cases() if case['text'] == text)
            begin = text.rfind(case['adjective'])
            return {'input_ids': [101, 10 + 2 * case['pair_index'] + case['side'], 1],
                    'attention_mask': [1, 1, 1], 'special_tokens_mask': [0, 0, 1],
                    'offset_mapping': [[0, 1], [begin, begin + len(case['adjective'])], [0, 0]]}
    def encode(case, _tokens):
        return torch.tensor([[[float(case['side'])]]])
    def decode(encoded, prefix):
        rows = torch.zeros((len(prefix), 5), dtype=torch.float32)
        rows[-1, (2 if encoded[0, 0, 0] == 0 else 3) if len(prefix) == 1 else 1] = 4
        return rows
    journal = study.Journal(tmp_path / 'attempt', byte_cap=2_000_000, forward_cap=462)
    result = study.run_study(Tokenizer(), encode, decode, journal, vocab_size=5)
    assert result['summary']['complete_pairs'] == 32
    assert result['summary']['feasible'] is True
    assert result['alignment']['behaviorally_usable_pairs'] == 32
    assert result['repeat_indices'] == [0, 32]
    assert journal.forwards == 330
    events = [json.loads(line) for line in (tmp_path / 'attempt' / 'events.jsonl').read_text().splitlines()]
    assert sum(event['kind'] == 'case_completed' for event in events) == 66


def test_decoder_mask_attends_start_and_any_emitted_pad():
    from types import SimpleNamespace
    seen = {}
    def model(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(logits=torch.zeros((1, 3, 5), dtype=torch.float32))
    result = study.decode_with_encoder(model, torch.zeros((1, 2, 3)), (0, 4, 0))
    assert result.shape == (3, 5)
    assert seen['decoder_attention_mask'].tolist() == [[1, 1, 1]]
    assert seen['decoder_input_ids'].tolist() == [[0, 4, 0]]
    assert seen['attention_mask'].tolist() == [[1, 1]]
    assert seen['use_cache'] is False


def test_tiny_untrained_t5_uses_real_encoder_and_decoder_without_cache(tmp_path):
    from transformers import T5Config, T5ForConditionalGeneration
    torch.manual_seed(123)
    config = T5Config(vocab_size=128, d_model=8, d_kv=4, d_ff=16, num_layers=1,
                      num_decoder_layers=1, num_heads=2, dropout_rate=0.0,
                      decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, use_cache=False)
    model = T5ForConditionalGeneration(config).eval()
    def encode():
        return model.encoder(input_ids=torch.tensor([[7, 8, 1]]),
                             attention_mask=torch.ones((1, 3), dtype=torch.long),
                             return_dict=True).last_hidden_state
    def decode(encoded, prefix):
        return study.decode_with_encoder(model, encoded, prefix)
    journal = study.Journal(tmp_path / 'attempt', byte_cap=1_000_000, forward_cap=7)
    with torch.inference_mode():
        result = study.execute_case(study.build_cases()[0], Decoder(),
                                    {'positive': [2, 1], 'negative': [3, 1]},
                                    encode, decode, journal, vocab_size=128, repeat=False)
    assert 1 <= len(result['generated_ids']) <= 4
    assert result['label_scores']['positive']['tensor']['shape'] == [2, 128]
    assert journal.forwards == 1 + len(result['generated_ids']) + 2
