"""Independent local-only tokenizer and status-rule review; no model is loaded."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys

work = Path(__file__).resolve().parent
repo = work/'teams/flan_t5_equal_length_response'
asset_dir = work/'t5-cache/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
plan = (repo/'docs/FLAN_T5_EQUAL_LENGTH_SELECTION_V1.md').read_bytes()
screen_raw = (repo/'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json').read_bytes()
cases_raw = (repo/'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json').read_bytes()
assert hashlib.sha256(plan).hexdigest() == '1fa773268a2bbfd73979c2ce9f83f6a1d821f6607c37c12460b2368b2ca0668c'
assert hashlib.sha256(screen_raw).hexdigest() == 'adabba92e451fb0c27415b8ec67b67134b59bb5aac293b3c056936cfe872c0a2'
assert hashlib.sha256(cases_raw).hexdigest() == '8d1ea3b124085ca8817dc80044dca6fd1e897bda7294510ad99e419c01e7708b'
asset_raw = (asset_dir/'asset-manifest.json').read_bytes()
assert hashlib.sha256(asset_raw).hexdigest() == 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
asset_manifest = json.loads(asset_raw)
for name in ('config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json'):
    raw = (asset_dir/name).read_bytes()
    assert len(raw) == asset_manifest['files'][name]['bytes']
    assert hashlib.sha256(raw).hexdigest() == asset_manifest['files'][name]['sha256']
assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('', '-1')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True, use_fast=True, trust_remote_code=False)
assert type(tokenizer).__name__ == 'T5TokenizerFast' and tokenizer.pad_token_id == 0 and tokenizer.eos_token_id == 1

screen, manifest = json.loads(screen_raw), json.loads(cases_raw)
pattern = re.compile(r'^\|\s*(\d+)\s*\|\s*(.*?)\s*/\s*(.*?)\s*\|\s*(.*?)\s*/\s*(.*?)\s*\|\s*$')
table = []
for line in plan.decode('utf-8').splitlines():
    match = pattern.fullmatch(line)
    if match:
        number, first, first_target, second, second_target = match.groups()
        table.append((int(number), first, first_target, second, second_target))
assert len(table) == len(screen['candidates']) == 16
assert [row[0] for row in table] == list(range(16))


def verify_tokenized(row, text):
    assert row['text'] == text
    tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    assert row['input_ids'] == tokenized['input_ids']
    assert row['attention_mask'] == tokenized['attention_mask']
    assert row['special_tokens_mask'] == tokenized['special_tokens_mask']
    assert row['offsets'] == [list(value) for value in tokenized['offset_mapping']]
    assert row['token_strings'] == tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    return tokenized


def eos_ok(row):
    return bool(row['input_ids'] and row['input_ids'][-1] == 1 and row['input_ids'].count(1) == 1
                and row['attention_mask'] == [1]*len(row['input_ids']))


computed_eligible = []
templates = [('translate', 'Translate English to German: {}', 'Translate English to German:', 5),
             ('question', 'What is the German word for {}?', 'What is the German word for', 7)]
for planned, candidate in zip(table, screen['candidates']):
    number, first, first_target, second, second_target = planned
    assert (candidate['candidate_index'], candidate['first_word'], candidate['first_target'],
            candidate['second_word'], candidate['second_target']) == planned
    per_template = []
    assert len(candidate['cases']) == 2
    for case, (template, form, context, punctuation) in zip(candidate['cases'], templates):
        assert case['template_id'] == template
        assert (case['word_position'], case['instruction_position'], case['context_position']) == (6, 0, punctuation)
        assert len(case['source']) == len(case['targets']) == 2
        for row, text in zip(case['source'], (form.format(first), form.format(second))):
            verify_tokenized(row, text)
        verify_tokenized(case['neutral'], context)
        for row, text in zip(case['targets'], (first_target, second_target)):
            verify_tokenized(row, text)
        a, b = case['source']
        source_ok = (all(eos_ok(row) for row in (a, b, case['neutral']))
                     and len(a['input_ids']) == len(b['input_ids']) <= 32
                     and a['attention_mask'] == b['attention_mask']
                     and a['special_tokens_mask'] == b['special_tokens_mask']
                     and [i for i, (x, y) in enumerate(zip(a['input_ids'], b['input_ids'])) if x != y] == [6]
                     and len(set((6, 0, punctuation))) == 3)
        for row, word in zip((a, b), (first, second)):
            prefix = form.split('{}')[0]
            begin, end = len(prefix), len(prefix)+len(word)
            word_positions = [i for i, (x, y) in enumerate(row['offsets']) if x < end and y > begin]
            source_ok = (source_ok and row['word'] == word and row['word_positions'] == word_positions == [6]
                         and all(pos < len(row['input_ids']) and not row['special_tokens_mask'][pos]
                                 for pos in (6, 0, punctuation)))
            if len(row['offsets']) > 6:
                x, y = row['offsets'][6]
                source_ok = source_ok and row['text'][x:y].strip() == word
            else:
                source_ok = False
        if len(a['input_ids']) > punctuation and len(b['input_ids']) > punctuation:
            source_ok = source_ok and all(a['input_ids'][i] == b['input_ids'][i] for i in (0, punctuation))
        else:
            source_ok = False
        targets = case['targets']
        target_ok = (first_target != second_target and targets[0]['input_ids'] != targets[1]['input_ids'])
        for row in targets:
            valid = len(row['input_ids']) == 2 and eos_ok(row)
            if valid:
                x, y = row['offsets'][0]
                valid = (row['special_tokens_mask'][0] == 0 and row['input_ids'][0] != tokenizer.unk_token_id
                         and row['text'][x:y].strip() == row['text'])
            target_ok = target_ok and valid
        eligible = bool(source_ok and target_ok)
        assert case['alignment_eligible'] is eligible
        assert bool(case['eligibility_reasons']) is not eligible
        per_template.append(eligible)
    ok = all(per_template)
    assert candidate['eligible'] is ok
    if ok:
        computed_eligible.append(number)
assert len(computed_eligible) == 9
selected = computed_eligible[:6]
assert selected == screen['selected_candidate_indices'] == [0, 1, 2, 3, 5, 7]
assert len(manifest['cases']) == 12
for i, row in enumerate(manifest['cases']):
    expected = screen['candidates'][selected[i//2]]['cases'][i%2]
    assert {k: v for k, v in row.items() if k not in ('candidate_index', 'pair_id', 'case_id')} == expected
    assert row['candidate_index'] == selected[i//2] and row['pair_id'] == i//2
    assert row['case_id'] == f"pair{i//2}-{'translate' if i%2 == 0 else 'question'}"
    assert all(len(item['input_ids']) <= 9 for item in (row['neutral'], *row['source']))

sys.path.insert(0, str(repo))
from looking_glass.flan_t5_equal_length_response import aggregate_decision, pair_decisions, fixed_evidence_bound
representatives = {'positive': ('positive', 'positive'), 'negative': ('positive', 'negative'),
                   'unevaluable': ('negative', 'baseline-unsupported')}
for choices in itertools.product(representatives, repeat=6):
    cells = [{'pair_id': i, 'template_id': ('translate', 'question')[j], 'status': status}
             for i, pair in enumerate(choices) for j, status in enumerate(representatives[pair])]
    assert pair_decisions(cells) == list(choices)
    p, n = choices.count('positive'), choices.count('negative')
    expected = 'positive' if p >= 3 and p > n else 'negative' if n >= 3 and n > p else 'inconclusive'
    assert aggregate_decision(cells) == expected
statuses = ('positive', 'negative', 'alignment-ineligible', 'baseline-unsupported', 'dose-ineligible', 'dose-unmatchable')
for left, right in itertools.product(statuses, repeat=2):
    cells = [{'pair_id': i//2, 'template_id': ('translate', 'question')[i%2], 'status': 'positive'}
             for i in range(12)]
    cells[0]['status'], cells[1]['status'] = left, right
    expected = ('unevaluable' if left not in ('positive', 'negative') or right not in ('positive', 'negative')
                else 'positive' if left == right == 'positive' else 'negative')
    assert pair_decisions(cells)[0] == expected
bound = fixed_evidence_bound()
assert bound['total_bytes'] < bound['limit_bytes'] == 512*1024*1024
report = {'schema': 'independent-flant5-equal-length-tokenizer-review-v1',
          'scope': 'tokenizer-only and fabricated status-rule checks; no model weights loaded or forwards',
          'plan_sha256': hashlib.sha256(plan).hexdigest(), 'screen_sha256': hashlib.sha256(screen_raw).hexdigest(),
          'cases_sha256': hashlib.sha256(cases_raw).hexdigest(), 'candidates': 16,
          'eligible_indices': computed_eligible, 'selected_indices': selected,
          'cases': len(manifest['cases']), 'aggregate_combinations_checked': 3**6,
          'pair_status_combinations_checked': len(statuses)**2, 'fixed_evidence_bound': bound}
raw = (json.dumps(report, sort_keys=True, separators=(',', ':'))+'\n').encode()
destination = work/'flan-t5-equal-length-independent-tokenizer-review.json'
with destination.open('xb') as out:
    out.write(raw)
print(json.dumps({'report': str(destination), 'sha256': hashlib.sha256(raw).hexdigest(), **report}, sort_keys=True))
