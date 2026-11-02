"""Apply the fixed, tokenizer-only selection plan; never load a model."""

import hashlib
import json
import os
from pathlib import Path
import re

PLAN_SHA = '1fa773268a2bbfd73979c2ce9f83f6a1d821f6607c37c12460b2368b2ca0668c'
ASSET_SHA = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
PLAN = Path('work/t5_length_matched/selection-plan.md')
OUTPUT = Path('work/t5_length_matched')
ASSETS = Path('work/t5-cache/flan-t5-small-' + REVISION)

if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
    raise ValueError('CUDA must be hidden')
if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
    raise ValueError('offline mode is required')
plan_bytes = PLAN.read_bytes()
if hashlib.sha256(plan_bytes).hexdigest() != PLAN_SHA:
    raise ValueError('selection plan differs from the pre-screen pin')
manifest_bytes = (ASSETS / 'asset-manifest.json').read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != ASSET_SHA:
    raise ValueError('asset manifest mismatch')
manifest = json.loads(manifest_bytes)
required = {'config.json', 'generation_config.json', 'model.safetensors',
            'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json'}
if (manifest['repo'] != 'google/flan-t5-small' or manifest['revision'] != REVISION
        or set(manifest['files']) != required):
    raise ValueError('asset identity mismatch')
for filename, expected in manifest['files'].items():
    path = ASSETS / filename
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('asset mismatch: ' + filename)

# Parse the exact pre-screen table rather than maintain a second candidate list.
rows = []
pattern = re.compile(r'^\|\s*(\d+)\s*\|\s*(.*?)\s*/\s*(.*?)\s*\|\s*(.*?)\s*/\s*(.*?)\s*\|\s*$')
for line in plan_bytes.decode('utf-8').splitlines():
    match = pattern.fullmatch(line)
    if match:
        n, first, first_target, second, second_target = match.groups()
        rows.append((int(n), first, first_target, second, second_target))
if len(rows) != 16 or [r[0] for r in rows] != list(range(16)):
    raise ValueError('selection table is not the fixed 16-row order')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(str(ASSETS), local_files_only=True,
                                          use_fast=True, trust_remote_code=False)
if type(tokenizer).__name__ != 'T5TokenizerFast' or tokenizer.eos_token_id != 1 or tokenizer.pad_token_id != 0:
    raise ValueError('tokenizer identity mismatch')
templates = (('translate', 'Translate English to German: {word}', 'Translate English to German:', 5),
             ('question', 'What is the German word for {word}?', 'What is the German word for', 7))


def encode(text):
    value = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    return {'text': text, 'input_ids': value['input_ids'], 'attention_mask': value['attention_mask'],
            'special_tokens_mask': value['special_tokens_mask'],
            'offsets': [list(x) for x in value['offset_mapping']],
            'token_strings': tokenizer.convert_ids_to_tokens(value['input_ids'])}


def eos_and_mask_ok(row):
    ids = row['input_ids']
    return bool(ids and ids[-1] == 1 and ids.count(1) == 1
                and row['attention_mask'] == [1] * len(ids))


def target_reasons(targets):
    reasons = []
    if targets[0]['input_ids'] == targets[1]['input_ids'] or targets[0]['text'] == targets[1]['text']:
        reasons.append('targets-not-distinct')
    for index, row in enumerate(targets):
        ids = row['input_ids']
        if len(ids) != 2 or not eos_and_mask_ok(row):
            reasons.append(f'target-{index}-not-one-lexical-plus-eos')
            continue
        if row['special_tokens_mask'][0] != 0 or ids[0] == tokenizer.unk_token_id:
            reasons.append(f'target-{index}-special-or-unknown')
        start, end = row['offsets'][0]
        if row['text'][start:end].strip() != row['text']:
            reasons.append(f'target-{index}-offset')
    return reasons


def source_case(first, second, template):
    template_id, form, neutral_text, punctuation = template
    prefix, _, _ = form.partition('{word}')
    sources = []
    for word in (first, second):
        text = form.format(word=word)
        start = len(prefix)
        if text[start:start + len(word)] != word:
            raise ValueError('template insertion mismatch')
        row = encode(text)
        row['word'] = word
        row['word_positions'] = [i for i, (a, b) in enumerate(row['offsets'])
                                 if a < start + len(word) and b > start]
        sources.append(row)
    a, b = sources
    neutral = encode(neutral_text)
    reasons = []
    if any(not eos_and_mask_ok(s) for s in sources + [neutral]):
        reasons.append('source-or-context-mask-eos')
    same_shape = (len(a['input_ids']) == len(b['input_ids']) <= 32
                  and a['attention_mask'] == b['attention_mask']
                  and a['special_tokens_mask'] == b['special_tokens_mask'])
    if not same_shape:
        reasons.append('source-shape-or-mask')
    if ([i for i, (x, y) in enumerate(zip(a['input_ids'], b['input_ids'])) if x != y] != [6]
            or a['word_positions'] != [6] or b['word_positions'] != [6]):
        reasons.append('source-word-alignment')
    positions = (6, 0, punctuation)
    if (len(set(positions)) != 3
            or any(p >= len(s['input_ids']) for p in positions for s in sources)
            or any(s['special_tokens_mask'][p] for p in positions for s in sources
                   if p < len(s['input_ids']))):
        reasons.append('position-special-or-missing')
    if (any(p >= len(s['input_ids']) for p in (0, punctuation) for s in sources)
            or any(a['input_ids'][p] != b['input_ids'][p] for p in (0, punctuation)
                   if p < len(a['input_ids']) and p < len(b['input_ids']))):
        reasons.append('control-position-changed')
    for index, row in enumerate(sources):
        if len(row['offsets']) <= 6:
            reasons.append(f'source-{index}-word-offset-missing')
        else:
            start, end = row['offsets'][6]
            if row['text'][start:end].strip() != row['word']:
                reasons.append(f'source-{index}-word-offset')
    return {'template_id': template_id, 'source': sources, 'neutral': neutral,
            'word_position': 6, 'instruction_position': 0, 'context_position': punctuation,
            'eligibility_reasons': reasons}


candidates = []
for number, first, first_target, second, second_target in rows:
    targets = [encode(first_target), encode(second_target)]
    target_errors = target_reasons(targets)
    cases = []
    for template in templates:
        case = source_case(first, second, template)
        case['targets'] = targets
        case['eligibility_reasons'] = target_errors + case['eligibility_reasons']
        case['alignment_eligible'] = not case['eligibility_reasons']
        cases.append(case)
    candidates.append({'candidate_index': number, 'first_word': first, 'second_word': second,
                       'first_target': first_target, 'second_target': second_target,
                       'eligible': all(case['alignment_eligible'] for case in cases), 'cases': cases})

selected = [c for c in candidates if c['eligible']][:6]
complete = len(selected) == 6
screen = {'schema': 'mlg-flan-t5-equal-length-candidate-screen-v1',
          'selection_plan_sha256': PLAN_SHA, 'asset_manifest_sha256': ASSET_SHA,
          'model': 'google/flan-t5-small', 'revision': REVISION,
          'tokenizer_class': type(tokenizer).__name__, 'eos_token_id': 1, 'pad_token_id': 0,
          'status': 'complete' if complete else 'selection-incomplete',
          'selected_candidate_indices': [c['candidate_index'] for c in selected],
          'candidates': candidates}


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + '\n').encode('utf-8')
    with path.open('xb') as f:
        if f.write(raw) != len(raw):
            raise OSError('short evidence write')
    return hashlib.sha256(raw).hexdigest()


screen_hash = write_new(OUTPUT / 'candidate-screen.json', screen)
cases_hash = None
if complete:
    cases = []
    for pair_id, candidate in enumerate(selected):
        for case in candidate['cases']:
            case = dict(case)
            case['case_id'] = f"pair{pair_id}-{case['template_id']}"
            case['pair_id'] = pair_id
            case['candidate_index'] = candidate['candidate_index']
            cases.append(case)
    result = {'schema': 'mlg-flan-t5-equal-length-dose-response-cases-v1',
              'model': 'google/flan-t5-small', 'revision': REVISION,
              'asset_manifest_sha256': ASSET_SHA, 'selection_plan_sha256': PLAN_SHA,
              'candidate_screen_sha256': screen_hash,
              'tokenizer_class': type(tokenizer).__name__, 'eos_token_id': 1, 'pad_token_id': 0,
              'cases': cases}
    cases_hash = write_new(OUTPUT / 'cases.json', result)
print(json.dumps({'status': screen['status'], 'screen_sha256': screen_hash,
                  'cases_sha256': cases_hash,
                  'selected': [{'candidate_index': c['candidate_index'], 'words': [c['first_word'], c['second_word']],
                                'targets': [c['first_target'], c['second_target']],
                                'target_ids': [x['input_ids'] for x in c['cases'][0]['targets']]}
                               for c in selected],
                  'eligible_count': sum(c['eligible'] for c in candidates)}))
