"""Replay locally held-out T5 evidence from saved bytes, never a model.

Usage: python AUDITOR RUN_DIR CUSTODY_DIR SOURCE_CHECKOUT ASSET_DIR
The source checkout must be the recorded frozen commit with its source files
unchanged. This is a saved-evidence consistency audit, not independent proof
of model execution, privacy, random-salt generation or chronology. This code
adapts the author's previous independent saved-byte replay for the public study.
Keep its report and diagnostics private until the protocol's authorized reveal.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys

import numpy as np
import torch

if not __debug__:
    raise RuntimeError('this evidence auditor must run without Python -O')
torch.set_num_threads(1)
assert torch.set_flush_denormal(False)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def same(a, b):
    return a.shape == b.shape and np.array_equal(a.view('<u4'), b.view('<u4'))

def distance(a, b):
    assert a.shape == b.shape
    delta = a.astype(np.float64).reshape(-1) - b.astype(np.float64).reshape(-1)
    return math.sqrt(math.fsum(float(x) * float(x) for x in delta))

def selected_logprobs(logits, ids):
    actual = torch.log_softmax(torch.from_numpy(logits[0].copy()), dim=-1)
    return [float(actual[i, token]) for i, token in enumerate(ids)]


def baseline_gate(margins, context):
    assert len(margins) == 2 and all(math.isfinite(x) for x in (*margins, context))
    a, b = margins
    delta = a - b
    raw = [name for name, failed in (('first-margin-not-positive', a <= 0),
                                      ('second-margin-not-negative', b >= 0),
                                      ('gap-below-one', delta < 1)) if failed]
    reference = [name for name, failed in (('first-not-above-context', a <= context),
                                            ('second-not-below-context', b >= context)) if failed]
    return {'delta': delta, 'context_relative_margins': [a-context, b-context],
            'gap_supported': delta >= 1, 'raw_supported': not raw,
            'strict_supported': not raw and not reference, 'raw_reasons': raw,
            'context_reasons': reference}


def track_statuses(gate, outcome):
    assert outcome in ('positive', 'negative', 'dose-ineligible', 'dose-unmatchable')
    if not gate['gap_supported']:
        return {'status': 'baseline-unsupported', 'raw_status': 'baseline-unsupported',
                'descriptive_status': 'gap-ineligible'}
    return {'status': outcome if gate['strict_supported'] else 'baseline-unsupported',
            'raw_status': outcome if gate['raw_supported'] else 'baseline-unsupported',
            'descriptive_status': outcome}


def track_aggregate(cells, key):
    assert len(cells) == 12
    pairs = []
    for index in range(6):
        values = [cells[index * 2 + offset][key] for offset in (0, 1)]
        pairs.append('unevaluable' if any(x not in ('positive', 'negative') for x in values)
                     else 'positive' if values == ['positive', 'positive'] else 'negative')
    p, n = pairs.count('positive'), pairs.count('negative')
    return {'status': 'positive' if p >= 3 and p > n else
                      'negative' if n >= 3 and n > p else 'inconclusive', 'pairs': pairs}


def check_track_statuses(saved, expected):
    assert all(saved[key] == value for key, value in expected.items())

# BEGIN EMBEDDED CUSTODY ORACLE
"""Independent, fabricated-testable checks for the prospective custody format.

This validates saved records and commitments; it cannot authenticate publication
time, salt generation, access controls, or tokenizer/model execution.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

IDENTITY = ('source_commit', 'protocol_sha256', 'asset_manifest_sha256', 'tokenizer_revision')
POOL_SCHEMA = 'mlg-flant5-heldout-pool-v1'
CASES_SCHEMA = 'mlg-flant5-heldout-cases-v1'
CARD_SCHEMA = 'mlg-flant5-heldout-commitment-v1'
EXPOSURE_ORDER = (
    'docs/FLAN_T5_PUBLIC_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json',
    'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json',
)
EXPECTED_EXPOSURES = frozenset(EXPOSURE_ORDER)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError('nonfinite JSON number')


def decode_json(raw, *, canonical=False, max_bytes=4 * 1024 * 1024):
    if type(raw) is not bytes or len(raw) > max_bytes:
        raise ValueError('invalid JSON evidence size')
    obj = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_pairs,
                     parse_constant=_reject_constant)
    if canonical:
        encoded = json.dumps(obj, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode('utf-8')
        if encoded != raw:
            raise ValueError('noncanonical committed payload')
    return obj


def _hex(value, length):
    return type(value) is str and re.fullmatch(f'[0-9a-f]{{{length}}}', value) is not None


def _identity(obj, expected_revision):
    if (not _hex(obj.get('source_commit'), 40)
            or not _hex(obj.get('protocol_sha256'), 64)
            or not _hex(obj.get('asset_manifest_sha256'), 64)
            or obj.get('tokenizer_revision') != expected_revision):
        raise ValueError('invalid source or asset identity')


def verify_card(payload_raw, salt, card, kind, *, expected_revision='0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'):
    if kind not in ('pool', 'cases'):
        raise ValueError('unknown commitment domain')
    payload = decode_json(payload_raw, canonical=True)
    schema = POOL_SCHEMA if kind == 'pool' else CASES_SCHEMA
    keys = ({'schema', *IDENTITY, 'exposure_inputs', 'templates', 'pairs'} if kind == 'pool'
            else {'schema', *IDENTITY, 'pool_commitment', 'screen', 'cases'})
    card_keys = {'schema', 'kind', 'commitment', *IDENTITY}
    if kind == 'cases':
        card_keys.add('pool_commitment')
    if type(payload) is not dict or set(payload) != keys or payload['schema'] != schema:
        raise ValueError('unexpected committed payload schema')
    if type(card) is not dict or set(card) != card_keys or card['schema'] != CARD_SCHEMA or card['kind'] != kind:
        raise ValueError('unexpected public card schema')
    _identity(payload, expected_revision)
    if any(card[key] != payload[key] for key in IDENTITY):
        raise ValueError('card identity differs from committed payload')
    if type(salt) is not bytes or len(salt) != 32:
        raise ValueError('salt is not 32 bytes')
    digest = hashlib.sha256(schema.encode('ascii') + b'\x00' + salt + payload_raw).hexdigest()
    if card['commitment'] != digest or not _hex(card['commitment'], 64):
        raise ValueError('commitment differs')
    if kind == 'cases' and (not _hex(payload['pool_commitment'], 64)
                            or payload['pool_commitment'] != card['pool_commitment']):
        raise ValueError('pool commitment differs')
    return payload


def _bounded_string(value, max_bytes):
    return type(value) is str and bool(value) and len(value.encode('utf-8')) <= max_bytes


def verify_pool_shape(pool, source_hashes):
    exposure = pool['exposure_inputs']
    if type(exposure) is not list or len(exposure) != len(EXPECTED_EXPOSURES):
        raise ValueError('incorrect exposure list')
    if [entry.get('path') if type(entry) is dict else None for entry in exposure] != list(EXPOSURE_ORDER):
        raise ValueError('incorrect exposure order')
    seen = set()
    for entry in exposure:
        if type(entry) is not dict or set(entry) != {'path', 'commit', 'sha256'}:
            raise ValueError('incorrect exposure entry')
        path = entry['path']
        if (path not in EXPECTED_EXPOSURES or path in seen
                or entry['commit'] != pool['source_commit']
                or entry['sha256'] != source_hashes.get(path)):
            raise ValueError('exposure source mismatch')
        seen.add(path)
    if seen != EXPECTED_EXPOSURES:
        raise ValueError('missing exposure source')
    templates = pool['templates']
    if type(templates) is not list or len(templates) != 2:
        raise ValueError('two templates are required')
    for index, item in enumerate(templates):
        if (type(item) is not dict or set(item) != {'template_id', 'prefix', 'suffix'}
                or item['template_id'] != f'template{index}'
                or not _bounded_string(item['prefix'], 256)
                or not _bounded_string(item['suffix'], 256)
                or any(ord(char) < 32 for char in item['prefix'] + item['suffix'])
                or item['suffix'][-1] not in '.?!:;'):
            raise ValueError('invalid template')
    normalized_templates = [tuple(unicodedata.normalize('NFC', item[key]).casefold()
                                  for key in ('prefix', 'suffix')) for item in templates]
    if normalized_templates[0] == normalized_templates[1]:
        raise ValueError('duplicate normalized template')
    prior_templates = {('translate english to german: ', ''),
                       ('what is the german word for ', '?')}
    if any(item in prior_templates for item in normalized_templates):
        raise ValueError('template appears in a prior public study')
    pairs = pool['pairs']
    if type(pairs) is not list or len(pairs) != 24:
        raise ValueError('24 fixed candidates are required')
    seen_words, seen_targets = set(), set()
    for index, pair in enumerate(pairs):
        if (type(pair) is not dict or set(pair) != {'candidate_index', 'source_words', 'target_words'}
                or type(pair['candidate_index']) is not int or pair['candidate_index'] != index):
            raise ValueError('incorrect candidate identity')
        for field, seen_values in (('source_words', seen_words), ('target_words', seen_targets)):
            values = pair[field]
            if type(values) is not list or len(values) != 2:
                raise ValueError('incorrect lexical pair')
            for value in values:
                if not _bounded_string(value, 64) or any(char.isspace() or ord(char) < 32 for char in value):
                    raise ValueError('invalid lexical item')
                folded = unicodedata.normalize('NFC', value).casefold()
                if folded in seen_values:
                    raise ValueError('duplicate normalized lexical item')
                seen_values.add(folded)


def prior_exposure(source_checkout, tokenizer):
    """Collect all public word-position and non-special target subtokens."""
    root = Path(source_checkout)
    prior_words, prior_targets, source_ids, target_ids = set(), set(), set(), set()
    special_ids = set(tokenizer.all_special_ids)

    def normalized(value):
        return unicodedata.normalize('NFC', value).casefold()

    def collect(case):
        sources = case.get('source', case.get('sources'))
        targets = case.get('targets')
        if type(sources) is not list or type(targets) is not list:
            raise ValueError('prior source or targets missing')
        for row in sources:
            if 'word_positions' in row:
                positions = row['word_positions']
            elif 'word_position' in row:
                positions = [row['word_position']]
            elif 'word_position' in case and case['word_position'] is not None:
                positions = [case['word_position']]
            else:
                positions = []
            if type(positions) is not list or any(type(i) is not int or not 0 <= i < len(row['input_ids'])
                                                   for i in positions):
                raise ValueError('prior word positions malformed')
            source_ids.update(row['input_ids'][i] for i in positions)
            word = row.get('word')
            if word is None and positions:
                start = min(row['offsets'][i][0] for i in positions)
                end = max(row['offsets'][i][1] for i in positions)
                word = row['text'][start:end].strip()
            if type(word) is str and word:
                prior_words.add(normalized(word))
        for row in targets:
            if type(row.get('text')) is not str or not row['text']:
                raise ValueError('prior target text missing')
            prior_targets.add(normalized(row['text']))
            ids = row['input_ids']
            mask = row.get('special_tokens_mask')
            if mask is not None and (type(mask) is not list or len(mask) != len(ids)):
                raise ValueError('prior target mask malformed')
            for index, token in enumerate(ids):
                if token not in special_ids and (mask is None or mask[index] == 0):
                    target_ids.add(token)

    for relative in sorted(EXPECTED_EXPOSURES):
        value = decode_json((root / relative).read_bytes())
        if relative.endswith('FLAN_T5_PUBLIC_CASES_V1.json'):
            cases = value['pairs']
        elif relative.endswith('FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json'):
            cases = [case for candidate in value['candidates'] for case in candidate['cases']]
        else:
            cases = value['cases']
        for case in cases:
            collect(case)
    return prior_words, prior_targets, source_ids, target_ids


def verify_novel_pool_words(pool, prior_words, prior_targets):
    forbidden = prior_words | prior_targets
    for pair in pool['pairs']:
        for field in ('source_words', 'target_words'):
            if any(unicodedata.normalize('NFC', value).casefold() in forbidden for value in pair[field]):
                raise ValueError('pool lexical item appears in recorded prior evidence')


def verify_observation(raw, kind, card_raw, source_commit):
    value = decode_json(raw, max_bytes=65536)
    keys = {'schema', 'kind', 'commit', 'commit_url', 'card_path', 'card_sha256', 'source_commit',
            'observer', 'observed_utc'}
    if (type(value) is not dict or set(value) != keys
            or value['schema'] != 'mlg-flant5-heldout-card-observation-v1'
            or value['kind'] != kind or not _hex(value['commit'], 40)
            or value['commit_url'] != f"https://github.com/EastonSell/model-looking-glass/commit/{value['commit']}"
            or value['card_sha256'] != hashlib.sha256(card_raw).hexdigest()
            or value['source_commit'] != source_commit
            or not _bounded_string(value['card_path'], 512)
            or value['card_path'].startswith('/') or '\\' in value['card_path']
            or any(part in ('', '.', '..') for part in value['card_path'].split('/'))
            or not value['card_path'].endswith('.json')
            or not _bounded_string(value['observer'], 1024)
            or not _bounded_string(value['observed_utc'], 1024)):
        raise ValueError('commitment observation mismatch')
    return value


def verify_selected_cell(case, pair, template, tokenizer, prior_source_ids, prior_target_ids):
    """Independently retokenize a selected cell and derive all three sites.

    This checks the saved tokenizer rows and fixed alignment. It does not judge
    translation quality, the linguistic-review verdict, or unrecorded exposure.
    """
    if type(case.get('semantic_eligible')) is not bool or type(case.get('semantic_reasons')) is not list:
        raise ValueError('missing semantic review')
    if case['semantic_eligible'] != (not case['semantic_reasons']):
        raise ValueError('semantic verdict and reasons disagree')

    def retokenize(row, expected_text):
        if type(row) is not dict or row.get('text') != expected_text:
            raise ValueError('saved text differs from committed pool')
        encoded = tokenizer(expected_text, return_offsets_mapping=True,
                            return_attention_mask=True, return_special_tokens_mask=True)
        for key, saved_key in (('input_ids', 'input_ids'), ('attention_mask', 'attention_mask'),
                               ('special_tokens_mask', 'special_tokens_mask'), ('offset_mapping', 'offsets')):
            actual = [list(value) if isinstance(value, tuple) else value for value in encoded[key]]
            if row.get(saved_key) != actual:
                raise ValueError('saved tokenization differs')
        if row.get('token_strings') != tokenizer.convert_ids_to_tokens(encoded['input_ids']):
            raise ValueError('saved token strings differ')
        ids = row['input_ids']
        if (not ids or ids[-1] != 1 or ids.count(1) != 1
                or row['attention_mask'] != [1] * len(ids)
                or row['special_tokens_mask'][-1] != 1
                or any(type(value) is not int or not 0 <= value < 32128 for value in ids)):
            raise ValueError('invalid token IDs or masks')
        return row

    prefix, suffix = template['prefix'], template['suffix']
    sources = [retokenize(case['source'][side], prefix + pair['source_words'][side] + suffix)
               for side in range(2)]
    neutral = retokenize(case['neutral'], prefix + suffix)
    targets = [retokenize(case['targets'][side], pair['target_words'][side]) for side in range(2)]
    if max(len(row['input_ids']) for row in (*sources, neutral)) > 16:
        raise ValueError('source or context exceeds the fixed length')
    specials = set(tokenizer.all_special_ids)
    for target in targets:
        ids = target['input_ids']
        offsets = target['offsets']
        if (len(ids) != 2 or ids[0] in specials or ids[0] in prior_target_ids
                or target['special_tokens_mask'] != [0, 1]
                or target['text'][offsets[0][0]:offsets[0][1]].strip() != target['text']):
            raise ValueError('target is not a novel single lexical token plus EOS')
    if targets[0]['input_ids'][0] == targets[1]['input_ids'][0]:
        raise ValueError('target tokens collide')
    a, b = sources
    if (len(a['input_ids']) != len(b['input_ids'])
            or a['attention_mask'] != b['attention_mask']
            or a['special_tokens_mask'] != b['special_tokens_mask']):
        raise ValueError('source masks or lengths differ')
    differences = [i for i, (x, y) in enumerate(zip(a['input_ids'], b['input_ids'])) if x != y]
    if len(differences) != 1 or case.get('word_position') != differences[0]:
        raise ValueError('source differs outside one word token')
    word = differences[0]
    for side, row in enumerate(sources):
        start, end = row['offsets'][word]
        if (row['input_ids'][word] in specials or row['input_ids'][word] in prior_source_ids
                or row['special_tokens_mask'][word] != 0
                or row.get('word') != pair['source_words'][side]
                or row.get('word_positions') != [word]
                or row['text'][start:end].strip() != pair['source_words'][side]):
            raise ValueError('source word token differs or was previously exposed')
    prefix_sites = []
    for index, special in enumerate(a['special_tokens_mask']):
        segments = [row['text'][row['offsets'][index][0]:row['offsets'][index][1]]
                    if 0 <= row['offsets'][index][0] < row['offsets'][index][1] <= len(prefix) else ''
                    for row in sources]
        if (index != word and not special
                and all(any(char.isalpha() for char in segment) for segment in segments)):
            prefix_sites.append(index)
    suffix_sites = []
    for index in range(len(a['input_ids'])):
        if index == word or a['special_tokens_mask'][index] or a['input_ids'][index] != b['input_ids'][index]:
            continue
        pieces = []
        for side, row in enumerate(sources):
            start, end = row['offsets'][index]
            boundary = len(prefix) + len(pair['source_words'][side])
            pieces.append(row['text'][start:end] if boundary <= start < end <= len(row['text']) else None)
        if type(pieces[0]) is str and pieces[0] in '.?!:;' and pieces[0] == pieces[1]:
            suffix_sites.append(index)
    if not prefix_sites or not suffix_sites:
        raise ValueError('required control site missing')
    if (case.get('instruction_position') != min(prefix_sites)
            or case.get('context_position') != max(suffix_sites)
            or len({word, min(prefix_sites), max(suffix_sites)}) != 3):
        raise ValueError('control sites differ from the frozen recipe')


def independently_screen(pool, saved, tokenizer, prior_source_ids, prior_target_ids):
    """Retokenize the entire committed pool and derive eligibility/first six."""
    def row(text):
        result = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                           add_special_tokens=True)
        ids = list(result['input_ids'])
        out = {'text': text, 'input_ids': ids, 'attention_mask': list(result['attention_mask']),
               'special_tokens_mask': list(result['special_tokens_mask']),
               'offsets': [list(item) for item in result['offset_mapping']],
               'token_strings': list(tokenizer.convert_ids_to_tokens(ids))}
        if (not 1 <= len(ids) <= 512 or any(len(out[key]) != len(ids) for key in
                ('attention_mask', 'special_tokens_mask', 'offsets', 'token_strings'))
                or any(type(i) is not int or i < 0 for i in ids)
                or any(type(i) is not int or i not in (0, 1) for key in ('attention_mask', 'special_tokens_mask')
                       for i in out[key])
                or any(type(offset) is not list or len(offset) != 2
                       or any(type(i) is not int for i in offset)
                       or not 0 <= offset[0] <= offset[1] <= len(text) for offset in out['offsets'])):
            raise ValueError('invalid tokenization')
        return out

    def source_shape(item):
        ids = item['input_ids']
        return (len(ids) <= 16 and ids[-1] == 1 and ids.count(1) == 1
                and item['attention_mask'] == [1] * len(ids) and item['special_tokens_mask'][-1] == 1)

    def target_shape(item):
        ids = item['input_ids']
        return (len(ids) == 2 and ids[-1] == 1 and ids.count(1) == 1
                and ids[0] not in tokenizer.all_special_ids and ids[0] != tokenizer.unk_token_id
                and item['attention_mask'] == [1, 1] and item['special_tokens_mask'] == [0, 1]
                and item['text'][item['offsets'][0][0]:item['offsets'][0][1]].strip() == item['text'])

    def evaluate(pair, template):
        prefix, suffix = template['prefix'], template['suffix']
        sources = []
        for word in pair['source_words']:
            item = row(prefix + word + suffix)
            lo, hi = len(prefix), len(prefix) + len(word)
            item['word'] = word
            item['word_positions'] = [i for i, (start, end) in enumerate(item['offsets'])
                                      if start < hi and end > lo]
            sources.append(item)
        neutral = row(prefix + suffix)
        targets = [row(word) for word in pair['target_words']]
        reasons = []
        if not all(source_shape(item) for item in (*sources, neutral)):
            reasons.append('source-or-context-shape')
        if not all(target_shape(item) for item in targets) or targets[0]['input_ids'] == targets[1]['input_ids']:
            reasons.append('target-tokenization')
        if any(len(item['input_ids']) >= 2 and item['input_ids'][0] in prior_target_ids for item in targets):
            reasons.append('target-token-exposed')
        first, second = sources
        positions = [item['word_positions'] for item in sources]
        word = positions[0][0] if len(positions[0]) == 1 else None
        word_ok = (word is not None and positions[1] == [word]
                   and len(first['input_ids']) == len(second['input_ids'])
                   and first['attention_mask'] == second['attention_mask']
                   and first['special_tokens_mask'] == second['special_tokens_mask']
                   and [i for i, (a, b) in enumerate(zip(first['input_ids'], second['input_ids'])) if a != b] == [word])
        if word_ok:
            for item in sources:
                lo, hi = item['offsets'][word]
                if (item['special_tokens_mask'][word] != 0 or item['input_ids'][word] in tokenizer.all_special_ids
                        or item['text'][lo:hi].strip() != item['word']):
                    word_ok = False
        if not word_ok:
            reasons.append('word-alignment')
        if any(item['input_ids'][position] in prior_source_ids for item in sources for position in item['word_positions']):
            reasons.append('word-token-exposed')
        instruction, punctuation = [], []
        if len(first['input_ids']) == len(second['input_ids']):
            for index in range(len(first['input_ids'])):
                if (index == word or first['input_ids'][index] != second['input_ids'][index]
                        or any(item['attention_mask'][index] != 1 or item['special_tokens_mask'][index] != 0
                               for item in sources)):
                    continue
                spans = [(item['offsets'][index][0], item['offsets'][index][1],
                          item['text'][item['offsets'][index][0]:item['offsets'][index][1]]) for item in sources]
                if all(0 <= lo < hi <= len(prefix) and any(ch.isalpha() for ch in text)
                       for lo, hi, text in spans):
                    instruction.append(index)
                if all(lo >= len(prefix) + len(item['word']) and hi <= len(item['text'])
                       and len(text) == 1 and text in '.?!:;'
                       for item, (lo, hi, text) in zip(sources, spans)):
                    punctuation.append(index)
        instruction_pos = instruction[0] if instruction else None
        punctuation_pos = punctuation[-1] if punctuation else None
        if instruction_pos is None or punctuation_pos is None or len({word, instruction_pos, punctuation_pos}) != 3:
            reasons.append('control-alignment')
        return {'template_id': template['template_id'], 'source': sources, 'targets': targets,
                'neutral': neutral, 'word_position': word, 'instruction_position': instruction_pos,
                'context_position': punctuation_pos, 'alignment_eligible': not reasons,
                'eligibility_reasons': reasons}

    if (type(saved) is not dict or set(saved) != {'schema', 'pool_commitment', 'candidates',
                                                  'selected_candidate_indices', 'status'}
            or saved['schema'] != 'mlg-flant5-heldout-screen-v1' or len(saved['candidates']) != 24):
        raise ValueError('saved screen shape differs')
    expected = []
    for pair in pool['pairs']:
        cases = [evaluate(pair, template) for template in pool['templates']]
        expected.append({'candidate_index': pair['candidate_index'],
                         'eligible': all(case['alignment_eligible'] for case in cases), 'cases': cases})
    selected = [candidate['candidate_index'] for candidate in expected if candidate['eligible']][:6]
    if (saved['candidates'] != expected or saved['selected_candidate_indices'] != selected
            or saved['status'] != ('complete' if len(selected) == 6 else 'selection-incomplete')):
        raise ValueError('saved screen or first-six selection differs')
    return selected

def verify_complete_owner(owner, custody_root, custody_inventory, completed_bytes):
    """Check a locally observed clean exit; this does not independently prove it."""
    assert type(owner) is dict and owner['status'] == 'complete' and owner['child_exit_code'] == 0
    assert owner['schema'] == 'mlg-flant5-heldout-owner-v1'
    assert owner['terminal_sha256'] == sha(completed_bytes)
    assert owner['final_custody_inventory'] == custody_inventory
    assert owner['streams_exceeded_bound'] is False and owner['reader_threads_finished'] is True
    assert owner['bytecode_prefix_empty'] is True and owner['owner_bytecode_prefix_empty'] is True
    assert owner['source_unchanged'] is True and owner['timed_out'] is False
    assert owner['owner_timeout_seconds'] == 3600
    assert owner['stop']['reaped'] is True and owner['stop']['terminate_attempted'] is False
    assert owner['stop']['kill_attempted'] is False and owner['stop']['error_type'] is None
    assert owner['error_type'] is None and owner['terminal_error_type'] is None
    for number, name in enumerate(('owner.stdout.bin', 'owner.stderr.bin')):
        stream = (custody_root / name).read_bytes()
        assert len(stream) <= 65536 and owner['stream_states'][number]['observed_bytes'] == len(stream)
        assert 'reader_error_type' not in owner['stream_states'][number]
    for name in ('owner-bytecode-prefix', 'owner-parent-bytecode-prefix'):
        prefix = custody_root / name
        assert prefix.is_dir() and not any(prefix.rglob('*'))


# BEGIN HELDOUT EVIDENCE STARTUP
if len(sys.argv) != 5:
    raise SystemExit('usage: AUDITOR RUN_DIR CUSTODY_DIR SOURCE_CHECKOUT ASSET_DIR')
root = Path(sys.argv[1]).resolve(strict=False)
custody_root = Path(sys.argv[2]).resolve(strict=True)
source_checkout = Path(sys.argv[3]).resolve(strict=True)
asset_dir = Path(sys.argv[4]).resolve(strict=True)
assert (not root.exists() or root.is_dir()) and custody_root.is_dir() and root != custody_root
assert root.parent == custody_root.parent


def inventory(directory):
    entries = []
    if not directory.exists():
        return {'bytes': 0, 'files': 0, 'entries': []}
    info = os.lstat(directory)
    assert stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
    assert not (getattr(info, 'st_file_attributes', 0) & 0x400)
    for path in directory.rglob('*'):
        info = os.lstat(path)
        assert not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 0x400)
        if stat.S_ISREG(info.st_mode):
            entries.append({'path': path.relative_to(directory).as_posix(), 'size': info.st_size})
        else:
            assert stat.S_ISDIR(info.st_mode)
    entries.sort(key=lambda item: item['path'])
    return {'bytes': sum(item['size'] for item in entries), 'files': len(entries), 'entries': entries}


custody_inventory = inventory(custody_root)
assert custody_inventory['bytes'] <= 4 * 1024 * 1024
run_inventory = inventory(root)
assert run_inventory['bytes'] <= 508 * 1024 * 1024
assert custody_inventory['bytes'] + run_inventory['bytes'] <= 512 * 1024 * 1024
owner_path = custody_root / 'owner-completed.json'
owner = decode_json(owner_path.read_bytes(), max_bytes=65536) if owner_path.exists() else None
terminal_path = root / 'completed.json'
if not terminal_path.is_file() or owner is None or owner.get('status') != 'complete':
    # A stopped or uncertain attempt is never promoted to a scientific result.
    assert owner is None or owner.get('status') == 'incomplete'
    print(json.dumps({'status': 'incomplete', 'terminal_present': terminal_path.is_file(),
                      'run_bytes': run_inventory['bytes'], 'custody_bytes': custody_inventory['bytes'],
                      'scope': 'observed inventory only; no scientific result or process quiescence audited'}, sort_keys=True))
    raise SystemExit(2)
completed_bytes = terminal_path.read_bytes()
completed = decode_json(completed_bytes, max_bytes=1024 * 1024)
verify_complete_owner(owner, custody_root, custody_inventory, completed_bytes)

started = decode_json((root / 'started.json').read_bytes(), max_bytes=65536)
assert started['schema'] == 'mlg-flant5-heldout-response-attempt-v1' and started['status'] == 'started'
assert completed['schema'] == 'mlg-flant5-heldout-response-v1' and completed['status'] == 'complete'
assert completed['source'] == started['source']
source = completed['source']
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_checkout, text=True).strip()
assert commit == source['commit'] == '5342de9e2df4c0c2ecef731154a369b5f8feae62'
source_paths = {
    'looking_glass/__init__.py', 'looking_glass/flan_t5_heldout_owner.py', 'looking_glass/flan_t5_heldout_response.py',
    'looking_glass/flan_t5_heldout_experiment.py', 'looking_glass/flan_t5_heldout_custody.py',
    'looking_glass/flan_t5_dose_response.py', 'looking_glass/flan_t5_equal_dose.py',
    'looking_glass/flan_t5_feasibility.py', 'looking_glass/interventions.py',
    'looking_glass/intervention_plan.py', 'looking_glass/model_ownership.py',
    'docs/FLAN_T5_HELDOUT_RESPONSE_V1.md', 'docs/FLAN_T5_HELDOUT_CUSTODY_V1.md',
    'docs/FLAN_T5_PUBLIC_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json', 'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json', 'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json',
}
assert set(source['source_sha256']) == source_paths
expected_hashes = {
    'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json': '629036188d62403669fe33291fa9e2a8759145be828918db25fa3ea7c0555a18',
    'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json': 'a0e320c9d24a8940fbca05cbc4ed6e3f2661b5b44b2e62e1140832c0e99e7cd6',
    'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json': '8d1ea3b124085ca8817dc80044dca6fd1e897bda7294510ad99e419c01e7708b',
    'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json': 'adabba92e451fb0c27415b8ec67b67134b59bb5aac293b3c056936cfe872c0a2',
    'docs/FLAN_T5_HELDOUT_CUSTODY_V1.md': 'f8de658b9acf28bc722680c633514cc31972f74ddb1d7826101ecc02f2b88138',
    'docs/FLAN_T5_HELDOUT_RESPONSE_V1.md': '29bb1d80c6e7d81197489b00fdeb942ff2fe95315110c7d0a04838fcebc2773b',
    'docs/FLAN_T5_PUBLIC_CASES_V1.json': 'e00d77cf45885b10e19fe689b8cdb13d2d36dd038c7ab2c1701f9367eb2fe9fc',
    'looking_glass/__init__.py': '9e937f02ec3e4ce0908849aa1741d8afafe378e64212362dbf67af73e11a4fe0',
    'looking_glass/flan_t5_dose_response.py': '986879b936e5b603b1d97223c92f7df1c14f30ab4e692c5873aa43efa8d7076f',
    'looking_glass/flan_t5_equal_dose.py': '6ed5452a604c0b32862153c9c4a059760759c3108f57a81c05fca1d41fcd9b04',
    'looking_glass/flan_t5_feasibility.py': '67eace7da47a644a78ec39c802472ab296c65a1fcb41136ee6dd361e0ff1b63c',
    'looking_glass/flan_t5_heldout_custody.py': 'a476d9891c7dbf88550128344ce241e217113a9c335adc86463a9e5b94aa3d68',
    'looking_glass/flan_t5_heldout_experiment.py': '70a108324239ed0cc0c3b27b93613b005283495645f4eff1396525dd4ac847c4',
    'looking_glass/flan_t5_heldout_owner.py': '6ad08a53b65ce20fc15f9945cfa3992b7af6c20b4f3546631dc4e12e5282baf0',
    'looking_glass/flan_t5_heldout_response.py': 'ad7e4f4ea981e917a1ea0b08dd9854b818a4dde95eddb11c6b64ffdd0612ffae',
    'looking_glass/intervention_plan.py': '6470676938e73e48234128c30ba1d10a49104ee8513434871bc6e973c55d6120',
    'looking_glass/interventions.py': '71cbcf4776660d2dd883b82815f710b0c909228fd1762504f9e3f984f7f6c60f',
    'looking_glass/model_ownership.py': '99fee108f56ff8b7b7361e39de982c242cff590cc06c5d7a640cc12123cfe434',
}
assert source['source_sha256'] == expected_hashes
for relative, expected in source['source_sha256'].items():
    working = (source_checkout / relative).read_bytes()
    committed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=source_checkout)
    assert working == committed and sha(working) == expected

pool_bytes = (custody_root / 'pool.json').read_bytes()
cases_bytes = (custody_root / 'cases-payload.json').read_bytes()
pool_card_bytes = (custody_root / 'pool-commitment.json').read_bytes()
case_card_bytes = (custody_root / 'case-commitment.json').read_bytes()
assert len(pool_bytes) <= 1024 * 1024 and len(cases_bytes) <= 3 * 1024 * 1024
pool_card = decode_json(pool_card_bytes, canonical=True, max_bytes=65536)
case_card = decode_json(case_card_bytes, canonical=True, max_bytes=65536)
pool = verify_card(pool_bytes, (custody_root / 'pool-salt.bin').read_bytes(), pool_card, 'pool')
manifest = verify_card(cases_bytes, (custody_root / 'case-salt.bin').read_bytes(), case_card, 'cases')
assert (custody_root / 'pool-salt.bin').read_bytes() != (custody_root / 'case-salt.bin').read_bytes()
verify_pool_shape(pool, source['source_sha256'])
assert pool['source_commit'] == manifest['source_commit'] == commit
assert pool['protocol_sha256'] == manifest['protocol_sha256'] == source['source_sha256']['docs/FLAN_T5_HELDOUT_RESPONSE_V1.md']
assert pool['asset_manifest_sha256'] == manifest['asset_manifest_sha256'] == started['asset_manifest_sha256']
assert completed['asset_manifest_sha256'] == started['asset_manifest_sha256'] == 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
assert pool_card['commitment'] == case_card['pool_commitment'] == started['pool_commitment'] == completed['pool_commitment']
assert case_card['commitment'] == started['case_commitment'] == completed['case_commitment']
screen = decode_json((custody_root / 'screen.json').read_bytes(), canonical=True, max_bytes=2 * 1024 * 1024)
assert screen == manifest['screen']
assert len(manifest['cases']) == len(completed['cases']) == 12
assert screen['pool_commitment'] == pool_card['commitment']

# The pinned tokenizer is loaded locally only; no model weights are opened as
# a model and no forward is run. We verify every asset's published bytes.
asset_manifest_raw = (asset_dir / 'asset-manifest.json').read_bytes()
assert sha(asset_manifest_raw) == completed['asset_manifest_sha256']
asset_manifest = decode_json(asset_manifest_raw, max_bytes=65536)
assert asset_manifest['repo'] == 'google/flan-t5-small'
assert asset_manifest['revision'] == pool['tokenizer_revision']
assert set(asset_manifest['files']) == {'config.json', 'generation_config.json', 'model.safetensors',
                                        'special_tokens_map.json', 'spiece.model', 'tokenizer.json',
                                        'tokenizer_config.json'}
for name, item in asset_manifest['files'].items():
    path = asset_dir / name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == item['sha256']
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True, use_fast=True,
                                          trust_remote_code=False)
assert type(tokenizer).__name__ == 'T5TokenizerFast'
assert (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id) == (0, 1, 2)
prior_words, prior_targets, prior_source_ids, prior_target_ids = prior_exposure(source_checkout, tokenizer)
verify_novel_pool_words(pool, prior_words, prior_targets)
selected = independently_screen(pool, screen, tokenizer, prior_source_ids, prior_target_ids)
assert screen['status'] == 'complete' and len(selected) == 6
for pair_index, candidate_index in enumerate(selected):
    for template_index in range(2):
        case = manifest['cases'][2 * pair_index + template_index]
        expected_case = dict(screen['candidates'][candidate_index]['cases'][template_index])
        expected_case.update(case_id=f'pair{pair_index}-template{template_index}', pair_id=pair_index,
                             candidate_index=candidate_index, semantic_eligible=case['semantic_eligible'],
                             semantic_reasons=case['semantic_reasons'])
        assert case == expected_case
        verify_selected_cell(case, pool['pairs'][candidate_index], pool['templates'][template_index],
                             tokenizer, prior_source_ids, prior_target_ids)


def observed_bytes(digest):
    assert type(digest) is str and len(digest) == 64
    matches = []
    for path in custody_root.rglob('*'):
        if path.is_file() and path.stat().st_size <= 65536:
            data = path.read_bytes()
            if sha(data) == digest:
                matches.append(data)
    assert matches and all(data == matches[0] for data in matches)
    return matches[0]


source_observation = decode_json(observed_bytes(started['source_observation_sha256']), max_bytes=65536)
assert set(source_observation) == {'schema', 'commit', 'commit_url', 'observer', 'observed_utc', 'source_sha256'}
assert source_observation['schema'] == 'mlg-flant5-public-observation-v1'
assert source_observation['commit'] == commit and source_observation['source_sha256'] == source['source_sha256']
assert source_observation['commit_url'] == f'https://github.com/EastonSell/model-looking-glass/commit/{commit}'
assert source_observation['observer'] and source_observation['observed_utc']
verify_observation(observed_bytes(started['pool_observation_sha256']), 'pool', pool_card_bytes, commit)
verify_observation(observed_bytes(started['case_observation_sha256']), 'cases', case_card_bytes, commit)
auditor_observation = decode_json(observed_bytes(started['auditor_observation_sha256']), max_bytes=65536)
assert set(auditor_observation) == {'schema', 'commit', 'commit_url', 'observer', 'observed_utc',
                                    'source_commit', 'auditor_path', 'auditor_sha256'}
assert auditor_observation['schema'] == 'mlg-flant5-heldout-auditor-observation-v1'
assert auditor_observation['source_commit'] == commit
assert auditor_observation['auditor_sha256'] == sha(Path(__file__).read_bytes())
assert type(auditor_observation['commit']) is str and re.fullmatch('[0-9a-f]{40}', auditor_observation['commit'])
assert auditor_observation['commit_url'] == f"https://github.com/EastonSell/model-looking-glass/commit/{auditor_observation['commit']}"
assert auditor_observation['auditor_path'] == 'tools/audit_flan_t5_heldout.py'
assert auditor_observation['observer'] and auditor_observation['observed_utc']

bound = {'valid_tensor_bytes': 936 * (2 * 32128 + 4 * 16 * 512) * 4,
         'forward_json_bytes': 936 * 2 * 65536, 'dose_vector_bytes': 12 * 6 * 5 * 512 * 4,
         'other_json_bytes': 26 * 65536 + 1024 * 1024,
         'failure_and_observation_bytes': 2 * 65536 + (8 * 32128 + 4 * 32 * 512) * 4,
         'next_forward_reserve': (8 * 32128 + 4 * 32 * 512) * 4 + 3 * 65536,
         'run_limit_bytes': 508 * 1024 * 1024, 'custody_limit_bytes': 4 * 1024 * 1024}
bound['total_bytes'] = sum(bound[k] for k in ('valid_tensor_bytes', 'forward_json_bytes',
                                               'dose_vector_bytes', 'other_json_bytes',
                                               'failure_and_observation_bytes'))
assert bound == started['fixed_evidence_bound'] == completed['fixed_evidence_bound']
assert bound['total_bytes'] + bound['next_forward_reserve'] <= bound['run_limit_bytes']
assert type(started['max_forwards']) is int and started['max_forwards'] == 936
assert started['run_evidence_cap'] == bound['run_limit_bytes']
assert started['custody_evidence_cap'] == bound['custody_limit_bytes']
assert completed['initial_custody_inventory'] == started['initial_custody_inventory'] == completed['final_custody_inventory']
initial = started['initial_custody_inventory']
assert initial['files'] == len(initial['entries'])
assert initial['bytes'] == sum(item['size'] for item in initial['entries']) <= bound['custody_limit_bytes']
actual_entries = {item['path']: item['size'] for item in custody_inventory['entries']}
assert len(actual_entries) == len(custody_inventory['entries'])
assert all(actual_entries.get(item['path']) == item['size'] for item in initial['entries'])
owner_started_bytes = (custody_root / 'owner-started.json').read_bytes()
owner_started = decode_json(owner_started_bytes, max_bytes=65536)
assert owner_started['schema'] == 'mlg-flant5-heldout-owner-v1' and owner_started['status'] == 'started'
assert owner_started['source'] == source
owner_initial = owner_started['initial_custody_inventory']
assert owner_initial['files'] == len(owner_initial['entries'])
assert owner_initial['bytes'] == sum(item['size'] for item in owner_initial['entries'])
expected_initial_entries = sorted([*owner_initial['entries'],
                                   {'path': 'owner-started.json', 'size': len(owner_started_bytes)}],
                                  key=lambda item: item['path'])
assert initial['entries'] == expected_initial_entries
assert completed['hooks_restored'] is True
runtime = decode_json((root / 'environment.json').read_bytes(), max_bytes=65536)
assert runtime == completed['environment']
assert (runtime['device'], runtime['dtype'], runtime['threads'], runtime['interop_threads'],
        runtime['deterministic'], runtime['attention_implementation']) == ('cpu', 'float32', 1, 1, True, 'eager')
assert type(runtime['threads']) is int and type(runtime['interop_threads']) is int
assert runtime['deterministic'] is True and type(runtime['seed']) is int and runtime['seed'] == 20261028
assert all(runtime['numeric_environment'][key] == value for key, value in
           (('MKL_CBWR', 'COMPATIBLE'), ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1')))
assert type(completed['forwards']) is int and 0 <= completed['forwards'] <= 936
assert not (root / 'failure.json').exists()


by_name = {}
expected_order = []
max_f64_difference = 0.0
for index in range(completed['forwards']):
    row = json.loads((root / f'forward-{index:03d}.json').read_bytes())
    observed = json.loads((root / f'forward-{index:03d}-observed.json').read_bytes())
    assert row['index'] == index and row['file'] == f'forward-{index:03d}.bin'
    assert row['name'] not in by_name and row['dtype'] == 'little-endian-float32'
    assert observed['index'] == index and observed['forward'] == row['name']
    matching_cases = [case['case_id'] for case in manifest['cases']
                      if row['name'].startswith(case['case_id'] + '-')]
    assert len(matching_cases) == 1 and observed['case_id'] == matching_cases[0]
    assert observed['phase'] == 'post-return-validation'
    assert observed['return_observed'] is True
    assert observed['saved_file'] == row['file'] and observed['saved_sha256'] == row['sha256']
    assert observed['tensor_layout'] == row['tensor_layout']
    assert observed['session_metadata'] == row['session_metadata']
    assert row['target_ids'][-1] == 1 and row['decoder_ids'] == [0] + row['target_ids'][:-1]
    assert row['decoder_mask'] == [1] * len(row['decoder_ids'])
    raw = (root / row['file']).read_bytes()
    assert sha(raw) == row['sha256']
    tensors, offset = {}, 0
    for name in ('logits', 'encoder', 'before', 'after', 'capture'):
        info = row['tensor_layout'][name]
        length = info['bytes']
        assert type(info['available']) is bool and type(length) is int and length >= 0
        assert type(info['offset']) is int
        assert info['offset'] == offset
        if not info['available']:
            assert info['shape'] is None and length == 0
            tensors[name] = None
            continue
        assert type(info['shape']) is list and all(type(i) is int and i > 0 for i in info['shape'])
        assert length == 4 * math.prod(info['shape'])
        tensors[name] = np.frombuffer(raw, dtype='<f4', count=length // 4, offset=offset).reshape(info['shape'])
        assert np.isfinite(tensors[name]).all()
        offset += length
    assert offset == len(raw)
    logits, state = tensors['logits'], tensors['encoder']
    assert logits.shape == (1, len(row['target_ids']), 32128)
    assert state.shape == (1, len(row['source_ids']), 512)
    selected = selected_logprobs(logits, row['target_ids'])
    assert selected == row['selected_logprobs'] and math.fsum(selected) == row['score']
    for token_index, token in enumerate(row['target_ids']):
        values = logits[0, token_index].astype(np.float64)
        peak = float(values.max())
        normalizer = peak + math.log(math.fsum(math.exp(float(x) - peak) for x in values))
        difference = abs((float(values[token]) - normalizer) - selected[token_index])
        max_f64_difference = max(max_f64_difference, difference)
        # Diagnostic bound fixed before any pretrained follow-up output. Exact
        # float32 selected log-probabilities were checked independently above.
        assert difference <= 1e-3
    by_name[row['name']] = {'row': row, **tensors}

def get(name, source, target):
    item = by_name[name]
    expected_order.append(name)
    row = item['row']
    assert row['source_ids'] == source['input_ids'] and row['source_mask'] == source['attention_mask']
    assert row['target_ids'] == target['input_ids']
    return item

def plan_check(item, action, position, sequence_length):
    steps = item['row']['session_metadata']['steps']
    assert len(steps) == 3
    for index, (step, name, operation) in enumerate(zip(steps, ('before', 'action', 'after'),
                                                        ('capture', action, 'capture'))):
        assert step['name'] == name and step['operation'] == operation and step['order'] == index
        assert step['module_path'] == 'encoder.final_layer_norm' and step['invocation'] == 0
        assert 'encoder.final_layer_norm' in step['module_aliases']
        assert step['forward_index'] == 0 and step['call_site_known'] is False
        assert step['batch_index'] == (None if position is None else 0) and step['token_index'] == position
        assert step['output_shape'] == [1, sequence_length, 512]
        assert step['selected_shape'] == ([1, sequence_length, 512] if position is None else [512])
        assert step['output_selector'] == [] and step['dtype'] == 'torch.float32' and step['device'] == 'cpu'
        assert step['source_capture'] is None and step['preceding_same_site_actions'] == ['before', 'action'][:index]

used = set()
outcomes = []
for ci, (case, saved) in enumerate(zip(manifest['cases'], completed['cases'])):
    assert saved['case_id'] == case['case_id'] and saved['pair_id'] == ci // 2
    assert saved['template_id'] == ('template0', 'template1')[ci % 2]
    assert json.loads((root / f'case-{ci:02d}.json').read_bytes()) == saved
    assert type(case['semantic_eligible']) is bool
    assert type(case['semantic_reasons']) is list
    if not case['semantic_eligible']:
        check_track_statuses(saved, {key: 'semantic-ineligible' for key in
                                     ('status', 'raw_status', 'descriptive_status')})
        assert saved['reasons'] == case['semantic_reasons']
        outcomes.append(saved)
        continue
    # Selected rows passed the committed first-six screen. A successful run
    # cannot silently turn one into a different alignment-ineligible row.
    assert saved['status'] != 'alignment-ineligible'
    targets = case['targets']
    context = [get(f"{case['case_id']}-context-{t}", case['neutral'], targets[t]) for t in range(2)]
    bases = {(side, t): get(f"{case['case_id']}-base-{side}-{t}", case['source'][side], targets[t])
             for side in range(2) for t in range(2)}
    used.update(x['row']['name'] for x in context + list(bases.values()))
    assert all(x['row']['action'] == 'clean' and x['row']['session_metadata'] is None
               and x['before'] is None and x['after'] is None and x['capture'] is None
               for x in context + list(bases.values()))
    assert all(same(bases[side, 0]['encoder'], bases[side, 1]['encoder']) for side in range(2))
    states = [bases[side, 0]['encoder'] for side in range(2)]
    margins = [bases[s, 0]['row']['score'] - bases[s, 1]['row']['score'] for s in range(2)]
    delta = margins[0] - margins[1]
    context_scores = [x['row']['score'] for x in context]
    assert saved['baseline_margins'] == margins and saved['delta'] == delta
    assert saved['context_scores'] == context_scores and saved['context_margin'] == context_scores[0] - context_scores[1]
    gate = baseline_gate(margins, context_scores[0] - context_scores[1])
    assert saved['baseline_gate'] == gate
    if not gate['gap_supported']:
        check_track_statuses(saved, track_statuses(gate, 'positive'))
        outcomes.append(saved)
        continue

    positions = {'word': case['word_position'], 'instruction': case['instruction_position'],
                 'punctuation': case['context_position']}
    assert len(set(positions.values())) == 3
    assert all(type(p) is int and 0 <= p < states[0].shape[1] for p in positions.values())
    dose_record = json.loads((root / f'dose-{ci:02d}.json').read_bytes())
    dose = dose_record['dose']
    assert dose_record['case_id'] == case['case_id'] and dose == saved['dose']
    assert dose_record['file'] == f'dose-{ci:02d}.bin'
    dose_raw = (root / dose_record['file']).read_bytes()
    assert sha(dose_raw) == dose_record['sha256']
    vectors, cursor = {}, 0
    for item in dose_record['layout']:
        assert item['offset'] == cursor and item['bytes'] == 2048
        assert item['dtype'] == 'little-endian-float32' and item['shape'] == [512]
        key = (item['site'], item['side'], item['role'])
        assert key not in vectors
        vectors[key] = np.frombuffer(dose_raw[cursor:cursor + 2048], dtype='<f4')
        cursor += 2048
    assert cursor == len(dose_raw)
    distances = {site: distance(states[0][0, pos], states[1][0, pos]) for site, pos in positions.items()}
    assert distances == dose['distances']
    for site, pos in positions.items():
        for side in range(2):
            assert same(vectors[site, side, 'own'], states[side][0, pos])
            assert same(vectors[site, side, 'donor'], states[1-side][0, pos])
    q = min(distances.values())
    assert dose['q'] == q
    if q / 3 <= 1e-6:
        assert dose['status'] == 'dose-ineligible' and len(vectors) == 12
        check_track_statuses(saved, track_statuses(gate, 'dose-ineligible'))
        outcomes.append(saved)
        continue
    levels = ((1, 3), (2, 3), (1, 1))
    assert len(vectors) == 30 and len(dose['levels']) == len(levels)
    receiver_norms = {f'{site}-{side}': distance(states[side][0, pos], np.zeros_like(states[side][0, pos]))
                      for site, pos in positions.items() for side in range(2)}
    matchable = True
    for (numerator, denominator), level in zip(levels, dose['levels']):
        level_q = q * (numerator / denominator)
        tolerance = min(max(1e-5, 1e-5 * level_q), .01 * level_q)
        assert (level['numerator'], level['denominator'], level['q'], level['tolerance']) == (
            numerator, denominator, level_q, tolerance)
        achieved = {}
        for site, pos in positions.items():
            fraction = level_q / distances[site]
            for side in range(2):
                own, donor = states[side][0, pos], states[1-side][0, pos]
                delivered = vectors[site, side, f'{numerator}-{denominator}']
                expected = b''.join(struct.pack('<f', float(a) + fraction * (float(b) - float(a)))
                                    for a, b in zip(own, donor))
                assert delivered.tobytes() == expected
                achieved[f'{site}-{side}'] = distance(delivered, own)
        assert level['achieved'] == achieved and level['receiver_norms'] == receiver_norms
        assert level['q_over_receiver_norm'] == {
            key: level_q / norm if norm > 0 else None for key, norm in receiver_norms.items()}
        level_match = (all(x > 0 and abs(x-level_q) <= tolerance for x in achieved.values())
                       and all(abs(achieved[f'word-{side}'] - achieved[f'{site}-{side}'])
                               <= .01 * achieved[f'word-{side}']
                               for side in range(2) for site in ('instruction', 'punctuation')))
        assert level['status'] == ('matchable' if level_match else 'dose-unmatchable')
        matchable = matchable and level_match
    if not matchable:
        assert dose['status'] == 'dose-unmatchable'
        check_track_statuses(saved, track_statuses(gate, 'dose-unmatchable'))
        outcomes.append(saved)
        continue
    assert dose['status'] == 'matchable'
    assert saved['positions'] == positions
    anchors = ('null_word', 'self_word', 'self_instruction', 'self_punctuation',
               'full_self', 'full_swap', 'full_word_swap', 'zero_word')
    conditions = (*anchors, *(f'matched_{site}_{a}of{b}' for a, b in levels for site in positions), 'repeat')
    controls = {}
    for condition in conditions:
        for side in range(2):
            own, donor = states[side], states[1-side]
            if condition == 'repeat':
                action, pos, expected = 'capture', None, None
            elif condition in ('full_self', 'full_swap'):
                action, pos, expected = 'patch', None, own if condition == 'full_self' else donor
            else:
                site = ('instruction' if condition.endswith('instruction') or condition.startswith('matched_instruction_') else
                        'punctuation' if condition.endswith('punctuation') or condition.startswith('matched_punctuation_') else 'word')
                pos = positions[site]
                action = 'null' if condition == 'null_word' else 'zero' if condition == 'zero_word' else 'patch'
                role = condition.rsplit('_', 1)[-1].replace('of', '-') if condition.startswith('matched_') else None
                expected = (np.zeros_like(own[0, pos]) if action == 'zero' else
                            own[0, pos] if action == 'null' else
                            vectors[site, side, role] if condition.startswith('matched_') else
                            donor[0, pos] if condition == 'full_word_swap' else own[0, pos])
            for t in range(2):
                item = get(f"{case['case_id']}-{condition}-{side}-{t}", case['source'][side], targets[t])
                used.add(item['row']['name'])
                controls[condition, side, t] = item
                assert item['row']['action'] == action and item['row']['token_index'] == pos
                if condition in ('repeat', 'null_word', 'self_word', 'self_instruction', 'self_punctuation', 'full_self'):
                    assert same(item['logits'], bases[side, t]['logits'])
                if condition == 'full_swap':
                    assert same(item['logits'], bases[1-side, t]['logits']) and same(item['encoder'], donor)
                if action == 'capture':
                    assert item['before'] is None and item['after'] is None and same(item['capture'], own)
                    assert same(item['encoder'], own)
                    metadata = item['row']['session_metadata']
                    assert metadata['module_path'] == 'encoder.final_layer_norm'
                    assert 'encoder.final_layer_norm' in metadata['module_aliases']
                    assert metadata['call_site_known'] is False and metadata['forward_index'] == 0
                    assert metadata['invocation'] == 0 and metadata['batch_index'] is None
                    assert metadata['token_index'] is None and metadata['output_shape'] == list(own.shape)
                    assert metadata['selected_shape'] == list(own.shape)
                    assert metadata['dtype'] == 'torch.float32' and metadata['device'] == 'cpu'
                else:
                    assert item['capture'] is None
                    plan_check(item, action, pos, len(case['source'][side]['input_ids']))
                    assert same(item['before'], own if pos is None else own[0, pos])
                    assert same(item['after'], expected)
                    if pos is None:
                        assert same(item['encoder'], expected)
                    else:
                        assert same(item['encoder'][0, pos], expected)
                        assert all(same(item['encoder'][0, p], own[0, p])
                                   for p in range(own.shape[1]) if p != pos)
    condition_margins = {c: [controls[c, s, 0]['row']['score'] - controls[c, s, 1]['row']['score']
                             for s in range(2)] for c in conditions}
    movements = {c: [margins[0] - v[0], v[1] - margins[1]] for c, v in condition_margins.items()}
    assert saved['condition_margins'] == condition_margins and saved['signed_movements'] == movements
    assert saved['absolute_movements'] == {c: [abs(x) for x in v] for c, v in movements.items()}
    curves = {site: [[0.0, 0.0], *[movements[f'matched_{site}_{a}of{b}'] for a, b in levels]]
              for site in positions}
    assert saved['response_curves'] == curves
    assert saved['absolute_response_curves'] == {
        site: [[abs(x) for x in row] for row in curve] for site, curve in curves.items()}
    reasons = []
    for side in range(2):
        previous = 0.0
        for i, (a, b) in enumerate(levels, 1):
            word = curves['word'][i][side]
            control = max(abs(curves['instruction'][i][side]), abs(curves['punctuation'][i][side]))
            if word < -1e-4:
                reasons.append(f'reversed-side{side}-level{i}')
            if word < previous - 1e-4:
                reasons.append(f'nonmonotone-side{side}-level{i}')
            if word < control + .02 * (a / b) * delta:
                reasons.append(f'control_comparison-side{side}-level{i}')
            previous = word
        if previous < .05 * delta:
            reasons.append(f'weak_endpoint-side{side}')
    assert saved['reasons'] == reasons
    check_track_statuses(saved, track_statuses(gate, 'negative' if reasons else 'positive'))
    outcomes.append(saved)

assert used == set(by_name)
assert expected_order == list(by_name), 'saved forward record order differs from frozen schedule'
assert len(list(root.glob('forward-*.bin'))) == completed['forwards']
decisions = {'primary': track_aggregate(outcomes, 'status'),
             'raw_gate': track_aggregate(outcomes, 'raw_status'),
             'gap_descriptive': track_aggregate(outcomes, 'descriptive_status')}
assert completed['decisions'] == decisions
expected_files = {'started.json', 'environment.json', 'completed.json'}
expected_files.update(f'case-{i:02d}.json' for i in range(12))
for i in range(completed['forwards']):
    expected_files.update((f'forward-{i:03d}.bin', f'forward-{i:03d}.json',
                           f'forward-{i:03d}-observed.json'))
for i, case in enumerate(completed['cases']):
    if 'dose' in case:
        expected_files.update((f'dose-{i:02d}.bin', f'dose-{i:02d}.json'))
files = [p for p in root.rglob('*') if p.is_file()]
assert {p.relative_to(root).as_posix() for p in files} == expected_files
assert type(completed['bytes_before_terminal']) is int
assert sum(p.stat().st_size for p in files if p.name != 'completed.json') == completed['bytes_before_terminal']
assert all(p.stat().st_size <= (1024 * 1024 if p.name == 'completed.json' else 65536)
           for p in files if p.suffix == '.json')
result = {'status': 'passed', 'terminal_sha256': sha(completed_bytes),
          'owner_receipt_sha256': sha(owner_path.read_bytes()),
          'pool_commitment': pool_card['commitment'], 'case_commitment': case_card['commitment'],
          'source': completed['source'], 'source_checkout_verified': True,
          'run_bytes': run_inventory['bytes'], 'custody_bytes': custody_inventory['bytes'],
          'forwards': len(by_name), 'max_abs_f64_vs_saved_f32_token_score': max_f64_difference,
          'f64_per_token_tolerance': 1e-3, 'decisions': decisions,
          'cases': outcomes,
          'scope': ('saved-evidence consistency only; not independent execution, access-control, '
                    'publication-timing, salt-generation or pretraining-independence proof')}
print(json.dumps(result, sort_keys=True, allow_nan=False))
