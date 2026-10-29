"""Local, model-free custody for a fixed held-out FLAN-T5 case pool."""
import hashlib
import json
import os
import re
import stat
import subprocess
import unicodedata
from pathlib import Path, PurePosixPath

EXPOSURE_PATHS = (
    'docs/FLAN_T5_PUBLIC_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json',
    'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json',
    'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json',
)
CUSTODY_LIMIT_BYTES = 4 * 1024 * 1024


def canonical_json(value):
    """Encode a committed JSON object exactly as specified by the protocol."""
    if type(value) is not dict:
        raise ValueError('committed payload must be an object')
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise ValueError('payload is not canonical JSON') from error


def commitment(kind, payload, salt):
    if kind not in ('pool', 'cases') or type(salt) is not bytes or len(salt) != 32:
        raise ValueError('invalid commitment kind or salt')
    domain = f'mlg-flant5-heldout-{kind}-v1'.encode('ascii')
    return hashlib.sha256(domain + bytes([0]) + salt + canonical_json(payload)).hexdigest()


def parse_canonical(raw):
    """Reject alternate serializations, duplicate keys and nonfinite values."""
    if type(raw) is not bytes or len(raw) > 4 * 1024 * 1024:
        raise ValueError('invalid canonical record')

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate key')
            value[key] = item
        return value

    def invalid_constant(_):
        raise ValueError('nonfinite value')

    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid_constant)
        if canonical_json(value) != raw:
            raise ValueError('noncanonical encoding')
        return value
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError('invalid canonical record') from error


def commitment_card(kind, payload, salt):
    if kind not in ('pool', 'cases') or payload.get('schema') != f'mlg-flant5-heldout-{kind}-v1':
        raise ValueError('invalid committed identity')
    keys = ('source_commit', 'protocol_sha256', 'asset_manifest_sha256', 'tokenizer_revision')
    if any(type(payload.get(k)) is not str or not re.fullmatch('[0-9a-f]{%d}' % n, payload[k])
           for k, n in zip(keys, (40, 64, 64, 40))):
        raise ValueError('invalid committed identity')
    card = {'schema': 'mlg-flant5-heldout-commitment-v1', 'kind': kind,
            'commitment': commitment(kind, payload, salt)}
    card.update({key: payload[key] for key in keys})
    if kind == 'cases':
        pool_commitment = payload.get('pool_commitment')
        if type(pool_commitment) is not str or not re.fullmatch('[0-9a-f]{64}', pool_commitment):
            raise ValueError('invalid pool commitment')
        card['pool_commitment'] = pool_commitment
    return card


def _normalized(text):
    if type(text) is not str or not text:
        raise ValueError('invalid lexical input')
    return unicodedata.normalize('NFC', text).casefold()


def exposure_from_documents(documents, *, special_ids):
    """Collect known word/target texts and all lexical subtokens from prior records."""
    if not isinstance(special_ids, (set, frozenset)) or any(type(i) is not int for i in special_ids):
        raise ValueError('invalid tokenizer special IDs')
    source_words, target_words, word_ids, target_ids = set(), set(), set(), set()

    def add_case(case):
        sources = case.get('source', case.get('sources'))
        if not isinstance(sources, list) or not isinstance(case.get('targets'), list):
            raise ValueError('invalid prior exposure schema')
        for source in sources:
            positions = source.get('word_positions')
            if positions is None:
                position = source.get('word_position', case.get('word_position'))
                positions = [position]
            if (not isinstance(positions, list) or any(type(i) is not int or i < 0
                    or i >= len(source['input_ids']) for i in positions)):
                raise ValueError('invalid prior word position')
            word = source.get('word')
            if word is None:
                if len(positions) != 1:
                    raise ValueError('prior word text unavailable')
                start, end = source['offsets'][positions[0]]
                word = source['text'][start:end].strip()
            source_words.add(_normalized(word))
            word_ids.update(source['input_ids'][i] for i in positions if source['input_ids'][i] not in special_ids)
        for target in case['targets']:
            target_words.add(_normalized(target['text']))
            ids = target['input_ids']
            mask = target.get('special_tokens_mask')
            if mask is not None and (len(mask) != len(ids) or any(x not in (0, 1) for x in mask)):
                raise ValueError('invalid prior target mask')
            target_ids.update(token for index, token in enumerate(ids)
                              if token not in special_ids and (mask is None or not mask[index]))

    try:
        for document in documents:
            if 'pairs' in document:
                cases = document['pairs']
            elif 'cases' in document:
                cases = document['cases']
            elif 'candidates' in document:
                cases = [case for candidate in document['candidates'] for case in candidate['cases']]
            else:
                raise ValueError('invalid prior exposure schema')
            for case in cases:
                add_case(case)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('invalid prior exposure schema') from error
    return {'source_words': source_words, 'target_words': target_words,
            'word_token_ids': word_ids, 'target_token_ids': target_ids}


def load_exposure(checkout, source_commit, tokenizer):
    """Bind prior public exposure inputs to the requested clean source blobs."""
    checkout = Path(checkout)
    if type(source_commit) is not str or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('invalid source identity')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    if head != source_commit:
        raise ValueError('source identity differs')
    documents, inputs = [], []
    for name in EXPOSURE_PATHS:
        raw = (checkout / name).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{source_commit}:{name}'], cwd=checkout)
        if raw != committed:
            raise ValueError('prior exposure bytes differ')
        documents.append(json.loads(raw))
        inputs.append({'path': name, 'commit': source_commit, 'sha256': hashlib.sha256(raw).hexdigest()})
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() != head:
        raise ValueError('source identity changed')
    exposure = exposure_from_documents(documents, special_ids=set(tokenizer.all_special_ids))
    exposure['inputs'] = inputs
    return exposure


def validate_pool(pool, exposure):
    """Check the fixed private pool without displaying its lexical contents."""
    try:
        keys = {'schema', 'source_commit', 'protocol_sha256', 'asset_manifest_sha256',
                'tokenizer_revision', 'exposure_inputs', 'templates', 'pairs'}
        if type(pool) is not dict or set(pool) != keys:
            raise ValueError
        # This verifies only the public-shaped identity; the owner independently binds
        # it to the published source, protocol and pinned asset manifest.
        commitment_card('pool', pool, bytes(32))
        if (type(exposure) is not dict or set(exposure) != {'inputs', 'source_words',
                'target_words', 'word_token_ids', 'target_token_ids'}
                or pool['exposure_inputs'] != exposure['inputs']):
            raise ValueError
        inputs = pool['exposure_inputs']
        if type(inputs) is not list or len(inputs) != len(EXPOSURE_PATHS):
            raise ValueError
        for name, ref in zip(EXPOSURE_PATHS, inputs):
            if (type(ref) is not dict or set(ref) != {'path', 'commit', 'sha256'}
                    or ref['path'] != name or ref['commit'] != pool['source_commit']
                    or type(ref['sha256']) is not str
                    or not re.fullmatch('[0-9a-f]{64}', ref['sha256'])):
                raise ValueError
        for name in ('source_words', 'target_words', 'word_token_ids', 'target_token_ids'):
            if not isinstance(exposure[name], (set, frozenset)):
                raise ValueError
        templates = pool['templates']
        if type(templates) is not list or len(templates) != 2:
            raise ValueError
        normalized_templates = []
        for index, template in enumerate(templates):
            if (type(template) is not dict or set(template) != {'template_id', 'prefix', 'suffix'}
                    or template['template_id'] != f'template{index}'):
                raise ValueError
            prefix, suffix = template['prefix'], template['suffix']
            if (type(prefix) is not str or type(suffix) is not str or not prefix or not suffix
                    or len(prefix.encode('utf-8')) > 256 or len(suffix.encode('utf-8')) > 256
                    or suffix[-1] not in '.?!:;' or any(ord(c) < 32 for c in prefix + suffix)):
                raise ValueError
            normalized_templates.append((_normalized(prefix), _normalized(suffix)))
        if normalized_templates[0] == normalized_templates[1]:
            raise ValueError
        old = {(_normalized('Translate English to German: '), ''),
               (_normalized('What is the German word for '), '?')}
        if any(template in old for template in normalized_templates):
            raise ValueError
        pairs = pool['pairs']
        if type(pairs) is not list or len(pairs) != 24:
            raise ValueError
        seen_source, seen_target = set(), set()
        previous_texts = exposure['source_words'] | exposure['target_words']
        for index, pair in enumerate(pairs):
            if (type(pair) is not dict or set(pair) != {'candidate_index', 'source_words', 'target_words'}
                    or type(pair['candidate_index']) is not int or pair['candidate_index'] != index):
                raise ValueError
            for key, seen in (('source_words', seen_source), ('target_words', seen_target)):
                words = pair[key]
                if type(words) is not list or len(words) != 2:
                    raise ValueError
                for word in words:
                    if (type(word) is not str or not word or len(word.encode('utf-8')) > 64
                            or any(c.isspace() or ord(c) < 32 for c in word)):
                        raise ValueError
                    normalized = _normalized(word)
                    if normalized in seen or normalized in previous_texts:
                        raise ValueError
                    seen.add(normalized)
    except (KeyError, TypeError, UnicodeError, ValueError):
        raise ValueError('invalid held-out pool or exposure') from None


def inventory_custody(root):
    """Measure every retained private file without following links or reparse points.

    This is an apparent-byte inventory under a trusted, quiescent root, not an
    operating-system quota or a guarantee against concurrent mutation.
    """
    try:
        root = Path(root)
        root_info = os.lstat(root)
        if (not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode)
                or getattr(root_info, 'st_file_attributes', 0) & 0x400):
            raise ValueError
        entries, pending, total, directories = [], [(root, '')], 0, 1
        while pending:
            directory, relative = pending.pop()
            with os.scandir(directory) as items:
                for item in items:
                    name = item.name
                    if name in ('.', '..') or '/' in name or '\\' in name:
                        raise ValueError
                    rel = f'{relative}/{name}' if relative else name
                    if len(rel.encode('utf-8')) > 1024:
                        raise ValueError
                    info = item.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                        raise ValueError
                    if stat.S_ISDIR(info.st_mode):
                        directories += 1
                        if directories > 256:
                            raise ValueError
                        pending.append((item.path, rel))
                    elif stat.S_ISREG(info.st_mode):
                        if info.st_size < 0:
                            raise ValueError
                        total += info.st_size
                        entries.append({'path': rel, 'size': info.st_size})
                        if total > CUSTODY_LIMIT_BYTES or len(entries) > 256:
                            raise ValueError
                    else:
                        raise ValueError
        entries.sort(key=lambda entry: entry['path'])
        result = {'bytes': total, 'files': len(entries), 'entries': entries}
        if len(canonical_json(result)) > 64 * 1024:
            raise ValueError
        return result
    except (OSError, UnicodeError, TypeError, ValueError):
        raise ValueError('invalid or oversized private custody inventory') from None


def _tokenize(tokenizer, text):
    try:
        value = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                          add_special_tokens=True)
        ids = list(value['input_ids'])
        attention = list(value['attention_mask'])
        special = list(value['special_tokens_mask'])
        offsets = [list(offset) for offset in value['offset_mapping']]
        tokens = list(tokenizer.convert_ids_to_tokens(ids))
        length = len(ids)
        if (not 1 <= length <= 512 or any(len(item) != length for item in
                (attention, special, offsets, tokens))
                or any(type(i) is not int or i < 0 for i in ids)
                or any(type(i) is not int or i not in (0, 1) for i in attention + special)
                or any(len(offset) != 2 or any(type(i) is not int for i in offset)
                       or not 0 <= offset[0] <= offset[1] <= len(text) for offset in offsets)
                or any(type(token) is not str or len(token.encode('utf-8')) > 256 for token in tokens)):
            raise ValueError
        return {'text': text, 'input_ids': ids, 'attention_mask': attention,
                'special_tokens_mask': special, 'offsets': offsets, 'token_strings': tokens}
    except (KeyError, TypeError, ValueError, UnicodeError, IndexError):
        raise ValueError('invalid or oversized tokenizer record') from None


def _valid_source(row):
    ids = row['input_ids']
    return (len(ids) <= 16 and ids[-1] == 1 and ids.count(1) == 1
            and row['attention_mask'] == [1] * len(ids)
            and row['special_tokens_mask'][-1] == 1)


def _screen_cell(tokenizer, pair, template, exposure):
    prefix, suffix = template['prefix'], template['suffix']
    source = []
    for word in pair['source_words']:
        text = prefix + word + suffix
        row = _tokenize(tokenizer, text)
        start, end = len(prefix), len(prefix) + len(word)
        row['word'] = word
        row['word_positions'] = [i for i, (a, b) in enumerate(row['offsets'])
                                 if a < end and b > start]
        source.append(row)
    neutral = _tokenize(tokenizer, prefix + suffix)
    targets = [_tokenize(tokenizer, word) for word in pair['target_words']]
    reasons = []
    if not all(_valid_source(row) for row in source + [neutral]):
        reasons.append('source-or-context-shape')
    target_ok = []
    for row in targets:
        ids = row['input_ids']
        good = (len(ids) == 2 and ids[-1] == 1 and ids.count(1) == 1
                and ids[0] not in tokenizer.all_special_ids and ids[0] != tokenizer.unk_token_id
                and row['attention_mask'] == [1, 1] and row['special_tokens_mask'] == [0, 1]
                and row['text'][row['offsets'][0][0]:row['offsets'][0][1]].strip() == row['text'])
        target_ok.append(good)
    if not all(target_ok) or targets[0]['input_ids'] == targets[1]['input_ids']:
        reasons.append('target-tokenization')
    if any(row['input_ids'][0] in exposure['target_token_ids'] for row in targets if len(row['input_ids']) >= 2):
        reasons.append('target-token-exposed')
    a, b = source
    positions = [row['word_positions'] for row in source]
    word_position = positions[0][0] if len(positions[0]) == 1 else None
    word_ok = (word_position is not None and positions[1] == [word_position]
               and len(a['input_ids']) == len(b['input_ids'])
               and a['attention_mask'] == b['attention_mask']
               and a['special_tokens_mask'] == b['special_tokens_mask']
               and [i for i, (x, y) in enumerate(zip(a['input_ids'], b['input_ids'])) if x != y] == [word_position])
    if word_ok:
        for row in source:
            lo, hi = row['offsets'][word_position]
            if (row['special_tokens_mask'][word_position] != 0
                    or row['input_ids'][word_position] in tokenizer.all_special_ids
                    or row['text'][lo:hi].strip() != row['word']):
                word_ok = False
    if not word_ok:
        reasons.append('word-alignment')
    if any(row['input_ids'][position] in exposure['word_token_ids']
           for row in source for position in row['word_positions']):
        reasons.append('word-token-exposed')
    instruction, punctuation = [], []
    if len(a['input_ids']) == len(b['input_ids']):
        for index in range(len(a['input_ids'])):
            if (index == word_position or a['input_ids'][index] != b['input_ids'][index]
                    or a['attention_mask'][index] != 1 or b['attention_mask'][index] != 1
                    or a['special_tokens_mask'][index] != 0 or b['special_tokens_mask'][index] != 0):
                continue
            slices = []
            for row in source:
                lo, hi = row['offsets'][index]
                slices.append((lo, hi, row['text'][lo:hi]))
            if all(0 <= lo < hi <= len(prefix) and any(c.isalpha() for c in text)
                   for lo, hi, text in slices):
                instruction.append(index)
            if all(lo >= len(prefix) + len(row['word']) and hi <= len(row['text'])
                   and text in '.?!:;' and len(text) == 1
                   for row, (lo, hi, text) in zip(source, slices)):
                punctuation.append(index)
    instruction_position = instruction[0] if instruction else None
    context_position = punctuation[-1] if punctuation else None
    if (instruction_position is None or context_position is None
            or len({word_position, instruction_position, context_position}) != 3):
        reasons.append('control-alignment')
    return {'template_id': template['template_id'], 'source': source, 'targets': targets,
            'neutral': neutral, 'word_position': word_position,
            'instruction_position': instruction_position, 'context_position': context_position,
            'alignment_eligible': not reasons, 'eligibility_reasons': reasons}


def screen_pool(pool, tokenizer, exposure, pool_commitment):
    """Tokenize every fixed candidate and select the first six eligible pairs."""
    validate_pool(pool, exposure)
    if (type(pool_commitment) is not str or not re.fullmatch('[0-9a-f]{64}', pool_commitment)
            or type(tokenizer).__name__ != 'T5TokenizerFast'
            or tokenizer.pad_token_id != 0 or tokenizer.eos_token_id != 1
            or tokenizer.unk_token_id != 2):
        raise ValueError('invalid tokenizer or public pool commitment')
    candidates = []
    for pair in pool['pairs']:
        cases = [_screen_cell(tokenizer, pair, template, exposure) for template in pool['templates']]
        candidates.append({'candidate_index': pair['candidate_index'],
                           'eligible': all(case['alignment_eligible'] for case in cases), 'cases': cases})
    eligible = [entry['candidate_index'] for entry in candidates if entry['eligible']]
    selected = eligible[:6]
    screen = {'schema': 'mlg-flant5-heldout-screen-v1', 'pool_commitment': pool_commitment,
              'candidates': candidates, 'selected_candidate_indices': selected,
              'status': 'complete' if len(selected) == 6 else 'selection-incomplete'}
    if len(canonical_json(screen)) > 2 * 1024 * 1024:
        raise ValueError('screen exceeds private evidence limit')
    return screen


def selected_cases(screen):
    """Make the runner's ordered case rows from a complete deterministic screen."""
    if screen.get('status') != 'complete' or len(screen.get('selected_candidate_indices', ())) != 6:
        raise ValueError('selection is incomplete')
    cases = []
    for pair_id, candidate_index in enumerate(screen['selected_candidate_indices']):
        candidate = screen['candidates'][candidate_index]
        if candidate['candidate_index'] != candidate_index or not candidate['eligible']:
            raise ValueError('selection differs')
        for index, cell in enumerate(candidate['cases']):
            case = dict(cell)
            case.update({'case_id': f'pair{pair_id}-template{index}', 'pair_id': pair_id,
                         'candidate_index': candidate_index})
            cases.append(case)
    return cases


def _new_salt():
    return os.urandom(32)


def _write_new(root, name, raw, *, reserve=4096):
    if type(raw) is not bytes or type(name) is not str or '/' in name or '\\' in name:
        raise ValueError('invalid private custody write')
    inventory = inventory_custody(root)
    if inventory['bytes'] + len(raw) + reserve > CUSTODY_LIMIT_BYTES or inventory['files'] >= 256:
        raise ValueError('private custody reservation exhausted')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(Path(root) / name, flags, 0o600)
    failure = None
    try:
        if os.write(fd, raw) != len(raw):
            raise OSError('short private custody write')
        os.fsync(fd)
    except BaseException as error:
        failure = error
    try:
        os.close(fd)
    except BaseException as cleanup:
        if failure is None:
            raise
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise failure from cleanup
        if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
            raise cleanup from failure
        raise failure from cleanup
    if failure is not None:
        raise failure


def _read_private(root, name, limit):
    if type(name) is not str or '/' in name or '\\' in name:
        raise ValueError('invalid private custody read')
    inventory_custody(root)
    try:
        with (Path(root) / name).open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError
        return raw
    except (OSError, ValueError):
        raise ValueError('private custody record unavailable or oversized') from None


def _record_failure(root, phase, error):
    error_type = type(error).__name__
    if error_type not in ('KeyboardInterrupt', 'SystemExit', 'MemoryError', 'ValueError', 'OSError'):
        error_type = 'other'
    record = {'schema': 'mlg-flant5-heldout-custody-failure-v1', 'phase': phase,
              'status': 'incomplete', 'error_type': error_type}
    _write_new(root, f'{phase}-failed.json', canonical_json(record), reserve=0)


def _handle_failure(root, phase, error):
    try:
        _record_failure(root, phase, error)
    except BaseException as cleanup:
        if (not isinstance(error, (KeyboardInterrupt, SystemExit))
                and isinstance(cleanup, (KeyboardInterrupt, SystemExit))):
            raise cleanup from error


def _marker(root, phase):
    _write_new(root, f'{phase}-started.json', canonical_json(
        {'schema': 'mlg-flant5-heldout-custody-start-v1', 'phase': phase, 'status': 'started'}))


def _pool_record(root, pool_card, exposure):
    pool = parse_canonical(_read_private(root, 'pool.json', 1024 * 1024))
    salt = _read_private(root, 'pool-salt.bin', 32)
    saved_card = parse_canonical(_read_private(root, 'pool-commitment.json', 64 * 1024))
    validate_pool(pool, exposure)
    if saved_card != pool_card or commitment_card('pool', pool, salt) != saved_card:
        raise ValueError('pool commitment differs')
    return pool, salt


def create_pool_attempt(root, pool, exposure):
    """Create one exclusive private pool attempt and one salted public card."""
    root = Path(root)
    root.mkdir()
    try:
        _marker(root, 'pool')
        validate_pool(pool, exposure)
        raw = canonical_json(pool)
        if len(raw) > 1024 * 1024:
            raise ValueError('private pool exceeds its bound')
        salt = _new_salt()
        if type(salt) is not bytes or len(salt) != 32:
            raise ValueError('pool salt unavailable')
        _write_new(root, 'pool-salt.bin', salt)
        _write_new(root, 'pool.json', raw)
        card = commitment_card('pool', pool, salt)
        _write_new(root, 'pool-commitment.json', canonical_json(card))
        return card
    except BaseException as error:
        _handle_failure(root, 'pool', error)
        raise


def _check_pool_observation(observation, card):
    keys = {'schema', 'kind', 'commit', 'commit_url', 'card_path', 'card_sha256', 'source_commit',
            'observer', 'observed_utc'}
    if type(observation) is not dict or set(observation) != keys:
        raise ValueError('public pool observation differs')
    commit = observation['commit']
    card_path = observation['card_path']
    if (observation['schema'] != 'mlg-flant5-heldout-card-observation-v1'
            or observation['kind'] != 'pool' or type(commit) is not str
            or not re.fullmatch('[0-9a-f]{40}', commit)
            or observation['commit_url'] !=
               f'https://github.com/EastonSell/model-looking-glass/commit/{commit}'
            or type(card_path) is not str or not card_path or len(card_path.encode('utf-8')) > 512
            or '\\' in card_path or card_path.startswith('/') or not card_path.endswith('.json')
            or PurePosixPath(card_path).as_posix() != card_path
            or any(part in ('.', '..') for part in card_path.split('/'))
            or observation['card_sha256'] != hashlib.sha256(canonical_json(card)).hexdigest()
            or observation['source_commit'] != card['source_commit']
            or any(type(observation[key]) is not str or not observation[key]
                   or len(observation[key].encode('utf-8')) > 256 for key in ('observer', 'observed_utc'))):
        raise ValueError('public pool observation differs')


def screen_attempt(root, pool_card, public_observation, tokenizer, exposure):
    """Screen once only after a separately observed public pool card."""
    _marker(root, 'screen')
    try:
        pool, _ = _pool_record(root, pool_card, exposure)
        _check_pool_observation(public_observation, pool_card)
        _write_new(root, 'pool-observation.json', canonical_json(public_observation))
        screen = screen_pool(pool, tokenizer, exposure, pool_card['commitment'])
        _write_new(root, 'screen.json', canonical_json(screen))
        return screen
    except BaseException as error:
        _handle_failure(root, 'screen', error)
        raise


def _semantic_cases(screen, verdicts):
    cases = selected_cases(screen)
    if type(verdicts) is not list or len(verdicts) != len(cases):
        raise ValueError('semantic review is incomplete')
    output = []
    for case, verdict in zip(cases, verdicts):
        if (type(verdict) is not dict
                or set(verdict) != {'case_id', 'semantic_eligible', 'semantic_reasons'}
                or verdict['case_id'] != case['case_id'] or type(verdict['semantic_eligible']) is not bool
                or type(verdict['semantic_reasons']) is not list or len(verdict['semantic_reasons']) > 4
                or any(type(reason) is not str or not reason or len(reason.encode('utf-8')) > 256
                       for reason in verdict['semantic_reasons'])
                or (verdict['semantic_eligible'] and verdict['semantic_reasons'])
                or (not verdict['semantic_eligible'] and not verdict['semantic_reasons'])):
            raise ValueError('semantic review is incomplete')
        result = dict(case)
        result['semantic_eligible'] = verdict['semantic_eligible']
        result['semantic_reasons'] = list(verdict['semantic_reasons'])
        output.append(result)
    return output


def seal_cases_attempt(root, pool_card, screen, semantic_verdicts):
    """Commit the first six eligible pairs with all twelve sealed verdicts."""
    _marker(root, 'cases')
    try:
        pool = parse_canonical(_read_private(root, 'pool.json', 1024 * 1024))
        pool_salt = _read_private(root, 'pool-salt.bin', 32)
        saved_card = parse_canonical(_read_private(root, 'pool-commitment.json', 64 * 1024))
        if saved_card != pool_card or commitment_card('pool', pool, pool_salt) != pool_card:
            raise ValueError('pool commitment differs')
        if parse_canonical(_read_private(root, 'screen.json', 2 * 1024 * 1024)) != screen:
            raise ValueError('private screen differs')
        cases = _semantic_cases(screen, semantic_verdicts)
        keys = ('source_commit', 'protocol_sha256', 'asset_manifest_sha256', 'tokenizer_revision')
        payload = {'schema': 'mlg-flant5-heldout-cases-v1',
                   'pool_commitment': pool_card['commitment'], 'screen': screen, 'cases': cases}
        payload.update({key: pool[key] for key in keys})
        raw = canonical_json(payload)
        if len(raw) > 3 * 1024 * 1024:
            raise ValueError('private case payload exceeds its bound')
        salt = _new_salt()
        if type(salt) is not bytes or len(salt) != 32 or salt == pool_salt:
            raise ValueError('case salt unavailable or repeated')
        _write_new(root, 'case-salt.bin', salt)
        _write_new(root, 'cases-payload.json', raw)
        card = commitment_card('cases', payload, salt)
        _write_new(root, 'case-commitment.json', canonical_json(card))
        return card
    except BaseException as error:
        _handle_failure(root, 'cases', error)
        raise


def verify_bundle(root, pool_card, case_card, tokenizer, exposure):
    """Recompute both commitments and all 48 private tokenizer screens."""
    pool, pool_salt = _pool_record(root, pool_card, exposure)
    payload = parse_canonical(_read_private(root, 'cases-payload.json', 3 * 1024 * 1024))
    case_salt = _read_private(root, 'case-salt.bin', 32)
    saved_card = parse_canonical(_read_private(root, 'case-commitment.json', 64 * 1024))
    if (len(case_salt) != 32 or case_salt == pool_salt or saved_card != case_card
            or commitment_card('cases', payload, case_salt) != saved_card):
        raise ValueError('case commitment differs')
    keys = {'schema', 'source_commit', 'protocol_sha256', 'asset_manifest_sha256',
            'tokenizer_revision', 'pool_commitment', 'screen', 'cases'}
    if (set(payload) != keys or payload['pool_commitment'] != pool_card['commitment']
            or any(payload[key] != pool[key] for key in
                   ('source_commit', 'protocol_sha256', 'asset_manifest_sha256', 'tokenizer_revision'))):
        raise ValueError('case identity differs')
    screen = screen_pool(pool, tokenizer, exposure, pool_card['commitment'])
    if (payload['screen'] != screen
            or parse_canonical(_read_private(root, 'screen.json', 2 * 1024 * 1024)) != screen):
        raise ValueError('private screen differs')
    verdicts = []
    for case in payload['cases']:
        try:
            verdicts.append({key: case[key] for key in ('case_id', 'semantic_eligible', 'semantic_reasons')})
        except (KeyError, TypeError):
            raise ValueError('semantic review is incomplete') from None
    if _semantic_cases(screen, verdicts) != payload['cases']:
        raise ValueError('selected cases differ')
    inventory = inventory_custody(root)
    return {'cases': payload['cases'], 'custody_bytes': inventory['bytes'],
            'identity': {key: pool[key] for key in
                         ('source_commit', 'protocol_sha256', 'asset_manifest_sha256', 'tokenizer_revision')}}
