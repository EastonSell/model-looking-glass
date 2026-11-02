"""Tokenizer-only selection and salted commitments for the FLAN sentiment study.

This module loads no model and makes no claim that the supplied public sources
cover unknown exposures or model pretraining. The separate custodian fixes the
complete ordered pool before any tokenizer selection or pretrained forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import unicodedata

from .flan_t5_sentiment_screen import INSTRUCTIONS


PROTOCOL_SHA256 = '83d666bc67af0951d21eceaa8efb8e7cb16f53a40aefdb72eff03f8b57bd7813'
ASSET_MANIFEST_SHA256 = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
POOL_DOMAIN = b'FLAN-SENTIMENT-HOLDOUT-V1\0'
SELECTION_DOMAIN = b'FLAN-SENTIMENT-SELECTION-V1\0'
SIDES = ('P1', 'N1', 'P2', 'N2')
OPPOSITE = {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
SAME = {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}
REQUIRED_SOURCE_HASHES = {
    'docs/FLAN_T5_PUBLIC_CASES_V1.json': 'e00d77cf45885b10e19fe689b8cdb13d2d36dd038c7ab2c1701f9367eb2fe9fc',
    'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json': 'a0e320c9d24a8940fbca05cbc4ed6e3f2661b5b44b2e62e1140832c0e99e7cd6',
    'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json': '629036188d62403669fe33291fa9e2a8759145be828918db25fa3ea7c0555a18',
    'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json': '8d1ea3b124085ca8817dc80044dca6fd1e897bda7294510ad99e419c01e7708b',
    'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json': 'adabba92e451fb0c27415b8ec67b67134b59bb5aac293b3c056936cfe872c0a2',
    'results/flan-t5-heldout-response-cpu-v1-20261029/custody/pool.json': '90b0f249f93bc48895431b9d60f656922ec199de4bf78cf50548ce4eb75e1386',
    'results/flan-t5-heldout-response-cpu-v1-20261029/custody/screen.json': 'ce163cda5a2cabc2ef2489e33a585f0885f70b9967395da70a40706c398577db',
    'results/flan-t5-sentiment-screen-cpu-v1-20261029/child-records/events.jsonl': 'eb17ca8482e0c28ed4d6146285aaf33fa63db3437f0fb308c5becfa5e50d3d6b',
}
REQUIRED_SOURCE_PATHS = frozenset(REQUIRED_SOURCE_HASHES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def commitment(domain, value, salt):
    require(type(salt) is bytes and len(salt) == 32, 'salt must contain 32 bytes')
    return hashlib.sha256(domain + salt + b'\0' + canonical(value)).hexdigest()


def pool_commitment(pool, salt):
    return commitment(POOL_DOMAIN, pool, salt)


def selection_commitment(selection, salt):
    return commitment(SELECTION_DOMAIN, selection, salt)


def _hex(value, length):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _source_descriptor(value):
    require(type(value) is dict and set(value) == {'path', 'revision', 'sha256'}, 'source keys differ')
    path = value['path']
    require(type(path) is str and 0 < len(path.encode('utf-8')) <= 256
            and '\\' not in path and ':' not in path and not PurePosixPath(path).is_absolute()
            and all(part not in ('', '.', '..') for part in path.split('/')), 'source path is invalid')
    require(_hex(value['revision'], 40) and _hex(value['sha256'], 64), 'source identity is invalid')


def _require_commit(repo, revision):
    require(_hex(revision, 40), 'revision is invalid')
    try:
        kind = subprocess.check_output(['git', '--no-replace-objects', 'cat-file', '-t', revision], cwd=repo,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError('revision is not an available Git commit') from exc
    require(kind == b'commit', 'revision is not a Git commit')


def validate_pool(pool, exposures):
    require(type(pool) is dict and set(pool) == {'schema', 'protocol_sha256', 'runner_revision',
            'asset_manifest_sha256', 'exposure_sources', 'units'}, 'pool keys differ')
    require(pool['schema'] == 'flan-sentiment-private-pool-v1'
            and pool['protocol_sha256'] == PROTOCOL_SHA256
            and pool['asset_manifest_sha256'] == ASSET_MANIFEST_SHA256
            and _hex(pool['runner_revision'], 40), 'pool binding differs')
    sources = pool['exposure_sources']
    require(type(sources) is list and 1 <= len(sources) <= 64, 'source count differs')
    for source in sources:
        _source_descriptor(source)
    require(len({source['path'] for source in sources}) == len(sources), 'duplicate exposure source')
    require(type(exposures) is list and len(exposures) == len(sources), 'exposure inventory differs')
    words = set()
    for source, exposure in zip(sources, exposures):
        require(type(exposure) is dict and set(exposure) == {'descriptor', 'rows', 'words'}
                and exposure['descriptor'] == source and type(exposure['rows']) is list
                and type(exposure['words']) is list, 'exposure data differs')
        words.update(unicodedata.normalize('NFC', x).casefold() for x in exposure['words'])
        for row in exposure['rows']:
            _strip_eos(row)
    units = pool['units']
    require(type(units) is list and len(units) == 12, 'pool requires twelve units')
    seen = set()
    for index, unit in enumerate(units):
        require(type(unit) is dict and set(unit) == {'id', 'subject', 'adjectives', 'semantic_ok',
                'semantic_reason'} and unit['id'] == f'u{index:02d}', 'unit identity differs')
        require(type(unit['adjectives']) is dict and set(unit['adjectives']) == set(SIDES),
                'adjective keys differ')
        require(type(unit['semantic_ok']) is bool and type(unit['semantic_reason']) is str
                and unicodedata.normalize('NFC', unit['semantic_reason']) == unit['semantic_reason']
                and len(unit['semantic_reason'].encode('utf-8')) <= 256
                and (unit['semantic_ok'] or bool(unit['semantic_reason'])), 'semantic disposition differs')
        for word in [unit['subject'], *(unit['adjectives'][side] for side in SIDES)]:
            require(type(word) is str and re.fullmatch('[a-z]{1,24}', word) is not None,
                    'variable words must be lowercase ASCII')
            require(word not in seen, 'variable words must be distinct across pool')
            require(word not in words, 'variable word is already exposed')
            seen.add(word)


def _strip_eos(row):
    require(type(row) is list and len(row) > 1 and all(type(x) is int and 0 <= x < 32128 for x in row)
            and row[-1] == 1 and row.count(1) == 1, 'token row must have exactly one final EOS')
    return row[:-1]


def prefix_conflict(a, b):
    a, b = _strip_eos(a), _strip_eos(b)
    size = min(len(a), len(b))
    return a[:size] == b[:size]


def case_text(unit, instruction, side):
    return f"{INSTRUCTIONS[instruction]} The {unit['subject']} was {unit['adjectives'][side]}."


def _position(text, offsets, masks, ids, start, end, special_ids):
    overlap = [i for i, (left, right) in enumerate(offsets) if left < end and right > start]
    if len(overlap) != 1:
        return None
    position = overlap[0]
    valid_offsets = [[start, end]]
    if start and text[start - 1] == ' ':
        valid_offsets.append([start - 1, end])
    return (position if offsets[position] in valid_offsets and masks['attention_mask'][position] == 1
            and masks['special_tokens_mask'][position] == 0 and ids[position] not in special_ids else None)


def _case(unit, instruction, side, tokenizer):
    text = case_text(unit, instruction, side)
    encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    keys = ('input_ids', 'attention_mask', 'special_tokens_mask', 'offset_mapping')
    require(all(key in encoded for key in keys), 'tokenizer record is missing fields')
    fields = {key: [list(v) if isinstance(v, tuple) else v for v in encoded[key]] for key in keys}
    n = len(fields['input_ids'])
    require(0 < n <= 256 and all(len(fields[key]) == n for key in keys), 'tokenizer record lengths differ')
    require(all(type(x) is int for x in fields['input_ids'])
            and all(type(x) is int and x in (0, 1) for key in ('attention_mask', 'special_tokens_mask')
                    for x in fields[key])
            and all(type(v) is list and len(v) == 2 and all(type(x) is int and 0 <= x <= len(text) for x in v)
                    and v[0] <= v[1] for v in fields['offset_mapping']), 'tokenizer record is malformed')
    _strip_eos(fields['input_ids'])
    require(fields['special_tokens_mask'][-1] == 1, 'final EOS must be marked special')
    subject_start = len(INSTRUCTIONS[instruction]) + len(' The ')
    adjective_start = subject_start + len(unit['subject']) + len(' was ')
    special_ids = set(tokenizer.all_special_ids)
    positions = {}
    for name, start, word in (('subject', subject_start, unit['subject']),
                              ('adjective', adjective_start, unit['adjectives'][side])):
        positions[name] = _position(text, fields['offset_mapping'], fields, fields['input_ids'],
                                    start, start + len(word), special_ids)
    return {'side': side, 'text': text, **fields,
            'adjective_index': positions['adjective'], 'subject_index': positions['subject']}


def _group(unit, instruction, tokenizer, exposures):
    cases = [_case(unit, instruction, side, tokenizer) for side in SIDES]
    reasons = []
    if not unit['semantic_ok']:
        reasons.append('semantic-rejected')
    if any(len(case['input_ids']) > 64 or any(x != 1 for x in case['attention_mask']) for case in cases):
        reasons.append('input-length-or-mask')
    adjective = [case['adjective_index'] for case in cases]
    subject = [case['subject_index'] for case in cases]
    if None in adjective or len(set(adjective)) != 1:
        reasons.append('adjective-alignment')
    if None in subject or len(set(subject)) != 1:
        reasons.append('subject-alignment')
    if (None not in adjective and None not in subject
            and (adjective[0] == subject[0] or 0 in (adjective[0], subject[0]))):
        reasons.append('sites-not-distinct')
    if any(case['attention_mask'][0] != 1 or case['special_tokens_mask'][0] != 0
           or case['input_ids'][0] in tokenizer.all_special_ids for case in cases):
        reasons.append('instruction-site')
    reference = cases[0]
    if (None not in adjective and len(set(adjective)) == 1
            and all(len(case['input_ids']) == len(reference['input_ids'])
                    and case['attention_mask'] == reference['attention_mask']
                    and case['special_tokens_mask'] == reference['special_tokens_mask'] for case in cases)):
        pos = adjective[0]
        if (len({case['input_ids'][pos] for case in cases}) != 4
                or any([i for i, (a, b) in enumerate(zip(reference['input_ids'], case['input_ids']))
                        if a != b] != [pos] for case in cases[1:])):
            reasons.append('token-difference')
    else:
        reasons.append('token-difference')
    comparisons = []
    for index, exposure in enumerate(exposures):
        conflict = False
        compared = 0
        for case in cases:
            for row in exposure['rows']:
                conflict = prefix_conflict(case['input_ids'], row) or conflict
                compared += 1
        comparisons.append({'source_index': index, 'comparisons': len(cases) * len(exposure['rows']),
                            'conflict': conflict})
        require(compared == comparisons[-1]['comparisons'], 'comparison count differs')
    if any(record['conflict'] for record in comparisons):
        reasons.append('prior-input-prefix')
    return {'instruction_index': instruction, 'cases': cases, 'source_comparisons': comparisons,
            'eligible': not reasons, 'failure_reasons': reasons}


def select(pool, tokenizer, exposures, pool_digest):
    validate_pool(pool, exposures)
    require(_hex(pool_digest, 64), 'pool commitment is invalid')
    require(tokenizer.eos_token_id == 1 and tokenizer.pad_token_id == 0, 'tokenizer special IDs differ')
    screens = []
    for unit in pool['units']:
        screens.append({'unit_id': unit['id'], 'semantic_ok': unit['semantic_ok'],
                        'lexical_exclusion_ok': True,
                        'instructions': [_group(unit, i, tokenizer, exposures) for i in range(2)]})
    selected = [unit['unit_id'] for unit in screens if unit['semantic_ok']
                and all(group['eligible'] for group in unit['instructions'])][:4]
    mapping = [{'unit_id': ident, 'opposite': dict(OPPOSITE), 'same': dict(SAME)} for ident in selected]
    return {'schema': 'flan-sentiment-private-selection-v1', 'pool_commitment': pool_digest,
            'protocol_sha256': pool['protocol_sha256'], 'runner_revision': pool['runner_revision'],
            'asset_manifest_sha256': pool['asset_manifest_sha256'], 'exposure_sources': pool['exposure_sources'],
            'unit_screens': screens, 'selected_units': selected, 'donor_mapping': mapping}


def load_exposures(repo, descriptors, *, require_minimum=True):
    """Read bound public Git blobs, collecting recorded words and EOS-ended IDs."""
    require(type(descriptors) is list and descriptors, 'no exposure sources')
    for descriptor in descriptors:
        _source_descriptor(descriptor)
    paths = {item['path'] for item in descriptors}
    require(len(paths) == len(descriptors), 'duplicate source path')
    if require_minimum:
        require(REQUIRED_SOURCE_PATHS <= paths, 'required known exposure source is missing')
        require(all(item['path'] not in REQUIRED_SOURCE_HASHES
                    or item['sha256'] == REQUIRED_SOURCE_HASHES[item['path']] for item in descriptors),
                'required known exposure source digest differs')
    result = []
    for revision in {item['revision'] for item in descriptors}:
        _require_commit(repo, revision)
    for descriptor in descriptors:
        raw = subprocess.check_output(['git', '--no-replace-objects', 'show',
                                       f"{descriptor['revision']}:{descriptor['path']}"], cwd=repo)
        require(hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'exposure source digest differs')
        payloads = ([strict_json(line) for line in raw.splitlines() if line]
                    if descriptor['path'].endswith('.jsonl') else [strict_json(raw)])
        words, rows = set(), []
        def walk(value):
            if type(value) is dict:
                if 'input_ids' in value:
                    _strip_eos(value['input_ids'])
                    rows.append(value['input_ids'])
                for key, child in value.items():
                    if key in ('text', 'subject', 'adjective', 'word', 'words', 'first_word',
                               'second_word', 'source_words', 'target_words'):
                        values = child if type(child) is list else [child]
                        for text in values:
                            if type(text) is str:
                                words.update(re.findall(r'[^\W\d_]+', unicodedata.normalize('NFC', text).casefold()))
                    walk(child)
            elif type(value) is list:
                for child in value:
                    walk(child)
        for payload in payloads:
            walk(payload)
        result.append({'descriptor': descriptor, 'rows': rows, 'words': sorted(words)})
    return result


def _write_new(path, data):
    with Path(path).open('xb') as stream:
        require(stream.write(data) == len(data), 'short write')
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('commit-pool', 'select'):
        p = sub.add_parser(name)
        p.add_argument('--repo', type=Path, required=True)
        p.add_argument('--pool', type=Path, required=True)
        p.add_argument('--pool-salt', type=Path, required=True)
    c = sub.choices['commit-pool']
    c.add_argument('--pool-card', type=Path, required=True)
    s = sub.choices['select']
    for key in ('pool-card', 'pool-observation', 'asset-dir', 'selection', 'selection-salt', 'selection-card'):
        s.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(argv)
    pool_raw = args.pool.read_bytes()
    require(len(pool_raw) <= 256 * 1024, 'pool custody limit exceeded')
    pool = strict_json(pool_raw)
    exposures = load_exposures(args.repo, pool['exposure_sources'])
    validate_pool(pool, exposures)
    _require_commit(args.repo, pool['runner_revision'])
    if args.command == 'commit-pool':
        salt = secrets.token_bytes(32)
        card = {'schema': 'flan-sentiment-pool-card-v1', 'commitment': pool_commitment(pool, salt),
                'protocol_sha256': pool['protocol_sha256'], 'runner_revision': pool['runner_revision'],
                'asset_manifest_sha256': pool['asset_manifest_sha256']}
        _write_new(args.pool_salt, salt)
        card_bytes = canonical(card) + b'\n'
        _write_new(args.pool_card, card_bytes)
        require(args.pool.read_bytes() == pool_raw and args.pool_salt.read_bytes() == salt
                and args.pool_card.read_bytes() == card_bytes
                and pool_commitment(strict_json(pool_raw), args.pool_salt.read_bytes()) == card['commitment'],
                'pool commitment readback differs')
        print(json.dumps({'status': 'pool-committed', 'commitment': card['commitment']}))
        return
    salt = args.pool_salt.read_bytes()
    card_raw = args.pool_card.read_bytes()
    card = strict_json(card_raw)
    expected = {'schema': 'flan-sentiment-pool-card-v1', 'commitment': pool_commitment(pool, salt),
                'protocol_sha256': pool['protocol_sha256'], 'runner_revision': pool['runner_revision'],
                'asset_manifest_sha256': pool['asset_manifest_sha256']}
    require(card == expected, 'pool card differs')
    observation_raw = args.pool_observation.read_bytes()
    observation = strict_json(observation_raw)
    require(type(observation) is dict and set(observation) == {'schema', 'path', 'revision', 'sha256'}
            and observation['schema'] == 'flan-sentiment-pool-observation-v1', 'pool observation differs')
    _source_descriptor({key: observation[key] for key in ('path', 'revision', 'sha256')})
    require(hashlib.sha256(card_raw).hexdigest() == observation['sha256'], 'pool card observation digest differs')
    _require_commit(args.repo, observation['revision'])
    published = subprocess.check_output(['git', '--no-replace-objects', 'show',
                                         f"{observation['revision']}:{observation['path']}"], cwd=args.repo)
    require(published == card_raw, 'published pool card differs')
    from .flan_t5_feasibility import verify_assets
    verify_assets(args.asset_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.asset_dir), local_files_only=True,
                                              use_fast=True, trust_remote_code=False)
    require(type(tokenizer).__name__ == 'T5TokenizerFast', 'tokenizer class differs')
    selection = select(pool, tokenizer, exposures, card['commitment'])
    selection_salt = secrets.token_bytes(32)
    selected_card = {'schema': 'flan-sentiment-selection-card-v1',
                     'commitment': selection_commitment(selection, selection_salt),
                     'pool_commitment': card['commitment'], 'protocol_sha256': pool['protocol_sha256'],
                     'runner_revision': pool['runner_revision'], 'asset_manifest_sha256': pool['asset_manifest_sha256']}
    selection_bytes = canonical(selection)
    selected_card_bytes = canonical(selected_card) + b'\n'
    require(len(pool_raw) + len(card_raw) + len(observation_raw) + len(selection_bytes)
            + len(selected_card_bytes) + 64 + 256 * 1024 <= 8 * 1024 * 1024,
            'selection custody limit exceeded')
    _write_new(args.selection, selection_bytes)
    _write_new(args.selection_salt, selection_salt)
    _write_new(args.selection_card, selected_card_bytes)
    require(args.pool.read_bytes() == pool_raw and args.pool_salt.read_bytes() == salt
            and args.pool_card.read_bytes() == card_raw and args.pool_observation.read_bytes() == observation_raw
            and args.selection.read_bytes() == selection_bytes and args.selection_salt.read_bytes() == selection_salt
            and args.selection_card.read_bytes() == selected_card_bytes
            and selection_commitment(strict_json(selection_bytes), args.selection_salt.read_bytes())
            == selected_card['commitment'], 'selection commitment readback differs')
    print(json.dumps({'status': 'selection-committed', 'commitment': selected_card['commitment'],
                      'enough_selected': len(selection['selected_units']) == 4}))


if __name__ == '__main__':
    main()
