"""Independently replay a closed FLAN sentiment tokenizer-only selection.

This checker imports neither the selector nor a model. It reports counts and
public salted commitments only; it does not publish private cases or salts.
Public-source origin/chronology and interpreter secrecy require separate
custodian observations.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import unicodedata

INSTRUCTIONS = ('Classify the sentiment of this review as positive or negative:',
                'Is this review positive or negative?')
SIDES = ('P1', 'N1', 'P2', 'N2')
OPPOSITE = {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
SAME = {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}
PROTOCOL = '83d666bc67af0951d21eceaa8efb8e7cb16f53a40aefdb72eff03f8b57bd7813'
ASSET_MANIFEST = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
SELECTOR_SHA = '4f0dc5b4e875a6e2f725639b3d2f336b0d17a882ac9b601ea4ab3ab93ee4da17'
SELECTOR_DOC_SHA = 'b67789859a1a4662359f32bdad91d3ffc8795d0c0bdeef147666af5f73ecb841'
STAGE_SOURCES = {
    'tools/run_flan_sentiment_selector_stage_v1.py',
    'tools/run_flan_sentiment_selector_v1.py',
    'looking_glass/__init__.py',
    'looking_glass/flan_t5_sentiment_selector_v1.py',
    'looking_glass/flan_t5_sentiment_screen.py',
    'looking_glass/flan_t5_feasibility.py',
    'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md',
    'docs/FLAN_T5_SENTIMENT_SELECTOR_V1.md',
}
STAGE_SCOPE = 'Same-process version observation; external direct-exit and custody checks remain required.'
EXIT_SCOPE = 'Local direct-child observation; no execution authentication.'
REQUIRED = {
    'docs/FLAN_T5_PUBLIC_CASES_V1.json': 'e00d77cf45885b10e19fe689b8cdb13d2d36dd038c7ab2c1701f9367eb2fe9fc',
    'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json': 'a0e320c9d24a8940fbca05cbc4ed6e3f2661b5b44b2e62e1140832c0e99e7cd6',
    'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json': '629036188d62403669fe33291fa9e2a8759145be828918db25fa3ea7c0555a18',
    'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json': '8d1ea3b124085ca8817dc80044dca6fd1e897bda7294510ad99e419c01e7708b',
    'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json': 'adabba92e451fb0c27415b8ec67b67134b59bb5aac293b3c056936cfe872c0a2',
    'results/flan-t5-heldout-response-cpu-v1-20261029/custody/pool.json': '90b0f249f93bc48895431b9d60f656922ec199de4bf78cf50548ce4eb75e1386',
    'results/flan-t5-heldout-response-cpu-v1-20261029/custody/screen.json': 'ce163cda5a2cabc2ef2489e33a585f0885f70b9967395da70a40706c398577db',
    'results/flan-t5-sentiment-screen-cpu-v1-20261029/child-records/events.jsonl': 'eb17ca8482e0c28ed4d6146285aaf33fa63db3437f0fb308c5becfa5e50d3d6b',
}


def require(condition, category):
    if not condition:
        raise ValueError(category)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite JSON constant')
    def finite(value):
        value = float(value)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_file(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def token_body(row):
    require(type(row) is list and len(row) > 1
            and all(type(token) is int and 0 <= token < 32128 for token in row)
            and row[-1] == 1 and row.count(1) == 1, 'invalid EOS token row')
    return tuple(row[:-1])


def conflicts(left, right):
    left, right = token_body(left), token_body(right)
    count = min(len(left), len(right))
    return left[:count] == right[:count]


def descriptor_ok(item):
    if type(item) is not dict or set(item) != {'path', 'revision', 'sha256'}:
        return False
    path, revision, sha = item['path'], item['revision'], item['sha256']
    return (type(path) is str and 0 < len(path.encode('utf-8')) <= 256 and '\\' not in path and ':' not in path
            and not PurePosixPath(path).is_absolute() and all(part not in ('', '.', '..') for part in path.split('/'))
            and type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision) is not None
            and type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha) is not None)


def exposure_from_bytes(descriptor, raw):
    require(descriptor_ok(descriptor) and digest(raw) == descriptor['sha256'], 'public source identity')
    require(len(raw) <= 64 * 1024 * 1024, 'public source size')
    payloads = ([strict_json(line) for line in raw.splitlines() if line]
                if descriptor['path'].endswith('.jsonl') else [strict_json(raw)])
    fields = {'text', 'subject', 'adjective', 'word', 'words', 'first_word', 'second_word',
              'source_words', 'target_words'}
    words, rows = set(), []
    def walk(value):
        if type(value) is dict:
            if 'input_ids' in value:
                token_body(value['input_ids'])
                rows.append(value['input_ids'])
            for key, child in value.items():
                if key in fields:
                    for candidate in child if type(child) is list else [child]:
                        if type(candidate) is str:
                            normalized = unicodedata.normalize('NFC', candidate).casefold()
                            words.update(re.findall(r'[^\W\d_]+', normalized))
                walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
    for value in payloads:
        walk(value)
    return {'descriptor': descriptor, 'rows': rows, 'words': sorted(words)}


def git_context(repo):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_CONFIG_NOSYSTEM': '1',
                'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_OPTIONAL_LOCKS': '0'})
    command = ['git.exe' if os.name == 'nt' else 'git', '--no-replace-objects', '-C', str(repo)]
    return command, env


def checked_git_commit(repo, revision):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision) is not None, 'Git revision format')
    command, env = git_context(repo)
    try:
        kind = subprocess.check_output(command + ['cat-file', '-t', revision],
                                       env=env, stderr=subprocess.DEVNULL, timeout=30).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('Git revision unavailable') from error
    require(kind == b'commit', 'Git revision is not a commit')


def public_git_bytes(repo, descriptor, limit=64 * 1024 * 1024):
    """Bind separately fetched public bytes to an extant, no-replacement Git commit.

    This assumes a trusted Git executable/checkout and does not prove that the
    commit was public at a particular time; that observation is separate.
    """
    checked_git_commit(repo, descriptor['revision'])
    command, env = git_context(repo)
    try:
        spec = descriptor['revision'] + ':' + descriptor['path']
        size_raw = subprocess.check_output(command + ['cat-file', '-s', spec],
                                           env=env, stderr=subprocess.DEVNULL, timeout=30).strip()
        require(size_raw.isdigit() and int(size_raw) <= limit, 'public Git blob size')
        raw = subprocess.check_output(command + ['cat-file', 'blob', spec],
                                      env=env, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('public Git source unavailable') from error
    require(len(raw) == int(size_raw), 'public Git blob length')
    return raw


def exposures_from_files(descriptors, root, git_repo):
    require(type(descriptors) is list and 1 <= len(descriptors) <= 64
            and all(descriptor_ok(item) for item in descriptors), 'public source descriptors')
    paths = [item['path'] for item in descriptors]
    require(len(set(paths)) == len(paths) and set(REQUIRED) <= set(paths), 'public source coverage')
    require(all(item['path'] not in REQUIRED or item['sha256'] == REQUIRED[item['path']]
                for item in descriptors), 'required public source pins')
    root, git_repo = root.resolve(strict=True), git_repo.resolve(strict=True)
    require(git_repo.is_dir(), 'public Git checkout')
    result = []
    for item in descriptors:
        path = root.joinpath(*item['path'].split('/')).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), 'public source path')
        require(path.stat().st_size <= 64 * 1024 * 1024, 'public source size')
        raw = read_bounded(path, 64 * 1024 * 1024)
        require(raw == public_git_bytes(git_repo, item), 'public source Git blob mismatch')
        result.append(exposure_from_bytes(item, raw))
    return result


def validate_pool(pool, exposures, expected_runner):
    require(type(pool) is dict and set(pool) == {'schema', 'protocol_sha256', 'runner_revision',
            'asset_manifest_sha256', 'exposure_sources', 'units'}, 'pool schema')
    require(pool['schema'] == 'flan-sentiment-private-pool-v1' and pool['protocol_sha256'] == PROTOCOL
            and pool['asset_manifest_sha256'] == ASSET_MANIFEST and pool['runner_revision'] == expected_runner
            and type(expected_runner) is str and re.fullmatch('[0-9a-f]{40}', expected_runner) is not None,
            'pool source identity')
    sources = pool['exposure_sources']
    require(type(sources) is list and 1 <= len(sources) <= 64 and all(descriptor_ok(x) for x in sources)
            and len({x['path'] for x in sources}) == len(sources), 'pool exposure descriptors')
    require(type(exposures) is list and len(exposures) == len(sources), 'public source count')
    seen_words = set()
    for descriptor, exposure in zip(sources, exposures):
        require(type(exposure) is dict and set(exposure) == {'descriptor', 'rows', 'words'}
                and canonical(exposure['descriptor']) == canonical(descriptor)
                and type(exposure['rows']) is list and type(exposure['words']) is list, 'public source records')
        for row in exposure['rows']:
            token_body(row)
        for word in exposure['words']:
            require(type(word) is str, 'public word type')
            seen_words.add(unicodedata.normalize('NFC', word).casefold())
    require(type(pool['units']) is list and len(pool['units']) == 12, 'pool unit count')
    used = set()
    for number, unit in enumerate(pool['units']):
        require(type(unit) is dict and set(unit) == {'id', 'subject', 'adjectives', 'semantic_ok', 'semantic_reason'}
                and unit['id'] == f'u{number:02}' and type(unit['semantic_ok']) is bool
                and type(unit['semantic_reason']) is str
                and unicodedata.normalize('NFC', unit['semantic_reason']) == unit['semantic_reason']
                and len(unit['semantic_reason'].encode('utf-8')) <= 256
                and (unit['semantic_ok'] or bool(unit['semantic_reason']))
                and type(unit['adjectives']) is dict and set(unit['adjectives']) == set(SIDES), 'pool unit schema')
        for word in [unit['subject'], *(unit['adjectives'][s] for s in SIDES)]:
            require(type(word) is str and re.fullmatch('[a-z]{1,24}', word) is not None
                    and word not in used and word not in seen_words, 'pool lexical exclusion')
            used.add(word)


def one_case(unit, instruction_index, side, tokenizer):
    instruction = INSTRUCTIONS[instruction_index]
    text = f"{instruction} The {unit['subject']} was {unit['adjectives'][side]}."
    encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    keys = ('input_ids', 'attention_mask', 'special_tokens_mask', 'offset_mapping')
    require(all(key in encoded for key in keys), 'tokenizer fields')
    fields = {key: [list(v) if isinstance(v, tuple) else v for v in encoded[key]] for key in keys}
    size = len(fields['input_ids'])
    require(0 < size <= 256 and all(len(fields[key]) == size for key in keys)
            and all(type(v) is int for v in fields['input_ids'])
            and all(type(v) is int and v in (0, 1) for key in ('attention_mask', 'special_tokens_mask') for v in fields[key])
            and all(type(v) is list and len(v) == 2 and all(type(x) is int and 0 <= x <= len(text) for x in v)
                    and v[0] <= v[1] for v in fields['offset_mapping']), 'tokenizer shape or types')
    token_body(fields['input_ids'])
    require(fields['special_tokens_mask'][-1] == 1, 'EOS special mask')
    subject_start = len(instruction) + len(' The ')
    adjective_start = subject_start + len(unit['subject']) + len(' was ')
    special = set(tokenizer.all_special_ids)
    def position(start, word):
        end = start + len(word)
        overlaps = [i for i, (left, right) in enumerate(fields['offset_mapping']) if left < end and right > start]
        if len(overlaps) != 1:
            return None
        index = overlaps[0]
        valid = [[start, end]] + ([[start - 1, end]] if start and text[start - 1] == ' ' else [])
        return (index if fields['offset_mapping'][index] in valid and fields['attention_mask'][index] == 1
                and fields['special_tokens_mask'][index] == 0 and fields['input_ids'][index] not in special else None)
    return {'side': side, 'text': text, **fields,
            'adjective_index': position(adjective_start, unit['adjectives'][side]),
            'subject_index': position(subject_start, unit['subject'])}


def one_group(unit, instruction, tokenizer, exposures):
    cases = [one_case(unit, instruction, side, tokenizer) for side in SIDES]
    reasons = []
    if not unit['semantic_ok']:
        reasons.append('semantic-rejected')
    if any(len(case['input_ids']) > 64 or any(x != 1 for x in case['attention_mask']) for case in cases):
        reasons.append('input-length-or-mask')
    adjectives = [case['adjective_index'] for case in cases]
    subjects = [case['subject_index'] for case in cases]
    if None in adjectives or len(set(adjectives)) != 1:
        reasons.append('adjective-alignment')
    if None in subjects or len(set(subjects)) != 1:
        reasons.append('subject-alignment')
    if None not in adjectives and None not in subjects and (adjectives[0] == subjects[0] or 0 in (adjectives[0], subjects[0])):
        reasons.append('sites-not-distinct')
    if any(case['attention_mask'][0] != 1 or case['special_tokens_mask'][0] != 0
           or case['input_ids'][0] in tokenizer.all_special_ids for case in cases):
        reasons.append('instruction-site')
    reference = cases[0]
    comparable = (None not in adjectives and len(set(adjectives)) == 1
                  and all(len(case['input_ids']) == len(reference['input_ids'])
                          and case['attention_mask'] == reference['attention_mask']
                          and case['special_tokens_mask'] == reference['special_tokens_mask'] for case in cases))
    if comparable:
        position = adjectives[0]
        if (len({case['input_ids'][position] for case in cases}) != 4
                or any([i for i, (left, right) in enumerate(zip(reference['input_ids'], case['input_ids']))
                       if left != right] != [position] for case in cases[1:])):
            reasons.append('token-difference')
    else:
        reasons.append('token-difference')
    comparisons = []
    for source_index, exposure in enumerate(exposures):
        collision = False
        count = 0
        for case in cases:
            for prior in exposure['rows']:
                collided = conflicts(case['input_ids'], prior)
                collision = collision or collided
                count += 1
        comparisons.append({'source_index': source_index, 'comparisons': count, 'conflict': collision})
    if any(item['conflict'] for item in comparisons):
        reasons.append('prior-input-prefix')
    return {'instruction_index': instruction, 'cases': cases, 'source_comparisons': comparisons,
            'eligible': not reasons, 'failure_reasons': reasons}


def reconstruct(pool, tokenizer, exposures, pool_commitment, expected_runner):
    validate_pool(pool, exposures, expected_runner)
    require(type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id == 1
            and type(tokenizer.pad_token_id) is int and tokenizer.pad_token_id == 0
            and type(tokenizer.all_special_ids) is list
            and all(type(item) is int for item in tokenizer.all_special_ids), 'tokenizer special IDs')
    screens = [{'unit_id': unit['id'], 'semantic_ok': unit['semantic_ok'], 'lexical_exclusion_ok': True,
                'instructions': [one_group(unit, index, tokenizer, exposures) for index in range(2)]}
               for unit in pool['units']]
    selected = [row['unit_id'] for row in screens if row['semantic_ok']
                and all(group['eligible'] for group in row['instructions'])][:4]
    mapping = [{'unit_id': ident, 'opposite': dict(OPPOSITE), 'same': dict(SAME)} for ident in selected]
    return {'schema': 'flan-sentiment-private-selection-v1', 'pool_commitment': pool_commitment,
            'protocol_sha256': pool['protocol_sha256'], 'runner_revision': pool['runner_revision'],
            'asset_manifest_sha256': pool['asset_manifest_sha256'], 'exposure_sources': pool['exposure_sources'],
            'unit_screens': screens, 'selected_units': selected, 'donor_mapping': mapping}


def check_saved(pool_raw, pool_salt, pool_card_raw, selection_raw, selection_salt, selection_card_raw,
                exposures, tokenizer, expected_runner):
    require(len(pool_raw) <= 256 * 1024 and len(selection_raw) <= 8 * 1024 * 1024
            and type(pool_salt) is bytes and len(pool_salt) == 32
            and type(selection_salt) is bytes and len(selection_salt) == 32, 'custody sizes')
    pool, pool_card = strict_json(pool_raw), strict_json(pool_card_raw)
    selection, selection_card = strict_json(selection_raw), strict_json(selection_card_raw)
    require(canonical(pool) == pool_raw and canonical(selection) == selection_raw
            and canonical(pool_card) + b'\n' == pool_card_raw
            and canonical(selection_card) + b'\n' == selection_card_raw, 'noncanonical custody bytes')
    pool_hash = digest(b'FLAN-SENTIMENT-HOLDOUT-V1\0' + pool_salt + b'\0' + pool_raw)
    expected_pool_card = {'schema': 'flan-sentiment-pool-card-v1', 'commitment': pool_hash,
                          'protocol_sha256': PROTOCOL, 'runner_revision': expected_runner,
                          'asset_manifest_sha256': ASSET_MANIFEST}
    require(canonical(pool_card) == canonical(expected_pool_card), 'pool commitment')
    predicted = reconstruct(pool, tokenizer, exposures, pool_hash, expected_runner)
    require(canonical(selection) == canonical(predicted), 'saved selection mismatch')
    selection_hash = digest(b'FLAN-SENTIMENT-SELECTION-V1\0' + selection_salt + b'\0' + selection_raw)
    expected_selection_card = {'schema': 'flan-sentiment-selection-card-v1', 'commitment': selection_hash,
                               'pool_commitment': pool_hash, 'protocol_sha256': PROTOCOL,
                               'runner_revision': expected_runner, 'asset_manifest_sha256': ASSET_MANIFEST}
    require(canonical(selection_card) == canonical(expected_selection_card), 'selection commitment')
    return {'schema': 'flan-t5-sentiment-independent-tokenizer-replay-v1', 'status': 'consistent',
            'unit_count': 12, 'token_rows': 96, 'selection_count': len(predicted['selected_units']),
            'source_count': len(exposures), 'source_rows': sum(len(source['rows']) for source in exposures),
            'enough_selected': len(predicted['selected_units']) == 4,
            'pool_commitment': pool_hash, 'selection_commitment': selection_hash,
            'protocol_sha256': PROTOCOL, 'runner_revision': expected_runner,
            'asset_manifest_sha256': ASSET_MANIFEST,
            'limitations': ['model-free saved consistency, not model execution or semantic judgment',
                            'public-source Git binding assumes a trusted checkout and does not prove publication timing',
                            'matching card bytes does not establish publication or observation chronology',
                            'version strings and asset hashes do not authenticate interpreter or native-code bytes',
                            'known exposure sources do not establish pretraining independence or secrecy',
                            'commitment bytes do not establish independent salt generation']}


def read_bounded(path, limit):
    require(path.stat().st_size <= limit, 'input size limit')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'input size limit')
    return raw


def checked_stage_receipt(raw, repo, expected_revision):
    stage = strict_json(raw)
    keys = {'schema', 'stage', 'status', 'selector_revision', 'source_sha256', 'versions',
            'pool_commitment', 'selection_commitment', 'scope'}
    require(type(stage) is dict and set(stage) == keys and canonical(stage) + b'\n' == raw
            and stage['schema'] == 'flan-t5-sentiment-selector-stage-runtime-v1'
            and stage['stage'] == 'select' and stage['status'] == 'completed'
            and stage['scope'] == STAGE_SCOPE, 'selector stage receipt')
    revision, sources, versions = stage['selector_revision'], stage['source_sha256'], stage['versions']
    require(type(expected_revision) is str and re.fullmatch('[0-9a-f]{40}', expected_revision) is not None
            and type(revision) is str and revision == expected_revision
            and type(sources) is dict and set(sources) == STAGE_SOURCES
            and type(versions) is dict and set(versions) == {'python', 'transformers', 'tokenizers'}
            and all(type(value) is str and 0 < len(value) <= 80 for value in versions.values())
            and all(type(stage[key]) is str and re.fullmatch('[0-9a-f]{64}', stage[key]) is not None
                    for key in ('pool_commitment', 'selection_commitment')), 'selector stage identity')
    required_pins = {'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md': PROTOCOL,
                     'docs/FLAN_T5_SENTIMENT_SELECTOR_V1.md': SELECTOR_DOC_SHA,
                     'looking_glass/flan_t5_sentiment_selector_v1.py': SELECTOR_SHA}
    require(all(sources[path] == pin for path,pin in required_pins.items()), 'selector source pins')
    for path,pin in sources.items():
        descriptor = {'path':path, 'revision':revision, 'sha256':pin}
        require(descriptor_ok(descriptor) and digest(public_git_bytes(repo, descriptor)) == pin,
                'selector stage Git source')
    return stage


def bind_stage_commitments(report, stage):
    require(report['pool_commitment'] == stage['pool_commitment']
            and report['selection_commitment'] == stage['selection_commitment'], 'selector stage commitments')


def checked_stage_exit(raw, stage_raw, stage, stdout, stderr):
    receipt = strict_json(raw)
    keys = {'schema', 'stage', 'status', 'stage_receipt_sha256', 'reaped', 'exit_code',
            'timed_out', 'exception_type', 'selector_revision', 'child_prefix_empty_after', 'logs', 'scope'}
    require(type(receipt) is dict and set(receipt) == keys and canonical(receipt) + b'\n' == raw
            and receipt['schema'] == 'flan-t5-sentiment-selector-stage-exit-v1'
            and receipt['stage'] == 'select' and receipt['status'] == 'completed'
            and receipt['stage_receipt_sha256'] == digest(stage_raw)
            and receipt['selector_revision'] == stage['selector_revision']
            and receipt['reaped'] is True and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
            and receipt['timed_out'] is False and receipt['exception_type'] is None
            and receipt['child_prefix_empty_after'] is True and receipt['scope'] == EXIT_SCOPE,
            'selector direct-exit receipt')
    logs = receipt['logs']
    require(type(logs) is dict and set(logs) == {'stdout', 'stderr'}, 'selector log inventory')
    for label, raw_log in (('stdout', stdout), ('stderr', stderr)):
        item = logs[label]
        require(type(item) is dict and set(item) == {'retained_bytes', 'truncated', 'error_type', 'finished', 'sha256'}
                and type(item['retained_bytes']) is int and item['retained_bytes'] == len(raw_log)
                and item['truncated'] is False and item['error_type'] is None and item['finished'] is True
                and type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None
                and item['sha256'] == digest(raw_log), 'selector retained log')
    return receipt


def verify_assets(directory):
    manifest_raw = (directory / 'asset-manifest.json').read_bytes()
    require(digest(manifest_raw) == ASSET_MANIFEST, 'asset manifest digest')
    manifest = strict_json(manifest_raw)
    files = {'config.json', 'generation_config.json', 'model.safetensors', 'special_tokens_map.json',
             'spiece.model', 'tokenizer.json', 'tokenizer_config.json'}
    require(manifest['repo'] == 'google/flan-t5-small'
            and manifest['revision'] == '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
            and type(manifest['files']) is dict and set(manifest['files']) == files, 'asset inventory')
    for name, item in manifest['files'].items():
        path = directory / name
        require(path.is_file() and type(item['bytes']) is int and path.stat().st_size == item['bytes']
                and hash_file(path) == item['sha256'], 'asset file digest')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'pool-salt', 'pool-card', 'selection', 'selection-salt', 'selection-card', 'runtime-receipt',
                 'stage-exit-receipt', 'stage-stdout', 'stage-stderr',
                 'public-source-root', 'public-source-git-repo', 'asset-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('runner-revision', 'selector-revision', 'replay-source-revision', 'replay-source-path'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode, 'isolated/no-bytecode runtime')
    replay_raw = read_bounded(Path(__file__).resolve(strict=True), 256 * 1024)
    replay_descriptor = {'path':args.replay_source_path, 'revision':args.replay_source_revision,
                         'sha256':digest(replay_raw)}
    require(descriptor_ok(replay_descriptor), 'replay source identity')
    require(public_git_bytes(args.public_source_git_repo.resolve(strict=True), replay_descriptor) == replay_raw,
            'replay source Git blob mismatch')
    runner_protocol = {'path':'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md',
                       'revision':args.runner_revision, 'sha256':PROTOCOL}
    require(descriptor_ok(runner_protocol)
            and digest(public_git_bytes(args.public_source_git_repo.resolve(strict=True), runner_protocol)) == PROTOCOL,
            'runner revision protocol binding')
    stage_raw = read_bounded(args.runtime_receipt, 64 * 1024)
    stage = checked_stage_receipt(stage_raw, args.public_source_git_repo.resolve(strict=True), args.selector_revision)
    exit_raw = read_bounded(args.stage_exit_receipt, 64 * 1024)
    checked_stage_exit(exit_raw, stage_raw, stage, read_bounded(args.stage_stdout, 1024 * 1024),
                       read_bounded(args.stage_stderr, 1024 * 1024))
    verify_assets(args.asset_dir)
    pool_raw = read_bounded(args.pool, 256 * 1024)
    pool = strict_json(pool_raw)
    require(canonical(pool) == pool_raw, 'noncanonical pool bytes')
    exposures = exposures_from_files(pool['exposure_sources'], args.public_source_root, args.public_source_git_repo)
    import platform
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    versions = {'python': platform.python_version(), 'transformers': transformers.__version__,
                'tokenizers': tokenizers.__version__}
    require(versions == stage['versions'], 'recorded screen runtime')
    tokenizer = AutoTokenizer.from_pretrained(str(args.asset_dir), local_files_only=True,
                                              use_fast=True, trust_remote_code=False)
    require(type(tokenizer).__name__ == 'T5TokenizerFast', 'tokenizer class')
    report = check_saved(pool_raw, read_bounded(args.pool_salt, 32), read_bounded(args.pool_card, 4096),
                         read_bounded(args.selection, 8 * 1024 * 1024), read_bounded(args.selection_salt, 32),
                         read_bounded(args.selection_card, 4096), exposures, tokenizer, args.runner_revision)
    bind_stage_commitments(report, stage)
    report['screen_runtime'] = versions
    report['screen_runtime_receipt_sha256'] = digest(stage_raw)
    report['screen_exit_receipt_sha256'] = digest(exit_raw)
    report['selector_source_revision'] = stage['selector_revision']
    report['replay_script_sha256'] = replay_descriptor['sha256']
    report['replay_source_revision'] = replay_descriptor['revision']
    report['replay_source_path'] = replay_descriptor['path']
    payload = canonical(report) + b'\n'
    with args.output.open('xb') as stream:
        remaining = memoryview(payload)
        while remaining:
            written = stream.write(remaining)
            require(type(written) is int and written > 0, 'short report write')
            remaining = remaining[written:]
        stream.flush()
        os.fsync(stream.fileno())
    require(args.output.read_bytes() == payload, 'report readback mismatch')
    print(json.dumps({'status': report['status'], 'report_sha256': digest(payload),
                      'enough_selected': report['enough_selected']}))


if __name__ == '__main__':
    main()
