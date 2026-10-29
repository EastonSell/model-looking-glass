"""Independent tokenizer-only replay of a closed FLAN sentiment selection.

This reads released private records, but writes aggregate counts and source
hashes only. It does not load model weights or authenticate execution/timing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

PROTOCOL = '83d666bc67af0951d21eceaa8efb8e7cb16f53a40aefdb72eff03f8b57bd7813'
ASSET_MANIFEST = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
SOURCE = '6fe4cc4e16ece717d672d8fc73fd3860e76ed459'
SIDES = ('P1', 'N1', 'P2', 'N2')
INSTRUCTIONS = ('Classify the sentiment of this review as positive or negative:',
                'Is this review positive or negative?')
OPPOSITE = {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
SAME = {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(raw):
    def pairs(items):
        output = {}
        for key, val in items:
            check(key not in output, 'duplicate JSON key')
            output[key] = val
        return output
    def bad(value):
        raise ValueError('nonfinite JSON')
    def finite_float(value):
        parsed = float(value)
        check(math.isfinite(parsed), 'nonfinite JSON float')
        return parsed
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad, parse_float=finite_float)


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def commit(domain, obj, salt):
    check(len(salt) == 32, 'salt length')
    return sha(domain + b'\0' + salt + b'\0' + canon(obj))


def git_blob(repo, rev, path):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git', '--no-replace-objects', 'show', f'{rev}:{path}'],
                                   cwd=repo, env=env, timeout=60)


def collect(value, rows, words):
    if type(value) is dict:
        ids = value.get('input_ids')
        if ids is not None:
            check(type(ids) is list and len(ids) > 1 and
                  all(type(x) is int and 0 <= x < 32128 for x in ids) and
                  ids[-1] == 1 and ids.count(1) == 1, 'bad exposure IDs')
            rows.append(tuple(ids[:-1]))
        for key, child in value.items():
            if key in ('text', 'subject', 'adjective', 'word', 'words', 'first_word',
                       'second_word', 'source_words', 'target_words'):
                values = child if type(child) is list else [child]
                for item in values:
                    if type(item) is str:
                        words.update(re.findall(r'[^\W\d_]+', unicodedata.normalize('NFC', item).casefold()))
            collect(child, rows, words)
    elif type(value) is list:
        for child in value:
            collect(child, rows, words)


def span_index(offsets, start, end, text, mask, specials, ids, special_ids):
    hits = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if len(hits) != 1:
        return None, len(hits)
    i = hits[0]
    allowed = [[start, end]]
    if start and text[start - 1] == ' ':
        allowed.append([start - 1, end])
    if offsets[i] in allowed and mask[i] == 1 and specials[i] == 0 and ids[i] not in special_ids:
        return i, 1
    return None, 1


def public_blob(repo, observation, card):
    check(sha(card) == observation['sha256'], 'card observation hash')
    check(git_blob(repo, observation['revision'], observation['path']) == card, 'public card blob')


def stage(root, name, revision):
    directory = root / (name + '-observer')
    exit_raw = (directory / 'stage-exit.json').read_bytes()
    record = load(exit_raw)
    runtime = (directory / 'stage-runtime.json').read_bytes()
    outer = load((root / (name + '-outer-exit.json')).read_bytes())
    check(record['status'] == 'completed' and record['exit_code'] == 0 and
          record['reaped'] is True and record['timed_out'] is False and
          record['child_prefix_empty_after'] is True and record['selector_revision'] == revision and
          record['stage_receipt_sha256'] == sha(runtime), 'stage exit/runtime')
    check(outer['exit_code'] == 0 and outer['stage'] == name.replace('commit-pool', 'commit-pool'),
          'outer stage exit')
    for stream in ('stdout', 'stderr'):
        raw = (directory / (stream + '.log')).read_bytes()
        saved = record['logs'][stream]
        check(saved['finished'] is True and saved['truncated'] is False and
              saved['retained_bytes'] == len(raw) and saved['sha256'] == sha(raw), 'stage log binding')
    return {'stage_exit_sha256': sha(exit_raw), 'runtime_sha256': sha(runtime),
            'direct_exit_code': 0, 'outer_exit_code': 0}


def run(root, repo, assets):
    from tokenizers import Tokenizer
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    pool_raw = (root / 'pool.json').read_bytes()
    sel_raw = (root / 'selection.json').read_bytes()
    pool, selection = load(pool_raw), load(sel_raw)
    check(pool_raw in (canon(pool), canon(pool) + b'\n') and
          sel_raw in (canon(selection), canon(selection) + b'\n'), 'noncanonical custody file')
    pool_card_raw, sel_card_raw = (root / 'pool-card.json').read_bytes(), (root / 'selection-card.json').read_bytes()
    pool_card, sel_card = load(pool_card_raw), load(sel_card_raw)
    pool_salt, sel_salt = (root / 'pool-salt.bin').read_bytes(), (root / 'selection-salt.bin').read_bytes()
    check(pool_salt != sel_salt, 'identical salts')
    pool_commit = commit(b'FLAN-SENTIMENT-HOLDOUT-V1', pool, pool_salt)
    sel_commit = commit(b'FLAN-SENTIMENT-SELECTION-V1', selection, sel_salt)
    check(pool_card['commitment'] == selection['pool_commitment'] == sel_card['pool_commitment'] == pool_commit,
          'pool commitment')
    check(sel_card['commitment'] == sel_commit, 'selection commitment')
    for record in (pool, selection, pool_card, sel_card):
        check(record['protocol_sha256'] == PROTOCOL and record['runner_revision'] == SOURCE and
              record['asset_manifest_sha256'] == ASSET_MANIFEST, 'source/asset binding')
    check(selection['exposure_sources'] == pool['exposure_sources'], 'exposure identities differ')
    check(pool['schema'] == 'flan-sentiment-private-pool-v1' and
          selection['schema'] == 'flan-sentiment-private-selection-v1', 'schema')
    public_blob(repo, load((root / 'pool-observation.json').read_bytes()), pool_card_raw)
    public_blob(repo, load((root / 'selection-observation.json').read_bytes()), sel_card_raw)

    manifest_raw = (assets / 'asset-manifest.json').read_bytes()
    check(sha(manifest_raw) == ASSET_MANIFEST, 'asset manifest hash')
    manifest = load(manifest_raw)
    for name in ('config.json', 'generation_config.json', 'special_tokens_map.json',
                 'spiece.model', 'tokenizer.json', 'tokenizer_config.json'):
        raw = (assets / name).read_bytes()
        check(len(raw) == manifest['files'][name]['bytes'] and
              sha(raw) == manifest['files'][name]['sha256'], 'tokenizer asset hash')
    fast = AutoTokenizer.from_pretrained(str(assets), local_files_only=True, use_fast=True)
    rust = Tokenizer.from_file(str(assets / 'tokenizer.json'))
    check(fast.eos_token_id == 1 and fast.pad_token_id == 0, 'tokenizer special IDs')
    special_ids = set(fast.all_special_ids)

    source_rows, source_words, source_hashes = [], set(), []
    descriptors = pool['exposure_sources']
    check(len(descriptors) == 10 and len({d['path'] for d in descriptors}) == 10, 'source count')
    for desc in descriptors:
        raw = git_blob(repo, desc['revision'], desc['path'])
        check(sha(raw) == desc['sha256'], 'public source hash')
        rows, words = [], set()
        payloads = [load(line) for line in raw.splitlines() if line] if desc['path'].endswith('.jsonl') else [load(raw)]
        for payload in payloads:
            collect(payload, rows, words)
        source_rows.append(rows)
        source_words.update(words)
        source_hashes.append({'path': desc['path'], 'sha256': sha(raw), 'rows': len(rows)})
    check(sum(map(len, source_rows)) == 625, 'public source row count')

    units, screens = pool['units'], selection['unit_screens']
    check(len(units) == len(screens) == 12, 'unit count')
    seen, qualifying, failures, overlap_hist = set(), [], Counter(), Counter()
    total_rows = comparisons = conflicting = eligible_groups = 0
    for index, (unit, screen) in enumerate(zip(units, screens, strict=True)):
        unit_id = f'u{index:02d}'
        check(unit['id'] == screen['unit_id'] == unit_id and type(unit['semantic_ok']) is bool and
              screen['semantic_ok'] is unit['semantic_ok'] and screen['lexical_exclusion_ok'] is True,
              'unit/screen identity')
        for word in (unit['subject'], *(unit['adjectives'][s] for s in SIDES)):
            check(type(word) is str and re.fullmatch(r'[a-z]{1,24}', word) is not None and
                  word not in seen and word not in source_words, 'lexical exclusion')
            seen.add(word)
        check(len(screen['instructions']) == 2, 'instruction count')
        group_flags = []
        for instruction, group in enumerate(screen['instructions']):
            check(group['instruction_index'] == instruction and len(group['cases']) == 4,
                  'instruction or case count')
            reasons = []
            if not unit['semantic_ok']:
                reasons.append('semantic-rejected')
            case_results = []
            for side, case in zip(SIDES, group['cases'], strict=True):
                text = f"{INSTRUCTIONS[instruction]} The {unit['subject']} was {unit['adjectives'][side]}."
                check(case['side'] == side and case['text'] == text, 'case text/order')
                encoded = fast(text, return_offsets_mapping=True, return_special_tokens_mask=True)
                values = {k: [list(v) if isinstance(v, tuple) else v for v in encoded[k]]
                          for k in ('input_ids', 'attention_mask', 'special_tokens_mask', 'offset_mapping')}
                check(all(case[k] == v for k, v in values.items()), 'HF retokenization mismatch')
                r = rust.encode(text, add_special_tokens=True)
                check(list(r.ids) == values['input_ids'] and list(r.attention_mask) == values['attention_mask'] and
                      list(r.special_tokens_mask) == values['special_tokens_mask'] and
                      [list(x) for x in r.offsets] == values['offset_mapping'], 'Rust retokenization mismatch')
                ids, mask, smask, offsets = (values['input_ids'], values['attention_mask'],
                                              values['special_tokens_mask'], values['offset_mapping'])
                check(all(type(x) is int for x in ids + mask + smask) and ids[-1] == 1 and
                      ids.count(1) == 1 and smask[-1] == 1, 'token/mask types or EOS')
                subject_start = len(INSTRUCTIONS[instruction]) + len(' The ')
                adjective_start = subject_start + len(unit['subject']) + len(' was ')
                subject, _ = span_index(offsets, subject_start, subject_start + len(unit['subject']),
                                        text, mask, smask, ids, special_ids)
                adjective, overlaps = span_index(offsets, adjective_start,
                                                adjective_start + len(unit['adjectives'][side]),
                                                text, mask, smask, ids, special_ids)
                check(case['subject_index'] == subject and case['adjective_index'] == adjective,
                      'saved site index mismatch')
                if adjective is None:
                    overlap_hist[str(overlaps)] += 1
                case_results.append((ids, mask, smask, adjective, subject))
                total_rows += 1
            if any(len(ids) > 64 or any(v != 1 for v in mask) for ids, mask, _, _, _ in case_results):
                reasons.append('input-length-or-mask')
            adjectives, subjects = [x[3] for x in case_results], [x[4] for x in case_results]
            if None in adjectives or len(set(adjectives)) != 1:
                reasons.append('adjective-alignment')
            if None in subjects or len(set(subjects)) != 1:
                reasons.append('subject-alignment')
            if None not in adjectives and None not in subjects and (adjectives[0] == subjects[0] or
                                                                    0 in (adjectives[0], subjects[0])):
                reasons.append('sites-not-distinct')
            if any(mask[0] != 1 or smask[0] != 0 or ids[0] in special_ids
                   for ids, mask, smask, _, _ in case_results):
                reasons.append('instruction-site')
            reference = case_results[0]
            if None not in adjectives and len(set(adjectives)) == 1 and all(
                    len(ids) == len(reference[0]) and mask == reference[1] and smask == reference[2]
                    for ids, mask, smask, _, _ in case_results):
                pos = adjectives[0]
                if (len({x[0][pos] for x in case_results}) != 4 or any(
                        [i for i, (a, b) in enumerate(zip(reference[0], x[0], strict=True)) if a != b] != [pos]
                        for x in case_results[1:])):
                    reasons.append('token-difference')
            else:
                reasons.append('token-difference')
            check(len(group['source_comparisons']) == len(source_rows), 'source comparison count')
            any_conflict = False
            for source_index, (record, prior) in enumerate(zip(group['source_comparisons'], source_rows, strict=True)):
                flags = []
                for ids, _, _, _, _ in case_results:
                    candidate = tuple(ids[:-1])
                    flags.extend(candidate[:min(len(candidate), len(row))] == row[:min(len(candidate), len(row))]
                                 for row in prior)
                count = 4 * len(prior)
                conflict = any(flags)
                check(type(record['source_index']) is int and record['source_index'] == source_index and
                      type(record['comparisons']) is int and record['comparisons'] == count and
                      type(record['conflict']) is bool and record['conflict'] is conflict, 'source comparison')
                comparisons += count
                conflicting += sum(flags)
                any_conflict |= conflict
            if any_conflict:
                reasons.append('prior-input-prefix')
            check(group['failure_reasons'] == reasons and type(group['eligible']) is bool and
                  group['eligible'] is (not reasons), 'group eligibility mismatch')
            failures.update(reasons)
            eligible_groups += not reasons
            group_flags.append(not reasons)
        if unit['semantic_ok'] and all(group_flags):
            qualifying.append(unit_id)
    selected = qualifying[:4]
    mapping = [{'unit_id': item, 'opposite': dict(OPPOSITE), 'same': dict(SAME)} for item in selected]
    check(selection['selected_units'] == selected and selection['donor_mapping'] == mapping,
          'first-four selection/mapping')
    check(total_rows == 96 and len(selected) < 4, 'selection outcome')
    stage_pool = stage(root, 'commit-pool', SOURCE)
    stage_select = stage(root, 'select', SOURCE)
    replay_raw = (root / 'independent-selection-replay.json').read_bytes()
    replay, closure = load(replay_raw), load((root / 'selection-incomplete-closure.json').read_bytes())
    check(replay['status'] == 'consistent' and replay['selection_count'] == len(selected) and
          replay['token_rows'] == total_rows and replay['pool_commitment'] == pool_commit and
          replay['selection_commitment'] == sel_commit, 'independent replay binding')
    check(closure['status'] == 'selection-incomplete' and closure['qualifying_units'] == len(selected) and
          closure['required_units'] == 4 and closure['screened_units'] == len(units) and
          closure['screened_rows'] == total_rows and closure['pool_card_sha256'] == sha(pool_card_raw) and
          closure['selection_card_sha256'] == sha(sel_card_raw) and
          closure['independent_replay_sha256'] == sha(replay_raw), 'closure binding')
    return {'schema': 'flan-sentiment-selection-second-independent-audit-v1',
            'status': 'selection-incomplete', 'screened_units': len(units),
            'screened_instruction_groups': len(units) * 2, 'screened_rows': total_rows,
            'qualifying_units': len(selected), 'required_units': 4,
            'semantic_rejected_units': sum(not u['semantic_ok'] for u in units),
            'eligible_instruction_groups': eligible_groups, 'failure_reasons_by_group': dict(failures),
            'unaligned_adjective_rows_by_overlap_count': dict(sorted(overlap_hist.items())),
            'source_count': len(source_rows), 'source_rows': sum(map(len, source_rows)),
            'all_case_source_comparisons': comparisons, 'prefix_conflicts': conflicting,
            'pool_card_sha256': sha(pool_card_raw), 'selection_card_sha256': sha(sel_card_raw),
            'pool_commitment': pool_commit, 'selection_commitment': sel_commit,
            'closure_sha256': sha((root / 'selection-incomplete-closure.json').read_bytes()),
            'independent_selection_replay_sha256': sha(replay_raw),
            'public_sources': source_hashes, 'pool_stage': stage_pool, 'selection_stage': stage_select,
            'asset_manifest_sha256': ASSET_MANIFEST, 'python': sys.version.split()[0],
            'tokenizers': tokenizers.__version__, 'transformers': transformers.__version__,
            'scope': ['saved commitment, source, disposition and tokenizer consistency; no model loaded',
                      'public card Git blobs do not authenticate publication chronology or execution',
                      'stage exits and no-forward disposition are cooperative observations',
                      'semantic judgment, unknown exposure and pretraining are not authenticated']}


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'repo', 'assets', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    report = run(args.root, args.repo, args.assets)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as file:
        check(file.write(raw) == len(raw), 'short report write')
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps({'status': report['status'], 'sha256': sha(raw),
                      'qualifying_units': report['qualifying_units'], 'rows': report['screened_rows'],
                      'failure_reasons_by_group': report['failure_reasons_by_group'],
                      'unaligned_adjective_rows_by_overlap_count': report['unaligned_adjective_rows_by_overlap_count'],
                      'comparisons': report['all_case_source_comparisons'], 'conflicts': report['prefix_conflicts']}))


if __name__ == '__main__':
    main()
