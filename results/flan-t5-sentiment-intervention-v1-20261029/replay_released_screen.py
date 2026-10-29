"""Replay the released FLAN sentiment tokenizer screen; this loads no model.

Use a trusted checkout containing the recorded Git objects and the pinned local
asset directory. This checks saved bytes and tokenization, not process execution,
semantic truth, publication timing, secrecy, or unknown prior exposure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import types

READER_REV = '0380e885ed5997186ee85bad5ab18aaa38fb252c'
READER_SHA = '7c638b3715e60aebeda13f3e51a2c3457d4bc5eae0cdd0916d73c5db977bce7d'
RUNNER_REV = '6fe4cc4e16ece717d672d8fc73fd3860e76ed459'
POOL_REV = 'a4039ebc0262b2756f3885134017495f5a26d3a9'
SELECTION_REV = '332ea04e73ae977915ea152a8140a31373febb7d'
ROOT_PATH = 'results/flan-t5-sentiment-intervention-v1-20261029/custody/'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_bytes(repo, revision, path):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_CONFIG_NOSYSTEM': '1',
                'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_OPTIONAL_LOCKS': '0'})
    command = ['git.exe' if os.name == 'nt' else 'git', '--no-replace-objects', '-C', str(repo)]
    kind = subprocess.check_output(command + ['cat-file', '-t', revision], env=env, timeout=30).strip()
    require(kind == b'commit', 'source revision is not a commit')
    return subprocess.check_output(command + ['show', revision + ':' + path], env=env, timeout=30)


def load_reader(repo):
    path = repo / 'tools/flan_sentiment_independent_selection_v1.py'
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == READER_SHA, 'reader hash mismatch')
    require(raw == git_bytes(repo, READER_REV, 'tools/flan_sentiment_independent_selection_v1.py'),
            'reader Git blob mismatch')
    module = types.ModuleType('_released_flan_selection_reader')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def salt_bytes(path):
    raw = path.read_bytes()
    require(re.fullmatch(b'[0-9a-f]{64}\\n', raw) is not None, 'salt must be lowercase hex plus newline')
    return bytes.fromhex(raw.decode('ascii').strip())


def compare_card(reader, repo, custody, kind, revision):
    card = (custody / (kind + '-card.json')).read_bytes()
    obs = reader.strict_json((custody / (kind + '-observation.json')).read_bytes())
    require(obs == {'schema': 'flan-sentiment-' + kind + '-observation-v1',
                    'path': ROOT_PATH + kind + '-card.json', 'revision': revision,
                    'sha256': hashlib.sha256(card).hexdigest()}, 'public card observation mismatch')
    require(card == git_bytes(repo, revision, obs['path']), 'public card Git blob mismatch')
    return card


def check_draft(pool, candidates, draft, verdicts, exposures):
    excluded = {word for source in exposures for word in source['words']}
    used = set()
    def take(words, count):
        output = []
        for word in words:
            if word not in excluded and word not in used:
                used.add(word)
                output.append(word)
                if len(output) == count:
                    return output
        raise ValueError('released lexical candidate list is insufficient')
    subjects = take(candidates['subjects'], 12)
    positive = take(candidates['positive'], 24)
    negative = take(candidates['negative'], 24)
    units = [{'id': f'u{i:02}', 'subject': subjects[i],
              'adjectives': {'P1': positive[2*i], 'N1': negative[2*i],
                             'P2': positive[2*i+1], 'N2': negative[2*i+1]}}
             for i in range(12)]
    require(draft == {'schema': 'flan-sentiment-semantic-draft-v1', 'units': units}, 'draft mismatch')
    require(type(verdicts) is list and len(verdicts) == 12, 'semantic verdict count')
    for unit, entry, verdict in zip(pool['units'], units, verdicts):
        require({k: unit[k] for k in ('id', 'subject', 'adjectives')} == entry, 'pool draft mismatch')
        require(verdict == {k: unit[k] for k in ('id', 'semantic_ok', 'semantic_reason')},
                'recorded semantic verdict mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='trusted checkout with the recorded Git objects')
    parser.add_argument('--asset-dir', type=Path, required=True, help='pinned seven-file local asset directory')
    parser.add_argument('--evidence', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode, 'run with python -I -B')
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'CUDA_VISIBLE_DEVICES': '-1', 'TOKENIZERS_PARALLELISM': 'false'})
    repo = args.repo.resolve(strict=True)
    custody = args.evidence.resolve(strict=True) / 'custody'
    reader = load_reader(repo)
    protocol = git_bytes(repo, RUNNER_REV, 'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md')
    require(hashlib.sha256(protocol).hexdigest() == reader.PROTOCOL, 'protocol mismatch')
    pool_raw = (custody / 'pool.json').read_bytes()
    selection_raw = (custody / 'selection.json').read_bytes()
    pool = reader.strict_json(pool_raw)
    descriptors = reader.strict_json((custody / 'exposure-descriptors.json').read_bytes())
    require(descriptors == pool['exposure_sources'], 'descriptor list mismatch')
    require(type(descriptors) is list and len({d['path'] for d in descriptors}) == len(descriptors),
            'invalid descriptor list')
    by_path = {d['path']: d for d in descriptors}
    require(set(reader.REQUIRED) <= set(by_path)
            and all(by_path[p]['sha256'] == h for p, h in reader.REQUIRED.items()),
            'required public exposure source mismatch')
    exposures = [reader.exposure_from_bytes(d, git_bytes(repo, d['revision'], d['path'])) for d in descriptors]
    check_draft(pool, reader.strict_json((custody / 'lexical-candidates.json').read_bytes()),
                reader.strict_json((custody / 'semantic-draft.json').read_bytes()),
                reader.strict_json((custody / 'semantic-verdicts.json').read_bytes()), exposures)
    reader.verify_assets(args.asset_dir.resolve(strict=True))
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    recorded = reader.strict_json((args.evidence / 'process/select-stage-runtime.json').read_bytes())
    versions = {'python': platform.python_version(), 'transformers': transformers.__version__,
                'tokenizers': tokenizers.__version__}
    require(versions == recorded['versions'], 'tokenizer runtime versions differ')
    tokenizer = AutoTokenizer.from_pretrained(str(args.asset_dir), local_files_only=True,
                                              use_fast=True, trust_remote_code=False)
    require(type(tokenizer).__name__ == 'T5TokenizerFast', 'unexpected tokenizer class')
    result = reader.check_saved(pool_raw, salt_bytes(custody / 'pool-salt.hex'),
                                compare_card(reader, repo, custody, 'pool', POOL_REV),
                                selection_raw, salt_bytes(custody / 'selection-salt.hex'),
                                compare_card(reader, repo, custody, 'selection', SELECTION_REV),
                                exposures, tokenizer, RUNNER_REV)
    replay = reader.strict_json((custody / 'independent-selection-replay.json').read_bytes())
    for key in ('status', 'unit_count', 'token_rows', 'selection_count', 'source_count', 'source_rows',
                'enough_selected', 'pool_commitment', 'selection_commitment', 'protocol_sha256',
                'runner_revision', 'asset_manifest_sha256'):
        require(result[key] == replay[key], 'independent replay result mismatch: ' + key)
    closure = reader.strict_json((custody / 'selection-incomplete-closure.json').read_bytes())
    require(closure['status'] == 'selection-incomplete' and closure['qualifying_units'] == result['selection_count']
            and closure['required_units'] == 4 and not result['enough_selected'], 'no-forward closure mismatch')
    print(json.dumps({'status': result['status'], 'units': result['unit_count'],
                      'rows': result['token_rows'], 'qualifying_units': result['selection_count'],
                      'required_units': 4, 'source_rows': result['source_rows'],
                      'scope': 'Saved tokenizer consistency only; process and chronology are separate.'},
                     sort_keys=True))


if __name__ == '__main__':
    main()
