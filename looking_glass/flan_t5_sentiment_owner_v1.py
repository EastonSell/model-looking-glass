"""Source-bound CPU owner for the preregistered FLAN sentiment study.

The surrounding operator observes the direct child and retains failures. The
cards and independent replay are local evidence, not authentication of a
reviewer, public chronology, model execution, or unknown prior exposures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys


SOURCE_PATHS = (
    'tools/run_flan_t5_sentiment_intervention_v1.py',
    'tools/run_flan_sentiment_selector_stage_v1.py',
    'tools/observe_flan_sentiment_selector_stage_v1.py',
    'tools/run_flan_sentiment_selector_v1.py',
    'looking_glass/__init__.py',
    'looking_glass/flan_t5_sentiment_owner_v1.py',
    'looking_glass/flan_t5_sentiment_selector_v1.py',
    'looking_glass/flan_t5_sentiment_screen.py',
    'looking_glass/flan_t5_feasibility.py',
    'looking_glass/bert_matched_numeric.py',
    'looking_glass/flan_t5_sentiment_numeric.py',
    'looking_glass/flan_t5_sentiment_journal.py',
    'looking_glass/flan_t5_sentiment_runner.py',
    'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md',
    'docs/FLAN_T5_SENTIMENT_SELECTOR_V1.md',
    'docs/FLAN_T5_SENTIMENT_CPU_CORE_V1.md',
    'docs/FLAN_T5_SENTIMENT_CPU_OWNER_V1.md',
)
MANIFEST_SHA256 = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
MAX_CUSTODY_BYTES = 8 * 1024**2
EXTERNAL_CUSTODY_RESERVE = 256 * 1024
REVIEW_SCHEMA = 'flan-t5-sentiment-independent-tokenizer-replay-v1'
SELECTOR_SOURCE_PATHS = (
    'tools/run_flan_sentiment_selector_stage_v1.py', 'tools/run_flan_sentiment_selector_v1.py',
    'looking_glass/__init__.py', 'looking_glass/flan_t5_sentiment_selector_v1.py',
    'looking_glass/flan_t5_sentiment_screen.py', 'looking_glass/flan_t5_feasibility.py',
    'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md', 'docs/FLAN_T5_SENTIMENT_SELECTOR_V1.md',
)
SCIENTIFIC_READER_PATHS = (
    'scripts/flan_sentiment_intervention_independent_audit.py',
    'scripts/flan_sentiment_dose_oracle.py',
    'scripts/bert_taste_matched_independent.py',
)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _clear_git_environment():
    for name in tuple(os.environ):
        if name.upper().startswith('GIT_'):
            os.environ.pop(name, None)


def _read(path, maximum):
    path = Path(path)
    _check(path.is_file() and not path.is_symlink(), 'custody input is absent or aliased')
    _check(path.stat().st_size <= maximum, 'custody input exceeds its bound')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    _check(len(raw) <= maximum, 'custody input exceeds its bound')
    return raw


def _write_new(path, value):
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')
    with Path(path).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short owner receipt write')
        stream.flush()
        os.fsync(stream.fileno())


def source_snapshot(root, revision):
    """Verify the actual owner/runner source against one locally observed commit."""
    _clear_git_environment()
    root = Path(root)
    _check(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision) is not None,
           'source revision is invalid')
    git = ['git', '--no-replace-objects']
    head = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=root).strip().decode('ascii')
    _check(head == revision and subprocess.check_output(git + ['cat-file', '-t', revision], cwd=root).strip()
           == b'commit', 'checkout is not at the expected commit')
    hashes = {}
    for name in SOURCE_PATHS:
        raw = (root / name).read_bytes()
        committed = subprocess.check_output(git + ['show', revision + ':' + name], cwd=root)
        _check(raw == committed, 'owner or runner source differs from the expected commit')
        hashes[name] = _hash(raw)
    return {'revision': revision, 'sha256': hashes}


def _observe_card(repo, raw, observation, kind):
    _clear_git_environment()
    from . import flan_t5_sentiment_selector_v1 as selector
    _check(type(observation) is dict and set(observation) == {'schema', 'path', 'revision', 'sha256'}
           and observation['schema'] == f'flan-sentiment-{kind}-observation-v1',
           'card observation schema differs')
    descriptor = {key: observation[key] for key in ('path', 'revision', 'sha256')}
    selector._source_descriptor(descriptor)
    _check(_hash(raw) == observation['sha256'], 'observed card hash differs')
    selector._require_commit(repo, observation['revision'])
    public = subprocess.check_output(['git', '--no-replace-objects', 'show',
                                      observation['revision'] + ':' + observation['path']], cwd=repo)
    _check(public == raw, 'observed public card differs')


def _check_stage_source(repo, runner_revision, stage):
    _clear_git_environment()
    from . import flan_t5_sentiment_selector_v1 as selector
    selector._require_commit(repo, stage['selector_revision'])
    _check(type(stage['source_sha256']) is dict and set(stage['source_sha256']) == set(SELECTOR_SOURCE_PATHS),
           'selector stage source inventory differs')
    git = ['git', '--no-replace-objects', 'show']
    for name in SELECTOR_SOURCE_PATHS:
        observed = subprocess.check_output(git + [stage['selector_revision'] + ':' + name], cwd=repo)
        reviewed = subprocess.check_output(git + [runner_revision + ':' + name], cwd=repo)
        _check(_hash(observed) == stage['source_sha256'][name] and observed == reviewed,
               'selector stage source does not match the reviewed runner revision')


def _check_replay_source(repo, review):
    _clear_git_environment()
    from . import flan_t5_sentiment_selector_v1 as selector
    descriptor = {'path': review['replay_source_path'], 'revision': review['replay_source_revision'],
                  'sha256': review['replay_script_sha256']}
    selector._source_descriptor(descriptor)
    selector._require_commit(repo, descriptor['revision'])
    source = subprocess.check_output(['git', '--no-replace-objects', 'show',
                                      descriptor['revision'] + ':' + descriptor['path']], cwd=repo)
    _check(_hash(source) == descriptor['sha256'], 'independent replay source differs')


def _check_scientific_reader(repo, pin):
    _clear_git_environment()
    from . import flan_t5_sentiment_selector_v1 as selector
    _check(type(pin) is dict and set(pin) == {'schema', 'revision', 'sha256'}
           and pin['schema'] == 'flan-t5-sentiment-scientific-reader-pin-v1'
           and type(pin['sha256']) is dict and set(pin['sha256']) == set(SCIENTIFIC_READER_PATHS),
           'scientific reader pin differs')
    selector._require_commit(repo, pin['revision'])
    for name in SCIENTIFIC_READER_PATHS:
        selector._source_descriptor({'path': name, 'revision': pin['revision'], 'sha256': pin['sha256'][name]})
        source = subprocess.check_output(['git', '--no-replace-objects', 'show', pin['revision'] + ':' + name],
                                         cwd=repo)
        _check(_hash(source) == pin['sha256'][name], 'scientific reader source differs')


def bind_custody(repo, revision, paths, *, observe=_observe_card):
    """Reconcile the two private commitments and observed public cards.

    The independent replay report is a separate observation. Its claimed
    independence and the completeness of the source inventory are external.
    """
    _clear_git_environment()
    from . import flan_t5_sentiment_selector_v1 as selector
    expected_names = ('pool', 'pool_salt', 'pool_card', 'pool_observation',
                      'selection', 'selection_salt', 'selection_card', 'selection_observation',
                      'screen_runtime_receipt', 'screen_exit_observation', 'screen_review', 'scientific_reader_pin')
    _check(type(paths) is dict and set(paths) == set(expected_names), 'custody path set differs')
    raw = {name: _read(paths[name], MAX_CUSTODY_BYTES) for name in expected_names}
    _check(sum(map(len, raw.values())) + EXTERNAL_CUSTODY_RESERVE <= MAX_CUSTODY_BYTES,
           'custody files exceed their reserved allowance')
    pool = selector.strict_json(raw['pool'])
    selection = selector.strict_json(raw['selection'])
    _check(raw['pool'] == selector.canonical(pool) and raw['selection'] == selector.canonical(selection),
           'private pool or selection is not canonical')
    _check(type(pool) is dict and type(selection) is dict and pool.get('runner_revision') == revision
           and selection.get('runner_revision') == revision, 'runner revision is not bound by custody')
    exposures = selector.load_exposures(repo, pool['exposure_sources'])
    selector.validate_pool(pool, exposures)
    selector._require_commit(repo, revision)
    pool_digest = selector.pool_commitment(pool, raw['pool_salt'])
    pool_card = {'schema': 'flan-sentiment-pool-card-v1', 'commitment': pool_digest,
                 'protocol_sha256': pool['protocol_sha256'], 'runner_revision': revision,
                 'asset_manifest_sha256': pool['asset_manifest_sha256']}
    _check(selector.strict_json(raw['pool_card']) == pool_card, 'pool card does not match private pool')
    observe(repo, raw['pool_card'], selector.strict_json(raw['pool_observation']), 'pool')
    for key in ('protocol_sha256', 'runner_revision', 'asset_manifest_sha256', 'exposure_sources'):
        _check(selection.get(key) == pool[key], 'pool and selection identities differ')
    _check(selection.get('pool_commitment') == pool_digest, 'selection does not bind the pool card')
    selection_digest = selector.selection_commitment(selection, raw['selection_salt'])
    selection_card = {'schema': 'flan-sentiment-selection-card-v1', 'commitment': selection_digest,
                      'pool_commitment': pool_digest, 'protocol_sha256': pool['protocol_sha256'],
                      'runner_revision': revision, 'asset_manifest_sha256': pool['asset_manifest_sha256']}
    _check(selector.strict_json(raw['selection_card']) == selection_card,
           'selection card does not match private selection')
    observe(repo, raw['selection_card'], selector.strict_json(raw['selection_observation']), 'selection')
    stage = selector.strict_json(raw['screen_runtime_receipt'])
    stage_keys = {'schema', 'stage', 'status', 'selector_revision', 'source_sha256', 'versions',
                  'pool_commitment', 'selection_commitment', 'scope'}
    _check(type(stage) is dict and set(stage) == stage_keys
           and stage['schema'] == 'flan-t5-sentiment-selector-stage-runtime-v1'
           and stage['stage'] == 'select' and stage['status'] == 'completed'
           and stage['pool_commitment'] == pool_digest and stage['selection_commitment'] == selection_digest
           and type(stage['versions']) is dict and set(stage['versions']) == {'python', 'transformers', 'tokenizers'}
           and all(type(v) is str and 0 < len(v) <= 128 for v in stage['versions'].values())
           and stage['scope'] == 'Same-process version observation; external direct-exit and custody checks remain required.'
           and raw['screen_runtime_receipt'] == selector.canonical(stage) + b'\n',
           'selector stage runtime receipt differs')
    _check_stage_source(repo, revision, stage)
    exit_record = selector.strict_json(raw['screen_exit_observation'])
    exit_keys = {'schema', 'stage', 'status', 'stage_receipt_sha256', 'reaped', 'exit_code', 'timed_out',
                 'exception_type', 'selector_revision', 'child_prefix_empty_after', 'logs', 'scope'}
    _check(type(exit_record) is dict and set(exit_record) == exit_keys
           and exit_record['schema'] == 'flan-t5-sentiment-selector-stage-exit-v1'
           and exit_record['stage'] == 'select' and exit_record['status'] == 'completed'
           and exit_record['stage_receipt_sha256'] == _hash(raw['screen_runtime_receipt'])
           and exit_record['reaped'] is True and type(exit_record['exit_code']) is int
           and exit_record['exit_code'] == 0 and exit_record['timed_out'] is False
           and exit_record['exception_type'] is None and exit_record['child_prefix_empty_after'] is True
           and exit_record['selector_revision'] == stage['selector_revision']
           and exit_record['scope'] == 'Local direct-child observation; no execution authentication.'
           and raw['screen_exit_observation'] == selector.canonical(exit_record) + b'\n'
           and type(exit_record['logs']) is dict and set(exit_record['logs']) == {'stdout', 'stderr'},
           'selector stage exit was not observed cleanly')
    log_paths, stage_logs = {}, {}
    for name, log in exit_record['logs'].items():
        _check(type(log) is dict and set(log) == {'retained_bytes', 'truncated', 'error_type', 'finished', 'sha256'}
               and type(log['retained_bytes']) is int and 0 <= log['retained_bytes'] <= 64 * 1024
               and log['truncated'] is False and log['error_type'] is None and log['finished'] is True
               and type(log['sha256']) is str and re.fullmatch('[0-9a-f]{64}', log['sha256']) is not None,
               'selector stage log observation differs')
        log_paths[name] = Path(paths['screen_exit_observation']).parent / (name + '.log')
        stage_logs[name] = _read(log_paths[name], 64 * 1024)
        _check(len(stage_logs[name]) == log['retained_bytes'] and _hash(stage_logs[name]) == log['sha256'],
               'selector stage log bytes differ from their observation')
    _check(sum(map(len, raw.values())) + sum(map(len, stage_logs.values())) + EXTERNAL_CUSTODY_RESERVE
           <= MAX_CUSTODY_BYTES, 'custody and stage logs exceed their reserved allowance')
    review = selector.strict_json(raw['screen_review'])
    keys = {'schema', 'status', 'pool_commitment', 'selection_commitment', 'protocol_sha256',
            'runner_revision', 'asset_manifest_sha256', 'unit_count', 'token_rows', 'selection_count',
            'enough_selected', 'source_count', 'source_rows', 'screen_runtime', 'limitations',
            'replay_script_sha256', 'replay_source_revision', 'replay_source_path',
            'selector_source_revision', 'screen_runtime_receipt_sha256', 'screen_exit_receipt_sha256'}
    _check(type(review) is dict and set(review) == keys and review['schema'] == REVIEW_SCHEMA
           and review['status'] == 'consistent' and review['pool_commitment'] == pool_digest
           and review['selection_commitment'] == selection_digest
           and review['protocol_sha256'] == pool['protocol_sha256'] and review['runner_revision'] == revision
           and review['asset_manifest_sha256'] == pool['asset_manifest_sha256']
           and type(review['unit_count']) is int and review['unit_count'] == 12
           and type(review['token_rows']) is int and review['token_rows'] == 96
           and type(review['selection_count']) is int and review['selection_count'] == 4
           and review['enough_selected'] is True
           and type(review['source_count']) is int and review['source_count'] == len(exposures)
           and type(review['source_rows']) is int
           and review['source_rows'] == sum(len(item['rows']) for item in exposures)
           and review['screen_runtime'] == stage['versions']
           and review['selector_source_revision'] == stage['selector_revision']
           and review['screen_runtime_receipt_sha256'] == _hash(raw['screen_runtime_receipt'])
           and review['screen_exit_receipt_sha256'] == _hash(raw['screen_exit_observation'])
           and type(review['limitations']) is list and 1 <= len(review['limitations']) <= 16
           and all(type(v) is str and 0 < len(v) <= 512 for v in review['limitations']),
           'independent tokenizer replay is incomplete or not bound')
    _check_replay_source(repo, review)
    scientific_reader = selector.strict_json(raw['scientific_reader_pin'])
    _check(raw['scientific_reader_pin'] == selector.canonical(scientific_reader) + b'\n',
           'scientific reader pin is not canonical')
    _check_scientific_reader(repo, scientific_reader)
    return {'pool': pool, 'selection': selection, 'exposures': exposures,
            'pool_commitment': pool_digest, 'selection_commitment': selection_digest,
            'screen_review': review, 'scientific_reader': scientific_reader,
            'raw': raw, 'stage_logs': stage_logs, 'stage_log_paths': log_paths}


def _file_stamps(paths):
    stamps = {}
    for name, path in paths.items():
        info = Path(path).stat()
        stamps[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    return stamps


def run(args):
    """Perform one fixed owner attempt, preserving a separate outcome receipt."""
    from .flan_t5_feasibility import verify_assets
    from .flan_t5_sentiment_journal import EvidenceJournal, Limits, require_protocol_allocation
    from .flan_t5_sentiment_runner import run_cpu_core, selection_to_groups
    from . import flan_t5_sentiment_selector_v1 as selector

    root, assets, attempt = Path(args.repo), Path(args.asset_dir), Path(args.attempt)
    receipt = attempt / 'owner-receipt.json'
    phase = 'preflight'
    status = 'incomplete'
    journal = None
    try:
        _check(root.is_absolute() and assets.is_absolute() and attempt.is_absolute() and attempt.is_dir(),
               'owner paths must be absolute and attempt must already exist')
        required_env = {'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}
        _check(all(os.environ.get(k) == v for k, v in required_env.items()), 'CPU/offline environment differs')
        source = source_snapshot(root, args.revision)
        paths = {name: getattr(args, name) for name in (
            'pool', 'pool_salt', 'pool_card', 'pool_observation', 'selection', 'selection_salt',
            'selection_card', 'selection_observation', 'screen_runtime_receipt', 'screen_exit_observation',
            'screen_review', 'scientific_reader_pin')}
        binding = bind_custody(root, args.revision, paths)
        require_protocol_allocation(Limits())
        journal = EvidenceJournal(attempt / 'child', {
            'schema': 'flan-t5-sentiment-intervention-owner-v1', 'source': source,
            'pool_commitment': binding['pool_commitment'], 'selection_commitment': binding['selection_commitment'],
            'asset_manifest_sha256': MANIFEST_SHA256,
            'scope': 'Saved-byte consistency and local direct-process observation; not execution authentication.'})
        asset_manifest = verify_assets(assets)
        phase = 'runtime_setup'
        import torch
        import numpy
        import transformers
        import tokenizers
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        site = Path(args.venv) / ('Lib/site-packages' if os.name == 'nt' else
                                  'lib/python%d.%d/site-packages' % sys.version_info[:2])
        site = site.resolve(strict=True)
        _check(all(Path(module.__file__).resolve().is_relative_to(site)
                   for module in (torch, numpy, transformers, tokenizers)),
               'model libraries are not from the selected venv')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision('highest')
        torch.manual_seed(20261028)
        _check(torch.set_flush_denormal(False), 'cannot preserve CPU denormals')
        tokenizer = AutoTokenizer.from_pretrained(str(assets), local_files_only=True, use_fast=True,
                                                  trust_remote_code=False)
        _check(type(tokenizer).__name__ == 'T5TokenizerFast' and tokenizer.eos_token_id == 1
               and tokenizer.pad_token_id == 0, 'tokenizer identity differs')
        replayed = selector.select(binding['pool'], tokenizer, binding['exposures'], binding['pool_commitment'])
        _check(replayed == binding['selection'], 'owner tokenization differs from the sealed selection')
        groups = selection_to_groups(binding['selection'], expected_runner_revision=args.revision,
                                     vocab_size=32128, special_ids=set(tokenizer.all_special_ids))
        _check(len(groups) == 8, 'selection has fewer than four eligible units')
        runtime = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__,
                   'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                   'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                   'deterministic': torch.are_deterministic_algorithms_enabled(),
                   'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                   'float32_matmul_precision': torch.get_float32_matmul_precision(),
                   'torch_build_sha256': _hash(torch.__config__.show().encode('utf8')),
                   'cpu_architecture': platform.machine(), 'cpu_count': os.cpu_count(),
                   'mkldnn_enabled': torch.backends.mkldnn.enabled,
                   'byte_order': sys.byteorder, 'python_float_rounds': sys.float_info.rounds,
                   'numeric_environment': {name: os.environ.get(name) for name in
                       ('MKL_CBWR', 'ATEN_CPU_CAPABILITY', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM')},
                   'environment': required_env, 'seed': 20261028, 'device': 'cpu', 'dtype': 'float32',
                   'attention': 'eager', 'decoder_cache': False, 'screen_replay': binding['screen_review'],
                   'scientific_reader': binding['scientific_reader']}
        journal.record('owner_contract', runtime)
        phase = 'model_load'
        _check(shutil.disk_usage(attempt).free >= Limits().free_bytes,
               'insufficient free space before model loading')
        model, info = T5ForConditionalGeneration.from_pretrained(
            str(assets), local_files_only=True, use_safetensors=True, trust_remote_code=False,
            output_loading_info=True, dtype=torch.float32, attn_implementation='eager')
        _check(not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
               'model loading diagnostics differ')
        model.eval()
        model.config.use_cache = False
        config = model.config
        _check(type(model).__name__ == 'T5ForConditionalGeneration' and config.d_model == 512
               and config.vocab_size == 32128 and config.decoder_start_token_id == 0
               and config.pad_token_id == 0 and config.eos_token_id == 1
               and getattr(config, '_attn_implementation', None) == 'eager'
               and not torch.cuda.is_initialized(), 'model configuration differs')
        watched = {**paths, **{'stage-log:' + name: path for name, path in binding['stage_log_paths'].items()},
                   **{'source:' + name: root / name for name in SOURCE_PATHS},
                   **{'asset:' + name: assets / name for name in asset_manifest['files']},
                   'asset:manifest': assets / 'asset-manifest.json'}
        initial_stamps = _file_stamps(watched)
        def custody_check():
            _check(_file_stamps(watched) == initial_stamps, 'source, asset or custody file changed')
        phase = 'forwards'
        result = run_cpu_core(model, tokenizer, groups, journal, custody_check=custody_check)
        phase = 'postflight'
        custody_check()
        _check(source_snapshot(root, args.revision) == source and verify_assets(assets) == asset_manifest,
               'source or assets changed during the attempt')
        _check(all(_read(paths[name], MAX_CUSTODY_BYTES) == value for name, value in binding['raw'].items()),
               'private custody bytes changed during the attempt')
        _check(all(_read(binding['stage_log_paths'][name], 64 * 1024) == value
                   for name, value in binding['stage_logs'].items()), 'selector stage logs changed')
        status = 'complete_candidate'
        _write_new(receipt, {'schema': 'flan-t5-sentiment-owner-receipt-v1', 'status': status,
                             'source_revision': args.revision, 'pool_commitment': binding['pool_commitment'],
                             'selection_commitment': binding['selection_commitment'],
                             'terminal_sha256': _hash((attempt / 'child' / 'terminal.json').read_bytes()),
                             'events_sha256': _hash((attempt / 'child' / 'events.jsonl').read_bytes()),
                             'scope': 'Saved-file consistency only; independent scientific replay is required.'})
        return result
    except BaseException as error:
        cleanup_interrupt = None
        if journal is not None and not (attempt / 'child' / 'terminal.json').exists():
            try:
                journal.finish('incomplete', {'phase': phase, 'error_type': type(error).__name__})
            except BaseException as terminal_error:
                if isinstance(error, Exception) and not isinstance(terminal_error, Exception):
                    cleanup_interrupt = terminal_error
        if not receipt.exists():
            try:
                _write_new(receipt, {'schema': 'flan-t5-sentiment-owner-receipt-v1', 'status': 'incomplete',
                                     'phase': phase, 'error_type': type(error).__name__})
            except BaseException as receipt_error:
                if (cleanup_interrupt is None and isinstance(error, Exception)
                        and not isinstance(receipt_error, Exception)):
                    cleanup_interrupt = receipt_error
        if cleanup_interrupt is not None:
            raise cleanup_interrupt from error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'asset-dir', 'attempt', 'pool', 'pool-salt', 'pool-card', 'pool-observation',
                 'selection', 'selection-salt', 'selection-card', 'selection-observation',
                 'screen-runtime-receipt', 'screen-exit-observation', 'screen-review', 'scientific-reader-pin'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--revision', required=True)
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
