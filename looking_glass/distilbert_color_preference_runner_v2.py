"""Supervised CPU owner for the fixed nine-row color-preference screen.

Every stage is one-shot in its run directory. Public predecessor archives are
checked before the next stage, and all retained original bytes and started
forwards share one budget. Receipts are cooperative process observations.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from . import distilbert_color_edit_metrics_v1 as metrics
from . import distilbert_color_edit_records_v1 as records
from . import distilbert_color_edit_publication_v1 as publication
from . import distilbert_color_edit_stages_v1 as stages
from . import distilbert_color_preference_screen_v2 as preference
from . import distilbert_color_preference_public_v2 as public
from .bert_agreement_study import verify_git_sources, _require_denormals

CASE_SHA256 = 'f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10'
CHECK_SHA256 = '6c29d8875238448a8fc20887832b02d058b6f76fcab0a41fe6385050e932a1c3'
DECODER_BIAS_SHA256 = '84bd316731d4479683221383be6553704dd27421c56ee1aa6a389f80ebb0ea43'
RUNTIME = {'torch': '2.8.0+cu128', 'transformers': '4.57.3', 'tokenizers': '0.22.2',
           'numpy': '2.4.6', 'safetensors': '0.8.0'}
PYTHON_VERSION = (3, 11, 15)
SOURCE_PATHS = (
    'docs/DISTILBERT_COLOR_PREFERENCE_V2.md',
    'docs/DISTILBERT_COLOR_PREFERENCE_SCREEN_V2.md',
    'experiments/bert_color_preference_public_cases_v2.json',
    'experiments/bert_color_preference_public_comparison_v2.json',
    'experiments/bert_color_preference_public_exposures_v1.json',
    'scripts/build_distilbert_color_edit_cases_v1.py',
    'scripts/build_distilbert_color_preference_cases_v2.py',
    'scripts/check_distilbert_color_preference_exposures_v2.py',
    'scripts/prepare_distilbert_color_prelaunch_v2.py',
    'scripts/observe_distilbert_color_supervisor_v1.py',
    'scripts/observe_distilbert_color_preference_v2.py',
    'looking_glass/__init__.py', 'looking_glass/pretrained.py', 'looking_glass/bert_agreement_study.py',
    'looking_glass/distilbert_color_edit_metrics_v1.py', 'looking_glass/distilbert_color_edit_records_v1.py',
    'looking_glass/distilbert_color_edit_action_v1.py', 'looking_glass/distilbert_color_edit_stages_v1.py',
    'looking_glass/distilbert_color_edit_publication_v1.py',
    'looking_glass/distilbert_color_preference_public_v2.py',
    'looking_glass/distilbert_color_preference_screen_v2.py',
    'looking_glass/distilbert_color_preference_runner_v2.py',
)
TERMINAL_RESERVE = 4 * 1024 * 1024
MAX_SCREEN_STARTS = 9
BOOTSTRAP = ("import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
             "sys.path.insert(1,sys.argv.pop(1));"
             "runpy.run_module('looking_glass.distilbert_color_preference_runner_v2',run_name='__main__')")


def _check_launch(site):
    """Require isolated Python startup and a fresh, empty bytecode prefix."""
    if (not sys.flags.ignore_environment or not sys.flags.no_user_site or
            not sys.dont_write_bytecode or sys.flags.optimize or not sys.pycache_prefix):
        raise RuntimeError('launch requires -E -s -B and a fresh -X pycache_prefix')
    site = Path(site)
    if (sys.prefix != sys.base_prefix or Path(sys.executable).resolve() != Path(sys._base_executable).resolve() or
            not site.is_absolute() or site.is_symlink() or not site.is_dir() or
            getattr(site.stat(), 'st_file_attributes', 0) & 0x400 or
            len(sys.path) < 2 or Path(sys.path[1]).resolve(strict=True) != site.resolve(strict=True)):
        raise RuntimeError('launch requires the direct base interpreter and explicit trusted site directory')
    prefix = Path(sys.pycache_prefix)
    if (not prefix.is_absolute() or prefix.is_symlink() or not prefix.is_dir() or
            getattr(prefix.stat(), 'st_file_attributes', 0) & 0x400 or
            prefix.resolve(strict=True) != prefix.absolute() or any(prefix.iterdir())):
        raise RuntimeError('bytecode prefix is not an empty regular directory')


def _remove_scratch(scratch, cache):
    scratch, cache = Path(scratch), Path(cache).resolve(strict=True)
    if (scratch.is_symlink() or scratch.resolve(strict=True).parent != cache or
            not scratch.name.startswith('color-owner-')):
        raise RuntimeError('refusing to remove unverified temporary directory')
    shutil.rmtree(scratch)


def _kill_and_reap(process):
    """Bound direct-child cleanup; never infer the status of descendants."""
    try:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        return process.wait(timeout=30), 'direct-child-reaped'
    except BaseException:
        return None, 'unknown'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _write(path, raw):
    with Path(path).open('xb', buffering=0) as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short evidence write')
        stream.flush()
        os.fsync(stream.fileno())
    if Path(path).read_bytes() != raw:
        raise OSError('evidence bytes changed on write')


def _read_json(path):
    return publication.strict_json(Path(path).read_bytes())


def _file_sha(path):
    digest, count = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            count += len(block)
            digest.update(block)
    return count, digest.hexdigest()


def _copy_asset(source, destination, expected):
    if _file_sha(source) != expected:
        raise ValueError('cached checkpoint asset differs from its pin')
    with source.open('rb') as src, destination.open('xb', buffering=0) as dst:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            if dst.write(block) != len(block):
                raise OSError('short checkpoint staging write')
        dst.flush()
        os.fsync(dst.fileno())
    if _file_sha(destination) != expected or _file_sha(source) != expected:
        raise ValueError('staged or cached checkpoint asset changed')


def _load_model(cache, manifest, repo, scratch):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('CPU-only offline environment must be set before dependency import')
    if (sys.version_info[:3] != PYTHON_VERSION or
            {name: importlib.metadata.version(name) for name in RUNTIME} != RUNTIME):
        raise RuntimeError('runtime package versions differ from the frozen owner')
    import torch
    import transformers
    from scripts import build_distilbert_color_edit_cases_v1 as builder
    from scripts import build_distilbert_color_preference_cases_v2 as preference_builder
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before CPU ownership')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or
            not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError('actual thread or deterministic-algorithm state differs')
    _require_denormals(torch)
    metrics.require_denormals()
    snapshot = Path(cache) / 'models--distilbert--distilbert-base-uncased' / 'snapshots' / builder.REVISION
    with tempfile.TemporaryDirectory(prefix='color-edit-stage-', dir=scratch) as name:
        staged = Path(name)
        for asset, expected in builder.ASSETS.items():
            _copy_asset(snapshot / asset, staged / asset, expected)
        with torch.device('cpu'):
            tokenizer = transformers.AutoTokenizer.from_pretrained(staged, local_files_only=True,
                                                                  trust_remote_code=False, use_fast=True)
            model, info = transformers.AutoModelForMaskedLM.from_pretrained(
                staged, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                attn_implementation='eager', dtype=torch.float32, output_loading_info=True)
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('checkpoint load reported incompatible weights')
        for asset, expected in builder.ASSETS.items():
            if _file_sha(staged / asset) != expected or _file_sha(snapshot / asset) != expected:
                raise ValueError('checkpoint asset changed during model loading')
    model.eval()
    if (type(model) is not transformers.DistilBertForMaskedLM or model.config.model_type != 'distilbert'
            or model.config.n_layers != 6 or model.config.dim != 768 or model.config.vocab_size != 30522
            or model.config._attn_implementation != 'eager' or torch.cuda.is_initialized()
            or tokenizer.mask_token_id != 103 or model.get_input_embeddings().weight is not model.vocab_projector.weight
            or model.get_input_embeddings().weight.data_ptr() != model.vocab_projector.weight.data_ptr()):
        raise RuntimeError('model, tied decoder or tokenizer identity differs')
    if builder.encode_inventory(tokenizer, preference_builder.specifications()) != manifest['cases']:
        raise RuntimeError('actual tokenizer differs from the exact frozen cases')
    from .distilbert_color_edit_action_v1 import ActionEngine, tensor_bytes
    linear = model.distilbert.transformer.layer[-1].ffn.lin2
    bias = model.vocab_projector.bias
    if (linear is not model.distilbert.transformer.layer[5].ffn.lin2 or bias is not model.vocab_projector.bias or
            _sha(tensor_bytes(bias.detach().contiguous())) != DECODER_BIAS_SHA256):
        raise RuntimeError('model intervention bindings differ')
    engine = ActionEngine(model, linear, bias, vocabulary=30522, hidden=768, intermediate=3072)
    return engine, {'runtime': RUNTIME, 'python': '.'.join(map(str, sys.version_info[:3])),
                    'model': builder.MODEL, 'revision': builder.REVISION,
                    'model_state_sha256': engine.state.sha256, 'device': 'cpu', 'dtype': 'float32',
                    'eager': True, 'intraop_threads': torch.get_num_threads(),
                    'interop_threads': torch.get_num_interop_threads(),
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'cuda_initialized': False,
                    'decoder_weight_tied_to_input': True, 'decoder_bias_sha256': DECODER_BIAS_SHA256}


def _verify_prelaunch(repo, source, phase, raw, *, check_main=True):
    """Reconstruct public inputs at preparation and again at the launch boundary."""
    from scripts import check_distilbert_color_preference_exposures_v2 as comparison
    document = publication.strict_json(raw)
    if (type(document) is not dict or document.get('schema') != 'distilbert-color-preference-prelaunch-v2'
            or document.get('phase') != phase or document.get('source_commit') != source
            or document.get('case_sha256') != CASE_SHA256 or document.get('public_check_sha256') != CHECK_SHA256
            or type(document.get('public_scan')) is not dict
            or type(document.get('additional_public')) is not list or len(document['additional_public']) > 20
            or type(document.get('scope')) is not str or not document['scope']):
        raise ValueError('invalid scoped prelaunch record')
    case_path = Path(repo) / public.CASE_PATH
    check_path = Path(repo) / 'experiments/bert_color_preference_public_comparison_v2.json'
    exposure_path = Path(repo) / public.EXPOSURE_PATH
    if _file_sha(case_path)[1] != CASE_SHA256 or _file_sha(check_path)[1] != CHECK_SHA256:
        raise ValueError('case or public comparison bytes differ')
    if _file_sha(exposure_path)[1] != public.EXPOSURE_SHA256:
        raise ValueError('planned exposure bytes differ')
    rerun = comparison.check(Path(repo), case_path)
    if _json_bytes(rerun) != _json_bytes(publication.strict_json(check_path.read_bytes())):
        raise ValueError('named public comparison does not reconstruct')
    proposed = publication.strict_json(case_path.read_bytes())['cases']
    observed = public.scan(repo, document['public_scan'].get('commit'), proposed)
    if _json_bytes(observed) != _json_bytes(document['public_scan']):
        raise ValueError('prepared public scan does not reconstruct')
    seen = set()
    for item in document['additional_public']:
        if type(item) is not dict or set(item) != {'commit', 'path', 'sha256', 'rows'}:
            raise ValueError('invalid additional public input source')
        key = (item['commit'], item['path'])
        if key in seen:
            raise ValueError('duplicate additional public source')
        seen.add(key)
        actual = public.additional(repo, item['commit'], item['path'], proposed)
        if _json_bytes(actual) != _json_bytes(item):
            raise ValueError('additional public source does not reconstruct')
    current = observed
    if check_main:
        remote = subprocess.run(['git', '-C', str(repo), 'ls-remote', '--heads', 'origin', 'refs/heads/main'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60).stdout.split()
        if len(remote) != 2 or remote[1] != b'refs/heads/main':
            raise RuntimeError('unable to observe official main')
        main = remote[0].decode('ascii')
        if main != observed['commit']:
            # Recheck changed named inputs rather than rejecting unrelated main advances.
            current = public.scan(repo, main, proposed)
    return {'document': document, 'current_scan': current}


def _inspect_run(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('a regular run directory is required')
    allowed_root = {'run.json', 'screen', 'development', 'evaluation'}
    used, starts = 0, 0
    for path in root.iterdir():
        if path.is_symlink() or path.name not in allowed_root:
            raise ValueError('unexpected run-root entry')
        if path.name == 'run.json':
            if not path.is_file():
                raise ValueError('run metadata is not a regular file')
            used += path.stat().st_size
            continue
        if not path.is_dir():
            raise ValueError('stage is not a regular directory')
        for artifact in path.iterdir():
            if artifact.is_symlink() or not artifact.is_file() or artifact.name not in publication.ALLOWED:
                raise ValueError('unexpected stage evidence')
            used += artifact.stat().st_size
            if artifact.name == 'claim.json':
                claim = _read_json(artifact)
                external = claim.get('external_prelaunch_bytes')
                if type(external) is not int or not 1 <= external <= 256 * 1024:
                    raise ValueError('invalid retained prelaunch-input byte count')
                used += external
            if artifact.name == 'supervisor-observation.json':
                observation = _read_json(artifact)
                external = observation.get('observer_external_bytes')
                if type(external) is not int or not 1 <= external <= 16384 - artifact.stat().st_size:
                    raise ValueError('invalid retained observer-metadata byte count')
                used += external
            if artifact.name == 'events.jsonl':
                events = records.parse_jsonl(artifact.read_bytes())
                starts += sum(event.get('kind') == 'start' for event in events)
    if used > records.MAX_BYTES or starts > records.MAX_FORWARDS:
        raise ValueError('aggregate evidence or forward budget exceeded')
    return used, starts


def _validate_runtime(runtime, claim):
    """Require the saved worker's complete frozen CPU/runtime contract."""
    fields = {'runtime', 'python', 'model', 'revision', 'model_state_sha256', 'device', 'dtype', 'eager',
              'intraop_threads', 'interop_threads', 'deterministic_algorithms', 'cuda_initialized',
              'decoder_weight_tied_to_input', 'decoder_bias_sha256', 'site_path_sha256', 'scratch_root_sha256'}
    if (type(runtime) is not dict or set(runtime) != fields or
            runtime['runtime'] != RUNTIME or runtime['python'] != '3.11.15' or
            runtime['model'] != 'distilbert/distilbert-base-uncased' or
            runtime['revision'] != '12040accade4e8a0f71eabdb258fecc2e7e948be' or
            type(runtime['model_state_sha256']) is not str or
            re.fullmatch('[0-9a-f]{64}', runtime['model_state_sha256']) is None or
            runtime['device'] != 'cpu' or runtime['dtype'] != 'float32' or runtime['eager'] is not True or
            type(runtime['intraop_threads']) is not int or runtime['intraop_threads'] != 1 or
            type(runtime['interop_threads']) is not int or runtime['interop_threads'] != 1 or
            runtime['deterministic_algorithms'] is not True or runtime['cuda_initialized'] is not False or
            runtime['decoder_weight_tied_to_input'] is not True or runtime['decoder_bias_sha256'] != DECODER_BIAS_SHA256 or
            runtime['site_path_sha256'] != claim.get('site_path_sha256') or
            runtime['scratch_root_sha256'] != claim.get('scratch_root_sha256')):
        raise ValueError('saved CPU runtime or model-state witness differs')
    for name in ('site_path_sha256', 'scratch_root_sha256'):
        if type(runtime[name]) is not str or re.fullmatch('[0-9a-f]{64}', runtime[name]) is None:
            raise ValueError('saved runtime path hash differs')


def _valid_prior(stage, phase, source, source_hashes, *, require_finalized=True):
    stage = Path(stage)
    if (not stage.is_dir() or (stage / 'failure.json').exists() or
            (stage / 'supervisor-observation-failure.json').exists() or
            not all((stage / name).is_file() for name in ('claim.json', 'worker-start.json', 'events.jsonl', 'result.json', 'exit.json')) or
            (require_finalized and not all((stage / name).is_file() for name in
                                           ('validated.json', 'supervisor-observation.json')))):
        raise ValueError('earlier stage is incomplete and cannot be retried')
    claim, result, exit_, worker = (_read_json(stage / name) for name in
                                    ('claim.json', 'result.json', 'exit.json', 'worker-start.json'))
    raw = (stage / 'events.jsonl').read_bytes()
    parsed = records.parse_jsonl(raw)
    pending = None
    for event in parsed:
        kind = event.get('kind')
        if kind == 'start' and pending is None and type(event.get('action')) is int and event.get('phase') == phase:
            pending = event['action']
        elif (kind == 'complete' and pending is not None and type(event.get('action')) is int and
              event['action'] == pending and event.get('phase') == phase):
            pending = None
        elif kind in ('control-factors', 'bias-origin') and phase == 'development' and pending is None:
            continue
        else:
            raise ValueError('earlier stage contains incomplete or contradictory actions')
    if pending is not None:
        raise ValueError('earlier stage ends with an incomplete action')
    if (claim.get('schema') != 'distilbert-color-stage-claim-v1' or
            result.get('schema') != 'distilbert-color-stage-result-v1' or
            exit_.get('schema') != 'distilbert-color-stage-exit-v1' or
            result.get('fabricated') is not None or claim.get('fabricated') is not None or
            claim.get('phase') != phase or claim.get('source_commit') != source or
            type(claim.get('source_sha256')) is not dict or _json_bytes(claim['source_sha256']) != _json_bytes(source_hashes) or
            result.get('phase') != phase or result.get('source_commit') != source or
            type(result.get('source_sha256')) is not dict or _json_bytes(result['source_sha256']) != _json_bytes(source_hashes) or
            result.get('status') != 'complete' or result.get('events_sha256') != _sha(raw) or
            result.get('case_sha256') != CASE_SHA256 or type(result.get('event_bytes')) is not int or
            result['event_bytes'] != len(raw) or type(result.get('starts')) is not int or
            result['starts'] != sum(event.get('kind') == 'start' for event in parsed) or
            exit_.get('phase') != phase or exit_.get('source_commit') != source or
            type(exit_.get('returncode')) is not int or exit_['returncode'] != 0 or exit_.get('timed_out') is not False or
            exit_.get('cleanup') != 'direct-child-reaped' or exit_.get('bytecode_prefix_empty') is not True or
            worker.get('phase') != phase or worker.get('source_commit') != source or
            type(worker.get('pid')) is not int or worker['pid'] != exit_.get('pid') or
            exit_.get('claim_sha256') != _file_sha(stage / 'claim.json')[1] or
            exit_.get('prelaunch_sha256') != _file_sha(stage / 'prelaunch.json')[1] or
            exit_.get('worker_start_sha256') != _file_sha(stage / 'worker-start.json')[1] or
            exit_.get('result_sha256') != _file_sha(stage / 'result.json')[1] or
            exit_.get('events_sha256') != _sha(raw)):
        raise ValueError('earlier stage receipt or saved bytes differ')
    _validate_runtime(result.get('runtime'), claim)
    if require_finalized:
        marker = _read_json(stage / 'validated.json')
        expected_files = ('claim.json', 'prelaunch.json', 'worker-start.json', 'events.jsonl', 'result.json', 'exit.json')
        if (type(marker) is not dict or marker.get('schema') != 'distilbert-color-stage-validation-v1' or
                marker.get('phase') != phase or marker.get('source_commit') != source or
                type(marker.get('supervisor_pid')) is not int or marker['supervisor_pid'] <= 0 or
                marker.get('files_sha256') != {name: _file_sha(stage / name)[1] for name in expected_files}):
            raise ValueError('earlier stage final validation marker differs')
        observed = _read_json(stage / 'supervisor-observation.json')
        if (type(observed) is not dict or observed.get('schema') != 'distilbert-color-supervisor-observation-v1' or
                observed.get('phase') != phase or observed.get('source_commit') != source or
                type(observed.get('supervisor_pid')) is not int or observed['supervisor_pid'] != marker['supervisor_pid'] or
                type(observed.get('returncode')) is not int or observed['returncode'] != 0 or
                observed.get('validation_sha256') != _file_sha(stage / 'validated.json')[1] or
                observed.get('run_sha256') != _file_sha(stage.parent / 'run.json')[1] or
                type(observed.get('observer_external_bytes')) is not int or
                not 1 <= observed['observer_external_bytes'] <= 16384 - (stage / 'supervisor-observation.json').stat().st_size or
                type(observed.get('command_sha256')) is not str or
                re.fullmatch('[0-9a-f]{64}', observed['command_sha256']) is None or
                type(observed.get('scope')) is not str or not observed['scope']):
            raise ValueError('earlier supervisor exit observation differs')
    return parsed, result


def _publication_ancestry(repo, source, phase, pubs):
    """Check commit ordering; an independent observer must verify publication."""
    ordered = [source]
    if phase in ('development', 'evaluation'):
        ordered.append(pubs['screen']['commit'])
    if phase == 'evaluation':
        ordered.append(pubs['development']['commit'])
    for parent, child in zip(ordered, ordered[1:]):
        if (type(parent) is not str or type(child) is not str or
                re.fullmatch('[0-9a-f]{40}', parent) is None or
                re.fullmatch('[0-9a-f]{40}', child) is None or parent == child):
            raise ValueError('publication ordering requires distinct immutable commits')
        subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', parent, child],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
    return ordered


def _check_prior_runtime(current, *prior_results):
    """Require the complete registered baseline and runtime to remain fixed."""
    if (type(current) is not dict or type(current.get('model_state_sha256')) is not str or
            re.fullmatch('[0-9a-f]{64}', current['model_state_sha256']) is None):
        raise ValueError('current model state digest is invalid')
    for result in prior_results:
        saved = result.get('runtime')
        if (type(saved) is not dict or type(saved.get('model_state_sha256')) is not str or
                re.fullmatch('[0-9a-f]{64}', saved['model_state_sha256']) is None or
                _json_bytes(saved) != _json_bytes(current)):
            raise ValueError('loaded original model state or runtime differs from the public predecessor')


def _worker(repo, cache, site, scratch, root, phase):
    if phase != 'screen':
        raise ValueError('this owner implements the screen only')
    _check_launch(site)
    stage = root / phase
    run, claim = _read_json(root / 'run.json'), _read_json(stage / 'claim.json')
    source = run['source_commit']
    scratch = Path(scratch)
    if (claim.get('source_commit') != source or claim.get('phase') != phase or scratch.is_symlink() or
            not scratch.is_dir() or not scratch.name.startswith('color-owner-') or
            claim.get('scratch_root_sha256') != _sha(os.fsencode(str(scratch.resolve(strict=True).parent))) or
            claim.get('site_path_sha256') != _sha(os.fsencode(str(Path(site).resolve(strict=True))))):
        raise ValueError('worker claim differs from run')
    _write(stage / 'worker-start.json', _json_bytes({'phase': phase, 'pid': os.getpid(), 'source_commit': source,
                                                      'started_utc': _now()}))
    try:
        if _file_sha(stage / 'prelaunch.json')[1] != claim.get('prelaunch_sha256'):
            raise ValueError('claimed prelaunch bytes differ')
        _verify_prelaunch(repo, source, phase, (stage / 'prelaunch.json').read_bytes(), check_main=False)
        used, starts = _inspect_run(root)
        current_stage_bytes = sum(path.stat().st_size for path in stage.iterdir()) + claim['external_prelaunch_bytes']
        if (type(claim.get('journal_byte_cap')) is not int or type(claim.get('remaining_starts')) is not int or
                type(claim.get('prior_public_index_bytes')) is not int or
                not 1 <= claim['journal_byte_cap'] <= records.MAX_BYTES or
                claim.get('prior_retained_bytes') != used - current_stage_bytes or
                claim.get('prior_starts') != starts or
                claim['journal_byte_cap'] > records.MAX_BYTES - used - claim['prior_public_index_bytes'] - TERMINAL_RESERVE or
                claim['remaining_starts'] != MAX_SCREEN_STARTS - starts):
            raise ValueError('worker aggregate budget differs from retained files')
        expected_prior = set(() if phase == 'screen' else ('screen',) if phase == 'development' else ('screen', 'development'))
        if type(claim.get('prior_publications')) is not dict or set(claim['prior_publications']) != expected_prior:
            raise ValueError('claimed predecessor set differs')
        source_hashes = verify_git_sources(repo, source, SOURCE_PATHS)
        if type(claim.get('source_sha256')) is not dict or _json_bytes(claim['source_sha256']) != _json_bytes(source_hashes):
            raise ValueError('worker claim source bytes differ')
        ordered = _publication_ancestry(repo, source, phase, claim['prior_publications'])
        if claim.get('publication_ancestry') != ordered:
            raise ValueError('claimed publication ancestry differs')
        index_bytes = 0
        for prior, ref in claim.get('prior_publications', {}).items():
            if prior not in ('screen', 'development') or set(ref) != {'commit', 'prefix'}:
                raise ValueError('invalid claimed predecessor')
            _valid_prior(root / prior, prior, source, source_hashes)
            proof = publication.verify_publication(repo, ref['commit'], ref['prefix'], root / prior, prior)
            index_bytes += proof['index_bytes'] + proof['run_sidecar_bytes']
        if index_bytes != claim['prior_public_index_bytes']:
            raise ValueError('claimed public archive-index budget differs')
        manifest = publication.strict_json((repo / public.CASE_PATH).read_bytes())
        if _file_sha(repo / public.CASE_PATH)[1] != CASE_SHA256:
            raise ValueError('fixed case bytes differ')
        current = public.scan(repo, claim.get('launch_main'), manifest['cases'])
        if claim.get('launch_public_scan_sha256') != _sha(_json_bytes(current)):
            raise ValueError('launch-boundary public scan does not reconstruct')
        engine, runtime = _load_model(cache, manifest, repo, scratch)
        runtime['site_path_sha256'] = _sha(os.fsencode(str(Path(site).resolve(strict=True))))
        runtime['scratch_root_sha256'] = claim['scratch_root_sha256']
        prior_screen = prior_dev = None
        previous_results = []
        if phase in ('development', 'evaluation'):
            prior_screen, screen_result = _valid_prior(root / 'screen', 'screen', source, source_hashes)
            previous_results.append(screen_result)
        if phase == 'evaluation':
            prior_dev, dev_result = _valid_prior(root / 'development', 'development', source, source_hashes)
            previous_results.append(dev_result)
        if previous_results:
            _check_prior_runtime(runtime, *previous_results)
        with records.Journal(stage / 'events.jsonl', byte_cap=claim['journal_byte_cap'],
                             forward_cap=claim['remaining_starts']) as journal:
            if phase == 'screen':
                stages.run_screen(engine, journal, manifest)
            elif phase == 'development':
                preliminary = stages.run_development(engine, journal, manifest, prior_screen)
            else:
                stages.run_evaluation(engine, journal, manifest, prior_screen, prior_dev, dev_result['summary'])
        events_raw = (stage / 'events.jsonl').read_bytes()
        events = records.parse_jsonl(events_raw)
        if phase == 'screen':
            summary = preference.rebuild_screen(manifest, events)
        elif phase == 'development':
            summary, _, _ = stages.rebuild_development(engine, manifest, prior_screen, events)
            if _json_bytes(summary) != _json_bytes(preliminary):
                raise ValueError('development decisions differ from saved raw evidence')
        else:
            summary = stages.rebuild_evaluation(engine, manifest, prior_screen, prior_dev, events)
        from .bert_agreement_study import verify_model_state
        verify_model_state(engine.model, engine.state)
        if verify_git_sources(repo, source, SOURCE_PATHS) != source_hashes:
            raise RuntimeError('frozen source changed during the stage')
        _check_launch(site)
        import torch
        _require_denormals(torch)
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA was initialized during the CPU stage')
        result = {'schema': 'distilbert-color-stage-result-v1', 'phase': phase, 'status': 'complete',
                  'source_commit': source, 'source_sha256': source_hashes, 'runtime': runtime,
                  'case_sha256': CASE_SHA256, 'events_sha256': _sha(events_raw),
                  'event_bytes': len(events_raw), 'starts': sum(e.get('kind') == 'start' for e in events),
                  'summary': summary}
        result_raw = _json_bytes(result)
        if len(result_raw) > TERMINAL_RESERVE - 2 * 1024 * 1024:
            raise ValueError('terminal result exceeds reserved bytes')
        _write(stage / 'result.json', result_raw)
    except BaseException as error:
        failure = {'schema': 'distilbert-color-stage-failure-v1', 'phase': phase,
                   'source_commit': source, 'type': type(error).__name__, 'message': str(error)[:2048],
                   'status': 'invalid-incomplete'}
        _write(stage / 'failure.json', _json_bytes(failure))
        raise


def supervise(*, repo, cache, site, scratch, root, phase, source, prelaunch_path, prelaunch_sha,
              publications=None, timeout=900):
    _check_launch(site)
    repo, cache, site, scratch = (Path(path).resolve(strict=True) for path in (repo, cache, site, scratch))
    root_path = Path(root).absolute()
    root = root_path.parent.resolve(strict=True) / root_path.name
    if root.is_symlink():
        raise ValueError('run root may not be a symbolic link')
    if phase != 'screen' or type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError('invalid phase or watchdog')
    if re.fullmatch('[0-9a-f]{40}', source or '') is None:
        raise ValueError('a full immutable source commit is required')
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in (repo, cache, site)):
        raise ValueError('run, checkout, cache and dependency site must be separate')
    if (not scratch.is_dir() or any(scratch == path or scratch.is_relative_to(path) or path.is_relative_to(scratch)
                                    for path in (repo, cache, site, root))):
        raise ValueError('scratch must be disjoint from checkout, cache, site and run')
    if (shutil.disk_usage(scratch).free < 512 * 1024 * 1024 or shutil.disk_usage(root.parent).free < 192 * 1024 * 1024 or
            any(shutil.disk_usage(path).free < 128 * 1024 * 1024 for path in (repo, cache, site))):
        raise RuntimeError('insufficient free space for bounded checkpoint staging and evidence')
    source_hashes = verify_git_sources(repo, source, SOURCE_PATHS)
    prelaunch_raw = Path(prelaunch_path).read_bytes()
    if not 1 <= len(prelaunch_raw) <= 256 * 1024 or _sha(prelaunch_raw) != prelaunch_sha:
        raise ValueError('prelaunch observation hash differs')
    admission = _verify_prelaunch(repo, source, phase, prelaunch_raw)
    current_scan = admission['current_scan']
    pubs = publications or {}
    expected_prior = set(() if phase == 'screen' else ('screen',) if phase == 'development' else ('screen', 'development'))
    if set(pubs) != expected_prior:
        raise ValueError('exactly the required predecessor publications must be specified')
    for ref in pubs.values():
        if type(ref) is not dict or set(ref) != {'commit', 'prefix'}:
            raise ValueError('invalid claimed predecessor')
    ordered = _publication_ancestry(repo, source, phase, pubs)
    if phase == 'screen':
        root.mkdir(exist_ok=False)
        _write(root / 'run.json', _json_bytes({'schema': 'distilbert-color-run-v1',
                                               'source_commit': source, 'case_sha256': CASE_SHA256,
                                               'created_utc': _now()}))
    else:
        run = _read_json(root / 'run.json')
        if run.get('source_commit') != source or run.get('case_sha256') != CASE_SHA256:
            raise ValueError('run identity differs')
    index_bytes = 0
    for prior in (() if phase == 'screen' else ('screen',) if phase == 'development' else ('screen', 'development')):
        events, result = _valid_prior(root / prior, prior, source, source_hashes)
        if prior not in pubs or set(pubs[prior]) != {'commit', 'prefix'}:
            raise ValueError('a public immutable predecessor archive is required')
        proof = publication.verify_publication(repo, pubs[prior]['commit'], pubs[prior]['prefix'], root / prior, prior)
        index_bytes += proof['index_bytes'] + proof['run_sidecar_bytes']
        if prior == 'screen':
            manifest = publication.strict_json((repo / public.CASE_PATH).read_bytes())
            decision = preference.rebuild_screen(manifest, events)
            if decision['status'] != 'eligible' or _json_bytes(decision) != _json_bytes(result['summary']):
                raise ValueError('public screen is ineligible or its saved decision differs')
    used, starts = _inspect_run(root)
    remaining = records.MAX_BYTES - used - index_bytes - 2 * len(prelaunch_raw) - TERMINAL_RESERVE - 16384
    if remaining <= 0 or starts >= MAX_SCREEN_STARTS:
        raise ValueError('no budget remains for a new stage')
    claim = {'schema': 'distilbert-color-stage-claim-v1', 'phase': phase, 'source_commit': source,
             'source_sha256': source_hashes, 'prelaunch_sha256': prelaunch_sha,
             'scratch_root_sha256': _sha(os.fsencode(str(scratch))),
             'site_path_sha256': _sha(os.fsencode(str(site))),
             'external_prelaunch_bytes': len(prelaunch_raw),
             'prior_publications': pubs, 'prior_retained_bytes': used, 'prior_public_index_bytes': index_bytes,
             'prior_starts': starts, 'journal_byte_cap': remaining,
             'remaining_starts': MAX_SCREEN_STARTS - starts, 'publication_ancestry': ordered,
             'launch_main': current_scan['commit'], 'launch_public_scan_sha256': _sha(_json_bytes(current_scan)),
             'claimed_utc': _now()}
    claim_raw = _json_bytes(claim)
    if len(claim_raw) > 16384:
        raise ValueError('serialized claim exceeds its reserved budget')
    stage = root / phase
    stage.mkdir(exist_ok=False)
    _write(stage / 'prelaunch.json', prelaunch_raw)
    _write(stage / 'claim.json', claim_raw)
    scratch_root = scratch
    child_scratch = Path(tempfile.mkdtemp(prefix='color-owner-', dir=scratch_root))
    pycache = child_scratch / 'pycache'
    temp = child_scratch / 'temp'
    pycache.mkdir()
    temp.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE', 'PYTHONPYCACHEPREFIX', 'PYTHONSTARTUP', 'PYTHONUSERBASE'):
        env.pop(key, None)
    env['TEMP'] = env['TMP'] = str(temp)
    command = [sys.executable, '-E', '-s', '-B', '-X', 'pycache_prefix=' + str(pycache),
               '-c', BOOTSTRAP, str(repo), str(site), '--worker', '--repo', str(repo), '--cache', str(cache),
               '--site', str(site), '--scratch', str(child_scratch), '--run-root', str(root), '--phase', phase]
    started = _now()
    began = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except BaseException as error:
        receipt = {'schema': 'distilbert-color-stage-exit-v1', 'phase': phase, 'source_commit': source,
                   'pid': None, 'started_utc': started, 'finished_utc': _now(),
                   'elapsed_seconds': time.monotonic() - began, 'returncode': None, 'timed_out': False,
                   'spawn_failure_type': type(error).__name__, 'claim_sha256': _file_sha(stage / 'claim.json')[1],
                   'prelaunch_sha256': _file_sha(stage / 'prelaunch.json')[1], 'worker_start_sha256': None,
                   'events_sha256': None, 'result_sha256': None,
                   'scope': 'Spawn did not return a child handle; whether a child started or exited is unknown.'}
        _write(stage / 'exit.json', _json_bytes(receipt))
        if isinstance(error, OSError):
            raise RuntimeError('spawn outcome is unknown; retain the one-shot record') from error
        raise
    timed_out = False
    cleanup = 'direct-child-reaped'
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        code, cleanup = _kill_and_reap(process)
    except BaseException as error:
        code, cleanup = _kill_and_reap(process)
        receipt = {'schema': 'distilbert-color-stage-exit-v1', 'phase': phase, 'source_commit': source,
                   'pid': process.pid, 'started_utc': started, 'finished_utc': _now(),
                   'elapsed_seconds': time.monotonic() - began, 'returncode': code, 'timed_out': False,
                   'interruption_type': type(error).__name__, 'cleanup': cleanup,
                   'claim_sha256': _file_sha(stage / 'claim.json')[1],
                   'prelaunch_sha256': _file_sha(stage / 'prelaunch.json')[1],
                   'worker_start_sha256': None, 'events_sha256': None, 'result_sha256': None,
                   'scope': 'Interrupted supervisor; child cleanup is reported only to the stated extent.'}
        _write(stage / 'exit.json', _json_bytes(receipt))
        if cleanup == 'direct-child-reaped':
            _remove_scratch(child_scratch, scratch_root)
        raise
    bytecode_empty = not any(pycache.iterdir()) if cleanup == 'direct-child-reaped' else None
    if cleanup == 'direct-child-reaped' and bytecode_empty:
        _remove_scratch(child_scratch, scratch_root)
    receipt = {'schema': 'distilbert-color-stage-exit-v1', 'phase': phase, 'source_commit': source,
               'pid': process.pid, 'started_utc': started, 'finished_utc': _now(),
               'elapsed_seconds': time.monotonic() - began, 'returncode': code, 'timed_out': timed_out,
               'cleanup': cleanup, 'bytecode_prefix_empty': bytecode_empty,
               'claim_sha256': _file_sha(stage / 'claim.json')[1],
               'prelaunch_sha256': _file_sha(stage / 'prelaunch.json')[1],
               'worker_start_sha256': _file_sha(stage / 'worker-start.json')[1] if cleanup == 'direct-child-reaped' and (stage / 'worker-start.json').is_file() else None,
               'events_sha256': _file_sha(stage / 'events.jsonl')[1] if cleanup == 'direct-child-reaped' and (stage / 'events.jsonl').is_file() else None,
               'result_sha256': _file_sha(stage / 'result.json')[1] if cleanup == 'direct-child-reaped' and (stage / 'result.json').is_file() else None,
               'scope': 'Direct child exit and saved files observed by this cooperative supervisor.'}
    _write(stage / 'exit.json', _json_bytes(receipt))
    final_bytes, final_starts = _inspect_run(root)
    if final_bytes + index_bytes > records.MAX_BYTES or final_starts > MAX_SCREEN_STARTS:
        raise RuntimeError('aggregate retained evidence or forward limit was exceeded')
    if code != 0 or timed_out or cleanup != 'direct-child-reaped' or not bytecode_empty:
        raise RuntimeError('stage exited unsuccessfully; retain the incomplete one-shot attempt')
    events, result = _valid_prior(stage, phase, source, source_hashes, require_finalized=False)
    if phase == 'screen':
        manifest = publication.strict_json((repo / public.CASE_PATH).read_bytes())
        if _json_bytes(preference.rebuild_screen(manifest, events)) != _json_bytes(result['summary']):
            raise ValueError('screen result does not match the nine raw records')
    _check_launch(site)
    validated_files = ('claim.json', 'prelaunch.json', 'worker-start.json', 'events.jsonl', 'result.json', 'exit.json')
    final_raw = _json_bytes({'schema': 'distilbert-color-stage-validation-v1', 'phase': phase,
                             'supervisor_pid': os.getpid(),
                             'source_commit': source, 'files_sha256': {name: _file_sha(stage / name)[1]
                                                                        for name in validated_files},
                             'validated_utc': _now(),
                             'scope': 'Cooperative supervisor completed checks; not an independent exit observation.'})
    if final_bytes + index_bytes + len(final_raw) + 16384 > records.MAX_BYTES:
        raise RuntimeError('final validation and observer reserve would exceed the retained evidence limit')
    _write(stage / 'validated.json', final_raw)
    return {'phase': phase, 'returncode': code, 'run_root': str(root), 'retained_bytes': final_bytes + index_bytes + len(final_raw),
            'started_forwards': final_starts, 'exit_sha256': _file_sha(stage / 'exit.json')[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--phase', choices=('screen',), required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--prelaunch', type=Path)
    parser.add_argument('--prelaunch-sha256')
    parser.add_argument('--screen-public-commit')
    parser.add_argument('--screen-public-prefix')
    parser.add_argument('--development-public-commit')
    parser.add_argument('--development-public-prefix')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.worker:
        _worker(args.repo.resolve(strict=True), args.cache.resolve(strict=True), args.site.resolve(strict=True),
                args.scratch.resolve(strict=True),
                args.run_root.resolve(strict=True), args.phase)
        return
    if not args.source_commit or not args.prelaunch or not args.prelaunch_sha256:
        parser.error('source commit, scoped prelaunch file and its SHA-256 are required')
    pubs = {}
    for name in ('screen', 'development'):
        commit, prefix = getattr(args, name + '_public_commit'), getattr(args, name + '_public_prefix')
        if commit or prefix:
            if not commit or not prefix:
                parser.error('both publication commit and prefix are required')
            pubs[name] = {'commit': commit, 'prefix': prefix}
    print(json.dumps(supervise(repo=args.repo, cache=args.cache, site=args.site, scratch=args.scratch,
                               root=args.run_root, phase=args.phase,
                               source=args.source_commit, prelaunch_path=args.prelaunch,
                               prelaunch_sha=args.prelaunch_sha256, publications=pubs, timeout=args.timeout),
                     sort_keys=True))


if __name__ == '__main__':
    main()
