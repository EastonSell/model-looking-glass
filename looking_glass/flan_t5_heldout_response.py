"""Source-bound CPU runner for a locally withheld FLAN-T5 response study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

from .flan_t5_equal_dose import (
    Evidence as BaseEvidence, MAX_RETURN_BYTES, check_publication_observation,
    hook_snapshot, model_snapshot, require_clean_hooks, run_forward,
)
from .flan_t5_feasibility import MANIFEST_HASH, REVISION, memory_bytes, verify_assets
from .flan_t5_heldout_experiment import all_track_decisions, run_cases

REPO = Path(__file__).resolve().parent.parent
PROTOCOL = REPO / 'docs/FLAN_T5_HELDOUT_RESPONSE_V1.md'
MAX_FORWARDS = 936
CUSTODY_LIMIT = 4 * 1024 * 1024
RUN_LIMIT = 508 * 1024 * 1024
SEED = 20261028
_JSON_LIMIT = 65536
_TERMINAL_LIMIT = 1024 * 1024
_HEX40 = re.compile(r'[0-9a-f]{40}\Z')
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def source_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    paths = ('looking_glass/__init__.py', 'looking_glass/flan_t5_heldout_owner.py', 'looking_glass/flan_t5_heldout_response.py',
             'looking_glass/flan_t5_heldout_experiment.py',
             'looking_glass/flan_t5_heldout_custody.py', 'looking_glass/flan_t5_dose_response.py',
             'looking_glass/flan_t5_equal_dose.py', 'looking_glass/flan_t5_feasibility.py',
             'looking_glass/interventions.py', 'looking_glass/intervention_plan.py',
             'looking_glass/model_ownership.py', 'docs/FLAN_T5_HELDOUT_RESPONSE_V1.md',
             'docs/FLAN_T5_HELDOUT_CUSTODY_V1.md',
             'docs/FLAN_T5_PUBLIC_CASES_V1.json', 'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json',
             'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json', 'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json',
             'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json')
    hashes = {}
    for name in paths:
        raw = (REPO / name).read_bytes()
        if raw != subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=REPO):
            raise ValueError(f'source differs from committed blob: {name}')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': commit, 'source_sha256': hashes}


def fixed_evidence_bound():
    tensors = MAX_FORWARDS * (2 * 32128 + 4 * 16 * 512) * 4
    forward_json = MAX_FORWARDS * 2 * _JSON_LIMIT
    dose_vectors = 12 * 6 * 5 * 512 * 4
    other_json = 26 * _JSON_LIMIT + _TERMINAL_LIMIT
    failure_and_observation = 2 * _JSON_LIMIT + MAX_RETURN_BYTES
    total = tensors + forward_json + dose_vectors + other_json + failure_and_observation
    next_forward = MAX_RETURN_BYTES + 3 * _JSON_LIMIT
    if total + next_forward > RUN_LIMIT:
        raise ValueError('fixed run evidence does not fit its reservation')
    return {'valid_tensor_bytes': tensors, 'forward_json_bytes': forward_json,
            'dose_vector_bytes': dose_vectors, 'other_json_bytes': other_json,
            'failure_and_observation_bytes': failure_and_observation, 'total_bytes': total,
            'next_forward_reserve': next_forward, 'run_limit_bytes': RUN_LIMIT,
            'custody_limit_bytes': CUSTODY_LIMIT}


class Evidence(BaseEvidence):
    def json(self, name, value, reserve=65536):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
        limit = _TERMINAL_LIMIT if name == 'completed.json' else _JSON_LIMIT
        if len(raw) > limit:
            raise ValueError('JSON evidence exceeds the fixed per-file bound')
        return self.write(name, raw, reserve)


def bounded_json(path, limit=_JSON_LIMIT):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('public observation exceeds its bound')
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError('public observation is malformed')
    return raw, value


def _repo_path(value, suffix):
    if (type(value) is not str or not value or '\\' in value or value.startswith('/')
            or PurePosixPath(value).as_posix() != value or any(p in ('.', '..') for p in value.split('/'))
            or not value.endswith(suffix)):
        return False
    return True


def check_card_observation(path, kind, card_raw, source):
    raw, value = bounded_json(path)
    commit = value.get('commit')
    if (value.get('schema') != 'mlg-flant5-heldout-card-observation-v1' or value.get('kind') != kind
            or type(commit) is not str or not _HEX40.fullmatch(commit)
            or value.get('commit_url') != f'https://github.com/EastonSell/model-looking-glass/commit/{commit}'
            or not _repo_path(value.get('card_path'), '.json')
            or value.get('card_sha256') != hashlib.sha256(card_raw).hexdigest()
            or value.get('source_commit') != source['commit']
            or not isinstance(value.get('observer'), str) or not value['observer']
            or not isinstance(value.get('observed_utc'), str) or not value['observed_utc']):
        raise ValueError('public commitment observation differs')
    return raw


def check_auditor_observation(path, source):
    raw, value = bounded_json(path)
    commit, digest = value.get('commit'), value.get('auditor_sha256')
    if (value.get('schema') != 'mlg-flant5-heldout-auditor-observation-v1'
            or type(commit) is not str or not _HEX40.fullmatch(commit)
            or type(digest) is not str or not _HEX64.fullmatch(digest)
            or value.get('commit_url') != f'https://github.com/EastonSell/model-looking-glass/commit/{commit}'
            or value.get('source_commit') != source['commit']
            or not _repo_path(value.get('auditor_path'), '.py')
            or not isinstance(value.get('observer'), str) or not value['observer']
            or not isinstance(value.get('observed_utc'), str) or not value['observed_utc']):
        raise ValueError('public auditor observation differs')
    return raw


def complete_track_decisions(summaries):
    decisions = all_track_decisions(summaries)
    if any(value['status'] == 'incomplete' for value in decisions.values()):
        raise ValueError('one or more outcome tracks are incomplete')
    return decisions


def check_custody_identity(verified, source):
    expected = {'source_commit': source['commit'],
                'protocol_sha256': source['source_sha256']['docs/FLAN_T5_HELDOUT_RESPONSE_V1.md'],
                'asset_manifest_sha256': MANIFEST_HASH, 'tokenizer_revision': REVISION}
    if (type(verified) is not dict or verified.get('identity') != expected
            or type(verified.get('cases')) is not list or len(verified['cases']) != 12
            or type(verified.get('custody_bytes')) is not int
            or not 0 <= verified['custody_bytes'] <= CUSTODY_LIMIT):
        raise ValueError('private custody identity or bounds differ')

def run(asset_dir, output, publication_observation, custody_root, pool_observation, case_observation, auditor_observation):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1')
            or os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1'
            or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or os.environ.get('OMP_NUM_THREADS') != '1'
            or os.environ.get('MKL_NUM_THREADS') != '1'):
        raise ValueError('fixed CPU/offline/numeric environment is required')
    if 'torch' in sys.modules:
        raise ValueError('PyTorch must not be preloaded before environment checks')
    from .flan_t5_heldout_custody import inventory_custody, load_exposure, verify_bundle
    custody_root = Path(custody_root).resolve(strict=True)
    output = Path(output)
    if (not output.parent.is_dir() or output.exists() or custody_root.parent != output.parent.resolve(strict=True)
            or custody_root == output.resolve(strict=False)):
        raise ValueError('custody and fresh attempt must be separate protected siblings')
    observations = (publication_observation, pool_observation, case_observation, auditor_observation)
    if any(Path(path).resolve(strict=True).parent != custody_root for path in observations):
        raise ValueError('all retained observations must be in the custody directory')
    source_before = source_identity()
    source_observation = check_publication_observation(publication_observation, source_before)
    pool_raw, pool_card = bounded_json(custody_root / 'pool-commitment.json')
    case_raw, case_card = bounded_json(custody_root / 'case-commitment.json')
    pool_readback = check_card_observation(pool_observation, 'pool', pool_raw, source_before)
    case_readback = check_card_observation(case_observation, 'cases', case_raw, source_before)
    auditor_readback = check_auditor_observation(auditor_observation, source_before)
    initial_inventory = inventory_custody(custody_root)
    if initial_inventory['bytes'] > CUSTODY_LIMIT:
        raise ValueError('custody evidence exceeds its reservation')
    bound = fixed_evidence_bound()
    assets_before = verify_assets(asset_dir)
    if shutil.disk_usage(output.parent).free < 2 * 512 * 1024 * 1024:
        raise ValueError('insufficient free space on output volume')
    evidence = Evidence(output, limit=RUN_LIMIT)
    evidence.json('started.json', {'schema': 'mlg-flant5-heldout-response-attempt-v1', 'status': 'started',
                                   'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH,
                                   'pool_commitment': pool_card.get('commitment'),
                                   'case_commitment': case_card.get('commitment'),
                                   'source_observation_sha256': hashlib.sha256(source_observation).hexdigest(),
                                   'pool_observation_sha256': hashlib.sha256(pool_readback).hexdigest(),
                                   'case_observation_sha256': hashlib.sha256(case_readback).hexdigest(),
                                   'auditor_observation_sha256': hashlib.sha256(auditor_readback).hexdigest(),
                                   'initial_custody_inventory': initial_inventory,
                                   'fixed_evidence_bound': bound, 'run_evidence_cap': RUN_LIMIT,
                                   'custody_evidence_cap': CUSTODY_LIMIT, 'max_forwards': MAX_FORWARDS})
    active = {'case_id': None, 'forward': None, 'phase': 'startup'}
    try:
        import torch
        import tokenizers
        import transformers
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise ValueError('cannot preserve denormals')
        if int((torch.tensor([2.0 ** -149], dtype=torch.float32) * 1.0).view(torch.int32)[0]) != 1:
            raise ValueError('unexpected float32 subnormal behavior')
        torch.random.default_generator.manual_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True,
                                                   use_fast=True, trust_remote_code=False)
        exposure = load_exposure(REPO, source_before['commit'], tokenizer)
        verified = verify_bundle(custody_root, pool_card, case_card, tokenizer, exposure)
        check_custody_identity(verified, source_before)
        manifest = {'cases': verified['cases']}
        eligibility = [[] for _ in manifest['cases']]
        before_load = memory_bytes()
        model, load_info = T5ForConditionalGeneration.from_pretrained(
            str(asset_dir), local_files_only=True, use_safetensors=True, trust_remote_code=False,
            output_loading_info=True, dtype=torch.float32)
        if any(load_info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('model loading diagnostics are not empty')
        model.eval()
        model.config.use_cache = False
        if (type(model).__name__ != 'T5ForConditionalGeneration' or model.config.d_model != 512
                or model.config.vocab_size != 32128 or model.config.decoder_start_token_id != 0
                or model.config.eos_token_id != 1 or model.config.pad_token_id != 0
                or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError('loaded model differs from the fixed CPU configuration')
        hooks_before = hook_snapshot(model)
        require_clean_hooks(hooks_before)
        model_before = model_snapshot(model)
        runtime = {'python': platform.python_version(), 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                   'device': 'cpu', 'dtype': 'float32', 'seed': SEED, 'threads': torch.get_num_threads(),
                   'interop_threads': torch.get_num_interop_threads(),
                   'deterministic': torch.are_deterministic_algorithms_enabled(),
                   'attention_implementation': getattr(model.config, '_attn_implementation', None),
                   'numeric_environment': {k: os.environ.get(k) for k in
                                           ('MKL_CBWR', 'ATEN_CPU_CAPABILITY', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                   'before_load': before_load, 'after_load': memory_bytes(),
                   'model_parameter_buffer_sha256': model_before[0],
                   'torch_build_config': torch.__config__.show()}
        if runtime['attention_implementation'] != 'eager':
            raise ValueError('attention implementation is not eager')
        evidence.json('environment.json', runtime)
        records = []

        def forward(source, target, name, action='clean', value=None, token_index=None):
            return run_forward(model, evidence, records, active, source, target, name, action, value, token_index,
                               forward_limit=MAX_FORWARDS)

        summaries = run_cases(manifest, eligibility, evidence, records, active, forward)
        if source_identity() != source_before or verify_assets(asset_dir) != assets_before or hook_snapshot(model) != hooks_before:
            raise ValueError('source, assets or hooks changed during execution')
        if model_snapshot(model) != model_before:
            raise ValueError('model parameters or buffers changed during execution')
        final_inventory = inventory_custody(custody_root)
        repeated = verify_bundle(custody_root, pool_card, case_card, tokenizer, exposure)
        if (final_inventory['bytes'] > CUSTODY_LIMIT or final_inventory != initial_inventory
                or repeated != verified):
            raise ValueError('private custody changed during execution')
        decisions = complete_track_decisions(summaries)
        result = {'schema': 'mlg-flant5-heldout-response-v1', 'status': 'complete',
                  'decisions': decisions, 'cases': summaries, 'forwards': len(records),
                  'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH,
                  'pool_commitment': pool_card.get('commitment'), 'case_commitment': case_card.get('commitment'),
                  'initial_custody_inventory': initial_inventory,
                  'final_custody_inventory': final_inventory, 'fixed_evidence_bound': bound,
                  'environment': runtime, 'final_memory': memory_bytes(), 'hooks_restored': True,
                  'bytes_before_terminal': evidence.used,
                  'limitations': ['Local withholding does not establish pretraining independence or generalization.',
                                  'Context-reference consistency does not identify or remove lexical priors.',
                                  'Matched input L2 does not match direction or downstream functional effect.']}
        sha = evidence.json('completed.json', result, reserve=0)
        print(json.dumps({'status': 'complete', 'result_sha256': sha}), flush=True)
        return result
    except BaseException as error:
        try:
            evidence.json('failure.json', {'schema': 'mlg-flant5-heldout-response-attempt-v1', 'status': 'incomplete',
                                          'error_type': type(error).__name__, 'active': active,
                                          'retained_bytes_before_failure': evidence.used}, reserve=0)
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--publication-observation', required=True, type=Path)
    parser.add_argument('--custody-root', required=True, type=Path)
    parser.add_argument('--pool-observation', required=True, type=Path)
    parser.add_argument('--case-observation', required=True, type=Path)
    parser.add_argument('--auditor-observation', required=True, type=Path)
    args = parser.parse_args()
    run(args.asset_dir.resolve(), args.output.resolve(), args.publication_observation.resolve(),
        args.custody_root.resolve(), args.pool_observation.resolve(), args.case_observation.resolve(),
        args.auditor_observation.resolve())


if __name__ == '__main__':
    main()
