"""Fixed public CPU FLAN-T5 dose response with equal-length targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .flan_t5_dose_response import check_cases as check_prior_cases, run_cases
from .flan_t5_equal_dose import (
    Evidence as BaseEvidence, MAX_RETURN_BYTES, check_publication_observation,
    hook_snapshot, model_snapshot, require_clean_hooks, run_forward,
)
from .flan_t5_feasibility import MANIFEST_HASH, REVISION, digest_file, memory_bytes, verify_assets

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / 'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json'
SCREEN = REPO / 'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json'
PLAN = REPO / 'docs/FLAN_T5_EQUAL_LENGTH_SELECTION_V1.md'
PROTOCOL = REPO / 'docs/FLAN_T5_EQUAL_LENGTH_RESPONSE_V1.md'
CASE_HASH = '8d1ea3b124085ca8817dc80044dca6fd1e897bda7294510ad99e419c01e7708b'
SCREEN_HASH = 'adabba92e451fb0c27415b8ec67b67134b59bb5aac293b3c056936cfe872c0a2'
PLAN_HASH = '1fa773268a2bbfd73979c2ce9f83f6a1d821f6607c37c12460b2368b2ca0668c'
PROTOCOL_HASH = 'c4167f17074b58297119514c9290d0a2b0ba95028b3ab55f2417a56abc6f9154'
SELECTED = (0, 1, 2, 3, 5, 7)
MAX_FORWARDS = 936
MAX_BYTES = 512 * 1024 * 1024
SEED = 20261028
_JSON_LIMIT = 65536
_TERMINAL_LIMIT = 1024 * 1024


def source_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    paths = ('looking_glass/flan_t5_equal_length_response.py', 'looking_glass/flan_t5_dose_response.py',
             'looking_glass/flan_t5_equal_dose.py', 'looking_glass/flan_t5_feasibility.py',
             'looking_glass/interventions.py', 'looking_glass/intervention_plan.py',
             'looking_glass/model_ownership.py', 'docs/FLAN_T5_EQUAL_LENGTH_RESPONSE_V1.md',
             'docs/FLAN_T5_EQUAL_LENGTH_SELECTION_V1.md', 'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json',
             'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json', 'tests/flan_t5_equal_length_tokenizer_screen.py')
    hashes = {}
    for name in paths:
        raw = (REPO / name).read_bytes()
        if raw != subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=REPO):
            raise ValueError(f'source differs from committed blob: {name}')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': commit, 'source_sha256': hashes}


def check_selection(manifest, screen):
    """Require the saved screen and selected rows to agree with the fixed rule."""
    candidates = screen.get('candidates', ())
    if (screen.get('schema') != 'mlg-flan-t5-equal-length-candidate-screen-v1'
            or screen.get('status') != 'complete' or screen.get('selection_plan_sha256') != PLAN_HASH
            or screen.get('revision') != REVISION or screen.get('asset_manifest_sha256') != MANIFEST_HASH
            or len(candidates) != 16 or [c.get('candidate_index') for c in candidates] != list(range(16))
            or any(type(c.get('eligible')) is not bool for c in candidates)
            or tuple(c['candidate_index'] for c in candidates if c['eligible'])[:6] != SELECTED
            or tuple(screen.get('selected_candidate_indices', ())) != SELECTED
            or manifest.get('schema') != 'mlg-flan-t5-equal-length-dose-response-cases-v1'
            or manifest.get('selection_plan_sha256') != PLAN_HASH
            or manifest.get('candidate_screen_sha256') != SCREEN_HASH
            or manifest.get('revision') != REVISION or manifest.get('asset_manifest_sha256') != MANIFEST_HASH
            or len(manifest.get('cases', ())) != 12):
        raise ValueError('candidate screening or selection differs')
    for i, case in enumerate(manifest['cases']):
        candidate = candidates[SELECTED[i // 2]]
        expected = candidate['cases'][i % 2]
        stripped = {k: v for k, v in case.items() if k not in ('candidate_index', 'pair_id', 'case_id')}
        if (case.get('candidate_index') != SELECTED[i // 2] or case.get('pair_id') != i // 2
                or case.get('case_id') != f"pair{i // 2}-{'translate' if i % 2 == 0 else 'question'}"
                or stripped != expected):
            raise ValueError('selected case differs from its screened candidate')


def check_cases(tokenizer, manifest):
    """Retokenize the fixed rows and apply the original alignment checks."""
    rows = manifest.get('cases', ())
    if len(rows) != 12:
        raise ValueError('expected 12 fixed cases')
    eligibility = []
    for start in (0, 6):
        normalized = []
        for i, row in enumerate(rows[start:start + 6]):
            if row.get('pair_id') != start // 2 + i // 2 or row.get('template_id') != ('translate', 'question')[i % 2]:
                raise ValueError('case order differs')
            for target in row['targets']:
                ids = target['input_ids']
                if (len(ids) != 2 or ids[1] != 1 or ids[0] == tokenizer.unk_token_id
                        or target['special_tokens_mask'] != [0, 1]
                        or target['text'][target['offsets'][0][0]:target['offsets'][0][1]].strip() != target['text']):
                    raise ValueError('target must be one lexical token and EOS')
            if any(len(source['input_ids']) > 9 for source in [row['neutral'], *row['source']]):
                raise ValueError('fixed source shape exceeds its evidence bound')
            copy = dict(row)
            copy['pair_id'] = i // 2
            copy['case_id'] = f"pair{i // 2}-{'translate' if i % 2 == 0 else 'question'}"
            normalized.append(copy)
        prior = dict(manifest, schema='mlg-flan-t5-dose-response-public-cases-v1', cases=normalized)
        eligibility.extend(check_prior_cases(tokenizer, prior))
    return eligibility


def pair_decisions(cells):
    allowed = ('positive', 'negative', 'alignment-ineligible', 'baseline-unsupported',
               'dose-ineligible', 'dose-unmatchable')
    expected = [(i, t) for i in range(6) for t in ('translate', 'question')]
    if (len(cells) != 12 or any(c.get('status') not in allowed for c in cells)
            or [(c.get('pair_id'), c.get('template_id')) for c in cells] != expected):
        return None
    pairs = []
    for i in range(6):
        statuses = [c['status'] for c in cells[2*i:2*i+2]]
        if any(s not in ('positive', 'negative') for s in statuses):
            pairs.append('unevaluable')
        elif all(s == 'positive' for s in statuses):
            pairs.append('positive')
        else:
            pairs.append('negative')
    return pairs


def aggregate_decision(cells):
    pairs = pair_decisions(cells)
    if pairs is None:
        return 'incomplete'
    positive, negative = pairs.count('positive'), pairs.count('negative')
    if positive >= 3 and positive > negative:
        return 'positive'
    if negative >= 3 and negative > positive:
        return 'negative'
    return 'inconclusive'


def fixed_evidence_bound():
    """Conservative apparent-byte budget for valid fixed-shape returns and one failed return."""
    tensors = MAX_FORWARDS * (2 * 32128 + 4 * 9 * 512) * 4
    forward_json = MAX_FORWARDS * 2 * _JSON_LIMIT
    dose_vectors = 12 * 6 * 5 * 512 * 4
    # At most 12 case summaries, 12 dose manifests, started and environment.
    other_json = 26 * _JSON_LIMIT + _TERMINAL_LIMIT
    failure_and_observation = 2 * _JSON_LIMIT + MAX_RETURN_BYTES
    total = tensors + forward_json + dose_vectors + other_json + failure_and_observation
    if total + MAX_RETURN_BYTES + 3 * _JSON_LIMIT > MAX_BYTES:
        raise ValueError('fixed evidence budget does not fit the reserve')
    return {'valid_tensor_bytes': tensors, 'forward_json_bytes': forward_json,
            'dose_vector_bytes': dose_vectors, 'other_json_bytes': other_json,
            'failure_and_observation_bytes': failure_and_observation, 'total_bytes': total,
            'limit_bytes': MAX_BYTES}


class Evidence(BaseEvidence):
    def json(self, name, value, reserve=65536):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
        limit = _TERMINAL_LIMIT if name == 'completed.json' else _JSON_LIMIT
        if len(raw) > limit:
            raise ValueError('JSON evidence exceeds the fixed per-file bound')
        return self.write(name, raw, reserve)

def run(asset_dir, output, publication_observation):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1')
            or os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1'
            or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or os.environ.get('OMP_NUM_THREADS') != '1'
            or os.environ.get('MKL_NUM_THREADS') != '1'):
        raise ValueError('fixed CPU/offline/numeric environment is required')
    if 'torch' in sys.modules:
        raise ValueError('PyTorch must not be preloaded before environment checks')
    if (digest_file(CASES) != CASE_HASH or digest_file(SCREEN) != SCREEN_HASH
            or digest_file(PLAN) != PLAN_HASH or digest_file(PROTOCOL) != PROTOCOL_HASH):
        raise ValueError('fixed protocol, selection or cases differ')
    bound = fixed_evidence_bound()
    manifest = json.loads(CASES.read_bytes())
    check_selection(manifest, json.loads(SCREEN.read_bytes()))
    source_before = source_identity()
    observation = check_publication_observation(publication_observation, source_before)
    assets_before = verify_assets(asset_dir)
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 2 * MAX_BYTES:
        raise ValueError('insufficient free space or missing output parent')
    evidence = Evidence(output, limit=MAX_BYTES)
    observation_hash = evidence.write('publication-observation.json', observation)
    evidence.json('started.json', {'schema': 'mlg-flant5-equal-length-response-attempt-v1', 'status': 'started',
                                   'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH,
                                   'case_manifest_sha256': CASE_HASH, 'evidence_cap': MAX_BYTES,
                                   'selection_plan_sha256': PLAN_HASH, 'candidate_screen_sha256': SCREEN_HASH,
                                   'fixed_evidence_bound': bound,
                                   'publication_observation_sha256': observation_hash,
                                   'max_forwards': MAX_FORWARDS})
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
        eligibility = check_cases(tokenizer, manifest)
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
        result = {'schema': 'mlg-flant5-equal-length-response-v1', 'status': 'complete',
                  'decision': aggregate_decision(summaries), 'pair_status': pair_decisions(summaries),
                  'cases': summaries, 'forwards': len(records), 'fixed_evidence_bound': bound,
                  'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH, 'case_manifest_sha256': CASE_HASH,
                  'selection_plan_sha256': PLAN_HASH, 'candidate_screen_sha256': SCREEN_HASH,
                  'environment': runtime, 'final_memory': memory_bytes(), 'hooks_restored': True,
                  'bytes_before_terminal': evidence.used,
                  'limitations': ['Public, correlated pairs and templates; no statistical or generalization claim.',
                                  'Matched input L2 does not match direction or downstream functional effect.',
                                  'Exact sequence scores include EOS; equal token counts do not equalize target priors.']}
        sha = evidence.json('completed.json', result, reserve=0)
        print(json.dumps({'status': 'complete', 'decision': result['decision'], 'forwards': len(records),
                          'result_sha256': sha, 'retained_bytes': evidence.used}), flush=True)
        return result
    except BaseException as error:
        try:
            evidence.json('failure.json', {'schema': 'mlg-flant5-equal-length-response-attempt-v1', 'status': 'incomplete',
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
    args = parser.parse_args()
    run(args.asset_dir.resolve(), args.output.resolve(), args.publication_observation.resolve())


if __name__ == '__main__':
    main()
