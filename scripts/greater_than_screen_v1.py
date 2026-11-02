"""Fixed public year-comparison screen; no intervention occurs here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CUDA_VISIBLE_DEVICES must be -1 before interpreter startup')
    if 'torch' in sys.modules:
        raise RuntimeError('torch was imported before the CPU environment snapshot')
CPU_ENV = {name: {'present': name in os.environ, 'value': os.environ.get(name)}
           for name in ('MKL_CBWR', 'ATEN_CPU_CAPABILITY')}
if __name__ == '__main__' and any(value['present'] for value in CPU_ENV.values()):
    raise RuntimeError('CPU backend overrides must be absent at interpreter startup')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network

GUARD_EVENTS, SAFE_QUERIES = [], {'cuda_is_available': 0, 'cuda_device_count': 0}
if __name__ == '__main__':
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)
from scripts import copying_head_eval_v1 as base
from looking_glass import greater_than_screen as screen
from looking_glass.copy_head_intervention import assert_no_forward_hooks

base.GUARD_EVENTS = GUARD_EVENTS
CASE_PATH = 'experiments/distilgpt2_greater_than_screen_v1.json'
CASE_SHA256 = '399994789ee4d3eff27dd6c9cce23ae27ab7761e4bc33caf58de4528194b01d2'
PROTOCOL_PATH = 'docs/DISTILGPT2_GREATER_THAN_SCREEN_V1.md'
SOURCES = ('scripts/greater_than_screen_v1.py', 'looking_glass/greater_than_screen.py',
           'tests/test_greater_than_screen.py', CASE_PATH, PROTOCOL_PATH,
           'scripts/copying_head_eval_v1.py', 'scripts/copying_dev_v1.py',
           'scripts/distilgpt2_intervention_dev_bootstrap.py', 'looking_glass/__init__.py',
           'looking_glass/pretrained.py', 'looking_glass/copy_head_intervention.py')
VOCAB = 50257
LOGITS_BYTES = VOCAB * 4
RAW_BYTES = 48 * LOGITS_BYTES


def source_identity():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, timeout=30)
    revision = git('rev-parse', 'HEAD').decode('ascii').strip()
    tree = git('show', '-s', '--format=%T', revision).decode('ascii').strip()
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('source checkout must be clean')
    sources = {}
    for path in SOURCES:
        raw = (ROOT / path).read_bytes()
        if raw != git('show', f'{revision}:{path}'):
            raise RuntimeError('source file differs from the committed bytes')
        sources[path] = hashlib.sha256(raw).hexdigest()
    return {'revision': revision, 'tree': tree, 'sha256': sources}


def persist_logits(raw_stream, journal, case, payload):
    if type(payload) is not bytes or len(payload) != LOGITS_BYTES:
        raise ValueError('a bounded final-position float32 vector is required')
    offset = raw_stream.tell()
    raw_stream.write(payload)
    raw_stream.flush()
    os.fsync(raw_stream.fileno())
    raw_meta = {'offset': offset, 'bytes': len(payload), 'sha256': base.sha(payload)}
    base._append_event(journal, {'event': 'raw_recorded', 'case_id': case['case_id'], 'raw': raw_meta})
    score = screen.score_logits(np.frombuffer(payload, dtype='<f4'), lower=case['lower_ids'],
                                upper=case['upper_ids'], equal=case['equal_id'], numeric=screen.DIGIT_IDS)
    return {'case_id': case['case_id'], 'raw': raw_meta, 'score': score}


def check_environment():
    current = {name: {'present': name in os.environ, 'value': os.environ.get(name)}
               for name in CPU_ENV}
    if current != CPU_ENV or any(value['present'] for value in current.values()):
        raise RuntimeError('CPU backend environment changed')
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if (not sys.dont_write_bytecode or os.environ.get('HF_HUB_OFFLINE') != '1'
            or os.environ.get('TRANSFORMERS_OFFLINE') != '1'):
        raise RuntimeError('use -B and offline environment flags')
    source = source_identity()
    check_environment()
    case_raw = (ROOT / CASE_PATH).read_bytes()
    if base.sha(case_raw) != CASE_SHA256:
        raise RuntimeError('case inventory hash changed')
    cases = screen.validate_cases(json.loads(case_raw))
    dependencies = base.dependency_identity()
    if not args.cache_dir.is_absolute() or not args.output_dir.is_absolute():
        raise ValueError('cache and output paths must be absolute')
    parent = args.output_dir.parent.resolve(strict=True)
    if parent == ROOT or ROOT in parent.parents or args.output_dir.exists():
        raise ValueError('output must be new and outside the checkout')
    if shutil.disk_usage(parent).free < RAW_BYTES + 8_388_608:
        raise OSError('insufficient free space for the fixed output')
    base.numeric_policy(initialize=True)
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('highest')
    torch.set_default_dtype(torch.float32)
    args.output_dir.mkdir()
    manifest = {'schema': 'distilgpt2-greater-than-screen-run-v1', 'source': source,
                'cases_sha256': CASE_SHA256, 'cases': cases, 'dependencies': dependencies,
                'cpu_backend_environment_at_import': CPU_ENV, 'expected_forwards': 48,
                'raw_tensor_bytes': RAW_BYTES,
                'metric': 'upper-minus-lower equal-size log-mass ratio; forward minus reverse',
                'scope': 'fixed public exploratory year-comparison screen, not a blind holdout'}
    manifest_sha = base.durable_json(args.output_dir / 'manifest.json', manifest)
    submitted = returned = recorded = 0
    current, root_return_observed = None, False
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        screen.validate_tokenizer(bundle.tokenizer, cases)
        model = bundle.model
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        state_before, buffers_before = base.state_hash(model), base.buffer_hash(model)
        runtime_before = base.runtime(model)
        assert_no_forward_hooks(model)
        records = []
        with (args.output_dir / 'attempt.jsonl').open('xb') as journal, \
                (args.output_dir / 'last_logits.f32').open('xb') as raw_stream:
            base._append_event(journal, {'event': 'runtime_ready', 'actual_runtime': runtime_before})
            for case in cases:
                current, root_return_observed = case['case_id'], False
                if base.runtime(model) != runtime_before or check_environment() != CPU_ENV:
                    raise RuntimeError('runtime or random state changed before forward')
                assert_no_forward_hooks(model)
                ids = torch.tensor([case['input_ids']], dtype=torch.long, device='cpu')
                base._append_event(journal, {'event': 'start', 'case_id': current})
                submitted += 1
                with torch.inference_mode():
                    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                output_attentions=False, output_hidden_states=False, return_dict=True)
                root_return_observed = True
                returned += 1
                logits = out.logits
                if (type(logits) is not torch.Tensor or logits.device.type != 'cpu'
                        or logits.dtype != torch.float32 or tuple(logits.shape) != (1, len(case['input_ids']), VOCAB)):
                    raise RuntimeError('model returned an unexpected logit shape')
                payload = logits[0, -1].detach().contiguous().numpy().astype('<f4', copy=False).tobytes()
                record = persist_logits(raw_stream, journal, case, payload)
                if (out.past_key_values is not None or out.attentions is not None or out.hidden_states is not None
                        or ids.tolist() != [case['input_ids']] or base.runtime(model) != runtime_before
                        or check_environment() != CPU_ENV):
                    raise RuntimeError('model output or runtime changed')
                assert_no_forward_hooks(model)
                base._append_event(journal, {'event': 'complete', **record})
                recorded += 1
                records.append(record)
        if (submitted != 48 or returned != 48 or recorded != 48
                or (args.output_dir / 'last_logits.f32').stat().st_size != RAW_BYTES
                or base.state_hash(model) != state_before or base.buffer_hash(model) != buffers_before
                or base.runtime(model) != runtime_before or source_identity() != source or GUARD_EVENTS):
            raise RuntimeError('final counts, source, state or runtime differ')
        summary = screen.summarize(cases, {r['case_id']: r['score']['log_mass_ratio'] for r in records})
        result = {'schema': 'distilgpt2-greater-than-screen-result-v1', 'status': 'complete',
                  'manifest_sha256': manifest_sha, 'checkpoint': bundle.metadata,
                  'model_state_sha256_before': state_before, 'model_state_sha256_after': base.state_hash(model),
                  'model_buffers_sha256_before': buffers_before, 'model_buffers_sha256_after': base.buffer_hash(model),
                  'runtime': runtime_before, 'submitted_forwards': submitted,
                  'model_call_returns_observed': returned, 'recorded_forwards': recorded,
                  'records': records, 'summary': summary}
        base.durable_json(args.output_dir / 'result.json', result)
        print(json.dumps({'status': 'complete', 'summary': summary}, sort_keys=True))
    except BaseException as error:
        failure = {'schema': 'distilgpt2-greater-than-screen-failure-v1', 'status': 'incomplete',
                   'exception_type': type(error).__name__, 'case_id': current,
                   'submitted_forwards': submitted, 'model_call_returns_observed': returned,
                   'recorded_forwards': recorded, 'model_call_return_observed': root_return_observed,
                   'note': 'False return observation does not prove the model call did not return.'}
        base.durable_json(args.output_dir / 'failure.json', failure)
        raise


if __name__ == '__main__':
    main()
