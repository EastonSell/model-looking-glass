"""Fixed public CPU explicit-year-direction study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback

if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CUDA_VISIBLE_DEVICES must be -1 before interpreter startup')
    if 'torch' in sys.modules:
        raise RuntimeError('torch was imported before the CPU environment snapshot')
ENV_NAMES = ('MKL_CBWR', 'ATEN_CPU_CAPABILITY')
ENV_AT_IMPORT = {name: {'present': name in os.environ, 'value': os.environ.get(name)} for name in ENV_NAMES}
EXPECTED_ENV = {name: {'present': False, 'value': None} for name in ENV_NAMES}
if __name__ == '__main__' and ENV_AT_IMPORT != EXPECTED_ENV:
    raise RuntimeError('CPU backend overrides must be absent before importing torch')

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network

GUARD_EVENTS = []
SAFE_QUERIES = {'cuda_is_available': 0, 'cuda_device_count': 0}
if __name__ == '__main__':
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from scripts import copying_head_eval_v1 as base
from looking_glass import explicit_year_direction as study
from looking_glass import year_transfer_intervention as intervention
from looking_glass.copy_head_factorial import binary64_rounding_probe, f32_bits_to_f64
from looking_glass.copy_head_intervention import assert_no_forward_hooks

base.GUARD_EVENTS = GUARD_EVENTS
CASE_PATH = 'experiments/distilgpt2_explicit_year_direction_v1.json'
CASE_SHA256 = '419d74c23333407f9a9dc0b6f7d7b70764ab5d1078e2b0a4c84d37ba194df8a9'
PROTOCOL_PATH = 'docs/DISTILGPT2_EXPLICIT_YEAR_DIRECTION_V1.md'
RESERVATION_VERSION = 'distilgpt2-development-exposures-v21'
RESERVATION_SHA256 = 'a391c5f3a8ca33ad32e9378417b424338b12ea55257979977d2afd5a0c7f90ae'
RESERVATION_MODULE = 'looking_glass.pretrained_exposures_v21'
VOCAB = 50257
LOGITS_BYTES = VOCAB * 4
MAX_RAW_BYTES = 114731328
MAX_PER_FORWARD_BYTES = LOGITS_BYTES + 2 * 32 * 768 * 4
MAX_JOURNAL_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
SOURCES = tuple(dict.fromkeys((
    'scripts/explicit_year_direction_v1.py', 'looking_glass/explicit_year_direction.py',
    'looking_glass/year_transfer.py', 'looking_glass/year_transfer_intervention.py',
    'tests/test_explicit_year_direction.py', 'tests/test_explicit_year_direction_runner.py',
    CASE_PATH, PROTOCOL_PATH, 'docs/EXPLICIT_YEAR_DIRECTION_RUNNER_V1.md', 'looking_glass/greater_than_screen.py',
    'looking_glass/copy_head_factorial.py', 'looking_glass/copy_head_intervention.py',
    'scripts/copying_head_eval_v1.py', 'scripts/copying_dev_v1.py',
    'scripts/distilgpt2_intervention_dev_bootstrap.py', 'looking_glass/pretrained.py',
    'looking_glass/__init__.py', 'looking_glass/pretrained_exposures.py',
    *(f'looking_glass/pretrained_exposures_v{i}.py' for i in range(2, 22)),
)))


def environment():
    current = {name: {'present': name in os.environ, 'value': os.environ.get(name)} for name in ENV_NAMES}
    if ENV_AT_IMPORT != EXPECTED_ENV or current != ENV_AT_IMPORT:
        raise RuntimeError('CPU backend environment changed')
    return current


def runtime(model):
    binary64_rounding_probe()
    block = model.transformer.h[3]
    if type(block) is not GPT2Block or block.forward.__func__ is not GPT2Block.forward:
        raise RuntimeError('block implementation differs from the frozen model')
    return {**base.runtime(model), 'cpu_backend_environment': environment()}


def plan_family(noops):
    """Use the reviewed planner while preserving this study's wording names."""
    aliases = {'after': 'forward', 'before': 'reverse', 'near': 'neutral'}
    if type(noops) is not dict or set(noops) != set(aliases):
        raise ValueError('expected all three explicit wordings')
    planned = intervention.plan_family({aliases[name]: noops[name] for name in study.CONDITIONS})
    return {'dose': planned['dose'],
            'norms': {name: planned['norms'][aliases[name]] for name in study.CONDITIONS},
            'plans': {name: planned['plans'][aliases[name]] for name in study.CONDITIONS}}


def source_identity():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, timeout=30)
    revision = git('rev-parse', 'HEAD').decode('ascii').strip()
    tree = git('show', '-s', '--format=%T', revision).decode('ascii').strip()
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('source checkout must be clean')
    sources = {}
    for path in SOURCES:
        content = (ROOT / path).read_bytes()
        if git('show', f'{revision}:{path}') != content:
            raise RuntimeError(f'source differs from the committed bytes: {path}')
        sources[path] = base.sha(content)
    return {'revision': revision, 'tree': tree, 'sha256': sources}


def inventory():
    protocol = (ROOT / PROTOCOL_PATH).read_bytes()
    cases_raw = (ROOT / CASE_PATH).read_bytes()
    if base.sha(protocol) != study.PROTOCOL_SHA256 or base.sha(cases_raw) != CASE_SHA256:
        raise RuntimeError('frozen protocol or cases changed')
    return study.validate_inventory(json.loads(cases_raw))


def check_exposure(cases):
    if RESERVATION_VERSION is None or RESERVATION_SHA256 is None or RESERVATION_MODULE is None:
        raise RuntimeError('the reviewed public reservation is not yet bound')
    from importlib import import_module
    baseline = import_module('looking_glass.pretrained_exposures_v17')
    reservation = import_module(RESERVATION_MODULE)
    if (baseline.INVENTORY_VERSION != study.EXPOSURE_VERSION
            or baseline.inventory_digest() != study.EXPOSURE_SHA256
            or reservation.INVENTORY_VERSION != RESERVATION_VERSION
            or reservation.inventory_digest() != RESERVATION_SHA256
            or tuple(tuple(row['input_ids']) for row in cases) != reservation.EXPLICIT_YEAR_DIRECTION_ROWS):
        raise RuntimeError('public exposure inventory or planned rows differ')
    return {'baseline_version': study.EXPOSURE_VERSION, 'baseline_sha256': study.EXPOSURE_SHA256,
            'reservation_version': RESERVATION_VERSION, 'reservation_sha256': RESERVATION_SHA256}


def append_event(stream, event):
    data = base.canonical(event) + b'\n'
    if len(data) > 65536 or stream.tell() + len(data) > MAX_JOURNAL_BYTES:
        raise RuntimeError('journal metadata limit exceeded')
    if stream.write(data) != len(data):
        raise OSError('short journal write')
    stream.flush()
    os.fsync(stream.fileno())


def durable_json(path, obj):
    if len(base.canonical(obj)) > MAX_JSON_BYTES:
        raise RuntimeError('JSON metadata limit exceeded')
    return base.durable_json(path, obj)


def decoded(raw):
    if type(raw) is not bytes or len(raw) % 4:
        raise ValueError('complete little-endian float32 bytes are required')
    return tuple(f32_bits_to_f64(value[0]) for value in struct.iter_unpack('<I', raw))


def available_payload(result, witness, length):
    payload = {}
    try:
        logits = result.logits if result is not None else None
        if (type(logits) is torch.Tensor and logits.device.type == 'cpu' and logits.dtype == torch.float32
                and tuple(logits.shape) == (1, length, VOCAB)):
            payload['logits'] = base._raw(logits[0, -1], LOGITS_BYTES)
    except Exception:
        pass
    if witness is not None:
        for name, attribute in (('block_before', 'before'), ('block_after', 'after')):
            value = getattr(witness, attribute, None)
            if type(value) is bytes and len(value) == length*intervention.ROW_BYTES:
                payload[name] = value
    return payload


def run_forward(model, ids, action, *, expected_before=None, plan=None, observation=None):
    if observation is None:
        observation = {}
    observation.update({'model_call_return_observed': False, 'payload': {}})
    if action not in ('clean', 'noop', 'clean_after', *intervention.ACTIONS):
        raise ValueError('unknown action')
    assert_no_forward_hooks(model)
    length = ids.shape[1]
    witness = None
    result = None
    try:
        with torch.inference_mode():
            if action in ('clean', 'clean_after'):
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               output_attentions=False, output_hidden_states=False, return_dict=True)
                observation['model_call_return_observed'] = True
                observation['payload'] = available_payload(result, None, length)
            else:
                witness = intervention.BlockOutputWitness(model.transformer.h[3], sequence_length=length,
                                                           expected_before=expected_before, plan=plan)
                with witness:
                    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                   output_attentions=False, output_hidden_states=False, return_dict=True)
                    observation['model_call_return_observed'] = True
                    observation['payload'] = available_payload(result, witness, length)
    except BaseException:
        observation['payload'].update(available_payload(result, witness, length))
        raise
    assert_no_forward_hooks(model)
    expected = {'logits'} if witness is None else {'logits', 'block_before', 'block_after'}
    if (set(observation['payload']) != expected or sum(map(len, observation['payload'].values())) > MAX_PER_FORWARD_BYTES
            or result.past_key_values is not None or result.attentions is not None or result.hidden_states is not None):
        raise RuntimeError('returned model/witness payload is incomplete or oversized')
    return observation['payload'], witness


def persist_action(log, raw, case, action, payload, cache, observation, postcheck, marker):
    offsets = base._write_payload(raw, payload)
    append_event(log, {'event': 'raw_recorded', 'case_id': case['case_id'], 'action': action, 'raw': offsets})
    marker['recorded'] = True
    decoded_payload = {name: decoded(data) for name, data in payload.items()}
    postcheck()
    if action in ('noop', 'self', 'clean_after') and payload['logits'] != cache.get('clean'):
        raise RuntimeError('clean/no-op/self/after full-logit bits differ')
    witness = observation.get('witness')
    if witness is not None:
        witness.verify_complete()
        if witness.observed is not None:
            observation['intervention'] = {**observation['intervention'],
                                           'observed_achieved': witness.observed['achieved'],
                                           'observed_changed_bits': witness.observed['changed_bits']}
        if action == 'noop' and payload['block_before'] != payload['block_after']:
            raise RuntimeError('no-op block output changed')
    if action in intervention.ACTIONS and payload['block_before'] != cache.get('before'):
        raise RuntimeError('pre-action block output differs from the no-op')
    scores = study.score_logits(decoded_payload['logits'], case)
    record = {'case_id': case['case_id'], 'action': action, 'scores': scores, 'raw': offsets,
              'intervention': observation.get('intervention')}
    append_event(log, {'event': 'complete', **record})
    return record


def persist_partial(log, path, case_id, action, observation, marker):
    partial = observation.get('payload', {})
    if not partial:
        return 0
    with path.open('xb') as raw:
        offsets = base._write_payload(raw, partial)
    append_event(log, {'event': 'raw_recorded_partial', 'case_id': case_id, 'action': action,
                       'model_call_return_observed': observation.get('model_call_return_observed', False),
                       'file': path.name, 'raw': offsets})
    marker['partial'] = True
    return sum(map(len, partial.values()))


def check_position_zero(cases, caches):
    """Check all same-wording low/high no-op position-zero snapshots."""
    if not study._same_typed(cases, study.make_inventory()['cases']):
        raise ValueError('unexpected cases')
    checked = 0
    for index in range(0, len(cases), 2):
        low, high = cases[index:index+2]
        left, right = (caches[row['case_id']]['before'] for row in (low, high))
        if (type(left) is not bytes or type(right) is not bytes
                or len(left) != len(low['input_ids'])*intervention.ROW_BYTES
                or len(right) != len(high['input_ids'])*intervention.ROW_BYTES
                or left[:intervention.ROW_BYTES] != right[:intervention.ROW_BYTES]):
            raise ValueError('same-wording position-zero no-op snapshots differ')
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if (not sys.dont_write_bytecode or os.environ.get('HF_HUB_OFFLINE') != '1'
            or os.environ.get('TRANSFORMERS_OFFLINE') != '1'):
        raise RuntimeError('use -B and offline environment flags')
    cases = inventory()
    source, exposure = source_identity(), check_exposure(cases)
    cpu_environment, dependencies = environment(), base.dependency_identity()
    if not args.cache_dir.is_absolute() or not args.output_dir.is_absolute():
        raise ValueError('cache and output directories must be absolute')
    output = args.output_dir
    parent = output.parent.resolve(strict=True)
    if parent == ROOT or ROOT in parent.parents or output.exists():
        raise ValueError('output must be new and outside the checkout')
    if shutil.disk_usage(parent).free < MAX_RAW_BYTES + MAX_JOURNAL_BYTES + 4*MAX_JSON_BYTES:
        raise OSError('insufficient free disk space')
    initial_probe = base.numeric_policy(initialize=True)
    binary64_rounding_probe()
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('highest')
    torch.set_default_dtype(torch.float32)
    output.mkdir()
    manifest = {'schema': 'distilgpt2-explicit-year-direction-run-v1', 'source': source,
                'protocol_revision': study.PROTOCOL_REVISION, 'protocol_sha256': study.PROTOCOL_SHA256,
                'cases_sha256': CASE_SHA256, 'cases': cases, 'exposure_inventories': exposure,
                'dependencies': dependencies, 'cpu_backend_environment_at_torch_import': cpu_environment,
                'initial_numeric_probe': initial_probe, 'max_raw_bytes': MAX_RAW_BYTES,
                'max_retained_bytes_per_forward': MAX_PER_FORWARD_BYTES,
                'max_journal_bytes': MAX_JOURNAL_BYTES, 'max_json_bytes': MAX_JSON_BYTES,
                'planned_forwards': 336,
                'intended_runtime': {'device': 'cpu', 'dtype': 'float32', 'model_eval': True,
                                     'attention_implementation': 'eager', 'use_cache': False,
                                     'intraop_threads': 1, 'deterministic_algorithms': False,
                                     'float32_matmul_precision': 'highest', 'cpu_rng_seed': base.CPU_RNG_SEED},
                'counter_semantics': {'submitted': 'forward helper entries, possibly before a root call',
                                      'model_returns_observed': 'root returns observed by the helper; absence is not proof',
                                      'returned': 'complete helper payloads',
                                      'partial': 'available bounded bytes saved after helper failure'},
                'scope': 'fixed public non-blind exploratory; does not prove a numerical circuit'}
    manifest_hash = durable_json(output/'manifest.json', manifest)
    completed, family_doses, caches = [], [], {row['case_id']: {} for row in cases}
    submitted = model_returns = returned = retained = 0
    phase, case_id, action = 'model_load', None, None
    root_return_observed, payload_returned = False, False
    marker = {'recorded': False, 'partial': False, 'partial_save_error': None}
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        if bundle.metadata['weights_sha256'] != base.WEIGHTS_SHA256:
            raise RuntimeError('checkpoint weight hash changed')
        study.validate_tokenizer(bundle.tokenizer, cases)
        model = bundle.model
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        model_before, buffers_before = base.state_hash(model), base.buffer_hash(model)
        runtime_before = runtime(model)
        assert_no_forward_hooks(model)
        with (output/'attempt.jsonl').open('xb') as log:
            append_event(log, {'event': 'runtime_ready', 'actual_runtime': runtime_before})

            def perform(case, kind, plan=None):
                nonlocal submitted, model_returns, returned, retained
                nonlocal phase, case_id, action, root_return_observed, payload_returned, marker
                cache = caches[case['case_id']]
                case_id, action, phase = case['case_id'], kind, 'forward'
                root_return_observed, payload_returned = False, False
                marker = {'recorded': False, 'partial': False, 'partial_save_error': None}
                ids = torch.tensor([case['input_ids']], dtype=torch.long, device='cpu')
                if runtime(model) != runtime_before:
                    raise RuntimeError('runtime changed before a forward')
                details = None
                if plan is not None:
                    details = {
                        **{name: plan[name] for name in ('position', 'requested', 'achieved', 'changed_bits')},
                        'expected_before_hex': plan['before'].hex(), 'expected_after_hex': plan['after'].hex(),
                    }
                append_event(log, {'event': 'start', 'case_id': case_id, 'action': action,
                                   'intervention': details})
                submitted += 1
                observation = {}
                payload = None
                model_returns_before, returned_before = model_returns, returned
                try:
                    payload, witness = run_forward(model, ids, kind,
                                                   expected_before=cache.get('before') if kind in intervention.ACTIONS else None,
                                                   plan=plan, observation=observation)
                    root_return_observed = observation['model_call_return_observed']
                    model_returns = model_returns_before + int(root_return_observed)
                    returned = returned_before + 1
                    payload_returned = True
                    retained += sum(map(len, payload.values()))
                    if retained > MAX_RAW_BYTES:
                        raise RuntimeError('aggregate raw payload exceeds its limit')
                    phase = 'persist_and_validate'
                    observation['witness'] = witness
                    observation['intervention'] = details

                    def postcheck():
                        if ids.tolist() != [case['input_ids']] or runtime(model) != runtime_before:
                            raise RuntimeError('input, runtime or CPU RNG changed')
                        assert_no_forward_hooks(model)

                    with (output/(case_id+'.f32')).open('xb' if kind == 'clean' else 'ab') as raw:
                        record = persist_action(log, raw, case, kind, payload, cache, observation, postcheck, marker)
                    completed.append(record)
                    if kind == 'clean':
                        cache['clean'] = payload['logits']
                        cache['clean_scores'] = record['scores']
                    elif kind == 'noop':
                        cache['before'] = payload['block_before']
                    elif kind in intervention.ACTIONS:
                        cache.setdefault('action_scores', {})[kind] = record['scores']['band_score']
                except BaseException:
                    root_return_observed = observation.get('model_call_return_observed', False)
                    model_returns = model_returns_before + int(root_return_observed)
                    payload_returned = payload is not None
                    returned = returned_before + int(payload_returned)
                    partial = observation.get('payload', {})
                    if partial and not marker['recorded']:
                        try:
                            physical = sum(path.stat().st_size for path in output.glob('*.f32'))
                            if physical + sum(map(len, partial.values())) > MAX_RAW_BYTES:
                                raise RuntimeError('partial evidence exceeds the raw limit')
                            persist_partial(log, output/(case_id+'.'+action+'.partial.f32'), case_id, action,
                                            observation, marker)
                        except Exception as save_error:
                            marker['partial_save_error'] = type(save_error).__name__
                    raise

            for case in cases:
                perform(case, 'clean')
                perform(case, 'noop')
            phase, case_id, action = 'position_zero_check', None, None
            root_return_observed, payload_returned = False, False
            marker = {'recorded': False, 'partial': False, 'partial_save_error': None}
            append_event(log, {'event': 'position_zero_check', 'pairs_checked': check_position_zero(cases, caches)})
            phase = 'clean_gates'
            clean_scores = {case['case_id']: caches[case['case_id']]['clean_scores'] for case in cases}
            clean_summary = study.summarize_clean(cases, clean_scores)
            append_event(log, {'event': 'clean_gates', 'summary': clean_summary})
            proceed = True
            if proceed:
                for index in range(8):
                    group = cases[index*6:(index+1)*6]
                    noops = {condition: {role: caches[next(row['case_id'] for row in group
                                  if row['condition'] == condition and row['year_role'] == role)]['before']
                                         for role in study.ROLES} for condition in study.CONDITIONS}
                    phase, case_id, action = 'family_dose', None, None
                    root_return_observed, payload_returned = False, False
                    marker = {'recorded': False, 'partial': False, 'partial_save_error': None}
                    plan = plan_family(noops)
                    dose_info = {'pair_noun_index': index, 'dose': plan['dose'], 'norms': plan['norms'],
                                 'fractions': {key: plan['dose']/value for key, value in plan['norms'].items()}}
                    family_doses.append(dose_info)
                    append_event(log, {'event': 'family_dose', **dose_info})
                    for case in group:
                        plans = plan['plans'][case['condition']][case['year_role']]
                        shift = case['rotation']
                        actions = intervention.ACTIONS[shift:] + intervention.ACTIONS[:shift]
                        for kind in actions:
                            perform(case, kind, plans[kind])
                        perform(case, 'clean_after')
        phase = 'verification'
        expected_count = 336
        expected_raw = (expected_count*LOGITS_BYTES +
                        10 * sum(len(row['input_ids'])*intervention.ROW_BYTES for row in cases))
        if (submitted != expected_count or model_returns != expected_count or returned != expected_count
                or len(completed) != expected_count or retained != expected_raw
                or base.state_hash(model) != model_before or base.buffer_hash(model) != buffers_before
                or runtime(model) != runtime_before or source_identity() != source or GUARD_EVENTS
                or torch.cuda.is_initialized()):
            raise RuntimeError('final count, model, runtime or source differs')
        assert_no_forward_hooks(model)
        effects = (study.summarize_interventions(cases, clean_scores,
                   {case['case_id']: caches[case['case_id']]['action_scores'] for case in cases}) if proceed else None)
        result = {'schema': 'distilgpt2-explicit-year-direction-result-v1', 'status': 'complete',
                  'manifest_sha256': manifest_hash, 'checkpoint': bundle.metadata,
                  'model_state_sha256_before': model_before, 'model_state_sha256_after': base.state_hash(model),
                  'model_buffers_sha256_before': buffers_before, 'model_buffers_sha256_after': base.buffer_hash(model),
                  'runtime': runtime_before, 'guard_safe_queries': SAFE_QUERIES,
                  'submitted_forwards': submitted, 'model_call_returns_observed': model_returns,
                  'returned_forwards': returned, 'raw_payload_bytes': retained,
                  'clean_summary': clean_summary, 'interventions_performed': proceed,
                  'family_doses': family_doses, 'records': completed, 'intervention_summary': effects}
        durable_json(output/'result.json', result)
    except BaseException as error:
        attempt = output/'attempt.jsonl'
        if attempt.is_file():
            try:
                with attempt.open('ab') as log:
                    append_event(log, {'event': 'failed', 'phase': phase, 'case_id': case_id, 'action': action,
                                       'error_type': type(error).__name__, 'model_call_return_observed': root_return_observed,
                                       'payload_returned': payload_returned, 'payload_recorded': marker['recorded'],
                                       'partial_payload_recorded': marker['partial'],
                                       'partial_save_error': marker['partial_save_error']})
            except Exception:
                pass
        durable_json(output/'failure.json', {'schema': 'distilgpt2-explicit-year-direction-failure-v1', 'status': 'incomplete',
                     'manifest_sha256': manifest_hash, 'phase': phase, 'case_id': case_id, 'action': action,
                     'submitted_forwards': submitted, 'model_call_returns_observed': model_returns,
                     'returned_forwards': returned, 'completed_records': len(completed), 'admitted_payload_bytes': retained,
                     'model_call_return_observed': root_return_observed, 'payload_returned': payload_returned,
                     'payload_recorded': marker['recorded'], 'partial_payload_recorded': marker['partial'],
                     'partial_save_error': marker['partial_save_error'],
                     'raw_files': base.raw_file_inventory(output, cases),
                     'partial_files': [{'file': path.name, 'bytes': path.stat().st_size,
                                        'sha256': base.sha(path.read_bytes())}
                                       for path in sorted(output.glob('*.partial.f32'))],
                     'error_type': type(error).__name__, 'traceback_frames': [
                         {'file': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
                         for frame in traceback.extract_tb(error.__traceback__)]})
        raise
    print(json.dumps({'status': 'complete', 'forwards': submitted, 'clean_summary': clean_summary,
                      'interventions_performed': proceed}, allow_nan=False))


if __name__ == '__main__':
    main()
