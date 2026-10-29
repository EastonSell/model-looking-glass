"""Fixed public CPU continuation on three previously exposed families."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback


def checked_empty_pycache_prefix():
    value = sys.pycache_prefix
    if type(value) is not str or not value:
        raise RuntimeError('an absolute empty pycache prefix is required')
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RuntimeError('pycache prefix must be an existing absolute non-symlink directory')
    if any(path.iterdir()):
        raise RuntimeError('pycache prefix must be empty')
    return str(path.resolve(strict=True))


PYCACHE_AT_IMPORT = None
if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CUDA_VISIBLE_DEVICES must be -1 before interpreter startup')
    if 'torch' in sys.modules:
        raise RuntimeError('torch was imported before the CPU environment snapshot')
    PYCACHE_AT_IMPORT = checked_empty_pycache_prefix()
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
from looking_glass import year_label_control_v2 as study
from looking_glass import year_label_intervention_v2 as intervention
from looking_glass.copy_head_factorial import binary64_rounding_probe, f32_bits_to_f64
from looking_glass.copy_head_intervention import assert_no_forward_hooks

base.GUARD_EVENTS = GUARD_EVENTS
CASE_PATH = 'experiments/distilgpt2_year_label_control_v1.json'
CASE_SHA256 = '07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'
PROTOCOL_PATH = 'docs/DISTILGPT2_YEAR_LABEL_CONTROL_V2.md'
RESERVATION_VERSION = 'distilgpt2-development-exposures-v26'
RESERVATION_SHA256 = '8fd80cb0b24225100628489dcd0fe2b7efd11e87dfcbb624cf039e54a8a3a62d'
RESERVATION_MODULE = 'looking_glass.pretrained_exposures_v26'
PRIOR_RECEIPT_SHA256 = 'f3b418c7760ed23e2f3c3f90bd39b33538b2b9913ffad2dcf0dd7546742f55bf'
PRIOR_MANIFEST_SHA256 = 'd43d181e1c7046c8e97e6310bd3edd38fe85089e0e69347cd2ddb2728d29d47a'
PRIOR_JOURNAL_SHA256 = 'a2f5ba4f980bde155bf4eb5729f154f7422956eb9c0920421f13327038e057f5'
PRIOR_FAILURE_SHA256 = 'b59687e3f63573ca4ecbe0b34ba98ead6cdcdf3e035e4e54247abd237ca6e6ef'
VOCAB = 50257
LOGITS_BYTES = VOCAB * 4
MAX_RAW_BYTES = 100363392
MAX_PER_FORWARD_BYTES = LOGITS_BYTES + 2 * 32 * 768 * 4
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
SOURCES = tuple(dict.fromkeys((
    'scripts/year_label_control_v2.py', 'looking_glass/year_label_control_v2.py',
    'looking_glass/year_label_intervention_v2.py', 'looking_glass/year_label_control.py',
    'looking_glass/year_label_intervention.py', 'looking_glass/explicit_year_direction.py',
    'looking_glass/year_transfer.py', 'looking_glass/year_transfer_intervention.py',
    'tests/test_year_label_control_v2.py', 'tests/test_year_label_intervention_v2.py', 'tests/test_year_label_runner_v2.py',
    CASE_PATH, PROTOCOL_PATH, 'docs/DISTILGPT2_YEAR_LABEL_CONTROL_V1.md',
    'docs/YEAR_LABEL_CONTROL_RUNNER_V2.md', 'looking_glass/greater_than_screen.py',
    'looking_glass/copy_head_factorial.py', 'looking_glass/copy_head_intervention.py',
    'scripts/year_label_receipt_v2.py', 'tests/test_year_label_receipt_v2.py',
    'scripts/copying_head_eval_v1.py', 'scripts/copying_dev_v1.py',
    'scripts/distilgpt2_intervention_dev_bootstrap.py', 'looking_glass/pretrained.py',
    'looking_glass/__init__.py', 'looking_glass/pretrained_exposures.py',
    *(f'looking_glass/pretrained_exposures_v{i}.py' for i in range(2, 27)),
)))


def environment():
    current_prefix = checked_empty_pycache_prefix()
    if PYCACHE_AT_IMPORT is not None and current_prefix != PYCACHE_AT_IMPORT:
        raise RuntimeError('pycache prefix changed after startup')
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


def plan_family(noops, labels):
    return intervention.plan_family(noops, labels)


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
    baseline = import_module('looking_glass.pretrained_exposures_v24')
    reservation = import_module(RESERVATION_MODULE)
    payload, digest = reservation._checked_inventory()
    if (baseline.INVENTORY_VERSION != study.BASELINE_VERSION
            or baseline.INVENTORY_SHA256 != study.BASELINE_SHA256
            or reservation.INVENTORY_VERSION != RESERVATION_VERSION
            or digest != RESERVATION_SHA256
            or payload['predecessor'] != {'version': 'distilgpt2-development-exposures-v25',
                                          'sha256': '32cc4bcbe4b482ded122715c5f96c6c5c765a76a1ef3532360cdf9890827f144'}
            or tuple(tuple(row['input_ids']) for row in cases) != reservation.YEAR_LABEL_CONTROL_ROWS[12:]):
        raise RuntimeError('public exposure inventory or planned rows differ')
    records = payload['records']
    for case in cases:
        own = [record for record in records if record.get('record_id') == 'planned-' + case['case_id']]
        if (len(own) != 1 or own[0].get('category') != 'planned'
                or own[0].get('token_ids') != case['input_ids']):
            raise RuntimeError('the historical v26 own reservation must be unique and exact')
    return {'baseline_version': study.BASELINE_VERSION, 'baseline_sha256': study.BASELINE_SHA256,
            'reservation_version': RESERVATION_VERSION, 'reservation_sha256': RESERVATION_SHA256,
            'own_v26_status': 'planned in the historical snapshot; observed in retained v1 evidence',
            'prior_receipt_sha256': PRIOR_RECEIPT_SHA256}


def _prior_segment(path, slot, expected_bytes):
    if (type(slot) is not dict or set(slot) != {'offset', 'bytes', 'sha256'}
            or type(slot['offset']) is not int or slot['offset'] < 0
            or type(slot['bytes']) is not int or slot['bytes'] != expected_bytes
            or type(slot['sha256']) is not str or len(slot['sha256']) != 64):
        raise ValueError('invalid prior raw segment')
    with path.open('rb') as source:
        source.seek(slot['offset'])
        data = source.read(expected_bytes)
    if len(data) != expected_bytes or base.sha(data) != slot['sha256']:
        raise ValueError('prior raw segment differs')
    return data


def load_prior(attempt, cases):
    """Bind v1's incomplete evidence and extract only the two comparison vectors."""
    if not attempt.is_absolute() or not attempt.is_dir():
        raise ValueError('prior attempt must be an existing absolute directory')
    receipt_raw = (attempt/'supervisor.json').read_bytes()
    if base.sha(receipt_raw) != PRIOR_RECEIPT_SHA256:
        raise ValueError('prior direct-process receipt differs')
    receipt = json.loads(receipt_raw)
    if (receipt.get('schema') != 'distilgpt2-year-label-control-process-observation-v1'
            or receipt.get('process_exit_code') != 1 or receipt.get('interrupted') is not False
            or receipt.get('failure_file_present') is not True
            or receipt.get('source_revision') != '72ff52feca7c102e611023b54f031fbb8ff17bd4'):
        raise ValueError('prior process outcome differs')
    run = attempt/'run'
    files = receipt.get('run_sha256')
    full = study.v1.make_inventory()['cases']
    expected_files = {'manifest.json', 'attempt.jsonl', 'failure.json'} | {r['case_id']+'.f32' for r in full}
    if (type(files) is not dict or set(files) != expected_files
            or {p.name for p in run.iterdir() if p.is_file()} != expected_files
            or files['manifest.json'] != PRIOR_MANIFEST_SHA256
            or files['attempt.jsonl'] != PRIOR_JOURNAL_SHA256
            or files['failure.json'] != PRIOR_FAILURE_SHA256):
        raise ValueError('prior file inventory differs')
    for name in sorted(expected_files):
        if base.sha((run/name).read_bytes()) != files[name]:
            raise ValueError('prior run file differs')
    manifest = json.loads((run/'manifest.json').read_bytes())
    failure = json.loads((run/'failure.json').read_bytes())
    if (not study.v1._same_typed(manifest.get('cases'), full)
            or failure.get('status') != 'incomplete' or failure.get('phase') != 'family_dose'
            or failure.get('submitted_forwards') != 168 or failure.get('completed_records') != 168):
        raise ValueError('prior incomplete attempt differs')
    events = [json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
    records = {}
    for event in events:
        if event.get('event') == 'complete':
            key = (event.get('case_id'), event.get('action'))
            if key in records:
                raise ValueError('duplicate prior complete event')
            records[key] = event
    if len(records) != 168:
        raise ValueError('prior completion count differs')
    saved = {}
    for row in cases:
        case_id = row['case_id']
        clean, noop = records.get((case_id, 'clean')), records.get((case_id, 'noop'))
        if type(clean) is not dict or type(noop) is not dict:
            raise ValueError('prior clean/no-op record is missing')
        path = run/(case_id+'.f32')
        saved[case_id] = {
            'clean': _prior_segment(path, clean['raw']['logits'], LOGITS_BYTES),
            'before': _prior_segment(path, noop['raw']['block_before'], len(row['input_ids'])*intervention.ROW_BYTES),
        }
    return saved


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
                                           'observed_changed_bits': witness.observed['changed_bits'],
                                           'observed_certificate': witness.observed['certificate']}
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
    """Check all four same-wording no-op position-zero snapshots."""
    if not study.v1._same_typed(cases, study.select_cases(study.v1.make_inventory()['cases'])):
        raise ValueError('unexpected cases')
    checked = 0
    for index in range(0, len(cases), 4):
        group = cases[index:index+4]
        snapshots = [caches[row['case_id']]['before'] for row in group]
        if (any(type(raw) is not bytes or len(raw) != len(row['input_ids'])*intervention.ROW_BYTES
                for row, raw in zip(group, snapshots))
                or any(raw[:intervention.ROW_BYTES] != snapshots[0][:intervention.ROW_BYTES]
                       for raw in snapshots[1:])):
            raise ValueError('same-wording position-zero no-op snapshots differ')
        checked += 3
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prior-attempt-dir', required=True, type=Path)
    args = parser.parse_args()
    if (not sys.dont_write_bytecode or os.environ.get('HF_HUB_OFFLINE') != '1'
            or os.environ.get('TRANSFORMERS_OFFLINE') != '1'):
        raise RuntimeError('use -B and offline environment flags')
    cases = inventory()
    prior = load_prior(args.prior_attempt_dir, cases)
    source, exposure = source_identity(), check_exposure(cases)
    cpu_environment, dependencies = environment(), base.dependency_identity()
    bytecode_prefix = checked_empty_pycache_prefix()
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
    manifest = {'schema': 'distilgpt2-year-label-control-run-v2', 'source': source,
                'protocol_revision': study.PROTOCOL_REVISION, 'protocol_sha256': study.PROTOCOL_SHA256,
                'cases_sha256': CASE_SHA256, 'cases': cases, 'exposure_inventories': exposure,
                'dependencies': dependencies, 'cpu_backend_environment_at_torch_import': cpu_environment,
                'empty_pycache_prefix_sha256': hashlib.sha256(os.path.normcase(bytecode_prefix).encode('utf-8')).hexdigest(),
                'initial_numeric_probe': initial_probe, 'max_raw_bytes': MAX_RAW_BYTES,
                'max_retained_bytes_per_forward': MAX_PER_FORWARD_BYTES,
                'max_journal_bytes': MAX_JOURNAL_BYTES, 'max_json_bytes': MAX_JSON_BYTES,
                'planned_forwards': 288,
                'prior_v1_receipt_sha256': PRIOR_RECEIPT_SHA256,
                'prior_v1_manifest_sha256': PRIOR_MANIFEST_SHA256,
                'prior_v1_journal_sha256': PRIOR_JOURNAL_SHA256,
                'prior_v1_failure_sha256': PRIOR_FAILURE_SHA256,
                'fidelity_rule': {'relative_vector_and_dose_cap': '1/5000',
                                  'relative_ideal_norm_cap': '1/1000000000000',
                                  'comparison': 'exact rational squares'},
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
                        'displacement_f64_hex': (None if plan['displacement'] is None else
                                                 struct.pack(f'<{intervention.WIDTH}d', *plan['displacement']).hex()),
                        'certificate': plan['certificate'],
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
            cross_run = [{'case_id': row['case_id'],
                          'clean_logits_match_v1': caches[row['case_id']]['clean'] == prior[row['case_id']]['clean'],
                          'noop_block_output_matches_v1': caches[row['case_id']]['before'] == prior[row['case_id']]['before']}
                         for row in cases]
            append_event(log, {'event': 'cross_run_baseline_comparison', 'cases': cross_run,
                               'prior_receipt_sha256': PRIOR_RECEIPT_SHA256})
            phase = 'clean_summary'
            clean_scores = {case['case_id']: caches[case['case_id']]['clean_scores'] for case in cases}
            clean_summary = study.summarize_clean(cases, clean_scores)
            append_event(log, {'event': 'clean_summary', 'summary': clean_summary})
            proceed = True
            if proceed:
                for index in (1, 2, 3):
                    offset = index - 1
                    group = cases[offset*12:(offset+1)*12]
                    labels = study.LABEL_PAIRS[index % 2]
                    noops = {condition: {label: {role: caches[next(row['case_id'] for row in group
                                  if row['condition'] == condition and row['label'] == label
                                  and row['year_role'] == role)]['before'] for role in study.ROLES}
                                         for label in labels} for condition in study.CONDITIONS}
                    phase, case_id, action = 'family_dose', None, None
                    root_return_observed, payload_returned = False, False
                    marker = {'recorded': False, 'partial': False, 'partial_save_error': None}
                    plan = plan_family(noops, labels)
                    dose_info = {'family_index': index, 'dose': plan['dose'], 'norms': plan['norms'],
                                 'fractions': {key: plan['dose']/value for key, value in plan['norms'].items()}}
                    family_doses.append(dose_info)
                    append_event(log, {'event': 'family_dose', **dose_info})
                    for case in group:
                        plans = plan['plans'][case['condition']][case['label']][case['year_role']]
                        shift = case['rotation']
                        actions = intervention.ACTIONS[shift:] + intervention.ACTIONS[:shift]
                        for kind in actions:
                            perform(case, kind, plans[kind])
                        perform(case, 'clean_after')
        phase = 'verification'
        expected_count = 288
        expected_raw = (expected_count*LOGITS_BYTES +
                        12 * sum(len(row['input_ids'])*intervention.ROW_BYTES for row in cases))
        if (submitted != expected_count or model_returns != expected_count or returned != expected_count
                or len(completed) != expected_count or retained != expected_raw
                or base.state_hash(model) != model_before or base.buffer_hash(model) != buffers_before
                or runtime(model) != runtime_before or source_identity() != source or GUARD_EVENTS
                or torch.cuda.is_initialized()):
            raise RuntimeError('final count, model, runtime or source differs')
        assert_no_forward_hooks(model)
        effects = (study.summarize_interventions(cases, clean_scores,
                   {case['case_id']: caches[case['case_id']]['action_scores'] for case in cases}) if proceed else None)
        result = {'schema': 'distilgpt2-year-label-control-result-v2', 'status': 'complete',
                  'manifest_sha256': manifest_hash, 'checkpoint': bundle.metadata,
                  'model_state_sha256_before': model_before, 'model_state_sha256_after': base.state_hash(model),
                  'model_buffers_sha256_before': buffers_before, 'model_buffers_sha256_after': base.buffer_hash(model),
                  'runtime': runtime_before, 'guard_safe_queries': SAFE_QUERIES,
                  'submitted_forwards': submitted, 'model_call_returns_observed': model_returns,
                  'returned_forwards': returned, 'raw_payload_bytes': retained,
                  'clean_summary': clean_summary, 'interventions_performed': proceed,
                  'family_doses': family_doses, 'records': completed, 'intervention_summary': effects,
                  'cross_run_baseline_comparison': cross_run}
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
        durable_json(output/'failure.json', {'schema': 'distilgpt2-year-label-control-failure-v2', 'status': 'incomplete',
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
