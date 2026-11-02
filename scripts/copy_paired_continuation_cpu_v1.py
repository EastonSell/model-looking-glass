"""Run the fixed paired-continuation screen and matched-donor patches on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import traceback

if __name__ == '__main__' and (
        not sys.dont_write_bytecode or not sys.pycache_prefix
        or not Path(sys.pycache_prefix).is_absolute() or not Path(sys.pycache_prefix).is_dir()
        or any(Path(sys.pycache_prefix).iterdir()) or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1'
        or os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1'
        or any(k in os.environ for k in ('MKL_CBWR', 'ATEN_CPU_CAPABILITY'))
        or not os.environ.get('COPY_PAIRED_BOOTSTRAP_SHA256') or 'torch' in sys.modules):
    raise RuntimeError('use the fixed CPU/offline bootstrap before project imports')

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network
GUARD_EVENTS = []
SAFE_QUERIES = {'cuda_is_available': 0, 'cuda_device_count': 0}
if __name__ == '__main__':
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)
from scripts import copying_head_output_dose_v1 as old
from scripts.copy_paired_continuation_bootstrap_v1 import verify_manifest
from looking_glass import copy_paired_continuation_v1 as m
from looking_glass.copy_head_intervention import HeadProjectionWitness, assert_no_forward_hooks
from looking_glass.copy_head_factorial import binary64_rounding_probe
from looking_glass.copy_head_output_dose import WEIGHT_BYTES, weights

base, prior = old.base, old.prior
base.GUARD_EVENTS = GUARD_EVENTS
CASES = 'experiments/distilgpt2_paired_continuation_cases_v1.json'
PROTOCOL = 'docs/DISTILGPT2_PAIRED_CONTINUATION_PATCH_V1.md'
AMENDMENT = 'docs/DISTILGPT2_PAIRED_CONTINUATION_POOL_AMENDMENT_V1.md'
CASE_SHA = '68b567718209285a9adedd588d0c0d269efb4b7d8ed9d278a2fb8201d672d1dd'
PROTOCOL_SHA = '61af860d38fe236bcac6a9d043cafd1ffe3dccc2cfa97fd2d0cc795c523f0924'
AMENDMENT_SHA = '7fa718ec0e375c460136812d3f06a6bebdfe065d182abeb6d4c0cdb009509154'
INVENTORY_SHA = '8c0e09c2a1bf276e931f3cd7580cbb5076ad696b3371002307cf8cc76e4e4b37'
RAW_LIMIT = 125_449_984
WEIGHT_LIMIT = 589_824
STAGING_BYTES = 355_679_200
META_RESERVE = 32 * 1024**2
FAILURE_RESERVE = 1024**2
WORK_ROOT = Path('E:/Codex/model-looking-glass/work')
ORDER = (2, 10, 7)
old.MAX_ACTION_BYTES = RAW_LIMIT


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_reservation(path, cases):
    raw = path.read_bytes()
    if sha(raw) != INVENTORY_SHA:
        raise ValueError('reservation canonical bytes differ')
    payload = json.loads(raw)
    records = payload['records']
    if (payload['version'] != 'distilgpt2-development-exposures-v45'
            or len(records) != 1557 or len({r['record_id'] for r in records}) != len(records)
            or payload['update'] != {'planned_paired_continuation_rows': 64}):
        raise ValueError('reservation structure differs')
    for c, r in zip(cases, records[-64:]):
        if (r['record_id'] != 'planned-' + c['case_id'] or r['category'] != 'planned'
                or r['token_ids'] != c['token_ids'] or r['candidate_token_ids'] != [c['candidate_f_id'], c['candidate_g_id']]
                or r['public_text'] != c['text']):
            raise ValueError('reserved prompt or scoring candidates differ')
    return {'sha256': INVENTORY_SHA, 'record_count': len(records), 'added': 64}


def validate_tokenization(tokenizer, cases):
    for c in cases:
        if tokenizer(c['text'], add_special_tokens=False)['input_ids'] != c['token_ids']:
            raise ValueError('tokenized fixed input differs')
        for token in (c['candidate_f_id'], c['candidate_g_id']):
            suffix = tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if tokenizer(c['text'] + suffix, add_special_tokens=False)['input_ids'] != c['token_ids'] + [token]:
                raise ValueError('scored continuation boundary differs')


def available_payload(result, witness):
    payload = {}
    try:
        logits = result.logits if result is not None else None
        if (type(logits) is torch.Tensor and logits.device.type == 'cpu' and logits.dtype == torch.float32
                and tuple(logits.shape) == (1, 12, m.VOCAB)):
            payload['logits'] = base._raw(logits[0, -1], m.ROW_BYTES)
    except Exception:
        pass
    for key, attr in (('cproj_before', 'before'), ('cproj_after', 'after'), ('cproj_output', 'output')):
        try:
            tensor = getattr(witness, attr, None)
            if (type(tensor) is torch.Tensor and tensor.device.type == 'cpu' and tensor.dtype == torch.float32
                    and tuple(tensor.shape) == (1, 12, 768)):
                payload[key] = base._raw(tensor, m.WITNESS_BYTES)
        except Exception:
            pass
    return payload


def run_forward(model, ids, kind, head, delivery, observation):
    observation.update({'model_call_return_observed': False, 'payload': {}})
    assert_no_forward_hooks(model)
    if kind not in ('clean', 'clean_after', 'noop', 'self', 'patch'):
        raise ValueError('unknown action')
    projection = model.transformer.h[3].attn.c_proj
    witness = None
    try:
        with torch.inference_mode():
            if kind in ('clean', 'clean_after'):
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               output_attentions=False, output_hidden_states=False, return_dict=True)
                observation['model_call_return_observed'] = True
                observation['payload'] = available_payload(result, None)
            else:
                if kind == 'noop':
                    witness = HeadProjectionWitness(projection, sequence_length=12, width=768, heads=12, mode='noop')
                else:
                    if head not in m.HEADS or type(delivery) is not bytes or len(delivery) != 256:
                        raise ValueError('exact head delivery required')
                    replacement = torch.frombuffer(bytearray(delivery), dtype=torch.int32).clone().view(torch.float32)
                    witness = HeadProjectionWitness(projection, sequence_length=12, width=768, heads=12,
                                                    mode='patch', head=head, position=11, replacement=replacement)
                with witness:
                    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                   output_attentions=False, output_hidden_states=False, return_dict=True)
                    observation['model_call_return_observed'] = True
                    observation['payload'] = available_payload(result, witness)
    except BaseException:
        observation['payload'].update(available_payload(None, witness))
        raise
    assert_no_forward_hooks(model)
    if result.past_key_values is not None or result.attentions is not None or result.hidden_states is not None:
        raise RuntimeError('unexpected model outputs')
    payload = observation['payload']
    expected = {'logits'} if witness is None else {'logits', 'cproj_before', 'cproj_after', 'cproj_output'}
    if set(payload) != expected:
        raise RuntimeError('bounded model or witness payload incomplete')
    return payload


def validate_payload(payload, kind, cache, head, delivery):
    for name, data in payload.items():
        if any((item[0] & 0x7f800000) == 0x7f800000 for item in struct.iter_unpack('<I', data)):
            raise RuntimeError('nonfinite returned ' + name)
    if kind in ('noop', 'self', 'clean_after') and payload['logits'] != cache['clean']:
        raise RuntimeError('control logits differ from clean')
    if kind == 'noop' and payload['cproj_before'] != payload['cproj_after']:
        raise RuntimeError('no-op changed projection input')
    if kind in ('self', 'patch'):
        if payload['cproj_before'] != cache['before']:
            raise RuntimeError('edited pre-input differs from no-op')
        expected = bytearray(cache['before'])
        start = (11 * 768 + 64 * head) * 4
        expected[start:start+256] = delivery
        if payload['cproj_after'] != bytes(expected):
            raise RuntimeError('head landing or untouched input differs')
        if payload['cproj_output'][:11*768*4] != cache['noop_output'][:11*768*4]:
            raise RuntimeError('untouched projection output differs')
        if kind == 'self' and (payload['cproj_after'] != cache['before']
                               or payload['cproj_output'] != cache['noop_output']):
            raise RuntimeError('self patch differs from no-op')


def perform_action(case, kind, cache, head, delivery, *, model, runtime_before,
                   output, log, counts, activity, completed):
    case_id = case['case_id']; observation = {}
    marker = {'recorded': False, 'partial': False, 'partial_save_error': None}
    activity.update({'phase': 'forward', 'case_id': case_id, 'action': kind,
                     'observation': observation, 'marker': marker})
    ids = torch.tensor([case['token_ids']], dtype=torch.int64, device='cpu')
    action = 'patch_h' + str(head) if kind == 'patch' else kind
    base._append_event(log, {'event': 'start', 'case_id': case_id, 'action': action})
    if prior.runtime(model) != runtime_before:
        raise RuntimeError('runtime changed before forward')
    path = output / (case_id + '.f32')
    start_offset = path.stat().st_size if path.exists() else 0
    before_returns, before_bound, before_retained = counts['model_returns'], counts['returned'], counts['retained']
    sentinel = object(); payload = sentinel
    counts['submitted'] += 1
    try:
        payload = run_forward(model, ids, kind, head, delivery, observation)
        observation['helper_return_observed'] = True
        counts['model_returns'] = before_returns + int(observation.get('model_call_return_observed', False))
        counts['returned'] = before_bound + 1
        counts['retained'] = before_retained + sum(map(len, payload.values()))
        if counts['retained'] > RAW_LIMIT:
            raise RuntimeError('raw payload exceeds fixed bound')
        activity['phase'] = 'persist_and_validate'
        with path.open('xb' if kind == 'clean' else 'ab') as raw:
            offsets = base._write_payload(raw, payload)
        base._append_event(log, {'event': 'raw_recorded', 'case_id': case_id, 'action': action, 'raw': offsets})
        marker['recorded'] = True
        if ids.tolist() != [case['token_ids']] or prior.runtime(model) != runtime_before:
            raise RuntimeError('input or runtime changed after forward')
        validate_payload(payload, kind, cache, head, delivery)
        other = case['candidate_g_id'] if case['correct_id'] == case['candidate_f_id'] else case['candidate_f_id']
        scored = m.score(payload['logits'], case['correct_id'], other)
        record = {'case_id': case_id, 'group': case['group'], 'variant': case['variant'],
                  'action': action, 'score': m.public_score(scored), 'raw': offsets}
        if kind == 'clean':
            cache.update({'clean': payload['logits'], 'clean_score': scored})
        elif kind == 'noop':
            cache.update({'before': payload['cproj_before'], 'noop_output': payload['cproj_output']})
        elif kind == 'patch':
            record['comparison'] = m.compare(cache['clean_score'], scored)
            cache.setdefault('patch_scores', {})[head] = scored
            cache.setdefault('arm_outputs', {})[head] = payload['cproj_output']
        base._append_event(log, {'event': 'complete', **record})
        completed.append(record)
    except BaseException:
        bound = payload is not sentinel
        observation['helper_return_observed'] = bound
        observation['helper_return_unbound_uncertain'] = bool(observation.get('model_call_return_observed', False) and not bound)
        counts['model_returns'] = before_returns + int(observation.get('model_call_return_observed', False))
        counts['returned'] = before_bound + int(bound)
        partial = observation.get('payload', {}); size = sum(map(len, partial.values()))
        counts['retained'] = before_retained + size if marker['recorded'] else before_retained
        if partial and not marker['recorded']:
            try:
                if before_retained + size > RAW_LIMIT:
                    raise RuntimeError('partial payload exceeds fixed bound')
                if old.old.recover_payload(log, path, case_id, action, partial, start_offset,
                                           observation.get('model_call_return_observed', False)):
                    marker['partial'] = True; counts['retained'] = before_retained + size
            except Exception as error:
                marker['partial_save_error'] = type(error).__name__
        raise


def describe_plan(plan):
    out = {'status': plan['status'], 'reason': plan['reason'], 'q_out': plan['q_out'],
           'source_output_norms': [{str(h): row[h] for h in m.HEADS} for row in plan['source_output_norms']],
           'minima': plan['minima'], 'plans': None}
    if plan['plans'] is not None:
        out['plans'] = [{str(h): {**{key: row[h][key] for key in
                     ('fraction', 'donor_gap_input_l2', 'delivered_input_l2', 'predicted_output_l2', 'changed_bits')},
                     'source_bits_hex': row[h]['source'].hex(), 'donor_bits_hex': row[h]['donor'].hex(),
                     'delivered_bits_hex': row[h]['delivered'].hex()} for h in m.HEADS} for row in plan['plans']]
    return out


def execute_family(batch, caches, perform, log, weight_raw):
    if (len(batch) != 4 or len(caches) != 4 or any(c['group'] != batch[0]['group'] or c['variant'] != v
                                                     for v, c in enumerate(batch))):
        raise ValueError('fixed family order differs')
    group = batch[0]['group']
    for case, cache in zip(batch, caches):
        perform(case, 'noop', cache)
    for case, cache in zip(batch, caches):
        perform(case, 'self', cache, 2, m.head_slice(cache['before'], 2))
    plan = m.plan_family(tuple(c['before'] for c in caches), weight_raw)
    description = describe_plan(plan)
    base._append_event(log, {'event': 'family_plan', 'group': group, 'plan': description})
    if plan['status'] == 'eligible':
        for v, (case, cache) in enumerate(zip(batch, caches)):
            shift = (4 * (group - 1) + v) % 3
            for head in ORDER[shift:] + ORDER[:shift]:
                perform(case, 'patch', cache, head, plan['plans'][v][head]['delivered'])
    for case, cache in zip(batch, caches):
        perform(case, 'clean_after', cache)
    if plan['status'] == 'eligible':
        assessment = m.inspect_actual(tuple(c['noop_output'] for c in caches),
                                      tuple(c['arm_outputs'] for c in caches), plan)
        effects = m.summarize_effects([c['clean_score'] for c in caches], [c['patch_scores'] for c in caches])
    else:
        assessment = {'status': 'dose-ineligible', 'reason': plan['reason'], 'q_out': plan['q_out']}
        effects = None
    baseline = m.baseline([c['clean_score'] for c in caches])
    summary = {'group': group, **baseline, 'dose_status': assessment['status'],
               'h2_control_rule': effects['h2_control_rule'] if effects is not None else None}
    family = {'summary': summary, 'plan': description, 'assessment': assessment, 'effects': effects}
    base._append_event(log, {'event': 'family_complete', **family})
    return family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir', 'stage-dir', 'output-dir', 'manifest', 'inventory'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if any(not p.is_absolute() for p in (args.cache_dir, args.stage_dir, args.output_dir, args.manifest, args.inventory)):
        raise ValueError('all paths must be absolute')
    digest = verify_manifest(args.manifest)
    if os.environ.get('COPY_PAIRED_BOOTSTRAP_SHA256') != digest:
        raise RuntimeError('bootstrap and manifest differ')
    manifest = json.loads(args.manifest.read_bytes())
    raw_cases = (ROOT / CASES).read_bytes()
    if (sha(raw_cases) != CASE_SHA or sha((ROOT / PROTOCOL).read_bytes()) != PROTOCOL_SHA
            or sha((ROOT / AMENDMENT).read_bytes()) != AMENDMENT_SHA):
        raise RuntimeError('fixed rules or cases differ')
    data = json.loads(raw_cases); cases = data['cases']
    if len(cases) != 64:
        raise RuntimeError('fixed case count differs')
    exposure = verify_reservation(args.inventory, cases)
    root = WORK_ROOT.resolve(strict=True); parent = args.output_dir.parent.resolve(strict=True)
    stage = args.stage_dir.resolve(strict=True)
    if (args.output_dir.exists() or not parent.is_relative_to(root) or not stage.is_relative_to(root)
            or parent.is_relative_to(ROOT) or stage.is_relative_to(ROOT)):
        raise ValueError('new output and staging paths must be under E work outside checkout')
    from looking_glass.pretrained import ASSETS
    if sum(size for size, _ in ASSETS.values()) != STAGING_BYTES:
        raise RuntimeError('pinned staged-asset size differs')
    needed = RAW_LIMIT + WEIGHT_LIMIT + STAGING_BYTES + META_RESERVE + FAILURE_RESERVE
    if shutil.disk_usage(parent).free < needed or shutil.disk_usage('C:/').free < 1024**3:
        raise OSError('insufficient E or C headroom at phase start')
    torch.set_num_interop_threads(1)
    initial = base.numeric_policy(initialize=True); binary64_rounding_probe()
    torch.use_deterministic_algorithms(False); torch.set_float32_matmul_precision('highest'); torch.set_default_dtype(torch.float32)
    output = args.output_dir; output.mkdir()
    run_manifest = {'schema': 'distilgpt2-paired-continuation-run-v1', 'source_revision': manifest['source_revision'],
                    'submitted_manifest_sha256': digest, 'cases_sha256': CASE_SHA, 'protocol_sha256': PROTOCOL_SHA,
                    'amendment_sha256': AMENDMENT_SHA, 'exposure': exposure, 'dependencies': base.dependency_identity(),
                    'model_id': base.MODEL_ID, 'model_revision': base.MODEL_REVISION, 'weights_sha256': base.WEIGHTS_SHA256,
                    'maximum_action_bytes': RAW_LIMIT, 'maximum_weight_bytes': WEIGHT_LIMIT, 'staging_bytes': STAGING_BYTES,
                    'required_e_free': needed, 'initial_numeric_probe': initial, 'stage_directory': 'E work',
                    'scope': 'fixed public outcome-informed paired continuation prompts; cooperative process observation'}
    run_sha = old.bounded_json(output / 'manifest.json', run_manifest, old.MAX_MANIFEST_BYTES)
    counts = {'submitted': 0, 'model_returns': 0, 'returned': 0, 'retained': 0}
    activity = {'phase': 'model_load', 'case_id': None, 'action': None, 'observation': {}, 'marker': {}}
    completed = []; families = []
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False, stage_dir=stage)
        if bundle.metadata['weights_sha256'] != base.WEIGHTS_SHA256:
            raise RuntimeError('checkpoint differs')
        model = bundle.model; validate_tokenization(bundle.tokenizer, cases)
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        state_before, buffers_before, runtime_before = base.state_hash(model), base.buffer_hash(model), prior.runtime(model)
        assert_no_forward_hooks(model)
        w = model.transformer.h[3].attn.c_proj.weight
        if tuple(w.shape) != (768, 768) or w.dtype != torch.float32 or w.device.type != 'cpu':
            raise RuntimeError('projection geometry differs')
        weight_raw = {h: base._raw(w[64*h:64*(h+1)].detach().contiguous(), WEIGHT_BYTES) for h in m.HEADS}
        weights(weight_raw); combined = b''.join(weight_raw[h] for h in m.HEADS)
        with (output / 'projection-weights.f32').open('xb') as stream:
            if len(combined) != WEIGHT_LIMIT or stream.write(combined) != len(combined):
                raise OSError('weight slice write failed')
            stream.flush(); os.fsync(stream.fileno())
        with (output / 'attempt.jsonl').open('xb') as stream:
            log = old.BoundedJournal(stream)
            base._append_event(log, {'event': 'runtime_ready', 'actual_runtime': runtime_before, 'weight_order': list(m.HEADS),
                'weight_sha256': sha(combined), 'weight_head_sha256': {str(h): sha(weight_raw[h]) for h in m.HEADS}})
            def perform(case, kind, cache, head=None, delivery=None):
                return perform_action(case, kind, cache, head, delivery, model=model, runtime_before=runtime_before,
                                      output=output, log=log, counts=counts, activity=activity, completed=completed)
            caches = [{} for _ in cases]
            for case, cache in zip(cases, caches):
                perform(case, 'clean', cache)
            for i in range(16):
                families.append(execute_family(cases[4*i:4*i+4], caches[4*i:4*i+4], perform, log, weight_raw))
        activity['phase'] = 'verification'
        planned = sum(f['plan']['status'] == 'eligible' for f in families)
        forwards = 64 + 16*12 + planned*12
        raw_bytes = 64*m.ROW_BYTES + 16*(4*m.ROW_BYTES + 8*(m.ROW_BYTES+3*m.WITNESS_BYTES)) \
                    + planned*12*(m.ROW_BYTES+3*m.WITNESS_BYTES)
        if (len(families) != 16 or len(completed) != forwards or counts != {'submitted': forwards, 'model_returns': forwards,
                'returned': forwards, 'retained': raw_bytes} or base.state_hash(model) != state_before
                or base.buffer_hash(model) != buffers_before or prior.runtime(model) != runtime_before
                or verify_manifest(args.manifest) != digest or GUARD_EVENTS or torch.cuda.is_initialized()):
            raise RuntimeError('final counts, state, runtime or source differ')
        assert_no_forward_hooks(model)
        outcome = m.decide([f['summary'] for f in families])
        result = {'schema': 'distilgpt2-paired-continuation-result-v1', 'status': 'complete', 'manifest_sha256': run_sha,
                  'checkpoint': bundle.metadata, 'weight_sha256': sha(combined),
                  'weight_head_sha256': {str(h): sha(weight_raw[h]) for h in m.HEADS},
                  'state_sha256_before': state_before, 'state_sha256_after': base.state_hash(model),
                  'buffers_sha256_before': buffers_before, 'buffers_sha256_after': base.buffer_hash(model),
                  'runtime': runtime_before, 'counts': counts, 'records': completed, 'families': families, 'decision': outcome}
        old.bounded_json(output / 'result.json', result, old.MAX_RESULT_BYTES)
    except BaseException as error:
        try:
            old.bounded_json(output / 'failure.json', {'schema': 'distilgpt2-paired-continuation-failure-v1',
                'status': 'incomplete', 'manifest_sha256': run_sha, 'phase': activity['phase'], 'case_id': activity['case_id'],
                'action': activity['action'], 'marker': activity['marker'],
                'model_call_return_observed': activity['observation'].get('model_call_return_observed', False),
                'helper_return_observed': activity['observation'].get('helper_return_observed', False),
                'helper_return_unbound_uncertain': activity['observation'].get('helper_return_unbound_uncertain', False),
                'counts': counts, 'completed_records': len(completed), 'raw_files': base.raw_file_inventory(output, cases),
                'error_type': type(error).__name__, 'traceback_frames': [{'file': Path(f.filename).name, 'line': f.lineno,
                    'function': f.name} for f in traceback.extract_tb(error.__traceback__)]}, old.MAX_FAILURE_BYTES)
        finally:
            raise
    print(json.dumps({'status': 'complete', 'forwards': forwards, 'decision': outcome}, allow_nan=False))


if __name__ == '__main__':
    main()
