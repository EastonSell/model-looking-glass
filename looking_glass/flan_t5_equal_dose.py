"""Fixed CPU-only FLAN-T5 matched-dose study; see FLAN_T5_EQUAL_DOSE_V1.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .flan_t5_feasibility import MANIFEST_HASH, REVISION, digest_file, memory_bytes, verify_assets

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / 'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json'
CASE_HASH = 'a0e320c9d24a8940fbca05cbc4ed6e3f2661b5b44b2e62e1140832c0e99e7cd6'
PROTOCOL = REPO / 'docs/FLAN_T5_EQUAL_DOSE_V1.md'
MAX_BYTES = 256 * 1024 * 1024
MAX_FORWARDS = 324
MAX_RETURN_BYTES = (8 * 32128 + 4 * 32 * 512) * 4
SEED = 20261028


def source_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    names = ('looking_glass/flan_t5_equal_dose.py', 'looking_glass/flan_t5_feasibility.py',
             'looking_glass/interventions.py', 'looking_glass/intervention_plan.py', 'looking_glass/model_ownership.py',
             'docs/FLAN_T5_EQUAL_DOSE_V1.md', 'docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json')
    hashes = {}
    for name in names:
        raw = (REPO / name).read_bytes()
        if raw != subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=REPO):
            raise ValueError(f'source differs from committed blob: {name}')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': commit, 'source_sha256': hashes}


def check_publication_observation(path, source):
    with path.open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('publication observation is too large')
    value = json.loads(raw)
    commit = source['commit']
    if (type(value) is not dict or value.get('schema') != 'mlg-flant5-public-observation-v1'
            or value.get('commit') != commit or value.get('source_sha256') != source['source_sha256']
            or value.get('commit_url') != f'https://github.com/EastonSell/model-looking-glass/commit/{commit}'
            or not isinstance(value.get('observer'), str) or not value['observer']
            or not isinstance(value.get('observed_utc'), str) or not value['observed_utc']):
        raise ValueError('public source observation differs from the runnable source')
    return raw


def check_cases(tokenizer, manifest):
    if (type(tokenizer).__name__ != 'T5TokenizerFast' or tokenizer.eos_token_id != 1
            or tokenizer.pad_token_id != 0 or manifest.get('schema') != 'mlg-flan-t5-equal-dose-public-cases-v1'
            or manifest.get('revision') != REVISION or manifest.get('asset_manifest_sha256') != MANIFEST_HASH
            or len(manifest.get('cases', ())) != 6):
        raise ValueError('case or tokenizer identity differs')
    ids = ('pair0-translate', 'pair0-question', 'pair1-translate', 'pair1-question',
           'pair2-translate', 'pair2-question')
    if tuple(c.get('case_id') for c in manifest['cases']) != ids:
        raise ValueError('case order differs')
    eligible = []
    for case in manifest['cases']:
        for row in [case['neutral'], *case['source'], *case['targets']]:
            value = tokenizer(row['text'], return_offsets_mapping=True, return_special_tokens_mask=True)
            for key, saved in (('input_ids', 'input_ids'), ('attention_mask', 'attention_mask'),
                               ('special_tokens_mask', 'special_tokens_mask'), ('offset_mapping', 'offsets')):
                actual = [list(v) if isinstance(v, tuple) else v for v in value[key]]
                if actual != row[saved]:
                    raise ValueError(f'tokenizer mismatch: {key}')
            if tokenizer.convert_ids_to_tokens(value['input_ids']) != row['token_strings']:
                raise ValueError('token string mismatch')
            if not row['input_ids'] or row['input_ids'][-1] != 1 or row['input_ids'].count(1) != 1:
                raise ValueError('missing or duplicate EOS')
        a, b = case['source']
        wa, wb = a['word_positions'], b['word_positions']
        reasons = []
        differences = [i for i, (x, y) in enumerate(zip(a['input_ids'], b['input_ids'])) if x != y]
        positions = [case.get(k) for k in ('word_position', 'instruction_position', 'context_position')]
        if (len(a['input_ids']) != len(b['input_ids']) or a['attention_mask'] != b['attention_mask']
                or a['special_tokens_mask'] != b['special_tokens_mask'] or len(a['input_ids']) > 32
                or wa != wb or wa != differences or len(wa) != 1
                or any(type(p) is not int or p < 0 or p >= len(a['input_ids']) for p in positions)
                or len(set(positions)) != 3 or positions[0] != wa[0]
                or any(a['special_tokens_mask'][p] or b['special_tokens_mask'][p] for p in positions)
                or any(a['input_ids'][p] != b['input_ids'][p] for p in positions[1:])):
            reasons.append('source-token-alignment')
        if not reasons:
            for source in (a, b):
                start, end = source['offsets'][positions[0]]
                if source['text'][start:end].strip() != source['word']:
                    reasons.append('source-word-offset')
        if any(len(target['input_ids']) > 8 for target in case['targets']):
            reasons.append('target-length')
        if bool(reasons) == bool(case['alignment_eligible']):
            raise ValueError('saved alignment status disagrees with current tokenization')
        eligible.append(reasons)
    return eligible


def vector_l2(left, right):
    if left.shape != right.shape:
        raise ValueError('dose vectors have different shapes')
    values = (left.double() - right.double()).reshape(-1).tolist()
    result = math.sqrt(math.fsum(float(x) * float(x) for x in values))
    if not math.isfinite(result):
        raise ValueError('dose is nonfinite')
    return result


def dose_matches(q, tolerance, achieved):
    return (all(x > 0 and abs(x-q) <= tolerance for x in achieved.values())
            and all(abs(achieved[f'word-{side}'] - achieved[f'{site}-{side}']) <= .01*achieved[f'word-{side}']
                    for side in range(2) for site in ('instruction', 'punctuation')))


def matched_replacements(states, positions):
    """Return one common q and exact float32 deliveries for both sides/sites."""
    import torch
    distances = {name: vector_l2(states[0][0, pos], states[1][0, pos])
                 for name, pos in positions.items()}
    if any(d <= 1e-6 for d in distances.values()):
        return {'status': 'dose-ineligible', 'distances': distances}, {}
    q = min(distances.values())
    tolerance = min(max(1e-5, 1e-5 * q), .01 * q)
    vectors = {}
    achieved = {}
    receiver_norms = {}
    for name, pos in positions.items():
        for side in range(2):
            own = states[side][0, pos]
            donor = states[1-side][0, pos]
            value = (own.double() + (q / distances[name]) * (donor.double() - own.double())).to(torch.float32)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('matched replacement is nonfinite')
            actual = vector_l2(value, own)
            vectors[name, side] = value
            achieved[f'{name}-{side}'] = actual
            receiver_norms[f'{name}-{side}'] = vector_l2(own, torch.zeros_like(own))
    result = {'status': 'matchable' if dose_matches(q, tolerance, achieved)
              else 'dose-unmatchable', 'q': q, 'tolerance': tolerance, 'distances': distances,
              'fractions': {name: q / d for name, d in distances.items()}, 'achieved': achieved,
              'receiver_norms': receiver_norms,
              'q_over_receiver_norm': {key: q / norm if norm > 0 else None
                                       for key, norm in receiver_norms.items()}}
    return result, vectors


def cell_decision(delta, movements):
    return all(movements['matched_word'][side] >= .20 * delta
               and all(movements['matched_word'][side] >= abs(movements[control][side]) + .05 * delta
                       for control in ('matched_instruction', 'matched_punctuation'))
               for side in range(2))


def aggregate_decision(cells):
    allowed = ('positive', 'negative', 'alignment-ineligible', 'baseline-unsupported',
               'dose-ineligible', 'dose-unmatchable')
    if (len(cells) != 6 or any(c.get('status') not in allowed for c in cells)
            or [(c.get('pair_id'), c.get('template_id')) for c in cells]
               != [(i, template) for i in range(3) for template in ('translate', 'question')]):
        return 'incomplete'
    pairs = []
    for pair in range(3):
        rows = [c for c in cells if c['pair_id'] == pair]
        if len(rows) == 2 and all(c['status'] in ('positive', 'negative') for c in rows):
            pairs.append(all(c['status'] == 'positive' for c in rows))
    return 'positive' if sum(pairs) >= 2 else 'negative' if len(pairs) - sum(pairs) >= 2 else 'inconclusive'


class Evidence:
    def __init__(self, path):
        path.mkdir(parents=True, exist_ok=False)
        self.path = path
        self.used = 0

    def write(self, name, raw, reserve=65536):
        if self.used + len(raw) + reserve > MAX_BYTES:
            raise ValueError('retained evidence cap exceeded')
        destination = self.path / name
        with destination.open('xb') as stream:
            try:
                if stream.write(raw) != len(raw):
                    raise OSError('short evidence write')
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                self.used += destination.stat().st_size
        return hashlib.sha256(raw).hexdigest()

    def json(self, name, value, reserve=65536):
        return self.write(name, (json.dumps(value, sort_keys=True, allow_nan=False,
                                            separators=(',', ':')) + '\n').encode(), reserve)

    def reserve_forward(self):
        if self.used + MAX_RETURN_BYTES + 3 * 65536 > MAX_BYTES:
            raise ValueError('insufficient evidence capacity for another bounded forward')


def score_logits(logits, target_ids):
    import torch
    if (type(target_ids) is not list or not target_ids or target_ids[-1] != 1
            or target_ids.count(1) != 1 or any(type(x) is not int or not 0 <= x < 32128 for x in target_ids)
            or tuple(logits.shape) != (1, len(target_ids), 32128) or logits.dtype != torch.float32):
        raise ValueError('target or logits do not match the fixed EOS-scoring contract')
    selected = torch.log_softmax(logits[0], dim=-1)[torch.arange(len(target_ids)),
                                                       torch.tensor(target_ids, dtype=torch.long)]
    per_token = [float(x) for x in selected]
    if not all(math.isfinite(x) for x in per_token):
        raise ValueError('nonfinite target log probabilities')
    return math.fsum(per_token), per_token


def model_snapshot(model):
    """Hash registered CPU parameters and buffers in bounded chunks."""
    import torch
    digest = hashlib.sha256()
    identity = []
    for kind, iterator in (('parameter', model.named_parameters(remove_duplicate=False)),
                           ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, value in iterator:
            if value.device.type != 'cpu' or value.layout != torch.strided:
                raise ValueError('registered model tensor is not a dense CPU tensor')
            descriptor = [kind, name, str(value.dtype), list(value.shape), bool(value.requires_grad)]
            raw = json.dumps(descriptor, separators=(',', ':')).encode()
            digest.update(len(raw).to_bytes(8, 'big'))
            digest.update(raw)
            data = value.detach().contiguous().reshape(-1).view(torch.uint8).reshape(-1)
            for start in range(0, data.numel(), 1024 * 1024):
                digest.update(data[start:start + 1024 * 1024].numpy().tobytes())
            identity.append((kind, name, id(value), value._version, tuple(value.shape), str(value.dtype)))
    return digest.hexdigest(), tuple(identity)


def hook_snapshot(model):
    import torch
    global_module = torch.nn.modules.module
    global_names = ('_global_forward_pre_hooks', '_global_forward_hooks',
                    '_global_backward_pre_hooks', '_global_backward_hooks')
    global_values = tuple((name, tuple(getattr(global_module, name).items())) for name in global_names)
    local = tuple((name, id(m), m.training, tuple(m._forward_pre_hooks.items()),
                   tuple(m._forward_hooks.items()), tuple(m._backward_pre_hooks.items()),
                   tuple(m._backward_hooks.items())) for name, m in model.named_modules(remove_duplicate=False))
    return global_values, local


def require_clean_hooks(snapshot):
    global_values, local = snapshot
    if any(items for _, items in global_values) or any(any(entry[3:]) for entry in local):
        raise ValueError('preexisting model or global hooks are not allowed')


def validate_plan_metadata(metadata, action, token_index, sequence_length):
    steps = metadata.get('steps') if type(metadata) is dict else None
    if type(steps) is not list or len(steps) != 3:
        raise ValueError('intervention plan has invalid steps')
    selected = [1, sequence_length, 512] if token_index is None else [512]
    batch = None if token_index is None else 0
    for index, (step, name, operation) in enumerate(zip(
            steps, ('before', 'action', 'after'), ('capture', action, 'capture'))):
        if (type(step) is not dict or step.get('name') != name or step.get('operation') != operation
                or type(step.get('order')) is not int or step['order'] != index
                or step.get('module_path') != 'encoder.final_layer_norm'
                or type(step.get('module_aliases')) is not list
                or 'encoder.final_layer_norm' not in step['module_aliases']
                or step.get('call_site_known') is not False or type(step.get('forward_index')) is not int
                or step['forward_index'] != 0 or type(step.get('invocation')) is not int
                or step['invocation'] != 0 or step.get('batch_index') != batch
                or (batch is not None and type(step.get('batch_index')) is not int)
                or step.get('token_index') != token_index
                or (token_index is not None and type(step.get('token_index')) is not int)
                or step.get('output_selector') != []
                or step.get('output_shape') != [1, sequence_length, 512] or step.get('selected_shape') != selected
                or step.get('dtype') != 'torch.float32' or step.get('device') != 'cpu'
                or step.get('source_capture') is not None
                or step.get('preceding_same_site_actions') != ['before', 'action'][:index]):
            raise ValueError('intervention plan metadata differs from fixed coordinates')


def validate_capture_metadata(metadata, sequence_length):
    shape = [1, sequence_length, 512]
    if (type(metadata) is not dict or metadata.get('module_path') != 'encoder.final_layer_norm'
            or type(metadata.get('module_aliases')) is not list
            or 'encoder.final_layer_norm' not in metadata['module_aliases']
            or metadata.get('call_site_known') is not False
            or type(metadata.get('forward_index')) is not int or metadata['forward_index'] != 0
            or type(metadata.get('invocation')) is not int or metadata['invocation'] != 0
            or metadata.get('batch_index') is not None or metadata.get('token_index') is not None
            or metadata.get('output_shape') != shape or metadata.get('selected_shape') != shape
            or metadata.get('dtype') != 'torch.float32' or metadata.get('device') != 'cpu'):
        raise ValueError('capture metadata differs from fixed coordinates')


def equal_bits(left, right):
    import torch
    return left.shape == right.shape and torch.equal(left.contiguous().view(torch.int32),
                                                       right.contiguous().view(torch.int32))


def save_dose_evidence(evidence, index, case_id, dose, states, positions, matched):
    chunks, layout = [], []
    for name, pos in positions.items():
        for side in range(2):
            vectors = [('own', states[side][0, pos]), ('donor', states[1-side][0, pos])]
            if (name, side) in matched:
                vectors.append(('delivered', matched[name, side]))
            for role, vector in vectors:
                raw = vector.contiguous().numpy().astype('<f4', copy=False).tobytes()
                layout.append({'site': name, 'side': side, 'role': role,
                               'offset': sum(len(x) for x in chunks), 'bytes': len(raw),
                               'shape': [512], 'dtype': 'little-endian-float32'})
                chunks.append(raw)
    filename = f'dose-{index:02d}.bin'
    sha = evidence.write(filename, b''.join(chunks))
    evidence.json(f'dose-{index:02d}.json', {'case_id': case_id, 'dose': dose,
                                             'file': filename, 'sha256': sha, 'layout': layout})


def validate_landing(condition, action, pos, own, donor, expected, value, state, before, after, dose):
    if action != 'capture':
        expected_pre = own if pos is None else own[0, pos]
        expected_post = value if pos is None else expected
        if not equal_bits(before, expected_pre) or not equal_bits(after, expected_post):
            raise ValueError('same-forward intervention witness differs')
        if condition.startswith('matched_'):
            delivered = vector_l2(after, before)
            if delivered <= 0 or abs(delivered - dose['q']) > dose['tolerance']:
                raise ValueError('same-forward dose differs from requested dose')
    if condition in ('repeat', 'full_self') and not equal_bits(state, own):
        raise ValueError('full-state self control failed')
    if condition == 'full_swap' and not equal_bits(state, donor):
        raise ValueError('full-state donor control failed')
    if pos is not None:
        if not equal_bits(state[0, pos], expected):
            raise ValueError('intervention did not land exactly')
        if any(not equal_bits(state[0, p], own[0, p]) for p in range(state.shape[1]) if p != pos):
            raise ValueError('unselected encoder position changed')


def run_forward(model, evidence, records, active, source, target, name, action='clean', value=None,
                token_index=None):
    """Execute one bounded forward and save returned bytes before validating them."""
    import torch
    from .interventions import ActivationCapture
    from .intervention_plan import ActivationPlan

    if len(records) >= MAX_FORWARDS:
        raise ValueError('forward cap exceeded')
    evidence.reserve_forward()
    source_ids = torch.tensor([source['input_ids']], dtype=torch.long)
    mask = torch.tensor([source['attention_mask']], dtype=torch.long)
    target_ids = target['input_ids']
    decoder_ids = [0] + target_ids[:-1]
    decoder = torch.tensor([decoder_ids], dtype=torch.long)
    decoder_mask = torch.ones_like(decoder)
    original = tuple(t.clone() for t in (source_ids, mask, decoder, decoder_mask))
    case_id = active.get('case_id')
    active.clear()
    active.update(case_id=case_id, forward=name, index=len(records), phase='forward', return_observed=False)
    kwargs = {'input_ids': source_ids, 'attention_mask': mask, 'decoder_input_ids': decoder,
              'decoder_attention_mask': decoder_mask, 'use_cache': False, 'return_dict': True}
    out = capture = before = after = metadata = None
    returned = False

    def bounded(tensor, limit):
        # Check the original size before materializing any noncontiguous return.
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                or tensor.dtype != torch.float32 or tensor.layout != torch.strided
                or not 0 < tensor.numel() <= limit):
            return None, b''
        tensor = tensor.detach().contiguous()
        return tensor, tensor.numpy().astype('<f4', copy=False).tobytes()

    def persist_return(failed):
        raw = (getattr(out, 'logits', None), getattr(out, 'encoder_last_hidden_state', None),
               before, after, capture)
        names = ('logits', 'encoder', 'before', 'after', 'capture')
        limits = (8 * 32128, 32 * 512, 32 * 512, 32 * 512, 32 * 512)
        items = [bounded(t, limit) for t, limit in zip(raw, limits)]
        layout, offset = {}, 0
        for key, (tensor, data) in zip(names, items):
            layout[key] = {'available': tensor is not None,
                           'shape': None if tensor is None else list(tensor.shape),
                           'offset': offset, 'bytes': len(data)}
            offset += len(data)
        active.update(return_observed=returned, tensor_layout=layout)
        packed = b''.join(data for _, data in items)
        if packed:
            filename = f'forward-{len(records):03d}' + ('-failed.bin' if failed else '.bin')
            sha = evidence.write(filename, packed)
            active.update(saved_file=filename, saved_sha256=sha)
        return [item[0] for item in items], layout

    try:
        with torch.no_grad():
            if action == 'clean':
                out = model(**kwargs)
                returned = True
            elif action == 'capture':
                with ActivationCapture(model, 'encoder.final_layer_norm') as session:
                    out = model(**kwargs)
                    returned = True
                capture = session.value
                metadata = session.metadata
            else:
                site = {'module_path': 'encoder.final_layer_norm', 'invocation': 0}
                if token_index is not None:
                    site.update(batch_index=0, token_index=token_index)
                steps = [dict(name='before', operation='capture', **site),
                         dict(name='action', operation=action, **site),
                         dict(name='after', operation='capture', **site)]
                if action == 'patch':
                    steps[1]['value'] = value
                with ActivationPlan(model, steps) as session:
                    out = model(**kwargs)
                    returned = True
                snapshots = session.captures
                before, after = snapshots['before'], snapshots['after']
                metadata = session.metadata
    except BaseException:
        # The model may have returned successfully before session validation failed.
        try:
            persist_return(failed=True)
            evidence.json(f'forward-{len(records):03d}-failed.json',
                          dict(active, status='forward-failed', session_metadata=metadata))
        except BaseException:
            pass
        raise

    (logits, state, before, after, capture), layout = persist_return(failed=False)
    active['phase'] = 'post-return-validation'
    evidence.json(f'forward-{len(records):03d}-observed.json', dict(active, session_metadata=metadata))
    if not all(torch.equal(x, y) for x, y in zip((source_ids, mask, decoder, decoder_mask), original)):
        raise ValueError('model mutated input or mask')
    if (logits is None or state is None or tuple(logits.shape) != (1, len(target_ids), 32128)
            or tuple(state.shape) != (1, len(source['input_ids']), 512)
            or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(state).all())
            or getattr(out, 'past_key_values', None) is not None):
        raise ValueError('nonfinite, cached, unavailable or malformed model output')
    if action == 'capture':
        validate_capture_metadata(metadata, len(source['input_ids']))
        if capture is None or not equal_bits(capture, state):
            raise ValueError('capture differs from root-model encoder state')
    elif action != 'clean':
        validate_plan_metadata(metadata, action, token_index, len(source['input_ids']))
        if before is None or after is None:
            raise ValueError('plan witnesses unavailable')
    active['phase'] = 'scoring'
    score, per_token = score_logits(logits, target_ids)
    row = {'index': len(records), 'name': name, 'action': action, 'token_index': token_index,
           'source_ids': source['input_ids'], 'source_mask': source['attention_mask'],
           'target_ids': target_ids, 'decoder_ids': decoder_ids, 'decoder_mask': [1] * len(decoder_ids),
           'selected_logprobs': per_token, 'score': score, 'file': active['saved_file'],
           'sha256': active['saved_sha256'], 'dtype': 'little-endian-float32', 'tensor_layout': layout,
           'session_metadata': metadata}
    evidence.json(f'forward-{len(records):03d}.json', row)
    records.append(row)
    active['phase'] = 'control-check'
    return score, logits, state, before, after


def run_cases(manifest, eligibility, evidence, records, active, forward):
    import torch
    equal = equal_bits
    summaries = []
    for index, case in enumerate(manifest['cases']):
        active.update(case_id=case['case_id'], forward=None, phase='case')
        summary = {'case_id': case['case_id'], 'pair_id': case['pair_id'], 'template_id': case['template_id']}
        if eligibility[index]:
            summary.update(status='alignment-ineligible', reasons=eligibility[index])
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        targets = case['targets']
        context_scores = [forward(case['neutral'], target, f'{case["case_id"]}-context-{j}')[0]
                          for j, target in enumerate(targets)]
        bases, states, scores = {}, {}, {}
        for side in range(2):
            for target in range(2):
                score, logits, state, _, _ = forward(case['source'][side], targets[target],
                                               f'{case["case_id"]}-base-{side}-{target}')
                bases[side, target] = logits
                scores['base', side, target] = score
                if side in states and not equal(states[side], state):
                    raise ValueError('encoder state differs between target forwards')
                states[side] = state
        margins = [scores['base', side, 0] - scores['base', side, 1] for side in range(2)]
        delta = margins[0] - margins[1]
        summary.update(baseline_margins=margins, delta=delta, context_scores=context_scores,
                       context_margin=context_scores[0]-context_scores[1])
        if not (margins[0] > 0 and margins[1] < 0 and delta >= 1.0):
            summary['status'] = 'baseline-unsupported'
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        positions = {'word': case['word_position'], 'instruction': case['instruction_position'],
                     'punctuation': case['context_position']}
        dose, matched = matched_replacements(states, positions)
        summary['dose'] = dose
        save_dose_evidence(evidence, index, case['case_id'], dose, states, positions, matched)
        if dose['status'] != 'matchable':
            summary['status'] = dose['status']
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        conditions = ('repeat', 'null_word', 'self_word', 'self_instruction', 'self_punctuation',
                      'full_self', 'full_swap', 'full_word_swap', 'zero_word',
                      'matched_word', 'matched_instruction', 'matched_punctuation')
        for condition in conditions:
            for side in range(2):
                own, donor = states[side], states[1-side]
                if condition == 'repeat':
                    action, pos, value = 'capture', None, None
                elif condition in ('full_self', 'full_swap'):
                    action, pos, value = 'patch', None, own if condition == 'full_self' else donor
                else:
                    site = ('instruction' if condition.endswith('instruction') else
                            'punctuation' if condition.endswith('punctuation') else 'word')
                    pos = positions[site]
                    action = 'null' if condition == 'null_word' else 'zero' if condition == 'zero_word' else 'patch'
                    value = (None if action != 'patch' else matched[site, side] if condition.startswith('matched_')
                             else donor[0, pos] if condition == 'full_word_swap' else own[0, pos])
                expected = None if pos is None else (torch.zeros_like(own[0, pos]) if action == 'zero' else
                                                      own[0, pos] if action == 'null' else value)
                for target in range(2):
                    score, logits, state, before, after = forward(case['source'][side], targets[target],
                        f'{case["case_id"]}-{condition}-{side}-{target}', action, value, pos)
                    scores[condition, side, target] = score
                    validate_landing(condition, action, pos, own, donor, expected, value,
                                     state, before, after, dose)
                    if condition in ('repeat', 'null_word', 'self_word', 'self_instruction',
                                     'self_punctuation', 'full_self') and not equal(logits, bases[side, target]):
                        raise ValueError(f'identity logits control failed: {condition}')
                    if condition == 'full_swap' and not equal(logits, bases[1-side, target]):
                        raise ValueError('full-swap logits control failed')
        condition_margins = {condition: [scores[condition, side, 0] - scores[condition, side, 1]
                                           for side in range(2)] for condition in conditions}
        movements = {condition: [margins[0]-values[0], values[1]-margins[1]]
                     for condition, values in condition_margins.items()}
        summary.update(status='positive' if cell_decision(delta, movements) else 'negative',
                       positions=positions, condition_margins=condition_margins,
                       signed_movements=movements,
                       absolute_movements={k: [abs(x) for x in v] for k, v in movements.items()})
        summaries.append(summary)
        evidence.json(f'case-{index:02d}.json', summary)
        print(json.dumps({'case_id': case['case_id'], 'status': summary['status'],
                          'q': dose['q'], 'forward_count': len(records)}), flush=True)
    return summaries


def run(asset_dir, output, publication_observation):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1')
            or os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1'
            or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or os.environ.get('OMP_NUM_THREADS') != '1'
            or os.environ.get('MKL_NUM_THREADS') != '1'):
        raise ValueError('fixed CPU/offline/numeric environment is required')
    if 'torch' in sys.modules:
        raise ValueError('PyTorch must not be preloaded before environment checks')
    if digest_file(CASES) != CASE_HASH:
        raise ValueError('fixed case manifest differs')
    source_before = source_identity()
    observation = check_publication_observation(publication_observation, source_before)
    assets_before = verify_assets(asset_dir)
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 2 * MAX_BYTES:
        raise ValueError('insufficient free space or missing output parent')
    manifest = json.loads(CASES.read_bytes())
    evidence = Evidence(output)
    observation_hash = evidence.write('publication-observation.json', observation)
    evidence.json('started.json', {'schema': 'mlg-flant5-equal-dose-attempt-v1', 'status': 'started',
                                    'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH,
                                    'case_manifest_sha256': CASE_HASH, 'evidence_cap': MAX_BYTES,
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

        equal = equal_bits

        def forward(source, target, name, action='clean', value=None, token_index=None):
            return run_forward(model, evidence, records, active, source, target, name, action, value, token_index)

        summaries = run_cases(manifest, eligibility, evidence, records, active, forward)
        if source_identity() != source_before or verify_assets(asset_dir) != assets_before or hook_snapshot(model) != hooks_before:
            raise ValueError('source, assets or hooks changed during execution')
        if model_snapshot(model) != model_before:
            raise ValueError('model parameters or buffers changed during execution')
        result = {'schema': 'mlg-flant5-equal-dose-v1', 'status': 'complete',
                  'decision': aggregate_decision(summaries), 'cases': summaries, 'forwards': len(records),
                  'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH, 'case_manifest_sha256': CASE_HASH,
                  'environment': runtime, 'final_memory': memory_bytes(), 'hooks_restored': True,
                  'bytes_before_terminal': evidence.used,
                  'limitations': ['Public, correlated pairs and templates; no statistical or generalization claim.',
                                  'Matched input L2 does not match direction or downstream effect.',
                                  'Exact sequence scores include EOS and differ in target length and priors.']}
        sha = evidence.json('completed.json', result, reserve=0)
        print(json.dumps({'status': 'complete', 'decision': result['decision'], 'forwards': len(records),
                          'result_sha256': sha, 'retained_bytes': evidence.used}), flush=True)
        return result
    except BaseException as error:
        try:
            evidence.json('failure.json', {'schema': 'mlg-flant5-equal-dose-attempt-v1', 'status': 'incomplete',
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
