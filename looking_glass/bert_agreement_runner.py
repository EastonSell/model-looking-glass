"""One bounded CPU DistilBERT forward with an ordered vector intervention.

This primitive neither loads a checkpoint nor establishes model provenance.
The caller must exclusively own the model and all input/source tensors. CUDA is
deliberately rejected until a separately supervised, profiled owner is wired in.
Logical tensor caps do not bound model, allocator, transient or process memory.
"""

from dataclasses import dataclass
import base64
import copy
import hashlib
import math
import sys

import torch

from .intervention_plan import ActivationPlan

SITES = (('candidate', 2, 2), ('block2_distractor', 2, 5),
         ('random_block3_distractor', 3, 5), ('random_block4_distractor', 4, 5),
         ('random_block0_subject', 0, 2))


@dataclass(frozen=True)
class Observation:
    """One mask logit vector and the selected pre/post action representations."""

    mask_logits: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor | None
    is_bits: int
    are_bits: int
    is_logit: float
    are_logit: float
    events: tuple
    capture_bytes: int
    source_bytes: int
    model_input_ids: tuple
    model_attention_mask: tuple


def float32_bytes(tensor):
    """Copy original CPU float32 bits in little-endian row-major order."""
    if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
            or tensor.dtype != torch.float32 or sys.byteorder != 'little'):
        raise ValueError('expected a little-endian CPU float32 tensor')
    # Reinterpret before copying so a float copy cannot alter denormal bits.
    return tensor.detach().view(torch.int32).contiguous().numpy().tobytes()


def decode_float32_bits(bits):
    """Decode finite IEEE binary32 bits into an exact binary64 value."""
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError('expected an unsigned 32-bit bit pattern')
    exponent, fraction = (bits >> 23) & 0xff, bits & 0x7fffff
    if exponent == 0xff:
        raise ValueError('nonfinite float32 bit pattern')
    magnitude = (math.ldexp(float(fraction), -149) if exponent == 0
                 else math.ldexp(float((1 << 23) | fraction), exponent - 150))
    return -magnitude if bits >> 31 else magnitude


def signed_donor_effect(target_number, cells):
    """Compute the prespecified four-cell signed margin contrast from bits."""
    keys = ('a_same', 'a_opposite', 'b_same', 'b_opposite')
    if target_number not in ('singular', 'plural') or type(cells) is not dict or set(cells) != set(keys):
        raise ValueError('expected one complete four-cell donor contrast')
    margins = {}
    for name in keys:
        pair = cells[name]
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise ValueError('each cell requires two float32 bit patterns')
        margins[name] = math.fsum((decode_float32_bits(pair[0]), -decode_float32_bits(pair[1])))
    delta_a = math.fsum((margins['a_opposite'], -margins['a_same']))
    delta_b = math.fsum((margins['b_opposite'], -margins['b_same']))
    sign = 1 if target_number == 'singular' else -1
    effect = -sign * (math.fsum((delta_a, delta_b)) / 2.0)
    if not math.isfinite(effect):
        raise ValueError('nonfinite donor contrast')
    return effect


def _has_hooks(model):
    names = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks',
             '_backward_hooks')
    if any(getattr(module, name, {}) for module in model.modules() for name in names):
        return True
    registry = torch.nn.modules.module
    return any(getattr(registry, name, {}) for name in
               ('_global_forward_pre_hooks', '_global_forward_hooks',
                '_global_backward_pre_hooks', '_global_backward_hooks'))


def run_observation(model, input_ids, *, layer, token, mask_token, is_id, are_id,
                    source=None, self_patch=False):
    """Capture or patch one selected block output and retain the mask logits.

    Cross-input ``source`` must be a one-dimensional float32 vector from the
    same site and runtime. A same-invocation ``self_patch`` captures its source
    in the same forward. The caller binds source identity and supplies the
    all-ones attention mask only for fixed, unpadded ten-token inputs.
    """
    from transformers import DistilBertForMaskedLM

    if type(model) is not DistilBertForMaskedLM:
        raise TypeError('expected an unmodified DistilBertForMaskedLM')
    if any(module.training for module in model.modules()):
        raise ValueError('model must be in evaluation mode')
    if _has_hooks(model):
        raise RuntimeError('preexisting model or global hooks are unsupported')
    if (type(input_ids) is not torch.Tensor or input_ids.device.type != 'cpu'
            or input_ids.dtype != torch.int64 or tuple(input_ids.shape) != (1, 10)):
        raise ValueError('expected one CPU int64 input row of length ten')
    if any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError('only CPU float32 model execution is enabled')
    if any(buffer.device.type != 'cpu' for buffer in model.buffers()):
        raise ValueError('all model buffers must remain on CPU')
    if (type(layer) is not int or not 0 <= layer < len(model.distilbert.transformer.layer)
            or type(token) is not int or not 0 <= token < 10
            or type(mask_token) is not int or not 0 <= mask_token < 10):
        raise ValueError('invalid block or token position')
    vocab, hidden = model.config.vocab_size, model.config.dim
    if (type(is_id) is not int or type(are_id) is not int or is_id == are_id
            or not 0 <= is_id < vocab or not 0 <= are_id < vocab
            or bool((input_ids < 0).any()) or bool((input_ids >= vocab).any())):
        raise ValueError('invalid vocabulary token identity')
    if type(self_patch) is not bool or (self_patch and source is not None):
        raise ValueError('self_patch and external source are mutually exclusive')
    if source is not None and (type(source) is not torch.Tensor or source.device.type != 'cpu'
                               or source.dtype != torch.float32 or tuple(source.shape) != (hidden,)
                               or not bool(torch.isfinite(source).all())):
        raise ValueError('source must be a finite CPU float32 hidden vector')

    path = f'distilbert.transformer.layer.{layer}'
    site = {'module_path': path, 'invocation': 0, 'batch_index': 0, 'token_index': token}
    steps = [{'name': 'before', 'operation': 'capture', **site}]
    if self_patch or source is not None:
        patch = {'name': 'patch', 'operation': 'patch', **site}
        if self_patch:
            patch['source_capture'] = 'before'
        else:
            patch['value'] = source
        steps.extend((patch, {'name': 'after', 'operation': 'capture', **site}))
    plan = ActivationPlan(model, steps, max_tensor_bytes=10 * hidden * 4,
                          max_capture_bytes=2 * hidden * 4,
                          max_source_bytes=hidden * 4, max_steps=3, max_invocations=1)
    attention_mask = torch.ones_like(input_ids)
    model_input_ids = tuple(input_ids[0].tolist())
    model_attention_mask = tuple(attention_mask[0].tolist())
    with torch.inference_mode(), plan:
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_attentions=False, output_hidden_states=False,
                       return_dict=True).logits
        if (tuple(input_ids[0].tolist()) != model_input_ids
                or tuple(attention_mask[0].tolist()) != model_attention_mask):
            raise RuntimeError('model input tensors changed during execution')
        if (type(logits) is not torch.Tensor or logits.dtype != torch.float32
                or logits.device.type != 'cpu' or tuple(logits.shape) != (1, 10, vocab)
                or not bool(torch.isfinite(logits).all())):
            raise RuntimeError('model returned invalid or nonfinite logits')
        mask_logits = logits[0, mask_token].detach().view(torch.int32).clone().view(torch.float32)
    captures, metadata = plan.captures, plan.metadata
    if _has_hooks(model):
        raise RuntimeError('hook cleanup is uncertain; discard the model')
    events = metadata['steps']
    expected_ops = ['capture', 'patch', 'capture'] if len(steps) == 3 else ['capture']
    names = ['before', 'patch', 'after'] if len(steps) == 3 else ['before']
    expected_capture_bytes = hidden * 4 * (2 if len(steps) == 3 else 1)
    expected_source_bytes = hidden * 4 if source is not None else 0
    if (len(events) != len(expected_ops)
            or type(metadata['capture_bytes']) is not int or type(metadata['source_bytes']) is not int
            or metadata['capture_bytes'] != expected_capture_bytes
            or metadata['source_bytes'] != expected_source_bytes
            or any(event['operation'] != op or event['name'] != names[index]
                   or any(type(event[field]) is not int for field in
                          ('order', 'forward_index', 'invocation', 'batch_index', 'token_index'))
                   or any(type(value) is not int for field in ('output_selector', 'output_shape', 'selected_shape')
                          for value in event[field])
                   or event['order'] != index or event['forward_index'] != 0
                   or event['source_capture'] != ('before' if self_patch and index == 1 else None)
                   or event['preceding_same_site_actions'] != names[:index]
                   or event['module_path'] != path or event['module_aliases'] != [path]
                   or event['invocation'] != 0 or event['batch_index'] != 0
                   or event['token_index'] != token or event['output_selector'] != [0]
                   or event['output_shape'] != [1, 10, hidden]
                   or event['selected_shape'] != [hidden]
                   or event['dtype'] != 'torch.float32' or event['device'] != 'cpu'
                   for index, (event, op) in enumerate(zip(events, expected_ops)))):
        raise RuntimeError('intervention metadata differs from the requested site')
    before, after = captures['before'], captures.get('after')
    if not bool(torch.isfinite(before).all()) or (after is not None and not bool(torch.isfinite(after).all())):
        raise RuntimeError('intervention captured nonfinite values')
    if after is not None:
        expected = before if self_patch else source
        if not torch.equal(after.view(torch.int32).contiguous(),
                           expected.view(torch.int32).contiguous()):
            raise RuntimeError('replacement bits did not land at the selected site')
    raw = mask_logits.view(torch.int32)
    is_bits, are_bits = int(raw[is_id]) & 0xffffffff, int(raw[are_id]) & 0xffffffff
    is_logit, are_logit = decode_float32_bits(is_bits), decode_float32_bits(are_bits)
    if not math.isfinite(is_logit) or not math.isfinite(are_logit):
        raise RuntimeError('nonfinite selected logit')
    return Observation(mask_logits, before, after, is_bits, are_bits,
                       is_logit, are_logit, tuple(events),
                       metadata['capture_bytes'], metadata['source_bytes'],
                       model_input_ids, model_attention_mask)


def _family_rows(manifest, split, family, mask_id):
    """Own and verify all eight input identities before any family forward."""
    if (type(manifest) is not dict or type(manifest.get('rows')) is not list
            or type(manifest.get('targets')) is not list or type(split) is not str
            or type(family) is not int or family < 1 or type(mask_id) is not int or mask_id < 0):
        raise ValueError('invalid manifest or family selection')
    rows = [copy.deepcopy(row) for row in manifest['rows']
            if row.get('split') == split and row.get('family') == family]
    targets = [copy.deepcopy(row) for row in manifest['targets']
               if row.get('split') == split and row.get('family') == family]
    if len(rows) != 8 or len(targets) != 4:
        raise ValueError('a family must contain eight rows and four targets')
    by_id = {}
    combinations = set()
    for row in rows:
        row_id, source, number, distractor = (row.get('row_id'), row.get('subject_source'),
                                               row.get('subject_number'), row.get('distractor_number'))
        if (type(row_id) is not str or not row_id or len(row_id) > 128 or row_id in by_id
                or source not in ('A', 'B') or number not in ('singular', 'plural')
                or distractor not in ('singular', 'plural')
                or type(row.get('subject_lemma')) is not str
                or type(row.get('distractor_lemma')) is not str
                or type(row.get('input_ids')) is not list or len(row['input_ids']) != 10
                or any(type(value) is not int or value < 0 for value in row['input_ids'])
                or type(row.get('attention_mask')) is not list
                or len(row['attention_mask']) != 10
                or any(type(value) is not int or value != 1 for value in row['attention_mask'])
                or row['input_ids'][6] != mask_id):
            raise ValueError('invalid or duplicate family row')
        combination = (source, number, distractor)
        if combination in combinations:
            raise ValueError('duplicate family factorial cell')
        combinations.add(combination)
        by_id[row_id] = row
    if (len(combinations) != 8 or len({row['distractor_lemma'] for row in rows}) != 1
            or len({row['subject_lemma'] for row in rows if row['subject_source'] == 'A'}) != 1
            or len({row['subject_lemma'] for row in rows if row['subject_source'] == 'B'}) != 1
            or {row['subject_lemma'] for row in rows if row['subject_source'] == 'A'} ==
               {row['subject_lemma'] for row in rows if row['subject_source'] == 'B'}):
        raise ValueError('inconsistent family lemmas')
    template_positions = (0, 1, 3, 4, 6, 7, 8, 9)
    if (len({tuple(row['input_ids'][i] for i in template_positions) for row in rows}) != 1
            or len({tuple(row['input_ids']) for row in rows}) != 8):
        raise ValueError('family rows must share a template and remain distinct')
    subject_ids = {}
    distractor_ids = {}
    for row in rows:
        subject_ids.setdefault((row['subject_source'], row['subject_number']), set()).add(row['input_ids'][2])
        distractor_ids.setdefault(row['distractor_number'], set()).add(row['input_ids'][5])
    if (any(len(values) != 1 for values in subject_ids.values())
            or len({next(iter(values)) for values in subject_ids.values()}) != 4
            or any(len(values) != 1 for values in distractor_ids.values())
            or len({next(iter(values)) for values in distractor_ids.values()}) != 2):
        raise ValueError('subject or distractor token identities are inconsistent')
    expected_order = [('singular', 'singular'), ('singular', 'plural'),
                      ('plural', 'singular'), ('plural', 'plural')]
    seen = set()
    for target, factors in zip(targets, expected_order):
        case_id, target_id, mapping = target.get('case_id'), target.get('target_row_id'), target.get('donor_row_ids')
        if (type(case_id) is not str or not case_id or case_id in seen or target_id not in by_id
                or type(mapping) is not dict or set(mapping) != {'A_same', 'A_opposite', 'B_same', 'B_opposite'}):
            raise ValueError('invalid target or donor mapping')
        seen.add(case_id)
        target_row = by_id[target_id]
        if (target_row['subject_source'] != 'A'
                or (target_row['subject_number'], target_row['distractor_number']) != factors
                or mapping['A_same'] != target_id):
            raise ValueError('target order or identity is invalid')
        for key, source in (('A_same', 'A'), ('A_opposite', 'A'), ('B_same', 'B'), ('B_opposite', 'B')):
            donor = by_id.get(mapping[key])
            want_number = target_row['subject_number']
            if key.endswith('opposite'):
                want_number = 'plural' if want_number == 'singular' else 'singular'
            if (donor is None or donor['subject_source'] != source
                    or donor['subject_number'] != want_number
                    or donor['distractor_number'] != target_row['distractor_number']
                    or donor['distractor_lemma'] != target_row['distractor_lemma']
                    or donor['input_ids'][:2] != target_row['input_ids'][:2]
                    or donor['input_ids'][3:] != target_row['input_ids'][3:]):
                raise ValueError('donor does not match the declared factorial cell')
    return rows, targets, by_id


def run_family(model, manifest, *, split, family, is_id, are_id, mask_id, emit):
    """Run one frozen family on CPU; emit every started/completed forward.

    Caller authenticates the complete manifest, checkpoint and execution mode,
    owns the model exclusively, and supplies a bounded durable event sink. The
    return value is published only after all 140 forwards and controls pass.
    An exception or sink failure makes the attempt incomplete.
    """
    return _run_family(model, manifest, split=split, family=family, is_id=is_id,
                       are_id=are_id, mask_id=mask_id, emit=emit, observer=run_observation)


def _run_family(model, manifest, *, split, family, is_id, are_id, mask_id, emit, observer):
    """Run the same schedule with a trusted, internally supplied observer.

    This private seam does not authenticate an observer or establish device,
    checkpoint, resource, or evidence custody. The public entry remains CPU.
    """
    if not callable(emit):
        raise ValueError('a forward event sink is required')
    if not callable(observer):
        raise ValueError('a trusted observation function is required')
    rows, targets, by_id = _family_rows(manifest, split, family, mask_id)
    sites = SITES
    if len(model.distilbert.transformer.layer) != 6:
        raise ValueError('the frozen study requires exactly six encoder blocks')
    references, baselines, by_case, controls = {}, {}, {}, []
    count = 0

    def forward(kind, row_id, site_name, layer, token, *, donor_id=None, source=None, self_patch=False):
        nonlocal count
        count += 1
        if count > 140:
            raise RuntimeError('family forward budget exceeded')
        record = {'split': split, 'family': family, 'forward_index': count, 'kind': kind, 'row_id': row_id,
                  'donor_row_id': donor_id, 'site': site_name, 'layer': layer,
                  'token': token}
        emit({'status': 'started', **record})
        try:
            inputs = torch.tensor([by_id[row_id]['input_ids']], dtype=torch.int64, device='cpu')
            observation = observer(model, inputs, layer=layer, token=token,
                                   mask_token=6, is_id=is_id, are_id=are_id,
                                   source=source, self_patch=self_patch)
            if type(observation) is not Observation:
                raise TypeError('observer returned an unsupported observation')
            if (observation.model_input_ids != tuple(by_id[row_id]['input_ids'])
                    or observation.model_attention_mask != tuple(by_id[row_id]['attention_mask'])):
                raise RuntimeError('model input values differ from the declared row')
        except BaseException as error:
            try:
                emit({'status': 'failed', **record, 'error_type': type(error).__name__})
            except (KeyboardInterrupt, SystemExit) as cancellation:
                raise cancellation from error
            except Exception:
                pass
            raise
        raw_before = float32_bytes(observation.before)
        raw_after = None if observation.after is None else float32_bytes(observation.after)
        emit({'status': 'completed', **record, 'is_bits': observation.is_bits,
              'are_bits': observation.are_bits, 'is_logit': observation.is_logit,
              'are_logit': observation.are_logit,
              'mask_logits_sha256': hashlib.sha256(float32_bytes(observation.mask_logits)).hexdigest(),
              'before_float32_le_base64': base64.b64encode(raw_before).decode('ascii'),
              'after_float32_le_base64': None if raw_after is None else base64.b64encode(raw_after).decode('ascii'),
              'events': list(observation.events), 'capture_bytes': observation.capture_bytes,
              'source_bytes': observation.source_bytes,
              'model_input_ids': list(observation.model_input_ids),
              'model_attention_mask': list(observation.model_attention_mask)})
        return observation

    def check(name, case, site_name, value, reference):
        passed = float32_bytes(value) == float32_bytes(reference)
        record = {'status': 'control', 'split': split, 'family': family,
                  'name': name, 'case_id': case, 'site': site_name, 'passed': passed}
        controls.append(record)
        emit(record)
        if not passed:
            raise RuntimeError('required agreement control failed: ' + name)

    for site_name, layer, token in sites:
        donors = {}
        for row in rows:
            row_id = row['row_id']
            observation = forward('donor_capture', row_id, site_name, layer, token)
            donors[row_id] = observation.before
            if row['subject_source'] == 'A':
                if site_name == 'candidate':
                    references[row_id] = observation.mask_logits
                    baselines[row_id] = (observation.is_bits, observation.are_bits)
                else:
                    check('clean_matches_candidate', row_id, site_name, observation.mask_logits, references[row_id])
        for target in targets:
            case, target_id = target['case_id'], target['target_row_id']
            cells = {}
            for cell in ('A_same', 'A_opposite', 'B_same', 'B_opposite'):
                donor_id = target['donor_row_ids'][cell]
                observation = forward('patch_' + cell, target_id, site_name, layer, token,
                                      donor_id=donor_id,
                                      source=None if cell == 'A_same' else donors[donor_id],
                                      self_patch=cell == 'A_same')
                if float32_bytes(observation.before) != float32_bytes(donors[target_id]):
                    raise RuntimeError('target pre-action vector differs from its clean capture')
                if cell == 'A_same':
                    check('self_patch', case, site_name, observation.mask_logits, references[target_id])
                cells[cell.lower()] = (observation.is_bits, observation.are_bits)
            effect = signed_donor_effect(by_id[target_id]['subject_number'], cells)
            by_case.setdefault(case, {})[site_name] = effect

    final_sources = {'final_subject': {}, 'final_mask': {}}
    final_before = {'final_subject': {}, 'final_mask': {}}
    for target in targets:
        case, target_id = target['case_id'], target['target_row_id']
        donor_id = target['donor_row_ids']['A_opposite']
        for name, token in (('final_subject', 2), ('final_mask', 6)):
            donor = forward('delivery_donor', donor_id, name, 5, token)
            patched = forward('delivery_patch', target_id, name, 5, token,
                              donor_id=donor_id, source=donor.before)
            final_sources[name][donor_id] = float32_bytes(donor.before)
            final_before[name][target_id] = float32_bytes(patched.before)
            reference = references[target_id] if name == 'final_subject' else donor.mask_logits
            check(name, case, name, patched.mask_logits, reference)
        fresh = forward('final_clean', target_id, 'candidate', 2, 2)
        check('final_clean', case, 'candidate', fresh.mask_logits, references[target_id])
    if any(final_sources[name] != final_before[name] for name in final_sources):
        raise RuntimeError('final-block pre-action vectors differ from clean captures')
    if count != 140 or any(len(by_case.get(target['case_id'], {})) != 5 for target in targets):
        raise RuntimeError('family execution did not complete its fixed inventory')
    cases = [{'case_id': target['case_id'], 'target_row_id': target['target_row_id'],
              'baseline_is_bits': baselines[target['target_row_id']][0],
              'baseline_are_bits': baselines[target['target_row_id']][1],
              'effects': by_case[target['case_id']]} for target in targets]
    means = {name: math.fsum(case['effects'][name] for case in cases) / 4.0
             for name, _, _ in sites}
    return {'split': split, 'family': family, 'forward_count': count,
            'cases': cases, 'site_means': means, 'controls': controls}
