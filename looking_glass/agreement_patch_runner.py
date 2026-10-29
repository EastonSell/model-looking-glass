"""Natural agreement patching on a trusted, exclusively owned eager model.

This low-level development executor takes already validated, public case rows;
it does not generate held-out cases, authenticate a preflight, launch a GPU
owner, or publish an authoritative study. The caller must supply those custody
and publication steps. CUDA is deliberately refused until the supervised,
measured-resource adapter is wired and independently reviewed.

This executor seeds and restores PyTorch's process-global CPU RNG. Use a
dedicated trusted process; ModelLease does not serialize unrelated threads.
When preserving an earlier failure, secondary cleanup or evidence exceptions,
including interruptions, are diagnostic rather than replacing that failure.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import sys
import threading

import torch

from .agreement_journal import AgreementJournal
from .agreement_metrics import score_factorial
from .agreement_numeric import compare_f32_le, decode_f32_le
from .distilgpt2_intervention_dev import _cleanup_status, _hook_state
from .intervention_plan import ActivationPlan
from .model_ownership import ModelLease


class StudyFailure(RuntimeError):
    """The natural track cannot be completed under its declared contract."""


class StudyCancelled(StudyFailure):
    """The owner cancelled before another forward could be published."""


class CancellationGate:
    """Serialize cancellation with the start of candidate completion."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancelled = False
        self._committed = False

    def cancel(self):
        with self._lock:
            if self._committed:
                return False
            self._cancelled = True
            return True

    def is_cancelled(self):
        with self._lock:
            return self._cancelled

    def commit(self):
        with self._lock:
            if self._cancelled or self._committed:
                return False
            self._committed = True
            return True


@dataclass(frozen=True)
class AgreementCase:
    case_id: str
    quartet: str
    lemma: str
    number: str
    input_ids: tuple
    subject_index: int


def _site(layer, component, token_role='last'):
    path = f'transformer.h.{layer}' + ('' if component == 'block' else '.' + component)
    return {'id': f'{layer}:{component}:{token_role}', 'module_path': path, 'layer': layer,
            'component': component, 'token_role': token_role,
            'output_selector': [0] if component in ('attn', 'block') else []}


_CANDIDATES = tuple(_site(layer, component) for layer in (1, 2, 3, 4)
                    for component in ('attn', 'mlp'))
_DELIVERY = _site(5, 'block')
_SUBJECT = _site(5, 'block', 'subject')
_SITES = _CANDIDATES + (_DELIVERY, _SUBJECT)
_TARGETS = (318, 389, 373, 547)
_RNG_SEED = 20261028


def _valid_identifier(value):
    return type(value) is str and bool(value) and len(value) <= 128


def _check_cpu_precision():
    try:
        enabled = torch.is_autocast_enabled('cpu')
    except TypeError:  # PyTorch versions predating the device-type argument.
        enabled = torch.is_autocast_cpu_enabled()
    if enabled:
        raise StudyFailure('CPU autocast is unsupported')


_CONFIG_RUNTIME_FIELDS = (
    'use_cache', '_attn_implementation', 'output_attentions', 'output_hidden_states',
    'use_return_dict', 'return_dict', 'add_cross_attention', 'n_layer', 'n_head',
    'n_embd', 'vocab_size', 'n_positions', 'n_ctx', 'activation_function',
    'layer_norm_epsilon', 'scale_attn_weights', 'scale_attn_by_inverse_layer_idx',
    'reorder_and_upcast_attn', 'tie_word_embeddings', 'model_type',
)
_MODULE_RUNTIME_FIELDS = (
    'scale_attn_weights', 'scale_attn_by_inverse_layer_idx', 'reorder_and_upcast_attn',
    'layer_idx', 'is_cross_attention', 'is_causal', 'embed_dim', 'num_heads',
    'head_dim', 'split_size', 'pruned_heads', 'model_parallel', 'gradient_checkpointing',
    '_attn_implementation', 'p', 'inplace', 'eps', 'normalized_shape',
    'elementwise_affine', 'approximate', 'negative_slope', 'padding_idx', 'nf',
)
_MISSING_RUNTIME_FIELD = object()


def _runtime_value(value):
    """Snapshot supported plain values without equating booleans and integers."""
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ('float', value.hex())
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_runtime_value(item) for item in value))
    if type(value) in (set, frozenset) and all(type(item) is int for item in value):
        return (type(value).__name__, tuple(sorted(value)))
    raise StudyFailure('unsupported model execution-configuration value')


def _runtime_fields(obj, names):
    result = []
    for name in names:
        value = getattr(obj, name, _MISSING_RUNTIME_FIELD)
        if value is not _MISSING_RUNTIME_FIELD:
            result.append((name, _runtime_value(value)))
    return tuple(result)


def _global_runtime_configuration():
    # A normal float32 input produces an exactly representable subnormal when
    # multiplied by 0.5. Integer reinterpretation reads the result without an
    # FTZ-sensitive float32-to-float64 conversion. This probes only the calling
    # thread, not every BLAS or worker thread's floating-point control state.
    normal_bits = torch.tensor([0x00800000], dtype=torch.int32, device='cpu')
    product_bits = int((normal_bits.view(torch.float32) * 0.5).view(torch.int32).item())
    if product_bits not in (0, 0x00400000):
        raise StudyFailure('unsupported CPU denormal probe result')
    return (('deterministic_algorithms', torch.are_deterministic_algorithms_enabled()),
            ('deterministic_warn_only', torch.is_deterministic_algorithms_warn_only_enabled()),
            ('cpu_threads', torch.get_num_threads()),
            ('cpu_interop_threads', torch.get_num_interop_threads()),
            ('float32_matmul_precision', torch.get_float32_matmul_precision()),
            ('calling_thread_denormal_product_bits', product_bits))


def _runtime_configuration(model):
    """Observe declared GPT-2 execution flags; this is not a hostile-code defense."""
    modules = []
    for path, module in model.named_modules(remove_duplicate=False):
        fields = _runtime_fields(module, _MODULE_RUNTIME_FIELDS)
        config = getattr(module, 'config', None)
        config_fields = _runtime_fields(config, _CONFIG_RUNTIME_FIELDS) if config is not None else ()
        modules.append((path, fields, config_fields))
    return (_global_runtime_configuration(), tuple(modules))


def _model_contract(model):
    """Validate the supported execution configuration while ownership is held."""
    _check_cpu_precision()
    if not isinstance(model, torch.nn.Module):
        raise StudyFailure('expected a torch module')
    if any(module.training is not False for module in model.modules()):
        raise StudyFailure('model evaluation mode changed or is unsupported')
    if (len(model.transformer.h) != 6 or model.config.use_cache is not False
            or model.config._attn_implementation != 'eager'
            or type(model.config.n_embd) is not int or model.config.n_embd <= 0
            or type(model.config.vocab_size) is not int or model.config.vocab_size <= max(_TARGETS)):
        raise StudyFailure('expected a six-block eager eval model without cache')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise StudyFailure('model parameters must be CPU float32')
    if any(b.device.type != 'cpu' or (b.is_floating_point() and b.dtype != torch.float32)
           for b in model.buffers()):
        raise StudyFailure('model buffers must be on CPU with float32 floating buffers')
    return (model.config.n_embd, model.config.vocab_size, _runtime_configuration(model))


def _error_message(error):
    try:
        return str(error)[:1000]
    except BaseException:
        return '<exception message unavailable>'


def _validate_cases(cases, expected_quartets, vocab_size):
    if (type(cases) not in (tuple, list) or len(cases) != 4 * expected_quartets
            or type(expected_quartets) is not int or not 1 <= expected_quartets <= 16):
        raise StudyFailure('case set does not have the declared quartet count')
    groups, identifiers, inputs = {}, set(), set()
    for item in cases:
        if (type(item) is not AgreementCase
                or not all(_valid_identifier(v) for v in (item.case_id, item.quartet, item.lemma))
                or item.case_id in identifiers or item.number not in ('singular', 'plural')
                or type(item.input_ids) is not tuple or not 2 <= len(item.input_ids) <= 32
                or any(type(token) is not int or token < 0 or token >= vocab_size for token in item.input_ids)
                or item.input_ids in inputs
                or type(item.subject_index) is not int or not 0 <= item.subject_index < len(item.input_ids) - 1):
            raise StudyFailure('invalid or duplicate case identity or token coordinates')
        identifiers.add(item.case_id)
        inputs.add(item.input_ids)
        groups.setdefault(item.quartet, []).append(item)
    if len(groups) != expected_quartets:
        raise StudyFailure('unexpected number of case quartets')
    for members in groups.values():
        if (len(members) != 4 or len({item.lemma for item in members}) != 2
                or len({(item.lemma, item.number) for item in members}) != 4
                or len({len(item.input_ids) for item in members}) != 1
                or len({item.subject_index for item in members}) != 1):
            raise StudyFailure('quartet is not a complete aligned two-by-two')
        subject_index = members[0].subject_index
        contexts = {item.input_ids[:subject_index] + item.input_ids[subject_index + 1:] for item in members}
        subjects = {item.input_ids[subject_index] for item in members}
        if len(contexts) != 1 or len(subjects) != 4:
            raise StudyFailure('quartet differs outside its unique subject token')
    return groups


def _input_sha(case):
    data = json.dumps({'ids': case.input_ids, 'subject_index': case.subject_index},
                      sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _payload(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
            or tensor.device.type != 'cpu' or sys.byteorder != 'little'):
        raise StudyFailure('expected CPU little-endian float32 output')
    # Copy integer bits rather than float values, including under FTZ.
    raw = tensor.detach().view(torch.int32).contiguous().numpy().tobytes()
    decode_f32_le(raw)
    return raw


def _state_hash(model):
    digest = hashlib.sha256()
    for group, entries in (('state_dict', model.state_dict().items()),
                           ('all_buffers', model.named_buffers(remove_duplicate=False))):
        for name, tensor in entries:
            if tensor.device.type != 'cpu' or not tensor.is_contiguous():
                raise StudyFailure('model state must be contiguous and on the declared device')
            header = json.dumps({'group': group, 'name': name, 'shape': list(tensor.shape),
                                 'dtype': str(tensor.dtype)}, sort_keys=True,
                                separators=(',', ':')).encode('utf-8')
            digest.update(len(header).to_bytes(4, 'big'))
            digest.update(header)
            raw = tensor.detach().reshape(-1).view(torch.uint8)
            for offset in range(0, raw.numel(), 1024 * 1024):
                digest.update(memoryview(raw[offset:offset + 1024 * 1024].numpy()))
    return digest.hexdigest()


def _state_identity(model):
    """Hold parameter/buffer objects and their original storage identities."""
    entries = []
    for group, named in (('parameter', model.named_parameters(remove_duplicate=False)),
                         ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, tensor in named:
            storage = tensor.untyped_storage()
            details = (group, name, storage.data_ptr(), storage.nbytes(), tensor.storage_offset(),
                       tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device),
                       tensor.requires_grad)
            # Keep both references alive so allocator reuse cannot mimic identity.
            entries.append((tensor, storage, details))
    return tuple(entries)


def _same_state_identity(baseline, model):
    current = _state_identity(model)
    return (len(baseline) == len(current)
            and all(before[0] is after[0] and before[2] == after[2]
                    for before, after in zip(baseline, current)))


def _scores(raw):
    if len(raw) % 4:
        raise StudyFailure('invalid vocabulary payload')
    values = []
    for index in _TARGETS:
        if (index + 1) * 4 > len(raw):
            raise StudyFailure('contrast token is outside vocabulary')
        values.append(float(decode_f32_le(raw[4 * index:4 * index + 4])[0]))
    return tuple(values)


def _tensor_from_bytes(raw):
    # Reconstruct from integer bytes, not FTZ-sensitive float32 conversion.
    return torch.frombuffer(bytearray(raw), dtype=torch.int32).clone().view(torch.float32)


def _vector_l2(left, right):
    """Calculate the protocol's binary64 norm from stored float32 bits."""
    if len(left) != len(right) or not left:
        raise StudyFailure('magnitude vectors differ in size')
    a, b = decode_f32_le(left), decode_f32_le(right)
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


def _pair_with_signs(pair):
    clean, patched = pair['clean_margin'], pair['patched_margin']
    clean_sign = int(clean > 0) - int(clean < 0)
    patched_sign = int(patched > 0) - int(patched < 0)
    return {**pair, 'clean_margin_sign': clean_sign, 'patched_margin_sign': patched_sign,
            'strict_sign_reversal': clean_sign * patched_sign == -1,
            'zero_transition': clean_sign != patched_sign and (clean_sign == 0 or patched_sign == 0)}


def _label_past_pair(pair):
    """Name the generic scorer's first/second logits by their actual tokens."""
    result = dict(pair)
    for original, actual in (('clean_is', 'clean_was'), ('clean_are', 'clean_were'),
                             ('patched_is', 'patched_was'), ('patched_are', 'patched_were')):
        result[actual] = result.pop(original)
    return result


def _discovery_selection(metrics, magnitudes, expected_quartets):
    """Freeze protocol-v1 ranking and baselines for complete discovery data."""
    site_ids = [site['id'] for site in _CANDIDATES]
    summaries = metrics['site_summaries']
    if len(summaries) != len(site_ids) or [row['site'] for row in summaries] != site_ids:
        raise StudyFailure('site summaries do not match protocol order')
    predictions = {row['site']: row['mean_number_main_contrast'] for row in summaries}
    if any(type(value) is not float or not math.isfinite(value) for value in predictions.values()):
        raise StudyFailure('site predictions must be finite')
    by_site = {site: [] for site in site_ids}
    seen = set()
    for record in magnitudes:
        if type(record) is not dict or set(record) != {'site', 'quartet', 'lemma', 'singular', 'plural', 'l2'}:
            raise StudyFailure('invalid magnitude record')
        key = (record['site'], record['quartet'], record['lemma'])
        if (record['site'] not in by_site or key in seen
                or not all(_valid_identifier(record[k]) for k in ('quartet', 'lemma', 'singular', 'plural'))
                or type(record['l2']) is not float or not math.isfinite(record['l2']) or record['l2'] < 0):
            raise StudyFailure('invalid or duplicate magnitude record')
        seen.add(key)
        by_site[record['site']].append(record['l2'])
    if any(len(values) != 2 * expected_quartets for values in by_site.values()):
        raise StudyFailure('incomplete magnitude baseline')
    family_pairs = [{(row['quartet'], row['lemma'], row['singular'], row['plural'])
                     for row in magnitudes if row['site'] == site} for site in site_ids]
    first = family_pairs[0]
    if (len(first) != 2 * expected_quartets or len({row[0] for row in first}) != expected_quartets
            or any(sum(1 for row in first if row[0] == quartet) != 2 for quartet in {row[0] for row in first})
            or any(pairs != first for pairs in family_pairs[1:])):
        raise StudyFailure('magnitude baselines do not use the same complete families')
    means = {site: math.fsum(by_site[site]) / (2 * expected_quartets) for site in site_ids}
    ranked = sorted(site_ids, key=lambda site: -predictions[site])  # stable protocol order resolves ties
    magnitude_ranked = sorted(site_ids, key=lambda site: -means[site])
    random_index = hashlib.sha256(b'MLG-number-v1-random\x00' + _RNG_SEED.to_bytes(8, 'big')).digest()[0] % len(site_ids)
    return {'ranking': ranked, 'predictions': predictions,
            'selected_site': ranked[0] if predictions[ranked[0]] > 0 else None,
            'earliest_site': site_ids[0], 'random_site': site_ids[random_index],
            'random_index': random_index, 'magnitude_site': magnitude_ranked[0],
            'magnitude_means': means, 'magnitude_pairs': magnitudes}


def _steps_for_capture(case):
    steps = []
    for index, site in enumerate(_SITES):
        token = case.subject_index if site['token_role'] == 'subject' else len(case.input_ids) - 1
        steps.append({'name': f'capture_{index}', 'operation': 'capture',
                      'module_path': site['module_path'], 'invocation': 0,
                      'batch_index': 0, 'token_index': token})
    return steps


def _steps_for_action(case, site, operation, donor=None):
    token = case.subject_index if site['token_role'] == 'subject' else len(case.input_ids) - 1
    common = {'module_path': site['module_path'], 'invocation': 0,
              'batch_index': 0, 'token_index': token}
    action = {'name': 'action', 'operation': operation, **common}
    if operation == 'patch':
        action.update({'source_capture': 'before'} if donor is None else {'value': donor})
    whole = {'module_path': site['module_path'], 'invocation': 0}
    return [{'name': 'whole_before', 'operation': 'capture', **whole},
            {'name': 'before', 'operation': 'capture', **common}, action,
            {'name': 'after', 'operation': 'capture', **common},
            {'name': 'whole_after', 'operation': 'capture', **whole}]


def _unchanged_outside_target(values, case, site, hidden_size):
    token = case.subject_index if site['token_role'] == 'subject' else len(case.input_ids) - 1
    start = 4 * hidden_size * token
    end = start + 4 * hidden_size
    before, after = values['whole_before'], values['whole_after']
    return (len(before) == len(after) == 4 * hidden_size * len(case.input_ids)
            and before[start:end] == values['before'] and after[start:end] == values['after']
            and before[:start] == after[:start] and before[end:] == after[end:])


class NaturalAgreementRunner:
    """Run complete natural-patching quartets under one process-local lease."""

    def __init__(self, model, journal, *, device='cpu', expected_quartets=4, cancel_event=None):
        if type(journal) is not AgreementJournal:
            raise ValueError('an actual bounded agreement journal is required')
        if device != 'cpu':
            raise ValueError('CUDA requires a separately reviewed supervised resource adapter')
        if type(expected_quartets) is not int or not 1 <= expected_quartets <= 16:
            raise ValueError('invalid expected quartet count')
        self.model, self.journal, self.device = model, journal, device
        if cancel_event is not None and type(cancel_event) is not CancellationGate:
            raise ValueError('cancellation must use a CancellationGate')
        self.expected_quartets = expected_quartets
        self.cancel_event = cancel_event if cancel_event is not None else CancellationGate()
        self._forwards = 0
        self._forwards_started = 0
        self._model_returns = 0
        self._forwards_recorded = 0
        self._failure_evidence_error = None
        self._pending_forward = None
        self._used = False
        self._started_work = False
        self.quarantined = False
        self._baseline_hooks = None
        self._baseline_contract = None
        self._baseline_state_identity = None
        self._lease = None

    def _cancel_check(self):
        if self.cancel_event.is_cancelled():
            raise StudyCancelled('study cancelled')

    # Device-specific implementations must establish ownership and settle all
    # work before returning. The public base class remains strictly CPU-only.
    def _execution_contract(self):
        return _model_contract(self.model)

    def _model_digest(self):
        return _state_hash(self.model)

    def _encode_payload(self, tensor):
        return _payload(tensor)

    def _prepare_steps(self, steps):
        return steps

    def _new_plan(self, steps):
        return ActivationPlan(self.model, steps, max_tensor_bytes=1024 * 1024,
                              max_capture_bytes=512 * 1024, max_source_bytes=128 * 1024,
                              max_steps=16, max_invocations=1, _parent_lease=self._lease)

    def _root_forward(self, name, value):
        pass

    def _finish_forward_roots(self):
        pass

    def _make_inputs(self, case):
        ids = torch.tensor([case.input_ids], dtype=torch.int64, device='cpu')
        return ids, torch.ones_like(ids)

    def _before_forward(self):
        pass

    def _after_forward(self):
        pass

    def _failed_forward(self, error):
        pass

    def _seed_device_rng(self):
        return None

    def _before_rng_cleanup(self, primary_error):
        """Let a device backend quarantine before cleanup after a failure."""
        pass

    def _restore_device_rng(self):
        pass

    def _runtime_metadata(self):
        return {'device': self.device}

    def _release_after_work(self, lease, initial_state, primary_error):
        """Keep model ownership if post-failure cleanup is changed or unknown."""
        unsafe = self.quarantined
        inspection_error = None
        if self._started_work:
            try:
                lease.verify_tree()
                unsafe = (unsafe or initial_state is None or self._model_digest() != initial_state
                          or _hook_state(self.model) != self._baseline_hooks
                          or self._execution_contract() != self._baseline_contract
                          or not _same_state_identity(self._baseline_state_identity, self.model))
            except BaseException as error:
                unsafe = True
                inspection_error = error
        if unsafe:
            self.quarantined = True
            try:
                lease._quarantine()
            except BaseException:
                pass  # Retain the lease even if the explicit latch failed.
            if primary_error is not None:
                if hasattr(primary_error, 'add_note'):
                    primary_error.add_note('Model cleanup is uncertain; retain ownership and discard the worker.')
                return
            if inspection_error is not None:
                raise inspection_error
            raise StudyFailure('model cleanup is uncertain; worker quarantined')
        try:
            lease.release()
            self._lease = None
        except BaseException as cleanup_error:
            self.quarantined = True
            if primary_error is not None:
                if hasattr(primary_error, 'add_note'):
                    primary_error.add_note('Model lease release failed; discard the worker: '
                                           + type(cleanup_error).__name__)
                return
            raise

    def _forward(self, case, steps):
        self._cancel_check()
        plan = None
        ids, mask = None, None
        output, captures, metadata = None, None, None
        started_here = False
        try:
            self._before_forward()
            if self._execution_contract() != self._baseline_contract:
                raise StudyFailure('model execution contract changed before forward')
            if _hook_state(self.model) != self._baseline_hooks:
                raise StudyFailure('unexpected hook before forward')
            self._lease.verify_tree()
            ids, mask = self._make_inputs(case)
            self._root_forward('inputs', (ids, mask))
            steps = self._prepare_steps(steps)
            self._root_forward('steps', steps)
            with torch.inference_mode():
                if steps:
                    plan = self._new_plan(steps)
                    self._root_forward('plan', plan)
                    with plan:
                        self._started_work = True
                        started_here = True
                        self._forwards_started += 1
                        output = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                        self._root_forward('output', output)
                        self._model_returns += 1
                    captures = plan.captures
                    self._root_forward('captures', captures)
                    metadata = plan.metadata
                else:
                    self._started_work = True
                    started_here = True
                    self._forwards_started += 1
                    output = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                    self._root_forward('output', output)
                    self._model_returns += 1
                    captures, metadata = {}, {'steps': []}
            if self._execution_contract() != self._baseline_contract:
                raise StudyFailure('model execution contract changed after forward')
            logits = output.logits
            if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
                    or str(logits.device) != self.device
                    or tuple(logits.shape) != (1, len(case.input_ids), self.model.config.vocab_size)
                    or getattr(output, 'past_key_values', None) is not None):
                raise StudyFailure('unexpected output shape, dtype, device or cache')
            if len(metadata['steps']) != len(steps or ()):
                raise StudyFailure('incomplete plan metadata')
            for index, (event, requested) in enumerate(zip(metadata['steps'], steps or ())):
                site = next((item for item in _SITES if item['module_path'] == requested['module_path']), None)
                full_output = requested.get('token_index') is None
                expected_shape = ([1, len(case.input_ids), self.model.config.n_embd]
                                  if full_output else [self.model.config.n_embd])
                prior = [step['name'] for step in (steps or ())[:index]
                         if step['module_path'] == requested['module_path']
                         and step.get('invocation', 0) == requested.get('invocation', 0)]
                if (site is None or event['name'] != requested['name']
                        or event['operation'] != requested['operation'] or event['order'] != index
                        or event['module_path'] != requested['module_path'] or event['invocation'] != 0
                        or event['batch_index'] != requested.get('batch_index')
                        or event['token_index'] != requested.get('token_index')
                        or event['forward_index'] != 0 or event['output_selector'] != site['output_selector']
                        or event['output_shape'] != [1, len(case.input_ids), self.model.config.n_embd]
                        or event['selected_shape'] != expected_shape
                        or event['dtype'] != 'torch.float32' or event['device'] != self.device
                        or event['source_capture'] != requested.get('source_capture')
                        or event['preceding_same_site_actions'] != prior):
                    raise StudyFailure('plan metadata does not match its requested coordinate')
            expected_captures = {step['name']: ([1, len(case.input_ids), self.model.config.n_embd]
                                                 if step.get('token_index') is None
                                                 else [self.model.config.n_embd])
                                 for step in (steps or ()) if step['operation'] == 'capture'}
            if set(captures) != set(expected_captures) or any(
                    list(captures[name].shape) != shape for name, shape in expected_captures.items()):
                raise StudyFailure('captured activation shape does not match metadata')
            encoded = {name: self._encode_payload(value) for name, value in captures.items()}
            if any(len(value) != 4 * math.prod(expected_captures[name]) for name, value in encoded.items()):
                raise StudyFailure('captured activation has the wrong size')
            last_logits = self._encode_payload(logits[0, -1])
            self._after_forward()
            self._forwards += 1
            self._lease.verify_tree()
            if _hook_state(self.model) != self._baseline_hooks:
                raise StudyFailure('hook registry changed after forward')
            self._pending_forward = (case, last_logits, encoded, metadata)
            self._finish_forward_roots()
            return last_logits, encoded, metadata
        except BaseException as error:
            # A device call may have returned into a local before an interrupt
            # prevented its normal custody-slot assignment. Retain every known
            # local before attempting any synchronization or cleanup.
            for name, value in (('ids', ids), ('mask', mask), ('steps', steps),
                                ('plan', plan), ('output', output), ('captures', captures)):
                if value is not None:
                    try:
                        self._root_forward(name, value)
                    except BaseException as root_error:
                        self.quarantined = True
                        if hasattr(error, 'add_note'):
                            error.add_note('Forward custody is uncertain: ' + type(root_error).__name__)
            try:
                self._failed_forward(error)
            except BaseException as cleanup_error:
                self.quarantined = True
                if hasattr(error, 'add_note'):
                    error.add_note('Device completion is unknown; discard the worker: '
                                   + type(cleanup_error).__name__)
            _cleanup_status(self.model, self._baseline_hooks, error)
            if started_here:
                self._record_rejected(case, output, captures, metadata, error)
            raise

    def _record_rejected(self, case, output, captures, metadata, primary_error):
        """Best-effort bounded evidence from a rejected or failed forward."""
        tensors, unavailable = {}, []
        if isinstance(captures, dict):
            for name, tensor in captures.items():
                try:
                    tensors[name] = (tuple(tensor.shape), self._encode_payload(tensor))
                except BaseException as error:
                    unavailable.append('capture ' + str(name) + ': ' + type(error).__name__)
        else:
            unavailable.append('plan did not publish captures')
        try:
            logits = getattr(output, 'logits', None)
            if (isinstance(logits, torch.Tensor) and logits.ndim == 3 and logits.shape[0] == 1
                    and logits.shape[1] > 0):
                selected = logits[0, -1]
                tensors['last_logits'] = (tuple(selected.shape), self._encode_payload(selected))
            else:
                unavailable.append('last logits unavailable')
        except BaseException as error:
            unavailable.append('last logits: ' + type(error).__name__)
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            metadata = None
            unavailable.append('plan metadata was not finite JSON')
        try:
            self.journal.record({'kind': 'rejected_forward', 'case': case.case_id,
                                 'input_sha256': _input_sha(case), 'error_type': type(primary_error).__name__,
                                 'plan_metadata': metadata, 'unavailable': unavailable}, tensors)
            self._forwards_recorded += 1
        except BaseException as error:
            self._failure_evidence_error = type(error).__name__
            if hasattr(primary_error, 'add_note'):
                primary_error.add_note('Rejected-forward evidence could not be saved: ' + type(error).__name__)

    def _record(self, event, last_logits, vectors, *, full_logits):
        payloads = {}
        for name, value in vectors.items():
            shape = ((1, len(value) // (4 * self.model.config.n_embd), self.model.config.n_embd)
                     if name in ('whole_before', 'whole_after') else (self.model.config.n_embd,))
            payloads[name] = (shape, value)
        if full_logits:
            payloads['last_logits'] = ((self.model.config.vocab_size,), last_logits)
        else:
            payloads['contrast_logits'] = ((4,), b''.join(last_logits[4 * i:4 * i + 4] for i in _TARGETS))
        refs = self.journal.record(event, payloads)
        self._forwards_recorded += 1
        self._pending_forward = None
        # Publish the completed forward before honoring cancellation observed
        # during that forward; never drop an already obtained observation.
        self._cancel_check()
        return refs

    def _record_pending_rejected(self, primary_error):
        """Persist a validated forward rejected by later checks, if still possible."""
        pending = self._pending_forward
        if pending is None:
            return
        case, logits, captures, metadata = pending
        tensors = {}
        try:
            hidden_size = self._baseline_contract[0]
            for name, value in captures.items():
                shape = ((1, len(value) // (4 * hidden_size), hidden_size)
                         if name in ('whole_before', 'whole_after') else (hidden_size,))
                tensors[name] = (shape, value)
            tensors['last_logits'] = ((len(logits) // 4,), logits)
            self.journal.record({'kind': 'rejected_forward', 'case': case.case_id,
                                 'input_sha256': _input_sha(case),
                                 'error_type': type(primary_error).__name__,
                                 'plan_metadata': metadata,
                                 'unavailable': []}, tensors)
            self._forwards_recorded += 1
            self._pending_forward = None
        except BaseException as error:
            self._failure_evidence_error = type(error).__name__
            if hasattr(primary_error, 'add_note'):
                primary_error.add_note('Pending-forward evidence could not be saved: ' + type(error).__name__)

    def run(self, cases, *, mode='discovery', frozen_predictions=None, heldout_binding=None):
        if self._used:
            raise StudyFailure('a runner is single-use')
        self._used = True
        controls = {'null': 0, 'self': 0, 'delivery': 0, 'subject_position': 0,
                    'repeat': 0, 'final_clean': 0}
        try:
            self._cancel_check()
            if mode not in ('discovery', 'heldout') or type(mode) is not str:
                raise StudyFailure('invalid natural study mode')
            if mode == 'discovery':
                if frozen_predictions is not None or heldout_binding is not None:
                    raise StudyFailure('discovery does not accept held-out predictions')
                frozen = frozen_selection = None
            else:
                if self.expected_quartets != 4:
                    raise StudyFailure('held-out evaluation requires four families')
                from .agreement_heldout import frozen_selection as load_selection, validate_heldout_binding
                frozen_selection, frozen = load_selection(frozen_predictions)
                frozen_evidence = json.loads(frozen.prediction_bytes)
                heldout_binding = validate_heldout_binding(heldout_binding)
            model = self.model
            metric_cases, metric_patches = [], []
            past_cases, past_patches, magnitudes = [], [], []
            lease = ModelLease(model)
            self._lease = lease  # Retained before acquisition, including interrupted acquisition.
            initial_state = None
            rng_before = None
            try:
                lease.acquire()
                self._cancel_check()
                self._baseline_contract = self._execution_contract()
                groups = _validate_cases(cases, self.expected_quartets, model.config.vocab_size)
                self._baseline_hooks = _hook_state(model)
                if any(any(registry for registry in module[1:]) for module in self._baseline_hooks[0]) or any(self._baseline_hooks[1:]):
                    raise StudyFailure('preexisting local or global hooks are unsupported')
                initial_state = self._model_digest()
                self._baseline_state_identity = _state_identity(model)
                rng_before = torch.random.get_rng_state()
                torch.random.default_generator.manual_seed(_RNG_SEED)
                device_rng = self._seed_device_rng()
                seeded_rng_sha256 = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
                contract = {'kind': 'run_contract', 'cpu_rng_seed': _RNG_SEED,
                            'seeded_cpu_rng_sha256': seeded_rng_sha256,
                            'device_rng': device_rng,
                            'runtime': self._runtime_metadata(),
                            'execution_configuration': self._baseline_contract[2],
                            'execution_configuration_sha256': hashlib.sha256(
                                json.dumps(self._baseline_contract[2], separators=(',', ':'),
                                           ensure_ascii=True).encode('ascii')).hexdigest(),
                            'cpu_threads': torch.get_num_threads(),
                            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                            'cases': [{'case_id': case.case_id, 'quartet': case.quartet,
                                       'lemma': case.lemma, 'number': case.number,
                                       'input_ids': list(case.input_ids),
                                       'attention_mask': [1] * len(case.input_ids),
                                       'subject_index': case.subject_index} for case in cases],
                            'sites': list(_SITES)}
                if mode == 'heldout':
                    contract['phase'] = 'heldout'
                    contract['prediction_sha256'] = hashlib.sha256(frozen.prediction_bytes).hexdigest()
                    contract['prediction_freeze_sha256'] = frozen.freeze.sha256
                    contract['heldout_commitment'] = heldout_binding.commitment_sha256
                    contract['exposure_inventory_sha256'] = heldout_binding.inventory_sha256
                    contract['discovery_event_sha256'] = frozen_evidence['discovery_event_sha256']
                    contract['independent_audit_sha256'] = frozen_evidence['independent_audit_sha256']
                self.journal.record(contract, {})
                for quartet in groups:
                    members = groups[quartet]
                    donors, clean_logits = {}, {}
                    # Capture all source sites in the actual study runtime.
                    for case in members:
                        logits, captures, metadata = self._forward(case, _steps_for_capture(case))
                        event = {'kind': 'clean_capture', 'case': case.case_id, 'quartet': quartet,
                                 'input_sha256': _input_sha(case), 'plan_metadata': metadata}
                        refs = self._record(event, logits, captures, full_logits=True)
                        clean_logits[case.case_id] = refs['last_logits']
                        for index, site in enumerate(_SITES):
                            donors[(case.case_id, site['id'])] = refs[f'capture_{index}']
                        values = _scores(logits)
                        metric_cases.append({'id': case.case_id, 'quartet': quartet, 'lemma': case.lemma,
                                             'number': case.number, 'clean_is': values[0], 'clean_are': values[1]})
                        past_cases.append({'id': case.case_id, 'quartet': quartet, 'lemma': case.lemma,
                                           'number': case.number, 'clean_is': values[2], 'clean_are': values[3]})
                    if mode == 'discovery':
                        for site in _CANDIDATES:
                            for lemma in sorted({case.lemma for case in members}):
                                singular = next(case for case in members if case.lemma == lemma and case.number == 'singular')
                                plural = next(case for case in members if case.lemma == lemma and case.number == 'plural')
                                left = self.journal.read_tensor(donors[(singular.case_id, site['id'])])
                                right = self.journal.read_tensor(donors[(plural.case_id, site['id'])])
                                magnitudes.append({'site': site['id'], 'quartet': quartet, 'lemma': lemma,
                                                   'singular': singular.case_id, 'plural': plural.case_id,
                                                   'l2': _vector_l2(left, right)})
                    for case in members:
                        logits, captures, metadata = self._forward(case, _steps_for_capture(case))
                        checks = {'last_logits': compare_f32_le(self.journal.read_tensor(clean_logits[case.case_id]), logits)}
                        for index, site in enumerate(_SITES):
                            checks[site['id']] = compare_f32_le(
                                self.journal.read_tensor(donors[(case.case_id, site['id'])]), captures[f'capture_{index}'])
                        self._record({'kind': 'clean_repeat', 'case': case.case_id, 'quartet': quartet,
                                      'checks': checks, 'plan_metadata': metadata}, logits, captures, full_logits=True)
                        if not all(check['within_tolerance'] for check in checks.values()):
                            raise StudyFailure('clean repeat differs beyond fixed tolerance')
                        controls['repeat'] += 1
                    for case in members:
                        site = _CANDIDATES[0]
                        logits, values, metadata = self._forward(case, _steps_for_action(case, site, 'null'))
                        checks = {'activation': compare_f32_le(values['before'], values['after']),
                                  'logits': compare_f32_le(self.journal.read_tensor(clean_logits[case.case_id]), logits)}
                        unchanged = _unchanged_outside_target(values, case, site, model.config.n_embd)
                        self._record({'kind': 'null', 'target': case.case_id, 'quartet': quartet,
                                      'site': site, 'checks': checks, 'untouched_output_bits_equal': unchanged,
                                      'plan_metadata': metadata},
                                     logits, values, full_logits=True)
                        if not unchanged or not all(check['within_tolerance'] for check in checks.values()):
                            raise StudyFailure('null control differs beyond fixed tolerance')
                        controls['null'] += 1
                    for site in _CANDIDATES:
                        for target in members:
                            for donor in members:
                                source = self.journal.read_tensor(donors[(donor.case_id, site['id'])])
                                self_patch = target.case_id == donor.case_id
                                value = None if self_patch else _tensor_from_bytes(source)
                                logits, values, metadata = self._forward(
                                    target, _steps_for_action(target, site, 'patch', value))
                                expected = values['before'] if self_patch else source
                                landed = values['after'] == expected
                                unchanged = _unchanged_outside_target(values, target, site, model.config.n_embd)
                                checks = {}
                                if self_patch:
                                    checks = {'activation': compare_f32_le(values['before'], values['after']),
                                              'logits': compare_f32_le(
                                                  self.journal.read_tensor(clean_logits[target.case_id]), logits)}
                                event = {'kind': 'patch', 'target': target.case_id, 'donor': donor.case_id,
                                         'quartet': quartet, 'site': site, 'source_input_sha256': _input_sha(donor),
                                         'target_input_sha256': _input_sha(target), 'self_patch': self_patch,
                                         'source_tensor_sha256': hashlib.sha256(expected).hexdigest(),
                                         'post_action_matches_donor_bits': landed,
                                         'untouched_output_bits_equal': unchanged, 'checks': checks,
                                         'plan_metadata': metadata}
                                self._record(event, logits, values, full_logits=self_patch)
                                if not landed or not unchanged or any(not check['within_tolerance'] for check in checks.values()):
                                    raise StudyFailure('patch landing or self control failed')
                                controls['self'] += int(self_patch)
                                scores = _scores(logits)
                                metric_patches.append({'site': site['id'], 'target': target.case_id,
                                                       'donor': donor.case_id, 'patched_is': scores[0],
                                                       'patched_are': scores[1]})
                                past_patches.append({'site': site['id'], 'target': target.case_id,
                                                     'donor': donor.case_id, 'patched_is': scores[2],
                                                     'patched_are': scores[3]})
                    for target in members:
                        donor = next(item for item in members if item.lemma == target.lemma
                                     and item.number != target.number)
                        for site, control in ((_DELIVERY, 'delivery'), (_SUBJECT, 'subject_position')):
                            source = self.journal.read_tensor(donors[(donor.case_id, site['id'])])
                            logits, values, metadata = self._forward(
                                target, _steps_for_action(target, site, 'patch', _tensor_from_bytes(source)))
                            reference_id = donor.case_id if control == 'delivery' else target.case_id
                            comparison = compare_f32_le(self.journal.read_tensor(clean_logits[reference_id]), logits)
                            landed = values['after'] == source
                            unchanged = _unchanged_outside_target(values, target, site, model.config.n_embd)
                            self._record({'kind': control, 'target': target.case_id, 'donor': donor.case_id,
                                          'quartet': quartet, 'site': site, 'reference': reference_id,
                                          'post_action_matches_donor_bits': landed,
                                          'untouched_output_bits_equal': unchanged, 'logits_check': comparison,
                                          'plan_metadata': metadata}, logits, values, full_logits=True)
                            if not landed or not unchanged or not comparison['within_tolerance']:
                                raise StudyFailure('delivery or subject-position control failed')
                            controls[control] += 1
                    for case in members:
                        logits, captures, metadata = self._forward(case, None)
                        comparison = compare_f32_le(self.journal.read_tensor(clean_logits[case.case_id]), logits)
                        self._record({'kind': 'final_clean', 'case': case.case_id, 'quartet': quartet,
                                      'logits_check': comparison}, logits, captures, full_logits=True)
                        if not comparison['within_tolerance']:
                            raise StudyFailure('final clean control failed')
                        controls['final_clean'] += 1
                final_state = self._model_digest()
                if (initial_state != final_state or _hook_state(model) != self._baseline_hooks
                        or not _same_state_identity(self._baseline_state_identity, model)):
                    raise StudyFailure('model state or hooks changed during the study')
                metrics = score_factorial(metric_cases, metric_patches,
                                          [site['id'] for site in _CANDIDATES],
                                          expected_quartets=self.expected_quartets)
                if mode == 'discovery':
                    selection = _discovery_selection(metrics, magnitudes, self.expected_quartets)
                else:
                    from .agreement_heldout import score_heldout
                    evaluation = score_heldout(metrics, frozen_selection)
                    evaluation['prediction_sha256'] = hashlib.sha256(frozen.prediction_bytes).hexdigest()
                    evaluation['prediction_freeze_sha256'] = frozen.freeze.sha256
                    evaluation['heldout_commitment'] = heldout_binding.commitment_sha256
                    evaluation['exposure_inventory_sha256'] = heldout_binding.inventory_sha256
                    evaluation['discovery_event_sha256'] = frozen_evidence['discovery_event_sha256']
                    evaluation['independent_audit_sha256'] = frozen_evidence['independent_audit_sha256']
                    evaluation['clean_baseline'] = metrics['baseline']
                    evaluation['clean_cases'] = []
                    for case in metric_cases:
                        margin = math.fsum((case['clean_is'], -case['clean_are']))
                        weight = 1 if case['number'] == 'singular' else -1
                        evaluation['clean_cases'].append({**case, 'clean_margin': margin,
                                                          'correct': weight * margin > 0})
                past_metrics = score_factorial(past_cases, past_patches,
                                               [site['id'] for site in _CANDIDATES],
                                               expected_quartets=self.expected_quartets)
                past_metrics['pairs'] = [_label_past_pair(pair) for pair in past_metrics['pairs']]
                # Bound each summary event independently, even with four
                # families and the full 512-pair factorial retained.
                for metric_kind, source in (('present_site_metrics', metrics),
                                            ('secondary_past_site_metrics', past_metrics)):
                    for site in (item['id'] for item in _CANDIDATES):
                        site_result = {'baseline': source['baseline'],
                                       'site_summary': next(row for row in source['site_summaries']
                                                            if row['site'] == site),
                                       'quartets': [row for row in source['quartets'] if row['site'] == site],
                                       'pairs': [_pair_with_signs(row) for row in source['pairs']
                                                 if row['site'] == site]}
                        contrast = ({'first_token': ' is', 'first_id': 318,
                                     'second_token': ' are', 'second_id': 389}
                                    if metric_kind == 'present_site_metrics' else
                                    {'first_token': ' was', 'first_id': 373,
                                     'second_token': ' were', 'second_id': 547})
                        self.journal.record({'kind': metric_kind, 'site': site, 'contrast': contrast,
                                             'metrics': site_result}, {})
                result = {'status': 'complete', 'forwards': self._forwards, 'controls': controls,
                          'model_state_unchanged': True, 'model_state_sha256': initial_state,
                          'cpu_rng_seed': _RNG_SEED, 'seeded_cpu_rng_sha256': seeded_rng_sha256,
                          'device_rng': device_rng}
                if mode == 'discovery':
                    result['selection'] = selection
                else:
                    result['heldout_evaluation'] = evaluation
                self.journal.record({'kind': 'summary', 'result': result}, {})
                result['metrics'] = metrics
                result['secondary_past_metrics'] = past_metrics
            finally:
                primary_error = sys.exc_info()[1]
                self._before_rng_cleanup(primary_error)
                try:
                    if rng_before is not None:
                        try:
                            try:
                                self._restore_device_rng()
                            finally:
                                torch.random.set_rng_state(rng_before)
                            if not torch.equal(torch.random.get_rng_state(), rng_before):
                                raise StudyFailure('CPU RNG state restoration failed')
                        except BaseException as rng_error:
                            self.quarantined = True
                            if primary_error is not None:
                                if hasattr(primary_error, 'add_note'):
                                    primary_error.add_note('RNG restoration failed; discard the worker: '
                                                           + type(rng_error).__name__)
                            else:
                                raise
                finally:
                    active_error = sys.exc_info()[1] or primary_error
                    try:
                        self._release_after_work(lease, initial_state, active_error)
                    except BaseException as cleanup_error:
                        # An interruption between verification and release can
                        # leave ownership held even if verification succeeded.
                        self.quarantined = True
                        try:
                            lease._quarantine()
                        except BaseException:
                            pass
                        if active_error is not None:
                            if hasattr(active_error, 'add_note'):
                                active_error.add_note('Lease cleanup was interrupted; discard the worker: '
                                                      + type(cleanup_error).__name__)
                        else:
                            raise
            if not self.cancel_event.commit():
                raise StudyCancelled('study cancelled before candidate completion')
            self.journal.finish('complete', {'forwards': self._forwards})
            return result
        except BaseException as error:
            self._record_pending_rejected(error)
            try:
                self.journal.finish('cancelled' if isinstance(error, StudyCancelled) else 'failed',
                                    {'error_type': type(error).__name__, 'message': _error_message(error),
                                     'forwards_started': self._forwards_started,
                                     'model_returns': self._model_returns,
                                     'forwards_validated': self._forwards,
                                     'forward_evidence_records': self._forwards_recorded,
                                     'failure_evidence_error': self._failure_evidence_error})
            except BaseException as journal_error:
                if hasattr(error, 'add_note'):
                    error.add_note('Journal terminal publication failed: ' + type(journal_error).__name__)
            raise
