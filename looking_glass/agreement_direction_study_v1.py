"""Fixed public agreement-direction schedule and execution helpers.

This module never prepares cases, loads a checkpoint, or authorizes a study.
The parent must bind publication, exposure and custodian records separately.
"""

import hashlib
import json
import math
import os
import struct
import sys

import torch

from .agreement_journal import AgreementJournal
from .agreement_numeric import decode_f32_le
from .agreement_patch_runner import (AgreementCase, NaturalAgreementRunner, StudyFailure,
                                     _same_state_identity, _site, _state_identity,
                                     _steps_for_action, _tensor_from_bytes,
                                     _unchanged_outside_target)
from .bert_matched_numeric import binary64_rounding_probe, f32_bits_to_f64, f64_to_f32_bits
from .distilgpt2_intervention_dev import _hook_state
from .model_ownership import ModelLease

TARGET_ARMS = ('same_diff', 'opp_same', 'opp_diff', 'u_plus', 'u_minus',
               'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus', 'l1_u_plus',
               'l1_u_minus', 'subject_u_plus', 'subject_u_minus',
               'center_u', 'center_v1', 'center_v2')
CONTEXT_ARMS = ('u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')
TARGET_PREFIX = ('clean', 'clean_repeat', 'noop_l3_final', 'noop_l3_subject', 'noop_l1_final')
CONTEXT_PREFIX = ('clean', 'clean_repeat', 'noop_l3_final')
ZERO_FLOOR = 2 ** -19


def _vector(raw):
    if type(raw) is not bytes or not raw or len(raw) % 4 or len(raw) > 768 * 4:
        raise ValueError('expected a bounded float32 vector')
    return tuple(f32_bits_to_f64(word) for (word,) in struct.iter_unpack('<I', raw))


def _norm(values):
    try:
        result = math.sqrt(math.fsum(value * value for value in values))
    except (OverflowError, ValueError) as error:
        raise ValueError('invalid vector norm') from error
    if not math.isfinite(result):
        raise ValueError('nonfinite vector norm')
    return result


def _direction(direction, width):
    if (type(direction) not in (tuple, list) or len(direction) != width
            or any(type(v) is not float or not math.isfinite(v) for v in direction)
            or abs(_norm(direction) - 1.0) > 1e-12):
        raise ValueError('direction must be finite, aligned and unit length')
    return tuple(direction)


def family_geometry(captures, u):
    """Compute fixed donor distances, common pair doses and centering signs."""
    binary64_rounding_probe()
    if type(captures) not in (tuple, list) or len(captures) != 4:
        raise ValueError('expected four ordered family captures')
    rows = tuple(_vector(raw) for raw in captures)
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('capture widths differ')
    u = _direction(u, len(rows[0]))
    distances = [[_norm(a - b for a, b in zip(rows[i ^ mask], rows[i]))
                  for mask in (2, 1, 3)] for i in range(4)]
    doses, centering, centering_gaps = [None] * 4, [None] * 4, []
    for singular, plural in ((0, 1), (2, 3)):
        q = min(distances[singular] + distances[plural]) / 2.0
        q = 0.0 if q <= ZERO_FLOOR else q
        doses[singular] = doses[plural] = q
        gap = (math.fsum(a * b for a, b in zip(rows[singular], u))
               - math.fsum(a * b for a, b in zip(rows[plural], u)))
        centering_gaps.append(gap)
        t = gap / 2.0
        if not math.isfinite(t):
            raise ValueError('nonfinite centering magnitude')
        t = 0.0 if abs(t) <= ZERO_FLOOR else t
        centering[singular], centering[plural] = t, -t if t else 0.0
    return {'distances': distances, 'doses': doses, 'centering': centering,
            'centering_gaps': centering_gaps}


def _delivery(base, base_values, values, requested, extra):
    binary64_rounding_probe()
    before_bits = tuple(word for (word,) in struct.iter_unpack('<I', base))
    if requested == 0.0:
        raw = base
        actual = base_values
        after_bits = before_bits
    else:
        words = []
        for item in values:
            value = float(item)
            if math.isfinite(value):
                words.append(f64_to_f32_bits(value))
            else:
                words.append((0x80000000 if math.copysign(1.0, value) < 0 else 0)
                             | (0x7fc00000 if math.isnan(value) else 0x7f800000))
        raw = struct.pack('<' + str(len(words)) + 'I', *words)
        after_bits = tuple(words)
        if any((word & 0x7f800000) == 0x7f800000 for word in words):
            return {'value': raw, 'requested_norm': requested, 'realized_norm': None,
                    'dose_valid': False, 'failure': 'nonfinite_rounded_candidate',
                    'relative_error': None, 'source_norm': _norm(base_values),
                    'realized_over_source_norm': None, 'changed_elements': None,
                    'changed_words': sum(a != b for a, b in zip(after_bits, before_bits)),
                    'changed_bit_count': sum((a ^ b).bit_count() for a, b in zip(after_bits, before_bits)),
                    **extra}
        actual = tuple(f32_bits_to_f64(word) for word in words)
    realized = _norm(a - b for a, b in zip(actual, base_values))
    error = None if requested == 0.0 else abs(realized - requested) / requested
    valid = raw == base if requested == 0.0 else error <= 1e-5
    source_norm = _norm(base_values)
    return {'value': raw, 'requested_norm': requested, 'realized_norm': realized, 'dose_valid': valid,
            'relative_error': error, 'source_norm': source_norm,
            'realized_over_source_norm': None if source_norm == 0 else realized / source_norm,
            'changed_elements': sum(a != b for a, b in zip(actual, base_values)),
            'changed_words': sum(a != b for a, b in zip(after_bits, before_bits)),
            'changed_bit_count': sum((a ^ b).bit_count() for a, b in zip(after_bits, before_bits)),
            **extra}


def donor_patch(base, donor, q):
    """Round one input-space donor interpolation from the unmodified base."""
    original, source = _vector(base), _vector(donor)
    if len(original) != len(source) or type(q) not in (int, float) or not math.isfinite(q) or q < 0:
        raise ValueError('invalid donor displacement')
    difference = tuple(a - b for a, b in zip(source, original))
    distance = _norm(difference)
    if q == 0:
        return _delivery(base, original, (), 0.0, {'donor_distance': distance, 'fraction': None})
    if distance == 0 or q > distance:
        raise ValueError('requested donor dose exceeds the segment')
    fraction = float(q / distance)
    values = (value + fraction * change for value, change in zip(original, difference))
    return _delivery(base, original, values, float(q), {'donor_distance': distance, 'fraction': fraction})


def direction_patch(base, direction, amount):
    """Round one signed unit-direction displacement from the unmodified base."""
    original = _vector(base)
    direction = _direction(direction, len(original))
    if type(amount) not in (int, float) or not math.isfinite(amount):
        raise ValueError('invalid direction displacement')
    if amount == 0:
        return _delivery(base, original, (), 0.0, {'signed_amount': 0.0})
    values = (value + amount * change for value, change in zip(original, direction))
    return _delivery(base, original, values, float(abs(amount)), {'signed_amount': float(amount)})


def fixed_schedule(cases):
    """Return the complete fixed schedule after checking case order and identity."""
    if type(cases) not in (tuple, list) or len(cases) != 40:
        raise ValueError('expected 32 ordered targets and eight contexts')
    ids = set()
    for index, row in enumerate(cases):
        if type(row) is not dict:
            raise ValueError('case must be an object')
        name = row.get('case_id')
        if type(name) is not str or not name or len(name) > 128 or name in ids:
            raise ValueError('invalid or duplicate case ID')
        ids.add(name)
        if index < 32:
            family, variant = divmod(index, 4)
            if (row.get('role') != 'target' or type(row.get('family_index')) is not int
                    or row['family_index'] != family or type(row.get('variant')) is not int
                    or row['variant'] != variant
                    or row.get('subject_number') != ('singular' if variant % 2 == 0 else 'plural')):
                raise ValueError('target order or number differs')
        elif (row.get('role') != 'context' or type(row.get('family_index')) is not int
              or row['family_index'] != index - 32):
            raise ValueError('context order differs')
    actions = []
    def append(row, action, slot=None):
        actions.append({'case_id': row['case_id'], 'role': row['role'],
                        'family_index': row['family_index'],
                        'subject_number': row.get('subject_number'),
                        'action': action, 'slot': slot})
    for family in range(8):
        targets = cases[family * 4:family * 4 + 4]
        for row in targets:
            for action in TARGET_PREFIX:
                append(row, action)
        for row in targets:
            offset = (family + 4 * row['variant']) % len(TARGET_ARMS)
            rotated = TARGET_ARMS[offset:] + TARGET_ARMS[:offset]
            for slot, action in enumerate(rotated):
                append(row, action, slot)
            append(row, 'clean_after')
        context = cases[32 + family]
        for action in CONTEXT_PREFIX:
            append(context, action)
        offset = family % len(CONTEXT_ARMS)
        for slot, action in enumerate(CONTEXT_ARMS[offset:] + CONTEXT_ARMS[:offset]):
            append(context, action, slot)
        append(context, 'clean_after')
    return tuple(actions)


def _validated_cases(cases, vocabulary):
    fixed_schedule(cases)
    converted, inputs = {}, set()
    for index, row in enumerate(cases):
        tokens = row.get('token_ids')
        if (type(tokens) is not list or not 2 <= len(tokens) <= 32
                or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
                or tuple(tokens) in inputs):
            raise StudyFailure('invalid or repeated public input')
        inputs.add(tuple(tokens))
        if index < 32:
            if len(tokens) != 5 or type(row.get('subject_index')) is not int or row['subject_index'] != 1:
                raise StudyFailure('target token coordinates differ')
            subject_index = row['subject_index']
        else:
            subject_index = 1
        converted[row['case_id']] = AgreementCase(row['case_id'], str(row['family_index']),
                                                   str(row.get('subject_lemma', 'context')),
                                                   row.get('subject_number', 'singular'),
                                                   tuple(tokens), subject_index)
    for family in range(8):
        members = cases[4 * family:4 * family + 4]
        contexts = {tuple(row['token_ids'][:1] + row['token_ids'][2:]) for row in members}
        subjects = {row['token_ids'][1] for row in members}
        if len(contexts) != 1 or len(subjects) != 4:
            raise StudyFailure('target family does not have the fixed suffix')
    return converted


def _checked_directions(directions, width):
    if type(directions) is not dict or set(directions) != {'u', 'v1', 'v2'}:
        raise StudyFailure('expected three fixed direction vectors')
    result = {name: _direction(value, width) for name, value in directions.items()}
    for left, right in (('u', 'v1'), ('u', 'v2'), ('v1', 'v2')):
        if abs(math.fsum(a * b for a, b in zip(result[left], result[right]))) > 1e-12:
            raise StudyFailure('direction vectors are not orthogonal')
    return result


class _DirectionRunner(NaturalAgreementRunner):
    """Use the existing exclusive forward/plan machinery with raw-bit evidence."""

    def __init__(self, model, journal, custody_check=None):
        super().__init__(model, journal, device='cpu', expected_quartets=8)
        self._custody_check = custody_check
        self._last_output = None
        self._last_plan = None
        self._current_action = None
        self._environment = {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'MKL_CBWR', 'ATEN_CPU_CAPABILITY', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
        self._default_dtype = torch.get_default_dtype()

    def _encode_payload(self, tensor):
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.device.type != 'cpu' or sys.byteorder != 'little'):
            raise StudyFailure('expected CPU little-endian float32 output')
        return tensor.detach().contiguous().view(torch.int32).numpy().tobytes()

    def _root_forward(self, name, value):
        if name == 'output':
            self._last_output = value
        elif name == 'plan':
            self._last_plan = value

    def _before_forward(self):
        if self._custody_check is not None:
            self._custody_check()
        binary64_rounding_probe()
        if (self._environment['CUDA_VISIBLE_DEVICES'] != '-1'
                or self._environment != {key: os.environ.get(key) for key in self._environment}
                or torch.cuda.is_initialized() or torch.get_default_dtype() != self._default_dtype):
            raise StudyFailure('CPU execution environment changed')
        self._last_output = None
        self._last_plan = None

    def _after_forward(self):
        if self._custody_check is not None:
            self._custody_check()
        binary64_rounding_probe()
        if (self._environment != {key: os.environ.get(key) for key in self._environment}
                or torch.cuda.is_initialized() or torch.get_default_dtype() != self._default_dtype):
            raise StudyFailure('CPU execution environment changed during forward')

    def _save_forward(self, case, action, steps, *, dose=None):
        width, length, vocabulary = self.model.config.n_embd, len(case.input_ids), self.model.config.vocab_size
        # Reserve the complete returned sequence even though successful records
        # retain only the last row. Hooked failures may also retain both whole
        # activation outputs and both selected vectors.
        reservation = 4 * (length * vocabulary + (2 * length * width + 2 * width if steps else 0))
        journal = self.journal
        if (reservation > journal._max_forward or journal.retained_bytes + reservation > journal._max_total
                or journal._events >= journal._max_events
                or journal._event_log_bytes + 256 * 1024 > journal._max_event_log_bytes):
            raise StudyFailure('insufficient bounded forward evidence reservation')
        self._current_action = dict(action)
        logits, captures, metadata = self._forward(case, steps)
        output = self._last_output.logits
        finite = bool(torch.isfinite(output).all())
        tensors = {}
        for name, raw in captures.items():
            shape = ((1, len(case.input_ids), self.model.config.n_embd)
                     if name in ('whole_before', 'whole_after') else (self.model.config.n_embd,))
            tensors[name] = (shape, raw)
        if finite:
            tensors['last_logits'] = ((self.model.config.vocab_size,), logits)
        else:
            tensors['full_sequence_logits'] = (tuple(output.shape), self._encode_payload(output))
        event = {'kind': 'forward', **action, 'input_ids': list(case.input_ids),
                 'attention_mask': [1] * len(case.input_ids), 'plan_metadata': metadata,
                 'all_sequence_logits_finite': finite, 'dose': dose}
        self.journal.record(event, tensors)
        self._forwards_recorded += 1
        self._pending_forward = None
        self._last_output = None
        self._last_plan = None
        self._current_action = None
        if not finite:
            raise StudyFailure('nonfinite sequence logits were retained')
        decode_f32_le(logits)
        self._cancel_check()
        return logits, captures

    def _record_rejected(self, case, output, captures, metadata, primary_error):
        """Retain the bounded returned sequence and available hook snapshots."""
        tensors, unavailable = {}, []
        remaining = max(0, min(self.journal._max_forward,
                               self.journal._max_total - self.journal.retained_bytes))
        logits_observation = None
        if self._last_plan is not None and (not isinstance(captures, dict) or metadata is None):
            try:
                # These properties publish only after a successful plan exit.
                recovered_captures = self._last_plan.captures
                recovered_metadata = self._last_plan.metadata
                if not isinstance(captures, dict):
                    captures = recovered_captures
                if metadata is None:
                    metadata = recovered_metadata
            except (KeyboardInterrupt, SystemExit) as error:
                raise error from primary_error
            except Exception:
                unavailable.append('plan did not expose committed captures')
        if isinstance(captures, dict):
            for name, tensor in captures.items():
                try:
                    if (not isinstance(tensor, torch.Tensor) or not 1 <= tensor.ndim <= 8
                            or any(dim <= 0 for dim in tensor.shape)
                            or tensor.numel() * 4 > remaining):
                        unavailable.append('capture ' + str(name) + ': exceeds remaining bound')
                        continue
                    payload = self._encode_payload(tensor)
                    tensors[name] = (tuple(tensor.shape), payload)
                    remaining -= len(payload)
                except BaseException as error:
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise error from primary_error
                    unavailable.append('capture ' + str(name) + ': ' + type(error).__name__)
        else:
            unavailable.append('plan did not publish captures')
        try:
            logits = getattr(output, 'logits', None)
            if (isinstance(logits, torch.Tensor) and logits.ndim == 3
                    and all(dim > 0 for dim in logits.shape)):
                logits_observation = {'shape': list(logits.shape), 'saved': 'unavailable'}
                if logits.numel() * 4 <= remaining:
                    tensors['full_sequence_logits'] = (tuple(logits.shape), self._encode_payload(logits))
                    logits_observation['saved'] = 'full_sequence'
                elif remaining >= 4:
                    count = min(logits.shape[2], remaining // 4)
                    # Slice before materializing: a malformed oversized result
                    # must not allocate a second copy beyond the journal bound.
                    prefix = logits[0, 0, :count]
                    tensors['sequence_logits_prefix'] = ((count,), self._encode_payload(prefix))
                    logits_observation['saved'] = 'first_batch_first_position_prefix'
                    logits_observation['saved_elements'] = count
                else:
                    unavailable.append('sequence logits exceed remaining evidence capacity')
            else:
                unavailable.append('sequence logits unavailable')
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error from primary_error
            unavailable.append('sequence logits: ' + type(error).__name__)
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            metadata = None
            unavailable.append('plan metadata was not finite JSON')
        try:
            self.journal.record({'kind': 'rejected_forward', 'case': case.case_id,
                                 'action': self._current_action, 'error_type': type(primary_error).__name__,
                                 'plan_metadata': metadata, 'unavailable': unavailable,
                                 'logits_observation': logits_observation}, tensors)
            self._forwards_recorded += 1
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error from primary_error
            self._failure_evidence_error = type(error).__name__
            if hasattr(primary_error, 'add_note'):
                primary_error.add_note('Rejected-forward evidence could not be saved: ' + type(error).__name__)

    def _record_pending_rejected(self, primary_error):
        pending = self._pending_forward
        if pending is None:
            return
        case, logits, captures, metadata = pending
        tensors = {}
        try:
            width = self._baseline_contract[0]
            for name, value in captures.items():
                shape = ((1, len(case.input_ids), width)
                         if name in ('whole_before', 'whole_after') else (width,))
                tensors[name] = (shape, value)
            output = getattr(self._last_output, 'logits', None)
            if isinstance(output, torch.Tensor) and output.ndim == 3:
                tensors['full_sequence_logits'] = (tuple(output.shape), self._encode_payload(output))
            else:
                tensors['last_logits'] = ((len(logits) // 4,), logits)
            self.journal.record({'kind': 'rejected_forward', 'case': case.case_id,
                                 'action': self._current_action, 'error_type': type(primary_error).__name__,
                                 'plan_metadata': metadata, 'unavailable': []}, tensors)
            self._forwards_recorded += 1
            self._pending_forward = None
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error from primary_error
            self._failure_evidence_error = type(error).__name__
            if hasattr(primary_error, 'add_note'):
                primary_error.add_note('Pending-forward evidence could not be saved: ' + type(error).__name__)


def _publish_bad_dose(journal, case, action, base, candidate):
    metadata = {key: value for key, value in candidate.items() if key != 'value'}
    journal.record({'kind': 'invalid_delivery', **action, 'dose': metadata},
                   {'source': ((len(base) // 4,), base),
                    'candidate': ((len(candidate['value']) // 4,), candidate['value'])})
    raise StudyFailure('rounded delivery failed the fixed dose rule')


def _reserve_candidate_evidence(journal, width):
    """Leave room to save both source and rejected rounded candidate bits."""
    size = 2 * width * 4
    if (size > journal._max_forward or journal.retained_bytes + size > journal._max_total
            or journal._events >= journal._max_events
            or journal._event_log_bytes + 256 * 1024 > journal._max_event_log_bytes):
        raise StudyFailure('insufficient candidate evidence reservation')


def _arm_candidate(arm, target, family, geometry, captures, directions):
    variant = target['variant']
    q, t = geometry['doses'][variant], geometry['centering'][variant]
    base = captures[target['case_id']]['l3_final']
    if arm in ('same_diff', 'opp_same', 'opp_diff'):
        other = variant ^ {'same_diff': 2, 'opp_same': 1, 'opp_diff': 3}[arm]
        donor = family[other]['case_id']
        return base, donor_patch(base, captures[donor]['l3_final'], q), donor, _site(3, 'attn')
    if arm.startswith('l1_'):
        base, site, name, sign = captures[target['case_id']]['l1_final'], _site(1, 'attn'), 'u', arm.endswith('plus')
        return base, direction_patch(base, directions[name], q if sign else -q), None, site
    if arm.startswith('subject_'):
        base, site, name, sign = captures[target['case_id']]['l3_subject'], _site(3, 'attn', 'subject'), 'u', arm.endswith('plus')
        return base, direction_patch(base, directions[name], q if sign else -q), None, site
    if arm.startswith('center_'):
        name = arm[7:]
        return base, direction_patch(base, directions[name], -t), None, _site(3, 'attn')
    name, sign = arm.rsplit('_', 1)
    return base, direction_patch(base, directions[name], q if sign == 'plus' else -q), None, _site(3, 'attn')


def _act(runner, case, action, clean, *, site=None, base=None, candidate=None, donor=None):
    if site is None:
        logits, captures = runner._save_forward(case, action, None)
    else:
        if candidate is not None and not candidate['dose_valid']:
            _publish_bad_dose(runner.journal, case, action, base, candidate)
        value = None if candidate is None else _tensor_from_bytes(candidate['value'])
        steps = _steps_for_action(case, site, 'null' if candidate is None else 'patch', value)
        dose = None if candidate is None else {key: value for key, value in candidate.items() if key != 'value'}
        action = {**action, 'site': site, 'donor': donor}
        logits, captures = runner._save_forward(case, action, steps, dose=dose)
        expected = captures['before'] if candidate is None else candidate['value']
        if ((base is not None and captures['before'] != base) or captures['after'] != expected
                or not _unchanged_outside_target(captures, case, site, runner.model.config.n_embd)):
            raise StudyFailure('saved intervention landing or untouched coordinates differ')
    if (clean is not None and (site is None or candidate is None
                              or candidate['requested_norm'] == 0.0) and logits != clean):
        raise StudyFailure('saved clean or no-op logits differ')
    return logits, captures


def run_direction_study(model, journal, cases, directions, *, binding=None, custody_check=None):
    """Run the fixed plan on an exclusively owned model, yielding only a candidate.

    This private worker does not load a model or establish source, exposure,
    custody, external process exit, or scientific audit. The trusted caller
    must perform those gates and must not retry an incomplete attempt.
    """
    if type(journal) is not AgreementJournal:
        raise ValueError('expected an actual bounded agreement journal')
    if custody_check is not None and not callable(custody_check):
        raise ValueError('custody check must be callable')
    if binding is not None:
        names = {'prelaunch_sha256', 'reservation_sha256', 'latest_exposure_sha256'}
        if (type(binding) is not dict or set(binding) != names
                or any(type(value) is not str or len(value) != 64
                       or any(char not in '0123456789abcdef' for char in value)
                       for value in binding.values())):
            raise ValueError('invalid externally observed binding')
        binding = dict(binding)
    runner = _DirectionRunner(model, journal, custody_check=custody_check)
    runner._used = True
    lease = ModelLease(model)
    runner._lease = lease
    initial_state = None
    old_rng = None
    primary = None
    chained = None
    result = None
    try:
        lease.acquire()
        runner._baseline_contract = runner._execution_contract()
        converted = _validated_cases(cases, model.config.vocab_size)
        directions = _checked_directions(directions, model.config.n_embd)
        runner._baseline_hooks = _hook_state(model)
        if (any(any(registry for registry in module[1:]) for module in runner._baseline_hooks[0])
                or any(runner._baseline_hooks[1:])):
            raise StudyFailure('preexisting model hooks are unsupported')
        initial_state = runner._model_digest()
        runner._baseline_state_identity = _state_identity(model)
        old_rng = torch.random.get_rng_state()
        torch.random.default_generator.manual_seed(20261030)
        schedule = fixed_schedule(cases)
        journal.record({'kind': 'run_contract', 'expected_forwards': 784,
                        'environment': runner._environment, 'seed': 20261030,
                        'execution_configuration': runner._baseline_contract[2],
                        'model_state_sha256': initial_state, 'external_binding': binding}, {})
        index = 0
        for family_index in range(8):
            family = cases[family_index * 4:family_index * 4 + 4]
            captured, clean = {}, {}
            for row in family:
                case = converted[row['case_id']]
                captured[case.case_id] = {}
                for position in range(5):
                    action = {'index': index, **schedule[index]}
                    name = action['action']
                    site = {'noop_l3_final': _site(3, 'attn'),
                            'noop_l3_subject': _site(3, 'attn', 'subject'),
                            'noop_l1_final': _site(1, 'attn')}.get(name)
                    logits, values = _act(runner, case, action, clean.get(case.case_id), site=site)
                    if name == 'clean':
                        clean[case.case_id] = logits
                    if site is not None:
                        captured[case.case_id][name[5:]] = values['before']
                    index += 1
            geometry = family_geometry([captured[row['case_id']]['l3_final'] for row in family], directions['u'])
            journal.record({'kind': 'family_geometry', 'family_index': family_index, **geometry}, {})
            centered = {}
            for row in family:
                case = converted[row['case_id']]
                for _ in range(16):
                    action = {'index': index, **schedule[index]}
                    _reserve_candidate_evidence(journal, model.config.n_embd)
                    base, candidate, donor, site = _arm_candidate(action['action'], row, family, geometry,
                                                                   captured, directions)
                    _act(runner, case, action, clean[case.case_id], site=site, base=base,
                         candidate=candidate, donor=donor)
                    if action['action'].startswith('center_'):
                        centered[(row['variant'], action['action'])] = candidate['value']
                    index += 1
                _act(runner, case, {'index': index, **schedule[index]}, clean[case.case_id])
                index += 1
            for first in (0, 2):
                gaps = {}
                for name in ('center_u', 'center_v1', 'center_v2'):
                    a, b = (_vector(centered[(i, name)]) for i in (first, first + 1))
                    gaps[name] = (math.fsum(x * y for x, y in zip(a, directions['u']))
                                  - math.fsum(x * y for x, y in zip(b, directions['u'])))
                journal.record({'kind': 'centering_result', 'family_index': family_index,
                                'singular_variant': first, 'achieved_u_gaps': gaps}, {})
            row = cases[32 + family_index]
            case = converted[row['case_id']]
            context_clean = None
            context_base = None
            for _ in range(3):
                action = {'index': index, **schedule[index]}
                site = _site(3, 'attn') if action['action'] == 'noop_l3_final' else None
                logits, values = _act(runner, case, action, context_clean, site=site)
                if action['action'] == 'clean':
                    context_clean = logits
                if site is not None:
                    context_base = values['before']
                index += 1
            q = math.fsum((geometry['doses'][0], geometry['doses'][2])) / 2.0
            for _ in range(6):
                action = {'index': index, **schedule[index]}
                name, sign = action['action'].rsplit('_', 1)
                _reserve_candidate_evidence(journal, model.config.n_embd)
                candidate = direction_patch(context_base, directions[name], q if sign == 'plus' else -q)
                _act(runner, case, action, context_clean, site=_site(3, 'attn'),
                     base=context_base, candidate=candidate)
                index += 1
            _act(runner, case, {'index': index, **schedule[index]}, context_clean)
            index += 1
        if index != 784 or runner._forwards_recorded != 784 or runner._forwards != 784:
            raise StudyFailure('forward inventory differs')
        if runner._model_digest() != initial_state or not _same_state_identity(runner._baseline_state_identity, model):
            raise StudyFailure('model state or buffer identity changed')
        result = {'status': 'complete_candidate', 'forwards': 784,
                  'model_state_sha256_before_after': initial_state,
                  'retained_tensor_bytes': journal.retained_bytes}
    except BaseException as error:
        primary = error
    finally:
        try:
            if old_rng is not None:
                torch.random.set_rng_state(old_rng)
                if not torch.equal(torch.random.get_rng_state(), old_rng):
                    raise StudyFailure('CPU RNG restoration failed')
        except BaseException as cleanup_error:
            runner.quarantined = True
            if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                chained, primary = primary, cleanup_error
            elif primary is None:
                primary = cleanup_error
            elif hasattr(primary, 'add_note'):
                primary.add_note('RNG cleanup failed: ' + type(cleanup_error).__name__)
        try:
            # Passing no primary forces the inherited lease helper to surface
            # cancellation or cleanup errors instead of only annotating them.
            runner._release_after_work(lease, initial_state, None)
        except BaseException as cleanup_error:
            runner.quarantined = True
            if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                chained, primary = primary, cleanup_error
            elif primary is None:
                primary = cleanup_error
            elif hasattr(primary, 'add_note'):
                primary.add_note('Model cleanup failed: ' + type(cleanup_error).__name__)
    if primary is not None:
        try:
            runner._record_pending_rejected(primary)
        except (KeyboardInterrupt, SystemExit) as evidence_error:
            chained, primary = primary, evidence_error
        try:
            journal.finish('cancelled' if isinstance(primary, (KeyboardInterrupt, SystemExit)) else 'failed',
                           {'error_type': type(primary).__name__, 'forwards_started': runner._forwards_started,
                            'model_returns': runner._model_returns, 'forward_evidence_records': runner._forwards_recorded,
                            'cleanup_quarantined': runner.quarantined})
        except BaseException as terminal_error:
            if isinstance(terminal_error, (KeyboardInterrupt, SystemExit)):
                raise terminal_error from primary
            if hasattr(primary, 'add_note'):
                primary.add_note('Failure terminal could not be written: ' + type(terminal_error).__name__)
        if chained is not None:
            raise primary from chained
        raise primary
    journal.finish('complete', result)
    return result
