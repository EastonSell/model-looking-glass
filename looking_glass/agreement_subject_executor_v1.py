"""Prospective agreement subject-dependence worker primitives.

This module does not prepare cases, load a checkpoint or authorize a run.
The source, protocol, cases, exposure and custody gates must be established
and independently reviewed by the supervised caller before a model forward.
"""

from collections import Counter
import hashlib
import json
import math
import re

import torch

from .agreement_direction_study_v1 import (_DirectionRunner, _act, _checked_directions,
                                           _direction, _reserve_candidate_evidence,
                                           _vector, direction_patch)
from .agreement_journal import AgreementJournal
from .agreement_numeric import decode_f32_le
from .agreement_patch_runner import AgreementCase, StudyFailure, _same_state_identity, _site, _state_identity
from . import distilgpt2_intervention_dev
from .distilgpt2_intervention_dev import _hook_state
from .model_ownership import ModelLease

CONDITIONS = ('none', 'mass', 'attractor_singular', 'attractor_plural')
FRAME_PAIRS = (('R1', 'I1'), ('R1', 'I2'), ('R2', 'I1'), ('R2', 'I2'))
CONTEXT_OFFSETS = ((0, 0, 2, 2), (2, 2, 4, 4), (2, 2, 4, 4), (4, 4, 4, 4))
PLUS_FIRST = ('u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')
MINUS_FIRST = ('u_minus', 'u_plus', 'v1_minus', 'v1_plus', 'v2_minus', 'v2_plus')


def fixed_schedule(cases):
    """Validate canonical identities and return the 720 ordered actions."""
    if type(cases) not in (tuple, list) or len(cases) != 80:
        raise ValueError('expected 80 canonical cases')
    identifiers = set()
    schedule = []
    for frame in range(4):
        for condition, name in enumerate(CONDITIONS):
            start = frame * 20 + condition * 5
            block = cases[start:start + 5]
            for position, row in enumerate(block):
                if type(row) is not dict:
                    raise ValueError('case must be an object')
                case_id = row.get('case_id')
                if (type(case_id) is not str or not case_id or len(case_id) > 128
                        or case_id in identifiers or type(row.get('frame_index')) is not int
                        or row['frame_index'] != frame or row.get('condition') != name
                        or type(row.get('condition_index')) is not int
                        or row['condition_index'] != condition):
                    raise ValueError('duplicate or invalid canonical case identity')
                identifiers.add(case_id)
                if position == 4:
                    if 'pair_index' not in row or row['pair_index'] is not None:
                        raise ValueError('context pair coordinate differs')
                    expected = ('context', None, None, None)
                else:
                    pair, number = divmod(position, 2)
                    if type(row.get('pair_index')) is not int or row['pair_index'] != pair:
                        raise ValueError('target pair coordinate differs')
                    expected = ('target', FRAME_PAIRS[frame][pair],
                                ('singular', 'plural')[number],
                                ('regular', 'irregular')[pair])
                if tuple(row.get(key) for key in ('role', 'subject_key', 'subject_number', 'morphology')) != expected:
                    raise ValueError('canonical role, pair or subject order differs')

            execution = []
            for pair in range(2):
                members = [(start + 2 * pair + number, block[2 * pair + number]) for number in range(2)]
                if (frame + frame // 2 + pair) % 2:
                    members.reverse()
                execution.extend((item, pair) for item in members)
            execution.append(((start + 4, block[4]), None))
            for (case_index, row), pair in execution:
                if pair is None:
                    offset = CONTEXT_OFFSETS[frame][condition]
                    plus_first = (frame + condition) % 2 == 0
                else:
                    offset = 2 * ((4 * frame + condition) % 3)
                    plus_first = (frame + condition + pair) % 2 == 0
                base = PLUS_FIRST if plus_first else MINUS_FIRST
                arms = base[offset:] + base[:offset]
                common = {'case_id': row['case_id'], 'case_index': case_index,
                          'frame_index': frame, 'condition': name, 'condition_index': condition,
                          'role': row['role'], 'pair_index': pair,
                          'subject_key': row.get('subject_key'),
                          'subject_number': row.get('subject_number'),
                          'morphology': row.get('morphology')}
                for action, slot in [('clean', None), ('noop_l3_final', None),
                                     *[(arm, i) for i, arm in enumerate(arms)], ('clean_after', None)]:
                    schedule.append({'index': len(schedule), **common, 'action': action, 'slot': slot})
    return tuple(schedule)


def signed_direction_patch(source, direction, amount):
    """Check the independently rounded dose and its signed projection."""
    if type(amount) is not float or abs(amount) != 2.0:
        raise ValueError('the fixed signed dose must be plus or minus 2.0')
    before = _vector(source)
    direction = _direction(direction, len(before))
    result = direction_patch(source, direction, amount)
    try:
        after = _vector(result['value'])
        projection = math.fsum((a - b) * d for a, b, d in zip(after, before, direction))
        relative_error = abs(abs(projection) - 2.0) / 2.0
        projected_valid = math.isfinite(projection) and projection * amount > 0 and relative_error <= 1e-5
    except (ValueError, OverflowError):
        projection, relative_error, projected_valid = None, None, False
    result['signed_projected_displacement'] = projection
    result['projected_relative_error'] = relative_error
    result['dose_valid'] = result['dose_valid'] and projected_valid
    return result


def _validated_cases(cases, vocabulary):
    fixed_schedule(cases)
    seen, converted = set(), {}
    for row in cases:
        tokens = row.get('token_ids')
        text = row.get('text')
        if (type(text) is not str or not text or len(text) > 512
                or type(tokens) is not list or not 1 <= len(tokens) <= 12
                or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
                or tuple(tokens) in seen or type(row.get('final_index')) is not int
                or row['final_index'] != len(tokens) - 1):
            raise ValueError('invalid or repeated input, text or final position')
        current = tuple(tokens)
        if any(current[:len(prior)] == prior or prior[:len(current)] == current for prior in seen):
            raise ValueError('input prefix overlap')
        seen.add(current)
        subject = row.get('subject_index')
        if row['role'] == 'target':
            if type(subject) is not int or not 0 <= subject < len(tokens) - 1:
                raise ValueError('invalid target subject coordinate')
        elif subject is not None:
            raise ValueError('context cannot have a subject coordinate')
        converted[row['case_id']] = AgreementCase(
            row['case_id'], str(row['frame_index']), row.get('subject_key') or 'context',
            row.get('subject_number') or 'singular', tuple(tokens), subject if subject is not None else 0)
    for frame in range(4):
        for condition in range(4):
            block = cases[frame * 20 + condition * 5:frame * 20 + condition * 5 + 5]
            for pair in range(2):
                left, right = block[pair * 2:pair * 2 + 2]
                a, b = left['token_ids'], right['token_ids']
                index = left['subject_index']
                if (right['subject_index'] != index or len(a) != len(b)
                        or a[index] == b[index] or a[:index] + a[index + 1:] != b[:index] + b[index + 1:]):
                    raise ValueError('paired inputs differ outside the subject position')
        blocks = [cases[frame * 20 + condition * 5:frame * 20 + condition * 5 + 5]
                  for condition in range(4)]
        for position in range(4):
            targets = [blocks[condition][position] for condition in (1, 2, 3)]
            if (len({row['subject_index'] for row in targets}) != 1
                    or len({tuple(row['token_ids'][:-1]) for row in targets}) != 1
                    or len({row['token_ids'][-1] for row in targets}) != 3):
                raise ValueError('mass and count targets differ outside their last noun')
        if (blocks[2][4]['token_ids'][:-1] != blocks[3][4]['token_ids'][:-1]
                or blocks[2][4]['token_ids'][-1] == blocks[3][4]['token_ids'][-1]):
            raise ValueError('count fragments differ outside their last noun')
    return converted


def _check_budget(journal, vocabulary, width):
    journal._check_owner()
    if (journal._faulted or journal._events != 0 or journal._event_log_bytes != 0
            or journal.retained_bytes != 0 or journal._known):
        raise ValueError('a fresh empty journal is required before execution')
    # A 12-token failed return and all hook witnesses remain separately
    # reservable even when the previous successful payloads were distinct.
    success = 720 * vocabulary * 4 + 560 * (2 * 12 * width * 4 + 2 * width * 4)
    failure = 4 * (12 * vocabulary + 2 * 12 * width + 2 * width)
    candidate = 2 * width * 4
    if (journal._max_forward < failure or journal._max_total < success + failure + candidate
            or journal._max_events < 722 or journal._max_event_log_bytes < 64 * 1024 * 1024):
        raise ValueError('insufficient evidence reservation before execution')
    return {'successful_logical_bytes_upper_bound': success,
            'failed_return_bytes_reserve': failure,
            'invalid_candidate_bytes_reserve': candidate,
            'retained_tensor_bytes_limit': journal._max_total,
            'per_forward_bytes_limit': journal._max_forward,
            'event_log_bytes_limit': journal._max_event_log_bytes}


def _finite_captures(values):
    for value in values.values():
        decode_f32_le(value)


class _SubjectRunner(_DirectionRunner):
    _propagate_fresh_interruptions = True

    def _failure_cleanup_status(self, error):
        try:
            restored = distilgpt2_intervention_dev._hook_state(self.model) == self._baseline_hooks
        except BaseException as inspection_error:
            self.quarantined = True
            if isinstance(inspection_error, (KeyboardInterrupt, SystemExit)):
                raise
            if hasattr(error, 'add_note'):
                error.add_note('Hook inspection failed: ' + type(inspection_error).__name__)
            return None
        if not restored:
            self.quarantined = True
            if hasattr(error, 'add_note'):
                error.add_note('Hook cleanup is uncertain; discard the model')
        return restored


def run_subject_study(model, journal, cases, directions, *, binding, custody_check=None):
    """Execute the fixed plan on an already authorized, exclusively held model.

    This worker returns a candidate only. It cannot establish that the supplied
    binding is published or approved, observe its own process exit, or replace
    independent replay. A failed or interrupted attempt must not be retried.
    """
    if type(journal) is not AgreementJournal:
        raise ValueError('expected an actual bounded agreement journal')
    names = {'prelaunch_sha256', 'reservation_sha256', 'latest_exposure_sha256'}
    if (type(binding) is not dict or set(binding) != names
            or any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
                   for value in binding.values())):
        raise ValueError('invalid externally observed binding')
    if custody_check is not None and not callable(custody_check):
        raise ValueError('custody check must be callable')
    runner = _SubjectRunner(model, journal, custody_check=custody_check)
    runner._used = True
    lease = ModelLease(model)
    runner._lease = lease
    initial_state = old_rng = primary = chained = result = None
    try:
        lease.acquire()
        runner._baseline_contract = runner._execution_contract()
        converted = _validated_cases(cases, model.config.vocab_size)
        directions = _checked_directions(directions, model.config.n_embd)
        budget = _check_budget(journal, model.config.vocab_size, model.config.n_embd)
        runner._baseline_hooks = _hook_state(model)
        if (any(any(registry for registry in module[1:]) for module in runner._baseline_hooks[0])
                or any(runner._baseline_hooks[1:])):
            raise StudyFailure('preexisting model hooks are unsupported')
        initial_state = runner._model_digest()
        runner._baseline_state_identity = _state_identity(model)
        old_rng = torch.random.get_rng_state()
        torch.random.default_generator.manual_seed(20261030)
        schedule = fixed_schedule(cases)
        schedule_bytes = json.dumps(schedule, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode('ascii')
        counts = Counter((row['action'], row['slot']) for row in schedule if row['slot'] is not None)
        placement = {arm: [counts[(arm, slot)] for slot in range(6)] for arm in PLUS_FIRST}
        journal.record({'kind': 'run_contract', 'expected_forwards': 720,
                        'environment': runner._environment, 'seed': 20261030,
                        'execution_configuration': runner._baseline_contract[2],
                        'model_state_sha256': initial_state, 'external_binding': dict(binding),
                        'schedule_sha256': hashlib.sha256(schedule_bytes).hexdigest(),
                        'arm_placement_counts': placement, 'evidence_budget': budget}, {})
        for start in range(0, len(schedule), 9):
            actions = schedule[start:start + 9]
            case = converted[actions[0]['case_id']]
            clean, _ = _act(runner, case, actions[0], None)
            _, captures = _act(runner, case, actions[1], clean, site=_site(3, 'attn'))
            _finite_captures(captures)
            source = captures['before']
            for action in actions[2:8]:
                name, sign = action['action'].rsplit('_', 1)
                _reserve_candidate_evidence(journal, model.config.n_embd)
                candidate = signed_direction_patch(source, directions[name], 2.0 if sign == 'plus' else -2.0)
                _, captures = _act(runner, case, action, clean, site=_site(3, 'attn'),
                                   base=source, candidate=candidate)
                _finite_captures(captures)
            _act(runner, case, actions[8], clean)
        if (runner._forwards_started != 720 or runner._model_returns != 720
                or runner._forwards_recorded != 720 or runner._forwards != 720):
            raise StudyFailure('forward inventory differs')
        if runner._model_digest() != initial_state or not _same_state_identity(runner._baseline_state_identity, model):
            raise StudyFailure('model state or buffer identity changed')
        result = {'status': 'complete_candidate', 'forwards': 720,
                  'model_state_sha256_before_after': initial_state,
                  'retained_tensor_bytes': journal.retained_bytes,
                  'schedule_sha256': hashlib.sha256(schedule_bytes).hexdigest()}
    except BaseException as error:
        primary = error
    finally:
        try:
            if old_rng is not None:
                torch.random.set_rng_state(old_rng)
                if not torch.equal(torch.random.get_rng_state(), old_rng):
                    raise StudyFailure('CPU RNG restoration failed')
        except BaseException as error:
            runner.quarantined = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                chained, primary = primary, error
            elif primary is None:
                primary = error
            elif hasattr(primary, 'add_note'):
                primary.add_note('RNG cleanup failed: ' + type(error).__name__)
        try:
            runner._release_after_work(lease, initial_state, None)
        except BaseException as error:
            runner.quarantined = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                chained, primary = primary, error
            elif primary is None:
                primary = error
            elif hasattr(primary, 'add_note'):
                primary.add_note('Model cleanup failed: ' + type(error).__name__)
    if primary is not None:
        try:
            runner._record_pending_rejected(primary)
        except (KeyboardInterrupt, SystemExit) as error:
            chained, primary = primary, error
        try:
            journal.finish('cancelled' if isinstance(primary, (KeyboardInterrupt, SystemExit)) else 'failed',
                           {'error_type': type(primary).__name__, 'forwards_started': runner._forwards_started,
                            'model_returns': runner._model_returns,
                            'forward_evidence_records': runner._forwards_recorded,
                            'cleanup_quarantined': runner.quarantined})
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error from primary
            if hasattr(primary, 'add_note'):
                primary.add_note('Failure terminal could not be written: ' + type(error).__name__)
        if chained is not None:
            raise primary from chained
        raise primary
    journal.finish('complete', result)
    return result
