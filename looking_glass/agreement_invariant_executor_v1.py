"""Bounded invariant-noun worker on an already authorized CPU model.

This worker cannot prepare cases, load a checkpoint, authorize forwarding or
observe its own process exit. The supervised caller must bind independently
reviewed source, public inputs, exposure and custody observations first.
"""

from collections import Counter
import hashlib
import json
import re

import torch

from . import agreement_invariant_schedule_v1 as plan
from .agreement_direction_study_v1 import _act, _checked_directions, _reserve_candidate_evidence
from .agreement_journal import AgreementJournal
from .agreement_patch_runner import AgreementCase, StudyFailure, _same_state_identity, _site, _state_identity
from .agreement_substitution_executor_v1 import _SubjectRunner, _finite_captures, signed_direction_patch
from .distilgpt2_intervention_dev import _hook_state
from .model_ownership import ModelLease

ACTION_SHA = 'fbf526fd867ed48127c24b3b35004727b1757782e90798d60dcfdec37ac9b2f6'


def fixed_schedule(cases):
    """Validate identity/metadata and derive the fixed 612 action records."""
    ordered = plan.cases()
    if type(cases) not in (tuple, list) or len(cases) != 68:
        raise ValueError('expected 68 cases')
    result = []
    for case_index, (row, (expected, arms)) in enumerate(zip(cases, ordered)):
        if type(row) is not dict or row.get('case_id') != expected:
            raise ValueError('case order or identity differs')
        target = re.fullmatch(r'(discovery|evaluation)_target_f([01])_c([012])_n([01])_(this|these)', expected)
        fragment = re.fullmatch(r'(discovery|evaluation)_fragment_f([01])_c([012])', expected)
        demonstrative = re.fullmatch(r'(discovery|evaluation)_demonstrative_f([01])_(this|these)', expected)
        if target:
            split, f, c, n, det = target.groups()
            wanted = (split, 'target', int(f), int(c), int(n), det)
        elif fragment:
            split, f, c = fragment.groups()
            wanted = (split, 'fragment', int(f), int(c), None, None)
        elif demonstrative:
            split, f, det = demonstrative.groups()
            wanted = (split, 'demonstrative', int(f), 0, None, det)
        else:
            raise ValueError('unexpected fixed case identity')
        keys = ('split', 'role', 'frame_index', 'context_index', 'noun_index', 'determiner')
        if tuple(row.get(key) for key in keys) != wanted:
            raise ValueError('case coordinate metadata differs')
        if type(row['frame_index']) is not int or type(row['context_index']) is not int:
            raise ValueError('case coordinates must be integers')
        if row['noun_index'] is not None and type(row['noun_index']) is not int:
            raise ValueError('noun coordinate must be an integer')
        common = {'case_id': expected, 'case_index': case_index,
                  **{key: row.get(key) for key in keys}}
        for action, slot in [('clean', None), ('noop', None),
                             *[(arm, slot) for slot, arm in enumerate(arms)], ('clean_after', None)]:
            result.append({'index': len(result), **common, 'action': action, 'slot': slot})
    expected_actions = [(row['case_id'], row['action']) for row in result]
    raw = json.dumps(expected_actions, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    if hashlib.sha256(raw).hexdigest() != ACTION_SHA:
        raise ValueError('fixed action digest differs')
    return tuple(result)


def _validated_cases(cases, vocabulary):
    fixed_schedule(cases)
    seen, converted, by_id = set(), {}, {}
    for row in cases:
        tokens, text = row.get('token_ids'), row.get('text')
        if (type(text) is not str or not text or len(text) > 512
                or type(tokens) is not list or not 1 <= len(tokens) <= 12
                or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
                or type(row.get('final_index')) is not int or row['final_index'] != len(tokens) - 1):
            raise ValueError('invalid input or final coordinate')
        current = tuple(tokens)
        if any(current[:len(prior)] == prior or prior[:len(current)] == current for prior in seen):
            raise ValueError('input rows overlap')
        seen.add(current)
        if row['role'] in ('target', 'demonstrative'):
            d = row.get('determiner_index')
            expected = 428 if row['determiner'] == 'this' else 777
            if type(d) is not int or not 0 <= d < len(tokens) - 1 or tokens[d] != expected:
                raise ValueError('invalid determiner coordinate')
        elif row.get('determiner_index') is not None:
            raise ValueError('fragment cannot have a determiner coordinate')
        subject = row.get('subject_index')
        if row['role'] == 'target':
            if type(subject) is not int or not 0 <= subject < len(tokens) - 1 or subject == d:
                raise ValueError('invalid target noun coordinate')
        elif subject is not None:
            raise ValueError('non-target cannot have a noun coordinate')
        converted[row['case_id']] = AgreementCase(
            row['case_id'], f"{row['split']}-{row['frame_index']}",
            f"noun-{row['noun_index']}" if row['role'] == 'target' else row['role'],
            'plural' if row['determiner'] == 'these' else 'singular',
            current, subject if subject is not None else 0)
        by_id[row['case_id']] = row
    for split in plan.SPLITS:
        for f in range(2):
            x = by_id[f'{split}_demonstrative_f{f}_this']
            y = by_id[f'{split}_demonstrative_f{f}_these']
            d = x['determiner_index']
            if (y['determiner_index'] != d or len(x['token_ids']) != len(y['token_ids'])
                    or x['token_ids'][:d] + x['token_ids'][d + 1:]
                    != y['token_ids'][:d] + y['token_ids'][d + 1:]):
                raise ValueError('demonstrative pair differs outside determiner')
            for c in range(3):
                rows = [by_id[f'{split}_target_f{f}_c{c}_n{n}_{det}']
                        for n in range(2) for det in ('this', 'these')]
                first = rows[0]
                d, n = first['determiner_index'], first['subject_index']
                for row in rows:
                    if (len(first['token_ids']) != len(row['token_ids'])
                            or row['determiner_index'] != d or row['subject_index'] != n
                            or any(a != b for i, (a, b) in enumerate(zip(first['token_ids'], row['token_ids']))
                                   if i not in (d, n))):
                        raise ValueError('target group differs outside determiner or noun')
                if (rows[0]['token_ids'][n] != rows[1]['token_ids'][n]
                        or rows[2]['token_ids'][n] != rows[3]['token_ids'][n]
                        or rows[0]['token_ids'][n] == rows[2]['token_ids'][n]):
                    raise ValueError('noun pairing differs')
            for n in range(2):
                for det in ('this', 'these'):
                    a = by_id[f'{split}_target_f{f}_c1_n{n}_{det}']['token_ids']
                    b = by_id[f'{split}_target_f{f}_c2_n{n}_{det}']['token_ids']
                    if len(a) != len(b) or a[:-1] != b[:-1] or a[-1] == b[-1]:
                        raise ValueError('target attractors differ outside last position')
            a = by_id[f'{split}_fragment_f{f}_c1']['token_ids']
            b = by_id[f'{split}_fragment_f{f}_c2']['token_ids']
            if len(a) != len(b) or a[:-1] != b[:-1] or a[-1] == b[-1]:
                raise ValueError('fragment attractors differ outside last position')
    return converted


def _check_budget(journal, vocabulary, width):
    journal._check_owner()
    if (journal._faulted or journal._events != 0 or journal._event_log_bytes != 0
            or journal.retained_bytes != 0 or journal._known):
        raise ValueError('a fresh empty journal is required')
    success = 612 * vocabulary * 4 + 476 * (2 * 12 * width * 4 + 2 * width * 4)
    failure = 4 * (12 * vocabulary + 2 * 12 * width + 2 * width)
    candidate = 2 * width * 4
    if (journal._max_forward < failure or journal._max_total < success + failure + candidate
            or journal._max_events < 614 or journal._max_event_log_bytes < 32 * 1024 * 1024):
        raise ValueError('insufficient evidence reservation before execution')
    return {'successful_logical_bytes_upper_bound': success,
            'failed_return_bytes_reserve': failure, 'invalid_candidate_bytes_reserve': candidate,
            'retained_tensor_bytes_limit': journal._max_total,
            'per_forward_bytes_limit': journal._max_forward,
            'event_log_bytes_limit': journal._max_event_log_bytes}


def run_invariant_study(model, journal, cases, directions, *, binding, custody_check=None):
    """Run exactly one fixed plan and return a candidate after clean teardown."""
    if type(journal) is not AgreementJournal:
        raise ValueError('expected bounded agreement journal')
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
        counts = Counter((row['action'], row['slot']) for row in schedule if row['slot'] is not None)
        placement = {arm: [counts[(arm, slot)] for slot in range(6)] for arm in plan.BASE}
        journal.record({'kind': 'run_contract', 'expected_forwards': 612,
                        'environment': runner._environment, 'seed': 20261030,
                        'execution_configuration': runner._baseline_contract[2],
                        'model_state_sha256': initial_state, 'external_binding': dict(binding),
                        'schedule_sha256': ACTION_SHA, 'arm_placement_counts': placement,
                        'evidence_budget': budget}, {})
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
        if (runner._forwards_started != 612 or runner._model_returns != 612
                or runner._forwards_recorded != 612 or runner._forwards != 612):
            raise StudyFailure('forward inventory differs')
        if runner._model_digest() != initial_state or not _same_state_identity(runner._baseline_state_identity, model):
            raise StudyFailure('model state or buffer identity changed')
        result = {'status': 'complete_candidate', 'forwards': 612,
                  'model_state_sha256_before_after': initial_state,
                  'retained_tensor_bytes': journal.retained_bytes, 'schedule_sha256': ACTION_SHA}
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
                            'model_returns_scope': 'normal-path observations only; a rejected forward may retain a returned output',
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
