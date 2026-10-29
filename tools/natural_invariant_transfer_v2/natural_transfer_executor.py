"""Fixed-site natural donor-patch worker on a caller-authorized CPU model.

The caller must verify the public protocol, mapping, source, exposure, closed
prior receipt/audit/model and frozen discovery before granting a phase. This
worker does not load a checkpoint or establish process/publication custody.
"""
import hashlib
import os
import re
from pathlib import Path

import torch

from looking_glass.agreement_direction_study_v1 import _act
from looking_glass.agreement_journal import AgreementJournal
from looking_glass.agreement_patch_runner import AgreementCase, StudyFailure, _same_state_identity, _site, _state_identity
from looking_glass.agreement_substitution_executor_v1 import _SubjectRunner, _finite_captures
from looking_glass.distilgpt2_intervention_dev import _hook_state
from looking_glass.intervention_plan import ActivationPlan
from looking_glass.model_ownership import ModelLease

from natural_transfer_numeric import InvalidPlan, ensure_round_to_nearest, plan_candidates

RAW_LIMIT = 64 * 1024 * 1024
METADATA_LIMIT = 8 * 1024 * 1024
HEX = re.compile(r'[0-9a-f]{64}\Z')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class _PrePatchGuard(ActivationPlan):
    """Reject a changed receiver at the hook before the replacement occurs."""
    def __init__(self, runner, steps):
        self.runner = runner
        super().__init__(runner.model, steps, max_tensor_bytes=1024 * 1024,
                         max_capture_bytes=512 * 1024, max_source_bytes=128 * 1024,
                         max_steps=16, max_invocations=1, _parent_lease=runner._lease)

    def _apply_step(self, step, output, preceding):
        if step['name'] == 'action' and step['operation'] == 'patch':
            before = self._captures.get('before')
            if before is None:
                raise StudyFailure('pre-patch guard has no preceding capture')
            actual = self.runner._encode_payload(before)
            self.runner._guard_observation = actual
            if actual != self.runner._guard_expected:
                raise StudyFailure('prior receiver differs before patch')
            self.runner._guard_hits += 1
        return super()._apply_step(step, output, preceding)


class _TransferRunner(_SubjectRunner):
    def __init__(self, model, journal, custody_check):
        super().__init__(model, journal, custody_check=custody_check)
        self._guard_expected = None
        self._guard_observation = None
        self._guard_hits = 0

    def _new_plan(self, steps):
        return _PrePatchGuard(self, steps)


def _validate_inputs(model, mapping, prior, phase):
    if phase not in ('discovery', 'evaluation'):
        raise ValueError('invalid fixed phase')
    if type(mapping) is not dict or type(mapping.get('receivers')) is not list:
        raise ValueError('invalid mapping')
    rows = [row for row in mapping['receivers'] if row.get('split') == phase]
    if len(rows) != 24 or len({row.get('receiver_case_id') for row in rows}) != 24:
        raise ValueError('phase must contain 24 distinct receiver rows')
    width, vocab = model.config.n_embd, model.config.vocab_size
    if type(width) is not int or type(vocab) is not int or width <= 0 or vocab <= 0:
        raise ValueError('invalid model dimensions')
    seen = set()
    for row in rows:
        case_id, tokens = row['receiver_case_id'], row['receiver_token_ids']
        if (type(tokens) is not list or not 1 <= len(tokens) <= 12
                or any(type(x) is not int or not 0 <= x < vocab for x in tokens)
                or row['final_index'] != len(tokens) - 1):
            raise ValueError('invalid receiver row')
        for key, donor_key in (('Q1', 'donor_case_id'), ('Q2', 'donor_case_id'),
                               ('Q3', 'source_donor_case_id'), ('Q4', 'donor_case_id')):
            candidate = row['candidates'][key]
            donor_id = candidate[donor_key]
            if type(donor_id) is not str or donor_id not in prior:
                raise ValueError('missing candidate donor')
        if row['candidates']['Q3']['source_donor_case_id'] != row['candidates']['Q2']['donor_case_id']:
            raise ValueError('rotation must use the direct donor')
        for name in (case_id, row['candidates']['Q1']['donor_case_id'],
                     row['candidates']['Q2']['donor_case_id'], row['candidates']['Q4']['donor_case_id']):
            item = prior.get(name)
            if (type(item) is not dict or type(item.get('vector_f32')) is not bytes
                    or len(item['vector_f32']) != width * 4
                    or item.get('vector_sha256') != _sha(item['vector_f32'])
                    or type(item.get('last_logits_f32')) is not bytes
                    or len(item['last_logits_f32']) != vocab * 4
                    or item.get('logits_sha256') != _sha(item['last_logits_f32'])):
                raise ValueError('missing or changed prior baseline bytes')
        seen.add(case_id)
    return rows


def _write_exclusive(path, data):
    with open(path, 'xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short displacement write')
        stream.flush()
        os.fsync(stream.fileno())


def _observe_sidecars(directory):
    """Report actual retained files, including partial writes; uncertainty stays explicit."""
    if directory is None or not directory.exists():
        return {'complete': True, 'bytes': 0, 'files': []}
    try:
        files = []
        for path in sorted(directory.iterdir(), key=lambda x: x.name):
            if not path.is_file() or path.is_symlink():
                return {'complete': False, 'bytes': None, 'files': [], 'error': 'unexpected entry'}
            data = path.read_bytes()
            files.append({'name': path.name, 'bytes': len(data), 'sha256': _sha(data)})
        return {'complete': True, 'bytes': sum(x['bytes'] for x in files), 'files': files}
    except (OSError, ValueError) as error:
        return {'complete': False, 'bytes': None, 'files': [], 'error': type(error).__name__}


def run_phase(model, journal, mapping, prior, phase, *, prior_model_sha256,
              binding, custody_check):
    """Execute exactly one granted 96-action phase or retain a failed attempt."""
    if type(journal) is not AgreementJournal or not callable(custody_check):
        raise ValueError('bounded journal and custody check required')
    names = {'pre_run_sha256', 'mapping_sha256', 'prior_receipt_sha256', 'latest_exposure_sha256'}
    if phase == 'evaluation':
        names.add('freeze_sha256')
    if (type(binding) is not dict or set(binding) != names
            or any(type(x) is not str or HEX.fullmatch(x) is None for x in binding.values())
            or type(prior_model_sha256) is not str or HEX.fullmatch(prior_model_sha256) is None):
        raise ValueError('invalid external source or evidence binding')
    if (journal._events != 0 or journal.retained_bytes != 0 or journal._known or journal._faulted
            or journal._max_total != RAW_LIMIT or journal._max_event_log_bytes != METADATA_LIMIT
            or journal._max_events < 100 or journal._max_forward < 4 * (12 * model.config.vocab_size
                 + 2 * 12 * model.config.n_embd + 2 * model.config.n_embd)):
        raise ValueError('fresh journal with fixed phase budgets required')
    rows = _validate_inputs(model, mapping, prior, phase)
    runner = _TransferRunner(model, journal, custody_check)
    runner._used = True
    lease = ModelLease(model)
    runner._lease = lease
    old_rng = initial_state = primary = chained = result = None
    sidecar_bytes = 0
    sidecars = None
    active = None
    try:
        ensure_round_to_nearest()
        custody_check()
        lease.acquire()
        runner._baseline_contract = runner._execution_contract()
        runner._baseline_hooks = _hook_state(model)
        if (any(any(registry for registry in module[1:]) for module in runner._baseline_hooks[0])
                or any(runner._baseline_hooks[1:])):
            raise StudyFailure('preexisting hooks are unsupported')
        initial_state = runner._model_digest()
        if initial_state != prior_model_sha256:
            raise StudyFailure('model state differs from prior evidence')
        runner._baseline_state_identity = _state_identity(model)
        old_rng = torch.random.get_rng_state()
        torch.random.default_generator.manual_seed(20261030)
        sidecars = Path(journal._directory) / 'displacements'
        sidecars.mkdir()
        journal.record({'kind': 'run_contract', 'phase': phase, 'expected_forwards': 96,
                        'model_state_sha256': initial_state, 'binding': dict(binding),
                        'raw_limit': RAW_LIMIT, 'metadata_limit': METADATA_LIMIT}, {})
        for row in rows:
            receiver = row['receiver_case_id']
            q1 = row['candidates']['Q1']['donor_case_id']
            q2 = row['candidates']['Q2']['donor_case_id']
            q4 = row['candidates']['Q4']['donor_case_id']
            base, direct, lexical, other = (prior[key]['vector_f32'] for key in (receiver, q2, q1, q4))
            case = AgreementCase(receiver, f"{phase}-{row['frame_index']}",
                                 f"noun-{row['noun_index']}",
                                 'plural' if row['determiner'] == 'these' else 'singular',
                                 tuple(row['receiver_token_ids']), row['subject_index'])
            try:
                plans = plan_candidates(base, direct, lexical, other)
            except InvalidPlan as error:
                if journal.retained_bytes + sidecar_bytes + 4 * len(base) > RAW_LIMIT:
                    raise StudyFailure('insufficient reserve for invalid-plan evidence') from error
                journal.record({'kind': 'invalid_plan', 'case_id': receiver, 'error_type': type(error).__name__,
                                'error': str(error), 'donor_ids': [q1, q2, q4]},
                               {name: ((model.config.n_embd,), raw) for name, raw in
                                (('receiver', base), ('direct_donor', direct),
                                 ('lexical_donor', lexical), ('other_donor', other))})
                raise StudyFailure('invalid fixed donor plan') from error
            for key in ('Q1', 'Q2', 'Q3', 'Q4'):
                active = {'case_id': receiver, 'candidate': key}
                plan = plans['candidates'][key]
                h = plan['displacement_f64']
                reserve = 4 * (len(case.input_ids) * model.config.vocab_size
                               + 2 * len(case.input_ids) * model.config.n_embd + 2 * model.config.n_embd)
                if journal.retained_bytes + sidecar_bytes + len(h) + reserve + 2 * len(base) > RAW_LIMIT:
                    raise StudyFailure('insufficient pre-forward raw evidence reserve')
                custody_check()
                path = sidecars / f"{runner._forwards_started:03d}-{key}.f64le"
                _write_exclusive(path, h)
                sidecar_bytes += len(h)
                dose = {'value': plan['replacement_f32'], 'dose_valid': plan['certificate']['pass'],
                        'requested_norm': plans['q'], 'certificate': plan['certificate'],
                        'displacement_file': 'displacements/' + path.name, 'displacement_sha256': _sha(h),
                        'receiver_vector_sha256': prior[receiver]['vector_sha256'],
                        'donor_vector_sha256': prior[q2 if key in ('Q2', 'Q3') else q1 if key == 'Q1' else q4]['vector_sha256']}
                runner._guard_expected, runner._guard_observation, runner._guard_hits = base, None, 0
                action = {'index': runner._forwards_started, 'phase': phase, 'case_id': receiver,
                          'candidate': key, 'receiver': receiver,
                          'donor': q2 if key in ('Q2', 'Q3') else q1 if key == 'Q1' else q4}
                _, captures = _act(runner, case, action, None, site=_site(3, 'attn'),
                                   base=base, candidate=dose, donor=action['donor'])
                if runner._guard_hits != 1:
                    raise StudyFailure('pre-patch guard did not run exactly once')
                _finite_captures(captures)
        if (runner._forwards_started != 96 or runner._model_returns != 96
                or runner._forwards_recorded != 96 or runner._forwards != 96):
            raise StudyFailure('fixed phase forward counts differ')
        if runner._model_digest() != initial_state or not _same_state_identity(runner._baseline_state_identity, model):
            raise StudyFailure('model state or identity changed')
        ensure_round_to_nearest()
        sidecar_inventory = _observe_sidecars(sidecars)
        if not sidecar_inventory['complete'] or sidecar_inventory['bytes'] != sidecar_bytes:
            raise StudyFailure('displacement inventory is incomplete or changed')
        result = {'status': 'complete_candidate', 'phase': phase, 'forwards': 96,
                  'model_state_sha256_before_after': initial_state,
                  'retained_tensor_bytes': journal.retained_bytes, 'displacement_bytes': sidecar_bytes,
                  'displacement_inventory': sidecar_inventory}
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
            if (runner._guard_observation is not None and runner._guard_expected is not None
                    and runner._guard_observation != runner._guard_expected):
                journal.record({'kind': 'prior_mismatch', **(active or {})},
                               {'expected': ((model.config.n_embd,), runner._guard_expected),
                                'actual': ((model.config.n_embd,), runner._guard_observation)})
        except (KeyboardInterrupt, SystemExit) as error:
            chained, primary = primary, error
        except BaseException as error:
            if hasattr(primary, 'add_note'):
                primary.add_note('Mismatch evidence could not be recorded: ' + type(error).__name__)
        try:
            journal.finish('cancelled' if isinstance(primary, (KeyboardInterrupt, SystemExit)) else 'failed',
                           {'error_type': type(primary).__name__, 'forwards_started': runner._forwards_started,
                            'model_returns': runner._model_returns, 'forward_evidence_records': runner._forwards_recorded,
                            'cleanup_quarantined': runner.quarantined, 'active': active,
                            'displacement_bytes_successfully_written': sidecar_bytes,
                            'displacement_inventory': _observe_sidecars(sidecars)})
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
