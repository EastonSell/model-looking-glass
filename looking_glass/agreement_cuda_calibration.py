"""Restricted, provisional calibration path for the agreement CUDA backend.

This measures an already-exposed public pair; it is not a study runner, a
launcher, an admission profile, or evidence of a worst-case bound. A separate
supervised collector must supply live device/model/host commitments and sample
the whole callback. Never use this path for held-out inputs or publish its
temporary logits or donor activations. No device work may begin until the
supervisor and this exact path have been reviewed and an owner window scheduled.
"""

import sys

import torch

from .agreement_cuda_runner import (CALIBRATION_DONOR_IDS, CALIBRATION_INPUT_IDS,
    CALIBRATION_KINDS, GIB, WARMUP_POLICY_SHA256, _CALIBRATION_ACTIONS,
    CudaAgreementRunner, _validate_workload_key, calibration_case_sha256)
from .agreement_patch_runner import (AgreementCase, StudyFailure, _RNG_SEED, _SITES,
    _error_message, _same_state_identity, _state_identity, _steps_for_action,
    _steps_for_capture, _tensor_from_bytes, _unchanged_outside_target)
from .distilgpt2_intervention_dev import _hook_state
from .memory_budget import MemoryLease
from .model_ownership import ModelLease


PROVISIONAL_JOB_BYTES = 3 * GIB // 2
PROVISIONAL_RESIDENT_BYTES = 2 * GIB


class CudaAgreementCalibrator(CudaAgreementRunner):
    """Only the fixed cold warmup and 13 ordered warm resource callbacks.

    The 1.5 GiB job and 2 GiB resident commitments are provisional, not
    measurements or bounds. A trusted owner must separately account for loading, persistent allocations,
    host scratch and native CUDA memory. The inherited owner gate and sticky
    quarantine apply exactly as for the measured study backend.

    The collector must treat an exception or uncertain return from *any*
    callback as terminal, call ``quarantine_after_uncertain_callback`` if it
    can, retain the claims, and terminate the owner without retry or clean
    attestation. An interrupt can occur at a callback return after its internal
    handler exits, so the runner's health flag alone cannot certify delivery.
    """

    def __init__(self, model, journal, *, session, device_lease, memory_lease,
                 resident_lease, memory_probe, expected_key, cancel_event=None):
        _validate_workload_key(expected_key)
        if (type(memory_lease) is not MemoryLease or type(resident_lease) is not MemoryLease
                or resident_lease is memory_lease or resident_lease._budget is not memory_lease._budget
                or resident_lease.released or resident_lease not in resident_lease._budget.active_leases()
                or resident_lease.estimated_incremental_bytes < PROVISIONAL_RESIDENT_BYTES):
            raise ValueError('a distinct active provisional resident commitment is required')
        self._resident_lease = resident_lease
        try:
            self._initialize_owner(model, journal, session=session, device_lease=device_lease,
                memory_lease=memory_lease, memory_probe=memory_probe, expected_key=expected_key,
                required_incremental_bytes=PROVISIONAL_JOB_BYTES, profile=None, warmup_profile=None,
                expected_quartets=1, cancel_event=cancel_event)
            self._calibration_index = 0
            self._calibration_donors = None
            self._calibration_in_callback = False
        except BaseException:
            self.quarantined = True
            raise

    def _check_owner(self, *, allow_quarantined=False, allow_pending=False):
        super()._check_owner(allow_quarantined=allow_quarantined, allow_pending=allow_pending)
        resident = self._resident_lease
        if (type(resident) is not MemoryLease or resident.released
                or resident._budget is not self._memory_lease._budget
                or resident not in resident._budget.active_leases()
                or resident.estimated_incremental_bytes < PROVISIONAL_RESIDENT_BYTES):
            raise StudyFailure('the provisional resident commitment is no longer held')

    def run(self, *args, **kwargs):
        raise StudyFailure('the provisional calibration owner cannot run a study')

    def _execution_contract(self):
        if not self._calibration_in_callback:
            raise StudyFailure('the provisional calibration owner cannot run a study')
        return super()._execution_contract()

    def warmup(self):
        if self._calibration_in_callback:
            self.quarantined = True
            raise StudyFailure('calibration callbacks cannot overlap')
        self._calibration_in_callback = True
        try:
            return super().warmup()
        finally:
            self._calibration_in_callback = False

    def _runtime_metadata(self):
        return {'device': self.device, 'device_uuid': self._expected_key.device_uuid,
                'workload_sha256': self._expected_key.fingerprint(),
                'bootstrap': 'provisional', 'job_commitment_bytes': PROVISIONAL_JOB_BYTES,
                'resident_commitment_bytes': PROVISIONAL_RESIDENT_BYTES,
                'warmup_policy_sha256': WARMUP_POLICY_SHA256}

    def quarantine_after_uncertain_callback(self):
        """Latch a collector-observed exception without touching CUDA."""
        self.quarantined = True

    def synchronize_for_measurement(self):
        try:
            self._sync()
        except BaseException:
            self.quarantined = True
            raise

    def measurement_snapshot(self):
        try:
            self._check_headroom()
            result = self._memory_probe.snapshot()
            self._check_owner()
            return result
        except BaseException:
            self.quarantined = True
            raise

    def reset_measurement_peak(self):
        try:
            self._check_headroom()
            self._memory_probe.reset_peak()
            self._check_owner()
        except BaseException:
            self.quarantined = True
            raise

    def measurement_peaks(self):
        try:
            self._check_owner()
            allocated = self._memory_probe.peak_allocated_bytes()
            self._check_owner()
            reserved = self._memory_probe.peak_reserved_bytes()
            self._check_owner()
            return allocated, reserved
        except BaseException:
            self.quarantined = True
            raise

    def calibrate_action(self, kind):
        """Run one fixed calibration case; never return model values."""
        if self._calibration_in_callback:
            self.quarantined = True
            raise StudyFailure('calibration callbacks cannot overlap')
        self._calibration_in_callback = True
        try:
            return self._calibrate_action(kind)
        finally:
            self._calibration_in_callback = False

    def _calibrate_action(self, kind):
        """Execute one exact action; return plan metadata, never model values.

        The collector must synchronize/reset/sample around this entire call.
        A failed callback quarantines the owner and retains all commitments;
        the collector must not retry a failed or uncertain CUDA drain.
        """
        if (not self._warmup_completed or self.quarantined or self._completed
                or self._calibration_index >= len(CALIBRATION_KINDS)
                or kind != CALIBRATION_KINDS[self._calibration_index]):
            self.quarantined = True
            raise StudyFailure('calibration action is out of order or owner is unavailable')
        self._started_work = False
        lease = ModelLease(self.model)
        self._lease = lease
        initial_state = None
        rng_before = None
        try:
            try:
                lease.acquire()
                self._baseline_contract = self._execution_contract()
                self._baseline_hooks = _hook_state(self.model)
                if (any(any(registry for registry in module[1:]) for module in self._baseline_hooks[0])
                        or any(self._baseline_hooks[1:])):
                    raise StudyFailure('preexisting local or global hooks are unsupported')
                initial_state = self._model_digest()
                self._baseline_state_identity = _state_identity(self.model)
                rng_before = torch.random.get_rng_state()
                torch.random.default_generator.manual_seed(_RNG_SEED)
                self._seed_device_rng()
                donor = AgreementCase('calibration-donor', 'calibration', 'cat', 'plural',
                                      CALIBRATION_DONOR_IDS, 1)
                target = AgreementCase('calibration-target', 'calibration', 'cat', 'singular',
                                       CALIBRATION_INPUT_IDS, 1)
                _, path, operation, token = _CALIBRATION_ACTIONS[self._calibration_index]
                if kind == 'clean_capture':
                    logits, captures, metadata = self._forward(donor, _steps_for_capture(donor))
                    if len(captures) != len(_SITES):
                        raise StudyFailure('calibration donor capture is incomplete')
                    self._calibration_donors = {site['id']: captures[f'capture_{index}']
                                                for index, site in enumerate(_SITES)}
                    checked = True
                else:
                    if self._calibration_donors is None:
                        raise StudyFailure('same-owner calibration donor is unavailable')
                    site = next(site for site in _SITES
                                if site['module_path'] == path and
                                (target.subject_index if site['token_role'] == 'subject'
                                 else len(target.input_ids) - 1) == token)
                    source = self._calibration_donors[site['id']] if operation == 'cross_patch' else None
                    step_operation = 'null' if operation == 'null' else 'patch'
                    tensor = _tensor_from_bytes(source) if source is not None else None
                    logits, captures, metadata = self._forward(
                        target, _steps_for_action(target, site, step_operation, tensor))
                    expected = captures['before'] if source is None else source
                    checked = (captures['after'] == expected
                               and _unchanged_outside_target(captures, target, site,
                                                             self.model.config.n_embd))
                    if not checked:
                        raise StudyFailure('calibration patch landing or untouched positions differ')
                # These are CPU bytes at this point; do not record or return them.
                self._pending_forward = None
                del logits, captures
                self._sync()
                if not _same_state_identity(self._baseline_state_identity, self.model):
                    raise StudyFailure('model identity changed during calibration')
                record = {'sample_id': kind, 'case_sha256': calibration_case_sha256(kind),
                          'seed': _RNG_SEED, 'plan_metadata': metadata,
                          'capture_or_landing_checked': checked}
                self.journal.record({'kind': 'gpu_calibration_action', **record}, {})
            finally:
                primary = sys.exc_info()[1]
                try:
                    if rng_before is not None:
                        try:
                            try:
                                self._restore_device_rng()
                            finally:
                                torch.random.set_rng_state(rng_before)
                            if not torch.equal(torch.random.get_rng_state(), rng_before):
                                raise StudyFailure('CPU RNG state restoration failed')
                        except BaseException as error:
                            self.quarantined = True
                            if primary is None:
                                raise
                            if hasattr(primary, 'add_note'):
                                primary.add_note('Calibration RNG cleanup failed: ' + type(error).__name__)
                finally:
                    active = sys.exc_info()[1] or primary
                    try:
                        self._release_after_work(lease, initial_state, active)
                    except BaseException as error:
                        self.quarantined = True
                        try:
                            lease._quarantine()
                        except BaseException:
                            pass
                        if active is None:
                            raise
                        if hasattr(active, 'add_note'):
                            active.add_note('Calibration lease cleanup failed: ' + type(error).__name__)
            self._calibration_index += 1
            if self._calibration_index == len(CALIBRATION_KINDS):
                self._calibration_donors = None
                self._completed = True
            return record
        except BaseException as error:
            self.quarantined = True
            try:
                self.journal.finish('failed', {'error_type': type(error).__name__,
                    'message': _error_message(error), 'stage': 'GPU calibration', 'action': kind})
            except BaseException:
                pass
            raise
