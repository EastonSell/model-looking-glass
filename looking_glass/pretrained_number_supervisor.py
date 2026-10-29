"""CPU discovery composition with private journal and local public archive.

This module does not establish publication order, actual checkpoint identity,
runtime provenance, or secrecy from same-account code. A reviewed launcher
must verify and record those facts before invoking it. No held-out cases are
drawn here. All failure details and journal receipts remain private.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

from .distilgpt2_intervention_dev import _model_state_sha256
from .pretrained_exposures_v2 import PUBLIC_DISCOVERY
from .pretrained_number_adapter import to_selector_observation
from .pretrained_number_archive import _CONDITIONS, _ROLES, _read, audit_discovery_archive, serialize_discovery_observation
from .pretrained_number_commitments import verify_site_reveal
from .pretrained_number_incomplete import encode_incomplete_forward
from .pretrained_number_injection import NumberInjectionExecutor
from .pretrained_number_journal import DiscoveryJournal, PairReceipt, inspect_discovery_journal
from .pretrained_number_requests import load_family_request_plan
from .pretrained_number_selection import DiscoveryCase, freeze_discovery
from .pretrained_number_witness import _required_operations, encode_private_forward


_MIB = 1024 * 1024
_FORWARD_RESERVE = _MIB
_FINAL_DISK_RESERVE = 10 * _MIB  # 8 MiB prediction plus bounded metadata/failure.
_STUDY_LIMIT = 512 * _MIB
_DISK_LIMIT = 1024 * _MIB


@dataclass(frozen=True)
class DiscoveryResult:
    status: str
    record_count: int
    prediction_sha256: str
    archive_sha256: str
    retained_tensor_bytes: int
    logical_limit_bytes: int
    disk_limit_bytes: int


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _close_after(fd, operation):
    try:
        result = operation()
    except BaseException as primary:
        try:
            os.close(fd)
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
    else:
        os.close(fd)
        return result


def _write_and_close(fd, data):
    def write():
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            if type(n) is not int or n <= 0 or n > len(view):
                raise OSError('incomplete evidence write')
            view = view[n:]
        os.fsync(fd)
    _close_after(fd, write)


def _write_new(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    _write_and_close(os.open(path, flags, 0o600), data)


def _receipt_bytes(receipt):
    return _canonical({'ordinal': receipt.ordinal, 'frame_sha256': receipt.frame_sha256,
                       'public_sha256': receipt.public_sha256}) + b'\n'


def _append_receipt(path, receipt):
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, 'O_BINARY', 0)
    _write_and_close(os.open(path, flags), _receipt_bytes(receipt))


def _read_receipts(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
    def read():
        result = bytearray()
        while len(result) <= 128 * 512:
            block = os.read(fd, 128 * 512 + 1 - len(result))
            if not block:
                break
            result.extend(block)
        return bytes(result)
    raw = _close_after(fd, read)
    if len(raw) > 128 * 512 or not raw.endswith(b'\n'):
        raise ValueError('receipt sidecar is incomplete or exceeds its bound')
    receipts = []
    for ordinal, line in enumerate(raw.splitlines()):
        value = json.loads(line)
        if (type(value) is not dict or set(value) != {'ordinal', 'frame_sha256', 'public_sha256'}
                or type(value['ordinal']) is not int or value['ordinal'] != ordinal
                or _canonical(value) != line):
            raise ValueError('receipt sidecar order or format differs')
        receipts.append(PairReceipt(value['ordinal'], value['frame_sha256'], value['public_sha256']))
    return tuple(receipts)


def _cases(records):
    observations = {}
    for raw in records:
        index, role, observation = _read(raw, 'cpu')
        if (index, role) in observations:
            raise ValueError('duplicate public observation')
        observations[index, role] = observation
    cases = []
    for index in range(8):
        take = lambda role: observations[index, role]
        cases.append(DiscoveryCase(index, take('clean'), take('repeat'), take('injected'),
                     tuple(take(f'forward_{i}') for i in range(1, 5)),
                     tuple(take(f'reverse_{i}') for i in range(1, 5)), take('null'), take('self'),
                     take('upstream'), take('downstream'), take('verify_clean')))
    return tuple(cases)


def _request_arguments(role, baselines):
    clean, injected = baselines
    if role.startswith('forward_'):
        layer = int(role[-1])
        return {'layer': layer, 'donor': clean.mlp_donor(layer)}
    if role.startswith('reverse_'):
        layer = int(role[-1])
        return {'layer': layer, 'donor': injected.mlp_donor(layer)}
    if role == 'upstream':
        return {'donor': clean.mlp_donor(0)}
    if role == 'downstream':
        return {'donor': clean.block_donor()}
    return {}


def run_discovery(*, model, attempt_dir, plan_bytes, source_commit, plan_sha256, runtime_sha256,
                  site_nonce, site_record_bytes, site_commitment,
                  logical_limit_bytes, disk_limit_bytes):
    """Execute one local, non-resumable discovery attempt on an owned CPU model.

    Limits are this phase's reserved shares of the global 512 MiB logical
    tensor cap and 1 GiB physical evidence allowance. The caller must reserve
    them across all tracks and check the whole-study free-space requirement.
    """
    if (type(logical_limit_bytes) is not int or not _FORWARD_RESERVE <= logical_limit_bytes <= _STUDY_LIMIT
            or type(disk_limit_bytes) is not int or not _FINAL_DISK_RESERVE <= disk_limit_bytes <= _DISK_LIMIT):
        raise ValueError('invalid phase resource limits')
    attempt = Path(attempt_dir)
    if not attempt.is_absolute() or not attempt.parent.is_dir() or attempt.exists():
        raise ValueError('a fresh absolute attempt directory is required')
    if shutil.disk_usage(attempt.parent).free < disk_limit_bytes:
        raise ValueError('insufficient free space for the reserved evidence allowance')
    plan = load_family_request_plan(plan_bytes, expected_source_commit=source_commit, expected_sha256=plan_sha256)
    site = verify_site_reveal(site_nonce, site_record_bytes, site_commitment, expected_source_commit=source_commit)
    attempt.mkdir(mode=0o700)
    private, public = attempt / 'private', attempt / 'public'
    private.mkdir(mode=0o700)
    journal = DiscoveryJournal(private / 'journal', plan_bytes=plan_bytes, expected_source_commit=source_commit,
        expected_plan_sha256=plan_sha256, runtime_sha256=runtime_sha256, site_nonce=site_nonce,
        site_record_bytes=site_record_bytes, site_commitment=site_commitment,
        max_disk_bytes=disk_limit_bytes, max_retained_payload_bytes=disk_limit_bytes)
    started = {'schema': 'mlg-number-discovery-local-v1', 'source_commit': source_commit,
               'plan_sha256': plan_sha256, 'runtime_sha256': runtime_sha256,
               'site_commitment': site_commitment, 'status': 'started'}
    _write_new(private / 'started.json', _canonical(started))
    receipt_path = private / 'receipts.jsonl'
    baseline = {}
    executor = None
    observation, actions = None, ()
    current_request = None
    phase = 'model-state-before'
    try:
        initial_state = _model_state_sha256(model)
        requests = json.loads(plan.record_bytes)['requests']
        _write_new(receipt_path, b'')
        receipt_bytes = 0
        with NumberInjectionExecutor(model, site=site, max_study_bytes=logical_limit_bytes) as executor:
            for ordinal, row in enumerate(requests):
                phase = 'admission'
                if executor.retained_tensor_bytes + _FORWARD_RESERVE > logical_limit_bytes:
                    raise ValueError('logical tensor allowance exhausted')
                reservation = journal.reservation()
                if (reservation.disk_bytes + reservation.payload_bytes + _FINAL_DISK_RESERVE
                        + 2 * _MIB + receipt_bytes > disk_limit_bytes):
                    raise ValueError('physical evidence allowance exhausted')
                index, role = row['case_index'], row['role']
                ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]
                phase = 'forward'
                # Keep the request identity and ordinal in one reference so an
                # interruption at a journal return can be reconciled safely.
                current_request = (row['request_id'], role, ordinal)
                observation = executor.forward(ids, request_id=row['request_id'], condition=_CONDITIONS[role],
                                               **_request_arguments(role, baseline.get(index, (None, None))))
                if role == 'clean':
                    baseline[index] = (observation, None)
                elif role == 'injected':
                    baseline[index] = (baseline[index][0], observation)
                elif role == 'verify_clean':
                    del baseline[index]
                phase = 'serialize'
                actions = tuple(item for item in executor.private_evidence() if item.request_id == row['request_id'])
                selected = to_selector_observation(observation)
                public_raw = serialize_discovery_observation(index, role, selected, execution_device='cpu')
                private_raw = encode_private_forward(index, role, selected, actions)
                phase = 'journal'
                receipt = journal.append(public_raw, private_raw)
                phase = 'receipt'
                _append_receipt(receipt_path, receipt)
                receipt_bytes += len(_receipt_bytes(receipt))
                observation = selected = actions = public_raw = private_raw = None
                current_request = None
        retained = executor.retained_tensor_bytes
        phase = 'model-state-after'
        if baseline or _model_state_sha256(model) != initial_state:
            raise ValueError('model state or family-final baseline changed')
        phase = 'inspect'
        recorded = _read_receipts(receipt_path)
        if len(recorded) != 128:
            raise ValueError('receipt sidecar does not cover every request')
        inspection = inspect_discovery_journal(private / 'journal', plan_bytes=plan_bytes,
            expected_source_commit=source_commit, expected_plan_sha256=plan_sha256,
            runtime_sha256=runtime_sha256, site_nonce=site_nonce, site_record_bytes=site_record_bytes,
            site_commitment=site_commitment, acknowledgments=recorded, max_read_bytes=disk_limit_bytes,
            externally_quiescent=True)
        if inspection.status != 'complete-prefix' or len(inspection.verified_pairs) != 128:
            raise ValueError('journal lacks a complete verified prefix')
        records = tuple(pair.public for pair in inspection.verified_pairs)
        phase = 'freeze'
        prediction = freeze_discovery(_cases(records), private_controls_verified=True, execution_device='cpu')
        audit = audit_discovery_archive(records, prediction, execution_device='cpu')
        phase = 'archive'
        public.mkdir(mode=0o700)
        for raw in records:
            index, role, _ = _read(raw, 'cpu')
            _write_new(public / f'observation-{index:02d}-{role}.json', raw)
        _write_new(public / 'prediction.json', prediction)
        result = DiscoveryResult('complete-local', 128, audit['prediction_sha256'], audit['archive_sha256'],
                                 retained, logical_limit_bytes, disk_limit_bytes)
        _write_new(private / 'completed.json', _canonical({'schema': 'mlg-number-discovery-completed-v1',
                   'prediction_sha256': result.prediction_sha256, 'archive_sha256': result.archive_sha256,
                   'record_count': result.record_count, 'model_state_sha256': initial_state,
                   'retained_tensor_bytes': retained}))
        return result
    except BaseException as error:
        # If the journal already staged a pair, its own terminal record and
        # independently persisted receipt prefix remain in place. A failed
        # plan may have discarded captures before the executor exposed them.
        partial, partial_status = None, 'unavailable'
        try:
            missing = False
            pending_cancel = None
            if executor is not None and current_request is not None:
                try:
                    current_request_id, current_role, _ = current_request
                    matches = tuple(item for item in executor.completed() if item.request_id == current_request_id)
                    if observation is None and len(matches) == 1:
                        observation = matches[0]
                    elif observation is None and len(matches) > 1:
                        missing = True
                    actions = tuple(item for item in executor.private_evidence()
                                    if item.request_id == current_request_id)
                    if observation is not None:
                        operations = [item.operation for item in actions]
                        expected = _required_operations(current_role)
                        if len(operations) != len(expected) or set(operations) != expected:
                            missing = True
                except BaseException as retrieval_error:
                    if isinstance(retrieval_error, (KeyboardInterrupt, SystemExit)) and not isinstance(error, (KeyboardInterrupt, SystemExit)):
                        pending_cancel = retrieval_error
                    missing = True
            try:
                failures = executor.private_failures() if executor is not None else ()
            except BaseException as retrieval_error:
                if (isinstance(retrieval_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None):
                    pending_cancel = retrieval_error
                missing = True
                failures = ()
            failure = failures[-1] if failures else None
            if observation is not None or failure is not None or actions:
                try:
                    encoded = encode_incomplete_forward(observation=observation,
                        actions=actions if type(actions) is tuple else (), failure=failure)
                except BaseException as serialization_error:
                    # Terminalize the journal even if this independent
                    # best-effort serializer cannot capture available data.
                    partial_status = 'unavailable'
                    if (isinstance(serialization_error, (KeyboardInterrupt, SystemExit))
                            and not isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None):
                        pending_cancel = serialization_error
                else:
                    partial = encoded.data
                    partial_status = 'retained-with-omissions' if missing or encoded.omissions else 'retained'
            sidecar_saved = False
            sidecar_attempted = False

            def save_sidecar():
                nonlocal sidecar_saved, sidecar_attempted, partial_status, pending_cancel
                sidecar_attempted = True
                try:
                    _write_new(private / 'incomplete-forward.bin', partial)
                except BaseException as save_error:
                    partial_status = 'unavailable'
                    if (isinstance(save_error, (KeyboardInterrupt, SystemExit))
                            and not isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None):
                        pending_cancel = save_error
                else:
                    sidecar_saved = True

            try:
                state = journal.reservation()
            except BaseException as journal_error:
                if (isinstance(journal_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None):
                    pending_cancel = journal_error
                state = None

            # An append can advance the journal before its separate receipt
            # is durable. Save evidence for that prior request separately,
            # but always try terminalization even if this save fails.
            embed_partial = (partial is not None and state is not None and not state.failed
                             and current_request is not None and state.acknowledged_pairs == current_request[2]
                             and state.acknowledged_pairs < 128)
            if partial is not None and not embed_partial:
                save_sidecar()
            try:
                if state is None or not state.failed:
                    journal.abort(phase='external',
                        cancelled=isinstance(error, (KeyboardInterrupt, SystemExit)) or pending_cancel is not None,
                        partial_evidence=partial if embed_partial else None)
            except BaseException as journal_error:
                if (isinstance(journal_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None):
                    pending_cancel = journal_error
                if partial is not None and not sidecar_saved and not sidecar_attempted:
                    save_sidecar()
            if pending_cancel is not None:
                raise pending_cancel from error
        except BaseException as secondary:
            partial_status = 'unavailable'
            if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise secondary from error
        try:
            _write_new(private / 'failed.json', _canonical({'schema': 'mlg-number-discovery-failed-v1',
                       'status': 'incomplete', 'phase': phase, 'partial_evidence': partial_status}))
        except BaseException as secondary:
            if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise secondary from error
        raise
