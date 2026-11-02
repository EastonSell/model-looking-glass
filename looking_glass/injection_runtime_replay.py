"""Site-blind public comparisons for a prospective runtime check.

This component does not authorize a model forward, protect the private site,
attest execution or establish runtime equivalence. Its detailed outputs remain
private until the separately authorized reveal.
"""
from dataclasses import asdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType

from .agreement_numeric import compare_f32_le
from .pretrained_number_archive import _ROLES, _read
from . import pretrained_number_journal
from .distilgpt2_intervention_dev import _model_state_sha256
from .pretrained_exposures_v2 import PUBLIC_DISCOVERY
from .pretrained_number_adapter import to_selector_observation
from .pretrained_number_archive import _CONDITIONS, serialize_discovery_observation
from .pretrained_number_commitments import verify_site_reveal
from .pretrained_number_incomplete import encode_incomplete_forward
from .pretrained_number_injection import NumberInjectionExecutor
from .pretrained_number_requests import load_family_request_plan, verify_request
from .pretrained_number_supervisor import _append_receipt, _request_arguments, _write_new
from .pretrained_number_witness import encode_private_forward


_JOURNAL_SOURCE_SHA256 = '9a67c485c0c4123ab77fb5252a1248b9cf6817e16fc066bc6ee07a6a6e686ec1'
_JOURNAL_MODULE = '_mlg_injection_runtime_replay_journal'
_REFERENCE_SHA256 = (
    '4d0815c9368397ffebb4257065e0fac1f5c7110b21bc4bbf53511199fb545fc2',
    '68f03dc86e14e653f9850aa241f4410fff121ec80d8c303987ed2f2966d795dc',
    '0a18dda10126cc532dbbe46712084a4013fabb00414b81141905fd1ae003dccc',
    '6350d70b8847ebf842169895c3a53ce6f469158ff7f4b44f405ce3ff3f83a8fc',
    '8e26558abae3fb1eac9a773b823bab94d9413cc9cec687941dbc16f23f9ae44d',
    'e67cb76ed14a693b8fe88bfcb39336740df0dd90fce6097e6c4aeb6b4301aa02',
    'b4955d40391ca411eea988f9ea44cb738b8b13f4a3ecb93db6e110064082be44',
    '4d3ab110bf6706194ced5ae67ac4f78845b78a84129065dce692f61fe1a2a79b',
    '08cb49bf4909dc6ea6f6b16d1786859fcbdb980fe162f0930b260f09ade0006d',
    'ef5bcaf33eb22f1b48c250e2643edc50b6599a097f9af0c8a3370abdc33a3c91',
    'af7a11a6ab88d745ffd17248c68c0e673ca183e67374f21b5fac3da1209c6c8e',
    'ecf6e7bd0c83aa6efc6b8a0223e891385423943977876362cc6621d3fb458c33',
    '4c327a552512bb5b3f4a39f6be0fda0672318f44fb327eb17c83ff4f3474940e',
    '9c5efeb0b2657f4eabc86718984b48b785c12170a5cc9060fe3390acad99e0a2',
    '958db3742fc9a7d9a06d9fe367290d088ccb76014abc0f9c789f12e40dad3719',
    'bd9b06b285d277bbce16f3058a8be1d6c1d9aa63e13bab29259e567cbcfbe58c',
)
_MIB = 1024 * 1024


@contextmanager
def isolated_replay_journal():
    """Isolate the frozen journal with only its fixed case-0 schedule changed.

    The caller must source-protect the original file and all its imports. This
    changes no discovery module globals, and is a separately bound adapter.
    """
    if _JOURNAL_MODULE in sys.modules:
        raise RuntimeError('a replay journal is already active in this process')
    path = Path(pretrained_number_journal.__file__)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _JOURNAL_SOURCE_SHA256:
        raise ValueError('frozen journal source differs')
    module = ModuleType(_JOURNAL_MODULE)
    module.__file__ = str(path)
    module.__package__ = 'looking_glass'
    sys.modules[_JOURNAL_MODULE] = module
    try:
        exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
        module._COUNT = 16
        module.family_request_schedule = lambda: tuple((0, role) for role in _ROLES)
        yield module
    finally:
        if sys.modules.get(_JOURNAL_MODULE) is module:
            del sys.modules[_JOURNAL_MODULE]


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _ordinary_info(path, *, directory=False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        raise ValueError('replay path is linked or nonordinary')
    return info


def _bounded_inventory(directory, expected):
    """Inspect only a fixed number of expected direct children."""
    found = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name not in expected or entry.name in found or len(found) >= len(expected):
                raise ValueError('replay output contains an unexpected entry')
            path = directory / entry.name
            found[entry.name] = _ordinary_info(path, directory=expected[entry.name] == 'directory')
    if set(found) != set(expected):
        raise ValueError('replay output is incomplete')
    return found


def _read_bounded_regular(path, limit, recorded):
    if recorded.st_size > limit:
        raise ValueError('replay side record exceeds its bound')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    after = _ordinary_info(path)
    before_id = (recorded.st_dev, recorded.st_ino, recorded.st_size, recorded.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if len(raw) != recorded.st_size or len(raw) > limit or before_id != after_id:
        raise ValueError('replay side record changed while reading')
    return raw


def compare_public_pair(role, reference, actual):
    """Compare one frozen role's full public tensors and exact metadata."""
    if type(role) is not str or role not in _ROLES:
        raise ValueError('a fixed replay role is required')
    reference_index, reference_role, left = _read(reference, 'cpu')
    actual_index, actual_role, right = _read(actual, 'cpu')
    if (reference_index != 0 or actual_index != 0 or reference_role != role or actual_role != role
            or left.request_id != right.request_id or left.input_sha256 != right.input_sha256
            or left.condition != right.condition
            or _canonical([asdict(item) for item in left.trace]) != _canonical([asdict(item) for item in right.trace])
            or _canonical(None if left.action is None else asdict(left.action))
            != _canonical(None if right.action is None else asdict(right.action))):
        raise ValueError('public role or metadata differs from the frozen reference')
    lhs = left.attention + left.mlp + (left.block, left.logits)
    rhs = right.attention + right.mlp + (right.block, right.logits)
    if len(lhs) != 14 or len(rhs) != 14:
        raise ValueError('public capture inventory differs')
    comparisons = [compare_f32_le(a, b) for a, b in zip(lhs, rhs)]
    return {'within_tolerance': all(item['within_tolerance'] for item in comparisons),
            'bit_equal': all(item['bit_equal'] for item in comparisons)}


def run_public_replay(*, model, attempt_dir, plan_bytes, source_commit, plan_sha256,
                      discovery_runtime_sha256, heldout_runtime_sha256, site_nonce,
                      site_record_bytes, site_commitment, references, expected_model_state_sha256):
    """Produce one private fixed-case replay; external gates remain mandatory."""
    for digest in (discovery_runtime_sha256, heldout_runtime_sha256, expected_model_state_sha256):
        if type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('a bound runtime or model digest is invalid')
    if (type(references) is not tuple or len(references) != 16
            or any(type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected
                   for raw, expected in zip(references, _REFERENCE_SHA256))):
        raise ValueError('fixed public reference bytes differ')
    plan = load_family_request_plan(plan_bytes, expected_source_commit=source_commit, expected_sha256=plan_sha256)
    rows = {item['role']: item for item in json.loads(plan.record_bytes)['requests'] if item['case_index'] == 0}
    if set(rows) != set(_ROLES) or len(rows) != 16:
        raise ValueError('fixed request schedule differs')
    for role, raw in zip(_ROLES, references):
        index, actual_role, reference = _read(raw, 'cpu')
        if index != 0 or actual_role != role:
            raise ValueError('fixed public reference order differs')
        verify_request(plan, 0, role, reference.request_id, reference.input_sha256,
                       expected_source_commit=source_commit, expected_sha256=plan_sha256)
    site = verify_site_reveal(site_nonce, site_record_bytes, site_commitment, expected_source_commit=source_commit)
    if _model_state_sha256(model) != expected_model_state_sha256:
        raise ValueError('model state differs before replay')
    attempt = Path(attempt_dir)
    if not attempt.is_absolute() or not attempt.parent.is_dir() or os.path.lexists(attempt):
        raise ValueError('a fresh absolute replay directory is required')
    attempt.mkdir(mode=0o700)
    receipts_path = attempt / 'receipts.jsonl'
    _write_new(receipts_path, b'')
    executor, journal, observation, current_request, current_ordinal = None, None, None, None, None
    receipts, comparisons, clean, injected = [], [], None, None
    try:
        with isolated_replay_journal() as frozen:
            kwargs = dict(plan_bytes=plan_bytes, expected_source_commit=source_commit,
                          expected_plan_sha256=plan_sha256, runtime_sha256=heldout_runtime_sha256,
                          site_nonce=site_nonce, site_record_bytes=site_record_bytes,
                          site_commitment=site_commitment)
            journal = frozen.DiscoveryJournal(attempt / 'journal', **kwargs,
                                              max_disk_bytes=14 * _MIB,
                                              max_retained_payload_bytes=8 * _MIB)
            with NumberInjectionExecutor(model, site=site, max_study_bytes=8 * _MIB) as executor:
                for ordinal, (role, reference) in enumerate(zip(_ROLES, references)):
                    reservation = journal.reservation()
                    if (executor.retained_tensor_bytes + _MIB > 8 * _MIB
                            or reservation.payload_bytes + _MIB > 8 * _MIB
                            or reservation.disk_bytes + 2 * _MIB > 14 * _MIB):
                        raise ValueError('replay evidence reservation exhausted')
                    row = rows[role]
                    current_request = row['request_id']
                    current_ordinal = ordinal
                    observation = executor.forward(PUBLIC_DISCOVERY[0][1], request_id=current_request,
                        condition=_CONDITIONS[role], **_request_arguments(role, (clean, injected)))
                    if role == 'clean':
                        clean = observation
                    elif role == 'injected':
                        injected = observation
                    selected = to_selector_observation(observation)
                    public = serialize_discovery_observation(0, role, selected, execution_device='cpu')
                    actions = tuple(item for item in executor.private_evidence() if item.request_id == current_request)
                    private = encode_private_forward(0, role, selected, actions)
                    receipt = journal.append(public, private)
                    _append_receipt(receipts_path, receipt)
                    receipts.append(receipt)
                    comparison = compare_public_pair(role, reference, public)
                    comparisons.append(comparison)
                    if not comparison['within_tolerance']:
                        raise ValueError('public compatibility check failed')
                    observation, current_request, current_ordinal = None, None, None
            if _model_state_sha256(model) != expected_model_state_sha256 or len(receipts) != 16:
                raise ValueError('model state or replay count differs')
            inspection = frozen.inspect_discovery_journal(attempt / 'journal', **kwargs,
                acknowledgments=tuple(receipts), max_read_bytes=14 * _MIB, externally_quiescent=True)
            if inspection.status != 'complete-prefix' or len(inspection.verified_pairs) != 16:
                raise ValueError('private replay journal is incomplete')
            outcome = {'schema': 'mlg-injection-runtime-replay-owner-v1',
                       'status': 'replay-produced-private-awaiting-independent-audit',
                       'discovery_runtime_sha256': discovery_runtime_sha256,
                       'heldout_runtime_sha256': heldout_runtime_sha256,
                       'model_state_sha256': expected_model_state_sha256,
                       'forward_count': 16, 'logical_limit_bytes': 8 * _MIB,
                       'physical_limit_bytes': 16 * _MIB, 'comparisons': comparisons}
            _write_new(attempt / 'owner-terminal.json', _canonical(outcome))
            return outcome
    except BaseException as error:
        # A bounded completed observation can survive a later serialization or
        # comparison failure. Preserve it privately when the journal permits.
        partial, pending_cancel, actions, failure = None, None, (), None
        def note_cancel(secondary):
            nonlocal pending_cancel
            if (pending_cancel is None and isinstance(secondary, (KeyboardInterrupt, SystemExit))
                    and not isinstance(error, (KeyboardInterrupt, SystemExit))):
                pending_cancel = secondary
        if executor is not None and current_request is not None:
            try:
                matches = tuple(item for item in executor.completed() if item.request_id == current_request)
                if observation is None and len(matches) == 1:
                    observation = matches[0]
            except BaseException as secondary:
                note_cancel(secondary)
            try:
                actions = tuple(item for item in executor.private_evidence() if item.request_id == current_request)
            except BaseException as secondary:
                note_cancel(secondary)
            try:
                failures = executor.private_failures()
                failure = failures[-1] if failures else None
            except BaseException as secondary:
                note_cancel(secondary)
        try:
            if observation is not None or actions or failure is not None:
                partial = encode_incomplete_forward(observation=observation, actions=actions, failure=failure).data
        except BaseException as secondary:
            note_cancel(secondary)
        try:
            reservation = journal.reservation() if journal is not None else None
        except BaseException as secondary:
            note_cancel(secondary)
            reservation = None
        embed = (partial is not None and reservation is not None and not reservation.failed
                 and current_ordinal is not None and reservation.acknowledged_pairs == current_ordinal)
        if partial is not None and not embed and len(partial) <= 2 * _MIB:
            try:
                if (reservation is None or reservation.payload_bytes + len(partial) > 8 * _MIB
                        or reservation.disk_bytes + len(partial) + _MIB > 16 * _MIB):
                    raise ValueError('partial evidence exceeds its reserve')
                _write_new(attempt / 'incomplete-forward.bin', partial)
            except BaseException as secondary:
                note_cancel(secondary)
        try:
            if reservation is not None and not reservation.failed:
                journal.abort(phase='external',
                              cancelled=isinstance(error, (KeyboardInterrupt, SystemExit)) or pending_cancel is not None,
                              partial_evidence=partial if embed else None)
        except BaseException as secondary:
            note_cancel(secondary)
        try:
            _write_new(attempt / 'failure.json', _canonical({'status': 'incomplete',
                'partial_evidence': 'retained-or-unavailable', 'error_type': type(error).__name__}))
        except BaseException as secondary:
            note_cancel(secondary)
        if pending_cancel is not None:
            raise pending_cancel from error
        raise


def audit_public_replay(attempt_dir, *, plan_bytes, source_commit, plan_sha256,
                        discovery_runtime_sha256, heldout_runtime_sha256, site_nonce,
                        site_record_bytes, site_commitment, references, expected_model_state_sha256,
                        externally_quiescent=False):
    """Reinspect private saved frames; caller must establish process exit independently."""
    if externally_quiescent is not True:
        raise ValueError('externally established quiescence is required')
    attempt = Path(attempt_dir)
    if not attempt.is_absolute():
        raise ValueError('replay directory is unavailable')
    _ordinary_info(attempt, directory=True)
    root_files = _bounded_inventory(attempt, {'journal': 'directory', 'receipts.jsonl': 'file',
                                              'owner-terminal.json': 'file'})
    journal_names = {f'{ordinal:03d}.{suffix}': 'file' for ordinal in range(16) for suffix in ('frame', 'ok')}
    journal_files = _bounded_inventory(attempt / 'journal', journal_names)
    total = (root_files['receipts.jsonl'].st_size + root_files['owner-terminal.json'].st_size
             + sum(info.st_size for info in journal_files.values()))
    if total > 16 * _MIB:
        raise ValueError('replay physical allowance exceeded')
    if (type(references) is not tuple or len(references) != 16
            or any(type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected
                   for raw, expected in zip(references, _REFERENCE_SHA256))):
        raise ValueError('fixed public reference bytes differ')
    receipts_raw = _read_bounded_regular(attempt / 'receipts.jsonl', 16384, root_files['receipts.jsonl'])
    terminal_raw = _read_bounded_regular(attempt / 'owner-terminal.json', 65536, root_files['owner-terminal.json'])
    if not receipts_raw.endswith(b'\n'):
        raise ValueError('replay side records are oversized or incomplete')
    terminal = json.loads(terminal_raw)
    expected_terminal = {'schema': 'mlg-injection-runtime-replay-owner-v1',
        'status': 'replay-produced-private-awaiting-independent-audit',
        'discovery_runtime_sha256': discovery_runtime_sha256,
        'heldout_runtime_sha256': heldout_runtime_sha256,
        'model_state_sha256': expected_model_state_sha256, 'forward_count': 16,
        'logical_limit_bytes': 8 * _MIB, 'physical_limit_bytes': 16 * _MIB,
        'comparisons': terminal.get('comparisons') if type(terminal) is dict else None}
    if _canonical(terminal) != terminal_raw or _canonical(terminal) != _canonical(expected_terminal):
        raise ValueError('private replay terminal differs')
    with isolated_replay_journal() as frozen:
        receipts = []
        for ordinal, line in enumerate(receipts_raw.splitlines()):
            item = json.loads(line)
            if (type(item) is not dict or set(item) != {'ordinal', 'frame_sha256', 'public_sha256'}
                    or type(item['ordinal']) is not int or item['ordinal'] != ordinal
                    or _canonical(item) != line):
                raise ValueError('replay receipt stream differs')
            receipts.append(frozen.PairReceipt(item['ordinal'], item['frame_sha256'], item['public_sha256']))
        if len(receipts) != 16:
            raise ValueError('all returned replay receipts are required')
        inspection = frozen.inspect_discovery_journal(attempt / 'journal', plan_bytes=plan_bytes,
            expected_source_commit=source_commit, expected_plan_sha256=plan_sha256,
            runtime_sha256=heldout_runtime_sha256, site_nonce=site_nonce, site_record_bytes=site_record_bytes,
            site_commitment=site_commitment, acknowledgments=tuple(receipts),
            max_read_bytes=14 * _MIB, externally_quiescent=True)
        if inspection.status != 'complete-prefix' or len(inspection.verified_pairs) != 16:
            raise ValueError('replay journal is incomplete')
        payload_bytes = sum(len(pair.public) + len(pair.private) for pair in inspection.verified_pairs)
        if payload_bytes > 8 * _MIB:
            raise ValueError('replay logical allowance exceeded')
        comparisons = [compare_public_pair(role, reference, pair.public)
                       for role, reference, pair in zip(_ROLES, references, inspection.verified_pairs)]
        if (not all(item['within_tolerance'] for item in comparisons)
                or _canonical(comparisons) != _canonical(terminal['comparisons'])):
            raise ValueError('saved replay comparison differs')
    return {'status': 'saved-evidence-consistent', 'forward_count': 16,
            'logical_payload_upper_bound': payload_bytes, 'physical_bytes': total,
            'receipts_sha256': hashlib.sha256(receipts_raw).hexdigest(),
            'owner_terminal_sha256': hashlib.sha256(terminal_raw).hexdigest()}
