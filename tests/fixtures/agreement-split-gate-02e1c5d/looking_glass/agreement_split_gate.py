"""Short cooperative cross-track allocation and first-forward ordering.

The markers contain public digests only. This lock is not a confidentiality
boundary and cannot govern a process which does not participate.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .device_lease import DeviceLease
from .injection_split_custody import _COMMIT_KEYS, _parse_canonical, _write_new
from .pretrained_number_commitments import canonical_json


_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_TRACKS = frozenset({'natural', 'injection'})
_MAX_GATE_BYTES = 64 * 1024
_GATE_FILES = frozenset({'gate.lock'} | {track + '-' + suffix + '.json'
    for track in _TRACKS for suffix in ('allocation', 'commitment', 'forward')})
_RECEIPT_KEYS = frozenset({'schema', 'natural_prediction_freeze_sha256',
    'natural_heldout_commitment', 'natural_exposure_inventory_sha256',
    'natural_commitment_file_sha256', 'injection_prediction_freeze_sha256',
    'injection_heldout_commitment', 'injection_exposure_inventory_sha256',
    'injection_commitment_file_sha256', 'comparator_source_sha256',
    'natural_case_count', 'injection_case_count', 'status'})


class GateError(ValueError):
    """The shared state is uncertain or does not authorize a forward."""


def _hex(value):
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise GateError('a lowercase SHA-256 digest is required')
    return value


def _track(value):
    if type(value) is not str or value not in _TRACKS:
        raise GateError('unknown track')
    return value


def _read_marker(path):
    if not os.path.lexists(path):
        return None
    st = path.lstat()
    if (not stat.S_ISREG(st.st_mode) or path.is_symlink() or st.st_nlink != 1
            or getattr(st, 'st_file_attributes', 0) & 1024):
        raise GateError('shared marker is not a regular private file')
    with path.open('rb') as stream:
        raw = stream.read(8193)
    if not raw or len(raw) > 8192:
        raise GateError('shared marker is incomplete or oversized')
    try:
        value = json.loads(raw)
        if type(value) is not dict or canonical_json(value) != raw:
            raise GateError('shared marker is not canonical')
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        if isinstance(error, GateError):
            raise
        raise GateError('shared marker is invalid') from None


def _allocation(root, track):
    value = _read_marker(root / (track + '-allocation.json'))
    if value is not None:
        if (set(value) != {'schema', 'track', 'freeze_sha256'}
                or value['schema'] != 'mlg-number-split-allocation-v1'
                or value['track'] != track):
            raise GateError('allocation marker is invalid')
        _hex(value['freeze_sha256'])
    return value


def _commitment(root, track):
    value = _read_marker(root / (track + '-commitment.json'))
    if value is not None:
        if (set(value) != {'schema', 'track', 'freeze_sha256', 'heldout_commitment',
                           'commitment_file_sha256', 'inventory_sha256'}
                or value['schema'] != 'mlg-number-split-commitment-observed-v1'
                or value['track'] != track):
            raise GateError('commitment marker is invalid')
        for key in ('freeze_sha256', 'heldout_commitment', 'commitment_file_sha256', 'inventory_sha256'):
            _hex(value[key])
    return value


def _forward(root, track):
    value = _read_marker(root / (track + '-forward.json'))
    if value is not None:
        keys = {'schema', 'track', 'freeze_sha256', 'heldout_commitment',
                'commitment_file_sha256', 'inventory_sha256', 'comparison'}
        comparison = value.get('comparison')
        if comparison == 'clear':
            keys.add('receipt_sha256')
        if (set(value) != keys or value.get('schema') != 'mlg-number-split-forward-v1'
                or value.get('track') != track or comparison not in {
                    'clear', 'other_not_allocated_unchecked', 'other_allocation_pending_unchecked'}):
            raise GateError('forward marker is invalid')
        for key in ('freeze_sha256', 'heldout_commitment', 'commitment_file_sha256', 'inventory_sha256'):
            _hex(value[key])
        if comparison == 'clear':
            _hex(value['receipt_sha256'])
    return value


def _check_receipt(raw, natural, injection):
    if type(raw) is not bytes or not 0 < len(raw) <= 8192:
        raise GateError('a bounded comparison receipt is required')
    try:
        receipt = json.loads(raw)
        if (type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS
                or canonical_json(receipt) != raw
                or receipt['schema'] != 'mlg-number-disjoint-split-check-v1'
                or receipt['status'] != 'clear'
                or type(receipt['natural_case_count']) is not int or receipt['natural_case_count'] != 16
                or type(receipt['injection_case_count']) is not int or receipt['injection_case_count'] != 16):
            raise GateError('a canonical clear receipt is required')
        _hex(receipt['comparator_source_sha256'])
        for track, marker in (('natural', natural), ('injection', injection)):
            for field, source in (('prediction_freeze_sha256', 'freeze_sha256'),
                                  ('heldout_commitment', 'heldout_commitment'),
                                  ('exposure_inventory_sha256', 'inventory_sha256'),
                                  ('commitment_file_sha256', 'commitment_file_sha256')):
                if receipt[track + '_' + field] != marker[source]:
                    raise GateError('clear receipt belongs to a different split')
        return receipt
    except (KeyError, ValueError, TypeError, UnicodeError, RecursionError) as error:
        if isinstance(error, GateError):
            raise
        raise GateError('comparison receipt is invalid') from None


class SplitGate:
    def __init__(self, root, lease, root_identity):
        self.root = root
        self.lease = lease
        self.root_identity = root_identity

    def _check(self):
        self.lease.check()
        st = self.root.lstat()
        if (not stat.S_ISDIR(st.st_mode) or self.root.is_symlink()
                or getattr(st, 'st_file_attributes', 0) & 1024
                or (st.st_dev, st.st_ino) != self.root_identity):
            raise GateError('shared gate root was replaced or is ambiguous')
        total = 0
        for path in self.root.iterdir():
            item = path.lstat()
            if (path.name not in _GATE_FILES or not stat.S_ISREG(item.st_mode)
                    or path.is_symlink() or item.st_nlink != 1
                    or getattr(item, 'st_file_attributes', 0) & 1024 or item.st_size > 8192):
                raise GateError('shared gate contains an unexpected or unsafe entry')
            total += item.st_size
        if total > _MAX_GATE_BYTES:
            raise GateError('shared gate exceeds its physical file-size cap')

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.lease.release()
        except BaseException as cleanup:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise exc from cleanup
            raise
        return False

    def announce_allocation(self, track, freeze_sha256):
        self._check()
        track = _track(track)
        freeze_sha256 = _hex(freeze_sha256)
        if (_allocation(self.root, track) is not None or _commitment(self.root, track) is not None
                or os.path.lexists(self.root / (track + '-forward.json'))):
            raise GateError('allocation has already been started or is uncertain')
        value = {'schema': 'mlg-number-split-allocation-v1', 'track': track,
                 'freeze_sha256': freeze_sha256}
        _write_new(self.root / (track + '-allocation.json'), canonical_json(value))
        self._check()

    def announce_commitment(self, track, freeze_sha256, heldout_commitment,
                            commitment_file_sha256, inventory_sha256, *,
                            commitment_bytes, publication_check):
        self._check()
        track = _track(track)
        values = {'schema': 'mlg-number-split-commitment-observed-v1', 'track': track,
                  'freeze_sha256': _hex(freeze_sha256), 'heldout_commitment': _hex(heldout_commitment),
                  'commitment_file_sha256': _hex(commitment_file_sha256),
                  'inventory_sha256': _hex(inventory_sha256)}
        allocation = _allocation(self.root, track)
        if (allocation is None or allocation['freeze_sha256'] != freeze_sha256
                or _commitment(self.root, track) is not None
                or os.path.lexists(self.root / (track + '-forward.json'))):
            raise GateError('commitment does not follow a single allocation')
        try:
            public = _parse_canonical(commitment_bytes)
            if (type(public) is not dict or set(public) != _COMMIT_KEYS[track]
                    or public.get('schema') != 'mlg-' + track + '-heldout-public-commitment-v1'
                    or public.get('prediction_freeze_sha256') != freeze_sha256
                    or public.get('heldout_commitment') != heldout_commitment
                    or public.get('exposure_inventory_sha256') != inventory_sha256
                    or hashlib.sha256(commitment_bytes).hexdigest() != commitment_file_sha256):
                raise GateError('published commitment bytes do not match')
            for key in _COMMIT_KEYS[track] - {'schema'}:
                _hex(public[key])
            if not callable(publication_check) or publication_check(commitment_bytes) is not True:
                raise GateError('commitment publication was not observed')
        except (TypeError, ValueError, KeyError) as error:
            if isinstance(error, GateError):
                raise
            raise GateError('published commitment is invalid') from None
        self._check()
        if (allocation != _allocation(self.root, track) or _commitment(self.root, track) is not None
                or os.path.lexists(self.root / (track + '-forward.json'))):
            raise GateError('allocation changed during commitment verification')
        _write_new(self.root / (track + '-commitment.json'), canonical_json(values))
        self._check()

    def announce_forward(self, track, *, receipt_bytes=None, receipt_check=None):
        """Durably announce a possible first forward before it can occur."""
        self._check()
        track = _track(track)
        other = 'injection' if track == 'natural' else 'natural'
        own_allocation, own = _allocation(self.root, track), _commitment(self.root, track)
        other_allocation, other_commitment = _allocation(self.root, other), _commitment(self.root, other)
        other_forward = _forward(self.root, other)
        if (other_forward is not None and (other_allocation is None or other_commitment is None
                or any(other_forward[key] != other_commitment[key] for key in
                       ('freeze_sha256', 'heldout_commitment', 'commitment_file_sha256', 'inventory_sha256')))):
            raise GateError('other forward marker lacks its matching committed split')
        if (own_allocation is None or own is None or own_allocation['freeze_sha256'] != own['freeze_sha256']
                or os.path.lexists(self.root / (track + '-forward.json'))):
            raise GateError('own split is incomplete or a forward was already announced')
        if other_allocation is None:
            if other_commitment is not None or os.path.lexists(self.root / (other + '-forward.json')):
                raise GateError('other track state is inconsistent')
            if receipt_bytes is not None or receipt_check is not None:
                raise GateError('an unallocated other track is unchecked, not clear')
            value = {'schema': 'mlg-number-split-forward-v1', 'track': track,
                     **{key: own[key] for key in ('freeze_sha256', 'heldout_commitment',
                                                  'commitment_file_sha256', 'inventory_sha256')},
                     'comparison': 'other_not_allocated_unchecked'}
        elif other_commitment is None:
            if os.path.lexists(self.root / (other + '-forward.json')):
                raise GateError('uncommitted other track has an uncertain forward')
            if receipt_bytes is not None or receipt_check is not None:
                raise GateError('a pending other track is unchecked, not clear')
            value = {'schema': 'mlg-number-split-forward-v1', 'track': track,
                     **{key: own[key] for key in ('freeze_sha256', 'heldout_commitment',
                                                  'commitment_file_sha256', 'inventory_sha256')},
                     'comparison': 'other_allocation_pending_unchecked'}
        else:
            if other_commitment['freeze_sha256'] != other_allocation['freeze_sha256']:
                raise GateError('other track allocation is unresolved')
            natural, injection = (own, other_commitment) if track == 'natural' else (other_commitment, own)
            _check_receipt(receipt_bytes, natural, injection)
            if not callable(receipt_check) or receipt_check(receipt_bytes) is not True:
                raise GateError('clear receipt verification did not pass')
            if (own_allocation != _allocation(self.root, track) or own != _commitment(self.root, track)
                    or other_allocation != _allocation(self.root, other)
                    or other_commitment != _commitment(self.root, other)
                    or other_forward != _forward(self.root, other)):
                raise GateError('shared state changed during receipt verification')
            value = {'schema': 'mlg-number-split-forward-v1', 'track': track,
                     **{key: own[key] for key in ('freeze_sha256', 'heldout_commitment',
                                                  'commitment_file_sha256', 'inventory_sha256')},
                     'comparison': 'clear',
                     'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest()}
        self._check()
        _write_new(self.root / (track + '-forward.json'), canonical_json(value))
        self._check()
        return value


def acquire(root, *, expected_root, timeout):
    """Acquire the common cooperative lock for a short state transition."""
    root, expected = Path(root), Path(expected_root)
    if (not root.is_absolute() or not expected.is_absolute() or not root.is_dir() or root.is_symlink()
            or getattr(root.lstat(), 'st_file_attributes', 0) & 1024):
        raise GateError('an existing protected gate directory is required')
    if not expected.is_dir() or not os.path.samefile(root, expected):
        raise GateError('gate root differs from the independently bound root')
    st = root.lstat()
    if not st.st_ino:
        raise GateError('shared gate root identity is unavailable')
    lease = DeviceLease.acquire(root / 'gate.lock', timeout=timeout)
    gate = SplitGate(root, lease, (st.st_dev, st.st_ino))
    try:
        gate._check()
    except BaseException as error:
        try:
            lease.release()
        except BaseException as cleanup:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error from cleanup
            raise
        raise
    return gate
