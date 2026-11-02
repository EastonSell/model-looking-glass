"""Cooperative successor markers under the existing shared split gate.

Call only with the source-verified original SplitGate held by this process.
The marker root and persistent gate/lock identities below are separate from the
original natural split.  This module does not authenticate the caller,
publication, a comparator, current inventory, ACLs or process exit.  A
successful result is a point-in-time cooperative observation; same-account
tampering or power loss requires external reconciliation.  No function here
draws a nonce or reads private cases.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat

from .agreement_split_gate import SplitGate, _allocation, _commitment, _forward


class SuccessorStateError(ValueError):
    """The shared state is incomplete or cannot authorize a transition."""


GATE_ROOT_SHA256 = '0ede711a758f4cd5a7b3651d12aa9b81ac7f0cac7e6419a309411e24c9f80e70'
GATE_ROOT_IDENTITY = (1219152768, 4503599632054271)
GATE_LOCK_IDENTITY = (1219152768, 6192449492328171)
SUCCESSOR_ROOT_SHA256 = 'ab323263391e82391148f479a359b3707774ebf2b64ca431f60bfe492570c639'
SUCCESSOR_ROOT_IDENTITY = (1219152768, 1125899911728318)
FREEZE = '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb'
POOL = 'b3323d296a9f56a4232b4a403f7c8935896634b3d2b6e9fe7eee4aee56ba1126'
INPUTS = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
PUBLIC_CLEARANCE = '262a3889fc4950de4eca39a7705a9028f248f19fdc0dd3b76370ce691638dc91'
AMENDMENT = 'd4460921f9688fe8da993ede89e49868e16a38311787a3c2dd136084f0f7c695'
PROTOCOL = '0c695a162507c6b6db7cf48088ca31848ae91cdb56ed22c679f14a3d0e99282b'
OLD_COMMITMENT = '8b475c373034201e5c36186207edbc56f0c8b62221f497d8d09faad1ce182150'
INJECTION_FREEZE = 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'
_DISPOSITION_SCHEMA = 'mlg-natural-successor-state-disposition-v1'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_ALLOC = 'natural-successor-allocation-v1.json'
_COMMIT = 'natural-successor-commitment-v1.json'
_FORWARD = 'natural-successor-forward-v1.json'
_CLAIM = 'phase-claim.json'


def _sha(value):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise SuccessorStateError('invalid digest')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def _ordinary(path, expected_hash, identity, *, directory):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise SuccessorStateError('an absolute protected path is required')
    normalized = os.path.normcase(os.path.realpath(path))
    if (normalized != os.path.normcase(os.path.abspath(path))
            or not hmac.compare_digest(hashlib.sha256(normalized.encode('utf-8')).hexdigest(), expected_hash)):
        raise SuccessorStateError('protected path differs')
    for part in (path, *path.parents):
        item = part.lstat()
        if stat.S_ISLNK(item.st_mode) or getattr(item, 'st_file_attributes', 0) & 1024:
            raise SuccessorStateError('protected path crosses a link')
    item = path.lstat()
    if ((not stat.S_ISDIR(item.st_mode) if directory else not stat.S_ISREG(item.st_mode))
            or (item.st_dev, item.st_ino) != identity
            or (not directory and item.st_nlink != 1)):
        raise SuccessorStateError('protected identity differs')
    return path


def _guard(gate, root):
    if type(gate) is not SplitGate:
        raise SuccessorStateError('a held source-verified split gate is required')
    gate_root = _ordinary(gate.root, GATE_ROOT_SHA256, GATE_ROOT_IDENTITY, directory=True)
    lock = gate_root / 'gate.lock'
    info = lock.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, 'st_file_attributes', 0) & 1024
            or (info.st_dev, info.st_ino) != GATE_LOCK_IDENTITY
            or os.path.normcase(os.path.abspath(gate.lease.path)) != os.path.normcase(os.path.abspath(lock))):
        raise SuccessorStateError('shared lock identity differs')
    gate._check()
    return _ordinary(root, SUCCESSOR_ROOT_SHA256, SUCCESSOR_ROOT_IDENTITY, directory=True)


def _read(path):
    if not os.path.lexists(path):
        return None, None
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, 'st_file_attributes', 0) & 1024 or not 1 <= info.st_size <= 8192):
        raise SuccessorStateError('marker is unsafe or incomplete')
    raw = path.read_bytes()
    after = path.lstat()
    if (len(raw) != info.st_size or (after.st_dev, after.st_ino, after.st_size) !=
            (info.st_dev, info.st_ino, info.st_size)):
        raise SuccessorStateError('marker changed while reading')
    try:
        value = json.loads(raw)
        if type(value) is not dict or _canonical(value) != raw:
            raise SuccessorStateError('marker is not canonical')
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise SuccessorStateError('marker is invalid') from None
    return value, hashlib.sha256(raw).hexdigest()


def _claim(value):
    if (type(value) is not dict or set(value) != {'schema', 'kind', 'operation_sha256',
             'public_clearance_sha256', 'inventory_sha256', 'amendment_sha256'}
            or value['schema'] != 'mlg-natural-successor-claim-v1'
            or value['kind'] != 'phase-public-support'
            or value['public_clearance_sha256'] != PUBLIC_CLEARANCE
            or value['amendment_sha256'] != AMENDMENT):
        raise SuccessorStateError('successor phase claim differs')
    _sha(value['operation_sha256'])
    _sha(value['inventory_sha256'])


def _allocation_value(claim, claim_sha):
    return {'schema': 'mlg-natural-successor-allocation-v1', 'track': 'natural-successor',
            'phase_claim_sha256': claim_sha, 'operation_sha256': claim['operation_sha256'],
            'freeze_sha256': FREEZE, 'pool_sha256': POOL, 'inputs_sha256': INPUTS,
            'public_clearance_sha256': PUBLIC_CLEARANCE, 'amendment_sha256': AMENDMENT,
            'inventory_sha256': claim['inventory_sha256']}


def _inspect(gate, root, *, allow_unannounced_claim=False):
    root = _guard(gate, root)
    # Other ordinary evidence may live in the phase root.  Reserve the named
    # transition/phase prefixes so a future or partial marker cannot look like
    # absence to an older caller.
    allowed = {_ALLOC, _COMMIT, _FORWARD, _CLAIM, 'owner-started.json',
               'nonce.bin', 'heldout-record.json', 'commitment.json'}
    prefixes = ('natural-successor-', 'pool-query-', 'phase-claim', 'owner-started',
                'nonce', 'heldout-record', 'commitment')
    for item in root.iterdir():
        if item.name.casefold().startswith(prefixes) and item.name not in allowed:
            raise SuccessorStateError('unknown successor or query marker')
    if any(os.path.lexists(root / item) for item in
           ('pool-query-invocation.json', 'pool-query-claim.json', 'pool-query-receipt.json',
            'pool-query-observation.json', 'pool-query-supervisor', 'pool-query-outer')):
        raise SuccessorStateError('legacy query state requires reconciliation')
    claim, claim_sha = _read(root / _CLAIM)
    allocation, allocation_sha = _read(root / _ALLOC)
    commitment, commitment_sha = _read(root / _COMMIT)
    forward, forward_sha = _read(root / _FORWARD)
    if os.path.lexists(root / 'owner-started.json') and (commitment is None or forward is None):
        raise SuccessorStateError('owner-start marker lacks a committed forward announcement')
    if os.path.lexists(root / 'owner') and commitment is None:
        raise SuccessorStateError('owner directory lacks a committed successor')
    nonce = root / 'nonce.bin'
    if os.path.lexists(nonce):
        info = nonce.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != 32
                or getattr(info, 'st_file_attributes', 0) & 1024):
            raise SuccessorStateError('nonce file metadata is incomplete or unsafe')
    if claim is None:
        if (allocation is not None or commitment is not None or forward is not None
                or any(os.path.lexists(root / item) for item in
                       ('nonce.bin', 'heldout-record.json', 'commitment.json', 'owner-started.json'))):
            raise SuccessorStateError('unannounced successor state is uncertain')
        state = {'schema': _DISPOSITION_SCHEMA, 'status': 'not_allocated_unchecked'}
    else:
        _claim(claim)
        if allocation is None:
            if (not allow_unannounced_claim or commitment is not None or forward is not None
                    or any(os.path.lexists(root / item) for item in
                           ('nonce.bin', 'heldout-record.json', 'commitment.json', 'owner-started.json'))):
                raise SuccessorStateError('phase claim lacks an allocation announcement')
            state = {'schema': _DISPOSITION_SCHEMA, 'status': 'claim_not_yet_announced',
                     'phase_claim_sha256': claim_sha}
        else:
            if allocation != _allocation_value(claim, claim_sha):
                raise SuccessorStateError('successor allocation differs from the claim')
            state = {'schema': _DISPOSITION_SCHEMA, 'status': 'allocation_pending_unchecked',
                     'phase_claim_sha256': claim_sha,
                     'allocation_sha256': allocation_sha, 'operation_sha256': claim['operation_sha256'],
                     'inventory_sha256': claim['inventory_sha256']}
            if commitment is not None:
                if (set(commitment) != {'schema', 'track', 'allocation_sha256', 'heldout_commitment',
                                        'commitment_file_sha256'}
                        or commitment['schema'] != 'mlg-natural-successor-commitment-observed-v1'
                        or commitment['track'] != 'natural-successor'
                        or commitment['allocation_sha256'] != allocation_sha):
                    raise SuccessorStateError('successor commitment differs')
                _sha(commitment['heldout_commitment'])
                _sha(commitment['commitment_file_sha256'])
                state.update(status='committed_comparison_required', commitment_marker_sha256=commitment_sha,
                             heldout_commitment=commitment['heldout_commitment'],
                             commitment_file_sha256=commitment['commitment_file_sha256'])
            if forward is not None:
                if (commitment is None or set(forward) != {'schema', 'track', 'allocation_sha256',
                         'commitment_marker_sha256', 'peer_disposition'}
                        or forward['schema'] != 'mlg-natural-successor-forward-v1'
                        or forward['track'] != 'natural-successor'
                        or forward['allocation_sha256'] != allocation_sha
                        or forward['commitment_marker_sha256'] != commitment_sha
                        or forward['peer_disposition'] not in
                           ('injection_not_allocated_unchecked', 'injection_allocation_pending_unchecked')):
                    raise SuccessorStateError('successor forward differs')
                state.update(status='possibly_exposed_comparison_required', forward_marker_sha256=forward_sha,
                             forward_peer_disposition=forward['peer_disposition'])
    _guard(gate, root)
    return state


def inspect(gate, root):
    """Return digest-only state while the caller holds the same shared gate.

    An allocation without a commitment is explicitly unchecked. A committed
    or possibly exposed successor requires an independently verified pairwise
    comparison before a later affected injection forward. The caller must keep
    the gate through its own persisted disposition and forward announcement.
    """
    return _inspect(gate, root)


def _publish(gate, root, name, value):
    root = _guard(gate, root)
    raw = _canonical(value)
    with (root / name).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise SuccessorStateError('marker write was incomplete')
        stream.flush()
        os.fsync(stream.fileno())
    _guard(gate, root)
    actual, digest = _read(root / name)
    if actual != value or digest != hashlib.sha256(raw).hexdigest():
        raise SuccessorStateError('marker readback differs')
    return digest


def announce_allocation(gate, root):
    """Announce the already claimed successor under the shared lock before nonce draw."""
    state = _inspect(gate, root, allow_unannounced_claim=True)
    if state['status'] != 'claim_not_yet_announced':
        raise SuccessorStateError('successor allocation has already started')
    claim, digest = _read(Path(root) / _CLAIM)
    if digest != state['phase_claim_sha256']:
        raise SuccessorStateError('phase claim changed')
    _publish(gate, root, _ALLOC, _allocation_value(claim, digest))
    return inspect(gate, root)


def announce_commitment(gate, root, commitment_bytes, *, publication_check):
    """Announce an externally authenticated public salted commitment."""
    state = inspect(gate, root)
    if state['status'] != 'allocation_pending_unchecked':
        raise SuccessorStateError('successor allocation is not pending')
    if type(commitment_bytes) is not bytes or not 1 <= len(commitment_bytes) <= 8192:
        raise SuccessorStateError('public commitment bytes are invalid')
    try:
        public = json.loads(commitment_bytes)
        if (type(public) is not dict or _canonical(public) != commitment_bytes
                or set(public) != {'schema', 'prediction_freeze_sha256', 'pool_sha256', 'inputs_sha256',
                    'pool_clearance_sha256', 'exposure_inventory_sha256', 'old_natural_commitment',
                    'protocol_sha256', 'amendment_sha256', 'operation_sha256', 'heldout_commitment'}
                or public['schema'] != 'mlg-natural-successor-public-commitment-v1'
                or public['prediction_freeze_sha256'] != FREEZE or public['pool_sha256'] != POOL
                or public['inputs_sha256'] != INPUTS or public['pool_clearance_sha256'] != PUBLIC_CLEARANCE
                or public['exposure_inventory_sha256'] != state['inventory_sha256']
                or public['old_natural_commitment'] != OLD_COMMITMENT or public['protocol_sha256'] != PROTOCOL
                or public['amendment_sha256'] != AMENDMENT or public['operation_sha256'] != state['operation_sha256']):
            raise SuccessorStateError('public commitment identity differs')
        _sha(public['heldout_commitment'])
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise SuccessorStateError('public commitment is invalid') from None
    if not callable(publication_check) or publication_check(commitment_bytes) is not True:
        raise SuccessorStateError('public commitment publication is unverified')
    if inspect(gate, root) != state:
        raise SuccessorStateError('successor state changed during publication check')
    marker = {'schema': 'mlg-natural-successor-commitment-observed-v1', 'track': 'natural-successor',
              'allocation_sha256': state['allocation_sha256'], 'heldout_commitment': public['heldout_commitment'],
              'commitment_file_sha256': hashlib.sha256(commitment_bytes).hexdigest()}
    _publish(gate, root, _COMMIT, marker)
    return inspect(gate, root)


def announce_forward(gate, root):
    """Announce a possible forward only when injection is absent or pending.

    A committed injection peer always stops here until a separately reviewed
    pairwise verifier is integrated. The caller keeps the shared gate through
    its final currentness check and first actual model return.
    """
    state = inspect(gate, root)
    if state['status'] != 'committed_comparison_required':
        raise SuccessorStateError('successor is not committed or already announced')
    allocation = _allocation(gate.root, 'injection')
    commitment = _commitment(gate.root, 'injection')
    forward = _forward(gate.root, 'injection')
    if allocation is not None and allocation['freeze_sha256'] != INJECTION_FREEZE:
        raise SuccessorStateError('injection allocation belongs to a different freeze')
    if allocation is None:
        if commitment is not None or forward is not None:
            raise SuccessorStateError('injection state is inconsistent')
        disposition = 'injection_not_allocated_unchecked'
    elif commitment is None:
        if forward is not None:
            raise SuccessorStateError('uncommitted injection has an uncertain forward')
        disposition = 'injection_allocation_pending_unchecked'
    else:
        raise SuccessorStateError('committed injection requires a reviewed comparison')
    if (inspect(gate, root) != state or allocation != _allocation(gate.root, 'injection')
            or commitment != _commitment(gate.root, 'injection') or forward != _forward(gate.root, 'injection')):
        raise SuccessorStateError('shared state changed during the forward decision')
    marker = {'schema': 'mlg-natural-successor-forward-v1', 'track': 'natural-successor',
              'allocation_sha256': state['allocation_sha256'],
              'commitment_marker_sha256': state['commitment_marker_sha256'],
              'peer_disposition': disposition}
    _publish(gate, root, _FORWARD, marker)
    return inspect(gate, root)
