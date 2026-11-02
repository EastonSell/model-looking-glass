"""Separate cooperative one-use markers for the prospective successor.

The published operation and caller must authenticate this module and bind one
protected root's normalized path and filesystem identity. These primitives do
not establish publication, ACLs, budget, source custody, comparator execution,
clearance, exposure currentness, peer state, or permission to draw. The caller
must establish each before the corresponding marker; arbitrary caller-chosen
root identities are not authority. Same-account replacement or rollback is not
prevented. File fsync does not establish directory-entry durability across an
OS crash or power loss; such a crash requires external reconciliation. An
incomplete or ambiguous marker consumes the one-use attempt, and an apparently
complete marker without an observed successful caller exit is still ambiguous.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat


_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def _sha_values(*values):
    if any(type(value) is not str or _SHA.fullmatch(value) is None for value in values):
        raise ValueError('invalid claim binding')


def _root(root, root_path_sha256, root_device, root_inode):
    _sha_values(root_path_sha256)
    if (type(root_device) is not int or root_device < 0
            or type(root_inode) is not int or root_inode <= 0):
        raise ValueError('invalid root identity')
    path = Path(root)
    if not path.is_absolute():
        raise ValueError('absolute protected root required')
    normalized = os.path.normcase(os.path.realpath(path))
    if (normalized != os.path.normcase(os.path.abspath(path))
            or not hmac.compare_digest(hashlib.sha256(normalized.encode('utf-8')).hexdigest(), root_path_sha256)):
        raise ValueError('protected root path differs')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('protected path contains a reparse point')
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (root_device, root_inode):
        raise ValueError('protected root identity differs')
    return path


def _write_new(path, data):
    with path.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('incomplete claim write')
        stream.flush()
        os.fsync(stream.fileno())


def _read_claim(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 1024 or info.st_nlink != 1
            or not 1 <= info.st_size <= 8192):
        raise ValueError('invalid query claim file')
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError('query claim changed while reading')
    try:
        value = json.loads(raw)
        if _canonical(value) != raw:
            raise ValueError('query claim is not canonical')
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('invalid query claim bytes') from error
    return raw, value


def _publish(root, root_path_sha256, root_device, root_inode, filename, value):
    path = _root(root, root_path_sha256, root_device, root_inode)
    raw = _canonical(value)
    if len(raw) > 8192:
        raise ValueError('claim exceeds its bound')
    _write_new(path / filename, raw)
    _root(root, root_path_sha256, root_device, root_inode)
    actual, _ = _read_claim(path / filename)
    if not hmac.compare_digest(actual, raw):
        raise ValueError('claim was not retained intact')
    return hashlib.sha256(raw).hexdigest()


def claim_query(*, root, root_path_sha256, root_device, root_inode, operation_sha256,
                pool_input_sha256, old_commitment, comparator_sha256):
    """Exclusively consume the fixed pool query before touching private rows."""
    _sha_values(operation_sha256, pool_input_sha256, old_commitment, comparator_sha256)
    value = {'schema': 'mlg-natural-successor-claim-v1', 'kind': 'pool-query',
             'operation_sha256': operation_sha256, 'pool_input_sha256': pool_input_sha256,
             'old_commitment': old_commitment, 'comparator_sha256': comparator_sha256}
    return _publish(root, root_path_sha256, root_device, root_inode, 'pool-query-claim.json', value)


def claim_phase(*, root, root_path_sha256, root_device, root_inode, operation_sha256,
                query_claim_sha256, pool_clearance_sha256, inventory_sha256):
    """Exclusively consume the separate phase before generating its one nonce.

    A digest here binds a claimed clearance; it does not authenticate the
    comparator or establish that it returned clear. Those checks are the
    reviewed caller's responsibility before this operation.
    """
    _sha_values(operation_sha256, query_claim_sha256, pool_clearance_sha256, inventory_sha256)
    path = _root(root, root_path_sha256, root_device, root_inode)
    raw, query = _read_claim(path / 'pool-query-claim.json')
    keys = {'schema', 'kind', 'operation_sha256', 'pool_input_sha256', 'old_commitment', 'comparator_sha256'}
    if (not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), query_claim_sha256)
            or type(query) is not dict or set(query) != keys
            or query['schema'] != 'mlg-natural-successor-claim-v1' or query['kind'] != 'pool-query'
            or query['operation_sha256'] != operation_sha256):
        raise ValueError('query claim or operation differs')
    _sha_values(query['pool_input_sha256'], query['old_commitment'], query['comparator_sha256'])
    value = {'schema': 'mlg-natural-successor-claim-v1', 'kind': 'phase',
             'operation_sha256': operation_sha256, 'query_claim_sha256': query_claim_sha256,
             'pool_clearance_sha256': pool_clearance_sha256, 'inventory_sha256': inventory_sha256}
    return _publish(root, root_path_sha256, root_device, root_inode, 'phase-claim.json', value)


def claim_public_phase(*, root, root_path_sha256, root_device, root_inode, operation_sha256,
                       public_clearance_sha256, inventory_sha256, amendment_sha256):
    """Consume a new phase following an independently verified public-support check.

    This marker binds the stated public receipt, inventory and amendment; it
    does not authenticate their contents, publication or currentness. A
    source-bound owner must verify them and hold their leases through this
    exclusive claim. Any old/partial query or phase artifact stops the attempt.
    The cooperative owner must also establish exclusive control of this root
    and no legacy query process in flight: checking different filenames cannot
    serialize a concurrent query and phase claim by itself.
    """
    _sha_values(operation_sha256, public_clearance_sha256, inventory_sha256, amendment_sha256)
    path = _root(root, root_path_sha256, root_device, root_inode)
    forbidden = ('pool-query-invocation.json', 'pool-query-claim.json', 'pool-query-receipt.json',
                 'pool-query-observation.json', 'pool-query-supervisor', 'pool-query-outer',
                 'phase-claim.json', 'nonce.bin', 'heldout-record.json', 'commitment.json', 'owner-started.json')
    if any(os.path.lexists(path / name) for name in forbidden):
        raise ValueError('earlier or uncertain query/phase state prevents a new claim')
    value = {'schema': 'mlg-natural-successor-claim-v1', 'kind': 'phase-public-support',
             'operation_sha256': operation_sha256, 'public_clearance_sha256': public_clearance_sha256,
             'inventory_sha256': inventory_sha256, 'amendment_sha256': amendment_sha256}
    return _publish(root, root_path_sha256, root_device, root_inode, 'phase-claim.json', value)
