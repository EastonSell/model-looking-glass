"""Fabricated roots only: no production claim, nonce, or private data."""
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

try:
    from looking_glass import natural_successor_claim as claim
except ImportError:
    claim = None


@pytest.fixture
def identity(tmp_path):
    root = tmp_path / 'new-phase'
    root.mkdir()
    info = root.stat()
    normalized = os.path.normcase(os.path.realpath(root))
    return {'root': root, 'root_path_sha256': hashlib.sha256(normalized.encode('utf-8')).hexdigest(),
            'root_device': info.st_dev, 'root_inode': info.st_ino,
            'operation_sha256': '1' * 64}


def query(identity):
    return claim.claim_query(**identity, pool_input_sha256='2' * 64,
                             old_commitment='3' * 64, comparator_sha256='4' * 64)


def test_query_and_phase_are_distinct_exclusive_markers(identity):
    assert claim is not None
    query_sha = query(identity)
    query_file = identity['root'] / 'pool-query-claim.json'
    assert hashlib.sha256(query_file.read_bytes()).hexdigest() == query_sha
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        query(identity)
    phase_sha = claim.claim_phase(**identity, query_claim_sha256=query_sha,
                                  pool_clearance_sha256='5' * 64, inventory_sha256='6' * 64)
    phase_file = identity['root'] / 'phase-claim.json'
    assert hashlib.sha256(phase_file.read_bytes()).hexdigest() == phase_sha
    assert json.loads(phase_file.read_bytes())['query_claim_sha256'] == query_sha
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        claim.claim_phase(**identity, query_claim_sha256=query_sha,
                          pool_clearance_sha256='5' * 64, inventory_sha256='6' * 64)


def test_copy_or_wrong_root_identity_cannot_consume_a_claim(identity, tmp_path):
    assert claim is not None
    other = tmp_path / 'other'
    other.mkdir()
    with pytest.raises((ValueError, RuntimeError)):
        query(dict(identity, root=other))
    with pytest.raises((ValueError, RuntimeError)):
        query(dict(identity, root_inode=identity['root_inode'] + 1))
    assert not list(other.iterdir()) and not list(identity['root'].iterdir())


def test_phase_requires_exact_query_claim_and_unmodified_operation(identity):
    assert claim is not None
    query_sha = query(identity)
    with pytest.raises((ValueError, RuntimeError)):
        claim.claim_phase(**identity, query_claim_sha256='f' * 64,
                          pool_clearance_sha256='5' * 64, inventory_sha256='6' * 64)
    with pytest.raises((ValueError, RuntimeError)):
        claim.claim_phase(**dict(identity, operation_sha256='f' * 64), query_claim_sha256=query_sha,
                          pool_clearance_sha256='5' * 64, inventory_sha256='6' * 64)
    assert not (identity['root'] / 'phase-claim.json').exists()


def test_incomplete_marker_consumes_the_one_use_query(identity, monkeypatch):
    assert claim is not None
    def interrupted(path, data):
        with path.open('xb') as stream:
            stream.write(data[:5])
            stream.flush()
            os.fsync(stream.fileno())
        raise OSError('fabricated interruption')
    monkeypatch.setattr(claim, '_write_new', interrupted)
    with pytest.raises(OSError):
        query(identity)
    assert (identity['root'] / 'pool-query-claim.json').read_bytes() == b'{"com'
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        query(identity)


def test_interruption_after_complete_write_still_consumes_query(identity, monkeypatch):
    assert claim is not None
    original = claim._write_new
    def interrupted(path, data):
        original(path, data)
        raise OSError('fabricated lost return')
    monkeypatch.setattr(claim, '_write_new', interrupted)
    with pytest.raises(OSError):
        query(identity)
    assert json.loads((identity['root'] / 'pool-query-claim.json').read_bytes())['kind'] == 'pool-query'
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        query(identity)


def test_concurrent_query_announcements_have_one_winner(identity):
    assert claim is not None
    def attempt():
        try:
            return query(identity)
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: attempt(), range(2)))
    assert len([value for value in outcomes if value is not None]) == 1
    assert hashlib.sha256((identity['root'] / 'pool-query-claim.json').read_bytes()).hexdigest() in outcomes
