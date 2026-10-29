"""Fabricated files only; never scan any protected study evidence in tests."""
from collections import namedtuple
import hashlib
import os

import pytest

try:
    from looking_glass import natural_successor_accounting as accounting
except ImportError:
    accounting = None


def test_all_phase_bytes_and_unknown_partials_are_charged(tmp_path):
    assert accounting is not None
    phase = tmp_path / 'new-phase'
    tensors = phase / 'owner' / 'attempt' / 'tensors'
    tensors.mkdir(parents=True)
    data = b'12345678'
    (tensors / (hashlib.sha256(data).hexdigest() + '.f32le')).write_bytes(data)
    (tensors.parent / 'events.jsonl').write_bytes(b'1234')
    (phase / 'orphan.partial').write_bytes(b'123')
    assert accounting.account_phase(phase) == {'tensor_bytes': 8, 'event_bytes': 4, 'side_bytes': 3}
    with pytest.raises(ValueError):
        accounting.account_phase(phase, reserve_side_bytes=accounting.SIDE_LIMIT)


def test_hardlinks_and_oversized_unknown_files_fail_closed(tmp_path):
    assert accounting is not None
    phase = tmp_path / 'new-phase'
    phase.mkdir()
    first = phase / 'first'
    first.write_bytes(b'a')
    os.link(first, phase / 'alias')
    with pytest.raises(ValueError):
        accounting.account_phase(phase)
    (phase / 'alias').unlink()
    with first.open('r+b') as stream:
        stream.truncate(accounting.SIDE_LIMIT + 1)
    with pytest.raises(ValueError):
        accounting.account_phase(phase)


def test_unknown_or_partial_files_in_tensor_tree_are_side_evidence(tmp_path):
    assert accounting is not None
    phase = tmp_path / 'phase'
    tensors = phase / 'owner' / 'attempt' / 'tensors'
    tensors.mkdir(parents=True)
    (tensors / 'unknown.partial').write_bytes(b'123')
    (tensors / ('a' * 64 + '.f32le')).write_bytes(b'wrong')
    nested = tensors / 'unexpected'
    nested.mkdir()
    (nested / ('b' * 64 + '.f32le')).write_bytes(b'1234')
    assert accounting.account_phase(phase) == {'tensor_bytes': 0, 'event_bytes': 0, 'side_bytes': 12}
    with (tensors / 'unknown.partial').open('r+b') as stream:
        stream.truncate(accounting.SIDE_LIMIT + 1)
    with pytest.raises(ValueError):
        accounting.account_phase(phase)


def test_free_space_and_orphan_staging_are_observed_without_deletion(tmp_path, monkeypatch):
    assert accounting is not None
    phase, cache = tmp_path / 'phase', tmp_path / 'cache'
    phase.mkdir()
    cache.mkdir()
    stage = cache / 'looking-glass-stage-fabricated'
    stage.mkdir()
    (stage / 'orphan.partial').write_bytes(b'12345')
    Usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(accounting.shutil, 'disk_usage', lambda _: Usage(10 ** 12, 0, 4 * 1024 ** 3))
    result = accounting.inspect_resources(phase, cache)
    assert result['stages'] == [{'name': stage.name, 'files': 1, 'apparent_bytes': 5}]
    assert (stage / 'orphan.partial').is_file()
    monkeypatch.setattr(accounting.shutil, 'disk_usage', lambda _: Usage(10 ** 12, 0, 3 * 1024 ** 3 - 1))
    with pytest.raises(ValueError):
        accounting.inspect_resources(phase, cache)
