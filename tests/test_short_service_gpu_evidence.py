"""Small, model-free examples for the fixed short-service saved-byte reader."""
import struct
import hashlib

import pytest

from looking_glass import short_service_gpu_evidence as evidence
from looking_glass.short_service_gpu_evidence import _compare, _compare_payloads, validate_supervision


def floats(*values):
    return struct.pack('<' + 'f' * len(values), *values)


def test_exact_float_bits_include_subnormal_and_reject_nonfinite():
    result = _compare(struct.pack('<II', 1, 0x80000001), floats(0., 0.))
    assert result['count'] == 2 and result['max_abs'] == 2. ** -149
    with pytest.raises(ValueError):
        _compare(struct.pack('<I', 0x7f800000), floats(0.))


def test_two_client_saved_bytes_and_distinct_last_active_rows():
    a = floats(1, 2, 3, 4, 5, 6)
    b = floats(8, 9, 10, 11, 12, 13)
    values = {'reference-alpha': a, 'reference-beta': b, 'client-alpha': a, 'client-beta': b}
    result = _compare_payloads(values, bucket=2, vocab=3, positions=(1, 0))
    assert result['reference_separation']['violations'] == 3
    assert [item['active_position'] for item in result['comparisons']] == [1, 0]
    assert all(item['full']['violations'] == 0 for item in result['comparisons'])
    values['client-alpha'] = b
    with pytest.raises(ValueError, match='own distinct'):
        _compare_payloads(values, bucket=2, vocab=3, positions=(1, 0))


def test_indistinguishable_references_and_bad_types_reject():
    a = floats(1, 2, 3, 4, 5, 6)
    values = {name: a for name in ('reference-alpha', 'reference-beta', 'client-alpha', 'client-beta')}
    with pytest.raises(ValueError, match='indistinguishable'):
        _compare_payloads(values, bucket=2, vocab=3, positions=(1, 1))
    with pytest.raises(ValueError, match='layout'):
        _compare_payloads(values, bucket=True, vocab=3, positions=(1, 1))


def test_supervisor_requires_exact_clean_reaped_exit_types():
    record = {'run_id': 'run', 'exit_code': 0, 'timed_out': False, 'clean': True,
              'pid': 123, 'supervisor_run_id': 'supervisor'}
    assert validate_supervision(record, run_id='run')
    for key, value in [('exit_code', False), ('timed_out', 0), ('clean', 1), ('pid', True)]:
        altered = dict(record, **{key: value})
        with pytest.raises(ValueError):
            validate_supervision(altered, run_id='run')


def test_saved_reader_rejects_rehashed_swap_missing_and_extra_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, 'VOCAB', 3)
    a, b = floats(*([1., 2., 3.] * 16)), floats(*([4., 5., 6.] * 16))
    payloads = {'reference-alpha': a, 'reference-beta': b, 'client-alpha': a, 'client-beta': b}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    for name, value in payloads.items():
        (tmp_path / (name + '.f32')).write_bytes(value)
    assert evidence.verify_payloads(tmp_path, bucket=16, positions=(15, 14), expected_hashes=hashes)
    (tmp_path / 'client-alpha.f32').write_bytes(b)
    hashes['client-alpha'] = hashlib.sha256(b).hexdigest()
    with pytest.raises(ValueError, match='own distinct'):
        evidence.verify_payloads(tmp_path, bucket=16, positions=(15, 14), expected_hashes=hashes)
    (tmp_path / 'client-alpha.f32').write_bytes(a)
    hashes['client-alpha'] = hashlib.sha256(a).hexdigest()
    (tmp_path / 'unexpected.f32').write_bytes(a)
    with pytest.raises(ValueError, match='inventory'):
        evidence.verify_payloads(tmp_path, bucket=16, positions=(15, 14), expected_hashes=hashes)
    (tmp_path / 'unexpected.f32').unlink()
    (tmp_path / 'client-beta.f32').unlink()
    with pytest.raises(ValueError, match='inventory'):
        evidence.verify_payloads(tmp_path, bucket=16, positions=(15, 14), expected_hashes=hashes)
