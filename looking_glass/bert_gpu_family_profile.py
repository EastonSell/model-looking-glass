"""Fixed development-family workload and bounded post-close journal audit.

This module does not load a model, acquire a device, allocate a resource
budget, authenticate callbacks, or authorize CUDA execution. A supervised
owner must establish those conditions and retain failure/partial evidence.
"""

import hashlib
import json
import math
from pathlib import Path

MAX_JOURNAL_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 12 * 1024
MAX_RECORDS = 328
MAX_LOGICAL_JOURNAL_PAYLOAD = 20 * 1024 * 1024


def _read_profile_journal(path):
    """Read a closed bounded journal; reject duplicate keys and invalid numbers."""
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_JOURNAL_BYTES + 1)
    if not raw or len(raw) > MAX_JOURNAL_BYTES or not raw.endswith(b'\n'):
        raise ValueError('profile journal is missing, oversized or incomplete')
    # Bound object count before splitlines allocates its per-record list. The
    # logical allowance covers the raw file, split bytes, parsed string data,
    # decoded vector data and two transient records; it is not an RSS cap.
    if (b'\r' in raw or raw.count(b'\n') > MAX_RECORDS
            or 4 * len(raw) + 2 * MAX_RECORD_BYTES > MAX_LOGICAL_JOURNAL_PAYLOAD):
        raise ValueError('profile journal exceeds its logical payload allowance')
    lines = raw[:-1].split(b'\n')
    if len(lines) > MAX_RECORDS:
        raise ValueError('profile journal has too many records')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate profile journal key')
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite profile journal number')
        return result

    records = []
    for line in lines:
        if not line or len(line) > MAX_RECORD_BYTES:
            raise ValueError('profile journal record exceeds its byte limit')
        try:
            record = json.loads(line.decode('ascii'), object_pairs_hook=unique, parse_float=number,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite journal number')))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError('invalid profile journal record') from error
        if type(record) is not dict:
            raise ValueError('profile journal records must be objects')
        records.append(record)
    return records, hashlib.sha256(raw).hexdigest()


def execute_family_profile(model, manifest, *, observer, checkpoint, journal_path,
                           is_id, are_id, mask_id, hidden_size, expected_device):
    """Run and audit only development family 1 through trusted owner callbacks."""
    from .bert_agreement_audit import audit_family_events
    from .bert_agreement_runner import _run_family
    from .bert_agreement_study import EventJournal

    if (not callable(observer) or not callable(checkpoint)
            or type(expected_device) is not str or expected_device not in ('cpu', 'cuda:0')):
        raise ValueError('trusted observation, checkpoint and fixed device are required')
    count = 0
    with EventJournal(Path(journal_path), max_bytes=MAX_JOURNAL_BYTES) as journal:
        def emit(record):
            raw = (json.dumps(record, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
            if len(raw) - 1 > MAX_RECORD_BYTES:
                raise ValueError('profile journal record exceeds its byte limit')
            journal.emit(record)

        def observe(model_arg, inputs, **options):
            nonlocal count
            count += 1
            if count > 140:
                raise RuntimeError('profile exceeded its fixed forward count')
            checkpoint('before', count)
            result = observer(model_arg, inputs, **options)
            checkpoint('after', count)
            return result

        live = _run_family(model, manifest, split='development', family=1,
                           is_id=is_id, are_id=are_id, mask_id=mask_id,
                           emit=emit, observer=observe)
    journal_sha = journal.sha256
    records, digest = _read_profile_journal(journal_path)
    if count != 140 or len(records) != MAX_RECORDS or digest != journal_sha:
        raise RuntimeError('profile journal is incomplete or changed after close')
    audited = audit_family_events(manifest, records, split='development', family=1,
        mask_id=mask_id, hidden_size=hidden_size, expected_device=expected_device)
    if live != audited:
        raise RuntimeError('closed profile journal does not reproduce the live family result')
    return {'forward_count': count, 'journal_sha256': digest, 'record_count': len(records),
            'expected_device': expected_device, 'live_cases': live['cases'],
            'audited_cases': audited['cases'], 'site_means': audited['site_means'],
            'control_count': len(audited['controls'])}
