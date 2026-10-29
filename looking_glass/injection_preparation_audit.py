"""Private structural audit of a completed injection split preparation.

This callback is for a trusted, source-verified process supervisor. It does
not start a process, observe an exit, publish a commitment, or authorize a
forward. Its digest-only result remains private until separately reviewed.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat

from . import injection_split_custody as custody
from . import pretrained_number_commitments as commitments


_LIMITS = {'claim.json': 8192, 'nonce.bin': 32, 'heldout-record.json': 1024 * 1024,
           'inventory.json': 16 * 1024 * 1024, 'commitment.json': 8192,
           'preparation-complete.json': 8192}
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink)


def _read(path, maximum):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('a private absolute path is required')
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_nlink != 1
            or not before.st_ino or getattr(before, 'st_file_attributes', 0) & _REPARSE
            or not 0 < before.st_size <= maximum):
        raise ValueError('a bounded private regular file is required')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(maximum + 1)
        finished = os.fstat(stream.fileno())
    after = path.lstat()
    if (len(data) != before.st_size or _identity(before) != _identity(opened)
            or _identity(before) != _identity(finished) or _identity(before) != _identity(after)):
        raise ValueError('a private file changed during audit')
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_study_preparation(phase_root, *, public, **kwargs):
    """Require the fixed published injection discovery before auditing files."""
    try:
        if type(public) is not custody.PublicInputs:
            raise ValueError('fixed study inputs are required')
        freeze = custody._parse_canonical(public.freeze_bytes)
        operation = custody._parse_canonical(public.operational_binding_bytes)
        track = freeze['tracks']['injection']
        if (type(operation) is not dict or set(operation) != custody._OP_V2_KEYS
                or operation.get('schema') != 'mlg-injection-heldout-operational-binding-v2'
                or public.freeze_sha256 != custody._STUDY_FREEZE
                or _sha(public.freeze_bytes) != custody._STUDY_FREEZE
                or type(freeze['tracks']) is not dict or set(freeze['tracks']) != {'injection'}
                or track.get('source_commit') != custody._STUDY_SOURCE
                or track.get('runtime_sha256') != custody._STUDY_RUNTIME
                or _sha(public.runtime_bytes) != custody._STUDY_RUNTIME
                or track.get('prediction_sha256') != custody._STUDY_PREDICTION
                or _sha(public.prediction_bytes) != custody._STUDY_PREDICTION
                or public.plan_sha256 != custody._STUDY_PLAN or _sha(public.plan_bytes) != custody._STUDY_PLAN
                or public.site_commitment != custody._STUDY_SITE_COMMITMENT):
            raise ValueError('the published discovery identity differs')
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ValueError('fixed injection discovery identity is invalid') from None
    return audit_preparation(phase_root, public=public, **kwargs)


def audit_preparation(phase_root, *, public, config_path, config_sha256, ledger_path,
                      site_nonce, site_record_bytes, tokenizer, publication_check,
                      source_check, prediction_check, observe_inventory, claim_check):
    """Recheck a closed, fabricated-or-real preparation without revealing it.

    This generic helper checks self-consistency; the real study must use
    ``audit_study_preparation`` to pin its published discovery. The caller must
    independently establish zero child exit, process cleanup,
    published source and configuration identities, private access controls and
    the real ledger row. Callbacks are trusted observations, not attestations.
    The returned object is suitable for the generic supervisor's structural
    callback; it does not attest the supervisor's own exit or publication of
    the newly produced commitment.
    """
    try:
        phase = Path(phase_root)
        ledger = Path(ledger_path)
        if not phase.is_absolute() or not ledger.is_absolute():
            raise ValueError('private paths must be absolute')
        info = phase.lstat()
        if (not stat.S_ISDIR(info.st_mode) or phase.is_symlink() or not info.st_ino
                or getattr(info, 'st_file_attributes', 0) & _REPARSE
                or phase.name != 'injection-heldout-' + public.freeze_sha256):
            raise ValueError('the private phase identity is invalid')
        if {entry.name for entry in phase.iterdir()} != set(_LIMITS):
            raise ValueError('preparation contains missing or unexpected files')
        if not callable(observe_inventory) or not callable(claim_check):
            raise ValueError('trusted inventory and ledger observations are required')
        config = _read(config_path, 65536)
        if _sha(config) != custody._hex(config_sha256):
            raise ValueError('preparation configuration differs')
        freeze, snapshot, plan_entropy = custody._check_inputs(public, site_nonce, site_record_bytes,
                                                               publication_check, source_check, prediction_check)
        if (observe_inventory() != snapshot.sha256
                or claim_check(ledger, 'injection_heldout', phase) is not True):
            raise ValueError('the inventory or consumed claim differs')
        files = {name: _read(phase / name, limit) for name, limit in _LIMITS.items()}
        if (len(files['nonce.bin']) != 32 or hmac.compare_digest(files['nonce.bin'], plan_entropy)
                or files['inventory.json'] != public.inventory_bytes):
            raise ValueError('private nonce or inventory differs')
        claim = {'schema': 'mlg-injection-heldout-claim-v1',
                 'prediction_freeze_sha256': freeze.sha256,
                 'exposure_inventory_sha256': snapshot.sha256,
                 'operational_binding_sha256': public.operational_binding_sha256,
                 'logical_limit_bytes': 72 * 1024 ** 2, 'disk_limit_bytes': 192 * 1024 ** 2}
        if files['claim.json'] != commitments.canonical_json(claim):
            raise ValueError('the phase claim differs')
        published = custody._parse_canonical(files['commitment.json'])
        if (type(published) is not dict or set(published) != custody._COMMIT_KEYS['injection']
                or published.get('schema') != 'mlg-injection-heldout-public-commitment-v1'
                or published.get('prediction_freeze_sha256') != freeze.sha256
                or published.get('exposure_inventory_sha256') != snapshot.sha256
                or published.get('custody_source_sha256') != custody._read_source_hash()
                or published.get('operational_binding_sha256') != public.operational_binding_sha256):
            raise ValueError('the prepared commitment differs')
        for key in custody._COMMIT_KEYS['injection'] - {'schema'}:
            custody._hex(published[key])
        complete = {'schema': 'mlg-injection-heldout-preparation-complete-v1',
                    'commitment_file_sha256': _sha(files['commitment.json']),
                    'operational_binding_sha256': public.operational_binding_sha256}
        if files['preparation-complete.json'] != commitments.canonical_json(complete):
            raise ValueError('preparation completion differs')
        commitments.verify_heldout_reveal(freeze, files['nonce.bin'], files['heldout-record.json'],
            published['heldout_commitment'], inventory_snapshot=snapshot, site_nonce=site_nonce,
            site_record_bytes=site_record_bytes, site_commitment=public.site_commitment)
        allocation = commitments.sample_families(freeze, files['nonce.bin'], inventory_snapshot=snapshot,
            current_inventory_sha256=snapshot.sha256, site_nonce=site_nonce,
            site_record_bytes=site_record_bytes, site_commitment=public.site_commitment)
        cases = custody.tokenize_assignment(allocation, tokenizer)
        record = json.loads(files['heldout-record.json'])
        if cases != [family['cases'] for family in record['families']]:
            raise ValueError('tokenizer does not match the sealed record')
        custody.reject_known_row_prefixes(cases, public.inventory_bytes)
        if (observe_inventory() != snapshot.sha256
                or claim_check(ledger, 'injection_heldout', phase) is not True
                or _read(config_path, 65536) != config
                or any(_read(phase / name, limit) != files[name] for name, limit in _LIMITS.items())
                or {entry.name for entry in phase.iterdir()} != set(_LIMITS)
                or _identity(phase.lstat()) != _identity(info)):
            raise ValueError('preparation changed during audit')
        custody._check_inputs(public, site_nonce, site_record_bytes, publication_check,
                              source_check, prediction_check)
        return {'status': 'evidence_complete', 'inventory': {'retained_tensor_bytes': 0},
                'config_sha256': config_sha256,
                'operational_binding_sha256': public.operational_binding_sha256,
                'preparation': {name: _sha(files[name]) for name in sorted(files)}}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise ValueError('injection preparation audit is incomplete') from None
