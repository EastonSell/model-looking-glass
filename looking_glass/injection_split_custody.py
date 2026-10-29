"""Trusted-caller preparation for one injection-only held-out split.

The caller supplies independently verified publication, current-inventory,
ledger and pinned-tokenizer operations. This module does not attest those
operations, create an OS security boundary, launch a model or publish anything.
It preserves the frozen discovery commitment and never draws a replacement.
"""
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets

from . import pretrained_number_commitments as commitments


_LOGICAL_LIMIT = 72 * 1024**2
_DISK_LIMIT = 192 * 1024**2
_MAX_PUBLIC_BYTES = 16 * 1024**2
_TARGETS = {' is': 318, ' are': 389, ' was': 373, ' were': 547}
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_HEX40 = re.compile(r'[0-9a-f]{40}\Z')
_OP_KEYS = frozenset({'schema', 'protocol_sha256', 'checkpoint_revision',
    'discovery_source_commit', 'prediction_freeze_sha256', 'runtime_sha256',
    'prediction_sha256', 'plan_sha256', 'site_commitment', 'inventory_sha256',
    'custody_source_sha256', 'executor_source_sha256', 'comparator_source_sha256',
    'ledger_source_sha256', 'tokenizer_sha256', 'supplement_sha256',
    'owner_source_sha256', 'journal_source_sha256', 'supervisor_source_sha256',
    'auditor_source_sha256', 'gate_source_sha256', 'inventory_source_sha256',
    'inventory_source_commit', 'accounting_amendment_sha256', 'lease_source_sha256'})
_OP_V2_KEYS = _OP_KEYS | {'preparation_child_source_sha256',
                          'preparation_audit_source_sha256', 'outer_observer_source_sha256',
                          'gate_allocation_caller_source_sha256', 'gate_transition_caller_source_sha256'}
_STUDY_FREEZE = 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'
_STUDY_SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
_STUDY_RUNTIME = 'b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e'
_STUDY_PREDICTION = '11376d8aae517db76d6a33c1ae5dc1f4b0c747a9d5216c9b02a17b1b0b300107'
_STUDY_PLAN = 'e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8'
_STUDY_SITE_COMMITMENT = 'c3739cb173d3c1bc3e3adda70025ff7f26a73c4690e50ecf5baf06ff8b359b9d'
_COMMIT_KEYS = {
    'natural': frozenset({'schema', 'prediction_freeze_sha256', 'exposure_inventory_sha256',
                          'heldout_commitment', 'wrapper_sha256', 'deviation_sha256',
                          'operational_binding_sha256'}),
    'injection': frozenset({'schema', 'prediction_freeze_sha256', 'exposure_inventory_sha256',
                            'heldout_commitment', 'custody_source_sha256', 'operational_binding_sha256'}),
}


class CustodyError(ValueError):
    """An incomplete preparation; never use it to justify another draw."""


@dataclass(frozen=True)
class PublicInputs:
    freeze_bytes: bytes = field(repr=False)
    freeze_sha256: str
    runtime_bytes: bytes = field(repr=False)
    prediction_bytes: bytes = field(repr=False)
    plan_bytes: bytes = field(repr=False)
    plan_sha256: str
    site_commitment: str
    inventory_bytes: bytes = field(repr=False)
    inventory_sha256: str
    operational_binding_bytes: bytes = field(repr=False)
    operational_binding_sha256: str


@dataclass(frozen=True)
class PreparedSplit:
    """A private phase path and public digest; not evidence of publication."""
    phase_root: Path = field(repr=False)
    commitment_sha256: str


@dataclass(frozen=True)
class CommittedSplit:
    """Private record bytes and the exact public commitment for one track."""
    track: str
    freeze_bytes: bytes = field(repr=False)
    freeze_sha256: str
    nonce: bytes = field(repr=False)
    record_bytes: bytes = field(repr=False)
    commitment_bytes: bytes = field(repr=False)
    inventory_bytes: bytes = field(repr=False)
    inventory_sha256: str
    site_nonce: bytes | None = field(default=None, repr=False)
    site_record_bytes: bytes | None = field(default=None, repr=False)
    site_commitment: str | None = None


def _sha(data):
    if type(data) is not bytes or not 0 < len(data) <= _MAX_PUBLIC_BYTES:
        raise CustodyError('bounded immutable bytes are required')
    return hashlib.sha256(data).hexdigest()


def _hex(value):
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise CustodyError('a lowercase SHA-256 digest is required')
    return value


def _canonical(value):
    return commitments.canonical_json(value)


def _parse_canonical(raw):
    _sha(raw)
    try:
        value = json.loads(raw)
        if _canonical(value) != raw:
            raise CustodyError('record is not canonical')
    except (TypeError, UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise CustodyError('invalid canonical record') from error
    return value


def _read_source_hash():
    source = Path(__file__)
    with source.open('rb') as stream:
        data = stream.read(_MAX_PUBLIC_BYTES + 1)
    return _sha(data)


def _check_inputs(public, site_nonce, site_record_bytes, publication_check, source_check, prediction_check):
    if (type(public) is not PublicInputs or not callable(publication_check)
            or not callable(source_check) or not callable(prediction_check)):
        raise CustodyError('trusted publication inputs are required')
    try:
        freeze = commitments.load_prediction_freeze(public.freeze_bytes, _hex(public.freeze_sha256))
        freeze_record = _parse_canonical(freeze.record_bytes)
        if set(freeze_record['tracks']) != {'injection'}:
            raise CustodyError('an injection-only freeze is required')
        track = freeze_record['tracks']['injection']
        snapshot = commitments.load_inventory_snapshot(public.inventory_bytes, _hex(public.inventory_sha256))
        site_commitment = _hex(public.site_commitment)
        commitments.verify_site_reveal(site_nonce, site_record_bytes, site_commitment,
                                       expected_source_commit=track['source_commit'])
        plan = _parse_canonical(public.plan_bytes)
        if (_sha(public.runtime_bytes) != track['runtime_sha256']
                or _sha(public.prediction_bytes) != track['prediction_sha256']
                or _sha(public.plan_bytes) != _hex(public.plan_sha256)
                or type(plan) is not dict
                or plan.get('schema') != 'mlg-number-injection-family-request-plan-v2'
                or plan.get('protocol_sha256') != commitments.PROTOCOL_SHA256
                or plan.get('checkpoint_revision') != commitments.CHECKPOINT_REVISION
                or plan.get('source_commit') != track['source_commit']):
            raise CustodyError('published discovery inputs differ from the freeze')
        entropy = bytes.fromhex(_hex(plan.get('public_entropy')))
        if hmac.compare_digest(entropy, site_nonce):
            raise CustodyError('private site nonce matches public entropy')
        binding = _parse_canonical(public.operational_binding_bytes)
        schema = binding.get('schema') if type(binding) is dict else None
        keys = (_OP_KEYS if schema == 'mlg-injection-heldout-operational-binding-v1'
                else _OP_V2_KEYS if schema == 'mlg-injection-heldout-operational-binding-v2' else None)
        if (keys is None or set(binding) != keys
                or _sha(public.operational_binding_bytes) != _hex(public.operational_binding_sha256)):
            raise CustodyError('operational binding is invalid')
        fixed = {'schema': schema,
                 'protocol_sha256': commitments.PROTOCOL_SHA256,
                 'checkpoint_revision': commitments.CHECKPOINT_REVISION,
                 'discovery_source_commit': track['source_commit'],
                 'prediction_freeze_sha256': freeze.sha256,
                 'runtime_sha256': track['runtime_sha256'],
                 'prediction_sha256': track['prediction_sha256'],
                 'plan_sha256': public.plan_sha256, 'site_commitment': site_commitment,
                 'inventory_sha256': snapshot.sha256, 'custody_source_sha256': _read_source_hash()}
        if any(binding.get(key) != value for key, value in fixed.items()):
            raise CustodyError('operational binding differs from frozen inputs or custody source')
        for key in ('executor_source_sha256', 'comparator_source_sha256',
                    'ledger_source_sha256', 'tokenizer_sha256', 'supplement_sha256',
                    'owner_source_sha256', 'journal_source_sha256', 'supervisor_source_sha256',
                    'auditor_source_sha256', 'gate_source_sha256', 'inventory_source_sha256',
                    'accounting_amendment_sha256', 'lease_source_sha256'):
            _hex(binding[key])
        if schema == 'mlg-injection-heldout-operational-binding-v2':
            for key in ('preparation_child_source_sha256', 'preparation_audit_source_sha256',
                        'outer_observer_source_sha256', 'gate_allocation_caller_source_sha256',
                        'gate_transition_caller_source_sha256'):
                _hex(binding[key])
        if type(binding['inventory_source_commit']) is not str or not _HEX40.fullmatch(binding['inventory_source_commit']):
            raise CustodyError('inventory source revision is invalid')
        published = {'freeze': public.freeze_bytes, 'runtime': public.runtime_bytes,
                     'prediction': public.prediction_bytes, 'plan': public.plan_bytes,
                     'site_commitment': site_commitment.encode('ascii') + b'\n',
                     'inventory': public.inventory_bytes,
                     'operational_binding': public.operational_binding_bytes}
        for name, data in published.items():
            if publication_check(name, data) is not True:
                raise CustodyError('published input readback did not match')
        if source_check(binding) is not True:
            raise CustodyError('operational source verification did not pass')
        if prediction_check(public.prediction_bytes, track['prediction_sha256']) is not True:
            raise CustodyError('frozen prediction validation did not pass')
        return freeze, snapshot, entropy
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        if isinstance(error, CustodyError):
            raise
        raise CustodyError('frozen discovery or site commitment is invalid') from error


def _valid_tokens(tokens):
    return (type(tokens) is list and bool(tokens)
            and all(type(token) is int and 0 <= token <= 50256 for token in tokens))


def tokenize_assignment(allocation, tokenizer):
    """Check the complete prefix, one-token subject, context and continuations."""
    prefix = tokenizer.encode('The', add_special_tokens=False)
    if not _valid_tokens(prefix):
        raise CustodyError('subject prefix is not tokenizable')
    result = []
    for family in allocation.families:
        suffix = tokenizer.encode(' ' + family.context, add_special_tokens=False)
        if not _valid_tokens(suffix) or len(family.subjects) != 4 or len(family.texts) != 4:
            raise CustodyError('family context is not tokenizable')
        members = []
        for subject, text in zip(family.subjects, family.texts):
            subject_prefix = tokenizer.encode('The ' + subject, add_special_tokens=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if (not _valid_tokens(subject_prefix) or len(subject_prefix) != len(prefix) + 1
                    or subject_prefix[:-1] != prefix or not _valid_tokens(ids)
                    or ids != subject_prefix + suffix or len(ids) > 32):
                raise CustodyError('subject or context token alignment failed')
            for continuation, target in _TARGETS.items():
                if tokenizer.encode(text + continuation, add_special_tokens=False) != ids + [target]:
                    raise CustodyError('continuation token alignment failed')
            members.append({'text': text, 'input_ids': list(ids),
                            'attention_mask': [1] * len(ids), 'subject_index': len(prefix)})
        result.append(members)
    if len(result) != 4 or any(len(family) != 4 for family in result):
        raise CustodyError('the allocation is incomplete')
    return result


def reject_known_row_prefixes(cases, inventory_bytes):
    """Reject either whole-row prefix direction against validated known rows."""
    try:
        known = [tuple(row['token_ids']) for row in json.loads(inventory_bytes)['records']]
        candidates = [tuple(item['input_ids']) for family in cases for item in family]
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        raise CustodyError('known row comparison is invalid') from error
    if len(candidates) != 16 or not known:
        raise CustodyError('known row comparison is incomplete')
    for candidate in candidates:
        for row in known:
            shared = min(len(candidate), len(row))
            if candidate[:shared] == row[:shared]:
                raise CustodyError('known whole-row prefix collision')


def _verify_committed_split(split, expected_track, publication_check):
    if type(split) is not CommittedSplit or split.track != expected_track:
        raise CustodyError('the committed track is invalid')
    try:
        freeze = commitments.load_prediction_freeze(split.freeze_bytes, _hex(split.freeze_sha256))
        freeze_record = _parse_canonical(freeze.record_bytes)
        if set(freeze_record['tracks']) != {expected_track}:
            raise CustodyError('a separate track freeze is required')
        snapshot = commitments.load_inventory_snapshot(split.inventory_bytes, _hex(split.inventory_sha256))
        public = _parse_canonical(split.commitment_bytes)
        if (type(public) is not dict or set(public) != _COMMIT_KEYS[expected_track]
                or public.get('schema') != 'mlg-' + expected_track + '-heldout-public-commitment-v1'
                or public.get('prediction_freeze_sha256') != freeze.sha256
                or public.get('exposure_inventory_sha256') != snapshot.sha256):
            raise CustodyError('the public commitment does not bind the frozen split')
        for key in _COMMIT_KEYS[expected_track] - {'schema'}:
            _hex(public[key])
        heldout_commitment = _hex(public.get('heldout_commitment'))
        if (publication_check(expected_track + '_freeze', split.freeze_bytes) is not True
                or publication_check(expected_track + '_commitment', split.commitment_bytes) is not True):
            raise CustodyError('the committed split was not observed publicly')
        commitments.verify_heldout_reveal(freeze, split.nonce, split.record_bytes, heldout_commitment,
            inventory_snapshot=snapshot, site_nonce=split.site_nonce,
            site_record_bytes=split.site_record_bytes, site_commitment=split.site_commitment)
        record = _parse_canonical(split.record_bytes)
        rows = [tuple(case['input_ids']) for family in record['families'] for case in family['cases']]
        if len(rows) != 16:
            raise CustodyError('the committed split does not contain sixteen cases')
        return freeze.sha256, snapshot.sha256, heldout_commitment, _sha(split.commitment_bytes), rows
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        if isinstance(error, CustodyError):
            raise
        raise CustodyError('the committed split could not be verified') from error


def compare_committed_splits(natural, injection, *, comparator_source_sha256, publication_check):
    """Return only a commitment-bound clear/collision receipt for both splits.

    The caller must independently establish publication and protect both
    records and the injection site tuple. A collision stops the affected later
    fixed evaluation without invalidating a previously completed independent
    track; it is not permission to redraw.
    """
    if not callable(publication_check) or _hex(comparator_source_sha256) != _read_source_hash():
        raise CustodyError('the published comparator source does not match')
    nfreeze, ninventory, ncommit, nfile, natural_rows = _verify_committed_split(
        natural, 'natural', publication_check)
    ifreeze, iinventory, icommit, ifile, injection_rows = _verify_committed_split(
        injection, 'injection', publication_check)
    collision = any(a[:min(len(a), len(b))] == b[:min(len(a), len(b))]
                    for a in natural_rows for b in injection_rows)
    return _canonical({'schema': 'mlg-number-disjoint-split-check-v1',
        'natural_prediction_freeze_sha256': nfreeze, 'natural_heldout_commitment': ncommit,
        'natural_exposure_inventory_sha256': ninventory, 'natural_commitment_file_sha256': nfile,
        'injection_prediction_freeze_sha256': ifreeze, 'injection_heldout_commitment': icommit,
        'injection_exposure_inventory_sha256': iinventory, 'injection_commitment_file_sha256': ifile,
        'comparator_source_sha256': comparator_source_sha256, 'natural_case_count': 16,
        'injection_case_count': 16, 'status': 'collision' if collision else 'clear'})


def verify_clear_receipt(raw, *, natural_freeze_sha256, injection_freeze_sha256,
                         natural_inventory_sha256, injection_inventory_sha256,
                         natural_commitment_bytes, injection_commitment_bytes,
                         comparator_source_sha256):
    """Check a clear receipt against both exact public commitment files.

    The caller must also observe the exact receipt bytes and published source;
    the receipt is a trusted custodian report, not an independent attestation.
    """
    try:
        source = _hex(comparator_source_sha256)
        if source != _read_source_hash():
            raise CustodyError('comparator source does not match')
        natural = _parse_canonical(natural_commitment_bytes)
        injection = _parse_canonical(injection_commitment_bytes)
        if (type(natural) is not dict or set(natural) != _COMMIT_KEYS['natural']
                or type(injection) is not dict or set(injection) != _COMMIT_KEYS['injection']
                or natural.get('schema') != 'mlg-natural-heldout-public-commitment-v1'
                or injection.get('schema') != 'mlg-injection-heldout-public-commitment-v1'
                or natural.get('prediction_freeze_sha256') != _hex(natural_freeze_sha256)
                or injection.get('prediction_freeze_sha256') != _hex(injection_freeze_sha256)
                or natural.get('exposure_inventory_sha256') != _hex(natural_inventory_sha256)
                or injection.get('exposure_inventory_sha256') != _hex(injection_inventory_sha256)):
            raise CustodyError('public split commitments do not match expected identities')
        for public, track in ((natural, 'natural'), (injection, 'injection')):
            for key in _COMMIT_KEYS[track] - {'schema'}:
                _hex(public[key])
        expected = {'schema': 'mlg-number-disjoint-split-check-v1',
            'natural_prediction_freeze_sha256': natural_freeze_sha256,
            'natural_heldout_commitment': _hex(natural.get('heldout_commitment')),
            'natural_exposure_inventory_sha256': natural_inventory_sha256,
            'natural_commitment_file_sha256': _sha(natural_commitment_bytes),
            'injection_prediction_freeze_sha256': injection_freeze_sha256,
            'injection_heldout_commitment': _hex(injection.get('heldout_commitment')),
            'injection_exposure_inventory_sha256': injection_inventory_sha256,
            'injection_commitment_file_sha256': _sha(injection_commitment_bytes),
            'comparator_source_sha256': source, 'natural_case_count': 16,
            'injection_case_count': 16, 'status': 'clear'}
        receipt = _parse_canonical(raw)
        if (type(receipt) is not dict or type(receipt.get('natural_case_count')) is not int
                or type(receipt.get('injection_case_count')) is not int or receipt != expected):
            raise CustodyError('a matching clear split receipt is required')
        return True
    except (ValueError, TypeError, AttributeError) as error:
        if isinstance(error, CustodyError):
            raise
        raise CustodyError('clear split receipt is invalid') from None


def _write_new(path, data):
    """Exclusively write and flush a small retained private record."""
    stream = path.open('xb')
    failure = None
    try:
        if stream.write(data) != len(data):
            raise OSError('private record write was incomplete')
        stream.flush()
        os.fsync(stream.fileno())
    except BaseException as error:
        failure = error
    try:
        stream.close()
    except BaseException as error:
        if failure is None:
            raise
        cancellation = (KeyboardInterrupt, SystemExit)
        if isinstance(error, cancellation) and not isinstance(failure, cancellation):
            raise error from failure
        raise failure from error
    if failure is not None:
        raise failure


def prepare_once(public, private_parent, ledger_path, *, site_nonce, site_record_bytes,
                 tokenizer, claim_phase, publication_check, observe_inventory,
                 source_check, prediction_check, gate_allocation):
    """Claim the injection phase, draw one OS nonce and seal a private split.

    Readback, currentness, tokenizer asset/source verification, private ACLs and
    source-first startup are trusted caller duties. A claim or failed draw is
    consumed; this function never retries or replaces a sampled case.
    """
    freeze, snapshot, plan_entropy = _check_inputs(public, site_nonce, site_record_bytes,
                                                    publication_check, source_check, prediction_check)
    if not callable(claim_phase) or not callable(observe_inventory) or not callable(gate_allocation):
        raise CustodyError('trusted ledger and exposure observers are required')
    if observe_inventory() != snapshot.sha256:
        raise CustodyError('selected exposure inventory is not current')
    parent = Path(private_parent)
    if not parent.is_absolute() or not parent.is_dir() or parent.is_symlink():
        raise CustodyError('an existing private absolute parent is required')
    phase = parent / ('injection-heldout-' + freeze.sha256)
    if os.path.lexists(phase):
        raise CustodyError('the one-use phase path already exists')
    grant = claim_phase(ledger_path, 'injection_heldout', output_path=phase)
    if (getattr(grant, 'phase', None) != 'injection_heldout'
            or Path(getattr(grant, 'output_path', '')) != phase
            or type(getattr(grant, 'logical_limit_bytes', None)) is not int
            or grant.logical_limit_bytes != _LOGICAL_LIMIT
            or type(getattr(grant, 'disk_limit_bytes', None)) is not int
            or grant.disk_limit_bytes != _DISK_LIMIT):
        raise CustodyError('phase grant does not match the approved reservation')
    phase.mkdir()
    try:
        _write_new(phase / 'claim.json', _canonical({
            'schema': 'mlg-injection-heldout-claim-v1', 'prediction_freeze_sha256': freeze.sha256,
            'exposure_inventory_sha256': snapshot.sha256,
            'operational_binding_sha256': public.operational_binding_sha256,
            'logical_limit_bytes': _LOGICAL_LIMIT, 'disk_limit_bytes': _DISK_LIMIT}))
        if observe_inventory() != snapshot.sha256:
            raise CustodyError('selected exposure inventory changed after the claim')
        if _read_source_hash() != json.loads(public.operational_binding_bytes)['custody_source_sha256']:
            raise CustodyError('custody source changed after the claim')
        if gate_allocation(freeze.sha256) is not True:
            raise CustodyError('shared allocation announcement did not complete')
        if (observe_inventory() != snapshot.sha256
                or _read_source_hash() != json.loads(public.operational_binding_bytes)['custody_source_sha256']):
            raise CustodyError('bound source or exposure inventory changed before the draw')
        nonce = secrets.token_bytes(32)
        if type(nonce) is not bytes or len(nonce) != 32:
            raise CustodyError('OS nonce generation failed')
        _write_new(phase / 'nonce.bin', nonce)
        if hmac.compare_digest(nonce, site_nonce) or hmac.compare_digest(nonce, plan_entropy):
            raise CustodyError('the private split nonce is not independent of prior bytes')
        allocation = commitments.sample_families(freeze, nonce, inventory_snapshot=snapshot,
            current_inventory_sha256=snapshot.sha256, site_nonce=site_nonce,
            site_record_bytes=site_record_bytes, site_commitment=public.site_commitment)
        cases = tokenize_assignment(allocation, tokenizer)
        reject_known_row_prefixes(cases, public.inventory_bytes)
        if (observe_inventory() != snapshot.sha256
                or _read_source_hash() != json.loads(public.operational_binding_bytes)['custody_source_sha256']):
            raise CustodyError('bound source or exposure inventory changed during preparation')
        sealed = commitments.seal_heldout(freeze, nonce, cases, allocation=allocation,
            inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256,
            site_nonce=site_nonce, site_record_bytes=site_record_bytes,
            site_commitment=public.site_commitment)
        _write_new(phase / 'heldout-record.json', sealed.record_bytes)
        _write_new(phase / 'inventory.json', public.inventory_bytes)
        commitment = {'schema': 'mlg-injection-heldout-public-commitment-v1',
                      'prediction_freeze_sha256': freeze.sha256,
                      'exposure_inventory_sha256': snapshot.sha256,
                      'heldout_commitment': sealed.commitment_sha256,
                      'custody_source_sha256': _read_source_hash(),
                      'operational_binding_sha256': public.operational_binding_sha256}
        commitment_bytes = _canonical(commitment)
        _write_new(phase / 'commitment.json', commitment_bytes)
        _write_new(phase / 'preparation-complete.json', _canonical({
            'schema': 'mlg-injection-heldout-preparation-complete-v1',
            'commitment_file_sha256': _sha(commitment_bytes),
            'operational_binding_sha256': public.operational_binding_sha256}))
        return PreparedSplit(phase, sealed.commitment_sha256)
    except BaseException as error:
        try:
            _write_new(phase / 'preparation-failure.json', _canonical({
                'status': 'incomplete', 'error_type': type(error).__name__}))
        except BaseException as cleanup_error:
            cancellation = (KeyboardInterrupt, SystemExit)
            if isinstance(error, cancellation):
                raise error from cleanup_error
            if isinstance(cleanup_error, cancellation):
                raise cleanup_error from error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise CustodyError('held-out preparation is incomplete') from None


def prepare_study_once(public, private_parent, ledger_path, **kwargs):
    """Require this study's immutable discovery identity before any claim."""
    if type(public) is not PublicInputs:
        raise CustodyError('fixed study inputs are required')
    try:
        freeze = _parse_canonical(public.freeze_bytes)
        operation = _parse_canonical(public.operational_binding_bytes)
        track = freeze['tracks']['injection']
        if (type(operation) is not dict or set(operation) != _OP_V2_KEYS
                or operation.get('schema') != 'mlg-injection-heldout-operational-binding-v2'
                or public.freeze_sha256 != _STUDY_FREEZE or _sha(public.freeze_bytes) != _STUDY_FREEZE
                or type(freeze['tracks']) is not dict or set(freeze['tracks']) != {'injection'}
                or track.get('source_commit') != _STUDY_SOURCE
                or track.get('runtime_sha256') != _STUDY_RUNTIME or _sha(public.runtime_bytes) != _STUDY_RUNTIME
                or track.get('prediction_sha256') != _STUDY_PREDICTION
                or _sha(public.prediction_bytes) != _STUDY_PREDICTION
                or public.plan_sha256 != _STUDY_PLAN or _sha(public.plan_bytes) != _STUDY_PLAN
                or public.site_commitment != _STUDY_SITE_COMMITMENT):
            raise CustodyError('fixed discovery identity does not match')
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        if isinstance(error, CustodyError):
            raise
        raise CustodyError('fixed discovery identity is invalid') from None
    return prepare_once(public, private_parent, ledger_path, **kwargs)
