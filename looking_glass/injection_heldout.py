"""Private adapter for the frozen injection science on committed held-out rows.

This module does not draw a split, publish a commitment or attest execution.
All inputs and artifacts remain private until the protocol permits reveal.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import current_thread
from types import FunctionType
from dataclasses import dataclass

from . import pretrained_number_archive as _archive
from . import pretrained_number_commitments as _commitments
from . import pretrained_number_journal as _journal
from . import pretrained_number_requests as _requests
from . import pretrained_number_selection as _selection
from . import pretrained_number_witness as _witness

INJECTION_SOURCE_COMMIT = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
INJECTION_RUNTIME_SHA256 = 'b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e'
INJECTION_PREDICTION_SHA256 = '11376d8aae517db76d6a33c1ae5dc1f4b0c747a9d5216c9b02a17b1b0b300107'
INJECTION_FREEZE_SHA256 = 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'
INJECTION_PLAN_SHA256 = 'e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8'
INJECTION_SITE_COMMITMENT = 'c3739cb173d3c1bc3e3adda70025ff7f26a73c4690e50ecf5baf06ff8b359b9d'


@dataclass(frozen=True, repr=False)
class HeldoutTarget:
    text: str
    input_ids: tuple


@dataclass(frozen=True, repr=False)
class PreparedInjection:
    """Private verified inputs; does not attest publication, custody or execution."""
    targets: tuple
    site: int
    heldout_commitment: str
    inventory_sha256: str
    prediction_bytes: bytes
    heldout_record_bytes: bytes
    split_nonce: bytes
    site_nonce: bytes
    site_record_bytes: bytes


def _tokenize_all(allocation, tokenizer):
    """Retokenize every committed row and its four contrast continuations."""
    def encode(text):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception as error:
            raise ValueError('held-out tokenizer failed') from error
        if (type(ids) is not list or not ids
                or any(type(token) is not int or not 0 <= token <= 50256 for token in ids)):
            raise ValueError('held-out tokenizer returned invalid token IDs')
        return ids

    prefix = encode('The')
    output = []
    for family in allocation.families:
        suffix = encode(' ' + family.context)
        members = []
        for subject, text in zip(family.subjects, family.texts):
            subject_prefix, ids = encode('The ' + subject), encode(text)
            if (len(subject_prefix) != len(prefix) + 1 or subject_prefix[:-1] != prefix
                    or ids != subject_prefix + suffix or len(ids) > 32):
                raise ValueError('held-out subject or context alignment failed')
            for continuation, target in ((' is', 318), (' are', 389), (' was', 373), (' were', 547)):
                if encode(text + continuation) != ids + [target]:
                    raise ValueError('held-out continuation alignment failed')
            members.append({'text': text, 'input_ids': list(ids), 'attention_mask': [1] * len(ids),
                            'subject_index': len(prefix)})
        if len(members) != 4:
            raise ValueError('held-out family is incomplete')
        output.append(members)
    if len(output) != 4:
        raise ValueError('held-out allocation is incomplete')
    return output


def verify_frozen_inputs(*, prediction_bytes, runtime_bytes, freeze_bytes, plan_bytes,
                         inventory_bytes, inventory_sha256, split_nonce, heldout_record_bytes,
                         heldout_commitment, site_nonce, site_record_bytes, site_commitment, tokenizer):
    """Verify the fixed science and private commitment, without a model forward.

    The caller must also establish public chronology, current inventory, source
    and runtime execution, ledger claim, and the serialized cross-track gate.
    This function does not establish any of those external facts.
    """
    for raw, expected in ((prediction_bytes, INJECTION_PREDICTION_SHA256),
                          (runtime_bytes, INJECTION_RUNTIME_SHA256),
                          (freeze_bytes, INJECTION_FREEZE_SHA256), (plan_bytes, INJECTION_PLAN_SHA256)):
        if type(raw) is not bytes or not 0 < len(raw) <= 2 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('frozen public input differs')
    if site_commitment != INJECTION_SITE_COMMITMENT:
        raise ValueError('published site commitment differs')
    freeze = _commitments.load_prediction_freeze(freeze_bytes, INJECTION_FREEZE_SHA256)
    aggregate = json.loads(freeze.record_bytes)
    if aggregate['tracks'] != {'injection': {'source_commit': INJECTION_SOURCE_COMMIT,
                                            'runtime_sha256': INJECTION_RUNTIME_SHA256,
                                            'prediction_sha256': INJECTION_PREDICTION_SHA256}}:
        raise ValueError('injection prediction freeze differs')
    _selection._load_prediction(prediction_bytes, INJECTION_PREDICTION_SHA256, 'cpu')
    plan = _requests.load_family_request_plan(plan_bytes, expected_source_commit=INJECTION_SOURCE_COMMIT,
                                              expected_sha256=INJECTION_PLAN_SHA256)
    entropy = bytes.fromhex(json.loads(plan.record_bytes)['public_entropy'])
    if (type(split_nonce) is not bytes or len(split_nonce) != 32 or type(site_nonce) is not bytes
            or len(site_nonce) != 32 or len({entropy, split_nonce, site_nonce}) != 3):
        raise ValueError('study nonces and request entropy must be distinct')
    snapshot = _commitments.load_inventory_snapshot(inventory_bytes, inventory_sha256)
    site = _commitments.verify_site_reveal(site_nonce, site_record_bytes, site_commitment,
                                           expected_source_commit=INJECTION_SOURCE_COMMIT)
    options = dict(site_nonce=site_nonce, site_record_bytes=site_record_bytes, site_commitment=site_commitment)
    _commitments.verify_heldout_reveal(freeze, split_nonce, heldout_record_bytes, heldout_commitment,
                                       inventory_snapshot=snapshot, **options)
    allocation = _commitments.sample_families(freeze, split_nonce, inventory_snapshot=snapshot,
                                               current_inventory_sha256=snapshot.sha256, **options)
    tokenized = _tokenize_all(allocation, tokenizer)
    known = tuple(tuple(row['token_ids']) for row in json.loads(snapshot.record_bytes)['records'])
    rows = tuple(tuple(case['input_ids']) for family in tokenized for case in family)
    if any(candidate[:min(len(candidate), len(row))] == row[:min(len(candidate), len(row))]
           for candidate in rows for row in known):
        raise ValueError('held-out complete-row prefix collision')
    checked = _commitments.seal_heldout(freeze, split_nonce, tokenized, allocation=allocation,
                                         inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256,
                                         **options)
    if checked.record_bytes != heldout_record_bytes or checked.commitment_sha256 != heldout_commitment:
        raise ValueError('retokenized held-out record differs')
    return PreparedInjection(committed_targets(heldout_record_bytes), site, heldout_commitment,
                             snapshot.sha256, prediction_bytes, heldout_record_bytes, split_nonce,
                             site_nonce, site_record_bytes)


@dataclass(frozen=True, repr=False)
class HeldoutBinding:
    record_bytes: bytes
    sha256: str
    heldout_commitment: str


def make_heldout_binding(*, source_commit, runtime_sha256, prediction_sha256, freeze_sha256,
                         heldout_commitment, site_commitment, inventory_sha256,
                         operation_sha256, gate_sha256, disjoint_status, disjoint_receipt_sha256):
    """Bind exact public identities; this is not proof of custody or publication."""
    digests = (runtime_sha256, prediction_sha256, freeze_sha256, heldout_commitment, site_commitment,
               inventory_sha256, operation_sha256, gate_sha256)
    if (type(source_commit) is not str or re.fullmatch(r'[0-9a-f]{40}', source_commit) is None
            or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None for value in digests)
            or type(disjoint_status) is not str or disjoint_status not in ('pending', 'clear')
            or (disjoint_status == 'pending' and disjoint_receipt_sha256 is not None)
            or (disjoint_status == 'clear' and (type(disjoint_receipt_sha256) is not str
                or re.fullmatch(r'[0-9a-f]{64}', disjoint_receipt_sha256) is None))):
        raise ValueError('held-out binding identities are invalid')
    request_ids = [_request_id(heldout_commitment, case, role)
                   for case, role in _requests.family_request_schedule()]
    if len(request_ids) != 128 or len(set(request_ids)) != 128:
        raise ValueError('held-out request plan has a collision')
    record = {'schema': 'mlg-injection-heldout-journal-binding-v1', 'source_commit': source_commit,
              'runtime_sha256': runtime_sha256, 'prediction_sha256': prediction_sha256,
              'freeze_sha256': freeze_sha256, 'heldout_commitment': heldout_commitment,
              'site_commitment': site_commitment, 'inventory_sha256': inventory_sha256,
              'operation_sha256': operation_sha256, 'gate_sha256': gate_sha256,
              'disjoint_status': disjoint_status, 'disjoint_receipt_sha256': disjoint_receipt_sha256}
    raw = json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('ascii')
    return HeldoutBinding(raw, hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + raw).hexdigest(),
                          heldout_commitment)


def _check_binding(binding):
    if type(binding) is not HeldoutBinding:
        raise ValueError('a validated held-out binding is required')
    try:
        record = json.loads(binding.record_bytes)
        if record.pop('schema') != 'mlg-injection-heldout-journal-binding-v1':
            raise ValueError('held-out binding schema differs')
        expected = make_heldout_binding(**record)
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        raise ValueError('held-out binding is malformed') from error
    if binding != expected:
        raise ValueError('held-out binding bytes differ')
    return binding


def _check_prepared(prepared):
    if (type(prepared) is not PreparedInjection or type(prepared.prediction_bytes) is not bytes
            or hashlib.sha256(prepared.prediction_bytes).hexdigest() != INJECTION_PREDICTION_SHA256
            or type(prepared.heldout_record_bytes) is not bytes or len(prepared.heldout_record_bytes) > 1_000_000
            or type(prepared.split_nonce) is not bytes or len(prepared.split_nonce) != 32
            or type(prepared.site_nonce) is not bytes or len(prepared.site_nonce) != 32
            or prepared.split_nonce == prepared.site_nonce or type(prepared.site) is not int
            or type(prepared.heldout_commitment) is not str
            or re.fullmatch(r'[0-9a-f]{64}', prepared.heldout_commitment) is None):
        raise ValueError('verified private inputs are malformed')
    record = _commitments._decode_canonical(prepared.heldout_record_bytes)
    commitment = hashlib.sha256(_commitments._HELDOUT_COMMIT_DOMAIN + prepared.split_nonce
                                + b'\0' + prepared.heldout_record_bytes).hexdigest()
    if (type(record) is not dict or record.get('protocol_sha256') != _commitments.PROTOCOL_SHA256
            or record.get('checkpoint_revision') != _commitments.CHECKPOINT_REVISION
            or commitment != prepared.heldout_commitment or record.get('prediction_freeze_sha256') != INJECTION_FREEZE_SHA256
            or record.get('exposure_inventory_sha256') != prepared.inventory_sha256
            or record.get('source_commits') != {'injection': INJECTION_SOURCE_COMMIT}
            or committed_targets(prepared.heldout_record_bytes) != prepared.targets
            or _commitments.verify_site_reveal(prepared.site_nonce, prepared.site_record_bytes,
                  INJECTION_SITE_COMMITMENT, expected_source_commit=INJECTION_SOURCE_COMMIT) != prepared.site):
        raise ValueError('verified private input identities differ')
    return prepared


def make_prepared_binding(prepared, *, operation_sha256, gate_sha256, disjoint_status,
                          disjoint_receipt_sha256):
    """Build journal identities from the verified immutable split and frozen science."""
    _check_prepared(prepared)
    return make_heldout_binding(source_commit=INJECTION_SOURCE_COMMIT, runtime_sha256=INJECTION_RUNTIME_SHA256,
                                prediction_sha256=INJECTION_PREDICTION_SHA256, freeze_sha256=INJECTION_FREEZE_SHA256,
                                heldout_commitment=prepared.heldout_commitment, site_commitment=INJECTION_SITE_COMMITMENT,
                                inventory_sha256=prepared.inventory_sha256, operation_sha256=operation_sha256,
                                gate_sha256=gate_sha256, disjoint_status=disjoint_status,
                                disjoint_receipt_sha256=disjoint_receipt_sha256)


def check_prepared_binding(binding, prepared):
    """Reject a journal binding that names a different verified split or freeze."""
    _check_binding(binding)
    _check_prepared(prepared)
    record = json.loads(binding.record_bytes)
    expected = make_prepared_binding(prepared, operation_sha256=record['operation_sha256'],
                                     gate_sha256=record['gate_sha256'], disjoint_status=record['disjoint_status'],
                                     disjoint_receipt_sha256=record['disjoint_receipt_sha256'])
    if binding != expected:
        raise ValueError('journal binding does not match verified private inputs')
    return binding


def _request_id(commitment, case_index, role):
    if (type(case_index) is not int or not 0 <= case_index < 8
            or type(role) is not str or role not in _archive._ROLES):
        raise ValueError('held-out request coordinates are invalid')
    ordinal = case_index * len(_archive._ROLES) + _archive._ROLES.index(role)
    return hashlib.sha256(b'MLG-injection-heldout-request-v1\0' + bytes.fromhex(commitment)
                          + ordinal.to_bytes(4, 'big')).digest()[:16].hex()


def heldout_request_id(binding, case_index, role):
    """Derive a site-independent request identity from the committed split."""
    _check_binding(binding)
    return _request_id(binding.heldout_commitment, case_index, role)


def heldout_request_plan(binding):
    """Return the complete collision-checked schedule before model work."""
    _check_binding(binding)
    return tuple((case, role, _request_id(binding.heldout_commitment, case, role))
                 for case, role in _requests.family_request_schedule())


def _clone(function, namespace):
    """Rebind an unchanged frozen function's globals within a private namespace."""
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class HeldoutCodec:
    """Use the frozen archive and witness checks for one immutable case snapshot."""
    __slots__ = ('_archive', '_witness')

    def __init__(self, rows):
        indexed = []
        for family in range(4):
            indexed.extend(((None, rows[2 * family]), (None, rows[2 * family + 1]),
                            (None, ()), (None, ())))
        archive = dict(vars(_archive))
        archive['PUBLIC_DISCOVERY'] = tuple(indexed)
        for name in ('_case_ids', '_check_observation', 'serialize_discovery_observation', '_read'):
            archive[name] = _clone(getattr(_archive, name), archive)
        witness = dict(vars(_witness))
        witness['_case_ids'] = archive['_case_ids']
        witness['_check_observation'] = archive['_check_observation']
        for name in ('_record_shape', 'encode_private_forward', 'decode_private_forward',
                     'audit_private_forward', 'audit_private_record'):
            witness[name] = _clone(getattr(_witness, name), witness)
        self._archive, self._witness = archive, witness

    def serialize_public(self, index, role, observation):
        return self._archive['serialize_discovery_observation'](index, role, observation, execution_device='cpu')

    def serialize_private(self, index, role, observation, evidence):
        return self._witness['encode_private_forward'](index, role, observation, evidence)

    def read_public(self, raw):
        return self._archive['_read'](raw, 'cpu')

    def audit_pair(self, public, private, *, site, clean=None, injected=None):
        index, role, observation = self._archive['_read'](public, 'cpu')
        record = self._witness['decode_private_forward'](private)
        if (record.case_index, record.role) != (index, role):
            raise ValueError('private and public held-out coordinates differ')
        return self._witness['audit_private_record'](private, observation, site=site,
                                                      clean=clean, injected=injected)


def bind_heldout_codec(rows):
    """Create an isolated codec for eight already-committed immutable rows."""
    if (type(rows) is not tuple or len(rows) != 8
            or any(type(row) is not tuple or not 1 <= len(row) <= 32
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)
            or len(set(rows)) != 8):
        raise ValueError('eight distinct immutable held-out token rows are required')
    return HeldoutCodec(rows)


def committed_targets(record_bytes):
    """Select each family's first lemma in its committed singular/plural order.

    The caller must separately verify the commitment, allocation, tokenizer and
    current exposure. This function does not establish those custody facts.
    """
    from .pretrained_number_commitments import _decode_canonical
    record = _decode_canonical(record_bytes)
    if (type(record) is not dict or set(record) != {'schema', 'protocol_sha256', 'source_commits',
                                                     'checkpoint_revision', 'prediction_freeze_sha256',
                                                     'exposure_inventory_sha256', 'families'}
            or record['schema'] != 'mlg-number-heldout-v1' or type(record['families']) is not list
            or len(record['families']) != 4):
        raise ValueError('held-out family record is malformed')
    all_rows, selected = [], []
    for family_index, family in enumerate(record['families']):
        if (type(family) is not dict or set(family) != {'family_index', 'context', 'subjects', 'cases'}
                or type(family['family_index']) is not int or family['family_index'] != family_index
                or type(family['context']) is not str or not family['context']
                or type(family['subjects']) is not list or len(family['subjects']) != 4
                or any(type(item) is not str or not item for item in family['subjects'])
                or type(family['cases']) is not list or len(family['cases']) != 4):
            raise ValueError('held-out family identity is invalid')
        reference = None
        for item_index, case in enumerate(family['cases']):
            if (type(case) is not dict or set(case) != {'text', 'input_ids', 'attention_mask', 'subject_index'}
                    or type(case['text']) is not str or not 0 < len(case['text']) <= 4096
                    or type(case['input_ids']) is not list or not 3 <= len(case['input_ids']) <= 32
                    or any(type(token) is not int or not 0 <= token < 50257 for token in case['input_ids'])
                    or type(case['attention_mask']) is not list or len(case['attention_mask']) != len(case['input_ids'])
                    or any(type(value) is not int or value != 1 for value in case['attention_mask'])
                    or type(case['subject_index']) is not int
                    or not 0 < case['subject_index'] < len(case['input_ids']) - 1):
                raise ValueError('held-out token metadata is invalid')
            ids, position = tuple(case['input_ids']), case['subject_index']
            alignment = (len(ids), position, ids[:position] + ids[position + 1:])
            if reference is None:
                reference = alignment
            elif alignment != reference:
                raise ValueError('held-out family token alignment differs')
            if any(ids[:len(other)] == other or other[:len(ids)] == ids for other in all_rows):
                raise ValueError('held-out rows duplicate or prefix one another')
            all_rows.append(ids)
            if item_index < 2:
                selected.append(HeldoutTarget(case['text'], ids))
    return tuple(selected)


def _payload_upper_bound(codec, public, private):
    """Count decoded tensor occurrences conservatively before staging.

    A failed bounded structural decode charges the entire pair, which is still
    staged when capacity permits. Semantic witness and schedule checks run
    after staging. Counting all occurrences bounds distinct tensor payloads.
    """
    try:
        index, role, observation = codec.read_public(public)
        record = codec._witness['decode_private_forward'](private)
        if ((record.case_index, record.role, record.request_id, record.input_sha256, record.condition) !=
                (index, role, observation.request_id, observation.input_sha256, observation.condition)):
            raise ValueError('pair identities differ')
        size = sum(map(len, observation.attention + observation.mlp + (observation.block, observation.logits)))
        for action in record.evidence:
            size += len(action.before) + len(action.after)
            size += len(action.donor) if action.donor is not None else 0
            size += len(action.realized_delta_f64_le) if action.realized_delta_f64_le is not None else 0
        return size
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        return len(public) + len(private)


class HeldoutJournal:
    """One-thread, fail-stop sequence of private, audited held-out pairs."""
    def __init__(self, directory, *, codec, binding, site, max_disk_bytes,
                 max_tensor_bytes=72 * 1024 * 1024):
        path = Path(directory)
        if (type(codec) is not HeldoutCodec
                or type(site) is not int or site not in range(1, 5)
                or type(max_disk_bytes) is not int or not 4096 < max_disk_bytes <= 192 * 1024 * 1024
                or type(max_tensor_bytes) is not int or not 0 < max_tensor_bytes <= 72 * 1024 * 1024
                or not path.is_absolute() or not path.parent.is_dir()):
            raise ValueError('held-out journal configuration is invalid')
        _check_binding(binding)
        path.mkdir(mode=0o700)
        self._path, self._codec, self._binding, self._site = path, codec, binding, site
        self._limit, self._used, self._ordinal, self._previous = max_disk_bytes, 4096, 0, bytes(32)
        self._tensor_limit, self._tensor_used = max_tensor_bytes, 0
        self._owner, self._failed, self._busy = current_thread(), False, False
        self._baseline, self._requests = {}, set()

    def append(self, public, private):
        if current_thread() is not self._owner or self._failed or self._busy:
            raise RuntimeError('held-out journal is unavailable')
        self._busy = True
        phase = 'preflight'
        try:
            if (self._ordinal >= 128 or type(public) is not bytes or type(private) is not bytes
                    or not 0 < len(public) <= _archive._MAX_RECORD_BYTES
                    or not 0 < len(private) <= _witness._MAX_PRIVATE_RECORD_BYTES
                    or len(public) + len(private) > 1024 * 1024):
                raise ValueError('held-out pair is outside its bounds')
            pending_decode_error = None
            try:
                tensor_size = _payload_upper_bound(self._codec, public, private)
            except Exception as error:
                # OOM and unexpected decoder failures are terminal. Keep the
                # available bounded pair if the conservative reservation fits.
                pending_decode_error = error
                tensor_size = len(public) + len(private)
            if self._tensor_used + tensor_size > self._tensor_limit:
                if pending_decode_error is not None:
                    raise pending_decode_error
                raise ValueError('held-out logical tensor allowance is exhausted')
            parts, digest = _journal._frame(self._ordinal, public, private,
                                            bytes.fromhex(self._binding.sha256), self._previous)
            ack = _journal._ACK.pack(_journal._ACK_MAGIC, self._ordinal, digest)
            size = sum(map(len, parts)) + len(ack)
            if self._used + size > self._limit:
                if pending_decode_error is not None:
                    raise pending_decode_error
                raise ValueError('held-out evidence reservation is exhausted')
            self._used += size
            self._tensor_used += tensor_size
            phase = 'stage'
            try:
                _journal._write_parts(self._path / f'{self._ordinal:03d}.frame', parts)
            except BaseException as secondary:
                if pending_decode_error is not None and not isinstance(secondary, (KeyboardInterrupt, SystemExit)):
                    raise pending_decode_error from secondary
                raise
            if pending_decode_error is not None:
                raise pending_decode_error
            phase = 'audit'
            index, role, observation = self._codec.read_public(public)
            if ((index, role) != _requests.family_request_schedule()[self._ordinal]
                    or observation.request_id in self._requests
                    or observation.request_id != heldout_request_id(self._binding, index, role)):
                raise ValueError('held-out pair is out of order or duplicated')
            clean, injected = self._baseline.get(index, (None, None))
            self._codec.audit_pair(public, private, site=self._site, clean=clean, injected=injected)
            if role == 'clean':
                self._baseline[index] = (observation, None)
            elif role == 'injected':
                self._baseline[index] = (clean, observation)
            elif role == 'verify_clean':
                del self._baseline[index]
            phase = 'acknowledge'
            _journal._write_parts(self._path / f'{self._ordinal:03d}.ok', (ack,))
            self._requests.add(observation.request_id)
            receipt = _journal.PairReceipt(self._ordinal, digest.hex(), hashlib.sha256(public).hexdigest())
            self._ordinal += 1
            self._previous = digest
            return receipt
        except BaseException as error:
            self._failed = True
            terminal = json.dumps({'schema': 'mlg-number-heldout-journal-failure-v1',
                                   'status': 'incomplete', 'phase': phase, 'ordinal': self._ordinal,
                                   'cancelled': isinstance(error, (KeyboardInterrupt, SystemExit))},
                                  sort_keys=True, separators=(',', ':')).encode('ascii')
            try:
                _journal._write_parts(self._path / 'terminal.json', (terminal,))
            except BaseException as secondary:
                if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise secondary from error
                raise error from secondary
            raise
        finally:
            self._busy = False


@dataclass(frozen=True, repr=False)
class HeldoutInspection:
    observations: tuple
    frame_sha256: str
    retained_tensor_bytes_upper_bound: int
    disk_bytes: int


def _read_journal_file(path, limit):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or info.st_size > limit):
        raise ValueError('held-out journal file is not bounded and ordinary')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size or len(raw) > limit:
        raise ValueError('held-out journal file changed while reading')
    return raw


def inspect_heldout_journal(directory, *, codec, binding, site, max_disk_bytes, acknowledgments,
                            externally_quiescent=False,
                            max_tensor_bytes=72 * 1024 * 1024):
    """Check journal and independent receipts; this does not attest owner success."""
    path = Path(directory)
    _check_binding(binding)
    if (type(codec) is not HeldoutCodec or type(site) is not int or site not in range(1, 5)
            or type(max_disk_bytes) is not int or not 0 < max_disk_bytes <= 192 * 1024 * 1024
            or type(max_tensor_bytes) is not int or not 0 < max_tensor_bytes <= 72 * 1024 * 1024
            or externally_quiescent is not True or type(acknowledgments) is not tuple
            or len(acknowledgments) != 128
            or not path.is_absolute()):
        raise ValueError('held-out inspection configuration is invalid')
    for ordinal, receipt in enumerate(acknowledgments):
        if (type(receipt) is not _journal.PairReceipt or type(receipt.ordinal) is not int
                or receipt.ordinal != ordinal or type(receipt.frame_sha256) is not str
                or re.fullmatch(r'[0-9a-f]{64}', receipt.frame_sha256) is None
                or type(receipt.public_sha256) is not str
                or re.fullmatch(r'[0-9a-f]{64}', receipt.public_sha256) is None):
            raise ValueError('held-out receipts are incomplete or malformed')
    info = path.lstat()
    expected = {f'{index:03d}.{suffix}' for index in range(128) for suffix in ('frame', 'ok')}
    if (not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or {entry.name for entry in path.iterdir()} != expected):
        raise ValueError('held-out journal is incomplete or contains unexpected files')
    previous, seen, baselines, observations = bytes(32), set(), {}, {}
    disk_bytes = tensor_bytes = 0
    for ordinal, (expected_index, expected_role) in enumerate(_requests.family_request_schedule()):
        raw = _read_journal_file(path / f'{ordinal:03d}.frame', 1024 * 1024 + 1024)
        ack = _read_journal_file(path / f'{ordinal:03d}.ok', _journal._ACK.size)
        disk_bytes += len(raw) + len(ack)
        if disk_bytes > max_disk_bytes:
            raise ValueError('held-out journal exceeds its physical allowance')
        public, private, digest = _journal._parse_frame(raw, ordinal, bytes.fromhex(binding.sha256), previous)
        if len(ack) != _journal._ACK.size or _journal._ACK.unpack(ack) != (_journal._ACK_MAGIC, ordinal, digest):
            raise ValueError('held-out acknowledgment differs from its frame')
        receipt = acknowledgments[ordinal]
        if receipt.frame_sha256 != digest.hex() or receipt.public_sha256 != hashlib.sha256(public).hexdigest():
            raise ValueError('independently observed receipt differs from its frame')
        index, role, observation = codec.read_public(public)
        if ((index, role) != (expected_index, expected_role) or observation.request_id in seen
                or observation.request_id != heldout_request_id(binding, index, role)):
            raise ValueError('held-out request plan differs')
        clean, injected = baselines.get(index, (None, None))
        codec.audit_pair(public, private, site=site, clean=clean, injected=injected)
        record = codec._witness['decode_private_forward'](private)
        tensor_bytes += sum(map(len, observation.attention + observation.mlp + (observation.block, observation.logits)))
        for action in record.evidence:
            tensor_bytes += len(action.before) + len(action.after)
            tensor_bytes += len(action.donor) if action.donor is not None else 0
            tensor_bytes += len(action.realized_delta_f64_le) if action.realized_delta_f64_le is not None else 0
        if tensor_bytes > max_tensor_bytes:
            raise ValueError('held-out tensor allowance is exceeded')
        if role == 'clean':
            baselines[index] = (observation, None)
        elif role == 'injected':
            baselines[index] = (clean, observation)
        elif role == 'verify_clean':
            del baselines[index]
        observations[index, role] = observation
        seen.add(observation.request_id)
        previous = digest
    if baselines or len(observations) != 128:
        raise ValueError('held-out journal is missing family-final controls')
    ordered = tuple(tuple(observations[index, role] for role in _archive._ROLES) for index in range(8))
    return HeldoutInspection(ordered, previous.hex(), tensor_bytes, disk_bytes)


@dataclass(frozen=True, repr=False)
class PairRunResult:
    """The producer returned; an external supervisor must still observe its exit."""
    status: str
    forward_count: int
    last_frame_sha256: str
    receipts_sha256: str
    retained_tensor_bytes: int


def read_pair_receipts(path):
    """Read the owner's independently persisted receipt prefix."""
    from .pretrained_number_supervisor import _read_receipts
    return _read_receipts(path)


def run_injection_pairs(*, model, prepared, binding, attempt_dir, expected_model_state_sha256,
                        logical_limit_bytes, disk_limit_bytes, reserved_side_bytes):
    """Produce one private held-out journal after external custody authorization.

    This routine neither authorizes a forward nor proves publication, a phase
    claim, cross-track coordination, actual runtime identity or process exit.
    A trusted source-bound owner must verify those separately before invoking
    it. Scoring and promotion require independently observed clean process exit.
    """
    from .distilgpt2_intervention_dev import _model_state_sha256
    from .pretrained_number_adapter import to_selector_observation
    from .pretrained_number_incomplete import encode_incomplete_forward
    from .pretrained_number_injection import NumberInjectionExecutor
    from .pretrained_number_supervisor import _request_arguments, _write_new, _append_receipt

    mib = 1024 * 1024
    side_reserve = 8 * mib
    if (type(logical_limit_bytes) is not int or not mib <= logical_limit_bytes <= 72 * mib
            or type(disk_limit_bytes) is not int or not side_reserve + 4096 < disk_limit_bytes <= 192 * mib
            or type(reserved_side_bytes) is not int or not 0 <= reserved_side_bytes < disk_limit_bytes - side_reserve - 4096
            or type(expected_model_state_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected_model_state_sha256) is None):
        raise ValueError('held-out phase reservation is invalid')
    check_prepared_binding(binding, prepared)
    plan = heldout_request_plan(binding)
    path = Path(attempt_dir)
    if not path.is_absolute() or not path.parent.is_dir() or os.path.lexists(path):
        raise ValueError('a fresh held-out attempt directory is required')
    path.mkdir(mode=0o700)
    codec = bind_heldout_codec(tuple(item.input_ids for item in prepared.targets))
    current, observation, executor = None, None, None
    phase = 'start'
    try:
        _write_new(path.parent / 'binding.json', binding.record_bytes)
        journal = HeldoutJournal(path / 'journal', codec=codec, binding=binding, site=prepared.site,
                                  max_disk_bytes=disk_limit_bytes - reserved_side_bytes - side_reserve,
                                  max_tensor_bytes=logical_limit_bytes)
        def record(value):
            return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                              separators=(',', ':')).encode('ascii')
        _write_new(path / 'started.json', record({'schema': 'mlg-injection-heldout-owner-started-v1',
            'binding_sha256': binding.sha256, 'heldout_commitment': prepared.heldout_commitment,
            'forward_count_planned': 128}))
        receipt_path = path / 'receipts.jsonl'
        _write_new(receipt_path, b'')
        phase = 'model-state-before'
        if _model_state_sha256(model) != expected_model_state_sha256:
            raise ValueError('model state differs from the verified checkpoint')
        baselines = {}
        last_receipt = None
        with NumberInjectionExecutor(model, site=prepared.site, max_study_bytes=logical_limit_bytes) as executor:
            for ordinal, (index, role, request_id) in enumerate(plan):
                phase = 'admission'
                check_prepared_binding(binding, prepared)
                if (executor.retained_tensor_bytes + mib > logical_limit_bytes
                        or journal._tensor_used + mib > logical_limit_bytes
                        or journal._used + mib + 1024 > journal._limit):
                    raise ValueError('held-out forward reservation is exhausted')
                current = (request_id, role, ordinal)
                phase = 'forward'
                observation = executor.forward(prepared.targets[index].input_ids, request_id=request_id,
                    condition=_archive._CONDITIONS[role], **_request_arguments(role, baselines.get(index, (None, None))))
                if role == 'clean':
                    baselines[index] = (observation, None)
                elif role == 'injected':
                    baselines[index] = (baselines[index][0], observation)
                elif role == 'verify_clean':
                    del baselines[index]
                phase = 'serialize'
                actions = tuple(item for item in executor.private_evidence() if item.request_id == request_id)
                selected = to_selector_observation(observation)
                public = codec.serialize_public(index, role, selected)
                private = codec.serialize_private(index, role, selected, actions)
                phase = 'journal'
                last_receipt = journal.append(public, private)
                phase = 'receipt'
                _append_receipt(receipt_path, last_receipt)
                observation = current = None
        phase = 'model-state-after'
        if (baselines or executor.poisoned or _model_state_sha256(model) != expected_model_state_sha256
                or last_receipt is None or last_receipt.ordinal != 127):
            raise ValueError('model state or final controls differ')
        check_prepared_binding(binding, prepared)
        receipts = read_pair_receipts(receipt_path)
        if len(receipts) != 128 or receipts[-1] != last_receipt:
            raise ValueError('owner receipt sequence is incomplete')
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        result = PairRunResult('owner-produced', 128, last_receipt.frame_sha256, receipt_sha,
                               executor.retained_tensor_bytes)
        _write_new(path / 'owner-terminal.json', record({'schema': 'mlg-injection-heldout-owner-terminal-v1',
            'status': result.status, 'forward_count': result.forward_count, 'binding_sha256': binding.sha256,
            'receipts_sha256': result.receipts_sha256, 'last_frame_sha256': result.last_frame_sha256,
            'model_state_sha256': expected_model_state_sha256, 'retained_tensor_bytes': result.retained_tensor_bytes,
            'scientific_validation': 'pending independent audit'}))
        return result
    except BaseException as error:
        partial_status, pending_cancel, missing = 'unavailable', None, False
        actions, failure = (), None
        def observe_secondary(secondary):
            nonlocal pending_cancel, missing
            missing = True
            if (isinstance(secondary, (KeyboardInterrupt, SystemExit))
                    and not isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None):
                pending_cancel = secondary
        if executor is not None:
            if current is not None:
                request_id, role, _ = current
                if observation is None:
                    try:
                        matches = tuple(item for item in executor.completed() if item.request_id == request_id)
                        if len(matches) == 1:
                            observation = matches[0]
                        elif len(matches) > 1:
                            missing = True
                    except BaseException as secondary:
                        observe_secondary(secondary)
                try:
                    actions = tuple(item for item in executor.private_evidence() if item.request_id == request_id)
                    if observation is not None:
                        operations = [item.operation for item in actions]
                        expected = _witness._required_operations(role)
                        if len(operations) != len(expected) or set(operations) != expected:
                            missing = True
                except BaseException as secondary:
                    observe_secondary(secondary)
            try:
                failures = executor.private_failures()
                failure = failures[-1] if failures else None
            except BaseException as secondary:
                observe_secondary(secondary)
        try:
            if observation is not None or actions or failure is not None:
                partial = encode_incomplete_forward(observation=observation, actions=actions, failure=failure)
                _write_new(path / 'incomplete-forward.bin', partial.data)
                partial_status = 'retained-with-omissions' if missing or partial.omissions else 'retained'
        except BaseException as secondary:
            observe_secondary(secondary)
        try:
            failure_record = json.dumps({'schema': 'mlg-injection-heldout-owner-failure-v1',
                'status': 'incomplete', 'phase': phase, 'partial_evidence': partial_status,
                'cancelled': isinstance(error, (KeyboardInterrupt, SystemExit)) or pending_cancel is not None},
                sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('ascii')
            _write_new(path / 'failure.json', failure_record)
        except BaseException as secondary:
            if (isinstance(secondary, (KeyboardInterrupt, SystemExit))
                    and not isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None):
                pending_cancel = secondary
        if pending_cancel is not None:
            raise pending_cancel from error
        raise


def _audit_produced_pairs(attempt_dir, *, prepared, binding, expected_model_state_sha256,
                          max_disk_bytes=192 * 1024 * 1024):
    """Inspect quiescent bytes after a trusted supervisor has reaped the owner."""
    check_prepared_binding(binding, prepared)
    if (type(max_disk_bytes) is not int or not 0 < max_disk_bytes <= 192 * 1024 * 1024
            or type(expected_model_state_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected_model_state_sha256) is None):
        raise ValueError('held-out audit limits or model identity are invalid')
    path = Path(attempt_dir)
    if not path.is_absolute():
        raise ValueError('held-out attempt path is invalid')
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or {entry.name for entry in path.iterdir()} != {'journal', 'started.json', 'receipts.jsonl',
                                                             'owner-terminal.json'}):
        raise ValueError('held-out owner output is incomplete')
    start_raw = _read_journal_file(path / 'started.json', 4096)
    receipt_raw = _read_journal_file(path / 'receipts.jsonl', 128 * 512)
    terminal_raw = _read_journal_file(path / 'owner-terminal.json', 4096)
    binding_raw = _read_journal_file(path.parent / 'binding.json', 65536)
    if binding_raw != binding.record_bytes:
        raise ValueError('saved held-out binding differs')
    try:
        start, terminal = json.loads(start_raw), json.loads(terminal_raw)
        canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                              separators=(',', ':')).encode('ascii')
        expected_start = {'schema': 'mlg-injection-heldout-owner-started-v1', 'binding_sha256': binding.sha256,
                          'heldout_commitment': prepared.heldout_commitment, 'forward_count_planned': 128}
        terminal_keys = {'schema', 'status', 'forward_count', 'binding_sha256', 'receipts_sha256',
                         'last_frame_sha256', 'model_state_sha256', 'retained_tensor_bytes', 'scientific_validation'}
        if (not _same_json(start, expected_start) or canonical(start) != start_raw or type(terminal) is not dict
                or set(terminal) != terminal_keys or canonical(terminal) != terminal_raw
                or terminal['schema'] != 'mlg-injection-heldout-owner-terminal-v1'
                or terminal['status'] != 'owner-produced' or type(terminal['forward_count']) is not int
                or terminal['forward_count'] != 128 or terminal['binding_sha256'] != binding.sha256
                or terminal['model_state_sha256'] != expected_model_state_sha256
                or terminal['scientific_validation'] != 'pending independent audit'
                or type(terminal['retained_tensor_bytes']) is not int
                or not 0 <= terminal['retained_tensor_bytes'] <= 72 * 1024 * 1024
                or terminal['receipts_sha256'] != hashlib.sha256(receipt_raw).hexdigest()):
            raise ValueError('held-out owner terminal differs')
    except (KeyError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('held-out owner evidence is malformed') from error
    receipts = read_pair_receipts(path / 'receipts.jsonl')
    codec = bind_heldout_codec(tuple(item.input_ids for item in prepared.targets))
    inspection = inspect_heldout_journal(path / 'journal', codec=codec, binding=binding, site=prepared.site,
        max_disk_bytes=max_disk_bytes, acknowledgments=receipts, externally_quiescent=True)
    if (inspection.frame_sha256 != terminal['last_frame_sha256']
            or terminal['retained_tensor_bytes'] > inspection.retained_tensor_bytes_upper_bound):
        raise ValueError('held-out owner and journal inventory differ')
    side_size = len(start_raw) + len(receipt_raw) + len(terminal_raw) + len(binding_raw)
    if side_size + inspection.disk_bytes > max_disk_bytes:
        raise ValueError('held-out physical allowance is exceeded')
    audit = {'status': 'evidence_complete', 'forward_count': 128, 'binding_sha256': binding.sha256,
             'receipts_sha256': terminal['receipts_sha256'], 'last_frame_sha256': inspection.frame_sha256,
             # The reusable supervisor counts raw files in an optional tensors/
             # directory; here all retained tensors are inside encoded frames.
             'inventory': {'retained_tensor_bytes': 0, 'tensor_encoding': 'journal-frames',
                 'encoded_tensor_occurrence_bytes': inspection.retained_tensor_bytes_upper_bound,
                 'journal_file_bytes': inspection.disk_bytes, 'owner_side_file_bytes': side_size},
             'scientific_validation': 'pending independent audit'}
    return audit, inspection


def audit_produced_pairs(attempt_dir, *, prepared, binding, expected_model_state_sha256,
                         max_disk_bytes=192 * 1024 * 1024):
    """Return a bounded structural callback result, not scientific validation."""
    return _audit_produced_pairs(attempt_dir, prepared=prepared, binding=binding,
                                 expected_model_state_sha256=expected_model_state_sha256,
                                 max_disk_bytes=max_disk_bytes)[0]


def _same_json(left, right):
    """Compare JSON values without treating booleans or floats as integers."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same_json(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def score_supervised_pairs(output_dir, *, prepared, binding, expected_command, expected_model_state_sha256,
                           reserved_side_bytes=0, max_disk_bytes=192 * 1024 * 1024):
    """Score only after a trusted supervisor records exit, reap and Job cleanup.

    The caller remains responsible for published commitment, source/runtime,
    gate and claim verification, and independent numerical/landing review. The
    returned frozen-scorer bytes remain private until authorized reveal.
    """
    if (type(expected_command) is not list or not expected_command
            or any(type(part) is not str or not part for part in expected_command)
            or type(reserved_side_bytes) is not int or reserved_side_bytes < 0
            or type(max_disk_bytes) is not int or not 0 < max_disk_bytes <= 192 * 1024 * 1024):
        raise ValueError('supervised scoring configuration is invalid')
    path = Path(output_dir)
    if not path.is_absolute():
        raise ValueError('supervisor output path is invalid')
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or {entry.name for entry in path.iterdir()} !=
               {'attempt', 'binding.json', 'supervisor.json', 'stdout.txt', 'stderr.txt'}):
        raise ValueError('supervisor output contains missing or unexpected files')
    supervisor_raw = _read_journal_file(path / 'supervisor.json', 1024 * 1024)
    stdout_raw = _read_journal_file(path / 'stdout.txt', 1024 * 1024)
    stderr_raw = _read_journal_file(path / 'stderr.txt', 1024 * 1024)
    try:
        supervisor = json.loads(supervisor_raw)
        if (type(supervisor) is not dict or supervisor.get('status') != 'evidence_complete'
                or type(supervisor.get('exit_code')) is not int or supervisor['exit_code'] != 0
                or type(supervisor.get('child_pid')) is not int or not 0 < supervisor['child_pid'] <= 0xffffffff
                or type(supervisor.get('postexit_active_processes')) is not int
                or supervisor['postexit_active_processes'] != 0
                or any(supervisor.get(name) is not True for name in
                       ('child_reaped', 'job_drained', 'job_closed', 'logs_complete', 'cleanup_verified'))
                or any(supervisor.get(name) is not False for name in
                       ('retry_safe', 'cancellation_requested', 'quarantined', 'log_overflow'))
                or supervisor.get('supervisor_error') is not None or supervisor.get('command') != expected_command
                or any(key in supervisor for key in ('reason', 'log_errors', 'audit_error_type'))
                or supervisor.get('stdout_sha256') != hashlib.sha256(stdout_raw).hexdigest()
                or supervisor.get('stderr_sha256') != hashlib.sha256(stderr_raw).hexdigest()
                or not _same_json(supervisor.get('retained_tensor_files'),
                                  {'accounting_complete': True, 'total_bytes': 0, 'files': []})):
            raise ValueError('supervised owner exit or cleanup is not verified')
    except (TypeError, KeyError, UnicodeError, RecursionError) as error:
        raise ValueError('supervisor receipt is malformed') from error
    audit, inspection = _audit_produced_pairs(path / 'attempt', prepared=prepared, binding=binding,
        expected_model_state_sha256=expected_model_state_sha256, max_disk_bytes=max_disk_bytes)
    if not _same_json(supervisor.get('audit'), audit):
        raise ValueError('supervisor audit does not match retained evidence')
    physical_bytes = (audit['inventory']['journal_file_bytes'] + audit['inventory']['owner_side_file_bytes']
                      + len(supervisor_raw) + len(stdout_raw) + len(stderr_raw) + reserved_side_bytes)
    if physical_bytes > max_disk_bytes:
        raise ValueError('held-out physical phase allowance is exceeded')
    cases = []
    for index, (target, row) in enumerate(zip(prepared.targets, inspection.observations)):
        cases.append(_selection.EvaluationCase(index, row[0], row[1], row[2], tuple(row[3:7]),
                     tuple(row[7:11]), row[11], row[12], row[13], row[14], row[15],
                     text=target.text, input_ids=target.input_ids))
    result = _selection.score_evaluation(tuple(cases), discovery_bytes=prepared.prediction_bytes,
        prediction_sha256=INJECTION_PREDICTION_SHA256, prediction_freeze_sha256=INJECTION_FREEZE_SHA256,
        heldout_commitment_sha256=prepared.heldout_commitment, custody_verified=True,
        private_controls_verified=True, execution_device='cpu')
    # Reserve a bounded score receipt and a separate future independent audit.
    if len(result) > 2 * 1024 * 1024 or physical_bytes + len(result) + 65536 + 1024 * 1024 > max_disk_bytes:
        raise ValueError('held-out score and independent audit reservation is exhausted')
    from .pretrained_number_supervisor import _write_new
    _write_new(path / 'evaluation.json', result)
    score_receipt = {'schema': 'mlg-injection-heldout-private-score-v1',
        'status': 'scored-private-awaiting-independent-audit',
        'binding_sha256': binding.sha256, 'binding_file_sha256': hashlib.sha256(binding.record_bytes).hexdigest(),
        'supervisor_sha256': hashlib.sha256(supervisor_raw).hexdigest(),
        'evaluation_sha256': hashlib.sha256(result).hexdigest(),
        'disjoint_status': json.loads(binding.record_bytes)['disjoint_status'],
        'source_qualification': 'prospective orchestration; same-full-source rule departed'}
    _write_new(path / 'post-score-receipt.json', json.dumps(score_receipt, sort_keys=True, ensure_ascii=True,
               allow_nan=False, separators=(',', ':')).encode('ascii'))
    return result
