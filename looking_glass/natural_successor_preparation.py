"""One-use successor preparation inside a separately authorized owner.

This component reads only public inputs until it creates the new phase claim.
It never opens the old private split or loads a model. The caller must first
authenticate exact published source/evidence, the no-model owner rehearsal,
root ACLs and identities, exclusive cooperative custody and the current
inventory, and must independently observe this process's exit. The callbacks
below are trusted owner checks, not authentication by this helper. An error or
uncertain exit after the phase claim consumes the attempt; never redraw.
The owner must also enforce the protocol's one-attempt stop on pre-claim
failure and suppress private diagnostics, including exceptions from tokenizer
or record processing. This helper does not enforce either outer obligation.
"""
import hashlib
import json
import os
from pathlib import Path

from . import natural_agreement_successor as sampler
from . import natural_successor_claim as claim
from . import natural_successor_execution as execution
from . import natural_successor_split_state as split_state
from . import pretrained_number_commitments as commitments


TOKENIZER = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
COMPARATOR = '005a1a3ce967fc7d254b89800f4099dfbf3f966a972ed018f31d7e885a5db554'


class PreparationError(ValueError):
    """The single preparation cannot safely proceed."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=sampler._unique_pairs)
        if _canonical(value) != raw:
            raise PreparationError('public JSON is not canonical')
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise PreparationError('public JSON is invalid') from None


def _write_new(path, raw):
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short preparation write')
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != raw:
        raise PreparationError('prepared file differs on readback')


def _nonce():
    return os.urandom(32)


def prepare(*, gate, root, operation_sha256, pool_bytes, input_bytes, clearance_bytes,
            inventory_bytes, inventory_sha256, tokenizer, check_before_claim, check_before_draw):
    """Claim, announce and draw once under an already held shared gate.

    Both callbacks must perform the source-bound owner's actual currentness,
    evidence/lease, ACL, budget and rehearsal checks as applicable.  The second
    runs after allocation and immediately before drawing.  This function does
    not establish that a callback is trustworthy.  Returned digests omit the
    nonce, cases and unsalted private record hash.
    """
    split_state._sha(operation_sha256)
    split_state._sha(inventory_sha256)
    for raw, expected in ((pool_bytes, split_state.POOL), (input_bytes, split_state.INPUTS),
                          (clearance_bytes, split_state.PUBLIC_CLEARANCE),
                          (inventory_bytes, inventory_sha256)):
        if type(raw) is not bytes or not raw or _sha(raw) != expected:
            raise PreparationError('bound public bytes differ')
    pool = _json(pool_bytes)
    inputs = _json(input_bytes)
    rows = sampler.tokenize_candidate_pool(pool, tokenizer)
    expected_inputs = {'schema': 'mlg-natural-successor-candidate-inputs-v1',
                       'pool_sha256': split_state.POOL, 'tokenizer_sha256': TOKENIZER,
                       'cases': [{'input_ids': list(row)} for row in rows]}
    if len(rows) != 256 or inputs != expected_inputs:
        raise PreparationError('candidate rows differ from the fixed pool')
    pins = {'pool_sha256': split_state.POOL, 'protocol_sha256': split_state.PROTOCOL,
            'exposure_sha256': inventory_sha256, 'old_commitment': split_state.OLD_COMMITMENT,
            'pool_clear_sha256': split_state.PUBLIC_CLEARANCE}
    execution._clearance(clearance_bytes, pins, split_state.INPUTS, TOKENIZER, COMPARATOR, split_state.FREEZE)
    snapshot = commitments.load_inventory_snapshot(inventory_bytes, inventory_sha256)
    records = _json(snapshot.record_bytes)['records']
    sampler.reject_known_rows(rows, records)
    if split_state.inspect(gate, root)['status'] != 'not_allocated_unchecked':
        raise PreparationError('successor phase is not fresh')
    if not callable(check_before_claim) or check_before_claim(inventory_sha256) is not True:
        raise PreparationError('pre-claim authorization is unavailable')
    if split_state.inspect(gate, root)['status'] != 'not_allocated_unchecked':
        raise PreparationError('successor or shared gate changed before claim')
    claim_sha = claim.claim_public_phase(root=root, root_path_sha256=split_state.SUCCESSOR_ROOT_SHA256,
        root_device=split_state.SUCCESSOR_ROOT_IDENTITY[0], root_inode=split_state.SUCCESSOR_ROOT_IDENTITY[1],
        operation_sha256=operation_sha256, public_clearance_sha256=split_state.PUBLIC_CLEARANCE,
        inventory_sha256=inventory_sha256, amendment_sha256=split_state.AMENDMENT)
    allocation = split_state.announce_allocation(gate, root)
    if (allocation['phase_claim_sha256'] != claim_sha
            or not callable(check_before_draw) or check_before_draw(inventory_sha256) is not True):
        raise PreparationError('pre-draw authorization is unavailable')
    if split_state.inspect(gate, root) != allocation:
        raise PreparationError('allocation or shared gate changed before drawing')
    root = Path(root)
    # Exclusive creation precedes entropy acquisition: a partial nonce file or
    # any later failure leaves this one-use phase consumed.
    with (root / 'nonce.bin').open('xb') as stream:
        nonce = _nonce()
        if type(nonce) is not bytes or len(nonce) != 32 or stream.write(nonce) != len(nonce):
            raise PreparationError('nonce creation failed')
        stream.flush()
        os.fsync(stream.fileno())
    if (root / 'nonce.bin').read_bytes() != nonce:
        raise PreparationError('nonce readback differs')
    heldout, record = sampler.seal_record(pool_bytes, nonce, tokenizer, **pins)
    selected_record = _json(record)
    selected = [tuple(case['input_ids']) for family in selected_record['families'] for case in family['cases']]
    if len(selected) != 16 or not set(selected).issubset(set(rows)):
        raise PreparationError('selected rows differ from the cleared pool')
    sampler.reject_known_rows(selected, records)
    _write_new(root / 'heldout-record.json', record)
    public = _canonical({'schema': 'mlg-natural-successor-public-commitment-v1',
        'prediction_freeze_sha256': split_state.FREEZE, 'pool_sha256': split_state.POOL,
        'inputs_sha256': split_state.INPUTS, 'pool_clearance_sha256': split_state.PUBLIC_CLEARANCE,
        'exposure_inventory_sha256': inventory_sha256, 'old_natural_commitment': split_state.OLD_COMMITMENT,
        'protocol_sha256': split_state.PROTOCOL, 'amendment_sha256': split_state.AMENDMENT,
        'operation_sha256': operation_sha256, 'heldout_commitment': heldout})
    _write_new(root / 'commitment.json', public)
    if split_state.inspect(gate, root) != allocation:
        raise PreparationError('allocation changed during preparation')
    return {'schema': 'mlg-natural-successor-preparation-result-v1', 'status': 'prepared_unpublished',
            'phase_claim_sha256': claim_sha, 'allocation_sha256': allocation['allocation_sha256'],
            'heldout_commitment': heldout, 'commitment_file_sha256': _sha(public),
            'inventory_sha256': inventory_sha256, 'operation_sha256': operation_sha256}
