"""Scientific adapter for the separately committed natural successor.

This does not authenticate a comparator process, published source, tokenizer,
inventory currentness, peer state, model, claim or runtime. A separately
reviewed owner must establish those facts and pin all supplied bytes before
calling. A valid-looking receipt alone is not proof a private check occurred.
This adapter neither creates a nonce nor authorizes or retries a run.
"""
import hashlib
import hmac
import json
import re

from . import agreement_heldout, natural_agreement_successor as successor
from . import pretrained_number_commitments as commitments
from .agreement_patch_runner import AgreementCase, NaturalAgreementRunner, StudyFailure


_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _digest(raw, expected, bound):
    if (type(expected) is not str or _SHA.fullmatch(expected) is None
            or type(raw) is not bytes or not 1 <= len(raw) <= bound
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected)):
        raise StudyFailure('bound successor bytes differ')


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=successor._unique_pairs)
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                             separators=(',', ':')).encode('ascii')
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise StudyFailure('invalid successor JSON') from error
    if encoded != raw:
        raise StudyFailure('successor JSON is not canonical')
    return value


def _clearance(raw, pins, input_sha256, comparator_sha256, freeze_sha256):
    _digest(raw, pins['pool_clear_sha256'], 8192)
    receipt = _json(raw)
    expected = {'schema', 'status', 'comparison', 'natural_heldout_commitment',
                'natural_prediction_freeze_sha256', 'public_input_file_sha256',
                'natural_case_count', 'public_case_count', 'comparator_source_sha256', 'observed_utc'}
    if (type(receipt) is not dict or set(receipt) != expected
            or receipt['schema'] != 'mlg-natural-external-input-comparison-v1'
            or receipt['status'] != 'clear'
            or receipt['comparison'] != 'bidirectional-complete-row-prefix'
            or receipt['natural_heldout_commitment'] != pins['old_commitment']
            or receipt['natural_prediction_freeze_sha256'] != freeze_sha256
            or receipt['public_input_file_sha256'] != input_sha256
            or type(receipt['natural_case_count']) is not int or receipt['natural_case_count'] != 16
            or type(receipt['public_case_count']) is not int or receipt['public_case_count'] != 256
            or receipt['comparator_source_sha256'] != comparator_sha256
            or type(receipt['observed_utc']) is not str or not 1 <= len(receipt['observed_utc']) <= 64):
        raise StudyFailure('pool clearance identity differs')


def run_successor_heldout(runner, *, freeze, prediction_bytes, pool_bytes, input_bytes,
                          input_sha256, tokenizer_sha256, comparator_sha256,
                          pool_clearance_bytes, split_nonce, record_bytes, heldout_commitment,
                          sealed_snapshot, current_snapshot, observed_current_sha256, pins, tokenizer):
    """Validate a fixed successor and invoke the unchanged four-family runner.

    The owner must verify that the clearance came from the bound, successfully
    reaped comparator; check the current inventory and peer state at the actual
    forward boundary; and source-bind every dependency and supplied pin. This
    function accepts no self-reservation exemption and therefore fails closed
    if the candidate rows appear in either supplied inventory.
    """
    if type(runner) is not NaturalAgreementRunner or runner.expected_quartets != 4:
        raise StudyFailure('successor requires the four-family natural runner')
    names = {'pool_sha256', 'protocol_sha256', 'exposure_sha256', 'old_commitment', 'pool_clear_sha256'}
    if (type(pins) is not dict or set(pins) != names
            or any(type(value) is not str or _SHA.fullmatch(value) is None for value in pins.values())
            or any(type(value) is not str or _SHA.fullmatch(value) is None
                   for value in (input_sha256, tokenizer_sha256, comparator_sha256))):
        raise StudyFailure('invalid successor source identities')
    frozen = agreement_heldout.load_frozen_natural_predictions(prediction_bytes, freeze)
    if frozen.freeze.sha256 != successor.PREDICTION_FREEZE_SHA256:
        raise StudyFailure('successor prediction freeze differs')
    _digest(input_bytes, input_sha256, 1000000)
    _clearance(pool_clearance_bytes, pins, input_sha256, comparator_sha256, frozen.freeze.sha256)
    try:
        record = successor.verify_record(pool_bytes, split_nonce, tokenizer, record_bytes,
                                         heldout_commitment, **pins)
        public = _json(input_bytes)
        rows = successor.tokenize_candidate_pool(json.loads(pool_bytes), tokenizer)
        expected = {'schema': 'mlg-natural-successor-candidate-inputs-v1',
                    'pool_sha256': pins['pool_sha256'], 'tokenizer_sha256': tokenizer_sha256,
                    'cases': [{'input_ids': list(row)} for row in rows]}
        if public != expected or len(rows) != 256:
            raise StudyFailure('candidate inputs do not match the fixed pool')
        if (type(sealed_snapshot) is not commitments.InventorySnapshot
                or sealed_snapshot.sha256 != pins['exposure_sha256']):
            raise StudyFailure('sealed exposure identity differs')
        commitments._inventory_prefixes(sealed_snapshot, pins['exposure_sha256'])
        commitments._inventory_prefixes(current_snapshot, observed_current_sha256)
        selected = [tuple(item['input_ids']) for family in record['families'] for item in family['cases']]
        if len(selected) != 16 or not set(selected).issubset(set(rows)):
            raise StudyFailure('selected rows are not the cleared pool subset')
        for snapshot in (sealed_snapshot, current_snapshot):
            successor.reject_known_rows(selected, json.loads(snapshot.record_bytes)['records'])
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise StudyFailure('successor record or exposure validation failed') from error
    cases = tuple(AgreementCase(case_id=f"heldout-{family['family_index']}-{index}",
                                quartet=f"heldout-{family['family_index']}", lemma=f'lemma-{index // 2}',
                                number='singular' if index % 2 == 0 else 'plural',
                                input_ids=tuple(item['input_ids']), subject_index=item['subject_index'])
                  for family in record['families'] for index, item in enumerate(family['cases']))
    binding = agreement_heldout.HeldoutBinding(heldout_commitment, sealed_snapshot.sha256)
    return runner.run(cases, mode='heldout', frozen_predictions=frozen, heldout_binding=binding)
