"""Append one GPU development-family attempt to the cited BERT history.

This is an evidence overlay on eight already-observed rows, not a complete
exposure history or an attestation of execution.
"""
import copy
from dataclasses import replace
import hashlib
import json

from . import bert_agreement_exposures_v2 as v2
from . import bert_agreement_exposures_v3 as v3


SCHEMA = 'distilbert-agreement-exposure-status-v4'
PREDECESSOR_SHA256 = '6945a478e3af37083af055bf20bf65628fa979e0ee8b253ea8bacb475cb9e9f2'
SNAPSHOT_SHA256 = '1f9ae67040646341097414f5ef6da1f66d0ccc5f0be47deeee36cfe66550e32e'
EVIDENCE_ID = 'gpu-family1-profile'
EVIDENCE_COMMIT = 'bca5a886303e070e90f92c7e23ef301f0ded836f'
SOURCE_COMMIT = '4a597641be89ee094145fbdacee4d3fcdd241e41'
RUN_ID = '1f67a55c190391f693c3f3739de6ff12'
EVIDENCE_ROOT = 'results/bert-gpu-family-profile-v1-20261029/'
RESULT_SHA256 = 'd7537ab322018794682d3e3ffd93eea83c6175a71d0ea20ecd156dac97cc2289'
JOURNAL_SHA256 = '152e3f463a9bc7b8f8f0ef1e05e9304aa93c521b74ee19e5a2484af149246190'


def build_snapshot():
    """Return a detached overlay after verifying the exact v3 predecessor."""
    if v3.snapshot_digest() != PREDECESSOR_SHA256:
        raise ValueError('predecessor exposure status does not match')
    payload = copy.deepcopy(v3.build_snapshot())
    payload['schema'] = SCHEMA
    payload['predecessor'] = {'schema': v3.SCHEMA, 'sha256': PREDECESSOR_SHA256}
    if any(item['id'] == EVIDENCE_ID for item in payload['evidence']):
        raise ValueError('duplicate evidence identifier')
    payload['evidence'].append({
        'id': EVIDENCE_ID, 'status': 'complete', 'reported_terminal_status': 'success',
        'reference_commit': EVIDENCE_COMMIT, 'source_commit': SOURCE_COMMIT, 'run_id': RUN_ID,
        'result_path': EVIDENCE_ROOT + 'result.json', 'result_sha256': RESULT_SHA256,
        'journal_path': EVIDENCE_ROOT + 'evidence/events.jsonl', 'journal_sha256': JOURNAL_SHA256,
        'forward_count': 140,
    })
    matched = set()
    for row in payload['rows']:
        if row['row_id'] in v2.FAMILY_ONE_IDS:
            if row['split'] != 'development' or row['status'] != 'observed-in-cited-evidence':
                raise ValueError('GPU profile does not match previously observed development rows')
            matched.add(row['row_id'])
            row['observed_in'].append(EVIDENCE_ID)
    if matched != v2.FAMILY_ONE_IDS:
        raise ValueError('GPU profile does not match all eight family-one rows')
    return payload


def canonical_snapshot_bytes():
    return json.dumps(build_snapshot(), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def snapshot_digest():
    digest = hashlib.sha256(canonical_snapshot_bytes()).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise ValueError('exposure-status snapshot does not match its pinned digest')
    return digest


def verify_snapshot_bytes(data):
    """Verify exact bytes and inherited full-row comparison, not authorization."""
    if type(data) is not bytes or len(data) > 1_000_000 or data != canonical_snapshot_bytes():
        raise ValueError('exposure snapshot is not the pinned immutable record')
    digest = snapshot_digest()
    predecessor = v3.verify_snapshot_bytes(v3.canonical_snapshot_bytes())
    return replace(predecessor, snapshot_sha256=digest)
