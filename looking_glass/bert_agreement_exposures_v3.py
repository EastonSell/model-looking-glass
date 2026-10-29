"""Record a supervised development repeat without rewriting earlier status.

The cited attempt reused eight already observed rows. This is a known-row
evidence overlay, not a complete exposure-history or execution attestation.
"""
import copy
from dataclasses import replace
import hashlib
import json

from . import bert_agreement_exposures_v2 as v2


SCHEMA = 'distilbert-agreement-exposure-status-v3'
PREDECESSOR_SHA256 = '224d683291b5b4f3a617697c8f9811351cf63bef19958fff9870c943e88fcdbc'
SNAPSHOT_SHA256 = '6945a478e3af37083af055bf20bf65628fa979e0ee8b253ea8bacb475cb9e9f2'
EVIDENCE_COMMIT = '6eb1a50bc65d056eb82a80d4c7068220eb11e8aa'
SOURCE_COMMIT = 'b0dde0dd66712771a3a5b314dd4f2431c73f852c'
RUN_ID = 'bb4d5c3acc37d606b527d1b1fd6b46b1'
EVIDENCE_ROOT = 'results/bert-agreement-supervised-dev-v1-20261028/'
RESULT_SHA256 = '552b09905b977cce3ba76a9aa58ec49f18f44f28864b4fcae16c838de5853493'
INVENTORY_SHA256 = '3c4357e2d85ac9a4acb213d4190a41985446052e8503547765c080fc27646d98'
EXIT_SHA256 = '80aa7448eb1d489c37b8f351a58642c2d1ae652ea50d2df5f10848af24117cc7'
EVIDENCE_ID = 'cpu-family1-supervised-repeat'


def build_snapshot():
    """Return a detached overlay after verifying the exact v2 predecessor."""
    if v2.snapshot_digest() != PREDECESSOR_SHA256:
        raise ValueError('predecessor exposure status does not match')
    payload = copy.deepcopy(v2.build_snapshot())
    payload['schema'] = SCHEMA
    payload['predecessor'] = {'schema': v2.SCHEMA, 'sha256': PREDECESSOR_SHA256}
    if any(row['id'] == EVIDENCE_ID for row in payload['evidence']):
        raise ValueError('duplicate evidence identifier')
    payload['evidence'].append({
        'id': EVIDENCE_ID, 'status': 'complete', 'reference_commit': EVIDENCE_COMMIT,
        'source_commit': SOURCE_COMMIT, 'run_id': RUN_ID,
        'result_path': EVIDENCE_ROOT + 'result.json', 'result_sha256': RESULT_SHA256,
        'inventory_path': EVIDENCE_ROOT + 'inventory.json', 'inventory_sha256': INVENTORY_SHA256,
        'controller_exit_path': EVIDENCE_ROOT + 'controller-exit.json', 'controller_exit_sha256': EXIT_SHA256,
        'journal_reference_commit': v2.CPU_EVIDENCE_COMMIT,
        'journal_path': 'results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz',
        'journal_uncompressed_sha256': v2.JOURNAL_SHA256,
    })
    matched = set()
    for row in payload['rows']:
        if row['row_id'] in v2.FAMILY_ONE_IDS:
            if row['split'] != 'development' or row['status'] != 'observed-in-cited-evidence':
                raise ValueError('repeat does not match observed predecessor rows')
            matched.add(row['row_id'])
            row['observed_in'].append(EVIDENCE_ID)
    if matched != v2.FAMILY_ONE_IDS:
        raise ValueError('repeat does not match all eight family-one rows')
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
    """Verify exact bytes and inherited known full-row comparison, not authorization."""
    if type(data) is not bytes or len(data) > 1_000_000 or data != canonical_snapshot_bytes():
        raise ValueError('exposure snapshot is not the pinned immutable record')
    digest = snapshot_digest()
    predecessor = v2.verify_snapshot_bytes(v2.canonical_snapshot_bytes())
    return replace(predecessor, snapshot_sha256=digest)
