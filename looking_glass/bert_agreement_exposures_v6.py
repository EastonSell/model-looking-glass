"""Append the matched-dose CPU attempt to the cited DistilBERT history.

The v5 record and the matched pre-run case manifest stay unchanged.  This is
an inventory of cited evidence, not complete exposure history or an execution
attestation, and it does not authorize another evaluation.
"""
import copy
from dataclasses import dataclass
import hashlib
import json

from . import bert_agreement_exposures_v5 as v5
from . import bert_matched_cases as cases


SCHEMA = 'distilbert-agreement-exposure-status-v6'
PREDECESSOR_SHA256 = '34db577a211d6036e7803b8cdb6aaab35687b56974de8cf1ac6bccbf3272fb3c'
MANIFEST_SHA256 = '5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb'
SNAPSHOT_SHA256 = '0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb'
EVIDENCE_ID = 'cpu-matched-magnitude-v1'
EVIDENCE_COMMIT = '48b490ec1dc5fe146fa8f752a7cc4a6be18e9aa7'
SOURCE_COMMIT = '4ac00abf6938b843abc5015fa83e7ad60d993262'
ATTEMPT_ID = '6ce9531f5be90f46574ba6771851817c'
EVIDENCE_ROOT = 'results/bert-matched-dose-v1-20261029/'
RESULT_SHA256 = 'c5045afb6b7dbe518517a8156e52469055d9bd578b33524c7c8beb1a3e7e0eb2'
RESERVATION_SHA256 = 'f9c66fe0e5e93013a2561a96bffc2438f72f66377dd0438cbb2b1d1cc20bc036'
JOURNAL_GZIP_SHA256 = '1a405510648a649b7561a88fa2a770532fcafb2e6e87d587e038eada99033495'
JOURNAL_SHA256 = '9f482a369742960e24273d595d33d2a5fdc8ba2dfc82f1d65f6046266301a3ca'
CONTROLLER_EXIT_SHA256 = 'ce014315ebb7bb6e54c784a0241f28f76dc2dd751cd07e220b98df76261bc996'


def build_snapshot():
    """Return a detached v5 successor with the 64 cited matched rows."""
    if v5.snapshot_digest() != PREDECESSOR_SHA256:
        raise ValueError('predecessor exposure status does not match')
    if cases.manifest_digest() != MANIFEST_SHA256:
        raise ValueError('matched pre-run manifest does not match')
    cases.verify_manifest_bytes(cases.canonical_manifest_bytes())
    payload = copy.deepcopy(v5.build_snapshot())
    manifest = cases.build_manifest()
    if manifest['identity'] != payload['identity']:
        raise ValueError('matched model identity differs from the predecessor')
    payload['schema'] = SCHEMA
    payload['predecessor'] = {'schema': v5.SCHEMA, 'sha256': PREDECESSOR_SHA256}
    payload['matched_pre_run_manifest'] = {'schema': cases.SCHEMA, 'sha256': MANIFEST_SHA256}
    if any(item['id'] == EVIDENCE_ID for item in payload['evidence']):
        raise ValueError('duplicate evidence identifier')
    payload['evidence'].append({
        'id': EVIDENCE_ID, 'status': 'complete',
        'reported_terminal_status': 'complete_candidate',
        'reference_commit': EVIDENCE_COMMIT, 'source_commit': SOURCE_COMMIT,
        'attempt_id': ATTEMPT_ID,
        'result_path': EVIDENCE_ROOT + 'controller-result.json',
        'result_sha256': RESULT_SHA256,
        'reservation_path': EVIDENCE_ROOT + 'reservation.json',
        'reservation_sha256': RESERVATION_SHA256,
        'journal_path': EVIDENCE_ROOT + 'events.jsonl.gz',
        'journal_sha256': JOURNAL_GZIP_SHA256,
        'journal_uncompressed_sha256': JOURNAL_SHA256,
        'controller_exit_path': EVIDENCE_ROOT + 'controller-exit-observation.json',
        'controller_exit_sha256': CONTROLLER_EXIT_SHA256,
        'forward_count': 480, 'control_count': 160, 'target_cases': 32, 'input_rows': 64,
    })
    old_ids = {row['row_id'] for row in payload['rows']}
    old_tokens = {tuple(row['input_ids']) for row in payload['rows']}
    new_ids, new_tokens = set(), set()
    for row in manifest['rows']:
        row_id, tokens = row['row_id'], tuple(row['input_ids'])
        if (row['split'] != 'evaluation'
                or row['exposure_status'] != 'public-evaluation-not-observed-in-cited-evidence'
                or row_id in old_ids or row_id in new_ids
                or tokens in old_tokens or tokens in new_tokens):
            raise ValueError('matched rows are duplicate, overlapping or not pre-run rows')
        new_ids.add(row_id)
        new_tokens.add(tokens)
        payload['rows'].append({
            'row_id': row_id, 'split': row['split'],
            'input_ids': list(row['input_ids']), 'attention_mask': list(row['attention_mask']),
            'source_manifest_sha256': MANIFEST_SHA256,
            'source_exposure_status': row['exposure_status'],
            'status': 'observed-in-cited-evidence', 'observed_in': [EVIDENCE_ID],
        })
    if len(new_ids) != 64 or len(manifest['targets']) != 32:
        raise ValueError('matched evidence does not cover 64 rows and 32 targets')
    return payload


def canonical_snapshot_bytes():
    return json.dumps(build_snapshot(), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def snapshot_digest():
    digest = hashlib.sha256(canonical_snapshot_bytes()).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise ValueError('exposure-status snapshot does not match its pinned digest')
    return digest


@dataclass(frozen=True)
class ExposureStatusCheck:
    snapshot_sha256: str
    inherited_rows: int
    matched_rows: int
    observed_matched_rows: int
    exact_prior_row_overlap: int
    scope: str = 'known-full-token-rows-only'
    complete_history_certified: bool = False
    execution_authorized: bool = False


def verify_snapshot_bytes(data):
    """Verify the pinned successor and the complete, disjoint row inventory."""
    if type(data) is not bytes or len(data) > 1_000_000 or data != canonical_snapshot_bytes():
        raise ValueError('exposure snapshot is not the pinned immutable record')
    digest = snapshot_digest()
    v5.verify_snapshot_bytes(v5.canonical_snapshot_bytes())
    inherited = v5.build_snapshot()['rows']
    rows = build_snapshot()['rows']
    added = rows[len(inherited):]
    if rows[:len(inherited)] != inherited:
        raise ValueError('predecessor rows changed')
    observed = sum(row['status'] == 'observed-in-cited-evidence' for row in added)
    overlap = len({tuple(row['input_ids']) for row in inherited}
                  & {tuple(row['input_ids']) for row in added})
    if len(inherited) != 96 or len(added) != 64 or observed != 64 or overlap:
        raise ValueError('matched observation inventory is incomplete or overlapping')
    return ExposureStatusCheck(digest, len(inherited), len(added), observed, overlap)
