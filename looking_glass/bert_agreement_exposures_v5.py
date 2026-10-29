"""Append the completed CPU evaluation to the cited DistilBERT history.

The v4 record remains the immutable pre-evaluation snapshot. This overlay
records the saved attempt; it neither certifies all history nor authenticates
execution, and it does not authorize a further evaluation.
"""
import copy
from dataclasses import dataclass
import hashlib
import json

from . import bert_agreement_exposures_v4 as v4


SCHEMA = 'distilbert-agreement-exposure-status-v5'
PREDECESSOR_SHA256 = '1f9ae67040646341097414f5ef6da1f66d0ccc5f0be47deeee36cfe66550e32e'
SNAPSHOT_SHA256 = '34db577a211d6036e7803b8cdb6aaab35687b56974de8cf1ac6bccbf3272fb3c'
EVIDENCE_ID = 'cpu-evaluation-v1'
EVIDENCE_COMMIT = 'ada1b7746a5541af2e6addbc943c08a6b737e0b4'
SOURCE_COMMIT = '4ee50069464228798071218b58b64221ac5960f4'
RUN_ID = 'ba65abf8bbf7680ba28d2a2ebea67d45'
EVIDENCE_ROOT = 'results/bert-agreement-cpu-evaluation-v1-20261029/'
RESULT_SHA256 = '7637da956e49bbae690cfea45debe7fb89f6bb600ee85a75dbb28223553e48d5'
JOURNAL_GZIP_SHA256 = 'ad791c45388d6d8a5ee34c6710dd9968760c7fc17007fa3609a24fe0a00f9c01'
JOURNAL_SHA256 = 'f30b6ae11e4fbf5276d9f5a5dc3ae5206fd2a3c9f2fdf4252aae0a844fb6f7ee'
CONTROLLER_EXIT_SHA256 = 'da61131a78d6f828b31584b78351d40a8db9fe083971f88951ca357e81e93ee5'


def build_snapshot():
    """Return a detached overlay after verifying the pinned v4 predecessor."""
    if v4.snapshot_digest() != PREDECESSOR_SHA256:
        raise ValueError('predecessor exposure status does not match')
    payload = copy.deepcopy(v4.build_snapshot())
    payload['schema'] = SCHEMA
    payload['predecessor'] = {'schema': v4.SCHEMA, 'sha256': PREDECESSOR_SHA256}
    if any(item['id'] == EVIDENCE_ID for item in payload['evidence']):
        raise ValueError('duplicate evidence identifier')
    payload['evidence'].append({
        'id': EVIDENCE_ID, 'status': 'complete', 'reported_terminal_status': 'complete',
        'reference_commit': EVIDENCE_COMMIT, 'source_commit': SOURCE_COMMIT, 'run_id': RUN_ID,
        'result_path': EVIDENCE_ROOT + 'evidence/result.json', 'result_sha256': RESULT_SHA256,
        'journal_path': EVIDENCE_ROOT + 'evidence/events.jsonl.gz',
        'journal_sha256': JOURNAL_GZIP_SHA256, 'journal_uncompressed_sha256': JOURNAL_SHA256,
        'controller_exit_path': EVIDENCE_ROOT + 'evidence/controller-exit.json',
        'controller_exit_sha256': CONTROLLER_EXIT_SHA256,
        'forward_count': 1120, 'control_count': 384, 'target_cases': 32, 'input_rows': 64,
    })
    matched = set()
    for row in payload['rows']:
        if row['split'] == 'evaluation':
            if (row['status'] != 'public-evaluation-not-observed-in-cited-evidence'
                    or row['observed_in'] or row['row_id'] in matched):
                raise ValueError('evaluation evidence does not match the predecessor')
            matched.add(row['row_id'])
            row['status'] = 'observed-in-cited-evidence'
            row['observed_in'].append(EVIDENCE_ID)
    if len(matched) != 64:
        raise ValueError('evaluation evidence does not match all 64 public rows')
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
    observed_development_rows: int
    reserved_development_rows: int
    public_evaluation_rows: int
    observed_evaluation_rows: int
    exact_development_evaluation_overlap: int
    scope: str = 'known-full-token-rows-only'
    complete_history_certified: bool = False
    execution_authorized: bool = False


def verify_snapshot_bytes(data):
    """Verify the exact cited status overlay and inherited row disjointness."""
    if type(data) is not bytes or len(data) > 1_000_000 or data != canonical_snapshot_bytes():
        raise ValueError('exposure snapshot is not the pinned immutable record')
    digest = snapshot_digest()
    prior = v4.verify_snapshot_bytes(v4.canonical_snapshot_bytes())
    rows = build_snapshot()['rows']
    observed = sum(row['status'] == 'observed-in-cited-evidence'
                   for row in rows if row['split'] == 'evaluation')
    if prior.public_evaluation_rows != 64 or observed != 64:
        raise ValueError('evaluation observation inventory is incomplete')
    return ExposureStatusCheck(digest, prior.observed_development_rows,
        prior.reserved_development_rows, prior.public_evaluation_rows, observed,
        prior.exact_development_evaluation_overlap)
