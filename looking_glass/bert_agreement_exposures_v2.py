"""Append-only status for the public DistilBERT agreement case inventory.

This does not load a model or attest that the saved records describe every
historical exposure. The frozen v1 case manifest and its row statuses remain
unchanged; this successor annotates only the cited public evidence.
"""
from dataclasses import dataclass
import hashlib
import json

from . import bert_agreement_cases as v1


SCHEMA = 'distilbert-agreement-exposure-status-v2'
PREDECESSOR_SHA256 = '20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5'
SNAPSHOT_SHA256 = '224d683291b5b4f3a617697c8f9811351cf63bef19958fff9870c943e88fcdbc'
FAMILY_ONE_IDS = frozenset(f'development-f01-{source}-{subject}-{distractor}'
                           for source in ('A', 'B') for subject in ('s', 'p')
                           for distractor in ('s', 'p'))
GPU_IDS = frozenset({'development-f01-A-s-p', 'development-f01-A-p-p'})
CPU_EVIDENCE_COMMIT = 'bf3e5f94c2a723806aaafdfdc609a6340c807dc5'
GPU_EVIDENCE_COMMIT = 'eef5522ebdb0319cc57882e1a1cafbc5881efe6f'
PILOT_EVIDENCE_COMMIT = '9b896e926be68fceaa20d3eaf25b03ed3aad8227'
JOURNAL_SHA256 = 'ff14574bf1b35058b3cff51e4752180ec589982cf3a547baba318a9591af8b67'
GPU_RESULT_SHA256 = '3ca46d8824a181e10836a701c5b6926dd865a86b171ba5cd3963a6db3477d7d2'


def _evidence():
    return [
        {'id': 'pilot-v0', 'status': 'development', 'reference_commit': PILOT_EVIDENCE_COMMIT,
         'result_path': 'results/bert-agreement-dev-v0-20261028/result.json',
         'result_sha256': v1.PILOT_RESULT_SHA256},
        {'id': 'pilot-v0-replay', 'status': 'development', 'reference_commit': PILOT_EVIDENCE_COMMIT,
         'result_path': 'results/bert-agreement-dev-v0-20261028/replay/result.json',
         'result_sha256': v1.PILOT_REPLAY_SHA256},
        {'id': 'cpu-family1-incomplete', 'status': 'incomplete', 'reference_commit': CPU_EVIDENCE_COMMIT,
         'source_commit': 'cc66516375c181eb6d40f04eed0ed96800e0dc84',
         'run_id': '411eb82f68859d3f15a6701dd2d44344',
         'result_path': 'results/bert-agreement-dev-v1-20261028/attempt-01/result.json',
         'result_sha256': '44cfc69c4a708e6dee98284b55cf5f4d9e993909f3bbc30e4cfb1781dd2e1278',
         'journal_path': 'results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz',
         'journal_uncompressed_sha256': JOURNAL_SHA256},
        {'id': 'cpu-family1-retry', 'status': 'complete', 'reference_commit': CPU_EVIDENCE_COMMIT,
         'source_commit': '186cbb8510caa28610348ee000b5504dc2cf6179',
         'run_id': 'dd100dc14f21292d5017a05828bb9288',
         'result_path': 'results/bert-agreement-dev-v1-20261028/attempt-02/result.json',
         'result_sha256': 'dc2b9160ac7b8acc694a65578261a2792509fae0fe9bbada9f48d590692d4ae8',
         'journal_path': 'results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz',
         'journal_uncompressed_sha256': JOURNAL_SHA256},
        {'id': 'gpu-calibration', 'status': 'success', 'reference_commit': GPU_EVIDENCE_COMMIT,
         'source_commit': 'c2bff8183003ac25ec28475d23ddd20e7026f370',
         'run_id': 'f7202654258699ad97b68f2141fab8a6',
         'result_path': 'results/bert-gpu-calibration-v1-20261029/result.json',
         'result_sha256': GPU_RESULT_SHA256},
    ]


def build_snapshot():
    """Build a fresh overlay from the exact predecessor and saved row IDs."""
    if v1.manifest_digest() != PREDECESSOR_SHA256:
        raise ValueError('predecessor case manifest does not match')
    manifest = v1.build_manifest()
    source = manifest['rows']
    if len(source) != 96 or len({row['row_id'] for row in source}) != 96:
        raise ValueError('predecessor rows are incomplete or duplicated')
    if {row['row_id'] for row in source if row['row_id'] in FAMILY_ONE_IDS} != FAMILY_ONE_IDS:
        raise ValueError('family-one evidence does not match the predecessor')
    gpu_rows = {row['row_id'] for row in source if row['row_id'] in GPU_IDS}
    if gpu_rows != GPU_IDS or not GPU_IDS <= FAMILY_ONE_IDS:
        raise ValueError('GPU evidence does not match the predecessor')
    rows = []
    for row in source:
        observed = []
        if tuple(row['input_ids']) in v1.PILOT_ROWS:
            observed.extend(('pilot-v0', 'pilot-v0-replay'))
        if row['row_id'] in FAMILY_ONE_IDS:
            observed.extend(('cpu-family1-incomplete', 'cpu-family1-retry'))
        if row['row_id'] in GPU_IDS:
            observed.append('gpu-calibration')
        status = ('observed-in-cited-evidence' if observed else
                  'reserved-development' if row['split'] == 'development' else
                  'public-evaluation-not-observed-in-cited-evidence')
        rows.append({'row_id': row['row_id'], 'split': row['split'], 'input_ids': list(row['input_ids']),
                     'attention_mask': list(row['attention_mask']), 'v1_exposure_status': row['exposure_status'],
                     'status': status, 'observed_in': observed})
    return {'schema': SCHEMA, 'predecessor': {'schema': v1.SCHEMA, 'sha256': PREDECESSOR_SHA256},
            'identity': manifest['identity'], 'scope': 'cited public evidence; unknown history and pretraining',
            'evidence': _evidence(), 'rows': rows}


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
    exact_development_evaluation_overlap: int
    scope: str = 'known-full-token-rows-only'
    complete_history_certified: bool = False
    execution_authorized: bool = False


def verify_snapshot_bytes(data):
    """Check exact successor bytes and full-row development/evaluation disjointness."""
    if type(data) is not bytes or len(data) > 1_000_000 or data != canonical_snapshot_bytes():
        raise ValueError('exposure snapshot is not the pinned immutable record')
    digest = snapshot_digest()
    rows = build_snapshot()['rows']
    development = [row for row in rows if row['split'] == 'development']
    evaluation = [row for row in rows if row['split'] == 'evaluation']
    dev_tokens = {tuple(row['input_ids']) for row in development}
    eval_tokens = {tuple(row['input_ids']) for row in evaluation}
    if len(development) != 32 or len(dev_tokens) != 32 or len(evaluation) != 64 or len(eval_tokens) != 64:
        raise ValueError('case rows are incomplete or duplicated')
    overlap = len(dev_tokens & eval_tokens)
    if overlap:
        raise ValueError('public evaluation overlaps known development rows')
    observed = sum(row['status'] == 'observed-in-cited-evidence' for row in development)
    return ExposureStatusCheck(digest, observed, len(development) - observed, len(evaluation), overlap)
