"""Verify a publicly read-back discovery freeze before loading an evaluation model.

The caller separately observes the immutable public commit/blob and verifies
closed direct-process receipts, source identity and prior baseline provenance.
Supplying identifiers here is not an independent publication attestation.
"""
import hashlib
import json
import re
from pathlib import Path

from natural_transfer_analyze import load_phase, make_freeze


def _hash(raw): return hashlib.sha256(raw).hexdigest()


def verify_before_evaluation(*, mapping_path, expected_mapping_sha256, prior,
                             prior_events_sha256, discovery_directory,
                             discovery_events_sha256, discovery_terminal_sha256,
                             discovery_binding, freeze_path, expected_freeze_sha256,
                             observed_public_commit, observed_public_blob,
                             width=768, vocabulary=50257):
    if (type(observed_public_commit) is not str or re.fullmatch('[0-9a-f]{40}', observed_public_commit) is None
            or type(observed_public_blob) is not str or re.fullmatch('[0-9a-f]{40}', observed_public_blob) is None):
        raise ValueError('immutable public freeze identifiers are required')
    mapping_bytes=Path(mapping_path).read_bytes()
    if _hash(mapping_bytes)!=expected_mapping_sha256 or discovery_binding.get('mapping_sha256')!=expected_mapping_sha256:
        raise ValueError('mapping bytes or discovery binding differ')
    mapping=json.loads(mapping_bytes)
    freeze_bytes=Path(freeze_path).read_bytes()
    if _hash(freeze_bytes)!=expected_freeze_sha256:
        raise ValueError('freeze SHA-256 differs')
    blob=hashlib.sha1(b'blob '+str(len(freeze_bytes)).encode()+b'\0'+freeze_bytes).hexdigest()
    if blob!=observed_public_blob:
        raise ValueError('freeze Git blob differs from external observation')
    rows=load_phase(discovery_directory,mapping,prior,'discovery',
                    events_sha256=discovery_events_sha256,terminal_sha256=discovery_terminal_sha256,
                    expected_binding=discovery_binding,width=width,vocabulary=vocabulary)
    frozen=make_freeze(rows,mapping,mapping_sha256=expected_mapping_sha256,
                       prior_events_sha256=prior_events_sha256,discovery_events_sha256=discovery_events_sha256)
    expected=(json.dumps(frozen,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if freeze_bytes!=expected:
        raise ValueError('freeze is not the saved discovery calculation')
    return {'freeze_sha256':expected_freeze_sha256, 'mapping_sha256':expected_mapping_sha256,
            'discovery_events_sha256':discovery_events_sha256, 'prior_events_sha256':prior_events_sha256,
            'observed_public_commit':observed_public_commit, 'observed_public_blob':observed_public_blob,
            'scope':'Public commit/blob and process chronology are external cooperative observations.'}
