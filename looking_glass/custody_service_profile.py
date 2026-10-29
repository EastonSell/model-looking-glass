"""Versioned identity for the fixed custody-bound two-token service path.

This records the planned execution protocol in an independently observed
runtime identity. The collector and owner must separately establish that the
real measured and served requests follow that protocol under supervised
ownership. These helpers do not prove execution or bound GPU allocation.
"""
import hashlib
import json
from pathlib import Path

from .live_profile import _canonical_json_bytes, _guard_owner, _make_workload_keys, _plain_file_hash
from .resource_profile import ProfileRegistry
from .trusted_service_profile import (
    DIRECT_EXECUTION_PATH, JOB_COMMITMENT, LOAD_COMMITMENT, TrustedServiceIdentity,
    _decode_canonical, _resolve_service_profiles, build_trusted_service_identity,
)


RUNTIME_SCHEMA = 'trusted-custody-service-runtime-v1'
EXECUTION_PATH = 'looking_glass.owner_completion.LifetimeWorkspaceWorker.execute'
_PROTOCOL_DOCUMENT_SHA256 = '0b4cd53e7272a5afcd91807c36ad29ea39bee7a87d734f6da567445a4f568044'
_PROTOCOL_BYTES = _canonical_json_bytes({
    'schema': 'custody-forward-execution-v1',
    'protocol_document_sha256': _PROTOCOL_DOCUMENT_SHA256,
    'call_chain': [EXECUTION_PATH,
                   'looking_glass.resident_worker.ResidentWorker._execute_with_custody',
                   'looking_glass.resident_worker.ResidentWorker._execute',
                   'looking_glass.resident_worker.ResidentWorker._perform',
                   'looking_glass.resident_worker.ResidentWorker._perform_with_custody'],
    'rng_implementation': 'looking_glass.resident_worker.ResidentWorker._fork_owned_rng',
    'guarded_rng': True,
    'device_guard': 'looking_glass.owner_completion.OwnerCompletionAuthority._before_device',
    'result_custody': 'looking_glass.resident_worker._ResidentResultCustody',
    'cold_transfer': 'looking_glass.owner_completion.LifetimeWorkspaceWorker._finish_cold',
    'warm_transfer': 'looking_glass.owner_completion.LifetimeWorkspaceWorker._transfer_result',
    'resident_commitment_bytes': LOAD_COMMITMENT,
    'workspace_commitment_bytes': JOB_COMMITMENT,
    'max_gpu_trace_bytes': 0,
    'max_host_trace_bytes': 0,
    'max_concurrent_forwards': 1,
    'commitments_retained_until_process_exit': True,
})


def _verify_protocol_document(preflight):
    path = Path(preflight.repo_root) / 'docs' / 'CUSTODY_SERVICE_PROFILE_V1.md'
    if path.lstat().st_size > 1024 ** 2:
        raise ValueError('custody protocol document exceeds its size limit')
    record = _plain_file_hash(path)
    if record['size'] > 1024 ** 2 or record['sha256'] != _PROTOCOL_DOCUMENT_SHA256:
        raise ValueError('custody protocol document differs from the versioned policy')


def _validated_direct_runtime(direct):
    if type(direct) is not TrustedServiceIdentity:
        raise TypeError('a freshly observed trusted service identity is required')
    source = _decode_canonical(direct.source_manifest, 'trusted-service-source-v1')
    runtime = _decode_canonical(direct.runtime_manifest, 'trusted-service-runtime-v1')
    if (runtime.get('execution_path') != DIRECT_EXECUTION_PATH or 'custody_execution' in runtime
            or type(source.get('base')) is not dict or 'assets' not in source['base']
            or _canonical_json_bytes(source['base']['assets']) != direct.asset_manifest):
        raise ValueError('source, assets or direct execution identity differs')
    source_hash = hashlib.sha256(direct.source_manifest).hexdigest()
    if direct.keys != _make_workload_keys(source_hash, runtime):
        raise ValueError('observed runtime and workload keys differ')
    return runtime, source_hash


def build_custody_service_identity(preflight, identity, owner_session, device_lease):
    """Build custody keys before loading; actual worker binding is checked later."""
    direct = build_trusted_service_identity(preflight, identity, owner_session, device_lease)
    _guard_owner(identity, owner_session, device_lease)
    _verify_protocol_document(preflight)
    runtime, source_hash = _validated_direct_runtime(direct)
    runtime['schema'] = RUNTIME_SCHEMA
    runtime['execution_path'] = EXECUTION_PATH
    runtime['custody_execution'] = json.loads(_PROTOCOL_BYTES)
    keys = _make_workload_keys(source_hash, runtime)
    result = TrustedServiceIdentity(keys, direct.source_manifest, _canonical_json_bytes(runtime),
                                    direct.asset_manifest)
    _guard_owner(identity, owner_session, device_lease)
    return result


def resolve_custody_service_profiles(registry, identity, approved_profile_ids):
    """Resolve only separately pinned samples for this exact custody protocol."""
    if type(registry) is not ProfileRegistry or type(identity) is not TrustedServiceIdentity:
        raise TypeError('a trusted registry and independently built service identity are required')
    runtime = _decode_canonical(identity.runtime_manifest, RUNTIME_SCHEMA)
    if (runtime.get('execution_path') != EXECUTION_PATH
            or _canonical_json_bytes(runtime.get('custody_execution')) != _PROTOCOL_BYTES):
        raise ValueError('custody execution protocol differs')
    return _resolve_service_profiles(registry, identity, approved_profile_ids,
                                     RUNTIME_SCHEMA, EXECUTION_PATH)
