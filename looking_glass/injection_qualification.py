"""Qualify a private saved injection evaluation without changing its scores.

The inputs are saved evidence, not proof of process exit, publication, actual
runtime, cross-track clearance or successful replay. An independently trusted
caller must establish those facts before treating this record as a result.
"""
import hashlib
import json
import re


PROTOCOL_SHA = '95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee'
CHECKPOINT = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
DISCOVERY_SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
DISCOVERY_RUNTIME_SHA = 'b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e'
HELDOUT_RUNTIME_SHA = '7a967712a896d19ccfe4164c11a57e3ab1c65ece1a0ae26fedb220b9fc80e327'
PREDICTION_SHA = '11376d8aae517db76d6a33c1ae5dc1f4b0c747a9d5216c9b02a17b1b0b300107'
FREEZE_SHA = 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'
DISCOVERY_PLAN_SHA = 'e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8'
SITE_COMMITMENT = 'c3739cb173d3c1bc3e3adda70025ff7f26a73c4690e50ecf5baf06ff8b359b9d'
SUPPLEMENT_SHA = '2ccd3c6515e18953e960b6d5fcdb87abafd177058bd3e06a6f855c7fd86a8834'
RUNTIME_DEVIATION_SHA = '6b333232d66caa348211c932d4b4244070651e625b0a0c0d7686551b23fd895b'
REPLAY_PLAN_SHA = '051f8263c0c2630fb6438b28dfebc9d8c4da644f727d4938bfa0d3904117512c'
REPLAY_SOURCE_SHA = '6bf925af465f4761cde6d32c3454c6d0810b089af3d17ca914dff676ea27e8cd'
REPLAY_CLAIM_SOURCE_SHA = '7c6ea44652a9aae3b66159308f8da49dc60c64bb0f40d5e3295fb2f5b2631b54'
REPLAY_OWNER_SOURCE_SHA = '2370697df2f30a86abf3b774935a0978687a45fe81d6e0cab79214e326c5fbb2'
PRECLAIM_GUARD_SOURCE_SHA = '425a8b70c4512ef99bdc840a6095f536c9d7e0df18edee0ff2db65d729d0a320'
AUDITOR_SOURCE_SHA = '67bf75fc4b96f3bb39cf7d8a0dec155db57e73a0e71330b21c2ea2c3920b8d65'
OP_KEYS = frozenset({'schema', 'protocol_sha256', 'checkpoint_revision', 'discovery_source_commit',
    'prediction_freeze_sha256', 'runtime_sha256', 'prediction_sha256', 'plan_sha256', 'site_commitment',
    'inventory_sha256', 'custody_source_sha256', 'executor_source_sha256', 'comparator_source_sha256',
    'ledger_source_sha256', 'tokenizer_sha256', 'supplement_sha256', 'owner_source_sha256',
    'journal_source_sha256', 'supervisor_source_sha256', 'auditor_source_sha256', 'gate_source_sha256',
    'inventory_source_sha256', 'inventory_source_commit', 'accounting_amendment_sha256', 'lease_source_sha256',
    'preparation_child_source_sha256', 'preparation_audit_source_sha256', 'outer_observer_source_sha256',
    'gate_allocation_caller_source_sha256', 'gate_transition_caller_source_sha256', 'runtime_deviation_sha256',
    'heldout_runtime_sha256', 'runtime_replay_plan_sha256', 'runtime_replay_source_sha256',
    'runtime_replay_claim_source_sha256', 'runtime_replay_owner_source_sha256', 'preclaim_guard_source_sha256',
    'phase_accounting_source_sha256', 'runtime_replay_evidence_sha256'})
_BINDING_KEYS = frozenset({'schema', 'source_commit', 'runtime_sha256', 'prediction_sha256', 'freeze_sha256',
    'heldout_commitment', 'site_commitment', 'inventory_sha256', 'operation_sha256', 'gate_sha256',
    'disjoint_status', 'disjoint_receipt_sha256'})
_EVALUATION_KEYS = frozenset({'schema', 'status', 'protocol_sha256', 'checkpoint_revision', 'execution_device',
    'private_controls', 'custody', 'prediction_sha256', 'prediction_freeze_sha256', 'heldout_commitment_sha256',
    'selection', 'clean_accuracy', 'sites', 'cases'})
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_HEX40 = re.compile(r'[0-9a-f]{40}\Z')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _record(raw, limit):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError('a bounded saved record is required')
    try:
        value = json.loads(raw, object_pairs_hook=_unique)
        if type(value) is not dict or _canonical(value) != raw:
            raise ValueError('a canonical saved object is required')
        return value
    except (UnicodeError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError('saved evidence is malformed') from error


def _digest(raw, expected=None):
    if type(raw) is not bytes or not 0 < len(raw) <= 4 * 1024 * 1024:
        raise ValueError('bounded evidence bytes are required')
    result = hashlib.sha256(raw).hexdigest()
    if expected is not None and result != expected:
        raise ValueError('the fixed evidence digest differs')
    return result


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _check_runtime_manifests(discovery_bytes, heldout_bytes):
    discovery = _record(discovery_bytes, 1024 * 1024)
    heldout = _record(heldout_bytes, 1024 * 1024)
    old_system = discovery.pop('system_environment_sha256', None)
    new_system = heldout.pop('system_environment_sha256', None)
    if (type(old_system) is not str or _HEX64.fullmatch(old_system) is None
            or type(new_system) is not str or _HEX64.fullmatch(new_system) is None
            or old_system == new_system or not _same(discovery, heldout)):
        raise ValueError('the recorded runtime difference exceeds the disclosed departure')


def qualify_saved_evaluation(*, evaluation_bytes, binding_bytes, audit_bytes, operation_bytes,
                             source_supplement_bytes, runtime_deviation_bytes, discovery_runtime_bytes,
                             heldout_runtime_bytes,
                             replay_plan_bytes, replay_evidence_bytes):
    """Return a private, explicitly provisional qualification of saved bytes.

    The separately sourced auditor's record is checked for consistency only.
    Its authenticity, the process exits, the actual runtime and compatibility
    replay, publication and any cross-track receipt remain external duties.
    """
    evaluation = _record(evaluation_bytes, 4 * 1024 * 1024)
    binding = _record(binding_bytes, 65536)
    audit = _record(audit_bytes, 1024 * 1024)
    operation = _record(operation_bytes, 65536)
    operation_sha = _digest(operation_bytes)
    replay_evidence_sha = _digest(replay_evidence_bytes)
    _digest(source_supplement_bytes, SUPPLEMENT_SHA)
    _digest(runtime_deviation_bytes, RUNTIME_DEVIATION_SHA)
    _digest(discovery_runtime_bytes, DISCOVERY_RUNTIME_SHA)
    _digest(heldout_runtime_bytes, HELDOUT_RUNTIME_SHA)
    _check_runtime_manifests(discovery_runtime_bytes, heldout_runtime_bytes)
    _digest(replay_plan_bytes, REPLAY_PLAN_SHA)
    fixed_operation = {'schema': 'mlg-injection-heldout-operational-binding-v3',
        'protocol_sha256': PROTOCOL_SHA, 'checkpoint_revision': CHECKPOINT,
        'discovery_source_commit': DISCOVERY_SOURCE, 'prediction_freeze_sha256': FREEZE_SHA,
        'runtime_sha256': DISCOVERY_RUNTIME_SHA, 'heldout_runtime_sha256': HELDOUT_RUNTIME_SHA,
        'prediction_sha256': PREDICTION_SHA, 'plan_sha256': DISCOVERY_PLAN_SHA,
        'site_commitment': SITE_COMMITMENT, 'supplement_sha256': SUPPLEMENT_SHA,
        'auditor_source_sha256': AUDITOR_SOURCE_SHA,
        'runtime_deviation_sha256': RUNTIME_DEVIATION_SHA, 'runtime_replay_plan_sha256': REPLAY_PLAN_SHA,
        'runtime_replay_source_sha256': REPLAY_SOURCE_SHA,
        'runtime_replay_claim_source_sha256': REPLAY_CLAIM_SOURCE_SHA,
        'runtime_replay_owner_source_sha256': REPLAY_OWNER_SOURCE_SHA,
        'preclaim_guard_source_sha256': PRECLAIM_GUARD_SOURCE_SHA,
        'runtime_replay_evidence_sha256': replay_evidence_sha}
    if (set(operation) != OP_KEYS or any(operation.get(k) != v for k, v in fixed_operation.items())
            or type(operation.get('inventory_source_commit')) is not str
            or _HEX40.fullmatch(operation['inventory_source_commit']) is None
            or any(type(operation[k]) is not str or _HEX64.fullmatch(operation[k]) is None
                   for k in OP_KEYS - set(fixed_operation) - {'inventory_source_commit'})):
        raise ValueError('the operational binding differs')
    binding_fixed = {'schema': 'mlg-injection-heldout-journal-binding-v1', 'source_commit': DISCOVERY_SOURCE,
        'runtime_sha256': DISCOVERY_RUNTIME_SHA, 'prediction_sha256': PREDICTION_SHA,
        'freeze_sha256': FREEZE_SHA, 'site_commitment': SITE_COMMITMENT,
        'inventory_sha256': operation['inventory_sha256'], 'operation_sha256': operation_sha}
    if (set(binding) != _BINDING_KEYS or any(binding.get(k) != v for k, v in binding_fixed.items())
            or any(type(binding.get(k)) is not str or _HEX64.fullmatch(binding[k]) is None
                   for k in ('heldout_commitment', 'gate_sha256'))
            or type(binding.get('disjoint_status')) is not str
            or binding['disjoint_status'] not in ('pending', 'clear')
            or (binding['disjoint_status'] == 'pending' and binding['disjoint_receipt_sha256'] is not None)
            or (binding['disjoint_status'] == 'clear' and (type(binding['disjoint_receipt_sha256']) is not str
                or _HEX64.fullmatch(binding['disjoint_receipt_sha256']) is None))):
        raise ValueError('the saved split or gate binding differs')
    evaluation_fixed = {'schema': 'mlg-number-injection-evaluation-v1', 'status': 'complete',
        'protocol_sha256': PROTOCOL_SHA, 'checkpoint_revision': CHECKPOINT, 'execution_device': 'cpu',
        'private_controls': 'caller-attested', 'custody': 'caller-attested',
        'prediction_sha256': PREDICTION_SHA, 'prediction_freeze_sha256': FREEZE_SHA,
        'heldout_commitment_sha256': binding['heldout_commitment']}
    if (set(evaluation) != _EVALUATION_KEYS or any(evaluation.get(k) != v for k, v in evaluation_fixed.items())
            or type(evaluation.get('selection')) is not dict or type(evaluation.get('clean_accuracy')) is not dict
            or type(evaluation.get('sites')) is not dict or set(evaluation['sites']) != {'1', '2', '3', '4'}
            or type(evaluation.get('cases')) is not list or len(evaluation['cases']) != 8
            or any(type(case) is not dict or type(case.get('index')) is not int or case['index'] != index
                   for index, case in enumerate(evaluation['cases']))):
        raise ValueError('the frozen-score evaluation differs')
    binding_sha = hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding_bytes).hexdigest()
    evaluation_sha = _digest(evaluation_bytes)
    audit_fixed = {'schema': 'mlg-independent-injection-heldout-audit-v1',
        'status': 'independent_saved_evidence_pass', 'binding_sha256': binding_sha,
        'heldout_commitment': binding['heldout_commitment'], 'site_commitment': SITE_COMMITMENT,
        'freeze_sha256': FREEZE_SHA, 'prediction_sha256': PREDICTION_SHA, 'evaluation_sha256': evaluation_sha}
    if (any(audit.get(k) != v for k, v in audit_fixed.items()) or type(audit.get('records')) is not int
            or audit['records'] != 128 or audit.get('source_deviation') is not True
            or any(not _same(audit.get(k), evaluation[k]) for k in ('selection', 'clean_accuracy', 'sites'))):
        raise ValueError('the independent saved-evidence audit differs')
    report = {'schema': 'mlg-injection-qualified-private-result-v1',
        'status': 'private_saved_evidence_awaiting_execution_attestation',
        'prediction_sha256': PREDICTION_SHA, 'prediction_freeze_sha256': FREEZE_SHA,
        'heldout_commitment_sha256': binding['heldout_commitment'], 'operation_sha256': operation_sha,
        'binding_sha256': binding_sha, 'evaluation_sha256': evaluation_sha, 'audit_sha256': _digest(audit_bytes),
        'audit_source_sha256': AUDITOR_SOURCE_SHA,
        'source': {'discovery_commit': DISCOVERY_SOURCE, 'same_full_source_rule_departed': True,
                   'supplement_sha256': SUPPLEMENT_SHA},
        'runtime': {'discovery_sha256': DISCOVERY_RUNTIME_SHA, 'heldout_sha256': HELDOUT_RUNTIME_SHA,
            'same_runtime_rule_departed': True, 'deviation_sha256': RUNTIME_DEVIATION_SHA,
            'recorded_manifest_difference': 'system_environment_sha256',
            'public_case_replay_plan_sha256': REPLAY_PLAN_SHA,
            'public_case_replay_source_sha256': REPLAY_SOURCE_SHA,
            'public_case_replay_evidence_sha256': replay_evidence_sha,
            'limitation': 'The fixed public-case check does not establish runtime equivalence.'},
        'cross_track': {'journal_recorded_status': binding['disjoint_status'],
                        'recorded_receipt_sha256': binding['disjoint_receipt_sha256'],
                        'qualification': 'Receipt identity, coverage of every applicable track, and forward authorization require independent verification.'},
        'evaluation': evaluation,
        'limitations': ['Private saved evidence only; execution, process exits, publication and custody remain external.',
                        'The compatibility replay and actual runtime observations require independent verification.']}
    return _canonical(report)
