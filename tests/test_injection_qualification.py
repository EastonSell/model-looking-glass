"""Fabricated scientific-result qualification; no model or held-out data."""
import hashlib
import json

import pytest

from looking_glass import injection_qualification as q


def raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def fixture(monkeypatch):
    supplement, deviation, replay_plan, replay_evidence = (
        b'supplement', b'deviation', b'fixed-plan', b'private-replay-evidence')
    discovery_runtime = raw({'schema': 'runtime', 'system_environment_sha256': '1' * 64,
                             'numerics': {'deterministic': True}})
    heldout_runtime = raw({'schema': 'runtime', 'system_environment_sha256': '2' * 64,
                           'numerics': {'deterministic': True}})
    for name, value in (('SUPPLEMENT_SHA', supplement), ('RUNTIME_DEVIATION_SHA', deviation),
                        ('DISCOVERY_RUNTIME_SHA', discovery_runtime), ('HELDOUT_RUNTIME_SHA', heldout_runtime),
                        ('REPLAY_PLAN_SHA', replay_plan)):
        monkeypatch.setattr(q, name, hashlib.sha256(value).hexdigest())
    op = {key: 'a' * 64 for key in q.OP_KEYS}
    op.update({'schema': 'mlg-injection-heldout-operational-binding-v3',
        'protocol_sha256': q.PROTOCOL_SHA, 'checkpoint_revision': q.CHECKPOINT,
        'discovery_source_commit': q.DISCOVERY_SOURCE, 'prediction_freeze_sha256': q.FREEZE_SHA,
        'runtime_sha256': q.DISCOVERY_RUNTIME_SHA, 'heldout_runtime_sha256': q.HELDOUT_RUNTIME_SHA,
        'prediction_sha256': q.PREDICTION_SHA, 'plan_sha256': q.DISCOVERY_PLAN_SHA,
        'site_commitment': q.SITE_COMMITMENT, 'supplement_sha256': q.SUPPLEMENT_SHA,
        'auditor_source_sha256': q.AUDITOR_SOURCE_SHA,
        'runtime_deviation_sha256': q.RUNTIME_DEVIATION_SHA, 'runtime_replay_plan_sha256': q.REPLAY_PLAN_SHA,
        'runtime_replay_source_sha256': q.REPLAY_SOURCE_SHA,
        'runtime_replay_claim_source_sha256': q.REPLAY_CLAIM_SOURCE_SHA,
        'runtime_replay_owner_source_sha256': q.REPLAY_OWNER_SOURCE_SHA,
        'preclaim_guard_source_sha256': q.PRECLAIM_GUARD_SOURCE_SHA,
        'runtime_replay_evidence_sha256': hashlib.sha256(replay_evidence).hexdigest(),
        'inventory_source_commit': 'b' * 40})
    operation = raw(op)
    binding = {'schema': 'mlg-injection-heldout-journal-binding-v1', 'source_commit': q.DISCOVERY_SOURCE,
        'runtime_sha256': q.DISCOVERY_RUNTIME_SHA, 'prediction_sha256': q.PREDICTION_SHA,
        'freeze_sha256': q.FREEZE_SHA, 'heldout_commitment': 'c' * 64,
        'site_commitment': q.SITE_COMMITMENT, 'inventory_sha256': op['inventory_sha256'],
        'operation_sha256': hashlib.sha256(operation).hexdigest(), 'gate_sha256': 'd' * 64,
        'disjoint_status': 'pending', 'disjoint_receipt_sha256': None}
    binding_raw = raw(binding)
    selection = {'first_changed': 3, 'ranking': [3, 2, 1, 4]}
    accuracy = {'correct': 3, 'incorrect': 5, 'ties': 1}
    sites = {str(i): {'predicted_effect': -0.125, 'effects': [-1.5, 0.0, 2.5],
                      'restoration': {'numerator': 1.0, 'denominator': 0.0, 'value': None}}
             for i in range(1, 5)}
    evaluation = {'schema': 'mlg-number-injection-evaluation-v1', 'status': 'complete',
        'protocol_sha256': q.PROTOCOL_SHA, 'checkpoint_revision': q.CHECKPOINT, 'execution_device': 'cpu',
        'private_controls': 'caller-attested', 'custody': 'caller-attested',
        'prediction_sha256': q.PREDICTION_SHA, 'prediction_freeze_sha256': q.FREEZE_SHA,
        'heldout_commitment_sha256': binding['heldout_commitment'], 'selection': selection,
        'clean_accuracy': accuracy, 'sites': sites, 'cases': [{'index': i} for i in range(8)]}
    evaluation_raw = raw(evaluation)
    audit = {'schema': 'mlg-independent-injection-heldout-audit-v1',
        'status': 'independent_saved_evidence_pass', 'records': 128,
        'binding_sha256': hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding_raw).hexdigest(),
        'heldout_commitment': binding['heldout_commitment'], 'site_commitment': q.SITE_COMMITMENT,
        'freeze_sha256': q.FREEZE_SHA, 'prediction_sha256': q.PREDICTION_SHA,
        'evaluation_sha256': hashlib.sha256(evaluation_raw).hexdigest(),
        'source_deviation': True,
        'selection': selection, 'clean_accuracy': accuracy, 'sites': sites}
    return {'evaluation_bytes': evaluation_raw, 'binding_bytes': binding_raw, 'audit_bytes': raw(audit),
            'operation_bytes': operation, 'source_supplement_bytes': supplement,
            'runtime_deviation_bytes': deviation, 'discovery_runtime_bytes': discovery_runtime,
            'heldout_runtime_bytes': heldout_runtime,
            'replay_plan_bytes': replay_plan, 'replay_evidence_bytes': replay_evidence}


def test_qualification_preserves_negative_and_undefined_results(monkeypatch):
    inputs = fixture(monkeypatch)
    report = json.loads(q.qualify_saved_evaluation(**inputs))
    assert report['status'] == 'private_saved_evidence_awaiting_execution_attestation'
    assert report['audit_source_sha256'] == q.AUDITOR_SOURCE_SHA
    assert report['runtime']['discovery_sha256'] == q.DISCOVERY_RUNTIME_SHA
    assert report['runtime']['heldout_sha256'] == q.HELDOUT_RUNTIME_SHA
    assert report['runtime']['same_runtime_rule_departed'] is True
    assert report['runtime']['recorded_manifest_difference'] == 'system_environment_sha256'
    assert report['source']['same_full_source_rule_departed'] is True
    assert report['cross_track']['journal_recorded_status'] == 'pending'
    assert report['evaluation'] == json.loads(inputs['evaluation_bytes'])
    assert report['evaluation']['sites']['1']['effects'][0] == -1.5
    assert report['evaluation']['sites']['1']['restoration']['value'] is None


def test_qualification_rejects_swapped_or_inconsistent_evidence(monkeypatch):
    inputs = fixture(monkeypatch)
    for key in ('runtime_deviation_bytes', 'discovery_runtime_bytes', 'heldout_runtime_bytes', 'replay_plan_bytes',
                'replay_evidence_bytes', 'source_supplement_bytes'):
        with pytest.raises(ValueError):
            q.qualify_saved_evaluation(**dict(inputs, **{key: b'wrong'}))
    altered = json.loads(inputs['audit_bytes'])
    altered['sites']['1']['effects'][0] = 99
    with pytest.raises(ValueError):
        q.qualify_saved_evaluation(**dict(inputs, audit_bytes=raw(altered)))
    altered = json.loads(inputs['binding_bytes'])
    altered['disjoint_status'] = 'clear'
    with pytest.raises(ValueError):
        q.qualify_saved_evaluation(**dict(inputs, binding_bytes=raw(altered)))
    with pytest.raises(ValueError):
        q.qualify_saved_evaluation(**dict(inputs, evaluation_bytes=b'{"status":"complete","status":"complete"}'))


def test_clear_receipt_identity_is_not_an_authorization(monkeypatch):
    inputs = fixture(monkeypatch)
    binding = json.loads(inputs['binding_bytes'])
    binding.update({'disjoint_status': 'clear', 'disjoint_receipt_sha256': 'e' * 64})
    binding_raw = raw(binding)
    audit = json.loads(inputs['audit_bytes'])
    audit['binding_sha256'] = hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding_raw).hexdigest()
    report = json.loads(q.qualify_saved_evaluation(**dict(inputs, binding_bytes=binding_raw, audit_bytes=raw(audit))))
    assert report['cross_track']['journal_recorded_status'] == 'clear'
    assert report['cross_track']['recorded_receipt_sha256'] == 'e' * 64
    assert 'independent verification' in report['cross_track']['qualification']
    assert 'every applicable' in report['cross_track']['qualification']
    assert report['status'] == 'private_saved_evidence_awaiting_execution_attestation'


@pytest.mark.parametrize('source_field', ['auditor_source_sha256', 'runtime_replay_source_sha256',
    'runtime_replay_claim_source_sha256', 'runtime_replay_owner_source_sha256', 'preclaim_guard_source_sha256'])
def test_swapped_reviewed_source_cannot_be_made_consistent_by_rehashing(monkeypatch, source_field):
    inputs = fixture(monkeypatch)
    operation = json.loads(inputs['operation_bytes'])
    operation[source_field] = 'f' * 64
    operation_raw = raw(operation)
    binding = json.loads(inputs['binding_bytes'])
    binding['operation_sha256'] = hashlib.sha256(operation_raw).hexdigest()
    binding_raw = raw(binding)
    audit = json.loads(inputs['audit_bytes'])
    audit['binding_sha256'] = hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding_raw).hexdigest()
    with pytest.raises(ValueError):
        q.qualify_saved_evaluation(**dict(inputs, operation_bytes=operation_raw, binding_bytes=binding_raw,
                                          audit_bytes=raw(audit)))


def test_accepts_exact_frozen_scorer_output_on_fabricated_cases(monkeypatch):
    from looking_glass.pretrained_number_selection import freeze_discovery, score_evaluation
    from tests.test_pretrained_number_selection import fabricated_cases, fabricated_evaluation
    prediction = freeze_discovery(fabricated_cases(), private_controls_verified=True, execution_device='cpu')
    monkeypatch.setattr(q, 'PREDICTION_SHA', hashlib.sha256(prediction).hexdigest())
    inputs = fixture(monkeypatch)
    binding = json.loads(inputs['binding_bytes'])
    evaluation = score_evaluation(fabricated_evaluation(), discovery_bytes=prediction,
        prediction_sha256=q.PREDICTION_SHA, prediction_freeze_sha256=q.FREEZE_SHA,
        heldout_commitment_sha256=binding['heldout_commitment'], custody_verified=True,
        private_controls_verified=True, execution_device='cpu')
    scored = json.loads(evaluation)
    audit = json.loads(inputs['audit_bytes'])
    audit.update({'evaluation_sha256': hashlib.sha256(evaluation).hexdigest(),
                  'selection': scored['selection'], 'clean_accuracy': scored['clean_accuracy'],
                  'sites': scored['sites']})
    report = json.loads(q.qualify_saved_evaluation(**dict(inputs, evaluation_bytes=evaluation,
                                                          audit_bytes=raw(audit))))
    assert report['evaluation'] == scored
    assert len(report['evaluation']['cases']) == 8
    assert set(report['evaluation']['sites']) == {'1', '2', '3', '4'}


def test_runtime_manifests_allow_only_the_disclosed_recorded_difference():
    original = {'schema': 'runtime', 'system_environment_sha256': '1' * 64,
                'numerics': {'deterministic': True}}
    observed = dict(original, system_environment_sha256='2' * 64)
    q._check_runtime_manifests(raw(original), raw(observed))
    for invalid in (dict(observed, numerics={'deterministic': False}),
                    dict(observed, system_environment_sha256='1' * 64),
                    dict(observed, system_environment_sha256='not-a-digest'),
                    {key: value for key, value in observed.items() if key != 'system_environment_sha256'}):
        with pytest.raises(ValueError):
            q._check_runtime_manifests(raw(original), raw(invalid))
