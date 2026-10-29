"""Model-free closed-directory integration and failure controls."""
import copy
import hashlib
import json

import pytest

from looking_glass import agreement_science_saved_v1 as saved
from looking_glass import agreement_science_v1 as science


def _zero_candidate(path):
    """Make a complete synthetic zero-dose journal; no model or public rows."""
    root = path / 'attempt'
    tensors = root / 'tensors'
    tensors.mkdir(parents=True)
    cases = [{'case_id': f't{f}-{v}', 'role': 'target', 'family_index': f,
              'variant': v, 'subject_number': 'singular' if v % 2 == 0 else 'plural',
              'morphology': 'regular' if v < 2 else 'irregular',
              'attractor_number': 'singular' if f % 2 == 0 else 'plural',
              'token_ids': [1, 10 + v, 2, 3, 4]}
             for f in range(8) for v in range(4)]
    cases += [{'case_id': f'c{f}', 'role': 'context', 'family_index': f,
               'subject_number': None, 'token_ids': [1, 3, 4]} for f in range(8)]
    by_id = {case['case_id']: case for case in cases}
    directions = {'u': (1.0, 0.0, 0.0), 'v1': (0.0, 1.0, 0.0), 'v2': (0.0, 0.0, 1.0)}
    events, sizes = [], {}

    def reference(shape):
        count = 1
        for dimension in shape:
            count *= dimension
        raw = bytes(count * 4)
        digest = hashlib.sha256(raw).hexdigest()
        (tensors / (digest + '.f32le')).write_bytes(raw)
        sizes[digest] = len(raw)
        return {'sha256': digest, 'shape': list(shape), 'dtype': 'float32-le', 'bytes': len(raw)}

    def add(event, refs=None):
        events.append({'index': len(events), 'event': event, 'tensors': refs or {}})

    def forward(item):
        case = by_id[item['case_id']]
        action = item['action']
        event = {**item, 'kind': 'forward', 'input_ids': case['token_ids'],
                 'attention_mask': [1] * len(case['token_ids']),
                 'all_sequence_logits_finite': True, 'dose': None, 'plan_metadata': {'steps': []}}
        refs = {'last_logits': reference((548,))}
        if action not in ('clean', 'clean_repeat', 'clean_after'):
            site, plan = science.expected_hook_plan(action, sequence_length=len(case['token_ids']), width=3)
            event.update(site=site, donor=None, plan_metadata=plan)
            refs.update(before=reference((3,)), after=reference((3,)),
                        whole_before=reference((1, len(case['token_ids']), 3)),
                        whole_after=reference((1, len(case['token_ids']), 3)))
            if not action.startswith('noop_'):
                dose = {'requested_norm': 0.0, 'realized_norm': 0.0, 'dose_valid': True,
                        'relative_error': None, 'source_norm': 0.0, 'realized_over_source_norm': None,
                        'changed_elements': 0, 'changed_words': 0, 'changed_bit_count': 0}
                if action in ('same_diff', 'opp_same', 'opp_diff'):
                    other = case['variant'] ^ {'same_diff': 2, 'opp_same': 1, 'opp_diff': 3}[action]
                    event['donor'] = cases[4 * case['family_index'] + other]['case_id']
                    dose.update(donor_distance=0.0, fraction=None)
                else:
                    dose['signed_amount'] = 0.0
                event['dose'] = dose
        add(event, refs)

    state_hash = 'a' * 64
    add({'kind': 'run_contract', 'expected_forwards': 784, 'seed': 20261030,
         'external_binding': None, 'model_state_sha256': state_hash})
    schedule = science.expected_schedule(cases)
    for family in range(8):
        group = schedule[family * 98:(family + 1) * 98]
        for item in group[:20]:
            forward(item)
        add({'kind': 'family_geometry', 'family_index': family,
             'distances': [[0.0] * 3 for _ in range(4)], 'doses': [0.0] * 4,
             'centering': [0.0] * 4, 'centering_gaps': [0.0] * 2})
        for item in group[20:88]:
            forward(item)
        for variant in (0, 2):
            add({'kind': 'centering_result', 'family_index': family, 'singular_variant': variant,
                 'achieved_u_gaps': {'center_u': 0.0, 'center_v1': 0.0, 'center_v2': 0.0}})
        for item in group[88:]:
            forward(item)

    def save_events(rows):
        raw = b''.join((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode()
                       for row in rows)
        (root / 'events.jsonl').write_bytes(raw)
        size = sum(sizes.values())
        terminal = {'status': 'complete_candidate', 'events': 809, 'event_log_bytes': len(raw),
                    'retained_tensor_bytes': size, 'storage_accounting_complete': True,
                    'detail': {'status': 'complete_candidate', 'forwards': 784,
                               'model_state_sha256_before_after': state_hash, 'retained_tensor_bytes': size}}
        (root / 'terminal.json').write_text(json.dumps(terminal), encoding='utf-8')

    (root / 'started.json').write_text('{"scope":"synthetic"}', encoding='utf-8')
    save_events(events)
    return root, cases, directions, events, save_events


def test_complete_synthetic_directory_and_tampering(tmp_path):
    root, cases, directions, events, save_events = _zero_candidate(tmp_path)

    def audit():
        return saved.audit_attempt(root, cases, directions, expected_binding=None,
                                   expected_source_revision=None, width=3, vocabulary=548,
                                   allow_synthetic=True)

    report = audit()
    assert report['journal']['status'] == 'consistent_candidate'
    assert report['audit_mode'] == 'synthetic_test'
    assert len(report['targets']) == 32 and len(report['contexts']) == 8
    assert len(report['family_geometry']) == 8 and len(report['centering_achieved']) == 16
    assert len(report['deliveries']['t0-0']) == 16
    assert report['deliveries']['t0-0']['same_diff']['metrics']['realized_norm'] == 0.0
    assert report['targets'][0]['responses']['u']['present']['odd_per_unit'] is None
    assert not report['signed_prediction']['passed'] and not report['projection_prediction']['passed']

    changed = copy.deepcopy(events)
    arm = next(row for row in changed if row['event'].get('action') == 'same_diff')
    arm['event']['donor'] = 'wrong'
    save_events(changed)
    with pytest.raises(ValueError, match='donor'):
        audit()

    changed = copy.deepcopy(events)
    centering = next(row for row in changed if row['event'].get('kind') == 'centering_result')
    centering['event']['status'] = 'failure'
    save_events(changed)
    with pytest.raises(ValueError, match='centering'):
        audit()

    save_events(events)
    (root / 'tensors' / 'extra.f32le').write_bytes(b'')
    with pytest.raises(ValueError, match='inventory'):
        audit()
