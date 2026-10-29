"""Controller contracts; no pretrained or held-out inputs are run here."""

import hashlib
import json
from pathlib import Path
import struct
import os
import subprocess
import sys

import pytest


def _controller():
    path = Path(__file__).resolve().parents[1] / 'looking_glass' / 'natural_agreement_controller.py'
    assert path.is_file(), 'the discovery controller is missing'
    from looking_glass import natural_agreement_controller
    return natural_agreement_controller


def _write_candidate(directory):
    """A small journal exercises storage independently of study semantics."""
    directory.mkdir()
    (directory / 'tensors').mkdir()
    (directory / 'started.json').write_text('{}\n', encoding='utf-8')
    payload = struct.pack('<ff', 1.0, -2.0)
    digest = hashlib.sha256(payload).hexdigest()
    (directory / 'tensors' / (digest + '.f32le')).write_bytes(payload)
    event = {'index': 0, 'event': {'kind': 'example'},
             'tensors': {'values': {'sha256': digest, 'shape': [2],
                                    'dtype': 'float32-le', 'bytes': 8}}}
    line = (json.dumps(event, sort_keys=True, separators=(',', ':')) + '\n').encode()
    (directory / 'events.jsonl').write_bytes(line)
    terminal = {'status': 'complete_candidate', 'detail': {}, 'events': 1,
                'retained_tensor_bytes': 8, 'event_log_bytes': len(line),
                'storage_accounting_complete': True}
    (directory / 'terminal.json').write_text(json.dumps(terminal) + '\n', encoding='utf-8')
    return digest


def test_inventory_rejects_changed_referenced_tensor(tmp_path):
    control = _controller()
    attempt = tmp_path / 'attempt'
    digest = _write_candidate(attempt)
    (attempt / 'tensors' / (digest + '.f32le')).write_bytes(struct.pack('<ff', 1.0, 3.0))
    with pytest.raises(control.InventoryError):
        control.audit_storage(attempt)


def test_inventory_rejects_orphan_tensor(tmp_path):
    control = _controller()
    attempt = tmp_path / 'attempt'
    _write_candidate(attempt)
    (attempt / 'tensors' / ('0' * 64 + '.f32le')).write_bytes(b'junk')
    with pytest.raises(control.InventoryError):
        control.audit_storage(attempt)


def test_nonzero_child_exit_never_becomes_success(tmp_path):
    control = _controller()
    attempt = tmp_path / 'attempt'
    _write_candidate(attempt)
    verdict = control.audit_result(attempt, exit_code=7, reaped=True)
    assert verdict['status'] == 'incomplete'
    assert verdict['exit_code'] == 7
    assert (attempt / 'terminal.json').is_file()


def test_missing_or_unreaped_child_never_becomes_success(tmp_path):
    control = _controller()
    attempt = tmp_path / 'attempt'
    _write_candidate(attempt)
    verdict = control.audit_result(attempt, exit_code=0, reaped=False)
    assert verdict['status'] == 'incomplete'
    assert verdict['child_reaped'] is False


def test_storage_audit_reports_intact_candidate_without_claiming_study_completion(tmp_path):
    control = _controller()
    attempt = tmp_path / 'attempt'
    digest = _write_candidate(attempt)
    inventory = control.audit_storage(attempt)
    assert inventory['events'] == 1
    assert inventory['retained_tensor_bytes'] == 8
    assert inventory['tensor_sha256'] == [digest]
    assert control.audit_result(attempt, exit_code=0, reaped=True)['status'] == 'incomplete'


@pytest.mark.parametrize('altered', [True, 1.0])
def test_start_binding_requires_exact_json_types_and_digest(altered):
    control = _controller()
    binding = {'runtime': {'cpu_threads': 1},
               'study_budget': {'phase_reservations_bytes': {'natural_discovery': 1024 * 1024}}}
    raw = (json.dumps(binding, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode()
    expected = {'schema': 'mlg-natural-cpu-attempt-v1',
                'binding_sha256': hashlib.sha256(raw).hexdigest(), 'binding': binding}
    control.check_start(expected, expected)
    changed = json.loads(json.dumps(expected))
    changed['binding']['runtime']['cpu_threads'] = altered
    with pytest.raises(control.InventoryError):
        control.check_start(changed, expected)


def _complete_records():
    cases = []
    for family in range(4):
        for row in range(4):
            cases.append({'case_id': f'c{family * 4 + row:02}', 'quartet': f'q{family + 1}',
                          'lemma': f'l{family * 2 + row // 2}',
                          'number': 'singular' if row % 2 == 0 else 'plural',
                          'input_ids': [464, 100 + family * 4 + row, family + 10, 262, 88],
                          'attention_mask': [1, 1, 1, 1, 1], 'subject_index': 1})
    sites = [f'{layer}:{component}:last' for layer in range(1, 5)
             for component in ('attn', 'mlp')]
    all_sites = sites + ['5:block:last', '5:block:subject']
    records = []
    def site_data(site):
        layer, component, role = site.split(':')
        path = f'transformer.h.{layer}' + ('' if component == 'block' else f'.{component}')
        return {'id': site, 'layer': int(layer), 'component': component, 'token_role': role,
                'module_path': path, 'output_selector': [0] if component in ('attn', 'block') else []}
    def step(name, operation, site, order, case, *, whole=False, preceding=(), source_capture=None):
        metadata = site_data(site)
        return {'name': name, 'operation': operation, 'order': order,
                'module_path': metadata['module_path'], 'module_aliases': [metadata['module_path']],
                'invocation': 0, 'forward_index': 0, 'call_site_known': False,
                'batch_index': None if whole else 0,
                'token_index': None if whole else (case['subject_index'] if metadata['token_role'] == 'subject' else 4),
                'output_selector': metadata['output_selector'], 'output_shape': [1, 5, 768],
                'selected_shape': [1, 5, 768] if whole else [768], 'dtype': 'torch.float32',
                'device': 'cpu', 'preceding_same_site_actions': list(preceding), 'source_capture': source_capture}
    def capture_plan(case):
        steps = [step(f'capture_{i}', 'capture', s, i, case,
                      preceding=('capture_8',) if i == 9 else ()) for i, s in enumerate(all_sites)]
        return {'steps': steps, 'capture_bytes': 10 * 768 * 4, 'source_bytes': 0}
    def action_plan(case, site, operation, self_patch=False):
        names = ('whole_before', 'before', 'action', 'after', 'whole_after')
        steps = [step(name, operation if name == 'action' else 'capture', site, i, case,
                      whole=name.startswith('whole_'), preceding=names[:i],
                      source_capture='before' if name == 'action' and self_patch else None)
                 for i, name in enumerate(names)]
        return {'steps': steps, 'capture_bytes': (2 * 5 + 2) * 768 * 4,
                'source_bytes': 0 if operation == 'null' or self_patch else 768 * 4}
    def ref(identity, shape):
        return {'sha256': hashlib.sha256(identity.encode()).hexdigest(), 'shape': shape,
                'dtype': 'float32-le', 'bytes': 4 * __import__('math').prod(shape)}
    def captures(case):
        return {f'capture_{i}': ref(f'{case}-capture-{i}', [768]) for i in range(10)}
    def input_sha(case):
        raw = json.dumps({'ids': case['input_ids'], 'subject_index': case['subject_index']},
                         sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()
    def add(event, tensors=None):
        records.append({'index': len(records), 'event': event, 'tensors': tensors or {}})
    config = [[['deterministic_algorithms', True], ['deterministic_warn_only', False],
               ['cpu_threads', 1], ['cpu_interop_threads', 1], ['float32_matmul_precision', 'highest'],
               ['calling_thread_denormal_product_bits', 4194304]], [['', [], []]]]
    config_sha = hashlib.sha256(json.dumps(config, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()
    add({'kind': 'run_contract', 'cpu_rng_seed': 20261028, 'cpu_threads': 1,
         'deterministic_algorithms': True, 'cases': cases, 'sites': [site_data(s) for s in all_sites],
         'seeded_cpu_rng_sha256': 'a' * 64, 'execution_configuration': config,
         'execution_configuration_sha256': config_sha})
    for family in range(4):
        quartet = f'q{family + 1}'
        members = cases[family * 4:family * 4 + 4]
        for kind in ('clean_capture', 'clean_repeat', 'null'):
            for case in members:
                event = {'kind': kind, 'quartet': quartet, 'case': case['case_id'],
                         'target': case['case_id'], 'logits_check': {'within_tolerance': True},
                         'untouched_output_bits_equal': True, 'input_sha256': input_sha(case)}
                if kind == 'clean_repeat':
                    event['checks'] = {s: {'within_tolerance': True} for s in ['last_logits'] + all_sites}
                if kind in ('clean_capture', 'clean_repeat'):
                    event['plan_metadata'] = capture_plan(case)
                if kind == 'null':
                    event['checks'] = {s: {'within_tolerance': True} for s in ('activation', 'logits')}
                    event['site'] = site_data(sites[0])
                    event['plan_metadata'] = action_plan(case, sites[0], 'null')
                tensors = {'last_logits': ref('logits', [50257])}
                if kind in ('clean_capture', 'clean_repeat'):
                    tensors.update(captures(case['case_id']))
                if kind == 'null':
                    before = ref('null', [768])
                    tensors.update({'before': before, 'after': before,
                                    'whole_before': ref('whole-before', [1, 5, 768]),
                                    'whole_after': ref('whole-after', [1, 5, 768])})
                add(event, tensors)
        for site in sites:
            for target in members:
                for donor in members:
                    source = (ref('self', [768]) if target == donor
                              else captures(donor['case_id'])[f'capture_{sites.index(site)}'])
                    add({'kind': 'patch', 'quartet': quartet, 'target': target['case_id'],
                         'donor': donor['case_id'], 'site': site_data(site),
                         'source_input_sha256': input_sha(donor), 'target_input_sha256': input_sha(target),
                         'plan_metadata': action_plan(target, site, 'patch', target == donor),
                         'self_patch': target == donor,
                         'source_tensor_sha256': source['sha256'],
                         'post_action_matches_donor_bits': True, 'untouched_output_bits_equal': True,
                         'checks': ({'activation': {'within_tolerance': True},
                                     'logits': {'within_tolerance': True}} if target == donor else {})},
                        {'before': source, 'after': source,
                         'whole_before': ref('whole-before', [1, 5, 768]),
                         'whole_after': ref('whole-after', [1, 5, 768]),
                         ('last_logits' if target == donor else 'contrast_logits'):
                         ref('logits', [50257] if target == donor else [4])})
        for case in members:
            for kind in ('delivery', 'subject_position'):
                donor = next(item for item in members if item['lemma'] == case['lemma']
                             and item['number'] != case['number'])
                site = '5:block:last' if kind == 'delivery' else '5:block:subject'
                source = captures(donor['case_id'])[f'capture_{all_sites.index(site)}']
                add({'kind': kind, 'quartet': quartet, 'target': case['case_id'],
                     'donor': donor['case_id'], 'site': site_data(site),
                     'plan_metadata': action_plan(case, site, 'patch'),
                     'reference': donor['case_id'] if kind == 'delivery' else case['case_id'],
                     'post_action_matches_donor_bits': True, 'untouched_output_bits_equal': True,
                     'logits_check': {'within_tolerance': True}},
                    {'before': ref('before', [768]), 'after': source,
                     'whole_before': ref('whole-before', [1, 5, 768]),
                     'whole_after': ref('whole-after', [1, 5, 768]),
                     'last_logits': ref('logits', [50257])})
        for case in members:
            add({'kind': 'final_clean', 'quartet': quartet, 'case': case['case_id'],
                 'logits_check': {'within_tolerance': True}}, {'last_logits': ref('logits', [50257])})
    for kind in ('present_site_metrics', 'secondary_past_site_metrics'):
        for site in sites:
            pairs = [{'site': site, 'quartet': target['quartet'], 'target': target['case_id'],
                      'donor': donor['case_id'], 'cell': 'same_lemma_same_number', 'weight': 1,
                      'clean_margin': 0.0, 'patched_margin': 0.0, 'signed_target_shift': 0.0,
                      'clean_margin_sign': 0, 'patched_margin_sign': 0,
                      'strict_sign_reversal': False, 'zero_transition': False}
                     for target in cases for donor in cases if target['quartet'] == donor['quartet']]
            for pair in pairs:
                if kind == 'present_site_metrics':
                    pair.update({'clean_is': 0.0, 'clean_are': 0.0, 'patched_is': 0.0, 'patched_are': 0.0})
                else:
                    pair.update({'clean_was': 0.0, 'clean_were': 0.0, 'patched_was': 0.0, 'patched_were': 0.0})
            metrics = {'baseline': {'n': 16, 'correct': 16, 'incorrect': 0, 'ties': 0},
                       'site_summary': {'site': site, 'n_quartets': 4,
                                        'mean_number_main_contrast': 0.0,
                                        'mean_lemma_number_interaction': 0.0},
                       'quartets': [{'site': site, 'quartet': f'q{i}',
                                     'cell_means': {c: 0.0 for c in ('same_lemma_same_number', 'same_lemma_opposite_number',
                                                                      'different_lemma_same_number', 'different_lemma_opposite_number')},
                                     'cell_counts': {c: 4 for c in ('same_lemma_same_number', 'same_lemma_opposite_number',
                                                                    'different_lemma_same_number', 'different_lemma_opposite_number')},
                                     'number_main_contrast': 0.0,
                                     'lemma_number_interaction': 0.0} for i in range(1, 5)],
                       'pairs': pairs}
            contrast = ({'first_token': ' is', 'first_id': 318, 'second_token': ' are', 'second_id': 389}
                        if kind == 'present_site_metrics' else
                        {'first_token': ' was', 'first_id': 373, 'second_token': ' were', 'second_id': 547})
            add({'kind': kind, 'site': site, 'metrics': metrics, 'contrast': contrast})
    magnitude_pairs = [{'site': site, 'quartet': case['quartet'], 'lemma': case['lemma'],
                        'singular': case['case_id'], 'plural': cases[i + 1]['case_id'], 'l2': 0.0}
                       for site in sites for i, case in enumerate(cases) if i % 2 == 0]
    random_index = hashlib.sha256(b'MLG-number-v1-random\x00' + (20261028).to_bytes(8, 'big')).digest()[0] % 8
    add({'kind': 'summary', 'result': {'status': 'complete', 'forwards': 608,
                                       'model_state_unchanged': True, 'model_state_sha256': 'b' * 64,
                                       'cpu_rng_seed': 20261028, 'seeded_cpu_rng_sha256': 'a' * 64,
                                       'controls': {'null': 16, 'self': 128, 'delivery': 16,
                                                    'subject_position': 16, 'repeat': 16,
                                                    'final_clean': 16},
                                       'selection': {'ranking': sites, 'predictions': {s: 0.0 for s in sites},
                                                     'selected_site': None, 'earliest_site': sites[0],
                                                     'random_site': sites[random_index], 'random_index': random_index,
                                                     'magnitude_site': sites[0],
                                                     'magnitude_means': {s: 0.0 for s in sites},
                                                     'magnitude_pairs': magnitude_pairs}}})
    return records, cases


def test_discovery_inventory_requires_each_target_donor_site():
    control = _controller()
    records, cases = _complete_records()
    terminal = {'detail': {'forwards': 608}}
    control.check_discovery(records, terminal, cases)
    patches = [row for row in records if row['event']['kind'] == 'patch']
    patches[-1]['event']['donor'] = patches[-2]['event']['donor']
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, terminal, cases)


def test_discovery_inventory_rejects_changed_frozen_case():
    control = _controller()
    records, cases = _complete_records()
    expected = json.loads(json.dumps(cases))
    records[0]['event']['cases'][0]['input_ids'][1] += 1
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, expected)


def test_discovery_inventory_rejects_failed_control():
    control = _controller()
    records, cases = _complete_records()
    event = next(row['event'] for row in records if row['event']['kind'] == 'delivery')
    event['logits_check']['within_tolerance'] = False
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_discovery_inventory_rejects_missing_tensor_or_wrong_donor_reference():
    control = _controller()
    records, cases = _complete_records()
    capture = next(row for row in records if row['event']['kind'] == 'clean_capture')
    del capture['tensors']['capture_0']
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)
    records, cases = _complete_records()
    patch = next(row for row in records if row['event']['kind'] == 'patch'
                 and row['event']['target'] != row['event']['donor'])
    patch['event']['source_tensor_sha256'] = '0' * 64
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_discovery_inventory_rejects_missing_metrics_or_prediction():
    control = _controller()
    records, cases = _complete_records()
    metric = next(row for row in records if row['event']['kind'] == 'present_site_metrics')
    del metric['event']['metrics']['pairs']
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)
    records, cases = _complete_records()
    del records[-1]['event']['result']['selection']['predictions']
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_discovery_inventory_rejects_missing_named_repeat_check():
    control = _controller()
    records, cases = _complete_records()
    repeat = next(row for row in records if row['event']['kind'] == 'clean_repeat')
    del repeat['event']['checks']['1:attn:last']
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_discovery_inventory_rejects_wrong_input_identity_or_control_reference():
    control = _controller()
    records, cases = _complete_records()
    capture = next(row['event'] for row in records if row['event']['kind'] == 'clean_capture')
    capture['input_sha256'] = '0' * 64
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)
    records, cases = _complete_records()
    delivery = next(row['event'] for row in records if row['event']['kind'] == 'delivery')
    delivery['reference'] = delivery['target']
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_discovery_inventory_rejects_mislabelled_past_contrast():
    control = _controller()
    records, cases = _complete_records()
    metric = next(row['event'] for row in records if row['event']['kind'] == 'secondary_past_site_metrics')
    metric['contrast']['first_id'] = 318
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_discovery_inventory_rejects_final_clean_before_patches():
    control = _controller()
    records, cases = _complete_records()
    final = next(i for i, row in enumerate(records) if row['event']['kind'] == 'final_clean')
    records.insert(1, records.pop(final))
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


@pytest.mark.parametrize('field,value', [('module_path', 'transformer.h.2.attn'),
                                         ('output_selector', []), ('forward_index', 1),
                                         ('invocation', 1), ('batch_index', 1), ('token_index', 1)])
def test_discovery_inventory_rejects_wrong_action_coordinate(field, value):
    control = _controller()
    records, cases = _complete_records()
    patch = next(row['event'] for row in records if row['event']['kind'] == 'patch')
    patch['plan_metadata']['steps'][2][field] = value
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_discovery_inventory_requires_exact_numeric_types_and_state_digest():
    control = _controller()
    records, cases = _complete_records()
    records[0]['event']['cpu_threads'] = True
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)
    records, cases = _complete_records()
    del records[-1]['event']['result']['model_state_sha256']
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)
    records, cases = _complete_records()
    records[-1]['event']['result']['forwards'] = 608.0
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)
    records, cases = _complete_records()
    records[-1]['event']['result']['controls']['null'] = 16.0
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_inventory_rejects_boolean_event_accounting(tmp_path):
    control = _controller()
    attempt = tmp_path / 'attempt'
    _write_candidate(attempt)
    terminal_path = attempt / 'terminal.json'
    terminal = json.loads(terminal_path.read_text())
    terminal['events'] = True
    terminal_path.write_text(json.dumps(terminal) + '\n')
    with pytest.raises(control.InventoryError):
        control.audit_storage(attempt)


def test_discovery_inventory_rejects_mislabelled_site_path():
    control = _controller()
    records, cases = _complete_records()
    patch = next(row['event'] for row in records if row['event']['kind'] == 'patch')
    patch['site']['module_path'] = 'transformer.h.2.attn'
    with pytest.raises(control.InventoryError):
        control.check_discovery(records, {'detail': {'forwards': 608}}, cases)


def test_complete_synthetic_journal_is_only_evidence_complete(tmp_path):
    control = _controller()
    records, cases = _complete_records()
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    (attempt / 'tensors').mkdir()
    binding = {'runtime': {'cpu_threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
                           'deterministic_warn_only': False, 'float32_matmul_precision': 'highest',
                           'denormal_probe': [1, 4194304]},
               'study_budget': {'phase_reservations_bytes': {'natural_discovery': 1024 * 1024}}}
    binding_bytes = (json.dumps(binding, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode()
    started = {'schema': 'mlg-natural-cpu-attempt-v1',
               'binding_sha256': hashlib.sha256(binding_bytes).hexdigest(), 'binding': binding}
    (attempt / 'started.json').write_text(json.dumps(started) + '\n', encoding='utf-8')
    payloads = {}
    for row in records:
        for ref in row['tensors'].values():
            size = ref['bytes']
            payload = bytes(size)
            digest = hashlib.sha256(payload).hexdigest()
            ref['sha256'] = digest
            payloads[digest] = payload
        if row['event']['kind'] == 'patch':
            row['event']['source_tensor_sha256'] = row['tensors']['after']['sha256']
    for digest, payload in payloads.items():
        (attempt / 'tensors' / (digest + '.f32le')).write_bytes(payload)
    event_bytes = b''.join((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode()
                           for row in records)
    (attempt / 'events.jsonl').write_bytes(event_bytes)
    terminal = {'status': 'complete_candidate', 'detail': {'forwards': 608}, 'events': len(records),
                'retained_tensor_bytes': sum(map(len, payloads.values())), 'event_log_bytes': len(event_bytes),
                'storage_accounting_complete': True}
    (attempt / 'terminal.json').write_text(json.dumps(terminal) + '\n', encoding='utf-8')
    verdict = control.audit_result(attempt, exit_code=0, reaped=True,
                                   expected_cases=cases, expected_start=started)
    assert verdict['status'] == 'evidence_complete'
    assert verdict['scientific_validation'].startswith('pending')
    started['binding']['study_budget']['phase_reservations_bytes']['natural_discovery'] = 1
    small = (json.dumps(started['binding'], sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode()
    started['binding_sha256'] = hashlib.sha256(small).hexdigest()
    (attempt / 'started.json').write_text(json.dumps(started) + '\n', encoding='utf-8')
    verdict = control.audit_result(attempt, exit_code=0, reaped=True,
                                   expected_cases=cases, expected_start=started)
    assert verdict['status'] == 'incomplete' and 'reservation' in verdict['reason']


def test_recorded_execution_configuration_requires_digest_and_bound_globals():
    control = _controller()
    records, _ = _complete_records()
    contract = records[0]['event']
    runtime = {'cpu_threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
               'deterministic_warn_only': False, 'float32_matmul_precision': 'highest',
               'denormal_probe': [1, 4194304]}
    control.check_execution_configuration(contract, runtime)
    changed = json.loads(json.dumps(contract))
    changed['execution_configuration'][0][2][1] = 2
    with pytest.raises(control.InventoryError):
        control.check_execution_configuration(changed, runtime)
    raw = json.dumps(changed['execution_configuration'], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    changed['execution_configuration_sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(control.InventoryError, match='frozen runtime'):
        control.check_execution_configuration(changed, runtime)


def test_bound_cases_preserve_public_rows_and_subject_coordinates():
    control = _controller()
    root = Path(__file__).resolve().parents[1]
    raw = (root / 'results/distilgpt2-agreement-preflight-v1-20261028/frozen.json').read_bytes()
    manifest = json.loads(raw)
    cases = control.bind_public_cases(raw, manifest['rows'], manifest['groups'])
    assert len(cases) == 16
    assert cases[0] == {'case_id': 'c00', 'quartet': 'q1', 'lemma': 'cat', 'number': 'singular',
                        'input_ids': [464, 3797, 1474, 262, 9168],
                        'attention_mask': [1, 1, 1, 1, 1], 'subject_index': 1}
    assert cases[-1]['lemma'] == 'sailor'
    assert cases[-1]['number'] == 'plural'


def test_bound_cases_reject_changed_manifest_or_tokenizer_alignment():
    control = _controller()
    root = Path(__file__).resolve().parents[1]
    raw = (root / 'results/distilgpt2-agreement-preflight-v1-20261028/frozen.json').read_bytes()
    manifest = json.loads(raw)
    with pytest.raises(control.InventoryError):
        control.bind_public_cases(raw + b' ', manifest['rows'], manifest['groups'])
    manifest['rows'][0]['subject_index'] = 2
    with pytest.raises(control.InventoryError):
        control.bind_public_cases(raw, manifest['rows'], manifest['groups'])


def test_supervisor_preserves_nonzero_child_output_and_exit(tmp_path):
    control = _controller()
    out = tmp_path / 'new-run'
    command = [sys.executable, '-c', 'import sys; sys.stdin.buffer.readline(); print("raw output"); print("raw error", file=sys.stderr); sys.exit(7)']
    verdict = control.supervise(command, out, [], os.environ.copy())
    assert verdict['status'] == 'incomplete'
    assert verdict['exit_code'] == 7 and verdict['child_reaped'] is True
    assert (out / 'stdout.txt').read_bytes().strip() == b'raw output'
    assert (out / 'stderr.txt').read_bytes().strip() == b'raw error'
    assert json.loads((out / 'supervisor.json').read_text())['exit_code'] == 7
    with pytest.raises(FileExistsError):
        control.supervise(command, out, [], os.environ.copy())


def test_supervisor_counts_unpublished_partial_tensor_after_failed_child(tmp_path):
    control = _controller()
    out = tmp_path / 'partial-run'
    code = ("import pathlib,sys; sys.stdin.buffer.readline(); "
            "d=pathlib.Path(sys.argv[1])/'attempt'/'tensors'; d.mkdir(parents=True); "
            "(d/'unfinished.partial').write_bytes(b'partial'); sys.exit(7)")
    verdict = control.supervise([sys._base_executable, '-c', code, str(out)], out, [], os.environ.copy())
    assert verdict['status'] == 'incomplete' and verdict['cleanup_verified'] is True
    observed = verdict['retained_tensor_files']
    assert observed['accounting_complete'] is True and observed['total_bytes'] == 7
    assert observed['files'] == [{'name': 'unfinished.partial', 'bytes': 7,
                                  'sha256': hashlib.sha256(b'partial').hexdigest()}]
    assert (out / 'attempt/tensors/unfinished.partial').read_bytes() == b'partial'
    assert json.loads((out / 'supervisor.json').read_text())['retained_tensor_files'] == observed


def test_uncertain_cleanup_does_not_read_live_tensor_files(tmp_path, monkeypatch):
    control = _controller()
    tensors = tmp_path / 'attempt/tensors'
    tensors.mkdir(parents=True)
    (tensors / 'live.partial').write_bytes(b'partial')
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('live file read'))
    observed = control._retained_tensor_files(tmp_path / 'attempt', False)
    assert observed['accounting_complete'] is False
    assert observed['files'] == [] and 'pending' in observed['reason']


def test_supervisor_reports_verified_keyboard_interrupt_as_cancelled(tmp_path, monkeypatch):
    control = _controller()
    actual_popen = subprocess.Popen
    def interrupted(*args, **kwargs):
        child = actual_popen(*args, **kwargs)
        actual_wait = child.wait
        first = [True]
        def wait(*args, **kwargs):
            if first[0]:
                first[0] = False
                raise KeyboardInterrupt()
            return actual_wait(*args, **kwargs)
        child.wait = wait
        return child
    monkeypatch.setattr(control.subprocess, 'Popen', interrupted)
    code = 'import sys,time; sys.stdin.buffer.readline(); time.sleep(30)'
    verdict = control.supervise([sys._base_executable, '-c', code], tmp_path / 'cancelled', [], os.environ.copy())
    assert verdict['status'] == 'cancelled'
    assert verdict['cleanup_verified'] is True and verdict['child_reaped'] is True


def test_supervisor_bounds_log_overflow(tmp_path):
    control = _controller()
    out = tmp_path / 'overflow'
    command = [sys.executable, '-c', 'import sys; sys.stdin.buffer.readline(); sys.stdout.write("a" * 3000000); sys.stdout.flush()']
    verdict = control.supervise(command, out, [], os.environ.copy())
    assert verdict['status'] == 'incomplete'
    assert verdict['log_overflow'] is True
    assert (out / 'stdout.txt').stat().st_size <= 1024 * 1024


def test_deep_json_is_reported_as_incomplete(tmp_path):
    control = _controller()
    attempt = tmp_path / 'attempt'
    _write_candidate(attempt)
    (attempt / 'started.json').write_bytes(b'[' * 2000 + b'0' + b']' * 2000)
    verdict = control.audit_result(attempt, exit_code=0, reaped=True)
    assert verdict['status'] == 'incomplete'


def test_owner_start_gate_times_out_without_permission(tmp_path):
    _controller()
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / 'study-started'
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from looking_glass.natural_agreement_controller import await_start; "
            "await_start(timeout=1); open(sys.argv[2], 'w').write('started')")
    child = subprocess.Popen([sys.executable, '-I', '-c', code, str(root), str(marker)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert child.wait(timeout=5) == 1
    assert b'startup signal' in child.stderr.read()
    assert not marker.exists()
    child.stdin.close()


def test_owner_start_gate_accepts_exact_supervisor_signal(tmp_path):
    _controller()
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / 'study-started'
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from looking_glass.natural_agreement_controller import await_start; "
            "await_start(timeout=2); open(sys.argv[2], 'w').write('started')")
    child = subprocess.Popen([sys.executable, '-I', '-c', code, str(root), str(marker)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = child.communicate(b'MLG-natural-discovery-start-v1\n', timeout=5)
    assert child.returncode == 0, err.decode(errors='replace')
    assert marker.read_text() == 'started'


def test_lost_spawn_handle_never_signals_study_and_is_unverified(tmp_path, monkeypatch):
    control = _controller()
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / 'study-started'
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from looking_glass.natural_agreement_controller import await_start; "
            "await_start(timeout=1); open(sys.argv[2], 'w').write('started')")
    actual_popen = subprocess.Popen
    children = []
    def lose_handle(*args, **kwargs):
        children.append(actual_popen(*args, **kwargs))
        raise RuntimeError('simulated lost Popen return')
    monkeypatch.setattr(control.subprocess, 'Popen', lose_handle)
    verdict = control.supervise([sys.executable, '-I', '-c', code, str(root), str(marker)],
                                tmp_path / 'run', [], os.environ.copy())
    assert verdict['status'] == 'incomplete'
    assert verdict['child_reaped'] is False and verdict['retry_safe'] is False
    child = children[0]
    assert child.wait(timeout=5) == 1
    assert not marker.exists()
    child.stdin.close()
    child.stdout.close()
    child.stderr.close()


def test_source_binding_rejects_changed_or_untracked_python(tmp_path):
    control = _controller()
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'looking_glass').mkdir()
    source = tmp_path / 'looking_glass' / 'one.py'
    source.write_text('value = 1\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    commit, inventory = control.source_inventory(tmp_path)
    assert len(commit) == 40
    assert inventory['looking_glass/one.py'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert control.verify_source_inventory(tmp_path, commit, inventory) == (commit, inventory)
    source.write_text('value = 2\n', encoding='utf-8')
    with pytest.raises(control.InventoryError):
        control.source_inventory(tmp_path)
    with pytest.raises(control.InventoryError):
        control.verify_source_inventory(tmp_path, commit, inventory)
    source.write_text('value = 1\n', encoding='utf-8')
    (tmp_path / 'looking_glass' / 'extra.py').write_text('x = 1\n', encoding='utf-8')
    with pytest.raises(control.InventoryError):
        control.source_inventory(tmp_path)


def test_source_binding_ignores_git_replace_and_environment_overrides(tmp_path, monkeypatch):
    control = _controller()
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'looking_glass').mkdir()
    source = tmp_path / 'looking_glass' / 'one.py'
    source.write_text('value = 1\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'original'], check=True)
    original = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD']).decode().strip()
    source.write_text('value = 2\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'replacement'], check=True)
    replacement = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD']).decode().strip()
    subprocess.run(['git', '-C', str(tmp_path), 'replace', original, replacement], check=True)
    subprocess.run(['git', '--no-replace-objects', '-C', str(tmp_path), 'reset', '--hard', '-q', original], check=True)
    commit, inventory = control.source_inventory(tmp_path)
    assert commit == original
    assert source.read_text() == 'value = 1\n'
    assert inventory['looking_glass/one.py'] == hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setenv('GIT_DIR', str(tmp_path / 'wrong-repository'))
    assert control.source_inventory(tmp_path) == (commit, inventory)


def test_supervisor_closes_job_even_when_termination_fails(tmp_path, monkeypatch):
    control = _controller()
    import scripts.v2_source_guards as guards
    jobs = []
    class FailingJob:
        def __init__(self):
            self.child = None
            self.closed = False
            jobs.append(self)
        def assign(self, child):
            self.child = child
        def terminate_and_wait(self):
            raise OSError('simulated job drain failure')
        def close(self):
            self.closed = True
            if self.child is not None and self.child.poll() is None:
                self.child.kill()
    monkeypatch.setattr(guards, 'WindowsJob', FailingJob)
    command = [sys.executable, '-c', 'import sys,time; sys.stdin.buffer.readline(); time.sleep(3)']
    verdict = control.supervise(command, tmp_path / 'run', [], os.environ.copy(), timeout=1)
    assert verdict['status'] == 'incomplete'
    assert jobs[0].closed is True
    assert verdict['child_reaped'] is True
    assert verdict['job_drained'] is False
