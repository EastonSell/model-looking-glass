"""Fabricated-only prediction freeze tests; no model or held-out inputs."""

import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from looking_glass import agreement_heldout, agreement_patch_runner, pretrained_number_commitments as commitments
from looking_glass import natural_prediction_freeze as freeze_builder
from looking_glass.agreement_metrics import score_factorial

try:
    from looking_glass.natural_prediction_freeze import build_natural_prediction_freeze
except ModuleNotFoundError as error:
    if error.name != 'looking_glass.natural_prediction_freeze':
        raise
    build_natural_prediction_freeze = None


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def pretty(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fabricated_evidence():
    """Fabricated events over the public manifest; never forward a model."""
    path = (Path(__file__).resolve().parents[1] / 'results' /
            'distilgpt2-agreement-preflight-v1-20261028' / 'frozen.json')
    preflight_manifest = path.read_bytes()
    rows = json.loads(preflight_manifest)['rows']
    bound_cases = [{'case_id': f'c{index:02}', 'quartet': f'q{row["group"]}',
                    'lemma': rows[index - index % 2]['subject'], 'number': row['subject_number'],
                    'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
                    'subject_index': row['subject_index']} for index, row in enumerate(rows)]
    sites = [site['id'] for site in agreement_patch_runner._CANDIDATES]
    cases, patches, magnitudes = [], [], []
    for family in range(4):
        members = []
        for bound in bound_cases[family * 4:family * 4 + 4]:
            number = bound['number']
            row = {'id': bound['case_id'], 'quartet': bound['quartet'], 'lemma': bound['lemma'],
                   'number': number, 'clean_is': 4.0 if number == 'singular' else 0.0,
                   'clean_are': 0.0 if number == 'singular' else 4.0}
            cases.append(row)
            members.append(row)
        for index, site in enumerate(sites):
            for lemma in (members[0]['lemma'], members[2]['lemma']):
                singular = next(row['id'] for row in members if row['lemma'] == lemma and row['number'] == 'singular')
                plural = next(row['id'] for row in members if row['lemma'] == lemma and row['number'] == 'plural')
                magnitudes.append({'site': site, 'quartet': members[0]['quartet'], 'lemma': lemma,
                                   'singular': singular, 'plural': plural, 'l2': float(index + 1)})
            for target in members:
                for donor in members:
                    shift = 0.125 * (index + 1) if donor['number'] != target['number'] else 0.0
                    weight = 1 if target['number'] == 'singular' else -1
                    patches.append({'site': site, 'target': target['id'], 'donor': donor['id'],
                                    'patched_is': target['clean_is'] - weight * shift,
                                    'patched_are': target['clean_are']})
    scored = score_factorial(cases, patches, sites)
    selection = agreement_patch_runner._discovery_selection(scored, magnitudes, 4)
    site_events = {}
    for kind in ('present_site_metrics', 'secondary_past_site_metrics'):
        contrast = ({'first_token': ' is', 'first_id': 318, 'second_token': ' are', 'second_id': 389}
                    if kind == 'present_site_metrics' else
                    {'first_token': ' was', 'first_id': 373, 'second_token': ' were', 'second_id': 547})
        events = []
        for site in sites:
            pairs = [agreement_patch_runner._pair_with_signs(row) for row in scored['pairs'] if row['site'] == site]
            if kind == 'secondary_past_site_metrics':
                pairs = [agreement_patch_runner._label_past_pair(row) for row in pairs]
            events.append({'kind': kind, 'site': site, 'contrast': contrast,
                           'metrics': {'baseline': scored['baseline'],
                                       'site_summary': next(row for row in scored['site_summaries'] if row['site'] == site),
                                       'quartets': [row for row in scored['quartets'] if row['site'] == site],
                                       'pairs': pairs}})
        site_events[kind] = events

    events = [{'kind': 'run_contract', 'cases': bound_cases}]
    for family in range(4):
        quartet = bound_cases[family * 4]['quartet']
        members = [case['case_id'] for case in bound_cases if case['quartet'] == quartet]
        for kind in ('clean_capture', 'clean_repeat', 'null'):
            for case in members:
                events.append({'kind': kind, 'quartet': quartet,
                               ('target' if kind == 'null' else 'case'): case})
        events.extend({'kind': 'patch', 'quartet': quartet, 'target': target,
                       'site': {'id': site}, 'donor': donor}
                      for site in sites for target in members for donor in members)
        events.extend({'kind': kind, 'quartet': quartet, 'target': target}
                      for target in members for kind in ('delivery', 'subject_position'))
        events.extend({'kind': 'final_clean', 'quartet': quartet, 'case': case} for case in members)
    events.extend(site_events['present_site_metrics'])
    events.extend(site_events['secondary_past_site_metrics'])
    events.append({'kind': 'summary', 'result': {'status': 'complete', 'forwards': 608,
                                                  'selection': selection}})
    referenced = {}
    records = []
    for index, event in enumerate(events):
        kind = event['kind']
        if kind in ('clean_capture', 'clean_repeat'):
            shapes = {f'capture_{i}': [768] for i in range(10)}
            shapes['last_logits'] = [50257]
        elif kind in ('null', 'patch', 'delivery', 'subject_position'):
            shapes = {'before': [768], 'after': [768], 'whole_before': [1, 5, 768],
                      'whole_after': [1, 5, 768]}
            name = ('contrast_logits' if kind == 'patch' and event['target'] != event['donor']
                    else 'last_logits')
            shapes[name] = [4] if name == 'contrast_logits' else [50257]
        elif kind == 'final_clean':
            shapes = {'last_logits': [50257]}
        else:
            shapes = {}
        tensors = {}
        for name, shape in shapes.items():
            digest = sha(b'fabricated-shape-' + canonical(shape))
            size = 4
            for dimension in shape:
                size *= dimension
            tensors[name] = {'sha256': digest, 'shape': shape, 'dtype': 'float32-le', 'bytes': size}
            referenced[digest] = size
        records.append({'index': index, 'event': event, 'tensors': tensors})
    event_bytes = b''.join(canonical(row) + b'\n' for row in records)
    binding = {'schema': 'mlg-natural-cpu-binding-v1', 'source_commit': 'a' * 40,
               'protocol_sha256': commitments.PROTOCOL_SHA256, 'checkpoint_revision': commitments.CHECKPOINT_REVISION,
               'preflight_manifest_sha256': '1b1adf4d72163bfa3e0399ed8ee92eba6b5c0eb82a47edf28cecde5ffb1deafe',
               'natural_prediction_clarification_sha256': '3' * 64,
               'independent_natural_audit_source_sha256': '4' * 64,
               'track': 'natural discovery only',
               'expected_forwards': 608, 'cases': bound_cases, 'runtime': {'device': 'cpu'},
               'study_budget': {'study_limit_bytes': 512 * 1024 * 1024,
                                'prior_unique_tensor_bytes': 3216448,
                                'prior_payloads': [{'sha256': '7c262f3fe81236888a8f5fdb6d31f7835666674be9ffbf36f8ce2ca6a375eb6c',
                                                    'bytes': 3216448,
                                                    'description': 'public preflight full-vocabulary logits'}],
                                'phase_reservations_bytes': {'natural_discovery': 128 * 1024 * 1024,
                                                             'natural_heldout': 160 * 1024 * 1024,
                                                             'injection_discovery': 48 * 1024 * 1024,
                                                             'injection_heldout': 72 * 1024 * 1024},
                                'unallocated_bytes': 105835456},
               'source_sha256': {'looking_glass/natural_prediction_freeze.py': '2' * 64}}
    binding_bytes = pretty(binding)
    started = {'schema': 'mlg-natural-cpu-attempt-v1', 'binding_sha256': sha(binding_bytes), 'binding': binding}
    started_bytes = pretty(started)
    terminal = {'status': 'complete_candidate', 'storage_accounting_complete': True, 'events': len(records),
                'event_log_bytes': len(event_bytes), 'retained_tensor_bytes': sum(referenced.values()),
                'detail': {'forwards': 608}}
    terminal_bytes = pretty(terminal)
    inventory = {'started_sha256': sha(started_bytes), 'events_sha256': sha(event_bytes),
                 'terminal_sha256': sha(terminal_bytes), 'retained_tensor_bytes': sum(referenced.values()),
                 'tensor_sha256': sorted(referenced)}
    supervisor = {'status': 'evidence_complete', 'exit_code': 0, 'child_reaped': True,
                  'job_drained': True, 'job_closed': True, 'cleanup_verified': True,
                  'retry_safe': True, 'log_overflow': False, 'inventory': inventory,
                  'retained_tensor_files': {'accounting_complete': True,
                                            'total_bytes': sum(referenced.values()),
                                            'files': [{'name': digest + '.f32le', 'bytes': size, 'sha256': digest}
                                                      for digest, size in sorted(referenced.items())]}}
    launcher = {'status': 'evidence_complete', 'phase': 'owner_supervision', 'child_started': True,
                'cleanup_verified': True, 'retry_safe': True, 'owner_record': 'owner/supervisor.json',
                'owner_supervisor_sha256': sha(pretty(supervisor))}
    audit = {'schema': 'mlg-independent-natural-discovery-audit-v1', 'status': 'independent_saved_evidence_pass',
             'journal_event_sha256': sha(event_bytes), 'journal_started_sha256': sha(started_bytes),
             'journal_terminal_sha256': sha(terminal_bytes), 'audit_source_sha256': '4' * 64,
             'preflight_manifest_sha256': binding['preflight_manifest_sha256'], 'forwards': 608,
             'tensor_count': len(referenced), 'tensor_bytes': sum(referenced.values()),
             'present_predictions': selection['predictions'],
             'present_family_effects': {e['site']: {q['quartet']: q['number_main_contrast']
                                                    for q in e['metrics']['quartets']}
                                        for e in site_events['present_site_metrics']},
             'present_strict_sign_reversals': {site: {'same_number': 0, 'opposite_number': 0} for site in sites},
             'secondary_past_predictions': {e['site']: e['metrics']['site_summary']['mean_number_main_contrast']
                                            for e in site_events['secondary_past_site_metrics']},
             'present_baseline': scored['baseline'], 'past_baseline': scored['baseline'],
             'selected_site': selection['selected_site'], 'earliest_site': selection['earliest_site'],
             'random_site': selection['random_site'], 'magnitude_site': selection['magnitude_site'],
             # Fake the real audit envelope solely for this in-memory fixture.
             'scope': 'Saved public development data only; no held-out or general causal claim.'}
    return {'events': event_bytes, 'started': started_bytes, 'terminal': terminal_bytes,
            'supervisor': pretty(supervisor), 'launcher_terminal': pretty(launcher),
            'binding': binding_bytes, 'audit': canonical(audit), 'preflight_manifest': preflight_manifest,
            'expected_source_commit': 'a' * 40,
            'expected_audit_source_sha256': '4' * 64, 'expected_builder_sha256': '2' * 64,
            'expected_clarification_sha256': '3' * 64}, site_events, selection


def replace_supervisor(inputs, value):
    inputs['supervisor'] = pretty(value)
    launcher = json.loads(inputs['launcher_terminal'])
    launcher['owner_supervisor_sha256'] = sha(inputs['supervisor'])
    inputs['launcher_terminal'] = pretty(launcher)


def replace_binding(inputs, value):
    inputs['binding'] = pretty(value)
    started = json.loads(inputs['started'])
    started['binding'] = value
    started['binding_sha256'] = sha(inputs['binding'])
    inputs['started'] = pretty(started)
    supervisor = json.loads(inputs['supervisor'])
    supervisor['inventory']['started_sha256'] = sha(inputs['started'])
    replace_supervisor(inputs, supervisor)
    audit = json.loads(inputs['audit'])
    audit['journal_started_sha256'] = sha(inputs['started'])
    inputs['audit'] = canonical(audit)


def replace_cases(inputs, cases):
    binding = json.loads(inputs['binding'])
    binding['cases'] = cases
    replace_binding(inputs, binding)
    rows = [json.loads(line) for line in inputs['events'].splitlines()]
    rows[0]['event']['cases'] = cases
    inputs['events'] = b''.join(canonical(row) + b'\n' for row in rows)
    terminal = json.loads(inputs['terminal'])
    terminal['event_log_bytes'] = len(inputs['events'])
    inputs['terminal'] = pretty(terminal)
    supervisor = json.loads(inputs['supervisor'])
    supervisor['inventory']['events_sha256'] = sha(inputs['events'])
    supervisor['inventory']['terminal_sha256'] = sha(inputs['terminal'])
    replace_supervisor(inputs, supervisor)
    audit = json.loads(inputs['audit'])
    audit['journal_event_sha256'] = sha(inputs['events'])
    audit['journal_terminal_sha256'] = sha(inputs['terminal'])
    inputs['audit'] = canonical(audit)


class NaturalPredictionFreezeTests(unittest.TestCase):
    def test_frozen_manifest_bytes_yield_exact_ordered_public_cases(self):
        path = (Path(__file__).resolve().parents[1] / 'results' /
                'distilgpt2-agreement-preflight-v1-20261028' / 'frozen.json')
        raw = path.read_bytes()
        manifest = json.loads(raw)
        rows = manifest['rows']
        expected = [{'case_id': f'c{index:02}', 'quartet': f'q{row["group"]}',
                     'lemma': rows[index - index % 2]['subject'], 'number': row['subject_number'],
                     'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
                     'subject_index': row['subject_index']} for index, row in enumerate(rows)]
        self.assertEqual(freeze_builder._public_cases(raw), expected)
        with self.assertRaises(freeze_builder.PredictionFreezeError):
            freeze_builder._public_cases(raw + b' ')

    def test_rejects_unaligned_or_duplicate_bound_token_rows(self):
        for change in ('duplicate', 'context', 'subject_final', 'subject_position', 'subject_first', 'short'):
            with self.subTest(change=change):
                inputs, _, _ = fabricated_evidence()
                cases = json.loads(inputs['binding'])['cases']
                rows = [json.loads(line) for line in inputs['events'].splitlines()]
                if change == 'duplicate':
                    cases[0]['input_ids'] = list(cases[1]['input_ids'])
                elif change == 'context':
                    cases[0]['input_ids'][2] += 1
                elif change == 'subject_final':
                    cases[0]['subject_index'] = len(cases[0]['input_ids']) - 1
                elif change == 'subject_position':
                    cases[0]['subject_index'] = 2
                elif change == 'subject_first':
                    cases[0]['subject_index'] = 0
                else:
                    cases[0]['input_ids'] = [464]
                    cases[0]['attention_mask'] = [1]
                with self.assertRaises(freeze_builder.PredictionFreezeError):
                    freeze_builder._event_order(rows, cases)

    def test_rejects_different_aligned_cases_despite_consistent_evidence_hashes(self):
        inputs, _, _ = fabricated_evidence()
        cases = json.loads(inputs['binding'])['cases']
        for case in cases[:4]:
            case['input_ids'][2] += 1
        replace_cases(inputs, cases)
        rows = [json.loads(line) for line in inputs['events'].splitlines()]
        freeze_builder._event_order(rows, cases)  # The modified quartet is still structurally valid.
        with self.assertRaisesRegex(freeze_builder.PredictionFreezeError, 'frozen public preflight'):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_changed_manifest_bytes_and_typed_case_aliases(self):
        inputs, _, _ = fabricated_evidence()
        bad = dict(inputs)
        bad['preflight_manifest'] += b' '
        with self.assertRaisesRegex(freeze_builder.PredictionFreezeError, 'frozen manifest'):
            build_natural_prediction_freeze(**bad)
        cases = json.loads(inputs['binding'])['cases']
        cases[0]['subject_index'] = float(cases[0]['subject_index'])
        replace_cases(inputs, cases)
        with self.assertRaisesRegex(freeze_builder.PredictionFreezeError, 'frozen public preflight'):
            build_natural_prediction_freeze(**inputs)

    def test_extracts_exact_ordered_adapter_compatible_prediction(self):
        inputs, events, selection = fabricated_evidence()
        with mock.patch.object(commitments, 'sample_families', side_effect=AssertionError('unexpected draw')):
            bundle = build_natural_prediction_freeze(**inputs)
        record = json.loads(bundle.prediction_bytes)
        self.assertEqual(record['schema'], 'mlg-number-natural-predictions-v2')
        self.assertEqual(record['discovery_event_sha256'], sha(inputs['events']))
        self.assertEqual(record['independent_audit_sha256'], sha(inputs['audit']))
        self.assertEqual(record['present_site_metrics'], events['present_site_metrics'])
        self.assertEqual(record['secondary_past_site_metrics'], events['secondary_past_site_metrics'])
        self.assertEqual(record['selection'], selection)
        self.assertEqual(bundle.prediction_bytes, canonical(record))
        freeze = commitments.load_prediction_freeze(bundle.aggregate_bytes, sha(bundle.aggregate_bytes))
        checked = agreement_heldout.load_frozen_natural_predictions(bundle.prediction_bytes, freeze)
        self.assertEqual(agreement_heldout.frozen_selection(checked)[0], selection)
        self.assertEqual(json.loads(bundle.aggregate_bytes)['tracks']['natural']['runtime_sha256'], sha(inputs['binding']))

    def test_rejects_incomplete_or_mismatched_evidence(self):
        inputs, _, _ = fabricated_evidence()
        for key, transform in (
            ('events', lambda raw: raw + b' '),
            ('binding', lambda raw: raw.replace(b'"device": "cpu"', b'"device": "gpu"')),
            ('audit', lambda raw: raw.replace(b'independent_saved_evidence_pass', b'pending')),
            ('expected_audit_source_sha256', lambda value: '5' * 64),
            ('expected_builder_sha256', lambda value: '6' * 64),
            ('expected_clarification_sha256', lambda value: '7' * 64),
            ('expected_source_commit', lambda value: 'b' * 40),
            ('launcher_terminal', lambda raw: raw.replace(b'evidence_complete', b'incomplete')),
            ('supervisor', lambda raw: raw.replace(b'"job_drained": true', b'"job_drained": false')),
        ):
            with self.subTest(key=key):
                bad = dict(inputs)
                bad[key] = transform(bad[key])
                if key == 'supervisor':
                    replace_supervisor(bad, json.loads(bad['supervisor']))
                with self.assertRaises((ValueError, RuntimeError)):
                    build_natural_prediction_freeze(**bad)

    def test_rejects_audit_prediction_inconsistent_with_events(self):
        inputs, _, _ = fabricated_evidence()
        audit = json.loads(inputs['audit'])
        audit['present_predictions']['1:attn:last'] += 1
        inputs['audit'] = canonical(audit)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_noncanonical_or_duplicate_key_audit(self):
        inputs, _, _ = fabricated_evidence()
        inputs['audit'] = inputs['audit'][:-1] + b',"status":"independent_saved_evidence_pass"}'
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_synthetic_scope_and_missing_frozen_builder(self):
        inputs, _, _ = fabricated_evidence()
        audit = json.loads(inputs['audit'])
        audit['scope'] = 'Fabricated synthetic data only.'
        inputs['audit'] = canonical(audit)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)
        inputs, _, _ = fabricated_evidence()
        binding = json.loads(inputs['binding'])
        del binding['source_sha256']['looking_glass/natural_prediction_freeze.py']
        replace_binding(inputs, binding)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_wrong_metric_order_even_after_all_hashes_are_updated(self):
        inputs, _, _ = fabricated_evidence()
        rows = [json.loads(line) for line in inputs['events'].splitlines()]
        positions = [i for i, row in enumerate(rows) if row['event']['kind'] == 'present_site_metrics']
        first, second = positions[:2]
        rows[first]['event'], rows[second]['event'] = rows[second]['event'], rows[first]['event']
        inputs['events'] = b''.join(canonical(row) + b'\n' for row in rows)
        terminal = json.loads(inputs['terminal'])
        terminal['event_log_bytes'] = len(inputs['events'])
        inputs['terminal'] = pretty(terminal)
        supervisor = json.loads(inputs['supervisor'])
        supervisor['inventory']['events_sha256'] = sha(inputs['events'])
        supervisor['inventory']['terminal_sha256'] = sha(inputs['terminal'])
        replace_supervisor(inputs, supervisor)
        audit = json.loads(inputs['audit'])
        audit['journal_event_sha256'] = sha(inputs['events'])
        audit['journal_terminal_sha256'] = sha(inputs['terminal'])
        inputs['audit'] = canonical(audit)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_conflicting_supervisor_error_despite_success_label(self):
        inputs, _, _ = fabricated_evidence()
        supervisor = json.loads(inputs['supervisor'])
        supervisor['supervisor_error'] = 'cleanup failed'
        replace_supervisor(inputs, supervisor)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_supervisor_mixed_with_a_different_launcher_terminal(self):
        inputs, _, _ = fabricated_evidence()
        launcher = json.loads(inputs['launcher_terminal'])
        launcher['owner_supervisor_sha256'] = '8' * 64
        inputs['launcher_terminal'] = pretty(launcher)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_unsafe_or_missing_supervisor_retry_state(self):
        for mode in ('unsafe', 'missing'):
            with self.subTest(mode=mode):
                inputs, _, _ = fabricated_evidence()
                supervisor = json.loads(inputs['supervisor'])
                if mode == 'unsafe':
                    supervisor['retry_safe'] = False
                else:
                    del supervisor['retry_safe']
                replace_supervisor(inputs, supervisor)
                with self.assertRaises((ValueError, RuntimeError)):
                    build_natural_prediction_freeze(**inputs)

    def test_rejects_incomplete_retained_file_accounting(self):
        inputs, _, _ = fabricated_evidence()
        supervisor = json.loads(inputs['supervisor'])
        supervisor['retained_tensor_files']['accounting_complete'] = False
        replace_supervisor(inputs, supervisor)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_forged_retained_file_metadata(self):
        for change in ('digest', 'size', 'duplicate'):
            with self.subTest(change=change):
                inputs, _, _ = fabricated_evidence()
                supervisor = json.loads(inputs['supervisor'])
                files = supervisor['retained_tensor_files']['files']
                if change == 'digest':
                    files[0]['sha256'] = '0' * 64
                elif change == 'size':
                    files[0]['bytes'] += 1
                else:
                    files[-1] = dict(files[0])
                replace_supervisor(inputs, supervisor)
                with self.assertRaises((ValueError, RuntimeError)):
                    build_natural_prediction_freeze(**inputs)

    def test_rejects_missing_witness_inventory_even_if_all_envelopes_match(self):
        inputs, _, _ = fabricated_evidence()
        rows = [json.loads(line) for line in inputs['events'].splitlines()]
        rows[1]['tensors'] = {}
        inputs['events'] = b''.join(canonical(row) + b'\n' for row in rows)
        terminal = json.loads(inputs['terminal'])
        terminal['event_log_bytes'] = len(inputs['events'])
        inputs['terminal'] = pretty(terminal)
        supervisor = json.loads(inputs['supervisor'])
        supervisor['inventory']['events_sha256'] = sha(inputs['events'])
        supervisor['inventory']['terminal_sha256'] = sha(inputs['terminal'])
        replace_supervisor(inputs, supervisor)
        audit = json.loads(inputs['audit'])
        audit['journal_event_sha256'] = sha(inputs['events'])
        audit['journal_terminal_sha256'] = sha(inputs['terminal'])
        inputs['audit'] = canonical(audit)
        with self.assertRaises((ValueError, RuntimeError)):
            build_natural_prediction_freeze(**inputs)

    def test_rejects_budget_omitting_preflight_or_reserved_phases(self):
        for change in ('prior', 'phase'):
            with self.subTest(change=change):
                inputs, _, _ = fabricated_evidence()
                binding = json.loads(inputs['binding'])
                if change == 'prior':
                    binding['study_budget']['prior_unique_tensor_bytes'] = 0
                else:
                    del binding['study_budget']['phase_reservations_bytes']['injection_heldout']
                replace_binding(inputs, binding)
                with self.assertRaises((ValueError, RuntimeError)):
                    build_natural_prediction_freeze(**inputs)
