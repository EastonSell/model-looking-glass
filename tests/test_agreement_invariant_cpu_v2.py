"""Model-free admission checks for the prospective invariant CPU entry."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

path = Path(__file__).with_name('agreement_invariant_cpu_v2.py')
if not path.is_file():
    path = Path(__file__).resolve().parent.parent / 'tools/agreement_invariant_cpu_v2.py'
entry = types.ModuleType('_invariant_cpu_entry')
entry.__file__ = str(path)
exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), entry.__dict__)

# Tests use fabricated reservation constants regardless of the production pin.
entry.RESERVATION_REVISION = 'a' * 40
entry.RESERVATION_SHA256 = 'b' * 64
entry.RESERVATION_VERSION = 'fabricated-v2'
entry.RESERVATION_MODULE = 'looking_glass/pretrained_exposures_v99.py'


def fake_cases():
    return [{'case_id': 'fake-' + str(i), 'role': ('target', 'fragment', 'demonstrative')[i % 3],
             'text': 'fake input ' + str(i), 'token_ids': [1000 + i, 2000]} for i in range(68)]


def fake_payload(cases):
    answers = [value for _, value in entry.ANSWERS]
    return {'records': [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
                         'case_role': row['role'], 'public_text': row['text'],
                         'token_ids': row['token_ids'], 'candidate_token_ids': answers}
                        for row in cases]}


def fake_prelaunch(status='no_protected_rows'):
    stamp = '2026-10-29T12:00:00Z'
    return {'schema': 'distilgpt2-agreement-invariant-noun-prelaunch-v2', 'purpose': 'rehearsal',
            'owner': {'source_revision': '1' * 40, 'source_tree': '2' * 40,
                      'executor_sha256': '3' * 64, 'entry_script_sha256': '4' * 64},
            'protocol': {'revision': entry.PROTOCOL_REVISION, 'path': entry.PROTOCOL_PATH,
                         'sha256': entry.PROTOCOL_SHA256},
            'cases': {'revision': entry.CASE_REVISION, 'path': entry.CASE_PATH,
                      'sha256': entry.CASE_SHA256, 'ordered_rows_sha256': entry.ROWS_SHA256},
            'directions': {'revision': entry.DIRECTION_REVISION, 'manifest_sha256': entry.DIRECTION_MANIFEST_SHA256,
                           **{key + '_sha256': value for key, value in entry.DIRECTION_SHA256.items()}},
            'checkpoint': {'model_id': entry.MODEL_ID, 'revision': entry.MODEL_REVISION,
                           'weights_sha256': entry.WEIGHTS_SHA256, 'tokenizer_sha256': entry.TOKENIZER_SHA256},
            'limits': entry.LIMITS.copy(),
            'runtime': {**entry.VERSIONS, 'cpu_threads': 1, 'cpu_interop_threads': 1,
                        'claim_cuda_visibility': '', 'environment': entry.REQUIRED_ENV.copy()},
            'exposure': {'reservation_revision': entry.RESERVATION_REVISION,
                         'reservation_sha256': entry.RESERVATION_SHA256,
                         'reservation_version': entry.RESERVATION_VERSION,
                         'latest_revision': '5' * 40, 'latest_sha256': '6' * 64,
                         'latest_version': 'fake-v40', 'latest_module': 'looking_glass/pretrained_exposures_v40.py',
                         'operator_revision': '5' * 40, 'operator_sha256': '6' * 64,
                         'operator_observation_sha256': '7' * 64, 'operator_version': 'fake-v40',
                         'operator_refreshed_utc': stamp},
            'custody': {'natural': {'status': 'clear', 'case_sha256': entry.CASE_SHA256,
                                    'support_receipt_sha256': entry.NATURAL_RECEIPT_SHA256,
                                    'scope': entry.NATURAL_SCOPE, 'observation_sha256': '8' * 64,
                                    'refreshed_utc': stamp},
                        'injection': {'status': status, 'direct_target_conflict': 'none reported',
                                      'observation_sha256': '9' * 64, 'refreshed_utc': stamp}},
            'independent_reader_status': 'under_review', 'scope': 'fabricated', 'recorded_utc': stamp}


class AdmissionTests(unittest.TestCase):
    def test_unpublished_reservation_fails_closed(self):
        with mock.patch.object(entry, 'RESERVATION_REVISION', None):
            with self.assertRaises(RuntimeError):
                entry.validate_prelaunch(fake_prelaunch())

    def test_fresh_observer_bytecode_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prefix = root / 'empty-pycache'
            prefix.mkdir()
            with mock.patch.object(entry.sys, 'pycache_prefix', str(prefix)):
                entry.verify_empty_pycache_prefix(prefix)
                (prefix / 'stale.pyc').write_bytes(b'fake')
                with self.assertRaises(ValueError):
                    entry.verify_empty_pycache_prefix(prefix)
            with mock.patch.object(entry.sys, 'pycache_prefix', None):
                with self.assertRaises(ValueError):
                    entry.verify_empty_pycache_prefix(prefix)

    def test_scoped_injection_observations_do_not_require_empty_private_allocation(self):
        for status in ('no_protected_rows', 'direct_target_nonconflict_reported'):
            with self.subTest(status=status):
                entry.validate_prelaunch(fake_prelaunch(status))
        with self.assertRaises(ValueError):
            entry.validate_prelaunch(fake_prelaunch('unknown'))

    def test_public_only_observation_preserves_unknown_protected_history(self):
        record = fake_prelaunch('protected_status_unknown')
        record['custody']['natural'].update(status='public_support_clear_protected_unknown',
                                             protected_history='unverified')
        record['custody']['injection'].update(protected_history='unverified', direct_target_conflict='unknown')
        entry.validate_prelaunch(record)
        for name in ('natural', 'injection'):
            with self.subTest(name=name):
                changed = json.loads(json.dumps(record))
                del changed['custody'][name]['protected_history']
                with self.assertRaises(ValueError):
                    entry.validate_prelaunch(changed)
        record['custody']['injection']['direct_target_conflict'] = 'none reported'
        with self.assertRaises(ValueError):
            entry.validate_prelaunch(record)

    def test_unknown_status_checks_public_receipt_without_asserting_private_state(self):
        stamp = '2026-10-29T12:00:00Z'
        zero = {'case_prefix_of_reference': 0, 'reference_prefix_of_case': 0}
        receipt = {'schema': 'natural-public-cases-crosscheck-v2', 'tokenizer_sha256': entry.TOKENIZER_SHA256,
                   'case_sha256': {'invariant': entry.CASE_SHA256}, 'candidate_sequences': {'invariant': 612},
                   'public_support_sha256': {'original': '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6',
                                              'successor': entry.SUCCESSOR_SUPPORT_SHA256},
                   'counts': {'invariant': {key: zero for key in ('original_public_support', 'successor_public_pool',
                                                                   'six_short_service_rows', 'public_timing_input')}},
                   'invariant_vs_head': zero}
        natural = {'schema': 'agreement-invariant-natural-refresh-v1', 'status': 'public_support_clear_protected_unknown',
                   'case_sha256': entry.CASE_SHA256, 'scope': entry.NATURAL_SCOPE,
                   'successor_support_sha256': entry.SUCCESSOR_SUPPORT_SHA256, 'successor_status': 'unknown',
                   'protected_history': 'unverified', 'refreshed_utc': stamp}
        injection = {'schema': 'agreement-invariant-injection-refresh-v1', 'status': 'protected_status_unknown',
                     'direct_target_conflict': 'unknown', 'site': 'transformer.h.3.attn',
                     'protected_history': 'unverified', 'refreshed_utc': stamp}
        record = fake_prelaunch('protected_status_unknown')
        record['custody']['natural'].update(status=natural['status'], protected_history='unverified')
        record['custody']['injection'].update(protected_history='unverified', direct_target_conflict='unknown')
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ('receipt.json', 'natural.json', 'injection.json')]
            paths[0].write_text(json.dumps(receipt), encoding='utf-8')
            digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
            natural['support_receipt_sha256'] = digest
            paths[1].write_text(json.dumps(natural), encoding='utf-8')
            paths[2].write_text(json.dumps(injection), encoding='utf-8')
            record['custody']['natural'].update(support_receipt_sha256=digest,
                                                observation_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest())
            record['custody']['injection']['observation_sha256'] = hashlib.sha256(paths[2].read_bytes()).hexdigest()
            with mock.patch.object(entry, 'NATURAL_RECEIPT_SHA256', digest):
                entry.validate_prelaunch(record)
                entry.verify_custody(record, *paths)
                natural['successor_status'] = 'undrawn'
                paths[1].write_text(json.dumps(natural), encoding='utf-8')
                record['custody']['natural']['observation_sha256'] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    entry.verify_custody(record, *paths)

    def test_actual_case_bytes_when_supplied(self):
        source = os.environ.get('INVARIANT_CASES')
        if not source:
            self.skipTest('no actual public case path supplied')
        self.assertEqual(len(entry.verify_case_document(Path(source).read_bytes())), 68)

    def test_exact_own_reservations_and_hidden_target(self):
        cases = fake_cases()
        payload = fake_payload(cases)
        entry.validate_exposure(payload, cases)
        payload['records'][0]['target_token_id'] = 42
        with self.assertRaises(ValueError):
            entry.validate_exposure(payload, cases)

    def test_foreign_prefix_and_extensions_rejected(self):
        cases = fake_cases()
        for tokens in ([1000], [1000, 2000, 318, 12], [1000, 2000]):
            payload = fake_payload(cases)
            payload['records'].append({'record_id': 'foreign', 'token_ids': tokens})
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                entry.validate_exposure(payload, cases)

    def test_masked_and_scored_prefixes_are_checked(self):
        cases = fake_cases()
        for foreign in ({'record_id': 'masked', 'token_ids': [1000, 2000, 50256],
                         'attention_mask': [1, 1, 0]},
                        {'record_id': 'scored', 'token_ids': [4, 5],
                         'scored_prefix_token_ids': [1000, 2000], 'scored_candidate_token_ids': [318]}):
            payload = fake_payload(cases)
            payload['records'].append(foreign)
            with self.subTest(foreign=foreign), self.assertRaises(ValueError):
                entry.validate_exposure(payload, cases)

    def test_invalid_scored_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            entry.record_sequences({'token_ids': [1], 'scored_candidate_token_ids': [318]})
        with self.assertRaises(ValueError):
            entry.record_sequences({'token_ids': [1, 2], 'attention_mask': [0, 0]})

    def test_model_identity_all_fields(self):
        identity = {'model_id': entry.MODEL_ID, 'model_revision': entry.MODEL_REVISION,
                    'tokenizer_id': entry.MODEL_ID, 'tokenizer_revision': entry.MODEL_REVISION,
                    'tokenizer_sha256': entry.TOKENIZER_SHA256, 'weights_sha256': entry.WEIGHTS_SHA256}
        entry.validate_inventory_identity({'identity': identity})
        for key in identity:
            changed = dict(identity)
            changed[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_inventory_identity({'identity': changed})

    def test_case_claim_is_exclusive(self):
        class Helper:
            @staticmethod
            def _write(target, value):
                with target.open('x', encoding='utf-8') as stream:
                    json.dump(value, stream)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            entry.claim_attempt(root, root / 'out', '0' * 64, Helper())
            with self.assertRaises(FileExistsError):
                entry.claim_attempt(root, root / 'out', '0' * 64, Helper())


if __name__ == '__main__':
    unittest.main()
