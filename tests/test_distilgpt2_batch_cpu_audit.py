"""Synthetic CPU-only checks of the independent batch evidence reader."""

import copy
import hashlib
import io
import json
import struct
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from looking_glass import distilgpt2_batch_cpu_audit as audit


FIXTURE = Path(__file__).parent / 'fixtures' / 'distilgpt2_batch_cpu_synthetic_frozen_v1.json'
PIN = 'a9efbc8f6a95a7f9551246fd4083df4eae4e00327560385345fc643fbf6b2c25'
COMMIT = 'b' * 40
REPO_PATH = 'results/synthetic/frozen.json'


def synthetic_report(protocol, status, case_status):
    groups = []
    for group in protocol['groups']:
        cases = []
        for frozen in group['cases']:
            cases.append({**frozen, 'status': case_status,
                          'max_abs_error': 0.0 if case_status == 'matched' else 1.0,
                          'absolute_error_overflow': False,
                          'max_relative_error': 0.0 if case_status == 'matched' else 1.0,
                          'relative_error_nonfinite_reason': None})
        groups.append({'group_id': group['group_id'], 'cases': cases})
    report = {'version': 'model-looking-glass-batch-evidence-v1',
              'protocol_sha256': protocol['protocol_sha256'], 'status': status, 'complete': True,
              'calls': {'serial': 4, 'mixed': 2}, 'groups': groups, 'tested_rows': [2], 'error': None}
    report['evidence_sha256'] = audit._digest(report)
    return report


def make_run(root, manifest):
    primary = synthetic_report(manifest['primary_protocol'], 'passed', 'matched')
    controls = {'positive_playback': primary,
                'reversed_playback': synthetic_report(manifest['reversal_protocol'], 'failed', 'mismatch'),
                'batch_mean': synthetic_report(manifest['batch_mean_protocol'], 'failed', 'mismatch')}
    metadata = {'model_id': audit.MODEL_ID, 'model_revision': audit.REVISION,
                'tokenizer_id': audit.MODEL_ID, 'tokenizer_revision': audit.REVISION,
                'weights_sha256': audit.ASSETS['model.safetensors'][1],
                'asset_sha256': {name: value[1] for name, value in audit.ASSETS.items()},
                'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3', 'device': 'cpu',
                'dtype': 'float32', 'attention_implementation': 'eager', 'use_cache': False,
                'parameter_count': 81912576}
    attempt = {'version': audit.VERSION, 'status': 'started', 'manifest_sha256': PIN,
               'manifest_commit': COMMIT, 'started_utc': '2026-10-28T00:00:00+00:00'}
    summary = {'version': audit.VERSION, 'status': 'passed', 'complete': True, 'manifest_sha256': PIN,
               'manifest_commit': COMMIT, 'manifest_repo_path': REPO_PATH, 'model_metadata': metadata,
               'model_state_sha256_before': 'f' * 64, 'model_state_sha256_after': 'f' * 64,
               'observed_model_calls': audit._observations(), 'output_archive': {}, 'controls': controls,
               'primary_evidence_sha256': primary['evidence_sha256'],
               'finished_utc': '2026-10-28T00:01:00+00:00'}
    for name, value in (('attempt.json', attempt), ('primary-report.json', primary), ('summary.json', summary)):
        (root / name).write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    (root / 'logits.zip').write_bytes(b'synthetic archive, checked separately')
    return summary


def make_archive(path, *, invalid=None, year=2025):
    entries = []
    with zipfile.ZipFile(path, 'x') as output:
        for index, (name, rows) in enumerate(audit.CALLS):
            number = float('nan') if invalid == name else float(index + 1)
            raw = struct.pack('<f', number) * (rows * 4 * audit.WIDTH)
            member = zipfile.ZipInfo(name + '.f32', (year, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(member, raw)
            entries.append({'request_id': name, 'member': member.filename,
                            'shape': [rows, 4, audit.WIDTH], 'dtype': 'little-endian-float32',
                            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return {'raw_bytes': sum(entry['bytes'] for entry in entries),
            'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'entries': entries}


class BatchArchiveAuditTests(unittest.TestCase):
    def test_fixed_six_members_are_reconstructed_and_zip_headers_may_differ(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, 'WIDTH', 7):
            first, second = Path(directory) / 'first.zip', Path(directory) / 'second.zip'
            result_a = audit.audit_archive(first, make_archive(first, year=2025))
            result_b = audit.audit_archive(second, make_archive(second, year=2026))
            self.assertEqual(audit.compare_archives(result_a, result_b),
                             {'matching_member_hashes': True, 'archive_hashes_match': False})
            changed = copy.deepcopy(result_b)
            changed['members'][0]['shape'][0] = True
            self.assertFalse(audit.compare_archives(result_a, changed)['matching_member_hashes'])

    def test_archive_and_member_mutations_fail_even_with_a_refreshed_outer_hash(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, 'WIDTH', 7):
            path = Path(directory) / 'logits.zip'
            info = make_archive(path)
            info['entries'][0]['sha256'] = '0' * 64
            with self.assertRaises(audit.EvidenceError):
                audit.audit_archive(path, info)
            info = make_archive(Path(directory) / 'nan.zip', invalid='mixed-0')
            with self.assertRaises(audit.EvidenceError):
                audit.audit_archive(Path(directory) / 'nan.zip', info)

    def test_shapes_order_names_and_raw_limits_are_type_strict(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, 'WIDTH', 7):
            for kind in ('bool-shape', 'wrong-name', 'wrong-order', 'size', 'duplicate'):
                with self.subTest(kind=kind):
                    path = Path(directory) / (kind + '.zip')
                    info = make_archive(path)
                    if kind == 'bool-shape':
                        info['entries'][0]['shape'][0] = True
                    elif kind == 'wrong-name':
                        info['entries'][0]['request_id'] = 'serial-1-0'
                    elif kind == 'wrong-order':
                        info['entries'][0], info['entries'][1] = info['entries'][1], info['entries'][0]
                    elif kind == 'size':
                        info['raw_bytes'] += 4
                    else:
                        info['entries'][0]['member'] = info['entries'][1]['member']
                    with self.assertRaises(audit.EvidenceError):
                        audit.audit_archive(path, info)

    def test_missing_extra_or_duplicate_zip_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, 'WIDTH', 7):
            path = Path(directory) / 'logits.zip'
            info = make_archive(path)
            with zipfile.ZipFile(path, 'a') as archive:
                archive.writestr('extra.f32', b'1234')
            info['archive_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(audit.EvidenceError):
                audit.audit_archive(path, info)

    def test_corrupt_deflate_is_a_normalized_evidence_error(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, 'WIDTH', 7):
            path = Path(directory) / 'logits.zip'
            info = make_archive(path)
            with zipfile.ZipFile(path) as archive:
                member = archive.infolist()[0]
                offset = member.header_offset
            data = bytearray(path.read_bytes())
            name_length, extra_length = struct.unpack_from('<HH', data, offset + 26)
            first_compressed = offset + 30 + name_length + extra_length
            data[first_compressed] = (data[first_compressed] & ~7) | 7
            path.write_bytes(data)
            info['archive_sha256'] = hashlib.sha256(data).hexdigest()
            with self.assertRaises(audit.EvidenceError):
                audit.audit_archive(path, info)


class BatchManifestAuditTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(FIXTURE.read_bytes())

    def test_producer_generated_synthetic_protocol_matches_the_external_pin(self):
        self.assertEqual(audit.audit_manifest(self.manifest, PIN)['source_commit'], 'a' * 40)
        with self.assertRaises(audit.EvidenceError):
            audit.audit_manifest(self.manifest, '0' * 64)

    def test_cli_sanitizes_cancellation_without_a_private_traceback(self):
        output = io.StringIO()
        args = ['private-run', '--manifest', 'private-manifest', '--expected-manifest-sha256', PIN,
                '--manifest-commit', COMMIT, '--manifest-repo-path', REPO_PATH, '--repository', 'private-repo']
        with patch.object(audit, '_load_json', side_effect=KeyboardInterrupt), redirect_stdout(output):
            self.assertEqual(audit.main(args), 130)
        self.assertEqual(json.loads(output.getvalue())['status'], 'cancelled')
        self.assertNotIn('private-', output.getvalue())

    def test_refreshed_outer_hash_cannot_hide_changed_frozen_inputs_or_runtime(self):
        for kind in ('seed', 'case', 'bool-thread', 'asset', 'source', 'runtime-field', 'env-field'):
            with self.subTest(kind=kind):
                altered = copy.deepcopy(self.manifest)
                if kind == 'seed':
                    altered['primary_protocol']['binding']['seed'] += 1
                elif kind == 'case':
                    altered['primary_protocol']['groups'][0]['cases'][0]['owner'] = 'team-b'
                elif kind == 'bool-thread':
                    altered['runtime']['intraop_threads'] = True
                elif kind == 'asset':
                    altered['asset_sha256']['model.safetensors'] = '0' * 64
                elif kind == 'runtime-field':
                    del altered['runtime']['torch_cpu_capability']
                elif kind == 'env-field':
                    del altered['runtime']['environment']['OMP_DYNAMIC']
                else:
                    del altered['source_sha256']['looking_glass/pretrained.py']
                for key in ('primary_protocol', 'reversal_protocol', 'batch_mean_protocol'):
                    protocol = altered[key]
                    protocol['protocol_sha256'] = audit._digest({k: v for k, v in protocol.items()
                                                                 if k != 'protocol_sha256'})
                altered['manifest_sha256'] = audit._digest({k: v for k, v in altered.items()
                                                            if k != 'manifest_sha256'})
                with self.assertRaises(audit.EvidenceError):
                    audit.audit_manifest(altered, altered['manifest_sha256'])

    def test_git_history_binds_manifest_and_source_blobs_not_just_claimed_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()
            git('init', '-q')
            (root / 'subject.py').write_bytes(b'value = 1\n')
            git('add', 'subject.py')
            git('-c', 'user.name=Audit Test', '-c', 'user.email=audit-test@example.invalid',
                'commit', '-qm', 'source')
            source = git('rev-parse', 'HEAD')
            manifest = {'source_commit': source,
                        'source_sha256': {'subject.py': hashlib.sha256((root / 'subject.py').read_bytes()).hexdigest()}}
            path = root / 'frozen.json'
            path.write_bytes(audit._canonical(manifest) + b'\n')
            git('add', 'frozen.json')
            git('-c', 'user.name=Audit Test', '-c', 'user.email=audit-test@example.invalid',
                'commit', '-qm', 'freeze')
            commit = git('rev-parse', 'HEAD')
            with patch.object(audit, 'SOURCE_PATHS', ('subject.py',)):
                for invalid in ({}, None, []):
                    with self.subTest(invalid=invalid), self.assertRaises(audit.EvidenceError):
                        audit.verify_repository(root, path, invalid, manifest_commit=commit,
                                                manifest_repo_path='frozen.json')
                result = audit.verify_repository(root, path, manifest, manifest_commit=commit,
                                                 manifest_repo_path='frozen.json', verify_worktree=True)
                self.assertFalse(result['publication_before_execution_independently_proven'])
                forged = dict(manifest, extra='not the committed manifest')
                with self.assertRaises(audit.EvidenceError):
                    audit.verify_repository(root, path, forged, manifest_commit=commit,
                                            manifest_repo_path='frozen.json')
                (root / 'subject.py').write_bytes(b'value = 2\n')
                with self.assertRaises(audit.EvidenceError):
                    audit.verify_repository(root, path, manifest, manifest_commit=commit,
                                            manifest_repo_path='frozen.json', verify_worktree=True)
                path.write_bytes(b'{"synthetic":false}\n')
                with self.assertRaises(audit.EvidenceError):
                    audit.verify_repository(root, path, manifest, manifest_commit=commit,
                                            manifest_repo_path='frozen.json')
                with self.assertRaises(audit.EvidenceError):
                    audit.verify_repository(root, path, manifest, manifest_commit=source,
                                            manifest_repo_path='frozen.json')


class BatchTerminalAuditTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(FIXTURE.read_bytes())

    def check(self, root):
        with patch.object(audit, 'audit_archive', return_value={'members': [], 'archive_sha256': 'd' * 64}):
            return audit.audit_run(root, self.manifest, expected_manifest_sha256=PIN,
                                   manifest_commit=COMMIT, manifest_repo_path=REPO_PATH)

    def test_valid_structural_record_does_not_claim_numeric_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, self.manifest)
            result = self.check(root)
            self.assertEqual(result['status'], 'structurally_consistent')
            self.assertFalse(result['numerical_metrics_independently_recomputed'])

    def test_oversized_numeric_metric_is_rejected_without_overflow(self):
        report = synthetic_report(self.manifest['primary_protocol'], 'passed', 'matched')
        report['groups'][0]['cases'][0]['max_abs_error'] = 10 ** 1000
        report['evidence_sha256'] = audit._digest({k: v for k, v in report.items() if k != 'evidence_sha256'})
        with self.assertRaises(audit.EvidenceError):
            audit._audit_report(report, self.manifest['primary_protocol'], 'passed', 'matched')

    def test_finite_float32_results_cannot_report_binary64_overflow(self):
        for field in ('absolute', 'relative'):
            with self.subTest(field=field):
                report = synthetic_report(self.manifest['primary_protocol'], 'passed', 'matched')
                case = report['groups'][0]['cases'][0]
                if field == 'absolute':
                    case.update(max_abs_error=None, absolute_error_overflow=True)
                else:
                    case.update(max_relative_error=None, relative_error_nonfinite_reason='overflow')
                report['evidence_sha256'] = audit._digest({k: v for k, v in report.items() if k != 'evidence_sha256'})
                with self.assertRaises(audit.EvidenceError):
                    audit._audit_report(report, self.manifest['primary_protocol'], 'passed', 'matched')

    def test_wrong_rows_manifest_path_or_cleanup_terminal_do_not_pass(self):
        for kind in ('row', 'bool-row', 'mask', 'path', 'state', 'time', 'nonutc', 'contradictory', 'extra'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = make_run(root, self.manifest)
                if kind == 'row':
                    summary['observed_model_calls'][2]['input_ids_values'].reverse()
                elif kind == 'bool-row':
                    summary['observed_model_calls'][0]['attention_mask_values'][0][0] = True
                elif kind == 'mask':
                    summary['observed_model_calls'][0]['attention_mask_values'][0][0] = 0
                elif kind == 'path':
                    summary['manifest_repo_path'] = 'other/frozen.json'
                elif kind == 'state':
                    summary['model_state_sha256_after'] = '0' * 64
                elif kind == 'time':
                    summary['finished_utc'] = '2026-10-27T00:00:00+00:00'
                elif kind == 'nonutc':
                    summary['finished_utc'] = '2026-10-28T00:01:00-07:00'
                elif kind == 'contradictory':
                    (root / 'failure.json').write_text('{}', encoding='utf-8')
                else:
                    (root / 'summary.json.tmp').write_text('{}', encoding='utf-8')
                (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
                with self.assertRaises(audit.EvidenceError):
                    self.check(root)

    def test_failure_or_missing_terminal_is_never_promoted_to_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = make_run(root, self.manifest)
            summary['status'] = 'failed'
            (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            self.assertEqual(self.check(root)['status'], 'failed')
            (root / 'summary.json').unlink()
            self.assertEqual(self.check(root)['status'], 'incomplete')
            failure = {'version': audit.VERSION, 'status': 'incomplete', 'exception_type': 'RuntimeError',
                       'cleanup_exception_type': None, 'observed_model_calls': [],
                       'finished_utc': '2026-10-28T00:01:00+00:00'}
            (root / 'failure.json').write_text(json.dumps(failure), encoding='utf-8')
            self.assertTrue(self.check(root)['failure_record_present'])

    def test_comparison_reports_incomplete_or_failed_runs_without_promoting_them(self):
        with patch.object(audit, 'audit_run', side_effect=[{'status': 'structurally_consistent'},
                                                          {'status': 'incomplete'}]):
            result = audit.compare_runs('first', 'second', self.manifest, expected_manifest_sha256=PIN,
                                        manifest_commit=COMMIT, manifest_repo_path=REPO_PATH)
        self.assertEqual(result, {'status': 'incomplete_or_nonpassing',
                                  'reference_status': 'structurally_consistent', 'replay_status': 'incomplete'})


if __name__ == '__main__':
    unittest.main()
