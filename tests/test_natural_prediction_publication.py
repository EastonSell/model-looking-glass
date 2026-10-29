"""Fabricated public-record publication checks; no model or held-out work."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_natural_prediction_freeze import canonical, fabricated_evidence, pretty, replace_binding, replace_supervisor, sha
from looking_glass import natural_prediction_publication as publication


def fake_inputs(tensor_dir):
    inputs, _, _ = fabricated_evidence()
    binding = json.loads(inputs['binding'])
    binding['development_exposure_inventory_sha256'] = '9' * 64
    replace_binding(inputs, binding)
    rows = [json.loads(line) for line in inputs['events'].splitlines()]
    payloads = {}
    for row in rows:
        for ref in row['tensors'].values():
            payload = bytes(ref['bytes'])
            digest = sha(payload)
            ref['sha256'] = digest
            payloads[digest] = payload
    inputs['events'] = b''.join(canonical(row) + b'\n' for row in rows)
    for digest, payload in payloads.items():
        (tensor_dir / (digest + '.f32le')).write_bytes(payload)
    total = sum(map(len, payloads.values()))
    terminal = json.loads(inputs['terminal'])
    terminal['event_log_bytes'] = len(inputs['events'])
    terminal['retained_tensor_bytes'] = total
    inputs['terminal'] = pretty(terminal)
    supervisor = json.loads(inputs['supervisor'])
    supervisor['inventory'].update(events_sha256=sha(inputs['events']), terminal_sha256=sha(inputs['terminal']),
                                   retained_tensor_bytes=total, tensor_sha256=sorted(payloads))
    supervisor['retained_tensor_files'] = {'accounting_complete': True, 'total_bytes': total,
        'files': [{'name': digest + '.f32le', 'bytes': len(payload), 'sha256': digest}
                  for digest, payload in sorted(payloads.items())]}
    replace_supervisor(inputs, supervisor)
    audit = json.loads(inputs['audit'])
    audit.update(journal_event_sha256=sha(inputs['events']), journal_terminal_sha256=sha(inputs['terminal']),
                 tensor_count=len(payloads), tensor_bytes=total)
    inputs['audit'] = canonical(audit)
    return inputs


class NaturalPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tensors = self.root / 'tensors'
        self.tensors.mkdir()
        self.inputs = fake_inputs(self.tensors)
        self.binding_commit = '1' * 40
        self.binding_path = 'results/distilgpt2-natural-agreement-v1-20261028/binding.json'
        self.binding_url = publication.raw_url(self.binding_commit, self.binding_path)
        self.output = self.root / 'stage'
        self.track_receipt = self.root / 'track-publication.json'
        self.aggregate_receipt = self.root / 'aggregate-publication.json'

    def stage(self):
        with mock.patch.object(publication, '_fetch', return_value=self.inputs['binding']):
            return publication.prepare_stage(inputs=self.inputs, tensor_dir=self.tensors,
                                             output_dir=self.output, binding_url=self.binding_url,
                                             wrapper_commit='f' * 40)

    def test_stages_audited_bundle_without_private_paths(self):
        result = self.stage()
        prediction = (self.output / 'natural-prediction-v2.json').read_bytes()
        aggregate = (self.output / 'natural-prediction-freeze.json').read_bytes()
        metadata = json.loads((self.output / 'stage.json').read_bytes())
        self.assertEqual(result['prediction_sha256'], sha(prediction))
        self.assertEqual(result['aggregate_sha256'], sha(aggregate))
        self.assertEqual(metadata['evidence']['events_sha256'], sha(self.inputs['events']))
        self.assertEqual(metadata['evidence']['audit_sha256'], sha(self.inputs['audit']))
        self.assertEqual(metadata['development_exposure_inventory_sha256'], '9' * 64)
        self.assertNotIn(str(self.root), (self.output / 'stage.json').read_text())
        self.assertNotIn(b'owner/supervisor.json', (self.output / 'stage.json').read_bytes())

    def test_rejects_changed_or_unlisted_payloads_before_output(self):
        for mode in ('corrupt', 'extra'):
            with self.subTest(mode=mode):
                inputs = fake_inputs(self.tensors)
                if mode == 'corrupt':
                    next(self.tensors.iterdir()).write_bytes(b'corrupt')
                else:
                    (self.tensors / 'orphan.f32le').write_bytes(b'unknown')
                with mock.patch.object(publication, '_fetch', return_value=inputs['binding']):
                    with self.assertRaises(publication.PublicationError):
                        publication.prepare_stage(inputs=inputs, tensor_dir=self.tensors,
                                                  output_dir=self.output, binding_url=self.binding_url,
                                                  wrapper_commit='f' * 40)
                self.assertFalse(self.output.exists())
                for path in self.tensors.iterdir():
                    path.unlink()

    def test_rejects_wrong_binding_or_failed_audit_without_overwriting(self):
        with mock.patch.object(publication, '_fetch', return_value=b'other'):
            with self.assertRaises(publication.PublicationError):
                publication.prepare_stage(inputs=self.inputs, tensor_dir=self.tensors,
                                          output_dir=self.output, binding_url=self.binding_url,
                                          wrapper_commit='f' * 40)
        self.assertFalse(self.output.exists())
        bad = dict(self.inputs)
        audit = json.loads(bad['audit'])
        audit['scope'] = 'Fabricated synthetic data only.'
        bad['audit'] = canonical(audit)
        with mock.patch.object(publication, '_fetch', return_value=self.inputs['binding']):
            with self.assertRaises(publication.PublicationError):
                publication.prepare_stage(inputs=bad, tensor_dir=self.tensors, output_dir=self.output,
                                          binding_url=self.binding_url, wrapper_commit='f' * 40)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_bytes(b'preserved')
        with mock.patch.object(publication, '_fetch', return_value=self.inputs['binding']):
            with self.assertRaises((publication.PublicationError, FileExistsError)):
                publication.prepare_stage(inputs=self.inputs, tensor_dir=self.tensors,
                                          output_dir=self.output, binding_url=self.binding_url,
                                          wrapper_commit='f' * 40)
        self.assertEqual(marker.read_bytes(), b'preserved')

    def test_failed_last_write_retains_partial_stage_without_returning_success(self):
        write = publication._write_new
        def fail_after_marker(path, raw):
            write(path, raw)
            if path.name == 'stage.json':
                raise OSError('simulated close or sync failure')
        with mock.patch.object(publication, '_fetch', return_value=self.inputs['binding']), \
                mock.patch.object(publication, '_write_new', side_effect=fail_after_marker):
            with self.assertRaises(OSError):
                publication.prepare_stage(inputs=self.inputs, tensor_dir=self.tensors,
                                          output_dir=self.output, binding_url=self.binding_url,
                                          wrapper_commit='f' * 40)
        self.assertTrue((self.output / 'stage.json').is_file())
        self.assertTrue((self.output / 'natural-prediction-v2.json').is_file())

    def test_stage_snapshots_the_input_mapping_before_calling_the_builder(self):
        original = self.inputs['binding']
        validate = publication._tensor_inventory
        def mutate_after_builder(directory, supervisor):
            self.inputs['binding'] = b'changed after validation'
            return validate(directory, supervisor)
        with mock.patch.object(publication, '_fetch', return_value=original), \
                mock.patch.object(publication, '_tensor_inventory', side_effect=mutate_after_builder):
            result = publication.prepare_stage(inputs=self.inputs, tensor_dir=self.tensors,
                                               output_dir=self.output, binding_url=self.binding_url,
                                               wrapper_commit='f' * 40)
        self.assertEqual((self.output / 'runtime-binding.json').read_bytes(), original)
        self.assertEqual(result['runtime_sha256'], sha(original))

    def test_publication_url_does_not_accept_another_host_or_traversal(self):
        for url in ('https://example.com/x', self.binding_url + '?x=1',
                    self.binding_url.replace('binding.json', '../binding.json')):
            with self.subTest(url=url), self.assertRaises(publication.PublicationError):
                publication.parse_binding_url(url)

    def fake_remote(self, ordered=True):
        prediction_commit, head = '2' * 40, '4' * 40
        aggregate_commit, final_head = (('5' * 40, '6' * 40) if ordered else ('3' * 40, head))
        stage = json.loads((self.output / 'stage.json').read_bytes())
        prediction = (self.output / 'natural-prediction-v2.json').read_bytes()
        aggregate = (self.output / 'natural-prediction-freeze.json').read_bytes()
        provenance = (self.output / 'stage.json').read_bytes()
        remote = {self.binding_url: self.inputs['binding'],
                  publication.raw_url(prediction_commit, stage['binding_path']): self.inputs['binding'],
                  publication.raw_url(prediction_commit, stage['prediction_path']): prediction,
                  publication.raw_url(prediction_commit, stage['provenance_path']): provenance,
                  publication.raw_url(head, stage['binding_path']): self.inputs['binding'],
                  publication.raw_url(head, stage['prediction_path']): prediction,
                  publication.raw_url(head, stage['provenance_path']): provenance,
                  publication.raw_url(aggregate_commit, stage['binding_path']): self.inputs['binding'],
                  publication.raw_url(aggregate_commit, stage['prediction_path']): prediction,
                  publication.raw_url(aggregate_commit, stage['provenance_path']): provenance,
                  publication.raw_url(aggregate_commit, stage['aggregate_path']): aggregate,
                  publication.raw_url(final_head, stage['binding_path']): self.inputs['binding'],
                  publication.raw_url(final_head, stage['prediction_path']): prediction,
                  publication.raw_url(final_head, stage['provenance_path']): provenance,
                  publication.raw_url(final_head, stage['aggregate_path']): aggregate,
                  publication._HEAD_URL: canonical({'ref': 'refs/heads/main',
                                                     'object': {'type': 'commit', 'sha': head}})}
        if not ordered:
            del remote[publication.raw_url(head, stage['aggregate_path'])]
        ancestry = [('a' * 40, self.binding_commit), (self.binding_commit, prediction_commit), ('f' * 40, prediction_commit),
                    (prediction_commit, aggregate_commit), (prediction_commit, head), (aggregate_commit, final_head)]
        if ordered:
            ancestry.append((head, aggregate_commit))
        for before, after in ancestry:
            remote[publication._compare_url(before, after)] = canonical(
                {'status': 'ahead', 'base_commit': {'sha': before}, 'merge_base_commit': {'sha': before}})
        def fetch(url, limit, allow_missing=False):
            if url not in remote:
                if allow_missing:
                    return None
                raise publication.PublicationError('missing fake remote')
            return remote[url]
        return prediction_commit, aggregate_commit, remote, fetch

    def test_two_remote_observations_bind_exact_artifacts_and_ancestry(self):
        self.stage()
        prediction_commit, aggregate_commit, remote, fetch = self.fake_remote()
        stage_sha = sha((self.output / 'stage.json').read_bytes())
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            first = publication.verify_track_publication(self.output, stage_sha, prediction_commit, self.track_receipt)
            self.assertEqual(first['prediction_commit'], prediction_commit)
            remote[publication._HEAD_URL] = canonical({'ref': 'refs/heads/main',
                                                        'object': {'type': 'commit', 'sha': '6' * 40}})
            second = publication.verify_aggregate_publication(self.output, stage_sha, aggregate_commit,
                                                              self.track_receipt, self.aggregate_receipt)
        self.assertEqual(second['aggregate_commit'], aggregate_commit)
        self.assertEqual(second['prediction_commit'], prediction_commit)
        self.assertEqual(second['stage_sha256'], stage_sha)
        self.assertTrue(self.aggregate_receipt.is_file())

    def test_rejects_wrong_remote_bytes_and_non_descendant_publication(self):
        self.stage()
        prediction_commit, _, remote, fetch = self.fake_remote()
        stage_sha = sha((self.output / 'stage.json').read_bytes())
        target = publication.raw_url(prediction_commit, publication._PREDICTION)
        remote[target] = b'changed'
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            with self.assertRaises(publication.PublicationError):
                publication.verify_track_publication(self.output, stage_sha, prediction_commit, self.track_receipt)
        self.assertFalse(self.track_receipt.exists())
        remote[target] = (self.output / 'natural-prediction-v2.json').read_bytes()
        remote[publication._compare_url(self.binding_commit, prediction_commit)] = canonical(
            {'status': 'diverged', 'base_commit': {'sha': self.binding_commit},
             'merge_base_commit': {'sha': '0' * 40}})
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            with self.assertRaises(publication.PublicationError):
                publication.verify_track_publication(self.output, stage_sha, prediction_commit, self.track_receipt)
        self.assertFalse(self.track_receipt.exists())

    def test_rejects_binding_that_does_not_descend_from_frozen_source(self):
        self.stage()
        prediction_commit, _, remote, fetch = self.fake_remote()
        stage_sha = sha((self.output / 'stage.json').read_bytes())
        remote[publication._compare_url('a' * 40, self.binding_commit)] = canonical(
            {'status': 'diverged', 'base_commit': {'sha': 'a' * 40},
             'merge_base_commit': {'sha': '0' * 40}})
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            with self.assertRaises(publication.PublicationError):
                publication.verify_track_publication(self.output, stage_sha, prediction_commit, self.track_receipt)
        self.assertFalse(self.track_receipt.exists())

    def test_rejects_changed_stage_and_aggregate_before_track_receipt(self):
        self.stage()
        _, aggregate_commit, _, fetch = self.fake_remote()
        stage_sha = sha((self.output / 'stage.json').read_bytes())
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            with self.assertRaises((publication.PublicationError, OSError)):
                publication.verify_aggregate_publication(self.output, stage_sha, aggregate_commit,
                                                         self.track_receipt, self.aggregate_receipt)
        (self.output / 'natural-prediction-v2.json').write_bytes(b'changed')
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            with self.assertRaises(publication.PublicationError):
                publication.verify_track_publication(self.output, stage_sha, '2' * 40, self.track_receipt)
        self.assertFalse(self.aggregate_receipt.exists())

    def test_stage_rejects_self_consistent_metadata_with_wrong_evidence_link(self):
        self.stage()
        path = self.output / 'stage.json'
        metadata = json.loads(path.read_bytes())
        metadata['evidence']['events_sha256'] = '0' * 64
        path.write_bytes(canonical(metadata))
        with self.assertRaises(publication.PublicationError):
            publication._read_stage(self.output, sha(path.read_bytes()))

    def test_aggregate_rechecks_track_observation_and_prior_absence(self):
        for change in ('unrelated_observed_head', 'aggregate_already_present', 'binding_changed'):
            with self.subTest(change=change):
                if self.output.exists():
                    for path in self.output.iterdir():
                        path.unlink()
                    self.output.rmdir()
                if self.track_receipt.exists():
                    self.track_receipt.unlink()
                if self.aggregate_receipt.exists():
                    self.aggregate_receipt.unlink()
                self.stage()
                prediction_commit, aggregate_commit, remote, fetch = self.fake_remote()
                stage_sha = sha((self.output / 'stage.json').read_bytes())
                with mock.patch.object(publication, '_fetch', side_effect=fetch):
                    publication.verify_track_publication(self.output, stage_sha, prediction_commit, self.track_receipt)
                    remote[publication._HEAD_URL] = canonical({'ref': 'refs/heads/main',
                                                                'object': {'type': 'commit', 'sha': '6' * 40}})
                    if change == 'unrelated_observed_head':
                        track = json.loads(self.track_receipt.read_bytes())
                        track['observed_main_head'] = '5' * 40
                        self.track_receipt.write_bytes(canonical(track))
                    elif change == 'aggregate_already_present':
                        remote[publication.raw_url(prediction_commit, publication._AGGREGATE)] = b'previous'
                    else:
                        remote[self.binding_url] = b'changed original binding'
                    with self.assertRaises(publication.PublicationError):
                        publication.verify_aggregate_publication(self.output, stage_sha, aggregate_commit,
                                                                 self.track_receipt, self.aggregate_receipt)
                self.assertFalse(self.aggregate_receipt.exists())

    def test_prediction_observation_rejects_existing_aggregate_on_main_head(self):
        self.stage()
        prediction_commit, _, remote, fetch = self.fake_remote()
        stage_sha = sha((self.output / 'stage.json').read_bytes())
        remote[publication.raw_url('4' * 40, publication._AGGREGATE)] = b'published too early'
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            with self.assertRaises(publication.PublicationError):
                publication.verify_track_publication(self.output, stage_sha, prediction_commit, self.track_receipt)
        self.assertFalse(self.track_receipt.exists())

    def test_aggregate_must_descend_from_the_observed_track_head(self):
        self.stage()
        prediction_commit, aggregate_commit, _, fetch = self.fake_remote(ordered=False)
        stage_sha = sha((self.output / 'stage.json').read_bytes())
        with mock.patch.object(publication, '_fetch', side_effect=fetch):
            publication.verify_track_publication(self.output, stage_sha, prediction_commit, self.track_receipt)
            # This fake aggregate was already an ancestor of the observed head;
            # prediction -> aggregate -> head does not establish the phase order.
            with self.assertRaises(publication.PublicationError):
                publication.verify_aggregate_publication(self.output, stage_sha, aggregate_commit,
                                                         self.track_receipt, self.aggregate_receipt)
        self.assertFalse(self.aggregate_receipt.exists())

    def test_observed_main_heads_must_still_contain_exact_artifacts(self):
        for phase in ('track', 'aggregate'):
            with self.subTest(phase=phase):
                if self.output.exists():
                    for path in self.output.iterdir():
                        path.unlink()
                    self.output.rmdir()
                if self.track_receipt.exists():
                    self.track_receipt.unlink()
                self.stage()
                prediction_commit, aggregate_commit, remote, fetch = self.fake_remote()
                stage_sha = sha((self.output / 'stage.json').read_bytes())
                with mock.patch.object(publication, '_fetch', side_effect=fetch):
                    if phase == 'track':
                        remote[publication.raw_url('4' * 40, publication._PREDICTION)] = b'changed on main'
                        with self.assertRaises(publication.PublicationError):
                            publication.verify_track_publication(self.output, stage_sha, prediction_commit,
                                                                 self.track_receipt)
                    else:
                        publication.verify_track_publication(self.output, stage_sha, prediction_commit,
                                                             self.track_receipt)
                        remote[publication._HEAD_URL] = canonical({'ref': 'refs/heads/main',
                                                                    'object': {'type': 'commit', 'sha': '6' * 40}})
                        remote[publication.raw_url('6' * 40, publication._AGGREGATE)] = b'changed on main'
                        with self.assertRaises(publication.PublicationError):
                            publication.verify_aggregate_publication(self.output, stage_sha, aggregate_commit,
                                                                     self.track_receipt, self.aggregate_receipt)
                self.assertFalse(self.aggregate_receipt.exists())


if __name__ == '__main__':
    unittest.main()
