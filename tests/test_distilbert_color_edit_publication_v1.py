"""Fabricated archive tests with no model or network access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from looking_glass import distilbert_color_edit_publication_v1 as publication


class PublicationTests(unittest.TestCase):
    def test_public_compressed_copy_matches_every_original_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'run'
            run.mkdir()
            (run / 'run.json').write_bytes(b'{}\n')
            stage = run / 'screen'
            stage.mkdir()
            for name, raw in {'claim.json': b'{"external_prelaunch_bytes":1,"prior_public_index_bytes":0}\n',
                              'exit.json': b'{"exit":0}\n',
                              'events.jsonl': b'{"seq":0,"kind":"start"}\n', 'result.json': b'{"ok":true}\n'}.items():
                (stage / name).write_bytes(raw)
            output = root / 'public'
            publication.pack_stage(stage, output, 'screen')
            def git_read(_repo, _commit, path):
                return (output / path.rsplit('/', 1)[-1]).read_bytes()
            with patch.object(publication, 'git_blob', side_effect=git_read):
                receipt = publication.verify_publication(root, 'a' * 40, 'results/test', stage, 'screen')
                self.assertEqual(receipt['files']['events.jsonl']['bytes'], len(b'{"seq":0,"kind":"start"}\n'))
                self.assertEqual(receipt['run_sidecar_bytes'], len(b'{}\n'))
                (run / 'run.json').write_bytes(b'{"tampered":true}\n')
                with self.assertRaisesRegex(ValueError, 'run sidecar'):
                    publication.verify_publication(root, 'a' * 40, 'results/test', stage, 'screen')
                (run / 'run.json').write_bytes(b'{}\n')
                (stage / 'events.jsonl').write_bytes(b'modified\n')
                with self.assertRaisesRegex(ValueError, 'stage bytes differ'):
                    publication.verify_publication(root, 'a' * 40, 'results/test', stage, 'screen')
            with self.assertRaises(FileExistsError):
                publication.pack_stage(stage, output, 'screen')

    def test_archive_index_is_charged_before_any_output_is_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'run'
            run.mkdir()
            (run / 'run.json').write_bytes(b'{}\n')
            stage = run / 'screen'
            stage.mkdir()
            (stage / 'claim.json').write_bytes(b'{"external_prelaunch_bytes":1,"prior_public_index_bytes":0}\n')
            (stage / 'exit.json').write_bytes(b'{}\n')
            output = root / 'public'
            with patch('looking_glass.distilbert_color_edit_runner_v1._inspect_run',
                       return_value=(publication.MAX_BYTES, 0)):
                with self.assertRaisesRegex(ValueError, 'retained-byte cap'):
                    publication.pack_stage(stage, output, 'screen')
            self.assertFalse(output.exists())

    def test_output_cannot_be_created_inside_the_retained_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            root.mkdir()
            (root / 'run.json').write_bytes(b'{}\n')
            stage = root / 'screen'
            stage.mkdir()
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                publication.pack_stage(stage, stage / 'archive', 'screen')
            self.assertFalse((stage / 'archive').exists())


if __name__ == '__main__':
    unittest.main()
