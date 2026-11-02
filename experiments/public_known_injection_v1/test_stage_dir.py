"""Fake offline lookups verify checkpoint staging placement and cleanup."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import huggingface_hub
import looking_glass.pretrained as pretrained


class StageDirectoryTests(unittest.TestCase):
    def test_explicit_stage_and_default_cache_preserve_offline_behavior(self):
        with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as stage:
            temporary_directory = tempfile.TemporaryDirectory
            observed = []
            def tracked(*args, **kwargs):
                observed.append(kwargs['dir'])
                return temporary_directory(*args, **kwargs)
            def missing(**kwargs):
                self.assertTrue(kwargs['local_files_only'])
                raise FileNotFoundError('fabricated offline cache miss')
            with patch.object(huggingface_hub, 'hf_hub_download', missing), patch.object(
                    pretrained.tempfile, 'TemporaryDirectory', tracked):
                with self.assertRaises(pretrained.CheckpointUnavailableError):
                    pretrained.load_distilgpt2(cache_dir=cache, stage_dir=stage)
                with self.assertRaises(pretrained.CheckpointUnavailableError):
                    pretrained.load_distilgpt2(cache_dir=cache)
            self.assertEqual(observed, [stage, cache])
            self.assertFalse(list(Path(stage).iterdir()))
            self.assertFalse(list(Path(cache).iterdir()))

    def test_invalid_stage_is_rejected_before_hub_lookup(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(huggingface_hub, 'hf_hub_download') as hub:
                with self.assertRaises(ValueError):
                    pretrained.load_distilgpt2(cache_dir=cache, stage_dir=Path(cache) / 'missing')
                hub.assert_not_called()


if __name__ == '__main__':
    unittest.main()
