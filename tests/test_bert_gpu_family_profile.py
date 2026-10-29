"""Synthetic CPU tests for the fixed-family profile and bounded journal."""

import os
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from looking_glass.bert_agreement_runner import decode_float32_bits, run_observation
from looking_glass.bert_gpu_family_profile import (MAX_RECORD_BYTES, _read_profile_journal,
                                                    execute_family_profile)
from test_bert_gpu_study_adapter import family_manifest


class FamilyProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        config = DistilBertConfig(vocab_size=64, dim=16, hidden_dim=32, n_layers=6,
            n_heads=4, dropout=0., attention_dropout=0., max_position_embeddings=32,
            attn_implementation='eager')
        cls.model = DistilBertForMaskedLM(config).eval()

    def test_full_family_journal_and_checkpoints_are_reconstructed(self):
        checkpoints = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            result = execute_family_profile(self.model, family_manifest(), observer=run_observation,
                checkpoint=lambda stage, number: checkpoints.append((stage, number)),
                journal_path=path, is_id=20, are_id=21, mask_id=7, hidden_size=16,
                expected_device='cpu')
            self.assertEqual(result['forward_count'], 140)
            self.assertEqual(checkpoints, [(stage, number) for number in range(1, 141)
                                           for stage in ('before', 'after')])
            records, digest = _read_profile_journal(path)
            self.assertEqual(digest, result['journal_sha256'])
            self.assertEqual(len(records), 328)
            self.assertEqual(sum(item['status'] == 'completed' for item in records), 140)
            self.assertEqual(sum(item['status'] == 'control' for item in records), 48)
            self.assertEqual(result['live_cases'], result['audited_cases'])

    def test_failed_checkpoint_keeps_started_and_failed_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            with patch.object(self.model, 'forward', wraps=self.model.forward) as model_forward:
                with self.assertRaisesRegex(RuntimeError, 'checkpoint failed'):
                    execute_family_profile(self.model, family_manifest(), observer=run_observation,
                        checkpoint=lambda stage, number: (_ for _ in ()).throw(RuntimeError('checkpoint failed')),
                        journal_path=path, is_id=20, are_id=21, mask_id=7, hidden_size=16,
                        expected_device='cpu')
                model_forward.assert_not_called()
            records, _ = _read_profile_journal(path)
            self.assertEqual([record['status'] for record in records], ['started', 'failed'])

    def test_journal_reader_rejects_duplicate_keys_truncation_and_oversize(self):
        bad_values = (b'{"a":1,"a":2}\n', b'{"a":1}', b'{"a":NaN}\n',
                      b' ' * (12 * 1024 + 1) + b'\n', b'\n' * 100000,
                      b'{}\r\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jsonl'
            for raw in bad_values:
                with self.subTest(size=len(raw), prefix=raw[:20]):
                    path.write_bytes(raw)
                    with self.assertRaises(ValueError):
                        _read_profile_journal(path)

    def test_saved_real_width_patch_records_fit_with_gpu_device_labels(self):
        path = (Path(__file__).resolve().parents[1] / 'results' / 'bert-agreement-dev-v1-20261028'
                / 'attempt-01' / 'events.jsonl.gz')
        with gzip.open(path, 'rb') as stream:
            records = [json.loads(line) for line in stream]
        patches = [record for record in records if record.get('status') == 'completed'
                   and record.get('after_float32_le_base64') is not None]
        self.assertEqual(len(patches), 88)
        for record in patches:
            for event in record['events']:
                event['device'] = 'cuda:0'
            record['is_bits'], record['are_bits'] = 0xff7fffff, 0x80000001
            record['is_logit'] = decode_float32_bits(record['is_bits'])
            record['are_logit'] = decode_float32_bits(record['are_bits'])
            encoded = json.dumps(record, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=True, allow_nan=False).encode('ascii')
            self.assertLessEqual(len(encoded), MAX_RECORD_BYTES)


if __name__ == '__main__':
    unittest.main()
