import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from looking_glass.experiment import first_difference, run_experiment, summarize_times


class ExperimentTests(unittest.TestCase):
    def test_first_difference_and_metadata_validation(self):
        a = [SimpleNamespace(node_id='x', occurrence=0, value=torch.tensor([1.])),
             SimpleNamespace(node_id='y', occurrence=0, value=torch.tensor([2.]))]
        b = [SimpleNamespace(node_id='x', occurrence=0, value=torch.tensor([1.])),
             SimpleNamespace(node_id='y', occurrence=0, value=torch.tensor([3.]))]
        self.assertEqual(first_difference(a,b), {'last_equal':'x','first_unequal':'y'})
        self.assertIsNone(first_difference(a,a))
        with self.assertRaises(ValueError):
            first_difference(a,b[:1])
        b[0].node_id = 'different'
        with self.assertRaises(ValueError):
            first_difference(a,b)

    def test_timing_summary(self):
        result = summarize_times([1000,2000,3000,4000,5000])
        self.assertEqual(result['median_us'],3.0)
        self.assertEqual(result['samples'],5)
        with self.assertRaises(ValueError):
            summarize_times([])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as path:
            marker = Path(path)/'keep.txt'
            marker.write_text('original')
            with self.assertRaises(FileExistsError):
                run_experiment(path, quick=True)
            self.assertEqual(marker.read_text(),'original')

    def test_explicit_cpu_float32_despite_ambient_default(self):
        from looking_glass.model import TinyResidual
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = TinyResidual(4)
            self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu'
                                for p in model.parameters()))
        finally:
            torch.set_default_dtype(previous)

    def test_started_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as path:
            output = Path(path)/'failure'
            with patch('looking_glass.experiment._evaluate', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    run_experiment(output,quick=True)
            self.assertIn('injected',(output/'failure.json').read_text())

    def test_ambient_cpu_autocast_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            output = Path(path)/'autocast'
            with torch.autocast('cpu'):
                with self.assertRaisesRegex(RuntimeError,'autocast'):
                    run_experiment(output,quick=True)
            self.assertIn('autocast',(output/'failure.json').read_text())

    def test_failed_evidence_write_preserves_original_error(self):
        with tempfile.TemporaryDirectory() as path:
            with patch('looking_glass.experiment._evaluate', side_effect=RuntimeError('original')):
                with patch.object(Path,'write_text',side_effect=OSError('disk failure')):
                    with self.assertRaisesRegex(RuntimeError,'original') as caught:
                        run_experiment(Path(path)/'failed',quick=True)
            self.assertIsInstance(caught.exception.__cause__,OSError)

    def test_quick_smoke_succeeds(self):
        with tempfile.TemporaryDirectory() as path:
            result = run_experiment(Path(path)/'new', quick=True)
            self.assertTrue(result['all_transparent'])
            self.assertTrue(result['channels_equal'])
            self.assertTrue(result['identical_branches_value_ambiguous'])
            self.assertEqual(len(result['timing']),7)
            self.assertEqual(result['protocol_mode'],'smoke, not full protocol')
            self.assertEqual(len((Path(path)/'new'/'raw_timing.csv').read_text().splitlines()),22)


if __name__ == '__main__':
    unittest.main()
