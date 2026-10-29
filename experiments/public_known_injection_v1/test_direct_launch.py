import unittest
from unittest.mock import Mock

from direct_launch import assess_receipt, observe_child


class DirectLaunchTests(unittest.TestCase):
    def test_acceptance_requires_exact_evidence_exit_prefix_and_no_stderr(self):
        files = {'events.jsonl': {}, 'phase-summary.json': {}}
        files.update({f'known-injection-{i:02d}.f32': {} for i in range(1, 9)})
        receipt = {'phase': 'discovery', 'exit_code': 0, 'interrupted': False,
                   'prefix_empty_after': True, 'stderr': {'bytes': 0},
                   'failure_file_present': False, 'run_files': files}
        summary = {'phase': 'discovery', 'cases': 8, 'completed_actions': 120,
                   'raw_bytes': 42, 'metadata_bytes': 42, 'status': 'complete'}
        self.assertTrue(assess_receipt(receipt, summary))
        for field, value in (('interrupted', True), ('prefix_empty_after', False),
                             ('exit_code', 1), ('failure_file_present', True)):
            altered = dict(receipt, **{field: value})
            self.assertFalse(assess_receipt(altered, summary))
        self.assertFalse(assess_receipt(dict(receipt, stderr={'bytes': 1}), summary))
        self.assertFalse(assess_receipt(dict(receipt, run_files={}), summary))

    def test_progress_failure_terminates_and_reaps_child(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.return_value = 1
        def broken(*args, **kwargs):
            raise BrokenPipeError('fabricated')
        out = observe_child(['fake'], cwd='.', env={}, stdout=None, stderr=None,
                            popen=lambda *args, **kwargs: child, progress=broken)
        self.assertTrue(out['interrupted'])
        self.assertEqual(out['launcher_error_type'], 'BrokenPipeError')
        child.terminate.assert_called_once()
        child.wait.assert_called_once()

    def test_lost_popen_return_is_uncertain_and_propagates(self):
        def lost(*args, **kwargs):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            observe_child(['fake'], cwd='.', env={}, stdout=None, stderr=None, popen=lost)


if __name__ == '__main__':
    unittest.main()
