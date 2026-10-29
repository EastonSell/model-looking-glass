import copy
import unittest
from unittest.mock import Mock

import natural_transfer_direct as direct


class DirectObserverTests(unittest.TestCase):
    def good(self):
        receipt = {'phase': 'discovery', 'exit_code': 0, 'interrupted': False,
                   'prefix_empty_after': True, 'stage_empty_after': True, 'stderr': {'bytes': 0},
                   'run_files': {'started.json': {}, 'events.jsonl': {}, 'terminal.json': {}}}
        terminal = {'status': 'complete_candidate', 'storage_accounting_complete': True, 'events': 97,
                    'detail': {'status': 'complete_candidate', 'phase': 'discovery', 'forwards': 96,
                               'displacement_inventory': {'complete': True, 'files': [{}] * 96}}}
        return receipt, terminal

    def test_candidate_requires_exit_phase_and_complete_accounting(self):
        receipt, terminal = self.good()
        self.assertTrue(direct.assess_receipt(receipt, terminal))
        for field, value in (('exit_code', False), ('exit_code', 1), ('interrupted', True),
                             ('prefix_empty_after', False), ('stage_empty_after', False),
                             ('stderr', {'bytes': 1}), ('stderr', {'bytes': False}), ('run_files', {}), ('phase', 'evaluation')):
            changed = dict(receipt, **{field: value})
            self.assertFalse(direct.assess_receipt(changed, terminal), field)
        for field, value in (('status', 'failed'), ('events', 96), ('events', True),
                             ('storage_accounting_complete', False)):
            changed = dict(terminal, **{field: value})
            self.assertFalse(direct.assess_receipt(receipt, changed), field)
        changed = copy.deepcopy(terminal)
        changed['detail']['displacement_inventory']['files'].pop()
        self.assertFalse(direct.assess_receipt(receipt, changed))

    def test_start_progress_failure_terminates_and_waits(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.return_value = 1
        def broken(*args, **kwargs):
            raise BrokenPipeError('fabricated')
        observed = direct.observe_child(['fake'], cwd='.', env={}, stdout=None, stderr=None,
                                        popen=lambda *args, **kwargs: child, progress=broken)
        self.assertTrue(observed['interrupted'])
        self.assertEqual(observed['launcher_error_type'], 'BrokenPipeError')
        child.terminate.assert_called_once()
        child.wait.assert_called_once()

    def test_lost_spawn_return_remains_uncertain(self):
        def lost(*args, **kwargs):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            direct.observe_child(['fake'], cwd='.', env={}, stdout=None, stderr=None, popen=lost)


if __name__ == '__main__':
    unittest.main()
