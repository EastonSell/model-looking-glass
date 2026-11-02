"""Regression for source-first gate tests sharing a pytest process."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent


class GateSuiteIsolationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'the recorder fixture requires torch')
    def test_gate_cases_do_not_break_later_recorder_or_custodian_tests(self):
        nodes = [
            'tests/test_ipc.py::IPCTest::test_team_configuration_rejects_invalid_or_ambiguous_identities',
            'tests/test_agreement_split_gate_allocation.py::GateCallerTests::test_changed_dependency_executes_no_project_code_or_marker',
            'tests/test_agreement_split_gate_allocation.py::GateCallerTests::test_unreviewed_gate_root_is_rejected_before_lock',
            'tests/test_agreement_split_gate_transition.py::TransitionTests::test_unobserved_commitment_cannot_create_marker',
            'tests/test_batched_recorder.py::test_observer_errors_are_sticky_but_regular_failures_do_not_escape',
            'tests/test_batched_recorder.py::test_interruption_marks_incomplete_and_remains_an_interruption',
            'tests/test_causal_v2_custodian_run.py::CustodianRunTests::test_discovery_freeze_commitment_precede_any_heldout_callback',
        ]
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = '-1'
        env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + str(ROOT / 'tests')
        env.pop('PYTEST_ADDOPTS', None)
        result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *nodes],
                                cwd=ROOT, env=env, capture_output=True, text=True, errors='replace', timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
