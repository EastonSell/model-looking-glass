"""Harness contract tests use ordinary synthetic outcomes, never opcode faults."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


class ExperimentalOpcodeHarnessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="mlg-harness-contract-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "tests").mkdir()
        shutil.copyfile(Path(__file__).with_name("conftest.py"), self.root / "tests/conftest.py")
        (self.root / "pytest.ini").write_text("[pytest]\n")
        self.source = self.root / "tests/test_cases.py"
        self.source.write_text('''import os, time, pytest, unittest

def test_supported():
    assert True

def test_experiment():
    assert True

def test_skipped():
    pytest.skip("genuine platform skip")

def test_failure():
    assert False, "original assertion preserved"

def test_exit():
    os._exit(7)

def test_timeout():
    time.sleep(5)

@pytest.mark.parametrize("value", [1, 2])
def test_parameter(value):
    assert value > 0

class Case(unittest.TestCase):
    def test_method(self):
        self.assertTrue(True)
''')

    def run_cases(self, nodes, *arguments):
        (self.root / "tests/experimental_opcode_faults.json").write_text(json.dumps({"schema_version": 1, "nodes": nodes}))
        env = os.environ.copy()
        for key in ("MLG_EXPERIMENTAL_OPCODE_CHILD_NODE", "MLG_EXPERIMENTAL_OPCODE_CHILD_REPORT", "PYTEST_ADDOPTS", "PYTHONPATH"):
            env.pop(key, None)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *arguments],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=25)
        return result.returncode, result.stdout + result.stderr

    def test_default_excludes_only_manifest_nodes(self):
        code, output = self.run_cases(["tests/test_cases.py::test_experiment"], "tests/test_cases.py::test_supported", "tests/test_cases.py::test_experiment")
        self.assertEqual(code, 0, output)
        self.assertIn("1 passed, 1 skipped", output)
        self.assertIn("1 selected; excluded by default", output)

    def test_opt_in_preserves_unittest_and_parameterized_selection(self):
        nodes = ["tests/test_cases.py::Case::test_method", "tests/test_cases.py::test_parameter[2]"]
        code, output = self.run_cases(nodes, *nodes, "--run-experimental-opcode-faults")
        self.assertEqual(code, 0, output)
        self.assertIn("2 passed", output)

    def test_original_skip_is_not_pass(self):
        node = "tests/test_cases.py::test_skipped"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults", "-rs")
        self.assertEqual(code, 0, output)
        self.assertIn("1 skipped", output)
        self.assertIn("genuine platform skip", output)

    def test_original_failure_propagates(self):
        node = "tests/test_cases.py::test_failure"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults")
        self.assertEqual(code, 1, output)
        self.assertIn("original assertion preserved", output)

    def test_nonzero_exit_fails(self):
        node = "tests/test_cases.py::test_exit"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults")
        self.assertEqual(code, 1, output)
        self.assertIn("exited 7", output)

    def test_timeout_fails(self):
        node = "tests/test_cases.py::test_timeout"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults", "--experimental-opcode-timeout=0.5")
        self.assertEqual(code, 1, output)
        self.assertIn("timed out", output)

    def test_unknown_base_node_rejected(self):
        code, output = self.run_cases(["tests/test_cases.py::test_nonexistent"], "--collect-only")
        self.assertEqual(code, 4, output)
        self.assertIn("unknown experimental node", output)

    def test_unknown_parameter_id_rejected(self):
        code, output = self.run_cases(["tests/test_cases.py::test_parameter[missing]"], "--collect-only")
        self.assertEqual(code, 4, output)
        self.assertIn("unknown experimental node IDs", output)

    def test_missing_child_report_is_failure(self):
        self.source.write_text("import os\ndef test_no_report():\n    os._exit(0)\n")
        node = "tests/test_cases.py::test_no_report"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults")
        self.assertEqual(code, 1, output)
        self.assertIn("Incomplete or failed experimental child lifecycle", output)

    def test_skipped_call_with_missing_teardown_fails(self):
        self.source.write_text("import os, pytest\n@pytest.fixture\ndef abort_teardown():\n    yield\n    os._exit(0)\ndef test_incomplete(abort_teardown):\n    pytest.skip('skip before teardown')\n")
        node = "tests/test_cases.py::test_incomplete"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults")
        self.assertEqual(code, 1, output)
        self.assertIn("Incomplete or failed experimental child lifecycle", output)

    def test_setup_skip_is_preserved_after_teardown(self):
        self.source.write_text("import pytest\n@pytest.fixture\ndef unavailable():\n    pytest.skip('setup unavailable')\ndef test_setup(unavailable):\n    pass\n")
        node = "tests/test_cases.py::test_setup"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults", "-rs")
        self.assertEqual(code, 0, output)
        self.assertIn("1 skipped", output)
        self.assertIn("setup unavailable", output)

    def test_malformed_report_is_failure_not_internal_error(self):
        self.source.write_text("import os\ndef test_malformed():\n    with open(os.environ['MLG_EXPERIMENTAL_OPCODE_CHILD_REPORT'], 'a') as f:\n        f.write('{}\\n')\n    os._exit(0)\n")
        node = "tests/test_cases.py::test_malformed"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults")
        self.assertEqual(code, 1, output)
        self.assertIn("Malformed experimental child lifecycle", output)
        self.assertNotIn("INTERNALERROR", output)

    @unittest.skipUnless(os.name == "posix", "POSIX owned process-group timeout cleanup")
    def test_timeout_kills_owned_descendant_group(self):
        marker = self.root / "descendant-ran"
        started = self.root / "descendant-started"
        script = f"import time; from pathlib import Path; Path({str(started)!r}).write_text('started'); time.sleep(1.5); Path({str(marker)!r}).write_text('escaped')"
        self.source.write_text(f"import subprocess, sys, time\ndef test_descendant():\n    subprocess.Popen([sys.executable, '-c', {script!r}])\n    time.sleep(5)\n")
        node = "tests/test_cases.py::test_descendant"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults", "--experimental-opcode-timeout=0.5")
        self.assertEqual(code, 1, output)
        self.assertIn("timed out", output)
        self.assertTrue(started.exists(), "descendant did not start before timeout")
        time.sleep(1.6)
        self.assertFalse(marker.exists(), "owned descendant survived timeout")

    @unittest.skipUnless(os.name == "posix", "POSIX signal result")
    def test_native_signal_is_failure(self):
        self.source.write_text("import os, signal\ndef test_signal():\n    os.kill(os.getpid(), signal.SIGTERM)\n")
        node = "tests/test_cases.py::test_signal"
        code, output = self.run_cases([node], node, "--run-experimental-opcode-faults")
        self.assertEqual(code, 1, output)
        self.assertIn("exited -15", output)


if __name__ == "__main__":
    unittest.main()
