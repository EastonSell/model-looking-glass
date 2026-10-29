"""Exercise the collector's unittest-parent isolation without model packages."""

import marshal
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest


COLLECTOR_PATH = Path(__file__).resolve().parent.parent / "tools" / "bootstrap_pretrained_cpu_evidence.py"
COLLECTOR_NAMESPACE = {"__name__": "_collector_under_test", "__file__": str(COLLECTOR_PATH)}
# The production collector runs as a script. Read its source directly here so
# an ordinary import loader cannot substitute a stale collector .pyc.
exec(compile(COLLECTOR_PATH.read_bytes(), str(COLLECTOR_PATH), "exec"), COLLECTOR_NAMESPACE)


@unittest.skipUnless(sys.implementation.name == "cpython", "test uses CPython timestamp pyc headers")
class CollectorBytecodeTests(unittest.TestCase):
    def test_unittest_parent_ignores_valid_stale_test_bytecode(self):
        # Dropping the collector's private cache prefix would execute this
        # timestamp/size-valid stale code, even when -B disables writes.
        scratch = os.environ.get("LOOKING_GLASS_TEST_SCRATCH") or None
        with tempfile.TemporaryDirectory(prefix="collector-bytecode-test-", dir=scratch) as temp:
            root = Path(temp)
            tests = root / "tests"
            tests.mkdir()
            source = tests / "test_bootstrap_pretrained_cpu.py"
            source.write_text(
                "import unittest\n"
                "class Fixture(unittest.TestCase):\n"
                "    def test_source_is_executed(self):\n"
                "        pass\n",
                encoding="utf-8",
            )
            cached = tests / "__pycache__" / (
                "test_bootstrap_pretrained_cpu." + sys.implementation.cache_tag + ".pyc"
            )
            cached.parent.mkdir()
            py_compile.compile(
                str(source), cfile=str(cached), doraise=True, optimize=0,
                invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
            )
            header = cached.read_bytes()[:16]
            stale = compile("raise RuntimeError('STALE_BYTECODE_EXECUTED')\n", str(source), "exec")
            cached.write_bytes(header + marshal.dumps(stale))

            env = os.environ.copy()
            control = subprocess.run(
                [sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", "tests", "-p",
                 "test_bootstrap_pretrained_cpu.py", "-v"],
                cwd=root, env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(control.returncode, 0)
            self.assertIn("STALE_BYTECODE_EXECUTED", control.stderr)

            prefix = root / "private-prefix"
            prefix.mkdir()
            command_builder = COLLECTOR_NAMESPACE.get("_unittest_command")
            self.assertTrue(callable(command_builder), "collector must provide its unittest command")
            result = subprocess.run(
                command_builder(sys.executable, prefix), cwd=root, env=env,
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Ran 1 test", result.stderr)
            self.assertNotIn("STALE_BYTECODE_EXECUTED", result.stderr)
            self.assertFalse(any(prefix.iterdir()))


if __name__ == "__main__":
    unittest.main()
