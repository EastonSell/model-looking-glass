import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_phase import validate_prefix


class StartupTests(unittest.TestCase):
    def test_requires_matching_fresh_empty_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp).resolve()
            digest = hashlib.sha256(os.path.normcase(str(prefix)).encode()).hexdigest()
            with patch.object(sys, 'dont_write_bytecode', True), patch.object(sys, 'pycache_prefix', str(prefix)):
                self.assertEqual(validate_prefix(digest), prefix)
                with self.assertRaises(ValueError):
                    validate_prefix('0' * 64)
                (prefix / 'preexisting.pyc').write_bytes(b'x')
                with self.assertRaises(ValueError):
                    validate_prefix(digest)


if __name__ == '__main__':
    unittest.main()
