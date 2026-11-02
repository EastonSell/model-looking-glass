"""Small model-free checks for archive reassembly."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name('rebuild_evidence.py')


class RebuildTests(unittest.TestCase):
    def test_reassembles_and_rejects_altered_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks = (b'first fabricated chunk', b'second fabricated chunk')
            parts = []
            for index, data in enumerate(chunks):
                name = 'part-%03d' % index
                (root / name).write_bytes(data)
                parts.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            whole = b''.join(chunks)
            index = {'schema': 'agreement-direction-evidence-archive-v1', 'parts': parts,
                     'archive_bytes': len(whole), 'archive_sha256': hashlib.sha256(whole).hexdigest()}
            (root / 'archive-index.json').write_text(json.dumps(index), encoding='utf-8')
            output = root / 'rebuilt'
            result = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), '--directory', str(root),
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), whole)
            (root / 'part-001').write_bytes(b'altered')
            bad = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), '--directory', str(root),
                                  '--output', str(root / 'bad')], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
