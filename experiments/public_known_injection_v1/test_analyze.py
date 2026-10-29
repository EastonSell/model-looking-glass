"""Saved-byte reader rejects metadata and tensor-shape tampering."""
import json
import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from analyze import discovery, heldout, load_phase
from build_fake_fixture import main as build_fixture


class SavedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.temporary.name) / 'fixture'
        with contextlib.redirect_stdout(io.StringIO()):
            build_fixture(str(cls.fixture))
        cls.cases = json.loads((cls.fixture / 'cases.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_fixture_and_frozen_prediction(self):
        discovery_rows, first = load_phase(self.fixture / 'discovery', self.cases[:8], vocab_size=41, hidden=8)
        frozen = discovery(discovery_rows, self.cases[:8], score_ids=(3, 4))
        held_rows, second = load_phase(self.fixture / 'held-out', self.cases[8:], vocab_size=41, hidden=8)
        outcome = heldout(held_rows, self.cases[8:], frozen, score_ids=(3, 4))
        self.assertEqual(first['actions'] + second['actions'], 240)
        self.assertEqual(outcome['A']['choices']['first_changed'], 1)
        self.assertEqual(outcome['B']['choices']['first_changed'], 4)

    def test_rejects_changed_hook_and_coherently_reshaped_tap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'copy'
            shutil.copytree(self.fixture / 'discovery', root)
            path = root / 'events.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            event = next(row for row in rows if row['type'] == 'raw-recorded' and row['action'] == 'injected')
            event['hook_events'][0]['module_path'] = 'transformer.h.5.mlp'
            path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'metadata or order'):
                load_phase(root, self.cases[:8], vocab_size=41, hidden=8)
            rows = [json.loads(line) for line in (self.fixture / 'discovery' / 'events.jsonl').read_text().splitlines()]
            tensor = next(row for row in rows if row['type'] == 'tensor' and row['tensor']['name'] == 'tap_1')
            ordinal, descriptor = tensor['ordinal'], tensor['tensor']
            descriptor['shape'] = [2, 4]
            raw = next(row for row in rows if row['type'] == 'raw-recorded' and row['ordinal'] == ordinal)
            next(item for item in raw['tensors'] if item['name'] == 'tap_1')['shape'] = [2, 4]
            path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'tensor names or shapes'):
                load_phase(root, self.cases[:8], vocab_size=41, hidden=8)
            rows = [json.loads(line) for line in (self.fixture / 'discovery' / 'events.jsonl').read_text().splitlines()]
            control = next(row for row in rows if row['type'] == 'complete' and row['action'] == 'null-self')
            control['checks']['logits_equal_clean'] = False
            path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'recorded control failed'):
                load_phase(root, self.cases[:8], vocab_size=41, hidden=8)


if __name__ == '__main__':
    unittest.main()
