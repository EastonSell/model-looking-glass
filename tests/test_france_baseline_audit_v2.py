"""Fabricated journal reader checks; no pretrained or prospective input use."""
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

_root = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_root / "src" if (_root / "src" / "looking_glass").is_dir() else _root))
from looking_glass import targeted_edit_france_baseline_audit_v2 as subject

spec = importlib.util.spec_from_file_location("_fake_france_baseline", pathlib.Path(__file__).with_name("test_france_baseline_v2.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make(directory):
    fixture.torch.set_flush_denormal(False)
    rows, locality = fixture.inputs()
    output = pathlib.Path(directory) / "screen"
    fixture.subject.run_rows(fixture.guard(), rows, locality, 1, 2, output,
                             vocabulary=40, identity={"fixture": True})
    return output, rows, locality


def rewrite(output, events, result):
    event_data = b"".join(canonical(event) for event in events)
    (output / "events.jsonl").write_bytes(event_data)
    result["journal_sha256"] = sha(event_data)
    result_bytes = canonical(result)
    (output / "provisional.json").write_bytes(result_bytes)
    done = json.loads((output / "completed.json").read_bytes())
    done["provisional_sha256"] = sha(result_bytes)
    (output / "completed.json").write_bytes(canonical(done))


class AuditTests(unittest.TestCase):
    def test_full_fixture_recomputes_all_tops_and_fixed_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            output, rows, locality = make(directory)
            report = subject.audit_rows(output, rows, locality, 1, 2, 40)
            self.assertEqual(report["status"], "saved-records-consistent")
            self.assertEqual((report["E"], report["H"], report["H_new"]), (4, 8, 5))
            self.assertEqual(report["forward_completions"], 27)

    def test_rehashed_wrong_entry_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output, rows, locality = make(directory)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            result = json.loads((output / "provisional.json").read_bytes())
            events[1]["actual_attention_mask"][0] = True
            rewrite(output, events, result)
            with self.assertRaises(ValueError):
                subject.audit_rows(output, rows, locality, 1, 2, 40)

    def test_rehashed_logit_change_conflicting_with_saved_science_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output, rows, locality = make(directory)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            result = json.loads((output / "provisional.json").read_bytes())
            raw = bytearray((output / "screen.f32").read_bytes())
            raw[4:8] = (0x41000000).to_bytes(4, "little")
            (output / "screen.f32").write_bytes(raw)
            digest = sha(raw[:160])
            events[1]["sha256"] = digest
            result["row_sha256"][0] = digest
            result["screen_sha256"] = sha(raw)
            rewrite(output, events, result)
            with self.assertRaises(ValueError):
                subject.audit_rows(output, rows, locality, 1, 2, 40)

    def test_strict_json_and_incomplete_marker_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            output, rows, locality = make(directory)
            (output / "started.json").write_bytes(b'{"identity":{},"identity":{},"rows":27}\n')
            with self.assertRaises(ValueError):
                subject.audit_rows(output, rows, locality, 1, 2, 40)
        with tempfile.TemporaryDirectory() as directory:
            output, rows, locality = make(directory)
            (output / "incomplete.json").write_bytes(b"{}\n")
            with self.assertRaises(ValueError):
                subject.audit_rows(output, rows, locality, 1, 2, 40)

    def test_integer_f32_decoder_preserves_signed_subnormals_and_ties(self):
        self.assertEqual(subject.decode_word(1), 2.0 ** -149)
        self.assertEqual(subject.decode_word(0x80000001), -(2.0 ** -149))
        top = subject.top_details([0x80000000, 0x00000000, 0x80000001])
        self.assertEqual(top["top_tie_count"], 2)
        self.assertIsNone(top["unique_top_id"])
        self.assertEqual(top["top_representative_id"], 0)
        with self.assertRaises(ValueError):
            subject.decode_word(0x7fc00000)


if __name__ == "__main__":
    unittest.main()
