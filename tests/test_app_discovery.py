"""Metadata-only local discovery; no ML packages, model loads or network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from looking_glass import app_discovery as discovery


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "models"
        self.root.mkdir()

    def checkpoint(self, name="model", family="gpt2", **extra):
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        config = {"model_type": family, "architectures": [discovery.HEADS[family]]}
        config.update(extra)
        (path / "config.json").write_text(json.dumps(config))
        (path / "model.safetensors").write_bytes(b"not parsed by discovery")
        (path / ("preprocessor_config.json" if family == "vit" else "tokenizer.json")).write_text("not parsed")
        return path

    def browse(self, path=None):
        return discovery.browse_model_folders(str(path or self.root), str(self.root))

    def test_five_exact_family_candidates_are_not_load_verification(self):
        for family in discovery.HEADS:
            self.checkpoint(family, family)
        result = self.browse()
        self.assertEqual({r["candidate"]["family"] for r in result["folders"]}, set(discovery.HEADS))
        for row in result["folders"]:
            self.assertEqual(row["candidate"]["status"], "candidate")
            self.assertIn("Inspect", row["candidate"]["reason"])
        self.assertEqual(result["mode"], "directory")

    def test_selected_checkpoint_itself_is_identified(self):
        path = self.checkpoint()
        result = self.browse(path)
        self.assertEqual(result["folders"], [])
        self.assertEqual(result["current"]["candidate"]["status"], "candidate")
        self.assertEqual(result["current"]["path"], str(path))

    def test_empty_browse_offers_locations_without_enumerating(self):
        with mock.patch.object(discovery.os, "scandir", side_effect=AssertionError("must not scan")):
            result = discovery.browse_model_folders()
        self.assertEqual(result["mode"], "locations")
        self.assertIsNone(result["path"])
        self.assertTrue(all(set(row) == {"name", "path"} for row in result["locations"]))

    def test_absent_head_is_unverified_not_fabricated(self):
        path = self.checkpoint(architectures=None)
        row = self.browse(path)["current"]
        self.assertEqual(row["candidate"]["status"], "unverified")
        self.assertIsNone(row["candidate"]["architecture"])

    def test_wrong_head_custom_code_quantization_and_other_family_rejected(self):
        variants = [dict(architectures=["GPT2Model"]), dict(auto_map={"AutoModel": "x.X"}),
                    dict(quantization_config={"bits": 4}), dict(model_type="qwen2")]
        for index, variant in enumerate(variants):
            path = self.checkpoint(str(index), **variant)
            self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "unsupported")

    def test_llama_unsupported_rope_and_partial_rotary_not_candidates(self):
        for index, extra in enumerate((dict(rope_scaling={"rope_type": "dynamic"}),
                                       dict(partial_rotary_factor=0.5),
                                       dict(rope_scaling={"rope_type": "llama3", "factor": float("inf")}))):
            path = self.checkpoint(str(index), "llama", **extra)
            self.assertIn(self.browse(path)["current"]["candidate"]["status"], {"unsupported", "unreadable"})

    def test_missing_weights_or_input_assets_are_incomplete(self):
        for index, missing in enumerate(("model.safetensors", "tokenizer.json")):
            path = self.checkpoint(str(index)); (path / missing).unlink()
            self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "incomplete")

    def test_shard_index_presence_is_only_unverified_weight_hint(self):
        path = self.checkpoint(); (path / "model.safetensors").unlink()
        (path / "model.safetensors.index.json").write_text("invalid index not parsed")
        row = self.browse(path)["current"]["candidate"]
        self.assertEqual(row["status"], "candidate")
        self.assertIn("not validated", row["reason"])

    def test_malformed_duplicate_nonobject_and_large_json_are_local_failures(self):
        for index, raw in enumerate((b"{", b'{"model_type":"qwen2","model_type":"gpt2"}', b"[]",
                                     b" " * (discovery.MAX_CONFIG_BYTES + 1))):
            path = self.checkpoint(str(index)); (path / "config.json").write_bytes(raw)
        result = self.browse()
        self.assertEqual(len(result["folders"]), 4)
        self.assertTrue(all(r["candidate"]["status"] == "unreadable" for r in result["folders"]))

    def test_no_config_is_ordinary_folder(self):
        (self.root / "plain").mkdir()
        row = self.browse()["folders"][0]
        self.assertFalse(row["is_checkpoint"])
        self.assertEqual(row["candidate"]["status"], "none")

    def test_configured_root_cannot_be_overridden(self):
        outside = self.root.parent / "outside"; outside.mkdir()
        with self.assertRaisesRegex(ValueError, "root"):
            discovery.browse_model_folders(str(outside), str(outside), models_dir=self.root)
        self.assertEqual(discovery.browse_model_folders(models_dir=self.root)["path"], str(self.root))

    def test_missing_and_file_roots_rejected(self):
        for path in (self.root / "missing", self.root / "file"):
            if path.name == "file": path.write_text("x")
            with self.assertRaises((OSError, ValueError)):
                discovery.browse_model_folders(str(path))

    def test_no_recursion(self):
        self.checkpoint("nested/deep/model")
        result = self.browse()
        self.assertEqual(len(result["folders"]), 1)
        self.assertEqual(result["folders"][0]["candidate"]["status"], "none")

    def test_folder_and_enumeration_bounds(self):
        for i in range(205): (self.root / f"folder{i:03}").mkdir()
        result = self.browse()
        self.assertEqual(len(result["folders"]), discovery.MAX_FOLDERS)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(result["discovery"]["entries_examined"], discovery.MAX_ENTRIES)

    def test_aggregate_config_read_budget(self):
        for i in range(50):
            path = self.checkpoint(str(i)); config = json.loads((path / "config.json").read_text())
            config["padding"] = "x" * 60000; (path / "config.json").write_text(json.dumps(config))
        result = self.browse()
        self.assertLessEqual(result["discovery"]["config_bytes_read"], discovery.MAX_TOTAL_CONFIG_BYTES)
        self.assertTrue(result["discovery"]["budget_exhausted"])
        self.assertTrue(any(r["candidate"]["status"] == "unreadable" for r in result["folders"]))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_directory_links_skipped_and_config_escape_unreadable(self):
        outside = self.root.parent / "outside"; outside.mkdir()
        (outside / "config.json").write_text('{"model_type":"gpt2"}')
        (self.root / "escape").symlink_to(outside, target_is_directory=True)
        path = self.checkpoint(); (path / "config.json").unlink()
        (path / "config.json").symlink_to(outside / "config.json")
        result = self.browse()
        self.assertEqual([r["name"] for r in result["folders"]], ["model"])
        self.assertEqual(result["folders"][0]["candidate"]["status"], "unreadable")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_in_root_hf_style_config_symlink_allowed(self):
        path = self.checkpoint("snapshots/revision")
        blob = self.root / "blob.json"; (path / "config.json").rename(blob)
        (path / "config.json").symlink_to(blob)
        self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "candidate")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO unavailable")
    def test_fifo_never_opened(self):
        path = self.root / "fifo"; path.mkdir(); os.mkfifo(path / "config.json")
        with mock.patch.object(discovery.os, "open", side_effect=AssertionError("FIFO must not open")):
            result = self.browse(path)
        self.assertEqual(result["current"]["candidate"]["status"], "unreadable")

    def test_deep_json_and_nonfinite_constants_do_not_abort_other_folders(self):
        for index, raw in enumerate((b'{"x":NaN}', b'{"x":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}')):
            path = self.checkpoint(str(index)); (path / "config.json").write_bytes(raw)
        self.checkpoint("valid")
        rows = self.browse()["folders"]
        self.assertEqual(rows[0]["candidate"]["status"], "unreadable")
        # Newer JSON decoders may support this depth; it is still bounded and not a model config.
        self.assertIn(rows[1]["candidate"]["status"], {"unreadable", "unsupported"})
        self.assertEqual(rows[2]["candidate"]["status"], "candidate")

    def test_json_decoder_recursion_failure_is_local(self):
        path = self.checkpoint()
        with mock.patch.object(discovery.json, "loads", side_effect=RecursionError("too deep")):
            self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "unreadable")

    def test_overflowing_json_numeric_literal_is_not_green(self):
        path = self.checkpoint()
        (path / "config.json").write_text('{"model_type":"gpt2","architectures":["GPT2LMHeadModel"],"overflow":1e999}')
        self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "unreadable")

    def test_exact_per_file_limit_can_be_read_without_overreading(self):
        path = self.checkpoint()
        raw = (path / "config.json").read_bytes()
        (path / "config.json").write_bytes(raw + b" " * (discovery.MAX_CONFIG_BYTES - len(raw)))
        result = self.browse(path)
        self.assertEqual(result["current"]["candidate"]["status"], "candidate")
        self.assertEqual(result["discovery"]["config_bytes_read"], discovery.MAX_CONFIG_BYTES)

    def test_inaccessible_config_is_local_and_does_not_change_permissions(self):
        bad = self.checkpoint("bad"); self.checkpoint("good")
        real_open = discovery.os.open
        def deny(path, *args, **kwargs):
            if path == bad / "config.json": raise PermissionError("synthetic denied")
            return real_open(path, *args, **kwargs)
        with mock.patch.object(discovery.os, "open", side_effect=deny):
            rows = self.browse()["folders"]
        self.assertEqual([r["candidate"]["status"] for r in rows], ["unreadable", "candidate"])

    def test_oversized_config_is_rejected_before_open(self):
        path = self.checkpoint(); (path / "config.json").write_bytes(b"x" * (discovery.MAX_CONFIG_BYTES + 1))
        with mock.patch.object(discovery.os, "open", side_effect=AssertionError("oversized file must not open")):
            row = self.browse(path)["current"]
        self.assertEqual(row["candidate"]["status"], "unreadable")

    def test_config_growth_after_stat_still_reads_at_most_hard_limit(self):
        path = self.checkpoint(); config = path / "config.json"
        real_open = discovery.os.open
        def grow(candidate, *args, **kwargs):
            if candidate == config:
                config.write_bytes(b"x" * (discovery.MAX_CONFIG_BYTES * 3))
            return real_open(candidate, *args, **kwargs)
        with mock.patch.object(discovery.os, "open", side_effect=grow):
            result = self.browse(path)
        self.assertEqual(result["current"]["candidate"]["status"], "unreadable")
        self.assertLessEqual(result["discovery"]["config_bytes_read"], discovery.MAX_CONFIG_BYTES + 1)

    def test_static_rope_is_not_advertised_as_validated(self):
        path = self.checkpoint("llama3", "llama", rope_scaling={"rope_type": "llama3", "factor": 8,
              "low_freq_factor": 1, "high_freq_factor": 4, "original_max_position_embeddings": 8192})
        row = self.browse(path)["current"]["candidate"]
        self.assertEqual(row["status"], "unverified")
        self.assertIn("geometry", row["reason"])

    def test_malformed_architecture_types_are_not_green(self):
        for index, architecture in enumerate(({}, "GPT2LMHeadModel", [], ["GPT2LMHeadModel", "Other"])):
            path = self.checkpoint(str(index), architectures=architecture)
            self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "unsupported")

    def test_malformed_family_types_are_not_green(self):
        for index, family in enumerate(([], {}, True, 7)):
            path = self.checkpoint(str(index), model_type=family)
            self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "unsupported")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_config_symlink_cycle_is_local_unreadable(self):
        path = self.checkpoint(); (path / "config.json").unlink()
        (path / "config.json").symlink_to(path / "config.json")
        self.assertEqual(self.browse(path)["current"]["candidate"]["status"], "unreadable")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_directory_alias_inside_root_is_not_followed(self):
        path = self.checkpoint()
        (self.root / "alias").symlink_to(path, target_is_directory=True)
        self.assertEqual([r["name"] for r in self.browse()["folders"]], ["model"])

    def test_selected_nonexistent_paths_do_not_create_directories(self):
        missing = self.root / "missing"
        with self.assertRaises(OSError): discovery.browse_model_folders(str(missing))
        self.assertFalse(missing.exists())

    def test_entry_enumeration_cap_counts_files_too(self):
        for index in range(discovery.MAX_ENTRIES + 10): (self.root / str(index)).write_bytes(b"")
        result = self.browse()
        self.assertEqual(result["discovery"]["entries_examined"], discovery.MAX_ENTRIES)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["folders"], [])

    def test_entry_cap_does_not_pull_an_extra_directory_entry(self):
        pulled = []
        class Entries:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __iter__(self): return self
            def __next__(self):
                pulled.append(1)
                entry = mock.Mock(); entry.name = ".ignored"
                return entry
        with mock.patch.object(discovery.os, "scandir", return_value=Entries()):
            self.browse()
        self.assertEqual(len(pulled), discovery.MAX_ENTRIES)

    def test_import_does_not_import_model_or_network_libraries(self):
        code = """import sys
from looking_glass import app_discovery
assert not any(name in sys.modules for name in ('torch','transformers','tokenizers','safetensors','requests','urllib.request'))
print('stdlib-only')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1],
                                text=True, capture_output=True, timeout=10, check=True)
        self.assertEqual(result.stdout.strip(), "stdlib-only")


if __name__ == "__main__": unittest.main()
