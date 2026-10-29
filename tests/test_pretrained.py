"""Offline contract checks for the opt-in pretrained model loader."""

import os
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import looking_glass.pretrained as pretrained
from looking_glass.pretrained import load_distilgpt2


class OptionalDependencyTests(unittest.TestCase):
    def test_missing_required_backend_never_attempts_download(self):
        for missing in ("torch", "safetensors", "tokenizers", "huggingface_hub"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as cache:
                requests = []
                hub = types.ModuleType("huggingface_hub")

                def unexpected(**kwargs):
                    requests.append(kwargs)
                    raise OSError("unexpected hub request")

                hub.hf_hub_download = unexpected
                fake_modules = {
                    "transformers": types.ModuleType("transformers"),
                    "torch": types.ModuleType("torch"),
                    "safetensors": types.ModuleType("safetensors"),
                    "tokenizers": types.ModuleType("tokenizers"),
                    "huggingface_hub": hub,
                }
                fake_modules[missing] = None
                with patch.dict(sys.modules, fake_modules):
                    with self.assertRaisesRegex(ImportError, missing):
                        load_distilgpt2(cache_dir=cache, allow_download=True)
                self.assertEqual(requests, [])

    def test_explicit_cache_and_boolean_download_opt_in_are_required(self):
        with self.assertRaises(ValueError):
            load_distilgpt2(cache_dir=None)
        with self.assertRaises(ValueError):
            load_distilgpt2(cache_dir="")
        with self.assertRaises(ValueError):
            load_distilgpt2(cache_dir="unused-cache", allow_download="yes")

    def test_missing_transformers_has_actionable_error_without_downloading(self):
        # -S makes the optional dependency unavailable even in a fully provisioned
        # interpreter. Importing the loader should not try to install anything.
        code = (
            "from looking_glass.pretrained import load_distilgpt2\n"
            "try:\n"
            "    load_distilgpt2(cache_dir='unused-cache')\n"
            "except ImportError as exc:\n"
            "    print(exc)\n"
            "    raise SystemExit(7)\n"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", code], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("transformers", result.stdout.lower())

    def test_empty_cache_offline_fails_without_requesting_download(self):
        # Hub access is the external boundary; assert that cache lookup is offline.
        requests = []
        hub = types.ModuleType("huggingface_hub")

        def missing(**kwargs):
            requests.append(kwargs)
            raise FileNotFoundError("cache entry missing")

        hub.hf_hub_download = missing
        fake_modules = {
            "transformers": types.ModuleType("transformers"),
            "torch": types.ModuleType("torch"),
            "safetensors": types.ModuleType("safetensors"),
            "tokenizers": types.ModuleType("tokenizers"),
            "huggingface_hub": hub,
        }
        with tempfile.TemporaryDirectory() as cache, patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                load_distilgpt2(cache_dir=cache)
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0]["local_files_only"], True)
        self.assertEqual(requests[0]["revision"],
                         "2290a62682d06624634c1f46a6ad5be0f47f38aa")

    def test_explicit_download_opt_in_permits_hub_fetches(self):
        requests = []
        hub = types.ModuleType("huggingface_hub")

        def missing(**kwargs):
            requests.append(kwargs)
            raise OSError("cannot access hub")

        hub.hf_hub_download = missing
        fake_modules = {
            "transformers": types.ModuleType("transformers"),
            "torch": types.ModuleType("torch"),
            "safetensors": types.ModuleType("safetensors"),
            "tokenizers": types.ModuleType("tokenizers"),
            "huggingface_hub": hub,
        }
        with tempfile.TemporaryDirectory() as cache, patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "download-enabled"):
                load_distilgpt2(cache_dir=cache, allow_download=True)
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0]["local_files_only"], False)

    def test_corrupted_asset_is_rejected_before_model_loading(self):
        hub = types.ModuleType("huggingface_hub")
        fake_modules = {
            "transformers": types.ModuleType("transformers"),
            "torch": types.ModuleType("torch"),
            "safetensors": types.ModuleType("safetensors"),
            "tokenizers": types.ModuleType("tokenizers"),
            "huggingface_hub": hub,
        }
        with tempfile.TemporaryDirectory() as cache:
            source = Path(cache) / "source"
            source.write_bytes(b"evil")
            hub.hf_hub_download = lambda **kwargs: str(source)
            # SHA-256 of the intended four-byte payload b"good".
            expected = {"config.json": (4, "770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c")}
            with patch.dict(sys.modules, fake_modules), patch.object(
                pretrained, "ASSETS", expected, create=True
            ):
                with self.assertRaisesRegex(RuntimeError, "integrity"):
                    load_distilgpt2(cache_dir=cache)

    def test_encoder_rejects_invalid_input_before_tokenization(self):
        bundle_class = getattr(pretrained, "LoadedPretrained", None)
        self.assertIsNotNone(bundle_class, "bounded encoder is not implemented")
        bundle = bundle_class(model=None, tokenizer=None, metadata={})
        for text, limit in [("", 128), ("x" * 8193, 128), (None, 128),
                            ("hello", 0), ("hello", 129), ("hello", True)]:
            with self.subTest(text_type=type(text), text_length=len(text) if text else 0,
                              limit=limit):
                with self.assertRaises(ValueError):
                    bundle.encode(text, max_tokens=limit)

    def test_encoder_rejects_overlong_tokens_and_returns_cpu_tensors(self):
        class Tokenizer:
            def __call__(self, text, *, truncation, max_length, add_special_tokens,
                         return_attention_mask, padding):
                if not truncation or add_special_tokens or not return_attention_mask or padding:
                    raise AssertionError("unexpected tokenization policy")
                ids = [ord(c) for c in text][:max_length]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        bundle = pretrained.LoadedPretrained(model=None, tokenizer=Tokenizer(), metadata={})
        encoded = bundle.encode("ab", max_tokens=2)
        self.assertEqual(encoded["input_ids"].tolist(), [[97, 98]])
        self.assertEqual(encoded["attention_mask"].tolist(), [[1, 1]])
        self.assertEqual(encoded["input_ids"].device.type, "cpu")
        with self.assertRaisesRegex(ValueError, "token"):
            bundle.encode("abc", max_tokens=2)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers is optional")
class LocalTransformersFixtureTests(unittest.TestCase):
    """Test real Transformers parsing/loading without a network or pretrained weights."""

    def setUp(self):
        import torch
        from tokenizers import ByteLevelBPETokenizer
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        byte_bpe = ByteLevelBPETokenizer()
        byte_bpe.train_from_iterator(["a little example", "another short example"],
                                     vocab_size=257, min_frequency=1,
                                     special_tokens=["<|endoftext|>"])
        byte_bpe.save_model(str(self.source))
        tokenizer = GPT2TokenizerFast(vocab_file=str(self.source / "vocab.json"),
                                      merges_file=str(self.source / "merges.txt"))
        tokenizer.save_pretrained(self.source)
        config = GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=8,
                            n_layer=1, n_head=2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.original = GPT2LMHeadModel(config)
        self.original.save_pretrained(self.source, safe_serialization=True)
        # If included in the stage, Transformers would try to interpret it.
        (self.source / "adapter_config.json").write_text("not valid JSON")

    def manifest(self):
        names = ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")
        assets = {}
        for name in names:
            data = (self.source / name).read_bytes()
            assets[name] = (len(data), hashlib.sha256(data).hexdigest())
        return assets

    def load_fixture(self):
        import huggingface_hub

        def cached_file(*, repo_id, filename, revision, cache_dir, local_files_only, token):
            self.assertEqual(repo_id, "distilbert/distilgpt2")
            self.assertEqual(revision, "2290a62682d06624634c1f46a6ad5be0f47f38aa")
            self.assertTrue(local_files_only)
            self.assertIs(token, False)
            return str(self.source / filename)

        with patch.object(huggingface_hub, "hf_hub_download", cached_file), patch.object(
            pretrained, "ASSETS", self.manifest()
        ):
            return load_distilgpt2(cache_dir=self.temp.name)

    def test_verified_assets_load_real_model_and_exclude_unverified_neighbors(self):
        import torch

        bundle = self.load_fixture()
        self.assertFalse(bundle.model.training)
        self.assertFalse(bundle.model.config.use_cache)
        self.assertTrue(torch.equal(bundle.model.transformer.wte.weight,
                                    self.original.transformer.wte.weight))
        self.assertTrue(all(p.device.type == "cpu" for p in bundle.model.parameters()))
        ids = bundle.encode("a little example")
        with torch.inference_mode():
            output = bundle.model(**ids).logits
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], ids["input_ids"].shape[1])
        self.assertFalse(list(Path(self.temp.name).glob("looking-glass-stage-*")))

    def test_checkpoint_missing_a_learned_weight_fails(self):
        from safetensors.torch import load_file, save_file

        path = self.source / "model.safetensors"
        weights = load_file(path)
        del weights["transformer.h.0.mlp.c_fc.weight"]
        save_file(weights, path, metadata={"format": "pt"})
        with self.assertRaisesRegex(pretrained.IncompleteCheckpointError, "missing_keys"):
            self.load_fixture()


@unittest.skipUnless(os.environ.get("LOOKING_GLASS_TEST_CACHE"), "explicit pretrained cache not configured")
class PinnedCheckpointSmokeTests(unittest.TestCase):
    def test_verified_pinned_checkpoint_runs_on_cpu_after_staging_cleanup(self):
        import torch

        cache = os.environ["LOOKING_GLASS_TEST_CACHE"]
        bundle = load_distilgpt2(cache_dir=cache)
        self.assertFalse(bundle.model.training)
        self.assertEqual(bundle.model.config._attn_implementation, "eager")
        self.assertFalse(bundle.model.config.use_cache)
        self.assertTrue(all(param.device.type == "cpu" for param in bundle.model.parameters()))
        self.assertEqual(bundle.metadata["weights_sha256"],
                         "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b")
        self.assertEqual(bundle.encode(" the")["input_ids"].tolist(), [[262]])
        self.assertEqual(bundle.encode(" a")["input_ids"].tolist(), [[257]])
        encoded = bundle.encode("The sky is blue.")
        with torch.inference_mode():
            logits = bundle.model(**encoded).logits
        self.assertEqual(tuple(logits.shape), (1, encoded["input_ids"].shape[1], 50257))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertFalse(list(Path(cache).glob("looking-glass-stage-*")))


if __name__ == "__main__":
    unittest.main()
