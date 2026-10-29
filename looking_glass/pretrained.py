"""Opt-in access to the pinned DistilGPT2 checkpoint."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "distilbert/distilgpt2"
REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
ASSETS = {
    "config.json": (762, "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a"),
    "generation_config.json": (124, "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805"),
    "tokenizer.json": (1355256, "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"),
    "tokenizer_config.json": (26, "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f"),
    "vocab.json": (1042301, "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
    "merges.txt": (456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "model.safetensors": (352824413, "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"),
}


class CheckpointUnavailableError(RuntimeError):
    """The pinned files could not be obtained under the requested policy."""


class CheckpointIntegrityError(RuntimeError):
    """A staged file does not match the pinned size and SHA-256."""


class IncompleteCheckpointError(RuntimeError):
    """Transformers reported missing, mismatched or unexpected weights."""


@dataclass
class LoadedPretrained:
    """A CPU model, its tokenizer and verified source-asset metadata."""

    model: object
    tokenizer: object
    metadata: dict

    def encode(self, text, *, max_tokens=128):
        """Encode one short prompt without silently truncating it."""
        if not isinstance(text, str) or not text or len(text) > 8192:
            raise ValueError("text must be a nonempty string of at most 8192 characters")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
            raise ValueError("max_tokens must be an integer from 1 through 128")
        tokenized = self.tokenizer(text, truncation=True, max_length=max_tokens + 1,
                                   add_special_tokens=False, return_attention_mask=True,
                                   padding=False)
        ids = tokenized["input_ids"]
        if not ids or len(ids) > max_tokens:
            raise ValueError("input must produce between 1 and max_tokens tokens")
        import torch
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device="cpu"),
            "attention_mask": torch.tensor([tokenized["attention_mask"]],
                                           dtype=torch.long, device="cpu"),
        }


def _copy_verified(source, destination, name, expected_size, expected_digest):
    digest = hashlib.sha256()
    total = 0
    with open(source, "rb") as src, open(destination, "xb") as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > expected_size:
                raise CheckpointIntegrityError(f"Checkpoint integrity failure for {name}: size mismatch")
            digest.update(block)
            dst.write(block)
    if total != expected_size or digest.hexdigest() != expected_digest:
        raise CheckpointIntegrityError(f"Checkpoint integrity failure for {name}: size or SHA-256 mismatch")


def load_distilgpt2(*, cache_dir, allow_download=False, stage_dir=None):
    """Load the pinned checkpoint, staging verified copies in ``stage_dir`` when supplied."""
    try:
        cache_dir = os.fspath(cache_dir)
    except TypeError as exc:
        raise ValueError("cache_dir must be an explicit nonempty path") from exc
    if not isinstance(cache_dir, str) or not cache_dir.strip():
        raise ValueError("cache_dir must be an explicit nonempty path")
    if type(allow_download) is not bool:
        raise ValueError("allow_download must be a boolean")
    if stage_dir is None:
        stage_dir = cache_dir
    else:
        try:
            stage_dir = os.fspath(stage_dir)
        except TypeError as exc:
            raise ValueError("stage_dir must be an existing directory") from exc
        if not isinstance(stage_dir, str) or not stage_dir.strip() or not os.path.isdir(stage_dir):
            raise ValueError("stage_dir must be an existing directory")
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "The pretrained loader requires transformers; install transformers 4.57.3."
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError("The pretrained loader requires torch; install PyTorch 2.6 or later.") from exc
    try:
        import safetensors  # noqa: F401
    except ImportError as exc:
        raise ImportError("The pretrained loader requires safetensors.") from exc
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:
        raise ImportError("The pretrained loader requires tokenizers.") from exc
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("The pretrained loader requires huggingface_hub.") from exc

    if not os.path.isdir(cache_dir) and not allow_download:
        raise CheckpointUnavailableError(
            "The supplied cache directory is unavailable in offline mode."
        )
    if allow_download:
        os.makedirs(cache_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="looking-glass-stage-", dir=stage_dir) as stage:
        for name, (size, digest) in ASSETS.items():
            try:
                source = hf_hub_download(
                    repo_id=MODEL_ID, filename=name, revision=REVISION,
                    cache_dir=cache_dir, local_files_only=not allow_download, token=False,
                )
            except Exception as exc:
                mode = "download-enabled" if allow_download else "offline"
                raise CheckpointUnavailableError(
                    f"Pinned checkpoint file {name} is unavailable in {mode} mode. "
                    "Check the supplied cache or opt in to a download."
                ) from exc
            _copy_verified(source, Path(stage) / name, name, size, digest)
        with torch.device("cpu"):
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                stage, local_files_only=True, trust_remote_code=False, use_fast=True,
            )
            model, info = transformers.AutoModelForCausalLM.from_pretrained(
                stage, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, attn_implementation="eager", dtype=torch.float32,
                output_loading_info=True,
            )
        errors = {key: info[key] for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
        ) if info.get(key)}
        if errors:
            raise IncompleteCheckpointError(f"Incomplete checkpoint weights: {errors}")
        model.to(device="cpu", dtype=torch.float32)
        model.eval()
        model.config.use_cache = False
        metadata = {
            "model_id": MODEL_ID,
            "model_revision": REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": REVISION,
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "weights_sha256": ASSETS["model.safetensors"][1],
            "asset_sha256": {name: digest for name, (_, digest) in ASSETS.items()},
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "use_cache": False,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        # A completed loader frame can remain reachable through imported
        # library traceback cycles when cyclic GC is deliberately disabled.
        # Transfer the successful result without leaving an extra model alias
        # in that frame. Failures before construction still retain their
        # original frame locals for the caller's failure custody.
        result = LoadedPretrained(model=model, tokenizer=tokenizer, metadata=metadata)
        model = None
        try:
            return result
        finally:
            result = None
