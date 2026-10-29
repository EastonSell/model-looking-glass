"""Verify the public case manifest against local pinned tokenizer assets.

This command loads a tokenizer only. It never loads or forwards a model.
Run from the repository root as `python -m scripts.verify_bert_agreement_tokenizer`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

from looking_glass import bert_agreement_cases as cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True,
                        help="directory holding the four pinned local tokenizer/config files")
    parser.add_argument("--manifest", type=Path,
                        default=Path("experiments/bert_agreement_public_cases_v1.json"))
    options = parser.parse_args()
    data = options.manifest.read_bytes()
    check = cases.verify_manifest_bytes(data)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        digest = hashlib.sha256((options.tokenizer_dir / name).read_bytes()).hexdigest()
        if digest != cases.ASSET_SHA256[name]:
            raise ValueError(f"local pinned asset does not match: {name}")
    # No network access is requested, and no checkpoint/model object is loaded.
    import transformers
    import tokenizers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        options.tokenizer_dir, local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if not tokenizer.is_fast:
        raise ValueError("the pinned fast tokenizer is required")
    manifest = json.loads(data)
    for row in manifest["rows"]:
        encoding = tokenizer(row["text"], add_special_tokens=True, return_attention_mask=True)
        if encoding["input_ids"] != row["input_ids"] or encoding["attention_mask"] != row["attention_mask"]:
            raise ValueError(f"tokenization differs for public row {row['row_id']}")
    for word, expected in cases.OUTPUT_TOKENS.items():
        if tokenizer.encode(word, add_special_tokens=False) != [expected]:
            raise ValueError("output tokenization differs from the pinned manifest")
    print(json.dumps({
        "schema": "distilbert-agreement-tokenizer-check-v1",
        "manifest_sha256": check.manifest_sha256,
        "verified_rows": len(manifest["rows"]),
        "verified_tokenizer_assets": 4,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_loaded": False,
        "checkpoint_weights_verified": False,
        "forward_executed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
