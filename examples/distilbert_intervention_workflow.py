"""Run a known intervention on an already-public cached DistilBERT input."""

import argparse
import json
from pathlib import Path

import torch
import transformers

from looking_glass.distilbert_intervention_workflow import run_known_distilbert_intervention


MODEL = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
TEXT = "The painter beside the artist [MASK] ready."
IDS = [101, 1996, 5276, 3875, 1996, 3063, 103, 3201, 1012, 102]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON report")
    parser.add_argument("--witnesses", type=Path, help="optional new PyTorch tensor file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, cache_dir=args.cache_dir, local_files_only=True,
        trust_remote_code=False, use_fast=True)
    model, loading = transformers.AutoModelForMaskedLM.from_pretrained(
        MODEL, revision=REVISION, cache_dir=args.cache_dir, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, attn_implementation="eager",
        dtype=torch.float32, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("checkpoint loading reported missing, unexpected or mismatched weights")
    model.eval()
    inputs = tokenizer(TEXT, return_tensors="pt")
    readout_id = tokenizer.convert_tokens_to_ids("is")
    if inputs["input_ids"][0].tolist() != IDS or tokenizer.mask_token_id != 103 or readout_id != 2003:
        raise RuntimeError("tokenizer does not reproduce the public input and readout")
    result = run_known_distilbert_intervention(
        model, inputs["input_ids"], inputs["attention_mask"], injection_layer=0,
        restoration_layer=1, target_token=2, control_token=5, readout_position=6,
        readout_token_id=readout_id, coordinate=0, amount=5.0, run_id="public-agreement-example")
    report = {"model": MODEL, "requested_revision": REVISION, "text": TEXT,
              "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
              "result": result.report}
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    if args.witnesses is not None:
        with args.witnesses.open("xb") as output:
            torch.save(result.witnesses, output)
    print(f"Saved report; off-target status: {result.report['off_target']['status']}")


if __name__ == "__main__":
    main()
