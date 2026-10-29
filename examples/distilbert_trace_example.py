"""Trace an already-public masked sentence using cached DistilBERT weights."""

import argparse
import json
from pathlib import Path

import torch
import transformers

from looking_glass.distilbert_trace import trace_distilbert_masked_lm


MODEL = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
TEXT = "The painter beside the artist [MASK] ready."
IDS = [101, 1996, 5276, 3875, 1996, 3063, 103, 3201, 1012, 102]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON summary file")
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
    if inputs["input_ids"][0].tolist() != IDS or tokenizer.mask_token_id != 103:
        raise RuntimeError("tokenizer does not reproduce the public input")
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    with torch.no_grad():
        clean = model(input_ids=ids, attention_mask=mask, output_attentions=False,
                      output_hidden_states=False, return_dict=True).logits
    trace = trace_distilbert_masked_lm(model, ids, mask, run_id="distilbert-example",
                                      layers=(0, 2), tokens=(6,), max_events=6,
                                      max_trace_bytes=32 * 1024)
    if not torch.equal(clean, trace.output.logits):
        raise RuntimeError("traced logits differ from the clean forward")
    top_id = int(clean[0, 6].argmax().item())
    report = {
        "schema": "distilbert-trace-example-v1", "model": MODEL, "revision": REVISION,
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "device": "cpu", "dtype": "float32", "text": TEXT, "input_ids": IDS,
        "attention_mask": [list(row) for row in trace.attention_mask], "mask_index": 6,
        "trace_tensor_bytes": trace.tensor_bytes, "clean_and_traced_logits_equal": True,
        "mask_top_token": {"token_id": top_id, "text": tokenizer.decode([top_id])},
        "events": [{"metadata": item.event.to_dict(),
                    "original_saved_shape": list(item.original_saved_shape),
                    "comparison_transform": item.comparison_transform,
                    "first_four_values": item.value[0, 0, :4].tolist()} for item in trace.events],
        "interpretation": "These are selected encoder activations and a masked-position argmax, not generated text or an explanation of the prediction.",
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Saved {len(trace.events)} events and {trace.tensor_bytes} trace bytes to {args.output}")


if __name__ == "__main__":
    main()
