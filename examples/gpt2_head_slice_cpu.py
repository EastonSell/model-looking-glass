"""Observe and zero selected GPT-2 head slices on a fixed public input."""

import argparse
import json
import math
from pathlib import Path

import torch

from looking_glass.copy_head_factorial import f32_bits_to_f64
from looking_glass.gpt2_head_slice import GPT2HeadSlicePoint, run_gpt2_head_slice
from looking_glass.pretrained import load_distilgpt2


PROMPT = "Hello world"
EXPECTED_IDS = [15496, 995]


def _same_bits(left, right):
    return left.shape == right.shape and torch.equal(left.contiguous().view(torch.int32),
                                                      right.contiguous().view(torch.int32))


def _values(value):
    return [f32_bits_to_f64(bits & 0xffffffff)
            for bits in value.contiguous().view(torch.int32).reshape(-1).tolist()]


def _distance(left, right):
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(_values(left), _values(right), strict=True)))


def _max_difference(left, right):
    return max(abs(a - b) for a, b in zip(_values(left), _values(right), strict=True))


def run_demo(model, inputs):
    """Run clean, observed, self-patch, two head ablations, and clean repeat."""
    ids = inputs["input_ids"]
    if tuple(ids.shape) != (1, 2) or not torch.equal(inputs["attention_mask"], torch.ones_like(ids)):
        raise ValueError("this example requires one unpadded two-token input")
    if model.config.n_head < 2:
        raise ValueError("the example needs two attention heads")
    target = GPT2HeadSlicePoint(layer=0, token=1, head=0)
    comparison = GPT2HeadSlicePoint(layer=0, token=1, head=1)
    def run(point, name, replacement=None):
        return run_gpt2_head_slice(model, ids, point=point, run_id=name, replacement=replacement,
                                   max_tensor_bytes=16 * 1024, max_capture_bytes=48 * 1024,
                                   max_source_bytes=4096, max_logits_bytes=1024 * 1024)
    with torch.inference_mode():
        clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        observed = run(target, "head-observe")
        self_patch = run(target, "head-self", observed.head_before.clone())
        target_zero = run(target, "head-zero", torch.zeros_like(observed.head_before))
        other_observed = run(comparison, "other-head-observe")
        other_zero = run(comparison, "other-head-zero", torch.zeros_like(other_observed.head_before))
        repeated = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    target_dose = _distance(target_zero.head_before, target_zero.head_after)
    other_dose = _distance(other_zero.head_before, other_zero.head_after)
    checks = {
        "observed_logits_equal_clean": _same_bits(observed.logits, clean),
        "other_observed_logits_equal_clean": _same_bits(other_observed.logits, clean),
        "self_patch_input_unchanged": _same_bits(self_patch.before, self_patch.after),
        "self_patch_logits_equal_clean": _same_bits(self_patch.logits, clean),
        "target_patch_input_matches_observation": _same_bits(target_zero.before, observed.before),
        "comparison_input_matches_observation": _same_bits(other_zero.before, other_observed.before),
        "both_heads_share_observed_input": _same_bits(other_observed.before, observed.before),
        "target_head_zero_delivered": _same_bits(target_zero.head_after, torch.zeros_like(target_zero.head_after)),
        "comparison_head_zero_delivered": _same_bits(other_zero.head_after, torch.zeros_like(other_zero.head_after)),
        "target_dose_nonzero": target_dose > 0,
        "comparison_dose_nonzero": other_dose > 0,
        "earlier_token_logits_unchanged": _same_bits(target_zero.logits[:, 0], clean[:, 0]),
        "post_session_logits_equal_clean": _same_bits(repeated, clean),
    }
    if not all(checks.values()):
        raise RuntimeError(f"example check failed: {checks}")
    return {
        "schema": "gpt2-head-slice-cpu-example-v1",
        "input_ids": ids[0].tolist(),
        "target": {"metadata": target_zero.metadata.to_dict(), "captured_tensor_bytes": target_zero.tensor_bytes,
                   "head_before_preview": _values(target_zero.head_before[:4]),
                   "head_after_preview": _values(target_zero.head_after[:4]),
                   "realized_input_l2": target_dose,
                   "max_final_token_logit_change": _max_difference(target_zero.logits[:, -1], clean[:, -1])},
        "comparison": {"metadata": other_zero.metadata.to_dict(), "captured_tensor_bytes": other_zero.tensor_bytes,
                       "realized_input_l2": other_dose,
                       "max_final_token_logit_change": _max_difference(other_zero.logits[:, -1], clean[:, -1])},
        "checks": checks,
        "interpretation": "These artificial head ablations show an API measurement. The comparison is not dose-matched; neither the selected head nor its slice establishes a unique circuit or attention-weight provenance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="offline Hugging Face hub cache")
    parser.add_argument("--output", type=Path, required=True, help="new summary JSON file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    bundle = load_distilgpt2(cache_dir=args.cache_dir)
    inputs = bundle.encode(PROMPT, max_tokens=2)
    if inputs["input_ids"][0].tolist() != EXPECTED_IDS:
        raise RuntimeError("the pinned tokenizer produced unexpected input IDs")
    result = {"prompt": PROMPT, "model": bundle.metadata, **run_demo(bundle.model, inputs)}
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Saved head-slice example to {args.output}")


if __name__ == "__main__":
    main()
