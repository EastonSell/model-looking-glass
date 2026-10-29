"""Trace and edit a small, fixed DistilGPT2 input on the CPU.

Run this file from an installed checkout; see the accompanying guide for the
cache and output arguments. The edit is deliberately chosen by the user: it is
not a claim about a naturally occurring feature or circuit.
"""

import argparse
import json
from pathlib import Path

import torch

from looking_glass.intervention_plan import ActivationPlan
from looking_glass.pretrained import load_distilgpt2
from looking_glass.trace_schema import GenerationContext
from looking_glass.transformer_trace import GPT2TraceSession


PROMPT = "Hello world"
EXPECTED_IDS = [15496, 995]
SITE = "transformer.h.0.mlp"
TOKEN = 1


def run_demo(model, inputs, decode_token):
    """Run five uncached forwards, returning a JSON-serializable summary.

    The model must be an unmodified CPU GPT2LMHeadModel in evaluation mode.
    Trace and intervention sessions run sequentially: both require exclusive
    ownership of the model. Limits below apply to captured tensors and plan
    sources, not to the model, its intermediate allocations, or logits.
    """
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("use an evaluation-mode CPU model")
    if tuple(inputs["input_ids"].shape) != (1, 2):
        raise ValueError("this example requires one two-token input")

    def forward():
        return model(**inputs, use_cache=False).logits.detach()

    with torch.no_grad():
        clean = forward()

        # Three layer-0 boundaries, for both input positions, in one forward.
        trace = GPT2TraceSession(
            model, run_id="pretrained-example", layers=(0,),
            targets=("attn", "mlp", "block"), samples=(0,), tokens=(0, 1),
            max_events=3, max_trace_bytes=32 * 1024, capture_device="cpu",
            forward_context=GenerationContext(stage="forward", use_cache=False),
        )
        with trace:
            traced = forward()
        if not trace.successful:
            raise RuntimeError("trace did not complete")

        # Capture the current vector and patch that very same vector back in.
        # This is an inference control, not a gradient-preserving operation.
        site = {"module_path": SITE, "invocation": 0,
                "batch_index": 0, "token_index": TOKEN}
        limits = {"max_tensor_bytes": 16 * 1024, "max_capture_bytes": 8 * 1024,
                  "max_source_bytes": 4 * 1024, "max_steps": 3,
                  "max_invocations": 1}
        self_plan = ActivationPlan(model, [
            dict(site, name="before", operation="capture"),
            dict(site, name="self_patch", operation="patch", source_capture="before"),
            dict(site, name="after", operation="capture"),
        ], **limits)
        with self_plan:
            self_logits = forward()

        # Add a known, deliberately artificial delta at the same site and
        # capture both sides of the action in this very forward.
        delta = torch.zeros(model.config.n_embd, dtype=torch.float32)
        delta[0] = 5.0
        edit_plan = ActivationPlan(model, [
            dict(site, name="before", operation="capture"),
            dict(site, name="add", operation="add", value=delta),
            dict(site, name="after", operation="capture"),
        ], **limits)
        with edit_plan:
            edited = forward()
        restored = forward()

    before, after = edit_plan.captures["before"], edit_plan.captures["after"]
    logit_change = (edited - clean).abs()
    final_logit_change = logit_change[0, -1].max()
    controls = {
        "traced_logits_equal_clean": torch.equal(traced, clean),
        "self_patch_logits_equal_clean": torch.equal(self_logits, clean),
        "self_patch_before_equals_after": torch.equal(
            self_plan.captures["before"], self_plan.captures["after"]),
        "edit_input_equals_self_capture": torch.equal(before, self_plan.captures["before"]),
        "edit_after_equals_before_plus_delta": torch.equal(after, before + delta),
        "edit_changed_activation": not torch.equal(before, after),
        "clean_and_edited_logits_finite": bool(torch.isfinite(clean).all() and torch.isfinite(edited).all()),
        "logit_difference_finite": bool(torch.isfinite(logit_change).all()),
        "edited_first_token_logits_equal_clean": torch.equal(edited[:, 0], clean[:, 0]),
        "edited_final_token_logits_changed": bool(torch.isfinite(final_logit_change) and final_logit_change > 0),
        "post_session_logits_equal_clean": torch.equal(restored, clean),
    }
    if not all(controls.values()):
        raise RuntimeError(f"a comparison or intervention check failed: {controls}")

    events = []
    for event in trace.events:
        # A short preview keeps the saved JSON small; event.value itself is the
        # complete bounded [batch, selected token, hidden] tensor in memory.
        events.append({
            "module_path": event.module_path,
            "kind": event.module_kind,
            "source_shape": list(event.shape),
            "saved_shape": list(event.saved_shape),
            "token_positions": list(event.token_positions),
            "tensor_bytes": event.tensor_bytes,
            "last_token_first_four_values": event.value[0, -1, :4].tolist(),
        })

    def top_next(logits):
        token_id = int(logits[0, -1].argmax().item())
        return {"token_id": token_id, "text": decode_token(token_id)}

    return {
        "trace": {"max_events": 3, "max_trace_bytes": 32 * 1024,
                  "captured_tensor_bytes": trace.tensor_bytes, "events": events},
        "intervention": {
            "module_path": SITE, "batch_index": 0, "token_index": TOKEN,
            "operation": "add", "delta_coordinate": 0, "delta_value": 5.0,
            "before_coordinate": float(before[0]), "after_coordinate": float(after[0]),
            "observed_displacement_l2": float(torch.linalg.vector_norm(after - before)),
            "max_absolute_logit_change": float(logit_change.max()),
            "max_absolute_final_token_logit_change": float(final_logit_change),
            "clean_next_token_top": top_next(clean),
            "edited_next_token_top": top_next(edited),
            "plan_limits_bytes": {key: limits[key] for key in (
                "max_tensor_bytes", "max_capture_bytes", "max_source_bytes")},
        },
        "controls": controls,
        "interpretation": (
            "The chosen edit demonstrates the API and its downstream effect on "
            "logits for this input. It does not explain the model's ordinary behavior. "
            "The next-token entries are argmax logits, not generated sentences."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Hugging Face hub cache containing the pinned model")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON file for the explanatory summary")
    parser.add_argument("--allow-download", action="store_true",
                        help="explicitly download missing pinned assets into the cache")
    args = parser.parse_args()

    torch.set_num_threads(1)
    bundle = load_distilgpt2(cache_dir=args.cache_dir, allow_download=args.allow_download)
    inputs = bundle.encode(PROMPT, max_tokens=2)
    ids = inputs["input_ids"][0].tolist()
    if ids != EXPECTED_IDS:
        raise RuntimeError("the pinned tokenizer produced unexpected input IDs")
    summary = run_demo(bundle.model, inputs, lambda token_id: bundle.tokenizer.decode([token_id]))
    summary = {"schema": "gpt2-trace-intervention-example-v1", "prompt": PROMPT,
               "input_ids": ids, "model": bundle.metadata, **summary}
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Saved summary to {args.output}")
    print(f"Captured {summary['trace']['captured_tensor_bytes']} trace bytes; "
          f"maximum absolute logit change: "
          f"{summary['intervention']['max_absolute_logit_change']:.6g}")


if __name__ == "__main__":
    main()
