"""Offline T5 pairing smoke on previously exposed public feasibility inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
            or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise ValueError("explicit CPU and offline settings are required")

    import torch
    import transformers
    from transformers import T5ForConditionalGeneration
    from looking_glass.flan_t5_feasibility import CASES, CASE_HASH, MANIFEST_HASH, REVISION, digest_file, verify_assets
    from looking_glass.flan_t5_trace import T5TracePoint
    from looking_glass.t5_forward import trace_t5_comparable_forward
    from looking_glass.transformer_localization import compare_transformer_traces

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized() or digest_file(CASES) != CASE_HASH:
        raise ValueError("unexpected CUDA initialization or public case bytes")
    assets = verify_assets(args.assets)
    pair = json.loads(CASES.read_bytes())["pairs"][0]
    source_a, source_b = (item["input_ids"] for item in pair["sources"])
    target = pair["targets"][0]["input_ids"]
    if (source_a != [30355, 15, 1566, 12, 2968, 10, 1712, 1]
            or source_b != [30355, 15, 1566, 12, 2968, 10, 1782, 1]
            or target != [16699, 15, 1]):
        raise ValueError("the previously exposed public case differs")
    model, load_info = T5ForConditionalGeneration.from_pretrained(
        str(args.assets), local_files_only=True, use_safetensors=True,
        attn_implementation="eager", output_loading_info=True)
    model.eval()
    if any(load_info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("checkpoint did not load exactly")
    if (model.config.decoder_start_token_id != 0
            or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("checkpoint must use the expected start token and CPU float32 parameters")
    decoder = torch.tensor([[0, *target[:-1]]], dtype=torch.long)
    points = (T5TracePoint("encoder", 0, "block", 0, 6),
              T5TracePoint("encoder", 6, "block", 0, 6),
              T5TracePoint("decoder", 0, "block", 0, 1),
              T5TracePoint("decoder", 6, "block", 0, 1))
    def run(name, source):
        ids = torch.tensor([source], dtype=torch.long)
        return trace_t5_comparable_forward(model, input_ids=ids, attention_mask=torch.ones_like(ids),
            decoder_input_ids=decoder, decoder_attention_mask=torch.ones_like(decoder),
            points=points, run_id=name)
    first, repeat, changed = run("first", source_a), run("repeat", source_a), run("changed", source_b)
    def compare(left, right, declared):
        return compare_transformer_traces(left.events, right.events,
            reference_complete=True, candidate_complete=True, pairing_verified=True,
            execution_order_verified=True, t5_contexts=(left.t5_context, right.t5_context),
            expected_t5_encoder_differences=declared, expected_t5_decoder_differences=())
    same = compare(first, repeat, ())
    paired = compare(first, changed, ((0, 6),))
    unverified = compare(first, changed, ())
    if (same["status"] != "no_difference_observed" or paired["status"] != "difference_observed"
            or unverified["status"] != "inconclusive"
            or not torch.equal(first.output.logits, repeat.output.logits)):
        raise ValueError("unexpected paired-comparison smoke result")
    if torch.cuda.is_initialized() or verify_assets(args.assets) != assets:
        raise ValueError("CUDA initialized or pinned asset bytes changed")
    def digest_tensor(value):
        return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
    repo = Path(__file__).resolve().parent.parent
    paths = ("looking_glass/t5_forward.py", "looking_glass/transformer_localization.py",
             "looking_glass/flan_t5_trace.py", "looking_glass/flan_t5_feasibility.py",
             "looking_glass/encoder_forward.py", "looking_glass/localization.py",
             "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
             "looking_glass/model_ownership.py", "looking_glass/trace_schema.py",
             "tools/flan_t5_paired_compare_cpu_smoke.py")
    report = {"schema": "flan-t5-paired-compare-cpu-smoke-v1", "status": "complete",
        "checkpoint_revision": REVISION, "asset_manifest_sha256": MANIFEST_HASH,
        "public_case_sha256": CASE_HASH, "case_scope": "previously exposed public feasibility pair",
        "forward_count": 3, "device": "cpu", "dtype": "torch.float32", "use_cache": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "repeat_logits_equal": True, "same_input_comparison": same, "changed_source_comparison": paired,
        "wrong_declaration_comparison": unverified,
        "event_value_sha256": {name: [digest_tensor(item.value) for item in trace.events]
                               for name, trace in (("first", first), ("repeat", repeat), ("changed", changed))},
        "source_sha256": {name: digest_file(repo / name) for name in paths},
        "scope": "Single-forward pairing check, not execution attestation, semantic alignment or generation."}
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as output_file:
        output_file.write(raw)
    print(json.dumps({"status": "complete", "report_sha256": hashlib.sha256(raw).hexdigest(), "forwards": 3}))


if __name__ == "__main__":
    main()
