"""Offline, forward-only adapter smoke using an already exposed public T5 case."""

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
            or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise ValueError("explicit CPU and offline settings are required")

    import torch
    import transformers
    from transformers import T5ForConditionalGeneration
    from looking_glass.flan_t5_feasibility import CASE_HASH, CASES, MANIFEST_HASH, REVISION, digest_file, verify_assets
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized() or digest_file(CASES) != CASE_HASH:
        raise ValueError("unexpected CUDA initialization or public case bytes")
    assets = verify_assets(args.assets)
    cases = json.loads(CASES.read_bytes())
    source = cases["pairs"][0]["sources"][0]["input_ids"]
    target = cases["pairs"][0]["targets"][0]["input_ids"]
    if source != [30355, 15, 1566, 12, 2968, 10, 1712, 1] or target != [16699, 15, 1]:
        raise ValueError("the previously exposed public case differs")
    model = T5ForConditionalGeneration.from_pretrained(
        str(args.assets), local_files_only=True, use_safetensors=True, attn_implementation="eager").eval()
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("checkpoint parameters must be CPU float32")
    source_ids = torch.tensor([source], dtype=torch.long)
    decoder_ids = torch.tensor([[model.config.decoder_start_token_id, *target[:-1]]], dtype=torch.long)
    source_mask, decoder_mask = torch.ones_like(source_ids), torch.ones_like(decoder_ids)
    paths = ("encoder.block.0.layer.0", "encoder.block.0.layer.1",
             "decoder.block.0.layer.0", "decoder.block.0.layer.2")
    observed, handles = {}, []
    for path in paths:
        def save(_module, _inputs, value, *, path=path):
            observed[path] = (value[0] if type(value) is tuple else value).detach().clone()
        handles.append(model.get_submodule(path).register_forward_hook(save))
    try:
        with torch.inference_mode():
            baseline = model(input_ids=source_ids, attention_mask=source_mask,
                             decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask,
                             use_cache=False, output_hidden_states=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    points = [T5TracePoint("encoder", 0, "self_attention", 0, 6),
              T5TracePoint("encoder", 0, "feed_forward", 0, 6),
              T5TracePoint("encoder", 0, "block", 0, 6),
              T5TracePoint("encoder", 6, "block", 0, 1),
              T5TracePoint("decoder", 0, "self_attention", 0, 1),
              T5TracePoint("decoder", 0, "feed_forward", 0, 1),
              T5TracePoint("decoder", 0, "block", 0, 1),
              T5TracePoint("decoder", 6, "block", 0, 2)]
    output, records = trace_t5_forward(model, input_ids=source_ids, attention_mask=source_mask,
        decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask, run_id="checkpoint-smoke", points=points)
    expected_paths = (paths[0], paths[1], "encoder.block.0", "encoder.block.6",
                      paths[2], paths[3], "decoder.block.0", "decoder.block.6")
    expected = (observed[paths[0]][0, 6], observed[paths[1]][0, 6],
                baseline.encoder_hidden_states[1][0, 6], baseline.encoder_hidden_states[7][0, 1],
                observed[paths[2]][0, 1], observed[paths[3]][0, 1],
                baseline.decoder_hidden_states[1][0, 1], baseline.decoder_hidden_states[7][0, 2])
    if not torch.equal(output.logits, baseline.logits) or output.past_key_values is not None:
        raise ValueError("trace changed the uncached model output")
    if (tuple(record.event.module_path for record in records) != expected_paths
            or any(not torch.equal(record.value, value) for record, value in zip(records, expected, strict=True))):
        raise ValueError("trace differs from untraced hidden-state observations")
    if torch.cuda.is_initialized() or verify_assets(args.assets) != assets:
        raise ValueError("CUDA initialized or pinned asset bytes changed")
    report = {"schema": "flan-t5-trace-cpu-smoke-v1", "status": "complete",
        "checkpoint_revision": REVISION, "asset_manifest_sha256": MANIFEST_HASH,
        "public_case_sha256": CASE_HASH, "case_scope": "previously exposed public feasibility case",
        "forwards": 2, "device": "cpu", "dtype": "torch.float32", "use_cache": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "logits_shape": list(output.logits.shape), "logits_equal_to_untraced": True,
        "trace_values_equal_to_untraced_boundaries": True,
        "events": [{"metadata": record.event.to_dict(),
                    "value_sha256": hashlib.sha256(record.value.contiguous().numpy().tobytes()).hexdigest()}
                   for record in records]}
    repo = Path(__file__).resolve().parent.parent
    paths = ("looking_glass/flan_t5_trace.py", "looking_glass/flan_t5_feasibility.py",
             "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
             "looking_glass/model_ownership.py", "looking_glass/trace_schema.py",
             "tools/flan_t5_trace_cpu_smoke.py")
    report["source_sha256"] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as output_file:
        output_file.write(raw)
    print(json.dumps({"status": "complete", "report_sha256": hashlib.sha256(raw).hexdigest(), "forwards": 2}))


if __name__ == "__main__":
    main()
