"""Offline mechanics example using one previously published FLAN-T5 pair."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="previously pinned FLAN-T5-small assets")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path")
    options = parser.parse_args()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
            or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise ValueError("explicit CPU and offline settings are required")

    import torch
    import transformers
    from transformers import T5ForConditionalGeneration
    from looking_glass.flan_t5_feasibility import CASE_HASH, CASES, MANIFEST_HASH, REVISION, digest_file, verify_assets
    from looking_glass.flan_t5_intervention_workflow import compare_t5_donor_patch
    from looking_glass.flan_t5_trace import T5TracePoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized() or digest_file(CASES) != CASE_HASH:
        raise ValueError("unexpected CUDA initialization or public case bytes")
    assets = verify_assets(options.assets)
    pair = json.loads(CASES.read_bytes())["pairs"][0]
    receiver, donor = (entry["input_ids"] for entry in pair["sources"])
    target = pair["targets"][0]["input_ids"]
    if (receiver != [30355, 15, 1566, 12, 2968, 10, 1712, 1]
            or donor != [30355, 15, 1566, 12, 2968, 10, 1782, 1] or target != [16699, 15, 1]):
        raise ValueError("public feasibility inputs differ")
    model = T5ForConditionalGeneration.from_pretrained(
        str(options.assets), local_files_only=True, use_safetensors=True, attn_implementation="eager").eval()
    source_ids, donor_ids = torch.tensor([receiver]), torch.tensor([donor])
    decoder_ids = torch.tensor([[model.config.decoder_start_token_id, *target[:-1]]])
    common = dict(receiver_input_ids=source_ids, donor_input_ids=donor_ids, attention_mask=torch.ones_like(source_ids),
                  receiver_decoder_input_ids=decoder_ids, donor_decoder_input_ids=decoder_ids.clone(),
                  decoder_attention_mask=torch.ones_like(decoder_ids), readout_position=1, readout_token_id=15)
    # Each call takes six separate uncached forwards; token coordinates are local to their stack.
    encoder = compare_t5_donor_patch(model, **common, run_id="public-encoder",
        target=T5TracePoint("encoder", 0, "block", 0, 6), control=T5TracePoint("encoder", 0, "block", 0, 5))
    decoder = compare_t5_donor_patch(model, **common, run_id="public-decoder",
        target=T5TracePoint("decoder", 1, "block", 0, 1), control=T5TracePoint("decoder", 1, "block", 0, 0))
    if torch.cuda.is_initialized() or verify_assets(options.assets) != assets:
        raise ValueError("CUDA initialized or pinned asset bytes changed")
    report = {"schema": "flan-t5-donor-patch-cpu-example-v1", "checkpoint_revision": REVISION,
              "asset_manifest_sha256": MANIFEST_HASH, "public_case_sha256": CASE_HASH,
              "case_scope": "previously exposed public feasibility pair", "device": "cpu", "forwards": 12,
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "encoder": encoder.report, "decoder": decoder.report,
              "witness_note": "The in-memory result contains full before/after witnesses; this JSON stores metadata and readout summaries only."}
    raw = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with options.output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps({"status": "complete", "forwards": 12, "report_sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
