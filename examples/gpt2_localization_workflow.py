"""Run the known-corruption trace workflow on the cached DistilGPT2 model."""

import argparse
import json
from pathlib import Path

import torch

from looking_glass.gpt2_trace_workflow import compare_known_corruption
from looking_glass.pretrained import load_distilgpt2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON output file")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    bundle = load_distilgpt2(cache_dir=args.cache_dir, allow_download=args.allow_download)
    inputs = bundle.encode("Hello world", max_tokens=2)
    if inputs["input_ids"][0].tolist() != [15496, 995]:
        raise RuntimeError("the pinned tokenizer produced unexpected IDs")
    report = compare_known_corruption(
        bundle.model, inputs["input_ids"], coordinate=0, amount=5.0,
        readout_token=13, layers=(0, 1), target_site=(0, "block"),
        comparison_site=(0, "attn"),
    )
    report["model"] = bundle.metadata
    report["prompt"] = "Hello world"
    report["readout_token_text"] = bundle.tokenizer.decode([13])
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Saved report to {args.output}")
    print(f"Trace comparison: {report['localization']['status']}; "
          f"{len(report['candidate_sites'])} changed sampled boundaries")
    print(f"Interventions: {report['interventions']['status']}")


if __name__ == "__main__":
    main()
