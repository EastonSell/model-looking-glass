"""Demonstrate a bounded ViT intervention with random weights and synthetic pixels."""

import json

import torch
from transformers import ViTConfig, ViTForImageClassification

from looking_glass.vit_intervention_workflow import compare_vit_donor_patch
from looking_glass.vit_trace import ViTTracePoint, patch_token_index


def main():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        config = ViTConfig(image_size=[32, 48], patch_size=[8, 16], num_channels=3,
                           hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=24, num_labels=7, hidden_dropout_prob=0,
                           attention_probs_dropout_prob=0)
        config._attn_implementation = "eager"
        model = ViTForImageClassification(config).eval()
    receiver = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    donor = receiver.flip(-1).contiguous()
    patch = patch_token_index(1, 2, grid_rows=4, grid_columns=3)
    result = compare_vit_donor_patch(model, receiver_pixels=receiver, donor_pixels=donor,
        target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, patch),
        readout_class=3, run_id="synthetic-vit-example")
    print(json.dumps({"scope": "random weights and synthetic pixels; mechanics only",
                      "sites": result.report["sites"], "doses": result.report["doses"],
                      "controls": result.report["controls"], "readout": result.report["readout"],
                      "forwards": result.report["forwards"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
