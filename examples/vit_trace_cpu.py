"""Offline, bounded ViT tracing example using an image and cached weights."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="pinned local ViT model and processor")
    parser.add_argument("--image", type=Path, required=True, help="local image to preprocess")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path")
    parser.add_argument("--verify-public-case", action="store_true", help="check the repository's exposed v1 cat image")
    args = parser.parse_args()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1") or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1" or sys.byteorder != "little"):
        raise ValueError("explicit CPU, offline and little-endian settings are required")
    if args.output.exists():
        raise FileExistsError("output path already exists")

    import torch
    import transformers
    from PIL import Image, __version__ as pillow_version
    from transformers import AutoImageProcessor, ViTForImageClassification
    from looking_glass.vision_coco_cases import MODEL_FILES, MODEL_REPO, MODEL_REVISION, verify_assets
    from looking_glass.vit_trace import ViTTracePoint, trace_vit_classifier

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("unexpected CUDA initialization")
    verify_assets(args.assets)
    raw_image = args.image.read_bytes()
    image_sha = hashlib.sha256(raw_image).hexdigest()
    if args.verify_public_case and image_sha != "41a423af33123a924fd6b38d5318c0f9db0f83c139d32b718b2b09fc6bace310":
        raise ValueError("image is not the exposed v1 example")
    processor = AutoImageProcessor.from_pretrained(str(args.assets), local_files_only=True, use_fast=False)
    if type(processor).__name__ != "ViTImageProcessor":
        raise ValueError("unexpected processor class")
    with Image.open(io.BytesIO(raw_image)) as image:
        original_size = list(image.size)
        pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    if pixels.device.type != "cpu" or pixels.dtype != torch.float32:
        raise ValueError("processor did not return CPU float32 pixels")
    pixel_sha = hashlib.sha256(pixels.view(torch.int32).contiguous().numpy().tobytes()).hexdigest()
    if args.verify_public_case and (original_size != [640, 427]
            or pixel_sha != "4cadfa22b04ef1fb0ab450559b8ceae8a3311f1901cd7ca6ebcf9053f7b17228"):
        raise ValueError("processed public example differs from the recorded bytes")
    model = ViTForImageClassification.from_pretrained(str(args.assets), local_files_only=True,
        use_safetensors=True, attn_implementation="eager").eval()
    # Independent direct hooks capture whole outputs on an otherwise untraced forward.
    paths = ("vit.encoder.layer.2.attention", "vit.encoder.layer.2.intermediate", "vit.encoder.layer.2")
    reference, handles = {}, []
    for path in paths:
        def capture(_module, _inputs, output, *, path=path):
            reference[path] = output.detach().clone()
        handles.append(model.get_submodule(path).register_forward_hook(capture))
    try:
        with torch.inference_mode():
            baseline = model(pixel_values=pixels, interpolate_pos_encoding=False, output_attentions=False,
                             output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    points = [ViTTracePoint(2, kind, 0, token)
              for kind in ("attention", "intermediate", "block") for token in (0, 1)]
    result = trace_vit_classifier(model, pixels, run_id="vit-cpu-example", points=points)
    def same_bits(a, b):
        return a.shape == b.shape and torch.equal(a.view(torch.int32).contiguous(), b.view(torch.int32).contiguous())
    if (not same_bits(result.output.logits, baseline.logits)
            or any(not same_bits(record.value, reference[record.event.module_path][0, record.event.token_positions[0]])
                   for record in result.records)):
        raise RuntimeError("trace differs from untraced direct-hook reference")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU example")
    verify_assets(args.assets)
    report = {"schema": "vit-trace-cpu-example-v1", "model": MODEL_REPO, "revision": MODEL_REVISION,
              "model_asset_sha256": MODEL_FILES, "image_sha256": image_sha, "original_size_xy": original_size,
              "processed_pixels_sha256": pixel_sha, "exif_transpose": False,
              "device": "cpu", "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "pillow_version": pillow_version,
              "forwards": 2, "logits_match_reference_bits": True, "selected_values_match_reference_bits": True,
              "top_logit_index": int(result.output.logits.argmax(dim=-1)[0]),
              "image_metadata": result.image_metadata, "tensor_bytes": result.tensor_bytes,
              "records": [{"event": record.event.to_dict(), "token_kind": record.token_kind,
                           "patch_row": record.patch_row, "patch_column": record.patch_column,
                           "processed_pixel_box_xyxy": record.processed_pixel_box_xyxy,
                           "value_sha256": hashlib.sha256(record.value.view(torch.int32).contiguous().numpy().tobytes()).hexdigest()}
                          for record in result.records],
              "note": "This report stores hashes and metadata, not raw image, logits or activation vectors."}
    raw = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with args.output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps({"status": "complete", "report_sha256": hashlib.sha256(raw).hexdigest(), "forwards": 2}))


if __name__ == "__main__":
    main()
