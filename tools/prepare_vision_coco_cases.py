"""Freeze the model-blind COCO image/geometry manifest; no model forward.

The pinned annotation and model files must already be in --assets. This tool
downloads only the predetermined photos, verifies their recorded dimensions,
and hashes both the source JPEG and the actual slow-processor float32 pixels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from looking_glass.vision_coco_cases import (
    ANNOTATION_SHA256, ANNOTATION_URL, GROUPS, MODEL_FILES, MODEL_REPO,
    MODEL_REVISION, select_pairs, verify_assets,
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    annotation_bytes = (args.assets / "instances_val2017.json").read_bytes()
    pairs, diagnostics = select_pairs(annotation_bytes, with_diagnostics=True)
    verify_assets(args.assets)
    config = json.loads((args.assets / "config.json").read_bytes())
    groups = {name: [{"id": index, "label": config["id2label"][str(index)]} for index in ids]
              for name, ids in GROUPS.items()}
    if (len(config["id2label"]) != 1000 or groups["dog"][0]["label"] != "Chihuahua"
            or groups["dog"][-1]["label"] != "Mexican hairless"
            or "tabby" not in groups["cat"][0]["label"]):
        raise ValueError("fixed ImageNet class group differs")
    image_dir = args.assets / "images"
    image_dir.mkdir(exist_ok=True)

    def download(image):
        destination = image_dir / image["file_name"]
        image["mirror_url"] = ("https://huggingface.co/datasets/merve/coco/resolve/"
                               "9e50abcdc1361852f34841af4939cbcd2d37c92f/val2017/"
                               + image["file_name"])
        request = urllib.request.Request(image["mirror_url"],
                                         headers={"User-Agent": "model-looking-glass-vision-study/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(5_000_001)
        if not raw or len(raw) > 5_000_000:
            raise ValueError("empty or oversized image")
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError("existing image differs from the pinned HTTPS mirror")
        else:
            with destination.open("xb") as stream:
                if stream.write(raw) != len(raw):
                    raise OSError("short image write")
        return destination

    images = [pair[side] for pair in pairs for side in ("cat", "dog")]
    if len({item["image_id"] for item in images}) != 24:
        raise ValueError("image IDs must be distinct across all splits")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, images))

    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoImageProcessor
    if torch.cuda.is_initialized():
        raise ValueError("this preparation must not initialize CUDA")
    processor = AutoImageProcessor.from_pretrained(str(args.assets), local_files_only=True, use_fast=False)
    if (type(processor).__name__ != "ViTImageProcessor" or processor.size != {"height": 224, "width": 224}
            or int(processor.resample) != 2 or processor.image_mean != [0.5] * 3
            or processor.image_std != [0.5] * 3 or processor.rescale_factor != 1 / 255):
        raise ValueError("image processor differs from the pinned preprocessing contract")
    for item in images:
        raw = (image_dir / item["file_name"]).read_bytes()
        with Image.open(image_dir / item["file_name"]) as image:
            if image.size != (item["width"], item["height"]) or image.getexif().get(274, 1) != 1:
                raise ValueError("decoded dimensions or EXIF orientation differ from COCO geometry")
            # Do not auto-rotate: the recorded COCO boxes refer to these pixels.
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        if pixels.shape != (1, 3, 224, 224) or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
            raise ValueError("invalid processed pixels")
        item["jpeg_bytes"] = len(raw)
        item["jpeg_sha256"] = digest(raw)
        item["pixel_sha256"] = digest(pixels.contiguous().numpy().astype("<f4", copy=False).tobytes())
    if len({item["pixel_sha256"] for item in images}) != len(images):
        raise ValueError("duplicate processed images across the selected pairs")
    manifest = {
        "schema": "vision-coco-vit-spatial-cases-v1", "annotation_url": ANNOTATION_URL,
        "annotation_sha256": ANNOTATION_SHA256, "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION, "model_files": MODEL_FILES, "class_groups": groups,
        "processor": {"class": type(processor).__name__, "size": processor.size,
                      "resample": "bilinear", "mean": processor.image_mean, "std": processor.image_std,
                      "rescale_factor": processor.rescale_factor, "exif_transpose": False},
        "preparation_versions": {"python": sys.version.split()[0], "pillow": PIL.__version__,
                                 "numpy": np.__version__, "torch": torch.__version__,
                                 "transformers": transformers.__version__},
        "selection_diagnostics": diagnostics, "pairs": pairs,
    }
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short manifest write")
    print(json.dumps({"manifest_sha256": digest(raw), "pairs": len(pairs),
                      "development": 4, "heldout": 8, "model_forwards": 0}))


if __name__ == "__main__":
    main()
