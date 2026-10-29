"""Prepare already-exposed ViT pixels and CPU evidence without model forwards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

CASE_SHA = "8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207"
TERMINAL_SHA = "2b61dfc59a8ab6c1072293e97d8e8ad42ceea8c4c1a0956bc9e7c992c2f3a9cc"
REFERENCE = {
    "forward-001": ("pair-04:cat:initial", "a27362618d930fac3a9f5195f4b3bfbace32f595ebdfe2929489fb37e5db001e",
                    "51f9649f275e60cbb7725ff946b71b2692a44c465bbd894dc648d7e6907d27ee"),
    "forward-002": ("pair-04:dog:initial", "40e05aa418154efdbe6fca10c4d8e0d79a2f77440f941717dfa8f0f7ec921b68",
                    "c40086eecdad0b63655abca098b9de143f1fb11c537b899d10f1b73c14697454"),
    "forward-024": ("pair-04:cat:natural_inside", "967a8810e7a3a9301335eb22981a1867108bfd802005f48f091a798c52794bb7",
                    "ae8915447c22e956fa83b4db125bbec48bd1f1f8243b68f5378eb305a4f357ad"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with path.open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short reference write")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(manifest_path, assets, cpu_run, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("reference preparation requires an empty CUDA mask")
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoImageProcessor

    if torch.cuda.is_initialized():
        raise ValueError("CUDA initialized during reference preparation")
    if (np.__version__ != "2.4.6" or PIL.__version__ != "11.3.0"
            or torch.__version__ != "2.8.0+cu128" or transformers.__version__ != "4.57.3"):
        raise ValueError("reference runtime differs from v1")
    case_raw = manifest_path.read_bytes()
    if sha(case_raw) != CASE_SHA:
        raise ValueError("case manifest differs")
    manifest = json.loads(case_raw)
    pair = next(row for row in manifest["pairs"] if row["pair_id"] == "pair-04")
    if (pair["split"] != "heldout" or pair["cat"]["image_id"] != 520531
            or pair["dog"]["image_id"] != 377575 or pair["box_tokens"] != [161, 162, 175, 176]):
        raise ValueError("the previously exposed pair differs")
    terminal_raw = (cpu_run / "data/terminal.json").read_bytes()
    supervisor = json.loads((cpu_run / "supervisor.json").read_bytes())
    terminal = json.loads(terminal_raw)
    if (sha(terminal_raw) != TERMINAL_SHA or supervisor.get("status") != "complete"
            or supervisor.get("clean") is not True or type(supervisor.get("exit_code")) is not int
            or supervisor["exit_code"] != 0 or supervisor.get("timed_out") is not False
            or supervisor.get("terminal_sha256") != TERMINAL_SHA
            or terminal.get("source_revision") != "b56d07e16d275a0f3ee9326bcd878277b89ec99d"):
        raise ValueError("CPU reference run did not match the recorded receipt")
    for name, expected in manifest["model_files"].items():
        if sha((assets / name).read_bytes()) != expected:
            raise ValueError("pinned model or processor asset differs")
    processor = AutoImageProcessor.from_pretrained(str(assets), local_files_only=True, use_fast=False)
    if (type(processor).__name__ != "ViTImageProcessor" or processor.size != {"height": 224, "width": 224}
            or int(processor.resample) != 2 or processor.image_mean != [.5] * 3
            or processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
        raise ValueError("processor differs from CPU study")
    # All validation above is read-only; reserve a fresh attempt before writing.
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for side in ("cat", "dog"):
        info = pair[side]
        path = assets / "images" / info["file_name"]
        raw = path.read_bytes()
        if len(raw) != info["jpeg_bytes"] or sha(raw) != info["jpeg_sha256"]:
            raise ValueError("previously exposed image bytes differ")
        with Image.open(path) as image:
            if image.size != (info["width"], info["height"]) or image.getexif().get(274, 1) != 1:
                raise ValueError("image geometry differs")
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        if (tuple(pixels.shape) != (1, 3, 224, 224) or pixels.dtype != torch.float32
                or not bool(torch.isfinite(pixels).all())):
            raise ValueError("invalid processed pixels")
        data = pixels.contiguous().numpy().astype("<f4", copy=False).tobytes()
        if len(data) != 602112 or sha(data) != info["pixel_sha256"]:
            raise ValueError("processed pixels differ from v1")
        name = f"{side}-pixels.f32"
        write_new(output / name, data)
        inventory[name] = {"bytes": len(data), "sha256": sha(data), "image_id": info["image_id"]}
    for name, (label, json_sha, binary_sha) in REFERENCE.items():
        record = (cpu_run / "data" / (name + ".json")).read_bytes()
        binary = (cpu_run / "data" / (name + ".bin")).read_bytes()
        if sha(record) != json_sha or sha(binary) != binary_sha:
            raise ValueError("saved CPU forward bytes differ")
        row = json.loads(record)
        if row.get("label") != label or row.get("sha256") != binary_sha or row.get("file") != name + ".bin":
            raise ValueError("saved CPU forward identity differs")
        for suffix, raw in ((".json", record), (".bin", binary)):
            filename = "cpu-" + name + suffix
            write_new(output / filename, raw)
            inventory[filename] = {"bytes": len(raw), "sha256": sha(raw)}
    # Confirm the cross-runtime comparison uses the exact donor bytes that the
    # CPU natural-inside intervention requested, not a recomputed direction.
    donor = (output / "cpu-forward-002.bin").read_bytes()
    patched = (output / "cpu-forward-024.bin").read_bytes()
    token_width = 192 * 4
    selected = b"".join(donor[4000 + token * token_width:4000 + (token + 1) * token_width]
                        for token in pair["box_tokens"])
    if selected != patched[312736:315808]:
        raise ValueError("CPU donor and natural-inside requested bytes differ")
    metadata = {"schema": "vision-vit-cuda-reference-v1", "status": "complete", "model_forwards": 0,
                "pair_id": "pair-04", "tokens": pair["box_tokens"], "case_sha256": CASE_SHA,
                "cpu_terminal_sha256": TERMINAL_SHA, "files": inventory,
                "scope": "previously exposed public images and already saved CPU forward bytes"}
    write_new(output / "reference.json", (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode())
    print(json.dumps({"status": "complete", "model_forwards": 0, "reference_sha256": sha((output / 'reference.json').read_bytes())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cpu-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1] / "experiments/vision_coco_spatial_v1.json",
            args.assets, args.cpu_run, args.output)


if __name__ == "__main__":
    main()
