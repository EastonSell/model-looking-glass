"""Model-blind, same-split descriptive donor selection for the vision study."""
from __future__ import annotations

import hashlib
import json

from .vision_coco_cases import ANNOTATION_SHA256, _squares


def select_donors(manifest_bytes, annotation_bytes):
    """Choose the lowest-image-ID eligible same-species donor at each site.

    Donors are other images from the same frozen split. Their selected animal
    box must contain the receiver's block and their rasterized animal mask
    must cover at least 512 of its 1,024 pixels. No model output is consulted.
    """
    from PIL import Image, ImageDraw
    import PIL
    if PIL.__version__ != "11.3.0":
        raise ValueError("donor selection requires Pillow 11.3.0")
    if hashlib.sha256(annotation_bytes).hexdigest() != ANNOTATION_SHA256:
        raise ValueError("unexpected annotation bytes")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "vision-coco-vit-spatial-cases-v1":
        raise ValueError("unexpected case manifest")
    pairs = manifest["pairs"]
    if len(pairs) != 12 or len({p[s]["image_id"] for p in pairs for s in ("cat", "dog")}) != 24:
        raise ValueError("unexpected pair inventory")
    data = json.loads(annotation_bytes)
    annotations = {a["id"]: a for a in data["annotations"]}
    rows = []
    for receiver in pairs:
        row, col = receiver["box_block_row_col"]
        for side in ("cat", "dog"):
            choices = []
            for candidate in pairs:
                if candidate["split"] != receiver["split"] or candidate["pair_id"] == receiver["pair_id"]:
                    continue
                image = candidate[side]
                annotation = annotations[image["annotation_id"]]
                if (annotation["image_id"] != image["image_id"]
                        or annotation["category_id"] != image["category_id"]):
                    raise ValueError("annotation identity mismatch")
                inside, _, _ = _squares(image, annotation)
                if (row, col) not in inside:
                    continue
                mask = Image.new("L", (224, 224), 0)
                draw = ImageDraw.Draw(mask)
                for polygon in annotation["segmentation"]:
                    draw.polygon([(polygon[i] * 224 / image["width"],
                                   polygon[i + 1] * 224 / image["height"])
                                  for i in range(0, len(polygon), 2)], fill=1)
                counts = [sum(mask.crop((16 * (col + dc), 16 * (row + dr),
                                         16 * (col + dc + 1), 16 * (row + dr + 1))).getdata())
                          for dr in range(2) for dc in range(2)]
                if sum(counts) >= 512:
                    choices.append({"donor_pair_id": candidate["pair_id"],
                                    "donor_image_id": image["image_id"],
                                    "donor_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                                    "donor_token_mask_pixels": counts,
                                    "donor_block_mask_pixels": sum(counts)})
            chosen = min(choices, key=lambda v: v["donor_image_id"]) if choices else None
            rows.append({"pair_id": receiver["pair_id"], "split": receiver["split"],
                         "receiver_side": side, "receiver_image_id": receiver[side]["image_id"],
                         "inside_row_col": [row, col], "donor": chosen})
    return {"schema": "vision-coco-same-species-donors-v1",
            "case_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "annotation_sha256": ANNOTATION_SHA256,
            "selection": "lowest image ID, same split and species, different pair, box contains block, mask >=512/1024",
            "rows": rows}


def canonical_donors(manifest_bytes, annotation_bytes):
    return (json.dumps(select_donors(manifest_bytes, annotation_bytes), sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
