"""Recompute same-species donor choices from the fixed annotation and split.

First check MANIFEST and image bytes with audit_vision_coco_manifest.py.
Usage: python audit_vision_coco_donors.py MANIFEST DONORS ANNOTATIONS PIL_DIR
"""

import json
from pathlib import Path
import sys

from audit_vision_coco_manifest import ANNOTATION_HASH, digest, parse, require, typed_equal


def audit(manifest_path, donor_path, annotations_path, pil_dir):
    sys.path.insert(0, str(pil_dir.resolve()))
    import PIL
    import numpy as np
    from PIL import Image, ImageDraw

    require(PIL.__version__ == "11.3.0", "polygon rasterizer version differs")
    manifest_bytes = manifest_path.read_bytes()
    donor_bytes = donor_path.read_bytes()
    ann_bytes = annotations_path.read_bytes()
    require(digest(ann_bytes) == ANNOTATION_HASH, "annotation bytes differ")
    manifest, donors, annotation = parse(manifest_bytes), parse(donor_bytes), parse(ann_bytes)
    require(donor_bytes == (json.dumps(donors, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(),
            "donor manifest is not canonical")
    require(donors["schema"] == "vision-coco-same-species-donors-v1"
            and donors["annotation_sha256"] == ANNOTATION_HASH
            and donors["case_manifest_sha256"] == digest(manifest_bytes)
            and donors["selection"] == "lowest image ID, same split and species, different pair, box contains block, mask >=512/1024",
            "donor provenance differs")
    pairs = manifest["pairs"]
    require(len(pairs) == 12 and len(donors["rows"]) == 24, "donor inventory size")
    by_id = {item["id"]: item for item in annotation["annotations"]}
    unavailable = {"development": 0, "heldout": 0}
    for index, receiver in enumerate(pairs):
        require(receiver["split"] == ("development" if index < 4 else "heldout"), "unexpected split")
        row, col = receiver["box_block_row_col"]
        for offset, side in enumerate(("cat", "dog")):
            item = donors["rows"][2 * index + offset]
            for key, value in {"pair_id": receiver["pair_id"], "split": receiver["split"],
                               "receiver_side": side, "receiver_image_id": receiver[side]["image_id"],
                               "inside_row_col": [row, col]}.items():
                require(typed_equal(item[key], value), "receiver identity differs")
            candidates = []
            for candidate in pairs:
                if candidate["split"] != receiver["split"] or candidate["pair_id"] == receiver["pair_id"]:
                    continue
                image = candidate[side]
                ann = by_id[image["annotation_id"]]
                require(ann["image_id"] == image["image_id"] and ann["category_id"] == image["category_id"],
                        "donor annotation identity differs")
                width, height = image["width"], image["height"]
                x, y, w, h = ann["bbox"]
                bounds = (x * 224 / width, y * 224 / height, (x + w) * 224 / width, (y + h) * 224 / height)
                if not (16 * col >= bounds[0] and 16 * row >= bounds[1]
                        and 16 * (col + 2) <= bounds[2] and 16 * (row + 2) <= bounds[3]):
                    continue
                mask = Image.new("L", (224, 224), 0)
                draw = ImageDraw.Draw(mask)
                for polygon in ann["segmentation"]:
                    draw.polygon([(polygon[i] * 224 / width, polygon[i + 1] * 224 / height)
                                  for i in range(0, len(polygon), 2)], fill=1)
                pixels = np.asarray(mask)
                counts = [int(pixels[16 * (row + dr):16 * (row + dr + 1),
                                     16 * (col + dc):16 * (col + dc + 1)].sum())
                          for dr in range(2) for dc in range(2)]
                if sum(counts) >= 512:
                    candidates.append({"donor_pair_id": candidate["pair_id"], "donor_image_id": image["image_id"],
                                       "donor_mask_sha256": digest(mask.tobytes()),
                                       "donor_token_mask_pixels": counts, "donor_block_mask_pixels": sum(counts)})
            expected = min(candidates, key=lambda value: value["donor_image_id"]) if candidates else None
            require(typed_equal(item["donor"], expected), "same-species donor differs")
            if expected is None:
                unavailable[receiver["split"]] += 1
    return {"status": "donor_manifest_consistency_verified", "case_manifest_sha256": digest(manifest_bytes),
            "donor_manifest_sha256": digest(donor_bytes), "receiver_sites": len(donors["rows"]),
            "unavailable_by_split": unavailable,
            "limitations": ["This checks saved donor selection, not model execution or donor intervention effects."]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("donors", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("pil_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.donors, args.annotations, args.pil_dir), indent=2))
