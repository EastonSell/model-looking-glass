"""Model-blind selection of a small, box-annotated COCO vision sample.

The source JSON is a separately pinned public mirror of COCO 2017 annotations.
No model output is used when forming the ordered, disjoint image pairs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ANNOTATION_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
ANNOTATION_URL = (
    "https://huggingface.co/datasets/pcuenq/coco2017-instances/resolve/"
    "0d2b1478a7c692d6227612841b28512da3def874/instances_val2017.json"
)
MODEL_REVISION = "77d1485af66b34d4ed0fe95dbb0c60c7496f950b"
MODEL_REPO = "WinKawaks/vit-tiny-patch16-224"
MODEL_FILES = {
    "config.json": "7e5aeb1add0a832153071a858bb18145cafb809e54e7fea4e5dc1183c82078f5",
    "preprocessor_config.json": "90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2",
    "model.safetensors": "a9146f3ee7cf6de399190132627e4aef1aeddd13692f5b333f24629dd36c37b0",
}
GROUPS = {"cat": list(range(281, 286)), "dog": list(range(151, 269))}
NUM_PAIRS = 12
DEVELOPMENT_PAIRS = 4


def _radius(square):
    row, col = square
    # 2x2 block center, in the 14x14 patch grid; image center is at (7, 7).
    return (row - 6) ** 2 + (col - 6) ** 2


def _token_ids(square):
    row, col = square
    return [1 + 14 * (row + dr) + col + dc for dr in range(2) for dc in range(2)]


def _squares(image, annotation):
    """Find blocks wholly inside or wholly outside this transformed box."""
    x, y, width, height = annotation["bbox"]
    image_width, image_height = image["width"], image["height"]
    box = (x * 224 / image_width, y * 224 / image_height,
           (x + width) * 224 / image_width, (y + height) * 224 / image_height)
    inside, outside = set(), set()
    for row in range(13):
        for col in range(13):
            left, top, right, bottom = 16 * col, 16 * row, 16 * (col + 2), 16 * (row + 2)
            if left >= box[0] and top >= box[1] and right <= box[2] and bottom <= box[3]:
                inside.add((row, col))
            if right <= box[0] or left >= box[2] or bottom <= box[1] or top >= box[3]:
                outside.add((row, col))
    return inside, outside, box


def select_pairs(annotation_bytes, *, with_diagnostics=False):
    """Return the 12 frozen image/position pairs from the pinned annotations.

    Rasterize non-crowd COCO polygons directly onto the 224px processor grid
    with Pillow 11.3's filled-polygon semantics. At least half of each chosen
    32x32 block must lie within each image's independently rasterized mask.
    """
    import PIL
    from PIL import Image, ImageDraw
    if PIL.__version__ != "11.3.0":
        raise ValueError("polygon rasterization requires Pillow 11.3.0")
    if hashlib.sha256(annotation_bytes).hexdigest() != ANNOTATION_SHA256:
        raise ValueError("COCO annotation mirror differs from its pinned bytes")
    data = json.loads(annotation_bytes)
    categories = {item["name"]: item["id"] for item in data["categories"]}
    if categories.get("cat") != 17 or categories.get("dog") != 18:
        raise ValueError("unexpected cat/dog category IDs")
    images = {item["id"]: item for item in data["images"]}
    animal_annotations, all_annotations = {}, {}
    for annotation in data["annotations"]:
        all_annotations.setdefault(annotation["image_id"], []).append(annotation)
        if annotation["category_id"] in (17, 18):
            animal_annotations.setdefault(annotation["image_id"], []).append(annotation)
    licenses = {item["id"]: item for item in data["licenses"]}
    candidates = {17: [], 18: []}
    geometry_candidates = {17: 0, 18: 0}
    rejected = {"no_polygon": 0, "no_qualified_inside_block": 0,
                "no_common_qualified_block": 0, "no_equal_radius_control": 0}
    for image_id, annotations in animal_annotations.items():
        image = images[image_id]
        if (len(annotations) != 1 or annotations[0]["iscrowd"]
                or image["license"] not in (4, 5, 7, 8) or image_id == 39769):
            continue
        annotation = annotations[0]
        x, y, width, height = annotation["bbox"]
        image_width, image_height = image["width"], image["height"]
        if (image_width < 224 or image_height < 224
                or max(image_width / image_height, image_height / image_width) > 2
                or not .08 <= width * height / (image_width * image_height) <= .70
                or width * 224 / image_width < 32 or height * 224 / image_height < 32):
            continue
        if not all(math.isfinite(float(v)) for v in (x, y, width, height)):
            raise ValueError("nonfinite COCO box")
        inside, outside, box = _squares(image, annotation)
        if inside and outside:
            geometry_candidates[annotation["category_id"]] += 1
            polygons = annotation.get("segmentation")
            if (type(polygons) is not list or not polygons
                    or any(type(p) is not list or len(p) < 6 or len(p) % 2 for p in polygons)):
                rejected["no_polygon"] += 1
                continue
            mask = Image.new("L", (224, 224), 0)
            draw = ImageDraw.Draw(mask)
            for polygon in polygons:
                if not all(math.isfinite(float(value)) for value in polygon):
                    raise ValueError("nonfinite COCO polygon")
                draw.polygon([(polygon[i] * 224 / image_width, polygon[i + 1] * 224 / image_height)
                              for i in range(0, len(polygon), 2)], fill=1)
            token_occupancy = {
                square: [sum(mask.crop((16 * (square[1] + dc), 16 * (square[0] + dr),
                                        16 * (square[1] + dc + 1), 16 * (square[0] + dr + 1))).getdata())
                         for dr in range(2) for dc in range(2)]
                for square in inside}
            occupancy = {square: sum(counts) for square, counts in token_occupancy.items()}
            qualified = {square for square, count in occupancy.items() if count >= 512}
            if not qualified:
                rejected["no_qualified_inside_block"] += 1
                continue
            candidates[annotation["category_id"]].append(
                (image, annotation, qualified, outside, box, occupancy, token_occupancy,
                 hashlib.sha256(mask.tobytes()).hexdigest()))
    for category, rows in candidates.items():
        rows.sort(key=lambda row: hashlib.sha256(
            f"vision-coco-v1:{category}:{row[0]['id']}".encode("ascii")).hexdigest())

    used_dogs, pairs = set(), []
    for cat in candidates[17]:
        for dog in candidates[18]:
            if dog[0]["id"] in used_dogs:
                continue
            inside, outside = cat[2] & dog[2], cat[3] & dog[3]
            if not inside or not outside:
                rejected["no_common_qualified_block"] += 1
                continue
            choices = ((a, b) for a in inside for b in outside if _radius(a) == _radius(b))
            try:
                interior, control = min(
                    choices, key=lambda value: (
                        abs(value[0][0] - value[1][0]) + abs(value[0][1] - value[1][1]),
                        _radius(value[0]), value))
            except ValueError:
                rejected["no_equal_radius_control"] += 1
                continue
            used_dogs.add(dog[0]["id"])
            def describe(row):
                image, annotation, _, _, box, occupancy, token_occupancy, mask_sha256 = row
                left, top = 16 * control[1], 16 * control[0]
                right, bottom = left + 32, top + 32
                other_overlaps = []
                for other in all_annotations[image["id"]]:
                    if other["id"] == annotation["id"]:
                        continue
                    x, y, width, height = other["bbox"]
                    x0, x1 = x * 224 / image["width"], (x + width) * 224 / image["width"]
                    y0, y1 = y * 224 / image["height"], (y + height) * 224 / image["height"]
                    if left < x1 and right > x0 and top < y1 and bottom > y0:
                        other_overlaps.append({"annotation_id": other["id"],
                                               "category_id": other["category_id"]})
                other_overlaps.sort(key=lambda value: value["annotation_id"])
                return {
                    "image_id": image["id"], "file_name": image["file_name"],
                    "width": image["width"], "height": image["height"],
                    "coco_url": image["coco_url"], "flickr_url": image["flickr_url"],
                    "license": licenses[image["license"]], "annotation_id": annotation["id"],
                    "category_id": annotation["category_id"], "bbox_xywh": annotation["bbox"],
                    "bbox_resized_xyxy": list(box), "mask_sha256": mask_sha256,
                    "inside_block_mask_pixels": occupancy[interior],
                    "inside_token_mask_pixels": token_occupancy[interior],
                    "control_other_box_overlaps": other_overlaps,
                }
            number = len(pairs)
            pairs.append({
                "pair_id": f"pair-{number:02d}",
                "split": "development" if number < DEVELOPMENT_PAIRS else "heldout",
                "cat": describe(cat), "dog": describe(dog),
                "box_block_row_col": list(interior), "control_block_row_col": list(control),
                "box_tokens": _token_ids(interior), "control_tokens": _token_ids(control),
                "squared_center_radius": _radius(interior),
            })
            break
        if len(pairs) == NUM_PAIRS:
            break
    if len(pairs) != NUM_PAIRS:
        raise ValueError("not enough geometry-matched pairs in the frozen annotations")
    diagnostics = {"geometry_candidates": {str(k): v for k, v in geometry_candidates.items()},
                   "mask_qualified_candidates": {str(k): len(v) for k, v in candidates.items()},
                   "rejected_attempt_counts": rejected, "minimum_mask_pixels_per_image": 512,
                   "mask_raster": "Pillow 11.3.0 ImageDraw filled COCO polygons directly at 224x224"}
    return (pairs, diagnostics) if with_diagnostics else pairs


def verify_assets(asset_dir):
    asset_dir = Path(asset_dir)
    for name, expected in MODEL_FILES.items():
        if hashlib.sha256((asset_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"model asset differs: {name}")
    return dict(MODEL_FILES)
