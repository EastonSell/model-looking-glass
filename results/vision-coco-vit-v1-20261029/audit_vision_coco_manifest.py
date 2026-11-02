"""Independently check a frozen COCO/ViT manifest without loading a model.

Usage: python audit_vision_coco_manifest.py MANIFEST ASSET_DIR PIL_DIR
       [--verify-https-mirror]

The image check uses PIL and NumPy independently of the Transformers processor.
HTTPS comparison establishes agreement with a pinned third-party mirror, not
that the mirror is the official COCO archive or that images were unexposed.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request


ANNOTATION_HASH = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
ANNOTATION_URL = ("https://huggingface.co/datasets/pcuenq/coco2017-instances/resolve/"
                  "0d2b1478a7c692d6227612841b28512da3def874/instances_val2017.json")
MODEL_REPO = "WinKawaks/vit-tiny-patch16-224"
MODEL_REVISION = "77d1485af66b34d4ed0fe95dbb0c60c7496f950b"
MODEL_HASHES = {
    "config.json": "7e5aeb1add0a832153071a858bb18145cafb809e54e7fea4e5dc1183c82078f5",
    "preprocessor_config.json": "90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2",
    "model.safetensors": "a9146f3ee7cf6de399190132627e4aef1aeddd13692f5b333f24629dd36c37b0",
}
MIRROR = ("https://huggingface.co/datasets/merve/coco/resolve/"
          "9e50abcdc1361852f34841af4939cbcd2d37c92f/val2017/")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(typed_equal(a[key], b[key]) for key in a)
    if type(a) is list:
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number")
        return number

    def constant(_):
        raise ValueError("nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=unique, parse_float=finite, parse_constant=constant)


def audit(manifest_path, asset_dir, pil_dir, verify_mirror):
    sys.path.insert(0, str(pil_dir.resolve()))
    import numpy as np
    import PIL
    from PIL import Image, ImageDraw

    raw = manifest_path.read_bytes()
    manifest = parse(raw)
    require(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(),
            "manifest is not canonical")
    require(manifest["annotation_sha256"] == ANNOTATION_HASH and manifest["annotation_url"] == ANNOTATION_URL
            and manifest["model_repo"] == MODEL_REPO
            and manifest["model_revision"] == MODEL_REVISION and manifest["model_files"] == MODEL_HASHES,
            "source pins differ")
    annotation_raw = (asset_dir / "instances_val2017.json").read_bytes()
    require(digest(annotation_raw) == ANNOTATION_HASH, "annotation bytes differ")
    for name, expected in MODEL_HASHES.items():
        require(digest((asset_dir / name).read_bytes()) == expected, "model asset differs: " + name)
    annotation = parse(annotation_raw)
    source_images = {item["id"]: item for item in annotation["images"]}
    source_annotations = {item["id"]: item for item in annotation["annotations"]}
    source_licenses = {item["id"]: item for item in annotation["licenses"]}
    other_annotations = {}
    for item in annotation["annotations"]:
        other_annotations.setdefault(item["image_id"], []).append(item)
    require(len(source_images) == len(annotation["images"])
            and len(source_annotations) == len(annotation["annotations"]), "duplicate annotation identity")
    config = parse((asset_dir / "config.json").read_bytes())
    wanted_groups = {label: [{"id": n, "label": config["id2label"][str(n)]} for n in indices]
                     for label, indices in (("cat", range(281, 286)), ("dog", range(151, 269)))}
    require(typed_equal(manifest["class_groups"], wanted_groups), "class labels differ from model config")
    processor = manifest["processor"]
    require(PIL.__version__ == "11.3.0" and manifest["preparation_versions"]["pillow"] == PIL.__version__,
            "polygon rasterizer version differs")
    require(processor["class"] == "ViTImageProcessor" and processor["size"] == {"height": 224, "width": 224}
            and processor["resample"] == "bilinear"
            and processor["mean"] == [0.5] * 3 and processor["std"] == [0.5] * 3
            and processor["rescale_factor"] == 1 / 255 and processor["exif_transpose"] is False,
            "processor contract differs")
    pairs = manifest["pairs"]
    require(type(pairs) is list and len(pairs) == 12, "pair count")
    image_ids, jpg_hashes, pixel_hashes, all_images = set(), set(), set(), []
    for number, pair in enumerate(pairs):
        require(pair["pair_id"] == f"pair-{number:02d}"
                and pair["split"] == ("development" if number < 4 else "heldout"), "split or pair order")
        boxblock, control = pair["box_block_row_col"], pair["control_block_row_col"]
        for block, token_key in ((boxblock, "box_tokens"), (control, "control_tokens")):
            require(type(block) is list and len(block) == 2 and all(type(x) is int and 0 <= x < 13 for x in block),
                    "invalid patch block")
            row, col = block
            require(typed_equal(pair[token_key], [1 + 14 * (row + dr) + col + dc for dr in range(2) for dc in range(2)]),
                    "patch token numbering")
        radius = lambda point: (point[0] - 6) ** 2 + (point[1] - 6) ** 2
        require(radius(boxblock) == radius(control) and typed_equal(radius(boxblock), pair["squared_center_radius"]),
                "unequal position radius")
        for side, category in (("cat", 17), ("dog", 18)):
            x = pair[side]
            image_id, ann_id = x["image_id"], x["annotation_id"]
            require(type(image_id) is int and image_id not in image_ids and type(ann_id) is int,
                    "reused or invalid image identity")
            image_ids.add(image_id)
            image, ann = source_images[image_id], source_annotations[ann_id]
            require(ann["image_id"] == image_id and ann["category_id"] == category
                    and x["category_id"] == category and x["bbox_xywh"] == ann["bbox"],
                    "annotation/box identity")
            for key in ("file_name", "width", "height", "coco_url", "flickr_url"):
                require(x[key] == image[key], "image metadata: " + key)
            require(x["license"] == source_licenses[image["license"]], "license metadata")
            name = x["file_name"]
            require(type(name) is str and name == f"{image_id:012d}.jpg", "invalid filename")
            require(x["mirror_url"] == MIRROR + name, "HTTPS mirror URL differs")
            width, height = image["width"], image["height"]
            left, top, w, h = ann["bbox"]
            box = [left * 224 / width, top * 224 / height, (left + w) * 224 / width, (top + h) * 224 / height]
            require(typed_equal(x["bbox_resized_xyxy"], box), "resized box differs")
            for block, inside in ((boxblock, True), (control, False)):
                row, col = block
                l, t, r, b = 16 * col, 16 * row, 16 * (col + 2), 16 * (row + 2)
                condition = (l >= box[0] and t >= box[1] and r <= box[2] and b <= box[3]) if inside else (
                    r <= box[0] or l >= box[2] or b <= box[1] or t >= box[3])
                require(condition, "block/box relationship")
            polygons = ann["segmentation"]
            require(type(polygons) is list and polygons and all(type(p) is list and len(p) >= 6 and len(p) % 2 == 0
                                                               for p in polygons), "invalid polygon")
            mask = Image.new("L", (224, 224), 0)
            painter = ImageDraw.Draw(mask)
            for polygon in polygons:
                painter.polygon([(polygon[i] * 224 / width, polygon[i + 1] * 224 / height)
                                 for i in range(0, len(polygon), 2)], fill=1)
            require(digest(mask.tobytes()) == x["mask_sha256"], "polygon mask digest differs")
            mask_array = np.asarray(mask)
            row, col = boxblock
            counts = [int(mask_array[16 * (row + dr):16 * (row + dr + 1),
                                     16 * (col + dc):16 * (col + dc + 1)].sum())
                      for dr in range(2) for dc in range(2)]
            require(typed_equal(x["inside_token_mask_pixels"], counts)
                    and typed_equal(x["inside_block_mask_pixels"], sum(counts))
                    and sum(counts) >= 512, "polygon coverage differs")
            row, col = control
            left, top, right, bottom = 16 * col, 16 * row, 16 * (col + 2), 16 * (row + 2)
            overlaps = []
            for other in other_annotations[image_id]:
                if other["id"] == ann_id:
                    continue
                ox, oy, ow, oh = other["bbox"]
                x0, x1, y0, y1 = ox * 224 / width, (ox + ow) * 224 / width, oy * 224 / height, (oy + oh) * 224 / height
                if left < x1 and right > x0 and top < y1 and bottom > y0:
                    overlaps.append({"annotation_id": other["id"], "category_id": other["category_id"]})
            overlaps.sort(key=lambda value: value["annotation_id"])
            require(typed_equal(overlaps, x["control_other_box_overlaps"]), "control overlap inventory differs")
            raw_jpg = (asset_dir / "images" / name).read_bytes()
            jhash = digest(raw_jpg)
            require(type(x["jpeg_bytes"]) is int and len(raw_jpg) == x["jpeg_bytes"]
                    and jhash == x["jpeg_sha256"] and jhash not in jpg_hashes, "JPEG mismatch or duplicate")
            jpg_hashes.add(jhash)
            with Image.open(asset_dir / "images" / name) as image_pixels:
                require(image_pixels.size == (width, height) and image_pixels.getexif().get(274, 1) == 1,
                        "image dimensions or orientation")
                resized = image_pixels.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                pixels = (np.asarray(resized).astype(np.float64) * (1 / 255)).astype(np.float32)
                pixels = (pixels - np.float32(.5)) / np.float32(.5)
                pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1).astype("<f4", copy=False))
            phash = digest(pixels.tobytes())
            require(phash == x["pixel_sha256"] and phash not in pixel_hashes, "pixel mismatch or duplicate")
            pixel_hashes.add(phash)
            all_images.append((name, len(raw_jpg), jhash))
    if verify_mirror:
        def verify(item):
            name, size, expected = item
            request = urllib.request.Request(MIRROR + name, headers={"User-Agent": "model-looking-glass-vision-audit/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(5_000_001)
            return len(data) == size and digest(data) == expected

        with ThreadPoolExecutor(max_workers=4) as pool:
            require(all(pool.map(verify, all_images)), "pinned HTTPS mirror mismatch")
    return {
        "status": "manifest_consistency_verified", "manifest_sha256": digest(raw),
        "auditor_sha256": digest(Path(__file__).read_bytes()),
        "annotation_sha256": ANNOTATION_HASH, "model_repository": MODEL_REPO,
        "model_revision": MODEL_REVISION, "model_file_sha256": MODEL_HASHES,
        "image_mirror_revision": "9e50abcdc1361852f34841af4939cbcd2d37c92f",
        "rasterizer_version": PIL.__version__, "numpy_version": np.__version__,
        "pairs": len(pairs), "development_images": 8, "heldout_images": 16,
        "distinct_jpeg_and_pixel_hashes": len(all_images), "https_mirror_checked": verify_mirror,
        "checks": ["JPEG bytes, decoded dimensions, EXIF orientation and independently processed pixel hashes",
                   "annotation identities, resized boxes, polygon masks, token occupancy and other-object overlaps",
                   "pair order, disjoint image IDs, patch coordinates and matched positional radii"],
        "limitations": ["The image and annotation sources are third-party mirrors, not authenticated official COCO provenance.",
                        "Byte hashes and fixed splits do not establish absence from model pretraining or prove model execution.",
                        "This check covers saved inputs and box geometry, not intervention correctness or scientific validity."],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("asset_dir", type=Path)
    parser.add_argument("pil_dir", type=Path)
    parser.add_argument("--verify-https-mirror", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.asset_dir, args.pil_dir, args.verify_https_mirror), indent=2))
