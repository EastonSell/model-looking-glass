"""Deterministic, annotation-only candidate inventory for the v4 screen.

No checkpoint, image pixels, or model output is used in this module.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import zipfile
from pathlib import Path

from .vision_coco_cases import _radius, _squares, _token_ids

ARCHIVE_SIZE = 252907541
ARCHIVE_SHA256 = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
ANNOTATION_MEMBER = 'annotations/instances_train2017.json'
ANNOTATION_SIZE = 469785474
ANNOTATION_SHA256 = '610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d'
PRIOR_V1_SHA256 = '8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207'
PRIOR_V3_SHA256 = '6fc5159a57cf07a91e93c8bc38f27d3f5925877083532ce92fec828c6d777bfb'


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_expected_image(path, size, digest):
    """Return the exact bounded JPEG bytes recorded at download time."""
    if type(size) is not int or not 0 < size <= 5_000_000:
        raise ValueError('image byte count differs')
    with Path(path).open('rb') as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('stored JPEG bytes differ')
    return raw


def _digest_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def stream_array(member, field):
    """Stream a top-level COCO array without loading the 470 MB JSON into RAM.

    The production caller first authenticates the entire pinned member. This
    parser is bounded per item and is not a general-purpose JSON validator.
    """
    if field not in {'images', 'annotations', 'categories', 'licenses'}:
        raise ValueError('unexpected COCO array')
    decoder, pattern = json.JSONDecoder(), re.compile(r'"' + field + r'"\s*:\s*\[')
    with io.TextIOWrapper(member, encoding='utf-8', newline='') as stream:
        buf, pos, eof = '', 0, False
        while True:
            found = pattern.search(buf)
            if found:
                pos = found.end()
                break
            block = stream.read(1024 * 1024)
            if not block:
                raise ValueError('missing COCO array')
            buf = (buf + block)[-2 * 1024 * 1024:]
        while True:
            if pos > 1024 * 1024:
                buf, pos = buf[pos:], 0
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    break
                block = stream.read(1024 * 1024)
                if not block:
                    raise ValueError('truncated COCO array')
                buf, pos = buf[pos:] + block, 0
            if buf[pos] == ']':
                return
            if buf[pos] == ',':
                pos += 1
                continue
            while True:
                try:
                    value, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    if len(buf) - pos > 16 * 1024 * 1024:
                        raise ValueError('oversized COCO item')
                    block = stream.read(1024 * 1024)
                    if not block:
                        raise ValueError('truncated or malformed COCO item')
                    buf = buf[pos:] + block
                    pos = 0
            if type(value) is not dict:
                raise ValueError('COCO array item must be an object')
            yield value
            pos = end


def verify_annotation_archive(stream):
    if os.fstat(stream.fileno()).st_size != ARCHIVE_SIZE or _digest_stream(stream) != ARCHIVE_SHA256:
        raise ValueError('annotation archive differs from the fixed bytes')
    with zipfile.ZipFile(stream) as archive:
        info = archive.getinfo(ANNOTATION_MEMBER)
        if info.file_size != ANNOTATION_SIZE:
            raise ValueError('annotation member length differs')
        digest = hashlib.sha256()
        with archive.open(info) as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != ANNOTATION_SHA256:
            raise ValueError('annotation member bytes differ')


def previous_images(v1_raw, v3_raw):
    if (hashlib.sha256(v1_raw).hexdigest() != PRIOR_V1_SHA256 or
            hashlib.sha256(v3_raw).hexdigest() != PRIOR_V3_SHA256):
        raise ValueError('previous exposure manifests differ')
    v1, v3 = json.loads(v1_raw), json.loads(v3_raw)
    images = {}
    for manifest, roles in ((v1, ('cat', 'dog')),
                            (v3, ('cat', 'dog', 'cat_donor', 'dog_donor'))):
        for pair in manifest['pairs']:
            for role in roles:
                row = pair[role]
                values = {key: row[key] for key in ('image_id', 'jpeg_sha256', 'pixel_sha256')}
                if row['image_id'] in images and images[row['image_id']] != values:
                    raise ValueError('conflicting earlier image identity')
                images[row['image_id']] = values
    if len(images) != 43:
        raise ValueError('earlier exposure image count differs')
    return list(images.values())


def rank(category, image_id):
    return (hashlib.sha256(f'vision-coco-recognized-v4:{category}:{image_id}'.encode('ascii')).hexdigest(),
            image_id)


def _candidate(image, annotations, *, ranker=rank):
    import PIL
    from PIL import Image, ImageDraw
    if PIL.__version__ != '11.3.0':
        raise ValueError('selection requires Pillow 11.3.0')
    if len({a['category_id'] for a in annotations}) != 1:
        return None
    valid = []
    for ann in annotations:
        polys = ann.get('segmentation')
        if (ann['iscrowd'] or type(polys) is not list or not polys or
                any(type(poly) is not list or len(poly) < 6 or len(poly) % 2 for poly in polys)):
            continue
        if not all(math.isfinite(float(v)) for v in ann['bbox']):
            raise ValueError('nonfinite animal box')
        valid.append(ann)
    if not valid:
        return None
    # Choose first, then filter geometry: never fall back to a smaller instance.
    ann = min(valid, key=lambda a: (-a['bbox'][2] * a['bbox'][3], a['id']))
    _, _, width, height = ann['bbox']
    iw, ih = image['width'], image['height']
    if (iw < 224 or ih < 224 or max(iw / ih, ih / iw) > 2 or
            not .08 <= width * height / (iw * ih) <= .70 or
            width * 224 / iw < 32 or height * 224 / ih < 32):
        return None
    inside, _, box = _squares(image, ann)
    if not inside:
        return None
    mask = Image.new('L', (224, 224), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in ann['segmentation']:
        if not all(math.isfinite(float(value)) for value in polygon):
            raise ValueError('nonfinite animal polygon')
        draw.polygon([(polygon[i] * 224 / iw, polygon[i + 1] * 224 / ih)
                      for i in range(0, len(polygon), 2)], fill=1)
    sites = []
    for row, col in sorted(inside, key=lambda p: (_radius(p), p)):
        counts = [sum(mask.crop((16 * (col + dc), 16 * (row + dr),
                                 16 * (col + dc + 1), 16 * (row + dr + 1))).getdata())
                  for dr in range(2) for dc in range(2)]
        if sum(counts) >= 512:
            sites.append({'row_col': [row, col], 'token_ids': _token_ids((row, col)),
                          'squared_center_radius': _radius((row, col)),
                          'mask_pixels': sum(counts), 'token_mask_pixels': counts})
    if not sites:
        return None
    return {'image_id': image['id'], 'category_id': ann['category_id'],
            'file_name': image['file_name'], 'width': iw, 'height': ih,
            'license_id': image['license'], 'annotation_id': ann['id'],
            'animal_annotation_ids': sorted(a['id'] for a in annotations),
            'bbox_xywh': ann['bbox'], 'bbox_resized_xyxy': list(box),
            'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(), 'sites': sites,
            'selection_rank_sha256': ranker(ann['category_id'], image['id'])[0]}


def _select_inventory(archive, excluded, count, *, ranker=rank):
    def rows(field):
        return stream_array(archive.open(ANNOTATION_MEMBER), field)
    categories = {r['name']: r['id'] for r in rows('categories')}
    if categories.get('cat') != 17 or categories.get('dog') != 18:
        raise ValueError('cat/dog category identities differ')
    licenses = {r['id']: r for r in rows('licenses')}
    images = {r['id']: r for r in rows('images') if
                  r['id'] not in excluded and r['license'] in (4, 5, 7, 8) and
              r['width'] >= 224 and r['height'] >= 224 and
              max(r['width'] / r['height'], r['height'] / r['width']) <= 2}
    animal = {}
    for ann in rows('annotations'):
        if ann['category_id'] in (17, 18) and ann['image_id'] in images:
            animal.setdefault(ann['image_id'], []).append(ann)
    groups = {17: [], 18: []}
    ranked = sorted(animal, key=lambda image_id: ranker(animal[image_id][0]['category_id'], image_id))
    for image_id in ranked:
        anns = animal[image_id]
        if len({a['category_id'] for a in anns}) != 1:
            continue
        category = anns[0]['category_id']
        if len(groups[category]) == count:
            continue
        result = _candidate(images[image_id], anns, ranker=ranker)
        if result:
            result['rank'] = len(groups[category])
            result['role'] = 'donor' if result['rank'] % 4 == 0 else 'target'
            result['license'] = licenses[images[image_id]['license']]
            groups[category].append(result)
    if any(len(group) != count for group in groups.values()):
        raise ValueError('insufficient annotation-qualified candidate inventory')
    selected = groups[17] + groups[18]
    indexed = {row['image_id']: row for row in selected}
    for row in selected:
        row['other_boxes'] = []
    for ann in rows('annotations'):
        row = indexed.get(ann['image_id'])
        if row is not None and ann['id'] != row['annotation_id']:
            row['other_boxes'].append({'annotation_id': ann['id'], 'category_id': ann['category_id'],
                                       'bbox_xywh': ann['bbox']})
    for row in selected:
        row['other_boxes'].sort(key=lambda other: other['annotation_id'])
    return selected

def select_inventory(archive_path, excluded_ids, *, count=32, authenticated=True, ranker=None):
    """Use one opened file throughout; ``authenticated=False`` is for toy tests.

    The source custodian must keep the opened file and path quiescent. The
    identity and before/after hashes detect ordinary replacement or mutation,
    but cannot establish that a hostile writer never changed and restored bytes.
    """
    if type(count) is not int or count < 1:
        raise ValueError('invalid candidate count')
    path = Path(archive_path)
    with path.open('rb') as source:
        initial = os.fstat(source.fileno())
        first_hash = _digest_stream(source)
        if authenticated:
            verify_annotation_archive(source)
        with zipfile.ZipFile(source) as archive:
            if ranker is None:
                selected = _select_inventory(archive, set(excluded_ids), count)
            else:
                selected = _select_inventory(archive, set(excluded_ids), count, ranker=ranker)
        last_hash = _digest_stream(source)
        final = os.fstat(source.fileno())
        present = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size)
        if identity(initial) != identity(final) or identity(initial) != identity(present) or first_hash != last_hash:
            raise ValueError('annotation archive changed during selection')
        return selected
