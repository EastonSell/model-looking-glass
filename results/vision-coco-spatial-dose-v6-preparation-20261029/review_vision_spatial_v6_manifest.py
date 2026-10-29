"""Model-free v6 selected-image audit against pinned source and saved pixels.

This checks selected identities, annotations, masks, ranks and exact prior
separation. It does not independently replay the complete candidate universe
or attest execution/publication timing, origin, or pretraining independence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import PIL
from PIL import Image, ImageDraw


ARCHIVE_SHA = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
MEMBER_SHA = '610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d'
MEMBER = 'annotations/instances_train2017.json'
MANIFEST_SHA = '586e0abe9c06a2eb02ad74dc8bda38137349cf1d5953363b2a190ae59d2f0075'
REVISION = '394e08972193156ff700c1ecdce205a0b714d365'
PRIOR = ['8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207',
         '6fc5159a57cf07a91e93c8bc38f27d3f5925877083532ce92fec828c6d777bfb',
         '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb',
         'd2fa15c4287f15f78ff5f1a1cc1eab2551d73ee2ac876b5f7bb956f31b1d2931']
MODEL = {'config.json': '7e5aeb1add0a832153071a858bb18145cafb809e54e7fea4e5dc1183c82078f5',
         'preprocessor_config.json': '90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2',
         'model.safetensors': 'a9146f3ee7cf6de399190132627e4aef1aeddd13692f5b333f24629dd36c37b0'}


def require(condition, why):
    if not condition:
        raise ValueError(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def stream_array(zf, key):
    # Independent incremental JSON-array reader for the pinned COCO member.
    with io.TextIOWrapper(zf.open(MEMBER), encoding='utf-8') as stream:
        pattern = re.compile(r'"' + re.escape(key) + r'"\s*:\s*\[')
        buffer = ''
        while True:
            chunk = stream.read(1024 * 1024)
            require(bool(chunk), 'missing source array')
            buffer += chunk
            found = pattern.search(buffer)
            if found:
                pos = found.end()
                break
            buffer = buffer[-1024:]
        decoder = json.JSONDecoder()
        while True:
            if pos > 1024 * 1024:
                buffer, pos = buffer[pos:], 0
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos == len(buffer):
                buffer = stream.read(1024 * 1024)
                pos = 0
                require(bool(buffer), 'truncated source array')
                continue
            if buffer[pos] == ']':
                return
            if buffer[pos] == ',':
                pos += 1
                continue
            while True:
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    buffer, pos = buffer[pos:], 0
                    require(len(buffer) <= 16 * 1024 * 1024, 'oversized source row')
                    chunk = stream.read(1024 * 1024)
                    require(bool(chunk), 'truncated source row')
                    buffer += chunk
            require(type(obj) is dict, 'source row is not an object')
            yield obj
            pos = end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    require(digest(raw) == MANIFEST_SHA, 'manifest digest differs')
    m = json.loads(raw)
    require(raw == (json.dumps(m, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode(),
            'manifest is not canonical')
    require(m['schema'] == 'vision-coco-spatial-dose-cases-v6' and
            m['annotation_archive_sha256'] == ARCHIVE_SHA and m['annotation_sha256'] == MEMBER_SHA and
            m['selection_source']['revision'] == REVISION and
            m['model_repo'] == 'WinKawaks/vit-tiny-patch16-224' and
            m['model_revision'] == '77d1485af66b34d4ed0fe95dbb0c60c7496f950b' and
            m['model_files'] == MODEL, 'source declarations differ')
    require(PIL.__version__ == m['preparation_versions']['pillow'] == '11.3.0', 'Pillow version differs')
    names = ['vision_coco_spatial_v1.json', 'vision_coco_same_site_v3.json',
             'vision_coco_recognized_v4.json', 'vision_coco_cls_compare_v5.json']
    old_raw = [(args.repo / 'experiments' / n).read_bytes() for n in names]
    require([digest(b) for b in old_raw] == PRIOR, 'prior inventory pins differ')
    require([m['prior_v1_sha256'], m['prior_v3_sha256'], m['prior_v4_sha256'], m['prior_v5_sha256']] == PRIOR,
            'manifest prior pins differ')
    old = [json.loads(b) for b in old_raw]
    require(m['processor'] == old[0]['processor'] ==
            {'class': 'ViTImageProcessor', 'exif_transpose': False, 'mean': [.5, .5, .5],
             'resample': 'bilinear', 'rescale_factor': 1/255,
             'size': {'height': 224, 'width': 224}, 'std': [.5, .5, .5]},
            'processor declaration differs')
    prior_rows = {}
    for doc, roles in ((old[0], ('cat', 'dog')), (old[1], ('cat', 'dog', 'cat_donor', 'dog_donor'))):
        for pair in doc['pairs']:
            for role in roles:
                row = pair[role]
                triple = tuple(row[k] for k in ('image_id', 'jpeg_sha256', 'pixel_sha256'))
                require(triple[0] not in prior_rows or prior_rows[triple[0]] == triple, 'prior row conflict')
                prior_rows[triple[0]] = triple
    for doc in old[2:]:
        for row in doc['images']:
            triple = tuple(row[k] for k in ('image_id', 'jpeg_sha256', 'pixel_sha256'))
            require(triple[0] not in prior_rows, 'v4 or v5 image overlaps an earlier ID')
            prior_rows[triple[0]] = triple
    require(len(prior_rows) == 171, 'prior population differs')
    prior_columns = [set(row[i] for row in prior_rows.values()) for i in range(3)]
    require(all(len(column) == 171 for column in prior_columns), 'prior byte identities collide')
    rows = m['images']
    require(type(rows) is list and len(rows) == 64, 'candidate count differs')
    for category, group in ((17, rows[:32]), (18, rows[32:])):
        keys = []
        for rank, row in enumerate(group):
            require(type(row['image_id']) is int and row['category_id'] == category and row['rank'] == rank and
                    row['role'] == ('donor' if rank % 4 == 0 else 'target'), 'role or rank differs')
            expected = digest(f"vision-coco-spatial-dose-v6:{category}:{row['image_id']}".encode('ascii'))
            require(row['selection_rank_sha256'] == expected, 'rank hash differs')
            keys.append((expected, row['image_id']))
        require(keys == sorted(keys), 'rank order differs')
    for index, key in enumerate(('image_id', 'jpeg_sha256', 'pixel_sha256')):
        values = [row[key] for row in rows]
        require(len(set(values)) == 64 and not set(values) & prior_columns[index], 'fresh identity overlaps')
    for name, expected in MODEL.items():
        require(file_digest(args.assets / name) == expected, 'model/processor asset differs')
    config = json.loads((args.assets / 'config.json').read_bytes())
    groups = {name: [{'id': i, 'label': config['id2label'][str(i)]} for i in ids]
              for name, ids in (('cat', range(281, 286)), ('dog', range(151, 269)))}
    require(m['class_groups'] == groups, 'class-group labels differ')
    require(file_digest(args.archive) == ARCHIVE_SHA, 'annotation archive bytes differ')
    selected = {r['image_id']: r for r in rows}
    with zipfile.ZipFile(args.archive) as zf:
        info = zf.getinfo(MEMBER)
        require(info.file_size == 469785474, 'annotation size differs')
        h = hashlib.sha256()
        with zf.open(MEMBER) as file:
            while block := file.read(1024 * 1024):
                h.update(block)
        require(h.hexdigest() == MEMBER_SHA, 'annotation member bytes differ')
        licenses = {r['id']: r for r in stream_array(zf, 'licenses')}
        images = {r['id']: r for r in stream_array(zf, 'images') if r['id'] in selected}
        all_annotations = {ident: [] for ident in selected}
        for annotation in stream_array(zf, 'annotations'):
            if annotation['image_id'] in selected:
                all_annotations[annotation['image_id']].append(annotation)
    require(len(images) == 64, 'source images missing')
    total_sites = 0
    for row in rows:
        image = images[row['image_id']]
        anns = all_annotations[row['image_id']]
        animal = [a for a in anns if a['category_id'] in (17, 18)]
        require(animal and {a['category_id'] for a in animal} == {row['category_id']}, 'animal category differs')
        require(sorted(a['id'] for a in animal) == row['animal_annotation_ids'], 'animal annotation ids differ')
        valid = [a for a in animal if not a['iscrowd'] and type(a.get('segmentation')) is list
                 and a['segmentation'] and all(type(p) is list and len(p) >= 6 and len(p) % 2 == 0
                                              for p in a['segmentation'])]
        require(bool(valid), 'no valid polygon animal')
        ann = min(valid, key=lambda a: (-a['bbox'][2] * a['bbox'][3], a['id']))
        require(row['annotation_id'] == ann['id'] and row['bbox_xywh'] == ann['bbox'], 'instance differs')
        require((row['width'], row['height'], row['file_name'], row['license_id'], row['license']) ==
                (image['width'], image['height'], image['file_name'], image['license'], licenses[image['license']]),
                'image metadata differs')
        require(image['license'] in (4, 5, 7, 8), 'license filter differs')
        others = sorted([{'annotation_id': a['id'], 'category_id': a['category_id'], 'bbox_xywh': a['bbox']}
                         for a in anns if a['id'] != ann['id']], key=lambda x: x['annotation_id'])
        require(others == row['other_boxes'], 'other-object boxes differ')
        width, height = image['width'], image['height']
        x, y, bw, bh = ann['bbox']
        require(width >= 224 and height >= 224 and max(width / height, height / width) <= 2 and
                .08 <= bw * bh / (width * height) <= .70 and bw * 224 / width >= 32 and
                bh * 224 / height >= 32, 'animal or image geometry differs')
        box = [x * 224 / width, y * 224 / height, (x + bw) * 224 / width, (y + bh) * 224 / height]
        require(box == row['bbox_resized_xyxy'], 'resized box differs')
        mask = Image.new('L', (224, 224), 0)
        painter = ImageDraw.Draw(mask)
        for poly in ann['segmentation']:
            painter.polygon([(poly[i] * 224 / width, poly[i + 1] * 224 / height)
                             for i in range(0, len(poly), 2)], fill=1)
        require(digest(mask.tobytes()) == row['mask_sha256'], 'raster mask differs')
        mask_array = np.asarray(mask, dtype=np.uint8)
        sites = []
        for yidx, xidx in sorted(((a, b) for a in range(13) for b in range(13)),
                                 key=lambda p: ((p[0]-6)**2 + (p[1]-6)**2, p)):
            if not (16*xidx >= box[0] and 16*yidx >= box[1] and
                    16*(xidx+2) <= box[2] and 16*(yidx+2) <= box[3]):
                continue
            counts = [int(mask_array[16*(yidx+dy):16*(yidx+dy+1),
                                     16*(xidx+dx):16*(xidx+dx+1)].sum())
                      for dy in range(2) for dx in range(2)]
            if sum(counts) >= 512:
                sites.append({'row_col': [yidx, xidx], 'token_ids':
                              [1+14*(yidx+dy)+xidx+dx for dy in range(2) for dx in range(2)],
                              'squared_center_radius': (yidx-6)**2 + (xidx-6)**2,
                              'mask_pixels': sum(counts), 'token_mask_pixels': counts})
        require(sites and sites == row['sites'], 'qualified spatial sites or mask counts differ')
        total_sites += len(sites)
        filename = f"{row['image_id']:012d}.jpg"
        require(row['file_name'] == filename and row['mirror_url'] ==
                'https://s3.amazonaws.com/images.cocodataset.org/train2017/' + filename, 'image path differs')
        jpeg = (args.assets / 'images' / filename).read_bytes()
        require(0 < len(jpeg) <= 5_000_000 and len(jpeg) == row['jpeg_bytes'] and
                digest(jpeg) == row['jpeg_sha256'], 'JPEG differs')
        with Image.open(io.BytesIO(jpeg)) as pic:
            require(pic.size == (width, height) and pic.getexif().get(274, 1) == 1, 'image dimensions/EXIF differ')
            resized = pic.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            pixels = (np.asarray(resized).astype(np.float64) * (1 / 255)).astype(np.float32)
            pixels = (pixels - np.float32(.5)) / np.float32(.5)
            pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1).astype('<f4', copy=False))
        require(digest(pixels.tobytes()) == row['pixel_sha256'], 'independent processed pixels differ')
    report = {'status': 'selected-input-consistency-verified', 'manifest_sha256': MANIFEST_SHA,
              'source_revision': REVISION, 'prior_exposures': 171, 'selected_images': len(rows),
              'mask_qualified_sites': total_sites, 'candidate_counts': {'cat': 32, 'dog': 32},
              'model_forwards': 0, 'scope': 'selected rows only; complete rank population, upstream authenticity, '
              'pretraining exposure, execution and chronology not attested'}
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
