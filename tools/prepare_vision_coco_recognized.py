"""Freeze v4 screen images from pinned annotations; never load a model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or
        not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute() or
        not Path(sys.pycache_prefix).is_dir() or any(Path(sys.pycache_prefix).iterdir())):
    raise ValueError('preparation requires an externally created empty private bytecode prefix')
spec = importlib.util.find_spec('looking_glass')
if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / 'looking_glass/__init__.py').resolve():
    raise ValueError('project package is shadowed')

from looking_glass.vision_coco_cases import MODEL_FILES, MODEL_REPO, MODEL_REVISION, verify_assets
from looking_glass.vision_coco_recognized_cases import (ANNOTATION_SHA256, ARCHIVE_SHA256,
                                                       PRIOR_V1_SHA256, PRIOR_V3_SHA256,
                                                       previous_images, read_expected_image, select_inventory)
from looking_glass.vision_coco_small_dose_cases import check_image_hashes

SOURCES = ('docs/VISION_COCO_RECOGNIZED_SCREEN_V4.md', 'tools/prepare_vision_coco_recognized.py',
           'looking_glass/__init__.py', 'looking_glass/vision_coco_recognized_cases.py',
           'looking_glass/vision_coco_cases.py', 'looking_glass/vision_coco_small_dose_cases.py',
           'experiments/vision_coco_spatial_v1.json', 'experiments/vision_coco_same_site_v3.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('selection revision must be a full lowercase SHA')
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
    def git(*args):
        return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false',
                                        *args], cwd=ROOT, env=env)
    if git('rev-parse', 'HEAD').decode().strip() != revision:
        raise ValueError('checkout differs from selection revision')
    sources = {}
    for path in SOURCES:
        raw = (ROOT / path).read_bytes()
        if raw != git('show', f'{revision}:{path}'):
            raise ValueError(f'selection source differs: {path}')
        sources[path] = sha(raw)
    if (sources['experiments/vision_coco_spatial_v1.json'] != PRIOR_V1_SHA256 or
            sources['experiments/vision_coco_same_site_v3.json'] != PRIOR_V3_SHA256):
        raise ValueError('previous case inventory differs')
    return {'revision': revision, 'sources': sources,
            'scope': 'trusted quiescent checkout and interpreter; not execution attestation'}


def prepare(assets, annotation_archive, output, revision):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden for image preparation')
    if output.exists():
        raise ValueError('manifest output already exists')
    source = snapshot(revision)
    prior_v1 = (ROOT / 'experiments/vision_coco_spatial_v1.json').read_bytes()
    prior_v3 = (ROOT / 'experiments/vision_coco_same_site_v3.json').read_bytes()
    previous = previous_images(prior_v1, prior_v3)
    images = select_inventory(annotation_archive, [row['image_id'] for row in previous])
    verify_assets(assets)
    prior = json.loads(prior_v1)
    config = json.loads((assets / 'config.json').read_bytes())
    for group in prior['class_groups'].values():
        if any(config['id2label'][str(row['id'])] != row['label'] for row in group):
            raise ValueError('class-group labels differ')
    image_dir = assets / 'images'
    image_dir.mkdir(exist_ok=True)

    def download(item):
        expected_name = f"{item['image_id']:012d}.jpg"
        if item['file_name'] != expected_name:
            raise ValueError('image name differs from its ID')
        item['mirror_url'] = 'https://s3.amazonaws.com/images.cocodataset.org/train2017/' + expected_name
        request = urllib.request.Request(item['mirror_url'], headers={'User-Agent': 'model-looking-glass-vision/4.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.geturl() != item['mirror_url']:
                raise ValueError('image mirror redirected')
            raw = response.read(5_000_001)
        if not raw or len(raw) > 5_000_000:
            raise ValueError('empty or oversized image')
        path = image_dir / expected_name
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('existing image differs from the mirror')
        else:
            with path.open('xb') as stream:
                if stream.write(raw) != len(raw):
                    raise OSError('short image write')
                stream.flush()
                os.fsync(stream.fileno())
        item['jpeg_bytes'], item['jpeg_sha256'] = len(raw), sha(raw)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, images))
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoImageProcessor
    if (torch.cuda.is_initialized() or PIL.__version__ != '11.3.0' or np.__version__ != '2.4.6' or
            torch.__version__ != '2.8.0+cu128' or transformers.__version__ != '4.57.3'):
        raise ValueError('preparation runtime differs')
    processor = AutoImageProcessor.from_pretrained(str(assets), local_files_only=True, use_fast=False)
    if (type(processor).__name__ != 'ViTImageProcessor' or processor.size != {'height': 224, 'width': 224}
            or int(processor.resample) != 2 or processor.image_mean != [.5] * 3
            or processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
        raise ValueError('processor differs')
    for item in images:
        path = image_dir / item['file_name']
        raw = read_expected_image(path, item['jpeg_bytes'], item['jpeg_sha256'])
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (item['width'], item['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('image dimensions or orientation differ')
            value = processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        if (tuple(value.shape) != (1, 3, 224, 224) or value.dtype != torch.float32 or
                not bool(torch.isfinite(value).all())):
            raise ValueError('invalid processed pixels')
        item['pixel_sha256'] = sha(value.contiguous().numpy().astype('<f4', copy=False).tobytes())
        if read_expected_image(path, item['jpeg_bytes'], item['jpeg_sha256']) != raw:
            raise ValueError('stored JPEG changed during processing')
    verify_assets(assets)
    check_image_hashes(previous, images)
    if snapshot(revision) != source:
        raise ValueError('selection source changed during preparation')
    manifest = {'schema': 'vision-coco-recognized-screen-cases-v4',
                'annotation_archive_sha256': ARCHIVE_SHA256, 'annotation_sha256': ANNOTATION_SHA256,
                'prior_v1_sha256': PRIOR_V1_SHA256, 'prior_v3_sha256': PRIOR_V3_SHA256,
                'selection_source': source, 'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                'model_files': MODEL_FILES, 'class_groups': prior['class_groups'],
                'processor': prior['processor'], 'images': images,
                'preparation_versions': {'python': sys.version.split()[0], 'pillow': PIL.__version__,
                                         'numpy': np.__version__, 'torch': torch.__version__,
                                         'transformers': transformers.__version__}}
    raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short manifest write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'manifest_sha256': sha(raw), 'images': len(images), 'model_forwards': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--annotation-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-revision', required=True)
    args = parser.parse_args()
    prepare(args.assets, args.annotation_archive, args.output, args.selection_revision)


if __name__ == '__main__':
    main()
