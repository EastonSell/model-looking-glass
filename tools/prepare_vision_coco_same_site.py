"""Freeze v3 same-site cases from annotations and image bytes; no model forward."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
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
    raise ValueError('selection requires an externally created empty private bytecode prefix and isolated startup')
spec = importlib.util.find_spec('looking_glass')
if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / 'looking_glass/__init__.py').resolve():
    raise ValueError('project package is shadowed by another installation')

from looking_glass.vision_coco_cases import (ANNOTATION_SHA256, ANNOTATION_URL, MODEL_FILES,
                                            MODEL_REPO, MODEL_REVISION, verify_assets)
from looking_glass.vision_coco_same_site_cases import (PRIOR_CASES_SHA256, check_reused_images,
                                                       select_same_site)

SOURCES = ('docs/VISION_COCO_SMALL_DOSE_V2.md', 'docs/VISION_COCO_SAME_SITE_SMALL_DOSE_V3.md',
           'tools/prepare_vision_coco_same_site.py', 'looking_glass/__init__.py',
           'looking_glass/vision_coco_same_site_cases.py', 'looking_glass/vision_coco_small_dose_cases.py',
           'looking_glass/vision_coco_cases.py',
           'experiments/vision_coco_spatial_v1.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_snapshot(revision):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('selection source revision must be a full lowercase SHA')
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_REPLACE_OBJECTS='1',
               GIT_TERMINAL_PROMPT='0')
    def git(*arguments):
        return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false',
                                        *arguments], cwd=ROOT, env=env)
    if git('rev-parse', 'HEAD').decode().strip() != revision:
        raise ValueError('selection checkout is not at the requested revision')
    sources = {}
    for path in SOURCES:
        raw = (ROOT / path).read_bytes()
        if raw != git('show', f'{revision}:{path}'):
            raise ValueError(f'selection source differs from committed bytes: {path}')
        sources[path] = sha(raw)
    if sources['experiments/vision_coco_spatial_v1.json'] != PRIOR_CASES_SHA256:
        raise ValueError('v1 exposure inventory differs')
    return {'revision': revision, 'source_sha256': sources,
            'scope': 'trusted quiescent checkout and interpreter; not execution attestation'}


def prepare(assets, output, selection_revision):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden during image-only preparation')
    snapshot = source_snapshot(selection_revision)
    prior_raw = (ROOT / 'experiments/vision_coco_spatial_v1.json').read_bytes()
    prior = json.loads(prior_raw)
    prior_images = [p[s] for p in prior['pairs'] for s in ('cat', 'dog')]
    annotation = (assets / 'instances_val2017.json').read_bytes()
    pairs, diagnostics = select_same_site(annotation, prior_raw)
    verify_assets(assets)
    config = json.loads((assets / 'config.json').read_bytes())
    groups = prior['class_groups']
    for group in groups.values():
        if any(config['id2label'][str(row['id'])] != row['label'] for row in group):
            raise ValueError('pinned class groups differ')
    image_dir = assets / 'images'
    image_dir.mkdir(exist_ok=True)
    references = [pair[role] for pair in pairs for role in ('cat', 'dog', 'cat_donor', 'dog_donor')]
    target_ids = [pair[role]['image_id'] for pair in pairs for role in ('cat', 'dog')]
    donor_ids = [pair[role]['image_id'] for pair in pairs for role in ('cat_donor', 'dog_donor')]
    if len(target_ids) != len(set(target_ids)) or len(target_ids) != 16 or set(target_ids) & set(donor_ids):
        raise ValueError('selection target/donor identities differ')
    unique = {}
    stable = ('file_name', 'width', 'height', 'coco_url', 'flickr_url', 'license', 'annotation_id',
              'animal_annotation_ids', 'category_id', 'bbox_xywh', 'bbox_resized_xyxy', 'mask_sha256',
              'selection_rank_sha256')
    for row in references:
        previous = unique.get(row['image_id'])
        if previous is not None and any(previous[key] != row[key] for key in stable):
            raise ValueError('reused donor metadata differs')
        unique[row['image_id']] = row
    images = list(unique.values())

    def download(image):
        path = image_dir / image['file_name']
        image['mirror_url'] = ('https://huggingface.co/datasets/merve/coco/resolve/'
                               '9e50abcdc1361852f34841af4939cbcd2d37c92f/val2017/' + image['file_name'])
        request = urllib.request.Request(image['mirror_url'],
                                         headers={'User-Agent': 'model-looking-glass-vision-study/3.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(5_000_001)
        if not raw or len(raw) > 5_000_000:
            raise ValueError('empty or oversized image')
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('existing image differs from mirror')
        else:
            with path.open('xb') as stream:
                if stream.write(raw) != len(raw):
                    raise OSError('short image write')
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, images))
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoImageProcessor
    if torch.cuda.is_initialized():
        raise ValueError('image preparation initialized CUDA')
    if (PIL.__version__ != '11.3.0' or np.__version__ != '2.4.6' or
            torch.__version__ != '2.8.0+cu128' or transformers.__version__ != '4.57.3'):
        raise ValueError('image preparation versions differ from v1')
    processor = AutoImageProcessor.from_pretrained(str(assets), local_files_only=True, use_fast=False)
    if (type(processor).__name__ != 'ViTImageProcessor' or processor.size != {'height': 224, 'width': 224}
            or int(processor.resample) != 2 or processor.image_mean != [.5] * 3
            or processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
        raise ValueError('image processor differs')
    for item in images:
        path = image_dir / item['file_name']
        raw = path.read_bytes()
        with Image.open(path) as image:
            if image.size != (item['width'], item['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('image dimensions or EXIF orientation differ')
            pixels = processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        if (tuple(pixels.shape) != (1, 3, 224, 224) or pixels.dtype != torch.float32 or
                not bool(torch.isfinite(pixels).all())):
            raise ValueError('invalid processed pixels')
        item['jpeg_bytes'] = len(raw)
        item['jpeg_sha256'] = sha(raw)
        item['pixel_sha256'] = sha(pixels.contiguous().numpy().astype('<f4', copy=False).tobytes())
    for row in references:
        observed = unique[row['image_id']]
        for key in ('mirror_url', 'jpeg_bytes', 'jpeg_sha256', 'pixel_sha256'):
            row[key] = observed[key]
    check_reused_images(prior_images, references, target_ids, donor_ids)
    if source_snapshot(selection_revision) != snapshot:
        raise ValueError('selection source changed during preparation')
    manifest = {'schema': 'vision-coco-vit-same-site-cases-v3',
                'annotation_url': ANNOTATION_URL, 'annotation_sha256': ANNOTATION_SHA256,
                'prior_manifest_sha256': PRIOR_CASES_SHA256, 'selection_source': snapshot,
                'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'model_files': MODEL_FILES,
                'class_groups': groups, 'processor': prior['processor'],
                'preparation_versions': {'python': sys.version.split()[0], 'pillow': PIL.__version__,
                                         'numpy': np.__version__, 'torch': torch.__version__,
                                         'transformers': transformers.__version__},
                'selection_diagnostics': diagnostics, 'pairs': pairs}
    raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short manifest write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'manifest_sha256': sha(raw), 'selected_pairs': len(pairs),
                      'selected_unique_images': len(images), 'donor_references': len(donor_ids),
                      'model_forwards': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-revision', required=True)
    args = parser.parse_args()
    prepare(args.assets, args.output, args.selection_revision)


if __name__ == '__main__':
    main()
