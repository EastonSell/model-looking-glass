"""Prepare four already-exposed pixels and saved CPU logits, with no model forward."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

MIB = 1024 ** 2
MANIFEST_SHA = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'
SCREEN_SHA = '711894680eac6bed8e60b8acb4413775cce51f5630115e561ecbfe0504efa648'
ARCHIVE_SHA = 'fdd48436398a9705e39b15fa410599a0d806e562e1770e54cdecd4c0babb9e81'
ROLES = {'cat': 405058, 'dog': 196065, 'cat_donor': 69231, 'dog_donor': 142769}
INDICES = {405058: 2, 196065: 35, 69231: 9, 142769: 37}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bounded(path, limit):
    with Path(path).open('rb') as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError('reference input exceeds its bound')
    return value


def write_new(path, data):
    with Path(path).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short reference write')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(manifest_path, screen_path, archive_path, assets, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('reference preparation requires an empty CUDA mask')
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoImageProcessor

    if (torch.cuda.is_initialized() or sys.byteorder != 'little' or np.__version__ != '2.4.6'
            or PIL.__version__ != '11.3.0' or torch.__version__ != '2.8.0+cu128'
            or transformers.__version__ != '4.57.3'):
        raise ValueError('CPU reference runtime differs')
    manifest_raw = read_bounded(manifest_path, 2 * MIB)
    screen_raw = read_bounded(screen_path, 2 * MIB)
    archive_raw = read_bounded(archive_path, 2 * MIB)
    if (sha(manifest_raw), sha(screen_raw), sha(archive_raw)) != (MANIFEST_SHA, SCREEN_SHA, ARCHIVE_SHA):
        raise ValueError('public reference source differs')
    manifest, screen = json.loads(manifest_raw), json.loads(screen_raw)
    if screen.get('manifest_sha256') != MANIFEST_SHA or len(screen.get('pairs', [])) < 1:
        raise ValueError('screen source binding differs')
    pair = screen['pairs'][0]
    if (pair.get('pair_id') != 'pair-00' or pair.get('box_tokens') != [90, 91, 104, 105]
            or any(pair.get(role, {}).get('image_id') != image_id for role, image_id in ROLES.items())):
        raise ValueError('fixed public pair differs')
    images = {item['image_id']: item for item in manifest['images'] if item['image_id'] in ROLES.values()}
    if set(images) != set(ROLES.values()):
        raise ValueError('public image inventory differs')
    assets = Path(assets)
    if set(manifest['model_files']) != {'config.json', 'preprocessor_config.json', 'model.safetensors'}:
        raise ValueError('model asset inventory differs')
    for name, digest in manifest['model_files'].items():
        if sha(read_bounded(assets / name, 32 * MIB)) != digest:
            raise ValueError('model asset differs')
    processor = AutoImageProcessor.from_pretrained(str(assets), local_files_only=True, use_fast=False)
    if (type(processor).__name__ != 'ViTImageProcessor' or processor.size != {'height': 224, 'width': 224}
            or int(processor.resample) != 2 or processor.image_mean != [.5] * 3
            or processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
        raise ValueError('processor differs')
    files = {}
    screen_rows = {item['image_id']: item for item in screen['screen'] if item['image_id'] in images}
    if set(screen_rows) != set(images):
        raise ValueError('saved CPU screen inventory differs')
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate archive entries')
        for role, image_id in ROLES.items():
            info = images[image_id]
            raw = read_bounded(assets / 'images' / info['file_name'], 4 * MIB)
            if len(raw) != info['jpeg_bytes'] or sha(raw) != info['jpeg_sha256']:
                raise ValueError('exposed image differs')
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (info['width'], info['height']) or image.getexif().get(274, 1) != 1:
                    raise ValueError('image geometry differs')
                pixels = processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
            if (type(pixels) is not torch.Tensor or tuple(pixels.shape) != (1, 3, 224, 224)
                    or pixels.device.type != 'cpu' or pixels.dtype != torch.float32
                    or not bool(torch.isfinite(pixels).all())):
                raise ValueError('processed pixels differ')
            pixel_bytes = pixels.contiguous().numpy().astype('<f4', copy=False).tobytes()
            if len(pixel_bytes) != 602112 or sha(pixel_bytes) != info['pixel_sha256']:
                raise ValueError('processed pixel hash differs')
            files[f'{role}-pixels.f32'] = pixel_bytes
            index = INDICES[image_id]
            row = json.loads(archive.read(f'forward-{index:03d}.json'))
            logits = archive.read(f'forward-{index:03d}.bin')
            if (len(logits) != 4000 or row.get('label') != f'screen:{image_id}'
                    or row.get('file') != f'forward-{index:03d}.bin'
                    or row.get('layout') != {'logits': {'bytes': 4000, 'offset': 0, 'shape': [1, 1000]}}
                    or row.get('sha256') != sha(logits)
                    or sha(logits) != screen_rows[image_id].get('logit_sha256')):
                raise ValueError('saved CPU screen logits differ')
            files[f'{role}-cpu-logits.f32'] = logits
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized during CPU reference preparation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for name, value in sorted(files.items()):
        write_new(output / name, value)
        inventory[name] = {'bytes': len(value), 'sha256': sha(value)}
    small_pair = {'pair_id': 'pair-00', 'split': pair['split'], 'box_tokens': [90, 91, 104, 105]}
    small_pair.update({role: {'image_id': image_id} for role, image_id in ROLES.items()})
    reference = {'schema': 'vision-cls-exposed-gpu-reference-v1', 'status': 'complete',
                 'model_forwards': 0, 'pair': small_pair, 'manifest_sha256': MANIFEST_SHA,
                 'screen_sha256': SCREEN_SHA, 'screen_evidence_sha256': ARCHIVE_SHA,
                 'model_files': manifest['model_files'], 'files': inventory}
    raw = (json.dumps(reference, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    write_new(output / 'reference.json', raw)
    return sha(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--screen', required=True, type=Path)
    parser.add_argument('--screen-evidence', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    digest = prepare(args.manifest, args.screen, args.screen_evidence, args.assets, args.output)
    print(json.dumps({'status': 'complete', 'model_forwards': 0, 'reference_sha256': digest}))


if __name__ == '__main__':
    main()
