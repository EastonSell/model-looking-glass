"""Run the fixed fresh-image response screen on CPU after public discovery."""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('screen owner requires isolated no-site no-bytecode startup')
if __name__ == '__main__':
    if sys.platform == 'win32':
        import nt as _prefix_os
        _prefix = sys.pycache_prefix
        _absolute = (_prefix.startswith('\\\\') or
                     len(_prefix) >= 3 and _prefix[1] == ':' and _prefix[2] in ('/', '\\'))
    else:
        import posix as _prefix_os
        _prefix = sys.pycache_prefix
        _absolute = _prefix.startswith('/')
    try:
        if not _absolute or _prefix_os.listdir(_prefix):
            raise RuntimeError('screen owner requires an absolute empty bytecode prefix')
    except OSError as error:
        raise RuntimeError('screen owner requires an absolute empty bytecode prefix') from error

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'experiments/vision_coco_cls_occlusion_v1.json'
MANIFEST_SHA = 'a52040786557fde9e503b6b6d1246bee0cbd41106b3cd110ea6788ab13a53487'
RESULT = 'results/vision-coco-cls-occlusion-v1-20261029'
DISCOVERY_COMMIT = '656929d7edceed3be964bd432131b351e0cd60d4'
HANDOFF = tuple(RESULT + '/' + name for name in ('discovery-evidence-index.json', 'discovery-freeze.json',
                'discovery-observer.json', 'discovery-independent-audit.json', 'discovery-public-readback.json'))
EXTRA_SOURCES = HANDOFF + ('looking_glass/vision_coco_cls_occlusion_screen.py',
                 'looking_glass/vision_coco_cls_occlusion_discovery_handoff.py',
                 'tools/run_vision_coco_cls_occlusion_screen.py',
                 'tools/observe_vision_coco_cls_occlusion_screen.py')


def run_clean_screen(*args, **kwargs):
    from looking_glass.vision_coco_occlusion_study import run_screen
    return run_screen(*args, **kwargs)


def run_response_masks(*args, **kwargs):
    from looking_glass.vision_coco_cls_occlusion_study import run_response_masks as implementation
    return implementation(*args, **kwargs)


def select_response_screen(*args, **kwargs):
    from looking_glass.vision_coco_cls_occlusion_study import select_response_screen as implementation
    return implementation(*args, **kwargs)


def run_fresh_screen(model, images, pixels, evidence, manifest_sha256, *, numeric_guard=None):
    observed, _unused, _counts = run_clean_screen(model, images, pixels, evidence, numeric_guard=numeric_guard)
    clean_logits = {}
    for index, (image, row) in enumerate(zip(images, observed), 1):
        raw = (evidence.root / f'forward-{index:03d}.bin').read_bytes()
        if (len(raw) != 4000 or image['image_id'] != row['image_id'] or
                hashlib.sha256(raw).hexdigest() != row['logit_sha256']):
            raise ValueError('clean screen bytes differ')
        clean_logits[image['image_id']] = raw
    masked = run_response_masks(model, images, observed, pixels, clean_logits, evidence, numeric_guard=numeric_guard)
    selected, counts, rows = select_response_screen(images, observed, masked)
    result = {'schema': 'vision-cls-occlusion-v1-screen-result', 'manifest_sha256': manifest_sha256,
              'screen': observed, 'masked': {str(key): value for key, value in masked.items()},
              'selected_ids': [row['image_id'] for row in selected], 'counts': counts, 'rows': rows,
              'decision': 'inconclusive' if not selected else 'pending'}
    return {'sha256': evidence.json('screen-result.json', result), 'result': result}


def load_fresh_inputs(assets, image_dir, images):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    from looking_glass.vision_coco_cases import verify_assets
    from looking_glass.vision_coco_recognized_cases import read_expected_image
    verify_assets(assets)
    processor = AutoImageProcessor.from_pretrained(str(assets), local_files_only=True, use_fast=False)
    if (type(processor).__name__ != 'ViTImageProcessor' or processor.size != {'height': 224, 'width': 224} or
            int(processor.resample) != 2 or processor.image_mean != [.5] * 3 or
            processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
        raise ValueError('processor differs')
    output = {}
    for image in images:
        path = image_dir / image['file_name']
        raw = read_expected_image(path, image['jpeg_bytes'], image['jpeg_sha256'])
        with Image.open(io.BytesIO(raw)) as decoded:
            if decoded.size != (image['width'], image['height']) or decoded.getexif().get(274, 1) != 1:
                raise ValueError('image geometry differs')
            value = processor(images=decoded.convert('RGB'), return_tensors='pt')['pixel_values']
        if (tuple(value.shape) != (1, 3, 224, 224) or value.dtype != torch.float32 or
                not bool(torch.isfinite(value).all()) or
                hashlib.sha256(value.contiguous().numpy().astype('<f4', copy=False).tobytes()).hexdigest() !=
                image['pixel_sha256'] or read_expected_image(path, image['jpeg_bytes'], image['jpeg_sha256']) != raw):
            raise ValueError('processed input or stored JPEG differs')
        output[image['image_id']] = value
    if len(output) != len(images):
        raise ValueError('duplicate fresh input identity')
    verify_assets(assets)
    return output


def _git(*parts):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', *parts],
                                   cwd=ROOT, env=env)


def _load_verified_module(revision, path, name):
    target = ROOT / path
    if target.read_bytes() != _git('show', f'{revision}:{path}'):
        raise ValueError('committed helper differs: ' + path)
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    paths = (args.assets, args.image_dir, args.discovery_data, args.discovery_archive, args.output)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or
            any(not path.is_absolute() for path in paths) or not args.assets.is_dir() or not args.image_dir.is_dir() or
            not args.discovery_data.is_dir() or not args.discovery_archive.is_file() or
            not args.output.parent.is_dir() or args.output.exists()):
        raise ValueError('screen environment or input/output paths differ')
    def overlaps(first, second):
        first, second = first.resolve(), second.resolve()
        return first.is_relative_to(second) or second.is_relative_to(first)
    if any(overlaps(args.output, item) for item in (ROOT, args.assets, args.image_dir, args.discovery_data)):
        raise ValueError('screen source, inputs and output must be separate')
    if shutil.disk_usage(ROOT).free < 3 * 1024**3 // 2 or shutil.disk_usage(args.output.parent).free < 1024**3:
        raise ValueError('insufficient source/cache or evidence space')
    if _git('rev-parse', 'HEAD').decode().strip() != args.revision:
        raise ValueError('screen source revision differs')
    spec = importlib.util.find_spec('looking_glass')
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / 'looking_glass/__init__.py').resolve():
        raise ValueError('project package is shadowed')
    discovery_owner = _load_verified_module(args.revision, 'tools/run_vision_coco_cls_occlusion_discovery.py',
                                            '_cls_occlusion_discovery_owner')
    source = discovery_owner.snapshot(args.revision)
    for path in EXTRA_SOURCES:
        raw = (ROOT / path).read_bytes()
        if raw != _git('show', f'{args.revision}:{path}'):
            raise ValueError('screen committed source differs: ' + path)
        source['sources'][path] = hashlib.sha256(raw).hexdigest()
    old = _load_verified_module(args.revision, 'tools/run_vision_coco_occlusion.py', '_prior_occlusion_owner')
    old.validate_cache_placement(args.output, sys.pycache_prefix, ROOT, os.environ)
    args.output.mkdir(exist_ok=False)
    (args.output / 'source.json').write_bytes(old.canonical(source))
    evidence = model = None
    try:
        import numpy as np
        import PIL
        import torch
        import transformers
        from transformers import ViTForImageClassification
        from looking_glass.vision_coco_cases import verify_assets
        from looking_glass.vision_coco_cls_occlusion_cases import previous_images
        from looking_glass.vision_coco_cls_occlusion_discovery_handoff import verify_discovery_handoff
        from looking_glass.vision_coco_recognized_cases import read_expected_image
        from looking_glass.vision_coco_recognized_handoff import validate_candidates, validate_runtime
        from looking_glass.vision_coco_recognized_screen import check_rounding
        from looking_glass.vision_coco_small_dose_cases import check_image_hashes
        from looking_glass.vision_coco_study import Evidence

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise ValueError('denormal preservation unavailable')
        runtime = old.runtime(torch, np, PIL, transformers)
        validate_runtime(runtime)
        manifest = json.loads((ROOT / MANIFEST).read_bytes())
        if manifest.get('schema') != 'vision-coco-cls-occlusion-cases-v1':
            raise ValueError('fresh manifest schema differs')
        images = manifest['images']
        validate_candidates(images)
        previous = previous_images(*[(ROOT / name).read_bytes() for name in (
            'experiments/vision_coco_spatial_v1.json', 'experiments/vision_coco_same_site_v3.json',
            'experiments/vision_coco_recognized_v4.json', 'experiments/vision_coco_cls_compare_v5.json',
            'experiments/vision_coco_spatial_dose_v6.json', 'experiments/vision_coco_occlusion_v1.json')])
        check_image_hashes(previous, images)
        handoff = verify_discovery_handoff(args.discovery_data, args.discovery_archive,
                                           *((ROOT / path).read_bytes() for path in HANDOFF),
                                           expected_manifest_sha256=MANIFEST_SHA,
                                           expected_public_commit=DISCOVERY_COMMIT, expected_model_state=old.MODEL_STATE)
        if (handoff['runtime'] != runtime or
                any(source['sources'].get(path) != digest for path, digest in handoff['source']['sources'].items())):
            raise ValueError('discovery runtime or scientific source differs')
        inputs = load_fresh_inputs(args.assets, args.image_dir, images)
        if args.mode == 'preflight':
            def forbidden(*_args, **_kwargs):
                raise RuntimeError('preflight model forward is forbidden')
            ViTForImageClassification.forward = forbidden
        model, info = ViTForImageClassification.from_pretrained(
            str(args.assets), local_files_only=True, use_safetensors=True,
            output_loading_info=True, attn_implementation='eager')
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('checkpoint load was incomplete')
        model.eval().to('cpu')
        if (model.config.hidden_size != 192 or model.config.num_hidden_layers != 12 or
                model.config.num_labels != 1000 or model.config.patch_size != 16 or model.config.image_size != 224):
            raise ValueError('checkpoint geometry differs')
        for group in manifest['class_groups'].values():
            if any(model.config.id2label[item['id']] != item['label'] for item in group):
                raise ValueError('checkpoint class labels differ')
        if old.state_digest(model) != old.MODEL_STATE:
            raise ValueError('checkpoint state differs')
        hooks = lambda: {name: (len(module._forward_pre_hooks), len(module._forward_hooks))
                         for name, module in model.named_modules()}
        initial_hooks = hooks()

        class BoundedEvidence(Evidence):
            def reserve(self, label):
                if shutil.disk_usage(ROOT).free < 1024**3 or shutil.disk_usage(args.output).free < 1024**3:
                    raise ValueError('disk headroom fell during screen')
                super().reserve(label)

        evidence = BoundedEvidence(args.output / 'data', 0 if args.mode == 'preflight' else 192,
                                   labels=model.config.id2label)
        evidence.json('started.json', {'schema': 'vision-cls-occlusion-v1-screen-start', 'mode': args.mode,
                                      'manifest_sha256': MANIFEST_SHA, 'source': source, 'runtime': runtime,
                                      'model_state_sha256': old.MODEL_STATE, 'image_ids': [r['image_id'] for r in images],
                                      'discovery_terminal_sha256': handoff['terminal_sha256'],
                                      'discovery_archive_sha256': handoff['archive_sha256'],
                                      'discovery_public_commit': DISCOVERY_COMMIT})
        terminal = {'schema': 'vision-cls-occlusion-v1-screen-terminal', 'status': 'complete', 'mode': args.mode,
                    'manifest_sha256': MANIFEST_SHA, 'source_revision': args.revision, 'runtime': runtime}
        if args.mode == 'screen':
            screened = run_fresh_screen(model, images, inputs, evidence, MANIFEST_SHA, numeric_guard=check_rounding)
            terminal.update(screen_result_sha256=screened['sha256'], counts=screened['result']['counts'],
                            selected_ids=screened['result']['selected_ids'], decision=screened['result']['decision'])
        if (old.state_digest(model) != old.MODEL_STATE or hooks() != initial_hooks or
                old.runtime(torch, np, PIL, transformers) != runtime):
            raise ValueError('model state, hooks or CPU runtime changed')
        verify_assets(args.assets)
        for image in images:
            read_expected_image(args.image_dir / image['file_name'], image['jpeg_bytes'], image['jpeg_sha256'])
        if verify_discovery_handoff(args.discovery_data, args.discovery_archive,
                                    *((ROOT / path).read_bytes() for path in HANDOFF),
                                    expected_manifest_sha256=MANIFEST_SHA,
                                    expected_public_commit=DISCOVERY_COMMIT,
                                    expected_model_state=old.MODEL_STATE) != handoff:
            raise ValueError('discovery handoff changed during screen')
        old.recheck(source)
        if evidence.submissions != len(evidence.records):
            raise ValueError('screen submissions and saved-forward counts differ')
        terminal.update(model_state_sha256_before_after=old.MODEL_STATE, submissions=evidence.submissions,
                        saved_forwards=len(evidence.records))
        evidence.json('terminal.json', terminal)
        return 0
    except BaseException as error:
        if evidence is not None:
            try:
                evidence.json('failure.json', {'status': 'incomplete', 'error_type': type(error).__name__,
                                               'active': evidence.active, 'submissions': evidence.submissions,
                                               'saved_forwards': len(evidence.records)})
            except Exception:
                pass
        raise
    finally:
        model = None
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'screen'), required=True)
    for name in ('assets', 'image-dir', 'discovery-data', 'discovery-archive', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    raise SystemExit(main(parser.parse_args()))
