"""Run one fixed CPU occlusion phase; retain every returned model output.

Source, runtime and process checks assume a trusted, quiescent workspace and
do not attest the interpreter, native libraries, chronology or descendants.
"""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('owner requires isolated no-site no-bytecode startup')

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'experiments/vision_coco_occlusion_v1.json'
MANIFEST_SHA = 'aa61b294710f5a6cd9682dc783085c7079f1addfbad8257fdbd893d143ce4cef'
V6_MANIFEST = 'experiments/vision_coco_spatial_dose_v6.json'
V6_SCREEN = 'results/vision-coco-spatial-dose-v6-20261029/screen-result.json'
V6_ARCHIVE = 'results/vision-coco-spatial-dose-v6-20261029/screen-evidence.zip'
V6_ARCHIVE_SHA = '7b88dc730498fad1418b8b0c54a33e673357252093a460611f440187a63e0f76'
RESULT = 'results/vision-coco-occlusion-v1-20261029'
DISCOVERY = RESULT + '/discovery-freeze.json'
SCREEN = RESULT + '/screen-result.json'
SCREEN_ARCHIVE = RESULT + '/screen-evidence.zip'
MODEL_STATE = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
SOURCES = (
    'docs/VISION_COCO_OCCLUSION_V1.md', MANIFEST, V6_MANIFEST, V6_SCREEN, V6_ARCHIVE,
    'tools/run_vision_coco_occlusion.py', 'tools/observe_vision_coco_occlusion.py',
    'tools/prepare_vision_coco_occlusion.py',
    'looking_glass/__init__.py', 'looking_glass/vision_coco_occlusion_cases.py',
    'looking_glass/vision_coco_occlusion_study.py', 'looking_glass/vision_coco_recognized_cases.py',
    'looking_glass/vision_coco_recognized_handoff.py', 'looking_glass/vision_coco_recognized_screen.py',
    'looking_glass/vision_coco_cls_cases.py', 'looking_glass/vision_coco_cls_study.py',
    'looking_glass/vision_coco_spatial_dose_cases.py', 'looking_glass/vision_coco_cases.py',
    'looking_glass/vision_coco_small_dose_cases.py', 'looking_glass/vision_coco_study.py',
    'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
    'experiments/vision_coco_spatial_v1.json', 'experiments/vision_coco_same_site_v3.json',
    'experiments/vision_coco_recognized_v4.json', 'experiments/vision_coco_cls_compare_v5.json',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', *args],
                                   cwd=ROOT, env=env)


def snapshot(revision, mode):
    if not re.fullmatch('[0-9a-f]{40}', revision) or git('rev-parse', 'HEAD').decode().strip() != revision:
        raise ValueError('local source revision differs')
    paths = list(SOURCES)
    if mode in ('screen', 'evaluation'):
        paths.append(DISCOVERY)
    if mode == 'evaluation':
        paths.extend((SCREEN, SCREEN_ARCHIVE))
    sources = {}
    for path in paths:
        raw = (ROOT / path).read_bytes()
        if raw != git('show', f'{revision}:{path}'):
            raise ValueError('committed source differs: ' + path)
        sources[path] = sha(raw)
    if sources[MANIFEST] != MANIFEST_SHA or sources[V6_ARCHIVE] != V6_ARCHIVE_SHA:
        raise ValueError('fixed manifest or prior screen archive differs')
    return {'revision': revision, 'sources': sources,
            'scope': 'trusted local checkout; public relevant-file equivalence is checked separately'}


def recheck(source):
    for path, expected in source['sources'].items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError('source changed during phase')


def runtime(torch, np, PIL, transformers):
    from looking_glass.vision_coco_recognized_screen import check_rounding
    check_rounding()
    return {'device': 'cpu', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_initialized': torch.cuda.is_initialized(), 'python': sys.version.split()[0],
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'numpy': np.__version__, 'pillow': PIL.__version__, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'mkldnn_enabled': torch.backends.mkldnn.enabled, 'default_dtype': str(torch.get_default_dtype()),
            'processor': 'ViTImageProcessor',
            'denormals_preserved': torch.tensor([1.401298464324817e-45], dtype=torch.float32).item() != 0,
            'mkl_cbwr': os.environ.get('MKL_CBWR'), 'binary64_rounding': 'nearest-even'}


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + b'\0')
        digest.update(json.dumps(list(value.shape), separators=(',', ':')).encode() + b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_inputs(assets, images):
    from PIL import Image
    import torch
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
    for item in images:
        path = assets / 'images' / item['file_name']
        raw = read_expected_image(path, item['jpeg_bytes'], item['jpeg_sha256'])
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (item['width'], item['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('image geometry differs')
            value = processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        if (tuple(value.shape) != (1, 3, 224, 224) or value.dtype != torch.float32 or
                not bool(torch.isfinite(value).all()) or
                sha(value.contiguous().numpy().astype('<f4', copy=False).tobytes()) != item['pixel_sha256'] or
                read_expected_image(path, item['jpeg_bytes'], item['jpeg_sha256']) != raw):
            raise ValueError('processed input or stored JPEG differs')
        output[item['image_id']] = value
    if len(output) != len(images):
        raise ValueError('duplicate input identity')
    verify_assets(assets)
    return output


def prior_screen_logits(images):
    """Bind fixed discovery clean outputs to all retained v6 screen vectors."""
    from looking_glass.vision_coco_recognized_screen import screen_score
    records = json.loads((ROOT / V6_SCREEN).read_bytes())['screen']
    wanted, found = {row['image_id'] for row in images}, {}
    raw_archive = (ROOT / V6_ARCHIVE).read_bytes()
    if sha(raw_archive) != V6_ARCHIVE_SHA:
        raise ValueError('v6 screen archive differs')
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        for index, score_record in enumerate(records, 1):
            if score_record['image_id'] not in wanted:
                continue
            row = json.loads(archive.read(f'forward-{index:03d}.json'))
            raw = archive.read(f'forward-{index:03d}.bin')
            if (len(raw) != 4000 or sha(raw) != row['sha256'] or
                    sha(raw) != score_record['logit_sha256']):
                raise ValueError('v6 discovery screen vector differs')
            image = next(item for item in images if item['image_id'] == score_record['image_id'])
            if screen_score(raw, image['category_id']) != {key: value for key, value in score_record.items()
                                                          if key != 'image_id'}:
                raise ValueError('v6 discovery screen score differs')
            found[image['image_id']] = raw
    if set(found) != wanted:
        raise ValueError('v6 discovery screen inventory differs')
    return found


def verify_public_screen_archive(data, archive_bytes):
    """Require the committed screen archive to match every closed record."""
    data = Path(data)
    expected = {'started.json', 'terminal.json', 'screen-result.json'} | {
        f'forward-{index:03d}.{suffix}' for index in range(1, 65) for suffix in ('json', 'bin')}
    if {item.name for item in data.iterdir()} != expected or len(archive_bytes) > 5_000_000:
        raise ValueError('screen evidence inventory differs')
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError('screen archive inventory differs')
        for name in names:
            if archive.read(name) != (data / name).read_bytes():
                raise ValueError('screen archive member differs')
    return sha(archive_bytes)


def main(args):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or
            not args.assets.is_absolute() or not args.output.is_absolute() or
            not args.output.parent.is_dir() or args.output.exists()):
        raise ValueError('owner environment or output path differs')
    def overlaps(first, second):
        first, second = first.resolve(), second.resolve()
        return first.is_relative_to(second) or second.is_relative_to(first)
    if (not args.assets.is_dir() or overlaps(args.output, ROOT) or overlaps(args.output, args.assets) or
            (args.screen_data is not None and (args.mode != 'evaluation' or
             any(overlaps(args.screen_data, item) for item in (args.output, ROOT, args.assets))))):
        raise ValueError('source, assets, screen and new output roots must be separate')
    if shutil.disk_usage(ROOT).free < 2 * 1024**3 or shutil.disk_usage(args.output.parent).free < 1024**3:
        raise ValueError('insufficient source/cache or evidence space')
    import importlib.util
    spec = importlib.util.find_spec('looking_glass')
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / 'looking_glass/__init__.py').resolve():
        raise ValueError('project package is shadowed')
    source = snapshot(args.revision, args.mode)
    args.output.mkdir(exist_ok=False)
    (args.output / 'source.json').write_bytes(canonical(source))
    evidence = model = None
    try:
        import numpy as np
        import PIL
        import torch
        import transformers
        from transformers import ViTForImageClassification
        from looking_glass.vision_coco_cases import verify_assets
        from looking_glass.vision_coco_occlusion_cases import previous_images, select_discovery
        from looking_glass.vision_coco_occlusion_study import (aggregate_images, run_case, run_screen,
                                                                verify_saved_screen)
        from looking_glass.vision_coco_recognized_handoff import validate_candidates, validate_runtime
        from looking_glass.vision_coco_recognized_screen import check_rounding
        from looking_glass.vision_coco_small_dose_cases import check_image_hashes
        from looking_glass.vision_coco_study import Evidence
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise ValueError('denormal preservation unavailable')
        current_runtime = runtime(torch, np, PIL, transformers)
        validate_runtime(current_runtime)
        manifest = json.loads((ROOT / MANIFEST).read_bytes())
        if manifest.get('schema') != 'vision-coco-occlusion-cases-v1':
            raise ValueError('occlusion manifest differs')
        validate_candidates(manifest['images'])
        previous = previous_images(*[(ROOT / path).read_bytes() for path in (
            'experiments/vision_coco_spatial_v1.json', 'experiments/vision_coco_same_site_v3.json',
            'experiments/vision_coco_recognized_v4.json', 'experiments/vision_coco_cls_compare_v5.json',
            V6_MANIFEST)])
        check_image_hashes(previous, manifest['images'])
        if args.mode in ('preflight', 'discovery'):
            images = select_discovery((ROOT / V6_MANIFEST).read_bytes(), (ROOT / V6_SCREEN).read_bytes())
            expected = prior_screen_logits(images)
        elif args.mode == 'screen':
            freeze = json.loads((ROOT / DISCOVERY).read_bytes())
            if (freeze.get('schema') != 'vision-occlusion-v1-discovery-freeze' or
                    freeze.get('status') != 'complete' or freeze.get('manifest_sha256') != MANIFEST_SHA or
                    freeze.get('selected_ids') != [328437, 142719, 476736, 503893] or
                    type(freeze.get('terminal_sha256')) is not str or
                    not re.fullmatch('[0-9a-f]{64}', freeze['terminal_sha256'])):
                raise ValueError('published discovery freeze differs')
            images, expected = manifest['images'], None
        else:
            if args.screen_data is None or not args.screen_data.is_absolute() or not args.screen_data.is_dir():
                raise ValueError('evaluation needs the closed screen data')
            screened = verify_saved_screen(manifest['images'], args.screen_data,
                                           expected_result=(ROOT / SCREEN).read_bytes(),
                                           expected_manifest_sha256=MANIFEST_SHA)
            verify_public_screen_archive(args.screen_data, (ROOT / SCREEN_ARCHIVE).read_bytes())
            started = json.loads((args.screen_data / 'started.json').read_bytes())
            terminal = json.loads((args.screen_data / 'terminal.json').read_bytes())
            if (started.get('runtime') != current_runtime or started.get('model_state_sha256') != MODEL_STATE or
                    terminal.get('status') != 'complete' or terminal.get('manifest_sha256') != MANIFEST_SHA or
                    type(terminal.get('submissions')) is not int or terminal['submissions'] != 64 or
                    type(terminal.get('saved_forwards')) is not int or terminal['saved_forwards'] != 64 or
                    terminal.get('screen_result_sha256') != screened['result_sha256']):
                raise ValueError('screen handoff identity or runtime differs')
            base_sources = {key: value for key, value in source['sources'].items()
                            if key not in (SCREEN, SCREEN_ARCHIVE)}
            screen_sources = started['source']['sources']
            if base_sources != screen_sources:
                raise ValueError('screen scientific source differs')
            images, expected = screened['selected'], screened['raw_by_id']
        inputs = load_inputs(args.assets, images)
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
        if state_digest(model) != MODEL_STATE:
            raise ValueError('checkpoint state differs')
        hooks = lambda: {name: (len(module._forward_pre_hooks), len(module._forward_hooks))
                         for name, module in model.named_modules()}
        initial_hooks = hooks()

        class BoundedEvidence(Evidence):
            def reserve(self, label):
                if (shutil.disk_usage(ROOT).free < 1024**3 or
                        shutil.disk_usage(args.output).free < 1024**3):
                    raise ValueError('disk headroom fell during the phase')
                super().reserve(label)

        cap = {'preflight': 0, 'discovery': 72, 'screen': 64, 'evaluation': 144}[args.mode]
        evidence = BoundedEvidence(args.output / 'data', cap, labels=model.config.id2label)
        evidence.json('started.json', {'schema': 'vision-occlusion-v1-start', 'mode': args.mode,
                                      'manifest_sha256': MANIFEST_SHA, 'source': source,
                                      'runtime': current_runtime, 'model_state_sha256': MODEL_STATE,
                                      'selected_ids': [image['image_id'] for image in images]})
        terminal = {'schema': 'vision-occlusion-v1-terminal', 'status': 'complete', 'mode': args.mode,
                    'manifest_sha256': MANIFEST_SHA, 'source_revision': args.revision, 'runtime': current_runtime}
        if args.mode == 'screen':
            observed, selected, counts = run_screen(model, images, inputs, evidence, numeric_guard=check_rounding)
            screen = {'schema': 'vision-occlusion-v1-screen-result', 'manifest_sha256': MANIFEST_SHA,
                      'screen': observed, 'selected_ids': [row['image_id'] for row in selected], 'counts': counts}
            terminal['screen_result_sha256'] = evidence.json('screen-result.json', screen)
        elif args.mode in ('discovery', 'evaluation'):
            rows = [run_case(model, image, inputs[image['image_id']], evidence,
                             expected_logits=expected[image['image_id']], numeric_guard=check_rounding)
                    for image in images]
            evaluable = [(row['category_id'], row['comparisons']['positive']) for row in rows
                         if row['dose']['status'] == 'matchable']
            terminal.update(cases=rows, aggregate=aggregate_images(evaluable), selected=len(rows))
        if (state_digest(model) != MODEL_STATE or hooks() != initial_hooks or
                runtime(torch, np, PIL, transformers) != current_runtime):
            raise ValueError('model state, hooks or CPU runtime changed')
        verify_assets(args.assets)
        recheck(source)
        if evidence.submissions != len(evidence.records):
            raise ValueError('submission and saved-forward counts differ')
        terminal.update(model_state_sha256_before_after=MODEL_STATE, submissions=evidence.submissions,
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
    parser.add_argument('--mode', choices=('preflight', 'discovery', 'screen', 'evaluation'), required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--screen-data', type=Path)
    raise SystemExit(main(parser.parse_args()))
