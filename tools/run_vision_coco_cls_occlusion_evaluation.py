"""Run the fixed, response-selected CLS/occlusion comparison on CPU."""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('evaluation owner requires isolated no-site no-bytecode startup')
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
            raise RuntimeError('evaluation owner requires an absolute empty bytecode prefix')
    except OSError as error:
        raise RuntimeError('evaluation owner requires an absolute empty bytecode prefix') from error

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULT = 'results/vision-coco-cls-occlusion-v1-20261029'
MANIFEST = 'experiments/vision_coco_cls_occlusion_v1.json'
MANIFEST_SHA = 'a52040786557fde9e503b6b6d1246bee0cbd41106b3cd110ea6788ab13a53487'
SCREEN_COMMIT = '5a1df2baf7929d8c03167ff142a85c2d0c8891ef'
SCREEN_RESULT_SHA = '49fa38a43ad78035c90f8cfcf6bc7558002cdc94c16fc443350bc46962d7b4a8'
SCREEN_ARCHIVE_SHA = '699eeb5f401bd78189263e0a4e77def091b440cc8843631bcf4bbd5ea967382e'
HANDOFF = tuple(RESULT + '/' + name for name in ('screen-evidence-index.json', 'screen-result.json',
                'screen-observer.json', 'screen-public-readback.json'))
EXTRA = HANDOFF + ('looking_glass/vision_coco_cls_occlusion_screen_handoff.py',
                   'looking_glass/vision_coco_cls_occlusion_evaluation.py',
                   'tools/run_vision_coco_cls_occlusion_evaluation.py',
                   'tools/observe_vision_coco_cls_occlusion_evaluation.py')


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


def snapshot(revision, discovery_owner, screen_owner):
    if not re.fullmatch('[0-9a-f]{40}', revision) or _git('rev-parse', 'HEAD').decode().strip() != revision:
        raise ValueError('evaluation source revision differs')
    source = discovery_owner.snapshot(revision)
    for path in screen_owner.EXTRA_SOURCES + EXTRA:
        raw = (ROOT / path).read_bytes()
        if raw != _git('show', f'{revision}:{path}'):
            raise ValueError('evaluation committed source differs: ' + path)
        source['sources'][path] = hashlib.sha256(raw).hexdigest()
    return source


def main(args):
    paths = (args.assets, args.image_dir, args.screen_data, args.screen_archive, args.output)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or
            any(not path.is_absolute() for path in paths) or not args.assets.is_dir() or not args.image_dir.is_dir() or
            not args.screen_data.is_dir() or not args.screen_archive.is_file() or
            not args.output.parent.is_dir() or args.output.exists()):
        raise ValueError('evaluation environment or input/output paths differ')
    def overlaps(first, second):
        first, second = first.resolve(), second.resolve()
        return first.is_relative_to(second) or second.is_relative_to(first)
    if any(overlaps(args.output, item) for item in (ROOT, args.assets, args.image_dir, args.screen_data)):
        raise ValueError('evaluation source, inputs and output must be separate')
    if shutil.disk_usage(ROOT).free < 3 * 1024**3 // 2 or shutil.disk_usage(args.output.parent).free < 1024**3:
        raise ValueError('insufficient source/cache or evidence space')
    spec = importlib.util.find_spec('looking_glass')
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / 'looking_glass/__init__.py').resolve():
        raise ValueError('project package is shadowed')
    discovery_owner = _load_verified_module(args.revision, 'tools/run_vision_coco_cls_occlusion_discovery.py',
                                            '_cls_occlusion_discovery_owner')
    screen_owner = _load_verified_module(args.revision, 'tools/run_vision_coco_cls_occlusion_screen.py',
                                         '_cls_occlusion_screen_owner')
    old = _load_verified_module(args.revision, 'tools/run_vision_coco_occlusion.py', '_prior_occlusion_owner')
    source = snapshot(args.revision, discovery_owner, screen_owner)
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
        from looking_glass.vision_coco_cls_occlusion_evaluation import run_evaluation
        from looking_glass.vision_coco_cls_occlusion_screen_handoff import verify_screen_handoff
        from looking_glass.vision_coco_recognized_cases import read_expected_image
        from looking_glass.vision_coco_recognized_handoff import validate_candidates, validate_runtime
        from looking_glass.vision_coco_recognized_screen import check_rounding
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
        index_raw, result_raw, observer_raw, readback_raw = ((ROOT / path).read_bytes() for path in HANDOFF)
        if hashlib.sha256(result_raw).hexdigest() != SCREEN_RESULT_SHA:
            raise ValueError('fixed public screen result differs')
        handoff = verify_screen_handoff(images, args.screen_data, args.screen_archive,
                                        index_raw, result_raw, observer_raw, readback_raw,
                                        manifest_sha256=MANIFEST_SHA, public_commit=SCREEN_COMMIT,
                                        model_state=old.MODEL_STATE)
        if (handoff['archive_sha256'] != SCREEN_ARCHIVE_SHA or handoff['runtime'] != runtime or
                any(source['sources'].get(path) != digest for path, digest in handoff['source']['sources'].items())):
            raise ValueError('screen archive, runtime or scientific source differs')
        selected = handoff['selected']
        if not selected:
            raise ValueError('empty selection must stop before evaluation')
        inputs = screen_owner.load_fresh_inputs(args.assets, args.image_dir, selected)
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
                    raise ValueError('disk headroom fell during evaluation')
                super().reserve(label)

        evidence = BoundedEvidence(args.output / 'data', 0 if args.mode == 'preflight' else 184,
                                   labels=model.config.id2label)
        ids = [image['image_id'] for image in selected]
        evidence.json('started.json', {'schema': 'vision-cls-occlusion-v1-evaluation-start', 'mode': args.mode,
                                      'manifest_sha256': MANIFEST_SHA, 'source': source, 'runtime': runtime,
                                      'model_state_sha256': old.MODEL_STATE, 'selected_ids': ids,
                                      'screen_result_sha256': SCREEN_RESULT_SHA,
                                      'screen_archive_sha256': SCREEN_ARCHIVE_SHA,
                                      'screen_terminal_sha256': handoff['terminal_sha256'],
                                      'screen_public_commit': SCREEN_COMMIT})
        terminal = {'schema': 'vision-cls-occlusion-v1-evaluation-terminal', 'status': 'complete',
                    'mode': args.mode, 'manifest_sha256': MANIFEST_SHA, 'source_revision': args.revision,
                    'runtime': runtime, 'selected_ids': ids}
        if args.mode == 'evaluation':
            terminal.update(run_evaluation(model, handoff, json.loads(result_raw)['rows'], inputs, evidence,
                                           numeric_guard=check_rounding))
        if (old.state_digest(model) != old.MODEL_STATE or hooks() != initial_hooks or
                old.runtime(torch, np, PIL, transformers) != runtime):
            raise ValueError('model state, hooks or CPU runtime changed')
        verify_assets(args.assets)
        for image in selected:
            read_expected_image(args.image_dir / image['file_name'], image['jpeg_bytes'], image['jpeg_sha256'])
        again = verify_screen_handoff(images, args.screen_data, args.screen_archive,
                                      *((ROOT / path).read_bytes() for path in HANDOFF),
                                      manifest_sha256=MANIFEST_SHA, public_commit=SCREEN_COMMIT,
                                      model_state=old.MODEL_STATE)
        if (again['terminal_sha256'] != handoff['terminal_sha256'] or
                again['archive_sha256'] != handoff['archive_sha256'] or
                [row['image_id'] for row in again['selected']] != ids):
            raise ValueError('screen handoff changed during evaluation')
        old.recheck(source)
        if evidence.submissions != len(evidence.records):
            raise ValueError('evaluation submissions and saved-forward counts differ')
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
    parser.add_argument('--mode', choices=('preflight', 'evaluation'), required=True)
    for name in ('assets', 'image-dir', 'screen-data', 'screen-archive', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    raise SystemExit(main(parser.parse_args()))
