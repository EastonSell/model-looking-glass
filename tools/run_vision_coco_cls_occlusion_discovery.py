"""Run the bounded exposed-image CLS/occlusion discovery on CPU.

The saved source and direct-child observations assume a trusted, quiescent
workspace and do not attest native execution, chronology or descendants.
"""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('owner requires isolated no-site no-bytecode startup')

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
            raise RuntimeError('owner requires an absolute empty bytecode prefix')
    except OSError as error:
        raise RuntimeError('owner requires an absolute empty bytecode prefix') from error

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'experiments/vision_coco_cls_occlusion_v1.json'
MANIFEST_SHA = 'a52040786557fde9e503b6b6d1246bee0cbd41106b3cd110ea6788ab13a53487'
PRIOR_AUDIT = 'results/vision-coco-occlusion-v1-20261029/evaluation-independent-audit.json'
PRIOR_TERMINAL_SHA = 'b15cb5c818df86746d35c62b1452e8c0f6f589adf3a17d579256d4675f779b64'
IDS = [404350, 151497, 470798, 135116]
SOURCES = (
    'docs/VISION_COCO_OCCLUSION_V1.md', 'docs/VISION_COCO_OCCLUSION_RUNTIME_AMENDMENT_V1.md',
    'experiments/vision_coco_occlusion_v1.json', 'experiments/vision_coco_spatial_dose_v6.json',
    'results/vision-coco-spatial-dose-v6-20261029/screen-result.json',
    'results/vision-coco-spatial-dose-v6-20261029/screen-evidence.zip',
    'tools/run_vision_coco_occlusion.py', 'tools/observe_vision_coco_occlusion.py',
    'tools/prepare_vision_coco_occlusion.py', 'looking_glass/__init__.py',
    'looking_glass/vision_coco_occlusion_cases.py', 'looking_glass/vision_coco_occlusion_study.py',
    'looking_glass/vision_coco_recognized_cases.py', 'looking_glass/vision_coco_recognized_handoff.py',
    'looking_glass/vision_coco_recognized_screen.py', 'looking_glass/vision_coco_cls_cases.py',
    'looking_glass/vision_coco_cls_study.py', 'looking_glass/vision_coco_spatial_dose_cases.py',
    'looking_glass/vision_coco_cases.py', 'looking_glass/vision_coco_small_dose_cases.py',
    'looking_glass/vision_coco_study.py', 'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
    'experiments/vision_coco_spatial_v1.json', 'experiments/vision_coco_same_site_v3.json',
    'experiments/vision_coco_recognized_v4.json', 'experiments/vision_coco_cls_compare_v5.json',
    MANIFEST, PRIOR_AUDIT, 'docs/VISION_COCO_CLS_OCCLUSION_V1.md',
    'looking_glass/vision_coco_cls_occlusion_cases.py',
    'looking_glass/vision_coco_cls_occlusion_study.py',
    'looking_glass/vision_coco_cls_occlusion_prior.py',
    'tools/run_vision_coco_cls_occlusion_discovery.py',
    'tools/observe_vision_coco_cls_occlusion_discovery.py',
)


def git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', *args],
                                   cwd=ROOT, env=env)


def capture_initials(*args, **kwargs):
    from looking_glass.vision_coco_cls_occlusion_study import capture_initials as implementation
    return implementation(*args, **kwargs)


def run_case(*args, **kwargs):
    from looking_glass.vision_coco_cls_occlusion_study import run_case as implementation
    return implementation(*args, **kwargs)


def snapshot(revision):
    if (not re.fullmatch('[0-9a-f]{40}', revision) or
            git('rev-parse', 'HEAD').decode().strip() != revision):
        raise ValueError('local source revision differs')
    sources = {}
    for path in SOURCES:
        raw = (ROOT / path).read_bytes()
        if raw != git('show', f'{revision}:{path}'):
            raise ValueError('committed source differs: ' + path)
        sources[path] = hashlib.sha256(raw).hexdigest()
    if sources[MANIFEST] != MANIFEST_SHA:
        raise ValueError('fixed manifest differs')
    return {'revision': revision, 'sources': sources,
            'scope': 'trusted quiescent local checkout; public relevant-file equivalence checked separately'}


def run_discovery(model, images, pixels, prior, evidence, *, numeric_guard=None):
    # This single call checks all three inputs and all logits for every case
    # before the first call to run_case can submit a patch.
    captured = capture_initials(model, images, pixels, prior['inputs'], prior['logits'], evidence,
                                numeric_guard=numeric_guard)
    return [run_case(model, image, captured[image['image_id']], evidence, numeric_guard=numeric_guard)
            for image in images]


def main(args):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or
            not args.assets.is_absolute() or not args.output.is_absolute() or not args.prior_data.is_absolute() or
            not args.assets.is_dir() or not args.prior_data.is_dir() or not args.output.parent.is_dir() or
            args.output.exists()):
        raise ValueError('owner environment or input/output paths differ')
    def overlaps(first, second):
        first, second = first.resolve(), second.resolve()
        return first.is_relative_to(second) or second.is_relative_to(first)
    if any(overlaps(args.output, item) for item in (ROOT, args.assets, args.prior_data)):
        raise ValueError('source, prior data, assets and output must be separate')
    if shutil.disk_usage(ROOT).free < 3 * 1024**3 // 2 or shutil.disk_usage(args.output.parent).free < 1024**3:
        raise ValueError('insufficient source/cache or evidence space')
    import importlib.util
    spec = importlib.util.find_spec('looking_glass')
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / 'looking_glass/__init__.py').resolve():
        raise ValueError('project package is shadowed')
    source = snapshot(args.revision)
    prior_spec = importlib.util.spec_from_file_location('_prior_occlusion_owner', ROOT / 'tools/run_vision_coco_occlusion.py')
    prior_owner = importlib.util.module_from_spec(prior_spec)
    prior_spec.loader.exec_module(prior_owner)
    prior_owner.validate_cache_placement(args.output, sys.pycache_prefix, ROOT, os.environ)
    args.output.mkdir(exist_ok=False)
    (args.output / 'source.json').write_bytes(prior_owner.canonical(source))
    evidence = model = None
    try:
        import numpy as np
        import PIL
        import torch
        import transformers
        from transformers import ViTForImageClassification
        from looking_glass.vision_coco_cases import verify_assets
        from looking_glass.vision_coco_cls_occlusion_cases import (PRIOR_EVALUATION_AUDIT_SHA256,
                                                                    previous_images, select_discovery)
        from looking_glass.vision_coco_cls_occlusion_prior import verify_prior_discovery
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
        current_runtime = prior_owner.runtime(torch, np, PIL, transformers)
        validate_runtime(current_runtime)
        manifest = json.loads((ROOT / MANIFEST).read_bytes())
        if manifest.get('schema') != 'vision-coco-cls-occlusion-cases-v1':
            raise ValueError('fresh manifest schema differs')
        validate_candidates(manifest['images'])
        previous = previous_images(*[(ROOT / name).read_bytes() for name in (
            'experiments/vision_coco_spatial_v1.json', 'experiments/vision_coco_same_site_v3.json',
            'experiments/vision_coco_recognized_v4.json', 'experiments/vision_coco_cls_compare_v5.json',
            'experiments/vision_coco_spatial_dose_v6.json', 'experiments/vision_coco_occlusion_v1.json')])
        check_image_hashes(previous, manifest['images'])
        audit_raw = (ROOT / PRIOR_AUDIT).read_bytes()
        images = select_discovery((ROOT / 'experiments/vision_coco_occlusion_v1.json').read_bytes(), audit_raw)
        if [row['image_id'] for row in images] != IDS:
            raise ValueError('discovery identities differ')
        prior = verify_prior_discovery(images, args.prior_data, audit_raw,
                                       expected_audit_sha256=PRIOR_EVALUATION_AUDIT_SHA256,
                                       expected_terminal_sha256=PRIOR_TERMINAL_SHA)
        inputs = prior_owner.load_inputs(args.assets, images)
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
        if prior_owner.state_digest(model) != prior_owner.MODEL_STATE:
            raise ValueError('checkpoint state differs')
        hooks = lambda: {name: (len(module._forward_pre_hooks), len(module._forward_hooks))
                         for name, module in model.named_modules()}
        initial_hooks = hooks()

        class BoundedEvidence(Evidence):
            def reserve(self, label):
                if shutil.disk_usage(ROOT).free < 1024**3 or shutil.disk_usage(args.output).free < 1024**3:
                    raise ValueError('disk headroom fell during the phase')
                super().reserve(label)

        evidence = BoundedEvidence(args.output / 'data', 0 if args.mode == 'preflight' else 92,
                                   labels=model.config.id2label)
        evidence.json('started.json', {'schema': 'vision-cls-occlusion-v1-start', 'mode': args.mode,
                                      'manifest_sha256': MANIFEST_SHA, 'source': source,
                                      'runtime': current_runtime, 'model_state_sha256': prior_owner.MODEL_STATE,
                                      'prior_audit_sha256': PRIOR_EVALUATION_AUDIT_SHA256,
                                      'prior_terminal_sha256': PRIOR_TERMINAL_SHA, 'selected_ids': IDS})
        terminal = {'schema': 'vision-cls-occlusion-v1-terminal', 'status': 'complete', 'mode': args.mode,
                    'manifest_sha256': MANIFEST_SHA, 'source_revision': args.revision, 'runtime': current_runtime}
        if args.mode == 'discovery':
            rows = run_discovery(model, images, inputs, prior, evidence, numeric_guard=check_rounding)
            terminal.update(cases=rows, selected=len(rows))
        if (prior_owner.state_digest(model) != prior_owner.MODEL_STATE or hooks() != initial_hooks or
                prior_owner.runtime(torch, np, PIL, transformers) != current_runtime):
            raise ValueError('model state, hooks or CPU runtime changed')
        verify_assets(args.assets)
        for image in images:
            read_expected_image(args.assets / 'images' / image['file_name'], image['jpeg_bytes'], image['jpeg_sha256'])
        prior_owner.recheck(source)
        if evidence.submissions != len(evidence.records):
            raise ValueError('submission and saved-forward counts differ')
        terminal.update(model_state_sha256_before_after=prior_owner.MODEL_STATE, submissions=evidence.submissions,
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
    parser.add_argument('--mode', choices=('preflight', 'discovery'), required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--prior-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    raise SystemExit(main(parser.parse_args()))
