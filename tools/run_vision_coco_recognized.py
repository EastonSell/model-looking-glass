"""Supervise the frozen v4 CPU screen or selected same-site interventions.

The checkout, interpreter and libraries are trusted and quiescent. Direct
child reaping and hashes are consistency checks, not process-tree, execution,
source-publication or chronology attestations. Retain incomplete attempts.
"""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('v4 owner requires isolated no-site no-bytecode startup and a fresh bytecode prefix')

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

if __name__ == '__main__':
    _prefix = Path(sys.pycache_prefix)
    if not _prefix.is_absolute() or not _prefix.is_dir() or any(_prefix.iterdir()):
        raise RuntimeError('v4 owner requires an externally created empty private bytecode prefix')

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'experiments/vision_coco_recognized_v4.json'
MANIFEST_SHA = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'
PUBLIC_SCREEN = 'results/vision-coco-recognized-v4-20261029/screen-result.json'
PUBLIC_SCREEN_ARCHIVE = 'results/vision-coco-recognized-v4-20261029/screen-evidence.zip'
PUBLIC_SCREEN_RECEIPT = 'results/vision-coco-recognized-v4-20261029/screen-custody.json'
PUBLIC_SCREEN_OBSERVER = 'results/vision-coco-recognized-v4-20261029/screen-observer.json'
SELECTION_REVISION = 'eaf3e984fca1a7bc942f9e1aaefffe6521344fc9'
MODEL_STATE = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
SOURCES = (
    MANIFEST, 'docs/VISION_COCO_RECOGNIZED_SCREEN_V4.md', 'docs/VISION_COCO_RECOGNIZED_CPU_V4.md',
    'tools/run_vision_coco_recognized.py',
    'tools/prepare_vision_coco_recognized.py', 'looking_glass/__init__.py',
    'looking_glass/vision_coco_recognized_cases.py', 'looking_glass/vision_coco_recognized_screen.py',
    'looking_glass/vision_coco_recognized_runner.py', 'looking_glass/vision_coco_recognized_handoff.py',
    'looking_glass/vision_coco_cases.py', 'looking_glass/vision_coco_small_dose_cases.py',
    'looking_glass/vision_coco_same_site_study.py', 'looking_glass/vision_coco_study.py',
    'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
    'looking_glass/model_ownership.py', 'looking_glass/cpu_study_supervision.py',
    'looking_glass/supervisor.py', 'looking_glass/device_lease.py', 'looking_glass/device_identity.py',
    'experiments/vision_coco_spatial_v1.json', 'experiments/vision_coco_same_site_v3.json',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_json(path, value):
    raw = canonical(value)
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_GRAFT_FILE=os.devnull, GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', *args],
                                   cwd=ROOT, env=env)


def source_snapshot(revision):
    if not re.fullmatch('[0-9a-f]{40}', revision) or git('rev-parse', 'HEAD').decode().strip() != revision:
        raise ValueError('checkout revision differs')
    git('merge-base', '--is-ancestor', SELECTION_REVISION, revision)
    sources = {}
    for path in SOURCES:
        data = (ROOT / path).read_bytes()
        if data != git('show', f'{revision}:{path}'):
            raise ValueError(f'committed source differs: {path}')
        sources[path] = sha(data)
    if sources[MANIFEST] != MANIFEST_SHA:
        raise ValueError('fixed v4 image manifest differs')
    return {'revision': revision, 'sources': sources,
            'scope': 'trusted quiescent checkout/interpreter; not execution or timing attestation'}


def recheck(snapshot):
    if set(snapshot['sources']) != set(SOURCES):
        raise ValueError('source inventory differs')
    for path, expected in snapshot['sources'].items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError('source changed during attempt')
    handoff = snapshot.get('public_handoff')
    if handoff is not None and (type(handoff) is not dict or
            set(handoff) != {PUBLIC_SCREEN, PUBLIC_SCREEN_ARCHIVE, PUBLIC_SCREEN_RECEIPT, PUBLIC_SCREEN_OBSERVER}):
        raise ValueError('public screen handoff inventory differs')
    for path, expected in (handoff or {}).items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError('public screen handoff changed during attempt')


def verify_project_origin():
    import importlib.util
    spec = importlib.util.find_spec('looking_glass')
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / 'looking_glass/__init__.py').resolve():
        raise ValueError('project package is shadowed')


def verify_intervention_schedule(data, terminal, pairs):
    """Structural receipt check; separate raw scientific replay is required."""
    from looking_glass.vision_coco_recognized_screen import aggregate_recognized
    rows = terminal.get('pairs')
    if (type(rows) is not list or len(rows) != len(pairs) or
            [r.get('pair_id') for r in rows] != [p['pair_id'] for p in pairs] or
            canonical(terminal.get('aggregate')) != canonical(aggregate_recognized(rows))):
        raise ValueError('intervention pair or aggregate differs')
    labels, seen = [], set()
    for pair in pairs:
        for role in ('cat', 'dog', 'cat_donor', 'dog_donor'):
            image_id = pair[role]['image_id']
            if image_id not in seen:
                seen.add(image_id)
                labels.append(f"{pair['pair_id']}:{role}:initial")
    for pair, row in zip(pairs, rows):
        if canonical(json.loads((data / f"{pair['pair_id']}.json").read_bytes())) != canonical(row):
            raise ValueError('saved pair differs')
        for side in ('cat', 'dog'):
            info = row.get('sides', {}).get(side, {})
            if (type(info.get('image_id')) is not int or info['image_id'] != pair[side]['image_id'] or
                    type(info.get('same_species_donor_id')) is not int or
                    info['same_species_donor_id'] != pair[side + '_donor']['image_id']):
                raise ValueError('saved target or donor differs')
            dose = info.get('dose', {}).get('status')
            if dose not in ('matchable', 'dose-ineligible', 'dose-unmatchable'):
                raise ValueError('saved dose status differs')
            names = ['repeat', 'null_inside', 'self_inside', 'full_self', 'full_swap', 'natural_inside']
            if dose == 'matchable':
                names += ['matched_opposite', 'matched_same_species', 'matched_random']
            names += ['final_inside', 'post_clean']
            if set(info.get('conditions', {})) != set(names):
                raise ValueError('saved intervention conditions differ')
            labels.extend(f"{pair['pair_id']}:{side}:{name}" for name in names)
    count = len(labels)
    if (type(terminal.get('submissions')) is not int or terminal['submissions'] != count or
            type(terminal.get('saved_forwards')) is not int or terminal['saved_forwards'] != count or count > 256):
        raise ValueError('intervention forward count differs')
    expected = {f'forward-{i:03d}.{ext}' for i in range(1, count + 1) for ext in ('json', 'bin')}
    expected |= {'started.json', 'terminal.json'} | {f"{pair['pair_id']}.json" for pair in pairs}
    if {path.name for path in data.iterdir()} != expected:
        raise ValueError('intervention forward inventory differs')
    for index, label in enumerate(labels, 1):
        row = json.loads((data / f'forward-{index:03d}.json').read_bytes())
        raw = (data / f'forward-{index:03d}.bin').read_bytes()
        if (type(row.get('submission')) is not int or row['submission'] != index or row.get('label') != label or
                row.get('file') != f'forward-{index:03d}.bin' or row.get('sha256') != sha(raw) or
                row.get('dtype') != 'little-endian-float32' or not 0 < len(raw) <= 1_000_000 or
                type(row.get('metadata', {}).get('steps')) is not list or not row['metadata']['steps']):
            raise ValueError('saved intervention forward differs')
        layout = row.get('layout')
        if type(layout) is not dict or layout.get('logits', {}).get('shape') != [1, 1000]:
            raise ValueError('saved intervention logits differ')
        needed = ({'logits', 'early', 'late'} if label.endswith((':initial', ':repeat', ':post_clean'))
                  else {'logits', 'before_all', 'after_all'})
        if not needed <= set(layout):
            raise ValueError('saved intervention witnesses missing')
        if not label.endswith((':initial', ':repeat', ':post_clean', ':null_inside')) and 'requested' not in layout:
            raise ValueError('requested intervention witness missing')
        if not label.endswith((':initial', ':repeat', ':post_clean', ':full_self', ':full_swap')):
            if not all(f'{prefix}_{i}' in layout for i in range(4) for prefix in ('before', 'after')):
                raise ValueError('token-level intervention witnesses missing')
        cursor = 0
        for part in sorted(layout.values(), key=lambda item: item.get('offset', -1)):
            shape, nbytes = part.get('shape'), part.get('bytes')
            if (type(shape) is not list or not shape or any(type(v) is not int or v <= 0 for v in shape) or
                    type(nbytes) is not int or nbytes != 4 * math.prod(shape) or part.get('offset') != cursor):
                raise ValueError('saved intervention layout differs')
            cursor += nbytes
        if cursor != len(raw):
            raise ValueError('saved intervention payload differs')


def load_inputs(assets, manifest, *, only_ids=None):
    """Recheck each frozen JPEG and processed tensor before model execution."""
    import io
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoImageProcessor
    from looking_glass.vision_coco_cases import verify_assets
    from looking_glass.vision_coco_recognized_cases import previous_images, read_expected_image
    from looking_glass.vision_coco_recognized_handoff import validate_candidates
    from looking_glass.vision_coco_small_dose_cases import check_image_hashes
    if (torch.cuda.is_initialized() or PIL.__version__ != '11.3.0' or np.__version__ != '2.4.6' or
            torch.__version__ != '2.8.0+cu128' or transformers.__version__ != '4.57.3'):
        raise ValueError('input runtime differs')
    images = manifest['images']
    validate_candidates(images)
    previous = previous_images((ROOT / 'experiments/vision_coco_spatial_v1.json').read_bytes(),
                               (ROOT / 'experiments/vision_coco_same_site_v3.json').read_bytes())
    check_image_hashes(previous, images)
    verify_assets(assets)
    processor = AutoImageProcessor.from_pretrained(str(assets), local_files_only=True, use_fast=False)
    if (type(processor).__name__ != 'ViTImageProcessor' or processor.size != {'height': 224, 'width': 224}
            or int(processor.resample) != 2 or processor.image_mean != [.5] * 3
            or processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
        raise ValueError('processor differs')
    output = {}
    for item in images:
        if only_ids is not None and item['image_id'] not in only_ids:
            continue
        path = assets / 'images' / item['file_name']
        raw = read_expected_image(path, item['jpeg_bytes'], item['jpeg_sha256'])
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (item['width'], item['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('image dimensions or EXIF orientation differ')
            value = processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        if (tuple(value.shape) != (1, 3, 224, 224) or value.dtype != torch.float32 or
                not bool(torch.isfinite(value).all()) or
                sha(value.contiguous().numpy().astype('<f4', copy=False).tobytes()) != item['pixel_sha256'] or
                read_expected_image(path, item['jpeg_bytes'], item['jpeg_sha256']) != raw):
            raise ValueError('processed tensor or saved JPEG differs')
        output[item['image_id']] = value
    if only_ids is not None and set(output) != only_ids:
        raise ValueError('selected processed tensor inventory differs')
    verify_assets(assets)
    return output


def _runtime(torch, np, PIL, transformers):
    from looking_glass.vision_coco_recognized_screen import check_rounding
    check_rounding()
    return {'device': 'cpu', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_initialized': torch.cuda.is_initialized(), 'python': sys.version.split()[0],
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'numpy': np.__version__, 'pillow': PIL.__version__,
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'mkldnn_enabled': torch.backends.mkldnn.enabled, 'default_dtype': str(torch.get_default_dtype()),
            'processor': 'ViTImageProcessor',
            'denormals_preserved': torch.tensor([1.401298464324817e-45], dtype=torch.float32).item() != 0,
            'mkl_cbwr': os.environ.get('MKL_CBWR'), 'binary64_rounding': 'nearest-even'}


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + b'\0')
        digest.update(json.dumps(list(value.shape), separators=(',', ':')).encode() + b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def worker(args):
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(args.custody)
    model = evidence = None
    try:
        source = json.loads((args.output / 'source.json').read_bytes())
        recheck(source)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import numpy as np
        import PIL
        import torch
        import transformers
        from transformers import ViTForImageClassification
        from looking_glass.vision_coco_cases import verify_assets
        from looking_glass.vision_coco_recognized_handoff import (screen_receipt, validate_runtime,
                                                                  verify_closed_screen, verify_screen_archive)
        from looking_glass.vision_coco_recognized_runner import run_clean_screen
        from looking_glass.vision_coco_recognized_screen import aggregate_recognized, check_rounding
        from looking_glass.vision_coco_same_site_study import run_same_site
        from looking_glass.vision_coco_study import Evidence, _bits
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise ValueError('denormal preservation unavailable')
        runtime = _runtime(torch, np, PIL, transformers)
        validate_runtime(runtime)
        raw_manifest = (ROOT / MANIFEST).read_bytes()
        if sha(raw_manifest) != MANIFEST_SHA:
            raise ValueError('manifest changed')
        manifest = json.loads(raw_manifest)
        if (manifest.get('schema') != 'vision-coco-recognized-screen-cases-v4' or
                manifest.get('selection_source', {}).get('revision') != SELECTION_REVISION):
            raise ValueError('selection identity differs')
        for path, digest in manifest['selection_source']['sources'].items():
            if source['sources'].get(path) != digest:
                raise ValueError('selection source differs')
        screen, pairs, needed = None, [], None
        if args.mode == 'intervention':
            public_raw = (ROOT / PUBLIC_SCREEN).read_bytes()
            if sha(public_raw) != source.get('public_handoff', {}).get(PUBLIC_SCREEN):
                raise ValueError('published screen differs from its parent-bound hash')
            screen = verify_closed_screen(manifest, args.screen_data, MANIFEST_SHA, expected_result=public_raw)
            archive_raw = (ROOT / PUBLIC_SCREEN_ARCHIVE).read_bytes()
            if sha(archive_raw) != source.get('public_handoff', {}).get(PUBLIC_SCREEN_ARCHIVE):
                raise ValueError('published screen archive differs from its parent-bound hash')
            screen['archive_sha256'] = verify_screen_archive(archive_raw, args.screen_data)
            receipt_raw = (ROOT / PUBLIC_SCREEN_RECEIPT).read_bytes()
            observer_raw = (ROOT / PUBLIC_SCREEN_OBSERVER).read_bytes()
            if (sha(receipt_raw) != source.get('public_handoff', {}).get(PUBLIC_SCREEN_RECEIPT) or
                    sha(observer_raw) != source.get('public_handoff', {}).get(PUBLIC_SCREEN_OBSERVER) or
                    receipt_raw != canonical(screen_receipt(screen, args.screen_data, screen['archive_sha256'],
                                                            observer_raw))):
                raise ValueError('published screen custody differs')
            if (screen['started']['source'].get('sources') != source['sources'] or
                    canonical(screen['started']['runtime']) != canonical(runtime)):
                raise ValueError('screen and intervention source/runtime differ')
            pairs = screen['result']['pairs']
            if not pairs:
                raise ValueError('no selected pair for intervention')
            needed = {p[role]['image_id'] for p in pairs for role in ('cat', 'dog', 'cat_donor', 'dog_donor')}
        pixels = load_inputs(args.assets, manifest, only_ids=needed)
        if args.mode == 'preflight':
            def forbid_forward(*_args, **_kwargs):
                raise RuntimeError('checkpoint forward is forbidden in preflight')
            ViTForImageClassification.forward = forbid_forward
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
        before = _state_digest(model)
        if before != MODEL_STATE:
            raise ValueError('checkpoint state differs')
        hooks = lambda: {name: (len(module._forward_pre_hooks), len(module._forward_hooks))
                         for name, module in model.named_modules()}
        initial_hooks = hooks()
        evidence = Evidence(args.output / 'data', 256, labels=model.config.id2label)
        evidence.json('started.json', {'schema': 'vision-recognized-v4-start', 'mode': args.mode,
                                       'manifest_sha256': MANIFEST_SHA, 'source': source, 'runtime': runtime,
                                       'model_state_sha256': before})
        terminal = {'schema': 'vision-recognized-v4-terminal', 'status': 'complete', 'mode': args.mode,
                    'manifest_sha256': MANIFEST_SHA, 'source_revision': args.revision, 'runtime': runtime}
        if args.mode == 'screen':
            observed, pairs, summary = run_clean_screen(model, manifest['images'], pixels, evidence)
            result = {'schema': 'vision-recognized-v4-screen-result', 'manifest_sha256': MANIFEST_SHA,
                      'screen': observed, 'pairs': pairs, 'summary': summary}
            terminal['screen_result_sha256'] = evidence.json('screen-result.json', result)
        elif args.mode == 'intervention':
            def check_clean(image_id, logits):
                if _bits(logits) != screen['raw_by_id'][image_id]:
                    raise ValueError('repeated clean output differs from published screen')
            rows = run_same_site(model, pairs, pixels, evidence, check_clean=check_clean,
                                 numeric_guard=check_rounding)
            terminal.update(pairs=rows, aggregate=aggregate_recognized(rows),
                            screen_result_sha256=screen['screen_result_sha256'],
                            screen_terminal_sha256=screen['terminal_sha256'],
                            screen_archive_sha256=screen['archive_sha256'])
        after = _state_digest(model)
        if (after != before or hooks() != initial_hooks or canonical(_runtime(torch, np, PIL, transformers)) !=
                canonical(runtime)):
            raise ValueError('checkpoint state, hooks or runtime changed')
        verify_assets(args.assets)
        recheck(source)
        terminal.update(model_state_sha256_before_after=after, submissions=evidence.submissions,
                        saved_forwards=len(evidence.records))
        if evidence.submissions != len(evidence.records):
            raise ValueError('submission and saved-forward counts differ')
        evidence.json('terminal.json', terminal)
        model = None
        gc.collect()
        session.check(lease)
        session.attest_clean(lease)
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
        lease.release()
    return 0


def verify_terminal(data, terminal, source, mode, manifest, screen_data):
    """Validate direct-child output before marking a receipt complete."""
    from looking_glass.vision_coco_recognized_handoff import (screen_receipt, validate_runtime,
                                                              verify_closed_screen, verify_screen_archive)
    if (data / 'failure.json').exists():
        raise ValueError('failure evidence exists')
    started = json.loads((data / 'started.json').read_bytes())
    if (started.get('schema') != 'vision-recognized-v4-start' or started.get('mode') != mode or
            canonical(started.get('source')) != canonical(source) or started.get('manifest_sha256') != MANIFEST_SHA or
            started.get('model_state_sha256') != MODEL_STATE or
            terminal.get('schema') != 'vision-recognized-v4-terminal' or terminal.get('status') != 'complete' or
            terminal.get('mode') != mode or terminal.get('manifest_sha256') != MANIFEST_SHA or
            terminal.get('source_revision') != source['revision'] or
            terminal.get('model_state_sha256_before_after') != MODEL_STATE or
            canonical(started.get('runtime')) != canonical(terminal.get('runtime'))):
        raise ValueError('start or terminal identity differs')
    validate_runtime(started.get('runtime'))
    if mode == 'preflight':
        if (type(terminal.get('submissions')) is not int or terminal['submissions'] != 0 or
                type(terminal.get('saved_forwards')) is not int or terminal['saved_forwards'] != 0 or
                {item.name for item in data.iterdir()} != {'started.json', 'terminal.json'}):
            raise ValueError('checkpoint preflight did not remain forward-free')
    elif mode == 'screen':
        verify_closed_screen(manifest, data, MANIFEST_SHA)
    else:
        raw = (ROOT / PUBLIC_SCREEN).read_bytes()
        screen = verify_closed_screen(manifest, screen_data, MANIFEST_SHA, expected_result=raw)
        archive_digest = verify_screen_archive((ROOT / PUBLIC_SCREEN_ARCHIVE).read_bytes(), screen_data)
        observer_raw = (ROOT / PUBLIC_SCREEN_OBSERVER).read_bytes()
        if (ROOT / PUBLIC_SCREEN_RECEIPT).read_bytes() != canonical(
                screen_receipt(screen, screen_data, archive_digest, observer_raw)):
            raise ValueError('screen supervisor projection differs')
        if (screen['started']['source'].get('sources') != source['sources'] or
                terminal.get('screen_result_sha256') != screen['screen_result_sha256'] or
                terminal.get('screen_terminal_sha256') != screen['terminal_sha256'] or
                terminal.get('screen_archive_sha256') != archive_digest):
            raise ValueError('screen linkage differs')
        verify_intervention_schedule(data, terminal, screen['result']['pairs'])


def controller(args):
    from looking_glass.cpu_study_supervision import run_cpu_study
    for path in (args.output, args.custody, args.assets, args.pillow, args.site_packages):
        if not path.is_absolute():
            raise ValueError('paths must be absolute')
    def overlaps(first, second):
        first, second = first.resolve(), second.resolve()
        return first.is_relative_to(second) or second.is_relative_to(first)
    if (overlaps(args.output, ROOT) or overlaps(args.custody, ROOT) or
            overlaps(args.output, args.assets) or overlaps(args.custody, args.assets) or
            overlaps(args.output, args.custody)):
        raise ValueError('evidence, source, assets and custody roots must be separate')
    if args.mode == 'intervention':
        if args.screen_data is None or not args.screen_data.is_absolute() or not args.screen_data.is_dir():
            raise ValueError('interventions require the closed screen data path')
        if any(overlaps(args.screen_data, root) for root in (args.output, args.custody, args.assets, ROOT)):
            raise ValueError('closed screen evidence must be separate from source and new roots')
    elif args.screen_data is not None:
        raise ValueError('screen data is valid only for intervention')
    source = source_snapshot(args.revision)
    if args.mode == 'intervention':
        handoff = {}
        for path in (PUBLIC_SCREEN, PUBLIC_SCREEN_ARCHIVE, PUBLIC_SCREEN_RECEIPT, PUBLIC_SCREEN_OBSERVER):
            raw = (ROOT / path).read_bytes()
            if raw != git('show', f'{args.revision}:{path}'):
                raise ValueError('public screen handoff differs from committed bytes')
            handoff[path] = sha(raw)
        screen_started = json.loads((args.screen_data / 'started.json').read_bytes())
        screen_revision = screen_started.get('source', {}).get('revision')
        if type(screen_revision) is not str or not re.fullmatch('[0-9a-f]{40}', screen_revision):
            raise ValueError('screen source revision differs')
        git('merge-base', '--is-ancestor', screen_revision, args.revision)
        source['public_handoff'] = handoff
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'source.json', source)
    try:
        if args.mode in ('screen', 'intervention'):
            write_json(args.custody / f'vision-recognized-v4-{args.mode}.claim.json',
                       {'schema': 'vision-recognized-v4-attempt-claim', 'mode': args.mode,
                        'manifest_sha256': MANIFEST_SHA, 'source_revision': args.revision})
        cache, temporary = args.output / 'empty-bytecode-cache', args.output / 'temporary'
        cache.mkdir()
        temporary.mkdir()
        site = args.site_packages.resolve(strict=True)
        if not site.is_dir():
            raise ValueError('site-packages path must be a directory')
        bootstrap = ('import runpy,sys; sys.path.extend([sys.argv[2],sys.argv[3],sys.argv[1]]); '
                     'sys.argv=sys.argv[4:]; runpy.run_path(sys.argv[0],run_name="__main__")')
        command = [str(Path(sys._base_executable)), '-I', '-S', '-B', '-X', f'pycache_prefix={cache}',
                   '-c', bootstrap, str(ROOT), str(site), str(args.pillow.resolve(strict=True)),
                   str(Path(__file__).resolve()), '--worker', '--mode', args.mode,
                   '--revision', args.revision, '--assets', str(args.assets), '--output', str(args.output),
                   '--custody', str(args.custody), '--pillow', str(args.pillow), '--site-packages', str(site)]
        if args.screen_data is not None:
            command += ['--screen-data', str(args.screen_data)]
        env = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE',
               'TEMP': str(temporary), 'TMP': str(temporary)}
        if os.name == 'nt':
            system = os.environ.get('SystemRoot')
            if not system:
                raise ValueError('Windows SystemRoot is unavailable')
            env.update(SystemRoot=system, WINDIR=system, PATH=str(Path(system) / 'System32'),
                       COMSPEC=str(Path(system) / 'System32/cmd.exe'))
        result = run_cpu_study(command, study_root=args.custody, timeout=args.timeout, cwd=ROOT, env=env)
    except BaseException as error:
        try:
            write_json(args.output / 'supervisor.json',
                       {'status': 'incomplete', 'error_type': type(error).__name__,
                        'scope': 'owner setup or supervisor did not return a direct-child exit receipt'})
        except BaseException as receipt_error:
            # Preserve an existing fatal cancellation; a new fatal interruption
            # takes precedence over an ordinary primary exception.
            if isinstance(error, Exception) and not isinstance(receipt_error, Exception):
                raise
        raise
    receipt = dataclasses.asdict(result)
    receipt.update(status='incomplete', scope='cooperative CPU custody; direct child reaped; no process-tree attestation',
                   failure_detail='consult retained start, terminal, failure and forward evidence')
    data = args.output / 'data'
    terminal_path = data / 'terminal.json'
    if (type(result.exit_code) is int and result.exit_code == 0 and result.clean is True and
            result.timed_out is False and terminal_path.is_file()):
        try:
            terminal_raw = terminal_path.read_bytes()
            terminal = json.loads(terminal_raw)
            manifest = json.loads((ROOT / MANIFEST).read_bytes())
            verify_terminal(data, terminal, source, args.mode, manifest, args.screen_data)
            recheck(source)
            receipt.update(status='complete', terminal_sha256=sha(terminal_raw))
            receipt.pop('failure_detail', None)
        except Exception as error:
            receipt['failure_detail'] = 'saved evidence validation failed: ' + type(error).__name__
    write_json(args.output / 'supervisor.json', receipt)
    print(json.dumps({'status': receipt['status'], 'mode': args.mode, 'owner_exit': result.exit_code,
                      'terminal_sha256': receipt.get('terminal_sha256')}))
    return 0 if receipt['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('preflight', 'screen', 'intervention'), required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--custody', type=Path, required=True)
    parser.add_argument('--pillow', type=Path, required=True)
    parser.add_argument('--site-packages', type=Path, required=True)
    parser.add_argument('--screen-data', type=Path)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    return worker(args) if args.worker else controller(args)


if __name__ == '__main__':
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    verify_project_origin()
    raise SystemExit(main())
