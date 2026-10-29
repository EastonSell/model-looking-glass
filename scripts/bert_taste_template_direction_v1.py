"""Crossed-template CPU taste study; no downloads or hidden inputs."""
import sys
import os

if __name__ == '__main__' and (not sys.flags.isolated or not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('isolated no-bytecode child with a fresh prefix is required')
if __name__ == '__main__':
    try:
        if not os.path.isabs(sys.pycache_prefix) or os.path.islink(sys.pycache_prefix):
            raise RuntimeError('fresh empty bytecode prefix is required')
        with os.scandir(sys.pycache_prefix) as entries:
            if next(entries, None) is not None:
                raise RuntimeError('fresh empty bytecode prefix is required')
    except OSError as exc:
        raise RuntimeError('fresh empty bytecode prefix is required') from exc
if __name__ == '__main__' and (os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1')
                               or os.environ.get('OMP_NUM_THREADS') != '1'
                               or os.environ.get('MKL_NUM_THREADS') != '1'
                               or os.environ.get('MKL_CBWR') != 'AUTO'
                               or 'ATEN_CPU_CAPABILITY' in os.environ):
    raise RuntimeError('CPU numerical environment differs')

import argparse
import base64
from contextlib import ExitStack
import hashlib
import importlib.machinery
import json
import math
from pathlib import Path
import platform
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = 'experiments/bert_taste_template_direction_public_cases_v1.json'
SEED = 20261031
SOURCE_PATHS = (
    'scripts/bert_taste_template_direction_v1.py', 'scripts/bert_taste_template_cases_v1.py',
    'scripts/bert_taste_template_geometry_v1.py', 'docs/BERT_TASTE_TEMPLATE_DIRECTION_V1.md', CASES_PATH,
    'scripts/bert_taste_matched_dose_v1.py', 'scripts/bert_taste_patch_v1.py',
    'experiments/bert_agreement_exposures_v6.json', 'experiments/bert_matched_public_cases_v1.json',
    'experiments/bert_taste_public_exposures_v2.json',
    'experiments/bert_taste_matched_dose_public_exposures_v2.json',
    'experiments/bert_taste_matched_dose_public_cases_v1.json',
    'looking_glass/__init__.py', 'looking_glass/pretrained.py', 'looking_glass/bert_agreement_study.py',
    'looking_glass/bert_matched_numeric.py',
    'looking_glass/bert_agreement_runner.py', 'looking_glass/intervention_plan.py',
    'looking_glass/interventions.py', 'looking_glass/model_ownership.py',
)


class _CheckoutImportGuard:
    """Allow checkout imports only from the listed source files."""
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        project = (fullname == 'scripts' or fullname.startswith('scripts.')
                   or fullname == 'looking_glass' or fullname.startswith('looking_glass.'))
        if spec is None:
            if project:
                raise ImportError('project module cannot be resolved')
            return None
        if spec.origin not in (None, 'built-in', 'frozen'):
            lexical, resolved = Path(spec.origin).absolute(), Path(spec.origin).resolve()
            if lexical.is_relative_to(ROOT) or resolved.is_relative_to(ROOT):
                allowed = {ROOT / name for name in SOURCE_PATHS if name.endswith('.py')}
                if resolved not in allowed or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                    raise ImportError('checkout import is not an allowed source file')
                return spec
            if project:
                raise ImportError('project module resolved outside the checkout')
        elif spec.submodule_search_locations is not None:
            locations = [Path(item).resolve() for item in spec.submodule_search_locations]
            if any(item.is_relative_to(ROOT) and item not in (ROOT / 'scripts', ROOT / 'looking_glass')
                   for item in locations):
                raise ImportError('checkout namespace is not allowed')
            if project:
                if fullname != 'scripts' or locations != [ROOT / 'scripts']:
                    raise ImportError('project namespace differs')
                return spec
        elif project:
            raise ImportError('project module has no source origin')
        return None


if __name__ == '__main__':
    sys.path[:] = [entry for entry in sys.path
                   if not Path(entry or os.getcwd()).resolve().is_relative_to(ROOT)]
    sys.path.append(str(ROOT))
    if any(name == 'scripts' or name.startswith('scripts.') or name == 'looking_glass'
           or name.startswith('looking_glass.') for name in sys.modules):
        raise ImportError('project modules loaded before the source guard')
    sys.meta_path.insert(0, _CheckoutImportGuard())

import torch
from scripts import bert_taste_matched_dose_v1 as previous
from scripts import bert_taste_template_cases_v1 as cases
from scripts import bert_taste_template_geometry_v1 as geometry

if any(Path(module.__file__).resolve() != ROOT / path for module, path in (
        (previous, 'scripts/bert_taste_matched_dose_v1.py'),
        (cases, 'scripts/bert_taste_template_cases_v1.py'),
        (geometry, 'scripts/bert_taste_template_geometry_v1.py'))):
    raise ImportError('project module source origin differs')


def source_identity():
    commit = previous.previous._git('rev-parse', 'HEAD').decode('ascii').strip()
    hashes = {}
    for name in SOURCE_PATHS:
        raw = (ROOT / name).read_bytes()
        if raw != previous.previous._git('show', f'{commit}:{name}'):
            raise RuntimeError('selected source differs from its Git blob')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': commit, 'selected_source_sha256': hashes}


def verify_cases(tokenizer):
    """Retokenize and independently rederive the already-public case selection."""
    manifest = json.loads((ROOT / CASES_PATH).read_bytes())
    if (manifest.get('schema') != 'bert-taste-template-direction-public-cases-v1'
            or manifest.get('model') != cases.MODEL or manifest.get('revision') != cases.REVISION
            or manifest.get('tokenizer_sha256') != cases.TOKENIZER_SHA256
            or manifest.get('prior_public_inventories') != cases.PRIOR
            or manifest.get('tokens') != cases.TOKENS
            or manifest.get('coordinates') != {'mask': 4, 'ingredient': 7, 'the': 1, 'sequence_length': 10}):
        raise ValueError('case manifest identity differs')
    if tokenizer.mask_token_id != 103:
        raise ValueError('mask token differs')
    for word in ('sugar', 'lemon', 'sweet', 'sour'):
        if tokenizer.encode(word, add_special_tokens=False) != [cases.TOKENS[word]]:
            raise ValueError('target word tokenization differs')
    def encode(text):
        data = tokenizer(text, add_special_tokens=True, padding=False,
                         truncation=False, return_attention_mask=True)
        return data['input_ids'], data['attention_mask']
    chosen = cases.select(encode, cases.load_priors())
    if any(manifest.get(key) != value for key, value in chosen.items()):
        raise ValueError('case selection differs from the frozen rule')
    commit, hashes = manifest.get('selection_source_commit'), manifest.get('selection_source_sha256')
    if type(commit) is not str or len(commit) != 40 or type(hashes) is not dict:
        raise ValueError('selection source binding is missing')
    if set(hashes) != set(cases.SELECTION_SOURCE_PATHS):
        raise ValueError('selection source inventory differs')
    for name, expected in hashes.items():
        raw = previous.previous._git('show', f'{commit}:{name}')
        if hashlib.sha256(raw).hexdigest() != expected or raw != (ROOT / name).read_bytes():
            raise ValueError('selection source changed after case selection')
    return manifest


def _tensor(raw):
    if sys.byteorder != 'little':
        raise RuntimeError('saved little-endian float32 requires a little-endian host')
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def _actual_cosine(after, before, donor):
    actual = geometry.displacement(previous._values(after), previous._values(before))
    direction = geometry.displacement(previous._values(donor), previous._values(before))
    return math.fsum(a*b for a, b in zip(actual, direction)) / (geometry.norm(actual) * geometry.norm(direction))


def run_cases(model, rows, *, sweet_id, sour_id, tokenizer, emit):
    """Run fixed crossed cases; inputs may be fabricated for small tests."""
    if type(rows) is not list or not rows or len(rows) % 4 or not callable(emit):
        raise ValueError('expected complete four-row crossed stems')
    baselines, eligibility, effects, doses, summaries = {}, {}, [], [], []
    completed = 0

    def forward(row, layer, position, mode, *, donor=None, patch=None, self_patch=False,
                expected_before=None, donor_clean=None, dose=None, baseline=False):
        nonlocal completed
        identity = {'row_id': row['id'], 'stem': row['stem'], 'template': row['template'],
                    'ingredient': row['ingredient'], 'layer': layer, 'position': position,
                    'mode': mode, 'donor': donor}
        emit({'event': 'forward_start', **identity})
        pre_probe = previous.previous.denormal_probe()
        observed, entry = previous.observe_entry(model, tuple(row['input_ids']), layer=layer,
                                                 position=position, mask_index=4,
                                                 source=patch, self_patch=self_patch)
        try:
            post_probe = previous.previous.denormal_probe()
            if expected_before is not None and not previous._same(observed.before, expected_before):
                raise RuntimeError('same-invocation pre-action differs from its clean capture')
            actual_norm = actual_cosine = None
            if dose is not None:
                actual_norm = geometry.norm(geometry.displacement(previous._values(observed.after),
                                                                   previous._values(observed.before)))
                actual_cosine = _actual_cosine(observed.after, observed.before, donor_clean)
                if actual_norm != dose['delivered_norm'] or actual_cosine != dose['cosine_with_donor_direction']:
                    raise RuntimeError('actual delivered geometry differs from the plan')
            summary = previous._summary(observed.mask_logits, sweet_id, sour_id, tokenizer)
            post_score = previous.previous.denormal_probe()
            emit({'event': 'forward_complete', **identity, 'model_entry': entry, 'summary': summary,
                  'denormal_before': pre_probe, 'denormal_after': post_probe, 'denormal_after_scoring': post_score,
                  'baseline_mask_logits_float32_le_base64': previous._bits(observed.mask_logits) if baseline else None,
                  'before_float32_le_base64': previous._bits(observed.before),
                  'after_float32_le_base64': None if observed.after is None else previous._bits(observed.after),
                  'source_float32_le_base64': None if patch is None else previous._bits(patch),
                  'donor_clean_float32_le_base64': None if donor_clean is None else previous._bits(donor_clean),
                  'full_before_sha256': observed.full_before_sha256, 'full_after_sha256': observed.full_after_sha256,
                  'untouched_positions_equal': observed.untouched_positions_equal,
                  'actual_pre_post_norm': actual_norm, 'actual_donor_cosine': actual_cosine,
                  'dose': None if dose is None else {k: v for k, v in dose.items() if k != 'raw'},
                  'events': list(observed.events)})
        except BaseException as original:
            try:
                emit({'event': 'forward_returned_invalid', **identity, 'model_entry': entry,
                      'exception_type': type(original).__name__,
                      'mask_logits_float32_le_base64': previous._bits(observed.mask_logits),
                      'before_float32_le_base64': previous._bits(observed.before),
                      'after_float32_le_base64': None if observed.after is None else previous._bits(observed.after),
                      'source_float32_le_base64': None if patch is None else previous._bits(patch),
                      'donor_clean_float32_le_base64': None if donor_clean is None else previous._bits(donor_clean)})
            except BaseException as recording_error:
                if (isinstance(recording_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(original, (KeyboardInterrupt, SystemExit))):
                    raise recording_error from original
                raise original from recording_error
            raise
        completed += 1
        return observed, summary

    for offset in range(0, len(rows), 4):
        group = rows[offset:offset+4]
        stem = group[0]['stem']
        if ([r['stem'] for r in group] != [stem] * 4
                or [(r['template'], r['ingredient']) for r in group]
                != [('after', 'sugar'), ('after', 'lemon'), ('with', 'sugar'), ('with', 'lemon')]):
            raise ValueError('crossed row order differs')
        references, captures, full_hash = {}, {}, {}
        for row in group:
            for position, site in ((7, 'ingredient'), (1, 'the')):
                observed, summary = forward(row, 3, position, 'capture_' + site, baseline=(site == 'ingredient'))
                captures[row['template'], row['ingredient'], site] = observed.before
                if site == 'ingredient':
                    references[row['id']] = observed.mask_logits
                    full_hash[row['id']] = observed.full_before_sha256
                    baselines[row['id']] = summary
                elif (not previous._same(references[row['id']], observed.mask_logits)
                      or full_hash[row['id']] != observed.full_before_sha256):
                    raise RuntimeError('clean captures disagree')
        for template in ('after', 'with'):
            eligibility[stem + '-' + template] = previous.eligible(
                baselines[stem + '-' + template + '-sugar'], baselines[stem + '-' + template + '-lemon'],
                sweet_id, sour_id)
        try:
            before_construction = previous.previous.denormal_probe()
            previous.rounding_probe()
            clean = {template: {ingredient: {site: previous._values(captures[template, ingredient, site])
                                            for site in ('ingredient', 'the')}
                                for ingredient in ('sugar', 'lemon')} for template in ('after', 'with')}
            plan = geometry.plan(stem, clean, width=next(iter(captures.values())).numel())
            after_construction = previous.previous.denormal_probe()
            previous.rounding_probe()
        except (ValueError, RuntimeError) as exc:
            emit({'event': 'dose_unavailable', 'stem': stem, 'reason': str(exc),
                  'exception_type': type(exc).__name__})
            raise
        dose_record = {'stem': stem, 'layer': 3, 'radius': plan['radius'],
                       'distances': plan['distances'], 'fractions': plan['fractions'],
                       'denormal_before_construction': before_construction,
                       'denormal_after_construction': after_construction}
        doses.append(dose_record)
        emit({'event': 'dose_plan', **dose_record})
        for row in group:
            template, ingredient = row['template'], row['ingredient']
            other = 'lemon' if ingredient == 'sugar' else 'sugar'
            donor_id = stem + '-' + template + '-' + other
            direction = 'lemon_to_sugar' if ingredient == 'sugar' else 'sugar_to_lemon'
            base = baselines[row['id']]['sweet_minus_sour']
            schedule = (('ingredient_donor', 'ingredient', False, None),
                        ('ingredient_self', 'ingredient', True, None), ('the_self', 'the', True, None),
                        ('ingredient_toward', 'ingredient', False, 'ingredient_toward'),
                        ('ingredient_away', 'ingredient', False, 'ingredient_away'),
                        ('orth_plus', 'ingredient', False, 'orth_plus'), ('orth_minus', 'ingredient', False, 'orth_minus'),
                        ('the_toward', 'the', False, 'the_toward'), ('the_away', 'the', False, 'the_away'))
            for mode, site, self_patch, action_name in schedule:
                donor_clean = None if self_patch else captures[template, other, site]
                action = None if action_name is None else plan['actions'][template][ingredient][action_name]
                patch = None if self_patch else _tensor(action['raw']) if action is not None else donor_clean
                observed, summary = forward(row, 3, 7 if site == 'ingredient' else 1, mode,
                                            donor=row['id'] if self_patch else donor_id,
                                            patch=patch, self_patch=self_patch,
                                            expected_before=captures[template, ingredient, site],
                                            donor_clean=donor_clean, dose=action)
                if self_patch and not previous._same(observed.mask_logits, references[row['id']]):
                    raise RuntimeError('self patch changed mask logits')
                effects.append({'stem': stem, 'template': template, 'recipient': row['id'],
                                'donor': row['id'] if self_patch else donor_id, 'direction': direction,
                                'layer': 3, 'position': 7 if site == 'ingredient' else 1, 'mode': mode,
                                'baseline_margin': base, 'patched_margin': summary['sweet_minus_sour'],
                                'effect': math.fsum((summary['sweet_minus_sour'], -base)), 'summary': summary})
        for row in group:
            observed, _ = forward(row, 3, 7, 'final_clean')
            if (not previous._same(observed.mask_logits, references[row['id']])
                    or not previous._same(observed.before, captures[row['template'], row['ingredient'], 'ingredient'])
                    or observed.full_before_sha256 != full_hash[row['id']]):
                raise RuntimeError('final clean control differs')
        final = {}
        for row in group:
            observed, _ = forward(row, 5, 7, 'final_block_capture')
            if not previous._same(observed.mask_logits, references[row['id']]):
                raise RuntimeError('final-block clean mask differs')
            final[row['template'], row['ingredient']] = observed.before
        for row in group:
            template, ingredient = row['template'], row['ingredient']
            other = 'lemon' if ingredient == 'sugar' else 'sugar'
            donor = final[template, other]
            donor_id = stem + '-' + template + '-' + other
            observed, summary = forward(row, 5, 7, 'final_block_donor', donor=donor_id,
                                        patch=donor, expected_before=final[template, ingredient], donor_clean=donor)
            if not previous._same(observed.mask_logits, references[row['id']]):
                raise RuntimeError('final-block donor changed mask logits')
            effects.append({'stem': stem, 'template': template, 'recipient': row['id'], 'donor': donor_id,
                            'direction': 'lemon_to_sugar' if ingredient == 'sugar' else 'sugar_to_lemon',
                            'layer': 5, 'position': 7, 'mode': 'final_block_donor',
                            'baseline_margin': baselines[row['id']]['sweet_minus_sour'],
                            'patched_margin': summary['sweet_minus_sour'],
                            'effect': math.fsum((summary['sweet_minus_sour'], -baselines[row['id']]['sweet_minus_sour'])),
                            'summary': summary})

    indexed = {(e['stem'], e['template'], e['direction'], e['mode']): e['effect'] for e in effects}
    comparisons = []
    for offset in range(0, len(rows), 4):
        stem = rows[offset]['stem']
        contrasts, specificity = {}, {}
        for template in ('after', 'with'):
            for direction, sign in (('sugar_to_lemon', 1), ('lemon_to_sugar', -1)):
                get = lambda mode: indexed[stem, template, direction, mode]
                contrast = math.fsum((sign * get('ingredient_toward'), -sign * get('the_toward')))
                odd = {site: math.fsum((get(site + '_toward'), -get(site + '_away'))) / 2
                       for site in ('ingredient', 'the')}
                even = {site: math.fsum((get(site + '_toward'), get(site + '_away'))) / 2
                        for site in ('ingredient', 'the')}
                odd['orth'] = math.fsum((get('orth_plus'), -get('orth_minus'))) / 2
                even['orth'] = math.fsum((get('orth_plus'), get('orth_minus'))) / 2
                contrasts[template, direction] = contrast
                specificity[template, direction] = math.fsum((sign * odd['ingredient'], -abs(odd['orth'])))
                comparisons.append({'stem': stem, 'template': template, 'direction': direction,
                                    'eligible_descriptive': eligibility[stem + '-' + template],
                                    'signed_matched_position_contrast': contrast, 'odd': odd, 'even': even,
                                    'donor_odd_minus_abs_orth': specificity[template, direction]})
        after = math.fsum(contrasts['after', d] for d in ('sugar_to_lemon', 'lemon_to_sugar')) / 2
        with_some = math.fsum(contrasts['with', d] for d in ('sugar_to_lemon', 'lemon_to_sugar')) / 2
        differences = {t: math.fsum((contrasts[t, 'sugar_to_lemon'], -contrasts[t, 'lemon_to_sugar']))
                       for t in ('after', 'with')}
        specificity_after = math.fsum(specificity['after', d] for d in ('sugar_to_lemon', 'lemon_to_sugar')) / 2
        specificity_with = math.fsum(specificity['with', d] for d in ('sugar_to_lemon', 'lemon_to_sugar')) / 2
        summaries.append({'stem': stem, 'after': after, 'with': with_some,
                          'template_difference': math.fsum((after, -with_some)),
                          'direction_main': math.fsum((differences['after'], differences['with'])) / 2,
                          'template_direction_interaction': math.fsum((differences['after'], -differences['with'])),
                          'direction_specificity_after': specificity_after, 'direction_specificity_with': specificity_with,
                          'direction_specificity_template_difference': math.fsum((specificity_after, -specificity_with))})
    means = {key: math.fsum(row[key] for row in summaries) / len(summaries)
             for key in ('after', 'with', 'template_difference', 'direction_main', 'template_direction_interaction',
                         'direction_specificity_after', 'direction_specificity_with', 'direction_specificity_template_difference')}
    return {'baselines': baselines, 'eligibility_descriptive': eligibility, 'effects': effects,
            'doses': doses, 'comparisons': comparisons, 'stem_summaries': summaries,
            'means': means, 'completed_forwards': completed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('output must be outside the source checkout')
    before = source_identity()
    output.mkdir(parents=False, exist_ok=False)
    counts = {'started': 0, 'completed': 0}
    phase = 'started_record'
    try:
        previous.previous._write_json(output / 'started.json',
                                      {'status': 'started', 'source': before, 'cases_path': CASES_PATH, 'seed': SEED})
        phase = 'runtime_setup'
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise RuntimeError('denormal preservation is unavailable')
        previous.previous.denormal_probe()
        previous.rounding_probe()
        torch.random.default_generator.manual_seed(SEED)
        random.seed(SEED)
        from looking_glass.bert_agreement_study import load_pinned_distilbert, capture_model_state, verify_model_state
        phase = 'offline_model_load'
        loaded = load_pinned_distilbert(args.cache_dir)
        model, tokenizer = loaded.model, loaded.tokenizer
        phase = 'case_and_exposure_check'
        manifest = verify_cases(tokenizer)
        if len(manifest['selected_stems']) != 6 or len(manifest['rows']) != 24:
            raise ValueError('fixed case count differs')
        previous.previous._git('merge-base', '--is-ancestor', manifest['selection_source_commit'], before['commit'])
        witness = capture_model_state(model)
        phase = 'public_forwards'
        with ExitStack() as stack:
            journals = {stem: stack.enter_context((output / (stem + '.jsonl')).open('xb'))
                        for stem in manifest['selected_stems']}
            def emit(record):
                raw = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
                stream = journals[record['stem']]
                previous.previous.write_all(stream, raw)
                stream.flush()
                if record['event'] == 'forward_start':
                    counts['started'] += 1
                elif record['event'] == 'forward_complete':
                    counts['completed'] += 1
            result = run_cases(model, manifest['rows'], sweet_id=4086, sour_id=14768,
                               tokenizer=tokenizer, emit=emit)
            for stream in journals.values():
                stream.flush()
                os.fsync(stream.fileno())
        files = {}
        for stem in manifest['selected_stems']:
            file = output / (stem + '.jsonl')
            raw = file.read_bytes()
            files[file.name] = {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        if counts != {'started': 336, 'completed': 336} or result['completed_forwards'] != 336:
            raise RuntimeError('forward inventory is incomplete')
        phase = 'final_checks'
        verify_model_state(model, witness)
        if previous.previous.has_hooks(model) or torch.cuda.is_initialized():
            raise RuntimeError('model cleanup or CPU-only state is uncertain')
        after = source_identity()
        if before != after:
            raise RuntimeError('selected source changed during execution')
        result.update({'status': 'complete', 'schema': 'bert-taste-template-direction-result-v1',
                       'source': before, 'selection_source_commit': manifest['selection_source_commit'],
                       'model': loaded.metadata, 'seed': SEED, 'journals': files,
                       'model_state_sha256_before_after': witness.sha256,
                       'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                   'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                                   'default_dtype': str(torch.get_default_dtype()),
                                   'mkldnn_enabled': torch.backends.mkldnn.enabled,
                                   'torch_build_sha256': hashlib.sha256(torch.__config__.show().encode('utf-8')).hexdigest(),
                                   'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
                                   'mkl_num_threads': os.environ.get('MKL_NUM_THREADS'),
                                   'mkl_cbwr': os.environ.get('MKL_CBWR'),
                                   'denormal_probe': previous.previous.denormal_probe(),
                                   'rounding_probe': previous.rounding_probe(),
                                   'cuda_initialized': torch.cuda.is_initialized()},
                       'limitations': ['Public convenience frames; no blind, pretraining or population claim.',
                           'Whole-output untouched coordinates and patched full-vocabulary summaries are owner-observed.',
                           'Source hashes and direct-process observation do not authenticate execution or whole-tree cleanup.']})
        previous.previous._write_json(output / 'result.json', result)
        return 0
    except BaseException as exc:
        previous.previous.record_incomplete(output, phase, counts, exc)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
