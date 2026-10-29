"""CPU taste-context propagation study; no downloads or hidden inputs."""
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
CASES_PATH = 'experiments/bert_taste_propagation_public_cases_v1.json'
SEED = 20261102
SOURCE_PATHS = (
    'scripts/bert_taste_propagation_v1.py', 'scripts/bert_taste_propagation_cases_v1.py',
    'scripts/bert_taste_propagation_geometry_v1.py', 'scripts/bert_taste_propagation_metrics_v1.py',
    'scripts/bert_taste_propagation_observer_v1.py', 'scripts/bert_taste_template_geometry_v1.py',
    'docs/BERT_TASTE_CONTEXT_PROPAGATION_V1.md', CASES_PATH,
    'scripts/bert_taste_matched_dose_v1.py', 'scripts/bert_taste_patch_v1.py',
    'experiments/bert_agreement_exposures_v6.json', 'experiments/bert_matched_public_cases_v1.json',
    'experiments/bert_taste_public_exposures_v2.json',
    'experiments/bert_taste_matched_dose_public_exposures_v2.json',
    'experiments/bert_taste_matched_dose_public_cases_v1.json',
    'experiments/bert_taste_template_direction_public_exposures_v2.json',
    'experiments/bert_taste_template_direction_public_cases_v1.json',
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
from scripts import bert_taste_propagation_cases_v1 as cases
from scripts import bert_taste_propagation_geometry_v1 as geometry
from scripts import bert_taste_propagation_metrics_v1 as metrics
from scripts import bert_taste_propagation_observer_v1 as observer

if any(Path(module.__file__).resolve() != ROOT / path for module, path in (
        (previous, 'scripts/bert_taste_matched_dose_v1.py'),
        (cases, 'scripts/bert_taste_propagation_cases_v1.py'),
        (geometry, 'scripts/bert_taste_propagation_geometry_v1.py'),
        (metrics, 'scripts/bert_taste_propagation_metrics_v1.py'),
        (observer, 'scripts/bert_taste_propagation_observer_v1.py'))):
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
    if (manifest.get('schema') != 'bert-taste-context-propagation-public-cases-v1'
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


def _encoded_f64(vector):
    return base64.b64encode(b''.join(struct.pack('<d', value) for value in vector)).decode('ascii')


def _specificity(lexical, first, second):
    if any(value is None for value in (lexical, first, second)):
        return None
    return math.fsum((lexical, -math.fsum((abs(first), abs(second))) / 2))


def _aggregate(cells, stems, groups, fields):
    """Propagate missing required values rather than averaging a subset."""
    result = {}
    for group in groups:
        chosen = [cell for cell in cells if cell['group'] == group]
        result[group] = {}
        for field in fields:
            by_stem = {}
            for stem in stems:
                by_stem[stem] = {}
                for template in ('after', 'with'):
                    values = [cell[field] for cell in chosen if cell['stem'] == stem and cell['template'] == template]
                    if len(values) != 2:
                        raise RuntimeError('aggregate recipient inventory differs')
                    by_stem[stem][template] = metrics.complete_mean(values)
                a, b = by_stem[stem]['after']['value'], by_stem[stem]['with']['value']
                by_stem[stem]['difference'] = (None if a is None or b is None else math.fsum((a, -b)))
            means = {template: metrics.complete_mean([by_stem[stem][template]['value'] for stem in stems])
                     for template in ('after', 'with')}
            means['difference'] = metrics.complete_mean([by_stem[stem]['difference'] for stem in stems])
            result[group][field] = {'stems': by_stem, 'means': means}
    return result


def run_cases(model, rows, *, sweet_id, sour_id, tokenizer, emit):
    """Run complete four-row crossed stems; fabricated rows are permitted in tests."""
    if type(rows) is not list or not rows or len(rows) % 4 or not callable(emit):
        raise ValueError('expected complete four-row crossed stems')
    baselines, eligibility, effects, doses, cells, propagation = {}, {}, [], [], [], []
    completed = 0

    def forward(row, layer, position, mode, *, donor=None, patch=None, self_patch=False,
                capture_late=False, expected_before=None, dose=None, full_logits=False, reference=None):
        nonlocal completed
        identity = {'row_id': row['id'], 'stem': row['stem'], 'template': row['template'],
                    'ingredient': row['ingredient'], 'layer': layer, 'position': position,
                    'mode': mode, 'donor': donor}
        emit({'event': 'forward_start', **identity})
        before_probe = previous.previous.denormal_probe()
        binary64_before = metrics.binary64_probe()
        observed = observer.observe(model, tuple(row['input_ids']), layer=layer, position=position,
                                    mask_index=4, source=patch, self_patch=self_patch,
                                    capture_late=capture_late)
        try:
            after_probe = previous.previous.denormal_probe()
            if expected_before is not None and not previous._same(observed.before, expected_before):
                raise RuntimeError('same-invocation pre-action differs from its clean capture')
            if reference is not None:
                if not previous._same(observed.mask_logits, reference.mask_logits):
                    raise RuntimeError(f'{mode} changed clean mask logits')
                if mode == 'fresh_clean' and not previous._same(observed.before, reference.before):
                    raise RuntimeError('fresh clean early capture differs')
                if (mode == 'fresh_clean' or (mode == 'self' and layer == 0)) and any(
                        not previous._same(observed.later[key], reference.later[key])
                        for key in observer.LATE_POSITIONS):
                    raise RuntimeError(f'{mode} changed a clean later capture')
            actual_norm = None
            if dose is not None:
                actual = geometry.base.displacement(previous._values(observed.after), previous._values(observed.before))
                actual_norm = geometry.base.norm(actual)
                if actual_norm != dose['delivered_norm']:
                    raise RuntimeError('actual displacement differs from the planned dose')
            summary = previous._summary(observed.mask_logits, sweet_id, sour_id, tokenizer)
            score_probe = previous.previous.denormal_probe()
            binary64_after = metrics.binary64_probe()
            emit({'event': 'forward_complete', **identity, 'model_entry': observed.model_entry,
                  'summary': summary, 'denormal_before': before_probe, 'denormal_after': after_probe,
                  'denormal_after_scoring': score_probe,
                  'binary64_before': binary64_before, 'binary64_after_scoring': binary64_after,
                  'mask_logits_float32_le_base64': previous._bits(observed.mask_logits) if full_logits else None,
                  'before_float32_le_base64': previous._bits(observed.before),
                  'after_float32_le_base64': None if observed.after is None else previous._bits(observed.after),
                  'source_float32_le_base64': None if patch is None else previous._bits(patch),
                  'later_float32_le_base64': {key: previous._bits(value) for key, value in observed.later.items()},
                  'late_full_sha256': observed.late_full_sha256,
                  'full_before_sha256': observed.full_before_sha256, 'full_after_sha256': observed.full_after_sha256,
                  'untouched_positions_equal': observed.untouched_positions_equal, 'actual_pre_post_norm': actual_norm,
                  'dose': None if dose is None else {k: v for k, v in dose.items() if k != 'raw'},
                  'events': list(observed.events)})
        except BaseException as original:
            try:
                emit({'event': 'forward_returned_invalid', **identity, 'model_entry': observed.model_entry,
                      'exception_type': type(original).__name__, 'mask_logits_float32_le_base64': previous._bits(observed.mask_logits),
                      'before_float32_le_base64': previous._bits(observed.before),
                      'after_float32_le_base64': None if observed.after is None else previous._bits(observed.after),
                      'source_float32_le_base64': None if patch is None else previous._bits(patch),
                      'later_float32_le_base64': {k: previous._bits(v) for k, v in observed.later.items()}})
            except BaseException as recording_error:
                if (isinstance(recording_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(original, (KeyboardInterrupt, SystemExit))):
                    raise recording_error from original
                raise original from recording_error
            raise
        completed += 1
        return observed, summary

    stems = []
    for offset in range(0, len(rows), 4):
        group = rows[offset:offset+4]
        stem = group[0]['stem']
        if ([r['stem'] for r in group] != [stem] * 4 or stem in stems
                or [(r['template'], r['ingredient']) for r in group]
                != [('after', 'sugar'), ('after', 'lemon'), ('with', 'sugar'), ('with', 'lemon')]):
            raise ValueError('crossed row order differs')
        stems.append(stem)
        clean, indexed = {}, {(r['template'], r['ingredient']): r for r in group}
        for row in group:
            observed, summary = forward(row, 0, 7, 'clean_capture', capture_late=True, full_logits=True)
            clean[row['id']] = observed
            baselines[row['id']] = summary
        for template in ('after', 'with'):
            eligibility[stem + '-' + template] = previous.eligible(
                baselines[stem + '-' + template + '-sugar'], baselines[stem + '-' + template + '-lemon'],
                sweet_id, sour_id)
        captures = {}
        for template in ('after', 'with'):
            captures[template] = {}
            for ingredient in ('sugar', 'lemon'):
                item = clean[indexed[template, ingredient]['id']]
                captures[template][ingredient] = {'early_ingredient': previous._values(item.before),
                    **{name: previous._values(value) for name, value in item.later.items()}}
        try:
            pre = previous.previous.denormal_probe()
            binary64_pre = metrics.binary64_probe()
            previous.rounding_probe()
            plan = geometry.plan(stem, captures, width=model.config.dim)
            post = previous.previous.denormal_probe()
            binary64_post = metrics.binary64_probe()
            previous.rounding_probe()
        except (ValueError, RuntimeError) as exc:
            emit({'event': 'dose_unavailable', 'stem': stem, 'exception_type': type(exc).__name__})
            raise
        dose_record = {'stem': stem, 'radius': plan['radius'], 'distances': plan['distances'],
                       'fractions': plan['fractions'], 'mutual_cosines': plan['mutual_cosines'],
                       'denormal_before_construction': pre, 'denormal_after_construction': post,
                       'binary64_before_construction': binary64_pre, 'binary64_after_construction': binary64_post}
        doses.append(dose_record)
        emit({'event': 'dose_plan', **dose_record})
        for row in group:
            template, ingredient = row['template'], row['ingredient']
            other = 'lemon' if ingredient == 'sugar' else 'sugar'
            other_template = 'with' if template == 'after' else 'after'
            lexical_row, context_row = indexed[template, other], indexed[other_template, ingredient]
            recipient, lexical, context = clean[row['id']], clean[lexical_row['id']], clean[context_row['id']]
            sign = 1 if other == 'sugar' else -1
            direction = 'sugar_to_lemon' if sign == 1 else 'lemon_to_sugar'
            metrics.binary64_probe()
            baseline_values = previous._values(recipient.mask_logits)
            donor_values = previous._values(lexical.mask_logits)
            output_direction = metrics.center(geometry.base.displacement(donor_values, baseline_values))
            for site in geometry.SITES:
                layer = 0 if site == 'early_ingredient' else 3
                position = {'early_ingredient': 7, 'late_ingredient': 7, 'late_the': 1, 'late_tasted': 3}[site]
                before = recipient.before if layer == 0 else recipient.later[site]
                action_group = plan['actions'][template][ingredient][site]
                paired = {}
                for axis in ('lex', 'context', 'orth1', 'orth2'):
                    values = []
                    for suffix in ('plus', 'minus'):
                        name = axis + '_' + suffix
                        action = action_group[name]
                        source = _tensor(action['raw'])
                        donor_id = context_row['id'] if axis == 'context' else lexical_row['id']
                        observed, summary = forward(row, layer, position, name, donor=donor_id, patch=source,
                                                    capture_late=(layer == 0), expected_before=before,
                                                    dose=action, full_logits=True)
                        delta = math.fsum((summary['sweet_minus_sour'], -baselines[row['id']]['sweet_minus_sour']))
                        effects.append({'stem': stem, 'template': template, 'recipient': row['id'],
                                        'direction': direction, 'site': site, 'mode': name,
                                        'baseline_margin': baselines[row['id']]['sweet_minus_sour'],
                                        'patched_margin': summary['sweet_minus_sour'], 'effect': delta,
                                        'summary': summary})
                        values.append((observed, summary, delta))
                    plus, minus = values
                    analysis_before = metrics.binary64_probe()
                    odd = math.fsum((plus[2], -minus[2])) / 2
                    even = math.fsum((plus[2], minus[2])) / 2
                    plus_logits, minus_logits = previous._values(plus[0].mask_logits), previous._values(minus[0].mask_logits)
                    odd_logits, _ = metrics.odd_even(plus_logits, minus_logits, baseline_values)
                    output = metrics.projection(metrics.center(odd_logits), output_direction)
                    paired[axis] = {'odd': odd, 'even': even, 'signed_odd': sign * odd,
                                    'tv_plus': metrics.total_variation(plus_logits, baseline_values),
                                    'tv_minus': metrics.total_variation(minus_logits, baseline_values),
                                    'output_projection': output}
                    if layer == 0:
                        for late in observer.LATE_POSITIONS:
                            base_late = previous._values(recipient.later[late])
                            donor_late = previous._values(lexical.later[late])
                            odd_vec, even_vec = metrics.odd_even(previous._values(plus[0].later[late]),
                                                                 previous._values(minus[0].later[late]), base_late)
                            direction_vec = geometry.base.displacement(donor_late, base_late)
                            paired[axis].setdefault('later', {})[late] = {
                                'odd_float64_le_base64': _encoded_f64(odd_vec),
                                'even_float64_le_base64': _encoded_f64(even_vec),
                                'projection': metrics.projection(odd_vec, direction_vec)}
                    paired[axis]['binary64_before_analysis'] = analysis_before
                    paired[axis]['binary64_after_analysis'] = metrics.binary64_probe()
                self_observed, self_summary = forward(row, layer, position, 'self', donor=row['id'],
                    self_patch=True, capture_late=(layer == 0), expected_before=before, reference=recipient)
                effects.append({'stem': stem, 'template': template, 'recipient': row['id'], 'direction': direction,
                                'site': site, 'mode': 'self', 'baseline_margin': baselines[row['id']]['sweet_minus_sour'],
                                'patched_margin': self_summary['sweet_minus_sour'], 'effect': 0., 'summary': self_summary})
                cell = {'stem': stem, 'template': template, 'recipient': row['id'], 'direction': direction,
                        'group': site, 'eligible_descriptive': eligibility[stem + '-' + template], 'axes': paired}
                cell['taste_specificity'] = _specificity(paired['lex']['signed_odd'], paired['orth1']['odd'], paired['orth2']['odd'])
                cell['context_specificity'] = math.fsum((paired['lex']['signed_odd'], -abs(paired['context']['odd'])))
                for axis, value in paired.items():
                    cell[axis + '_signed_odd'] = value['signed_odd']
                    cell[axis + '_output_projection'] = value['output_projection']['coefficient']
                    cell[axis + '_output_cosine'] = value['output_projection']['cosine']
                    cell[axis + '_output_norm'] = value['output_projection']['vector_norm']
                    cell[axis + '_output_residual'] = value['output_projection']['residual_norm']
                    cell[axis + '_tv_plus'], cell[axis + '_tv_minus'] = value['tv_plus'], value['tv_minus']
                cells.append(cell)
                if layer == 0:
                    for late in observer.LATE_POSITIONS:
                        item = {'stem': stem, 'template': template, 'recipient': row['id'],
                                'direction': direction, 'group': late}
                        for axis, value in paired.items():
                            projection = value['later'][late]['projection']
                            item[axis + '_projection'] = projection['coefficient']
                            item[axis + '_cosine'] = projection['cosine']
                            item[axis + '_norm'] = projection['vector_norm']
                        item['propagation_specificity'] = _specificity(item['lex_projection'],
                            item['orth1_projection'], item['orth2_projection'])
                        propagation.append(item)
        for row in group:
            reference = clean[row['id']]
            forward(row, 0, 7, 'fresh_clean', capture_late=True, reference=reference)
        final = {}
        for row in group:
            observed, _ = forward(row, 5, 7, 'final_capture', reference=clean[row['id']])
            final[row['id']] = observed.before
        for row in group:
            other = 'lemon' if row['ingredient'] == 'sugar' else 'sugar'
            donor = indexed[row['template'], other]
            forward(row, 5, 7, 'final_donor', donor=donor['id'], patch=final[donor['id']],
                    expected_before=final[row['id']], reference=clean[row['id']])

    cell_fields = ['taste_specificity', 'context_specificity']
    for axis in ('lex', 'context', 'orth1', 'orth2'):
        cell_fields.extend(axis + suffix for suffix in ('_signed_odd', '_output_projection', '_output_cosine',
                                                        '_output_norm', '_output_residual', '_tv_plus', '_tv_minus'))
    prop_fields = ['propagation_specificity']
    for axis in ('lex', 'context', 'orth1', 'orth2'):
        prop_fields.extend(axis + suffix for suffix in ('_projection', '_cosine', '_norm'))
    analysis_before = metrics.binary64_probe()
    result = {'baselines': baselines, 'eligibility_descriptive': eligibility, 'doses': doses, 'effects': effects,
            'comparisons': cells, 'propagation': propagation,
            'taste_aggregates': _aggregate(cells, stems, geometry.SITES, cell_fields),
            'propagation_aggregates': _aggregate(propagation, stems, observer.LATE_POSITIONS, prop_fields),
            'completed_forwards': completed}
    result['binary64_before_aggregation'] = analysis_before
    result['binary64_after_aggregation'] = metrics.binary64_probe()
    return result


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
        metrics.binary64_probe()
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
        if counts != {'started': 960, 'completed': 960} or result['completed_forwards'] != 960:
            raise RuntimeError('forward inventory is incomplete')
        phase = 'final_checks'
        verify_model_state(model, witness)
        if previous.previous.has_hooks(model) or torch.cuda.is_initialized():
            raise RuntimeError('model cleanup or CPU-only state is uncertain')
        after = source_identity()
        if before != after:
            raise RuntimeError('selected source changed during execution')
        result.update({'status': 'complete', 'schema': 'bert-taste-context-propagation-result-v1',
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
                                   'binary64_probe': metrics.binary64_probe(),
                                   'rounding_probe': previous.rounding_probe(),
                                   'cuda_initialized': torch.cuda.is_initialized()},
                       'limitations': ['Public convenience frames; no blind, pretraining or population claim.',
                           'Whole-output untouched coordinates are owner-observed.',
                           'Source hashes and direct-process observation do not authenticate execution or whole-tree cleanup.']})
        previous.previous._write_json(output / 'result.json', result)
        return 0
    except BaseException as exc:
        previous.previous.record_incomplete(output, phase, counts, exc)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
