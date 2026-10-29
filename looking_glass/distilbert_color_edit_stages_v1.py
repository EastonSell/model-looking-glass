"""Frozen screen, development and evaluation schedules for masked-color edits."""
import base64
import hashlib
import json
import math
import sys

from . import distilbert_color_edit_metrics_v1 as metrics
from . import distilbert_color_edit_records_v1 as records
from .distilbert_color_edit_action_v1 import tensor_bytes

FRACTIONS = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016)
RANDOM_SEED = 20261121


def case(manifest, obj, role):
    rows = [row for row in manifest['cases'] if row['object'] == obj and row['role'] == role]
    if len(rows) != 1:
        raise ValueError('the fixed case role must occur once')
    return rows[0]


def screen_rows(manifest):
    order = manifest['candidate_order']
    if order != ['banana', 'strawberry', 'lemon']:
        raise ValueError('candidate order differs from the public protocol')
    return [case(manifest, obj, role) for obj in order
            for role in ('discovery', 'calibration-01', 'calibration-02')]


def evaluate_rows(manifest, obj):
    target = [case(manifest, obj, 'heldout-%02d' % number) for number in range(1, 5)]
    same = [row for row in manifest['cases'] if row['object'] == obj and row['role'] == 'same_object_locality']
    shared = [row for row in manifest['cases'] if row['role'] in ('other_color', 'unrelated')]
    if len(target) != 4 or len(same) != 2 or len(shared) != 10:
        raise ValueError('the fixed evaluation inventory is incomplete')
    return target, same + shared


def tensor_row(tensor):
    return list(records.decode_raw(records.encode_raw(tensor_bytes(tensor)), tensor.numel()))


def scaled_delta(direction, desired):
    import torch
    if (not isinstance(direction, torch.Tensor) or not math.isfinite(desired) or desired <= 0 or
            not bool(torch.isfinite(direction).all())):
        raise ValueError('finite nonzero direction and requested norm are required')
    direction = direction.detach().to(dtype=torch.float64, device='cpu')
    norm = float(torch.linalg.vector_norm(direction).item())
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('zero or nonfinite direction')
    delta = (direction * (desired / norm)).to(torch.float32).contiguous()
    if not bool(torch.isfinite(delta).all()) or not bool(torch.any(delta != 0)):
        raise ValueError('direction cannot be represented as a finite nonzero f32 update')
    return delta


def _delta_sha(delta):
    return hashlib.sha256(tensor_bytes(delta)).hexdigest()


def require_unchanged_other_logits(original, bias, purple, factual):
    """The two-entry bias edit must leave every other raw f32 logit unchanged."""
    if (type(original) is not bytes or type(bias) is not bytes or len(original) != len(bias) or
            len(original) % 4 or type(purple) is not int or type(factual) is not int or
            purple == factual or not 0 <= purple < len(original) // 4 or not 0 <= factual < len(original) // 4):
        raise ValueError('invalid bias comparison bytes or IDs')
    previous = 0
    for index in sorted((purple, factual)):
        if original[previous:4 * index] != bias[previous:4 * index]:
            raise ValueError('bias action changed an unmodified vocabulary logit')
        previous = 4 * (index + 1)
    if original[previous:] != bias[previous:]:
        raise ValueError('bias action changed an unmodified vocabulary logit')


def _norm_matches(realized, planned):
    return math.isfinite(realized) and 0.999 * planned <= realized <= 1.001 * planned


def _random_direction(hidden, intermediate):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(RANDOM_SEED)
    output = torch.randn(hidden, dtype=torch.float64, generator=generator)
    input_ = torch.randn(intermediate, dtype=torch.float64, generator=generator)
    direction = torch.outer(output, input_)
    if not bool(torch.isfinite(direction).all()) or not bool(torch.any(direction != 0)):
        raise ValueError('fixed random direction is unavailable')
    return output, input_, direction


def _f64_envelope(tensor):
    import torch
    if (sys.byteorder != 'little' or tensor.dtype != torch.float64 or
            tensor.device.type != 'cpu' or not tensor.is_contiguous()):
        raise ValueError('a contiguous CPU f64 factor is required')
    metrics.require_denormals()
    raw = tensor.detach().view(torch.uint8).numpy().tobytes()
    metrics.require_denormals()
    return {'encoding': 'f64le', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'data': base64.b64encode(raw).decode('ascii')}


def run_screen(engine, journal, manifest):
    for row in screen_rows(manifest):
        engine.run(journal, row, phase='screen', condition='original')


def run_development(engine, journal, manifest, screen_events):
    import torch
    screen = records.rebuild_screen(manifest, screen_events, vocabulary=engine.vocabulary)
    if screen['status'] != 'eligible':
        raise ValueError('a complete eligible public screen is required')
    obj = screen['selected_object']
    discovery = case(manifest, obj, 'discovery')
    calibration = [case(manifest, obj, role) for role in ('calibration-01', 'calibration-02')]
    screen_by_id = {events['case_id']: records.decode_raw(events['logits'], engine.vocabulary)
                    for events in screen_events if events['kind'] == 'complete'}
    screen_raw = {event['case_id']: records.raw_f32_bytes(event['logits'], engine.vocabulary)
                  for event in screen_events if event['kind'] == 'complete'}
    original = screen_by_id[discovery['id']]
    purple = discovery['word_ids'][discovery['counterfactual']]
    factual = discovery['word_ids'][discovery['factual']]
    blue = discovery['word_ids'][discovery['wrong_target']]
    clean = engine.run(journal, discovery, phase='development', condition='clean-site', capture_site=True)
    original_envelope = next(event['logits'] for event in screen_events
                             if event['kind'] == 'complete' and event['case_id'] == discovery['id'])
    if records.encode_raw(tensor_bytes(clean['logits'])) != original_envelope:
        raise RuntimeError('screen and clean discovery bytes differ')
    primary_grad = engine.run(journal, discovery, phase='development', condition='primary-gradient',
                              gradient_targets=(purple, factual), capture_site=True)
    if (tensor_bytes(clean['logits']) != tensor_bytes(primary_grad['logits']) or
            tensor_bytes(clean['site']['input']) != tensor_bytes(primary_grad['site']['input']) or
            tensor_bytes(clean['site']['output']) != tensor_bytes(primary_grad['site']['output'])):
        raise RuntimeError('clean and gradient discovery witnesses differ')
    weight_norm = float(torch.linalg.vector_norm(engine.weight_original.to(torch.float64)).item())
    if not math.isfinite(weight_norm) or weight_norm == 0:
        raise RuntimeError('original FFN weight norm is unavailable')
    grid = []
    for index, fraction in enumerate(FRACTIONS):
        planned = fraction * weight_norm
        delta = scaled_delta(primary_grad['gradient'], planned)
        result = engine.run(journal, discovery, phase='development', condition='primary-grid-%02d' % index,
                            delta=delta, parameter='weight')
        if not _norm_matches(result['realized_norm'], planned):
            raise RuntimeError('primary grid realized norm differs from plan')
        info = metrics.describe(tensor_row(result['logits']), (purple, factual))
        grid.append({'fraction': fraction, 'planned_norm': planned, 'realized_norm': result['realized_norm'],
                     'delta_sha256': _delta_sha(delta), 'parameter_sha256': result['parameter_sha256'],
                     'purple_unique_top': info['unique_top'] == purple, 'logits': result['logits']})
    chosen = next((index for index, row in enumerate(grid) if row['purple_unique_top']), len(grid) - 1)
    chosen_grid = grid[chosen]
    primary_delta = scaled_delta(primary_grad['gradient'], chosen_grid['planned_norm'])
    if _delta_sha(primary_delta) != chosen_grid['delta_sha256']:
        raise RuntimeError('chosen primary delta is not reproducible')
    wrong_grad = engine.run(journal, discovery, phase='development', condition='wrong-gradient',
                            gradient_targets=(blue, factual), capture_site=True)
    if (tensor_bytes(clean['logits']) != tensor_bytes(wrong_grad['logits']) or
            tensor_bytes(clean['site']['input']) != tensor_bytes(wrong_grad['site']['input']) or
            tensor_bytes(clean['site']['output']) != tensor_bytes(wrong_grad['site']['output'])):
        raise RuntimeError('wrong-target gradient baseline differs')
    output_factor, input_factor, random = _random_direction(engine.hidden, engine.intermediate)
    journal.emit({'kind': 'control-factors', 'phase': 'development', 'seed': RANDOM_SEED,
                  'output_factor': _f64_envelope(output_factor), 'input_factor': _f64_envelope(input_factor)})
    control_norm = chosen_grid['realized_norm']
    directions = {'opposite': -primary_grad['gradient'], 'random': random,
                  'wrong-target': wrong_grad['gradient']}
    deltas = {'primary': primary_delta}
    for name, direction in directions.items():
        deltas[name] = scaled_delta(direction, control_norm)
    controls = {}
    for name in ('opposite', 'random', 'wrong-target'):
        controls[name] = []
        for row in (discovery, *calibration):
            result = engine.run(journal, row, phase='development', condition=name,
                                delta=deltas[name], parameter='weight')
            if not _norm_matches(result['realized_norm'], control_norm):
                raise RuntimeError('required control norm is unavailable')
            controls[name].append(result)
    primary_calibration = []
    for row in calibration:
        result = engine.run(journal, row, phase='development', condition='primary',
                            delta=primary_delta, parameter='weight')
        if (result['parameter_sha256'] != chosen_grid['parameter_sha256'] or
                result['realized_norm'] != chosen_grid['realized_norm']):
            raise RuntimeError('primary edit differs from the frozen discovery edit')
        primary_calibration.append(result)
    journal.emit({'kind': 'bias-origin', 'phase': 'development',
                  'decoder_bias': records.encode_raw(tensor_bytes(engine.bias_original)),
                  'model_state_sha256': engine.state.sha256})
    primary_discovery = tensor_row(chosen_grid['logits'])
    discovery_effect = metrics.effect(original, primary_discovery, purple, factual)
    bias_available = False
    bias_delta = None
    bias_discovery = None
    bias_calibration = []
    match_rows = []
    bias_status = 'below-discovery-gain-floor'
    if discovery_effect['margin_gain'] >= 0.1 and discovery_effect['logp_gain'] >= 0.025:
        try:
            p_delta, f_delta = metrics.solve_bias(original, primary_discovery, purple, factual)
            bias_delta = torch.zeros_like(engine.bias_original)
            bias_delta[purple], bias_delta[factual] = p_delta, f_delta
            if (not bool(torch.isfinite(bias_delta).all()) or not bool(torch.any(bias_delta != 0)) or
                    torch.equal(engine.bias_original + bias_delta, engine.bias_original)):
                bias_delta = None
                bias_status = 'rounding-unavailable'
        except (OverflowError, RuntimeError, ValueError):
            bias_delta = None
            bias_status = 'solve-unavailable'
        if bias_delta is not None:
            bias_discovery = engine.run(journal, discovery, phase='development', condition='bias',
                                        delta=bias_delta, parameter='bias')
            require_unchanged_other_logits(screen_raw[discovery['id']], tensor_bytes(bias_discovery['logits']), purple, factual)
            replay = tensor_row(bias_discovery['logits'])
            target_margin = primary_discovery[purple] - primary_discovery[factual]
            actual_margin = replay[purple] - replay[factual]
            target_logp = metrics.log_probabilities(primary_discovery)[purple]
            actual_logp = metrics.log_probabilities(replay)[purple]
            bias_available = abs(actual_margin - target_margin) <= 0.001 and abs(actual_logp - target_logp) <= 0.001
            bias_status = 'available' if bias_available else 'replay-mismatch'
            if bias_available:
                for row, primary in zip(calibration, primary_calibration):
                    bias_result = engine.run(journal, row, phase='development', condition='bias',
                                             delta=bias_delta, parameter='bias')
                    bias_calibration.append(bias_result)
                    factual_id = row['word_ids'][row['factual']]
                    purple_id = row['word_ids'][row['counterfactual']]
                    require_unchanged_other_logits(screen_raw[row['id']], tensor_bytes(bias_result['logits']), purple_id, factual_id)
                    p_effect = metrics.effect(screen_by_id[row['id']], tensor_row(primary['logits']), purple_id, factual_id)
                    b_effect = metrics.effect(screen_by_id[row['id']], tensor_row(bias_result['logits']), purple_id, factual_id)
                    match_rows.append({'case_id': row['id'], 'primary': p_effect, 'bias': b_effect,
                                       'matched': metrics.row_matches(p_effect, b_effect)})
    summary = {'selected_object': obj, 'purple_id': purple, 'factual_id': factual, 'blue_id': blue,
               'weight_norm': weight_norm, 'grid': [{k: v for k, v in row.items() if k != 'logits'} for row in grid],
               'chosen_grid_index': chosen, 'discovery_top_achieved': chosen_grid['purple_unique_top'],
               'discovery_effect': discovery_effect, 'bias_available': bias_available,
               'bias_status': bias_status,
               'bias_attempted': bias_discovery is not None,
               'bias_delta': ([float(bias_delta[purple]), float(bias_delta[factual])]
                              if bias_delta is not None else None),
               'bias_delta_sha256': _delta_sha(bias_delta) if bias_delta is not None else None,
               'calibration_matches': match_rows,
               'calibration_match_available': bool(bias_available and len(match_rows) == 2 and
                                                   all(row['matched'] for row in match_rows)),
               'weight_delta_sha256': {name: _delta_sha(delta) for name, delta in deltas.items()},
               'control_realized_norm': control_norm}
    return summary


def _completed_actions(manifest, events, phase, vocabulary):
    """Reject incomplete actions and bind both sides to the fixed manifest."""
    by_id = {row['id']: row for row in manifest['cases']}
    if len(by_id) != len(manifest['cases']):
        raise ValueError('duplicate fixed case IDs')
    actions, extras, position = [], [], 0
    for seq, event in enumerate(events):
        if type(event.get('seq')) is not int or event['seq'] != seq:
            raise ValueError('invalid phase sequence')
    while position < len(events):
        start = events[position]
        if start.get('kind') in ('control-factors', 'bias-origin') and phase == 'development':
            extras.append(start)
            position += 1
            continue
        if position + 1 >= len(events):
            raise ValueError('phase has an incomplete action')
        done = events[position + 1]
        if start.get('kind') != 'start' or done.get('kind') != 'complete':
            raise ValueError('phase has an incomplete or unknown action')
        case_id = start.get('case_id')
        if type(case_id) is not str or case_id not in by_id:
            raise ValueError('unknown action case')
        row = by_id[case_id]
        binding = {'action': len(actions), 'phase': phase, 'case_id': case_id,
                   'object': row['object'], 'condition': start.get('condition'),
                   'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
                   'mask_index': row['mask_index']}
        if (type(start.get('condition')) is not str or not start['condition'] or
                any(type(event.get('action')) is not int or type(event.get('mask_index')) is not int
                    or any(type(event.get(k)) is not list or any(type(x) is not int for x in event[k])
                           for k in ('input_ids', 'attention_mask')) for event in (start, done)) or
                any(event.get(key) != value for event in (start, done) for key, value in binding.items())):
            raise ValueError('action source, identity or input binding differs')
        records.decode_raw(done.get('logits'), vocabulary)
        actions.append(done)
        position += 2
    return actions, extras


def _reconstruct_tensor(envelope, shape):
    import torch
    count = math.prod(shape)
    raw = records.raw_f32_bytes(envelope, count)
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError('saved tensor contains nonfinite f32 values')
    metrics.require_denormals()
    return tensor


def _expected_parameter(engine, delta, parameter):
    import torch
    original = engine.weight_original if parameter == 'weight' else engine.bias_original
    realized = (original + delta).contiguous()
    difference = realized.to(torch.float64) - original.to(torch.float64)
    return hashlib.sha256(tensor_bytes(realized)).hexdigest(), float(torch.linalg.vector_norm(difference).item())


def _check_parameter(event, engine, delta, parameter, desired=None):
    sha, norm = _expected_parameter(engine, delta, parameter)
    if (event.get('parameter') != parameter or event.get('parameter_sha256') != sha or
            type(event.get('realized_norm')) not in (int, float) or event['realized_norm'] != norm or
            not math.isfinite(norm) or norm == 0 or (desired is not None and not _norm_matches(norm, desired))):
        raise ValueError('saved realized parameter change differs from its reconstructed plan')
    return sha, norm


def rebuild_development(engine, manifest, screen_events, development_events):
    """Rebuild all edit choices and calibration matches from exact saved rows."""
    import torch
    screen = records.rebuild_screen(manifest, screen_events, vocabulary=engine.vocabulary)
    if screen['status'] != 'eligible':
        raise ValueError('an eligible completed screen is required')
    obj = screen['selected_object']
    discovery = case(manifest, obj, 'discovery')
    calibration = [case(manifest, obj, role) for role in ('calibration-01', 'calibration-02')]
    prior = {event['case_id']: records.decode_raw(event['logits'], engine.vocabulary)
             for event in screen_events if event['kind'] == 'complete'}
    actions, extras = _completed_actions(manifest, development_events, 'development', engine.vocabulary)
    factors = [event for event in extras if event.get('kind') == 'control-factors']
    biases = [event for event in extras if event.get('kind') == 'bias-origin']
    if (len(extras) != 2 or len(factors) != 1 or len(biases) != 1 or
            factors[0].get('phase') != 'development' or biases[0].get('phase') != 'development' or
            type(factors[0].get('seed')) is not int or factors[0]['seed'] != RANDOM_SEED or
            len(actions) < 20 or factors[0]['seq'] != actions[8]['seq'] + 1 or
            biases[0]['seq'] != actions[19]['seq'] + 1 or
            biases[0].get('decoder_bias') != records.encode_raw(tensor_bytes(engine.bias_original)) or
            biases[0].get('model_state_sha256') != engine.state.sha256):
        raise ValueError('fixed random control witness is missing')
    output_factor, input_factor, random = _random_direction(engine.hidden, engine.intermediate)
    if (factors[0].get('output_factor') != _f64_envelope(output_factor) or
            factors[0].get('input_factor') != _f64_envelope(input_factor)):
        raise ValueError('fixed random factors differ')
    base_schedule = [('clean-site', discovery), ('primary-gradient', discovery)]
    base_schedule += [('primary-grid-%02d' % i, discovery) for i in range(len(FRACTIONS))]
    base_schedule += [('wrong-gradient', discovery)]
    base_schedule += [(name, row) for name in ('opposite', 'random', 'wrong-target')
                      for row in (discovery, *calibration)]
    base_schedule += [('primary', row) for row in calibration]
    if len(actions) < len(base_schedule):
        raise ValueError('development is missing required actions')
    if any(event['condition'] != condition or event['case_id'] != row['id']
           for event, (condition, row) in zip(actions, base_schedule)):
        raise ValueError('development schedule differs')
    clean, primary_grad = actions[0], actions[1]
    for event in (clean, primary_grad, actions[8]):
        if (not {'parameter', 'parameter_sha256', 'realized_norm'}.issubset(event) or
                event.get('parameter') is not None or event.get('parameter_sha256') is not None or
                type(event.get('realized_norm')) is not float or event['realized_norm'] != 0.0):
            raise ValueError('clean or gradient action reports a parameter update')
    p, f, b = (discovery['word_ids'][discovery[key]] for key in ('counterfactual', 'factual', 'wrong_target'))
    screen_discovery = next(event for event in screen_events if event['kind'] == 'complete'
                            and event['case_id'] == discovery['id'])
    if (clean['logits'] != screen_discovery['logits'] or clean['logits'] != primary_grad['logits'] or
            clean.get('site_input') != primary_grad.get('site_input') or
            clean.get('site_output') != primary_grad.get('site_output') or
            primary_grad.get('gradient_targets') != [p, f] or
            any(type(value) is not int for value in primary_grad.get('gradient_targets', ()))):
        raise ValueError('primary gradient did not share the clean discovery state')
    primary = _reconstruct_tensor(primary_grad.get('gradient'), (engine.hidden, engine.intermediate))
    site_input = _reconstruct_tensor(primary_grad.get('site_input'), (engine.intermediate,))
    _reconstruct_tensor(primary_grad.get('site_output'), (engine.hidden,))
    site_grad = _reconstruct_tensor(primary_grad.get('site_output_gradient'), (engine.hidden,))
    norm = float(torch.linalg.vector_norm(primary.to(torch.float64)).item())
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('saved primary gradient is unavailable')
    residual = float(torch.linalg.vector_norm(primary.to(torch.float64) -
                     torch.outer(site_grad.to(torch.float64), site_input.to(torch.float64))).item()) / norm
    if residual > 1e-5 or primary_grad.get('relative_rank_one_residual') != residual:
        raise ValueError('saved primary gradient reconstruction differs')
    weight_norm = float(torch.linalg.vector_norm(engine.weight_original.to(torch.float64)).item())
    grid = []
    for i, fraction in enumerate(FRACTIONS):
        event = actions[2 + i]
        desired = fraction * weight_norm
        delta = scaled_delta(primary, desired)
        sha, realized = _check_parameter(event, engine, delta, 'weight', desired)
        info = metrics.describe(records.decode_raw(event['logits'], engine.vocabulary), (p, f))
        grid.append({'fraction': fraction, 'planned_norm': desired, 'realized_norm': realized,
                     'delta_sha256': _delta_sha(delta), 'parameter_sha256': sha,
                     'purple_unique_top': info['unique_top'] == p})
    chosen = next((i for i, row in enumerate(grid) if row['purple_unique_top']), len(grid) - 1)
    primary_delta = scaled_delta(primary, grid[chosen]['planned_norm'])
    wrong_event = actions[8]
    if (wrong_event['logits'] != clean['logits'] or wrong_event.get('site_input') != clean.get('site_input') or
            wrong_event.get('site_output') != clean.get('site_output') or wrong_event.get('gradient_targets') != [b, f] or
            any(type(value) is not int for value in wrong_event.get('gradient_targets', ()))):
        raise ValueError('wrong-target gradient baseline differs')
    wrong = _reconstruct_tensor(wrong_event.get('gradient'), (engine.hidden, engine.intermediate))
    wrong_input = _reconstruct_tensor(wrong_event.get('site_input'), (engine.intermediate,))
    wrong_site_grad = _reconstruct_tensor(wrong_event.get('site_output_gradient'), (engine.hidden,))
    wrong_norm = float(torch.linalg.vector_norm(wrong.to(torch.float64)).item())
    if not math.isfinite(wrong_norm) or wrong_norm == 0:
        raise ValueError('saved wrong-target gradient is unavailable')
    wrong_residual = float(torch.linalg.vector_norm(wrong.to(torch.float64) -
                           torch.outer(wrong_site_grad.to(torch.float64), wrong_input.to(torch.float64))).item()) / wrong_norm
    if wrong_residual > 1e-5 or wrong_event.get('relative_rank_one_residual') != wrong_residual:
        raise ValueError('saved wrong-target gradient reconstruction differs')
    control_norm = grid[chosen]['realized_norm']
    deltas = {'primary': primary_delta, 'opposite': scaled_delta(-primary, control_norm),
              'random': scaled_delta(random, control_norm), 'wrong-target': scaled_delta(wrong, control_norm)}
    for index, (name, _row) in enumerate(base_schedule[9:18], start=9):
        _check_parameter(actions[index], engine, deltas[name], 'weight', control_norm)
    for event in actions[18:20]:
        sha, actual = _check_parameter(event, engine, primary_delta, 'weight')
        if sha != grid[chosen]['parameter_sha256'] or actual != control_norm:
            raise ValueError('primary calibration edit differs')
    primary_discovery = records.decode_raw(actions[2 + chosen]['logits'], engine.vocabulary)
    discovery_effect = metrics.effect(prior[discovery['id']], primary_discovery, p, f)
    bias_delta = None
    bias_status = 'below-discovery-gain-floor'
    if discovery_effect['margin_gain'] >= 0.1 and discovery_effect['logp_gain'] >= 0.025:
        try:
            pd, fd = metrics.solve_bias(prior[discovery['id']], primary_discovery, p, f)
            bias_delta = torch.zeros_like(engine.bias_original)
            bias_delta[p], bias_delta[f] = pd, fd
            if (not bool(torch.isfinite(bias_delta).all()) or not bool(torch.any(bias_delta != 0)) or
                    torch.equal(engine.bias_original + bias_delta, engine.bias_original)):
                bias_delta = None
                bias_status = 'rounding-unavailable'
        except (OverflowError, RuntimeError, ValueError):
            bias_delta = None
            bias_status = 'solve-unavailable'
    bias_attempted = bias_delta is not None
    bias_available, match_rows = False, []
    expected = list(base_schedule)
    if bias_attempted:
        expected.append(('bias', discovery))
        if len(actions) <= 20:
            raise ValueError('bias discovery action is missing')
        _check_parameter(actions[20], engine, bias_delta, 'bias')
        replay = records.decode_raw(actions[20]['logits'], engine.vocabulary)
        clean_event = next(event for event in screen_events if event['kind'] == 'complete'
                           and event['case_id'] == discovery['id'])
        require_unchanged_other_logits(records.raw_f32_bytes(clean_event['logits'], engine.vocabulary),
                                       records.raw_f32_bytes(actions[20]['logits'], engine.vocabulary), p, f)
        bias_available = (abs((replay[p] - replay[f]) - (primary_discovery[p] - primary_discovery[f])) <= 0.001
                          and abs(metrics.log_probabilities(replay)[p] - metrics.log_probabilities(primary_discovery)[p]) <= 0.001)
        bias_status = 'available' if bias_available else 'replay-mismatch'
        if bias_available:
            expected.extend(('bias', row) for row in calibration)
            if len(actions) < 23:
                raise ValueError('bias calibration actions are missing')
            for index, row in enumerate(calibration):
                bias_event, primary_event = actions[21 + index], actions[18 + index]
                _check_parameter(bias_event, engine, bias_delta, 'bias')
                pp, ff = row['word_ids'][row['counterfactual']], row['word_ids'][row['factual']]
                clean_event = next(event for event in screen_events if event['kind'] == 'complete'
                                   and event['case_id'] == row['id'])
                require_unchanged_other_logits(records.raw_f32_bytes(clean_event['logits'], engine.vocabulary),
                                               records.raw_f32_bytes(bias_event['logits'], engine.vocabulary), pp, ff)
                pe = metrics.effect(prior[row['id']], records.decode_raw(primary_event['logits'], engine.vocabulary), pp, ff)
                be = metrics.effect(prior[row['id']], records.decode_raw(bias_event['logits'], engine.vocabulary), pp, ff)
                match_rows.append({'case_id': row['id'], 'primary': pe, 'bias': be,
                                   'matched': metrics.row_matches(pe, be)})
    if len(actions) != len(expected) or any(event['condition'] != name or event['case_id'] != row['id']
                                            for event, (name, row) in zip(actions, expected)):
        raise ValueError('development final schedule differs')
    summary = {'selected_object': obj, 'purple_id': p, 'factual_id': f, 'blue_id': b,
               'weight_norm': weight_norm, 'grid': grid, 'chosen_grid_index': chosen,
               'discovery_top_achieved': grid[chosen]['purple_unique_top'],
               'discovery_effect': discovery_effect, 'bias_available': bias_available,
               'bias_status': bias_status,
               'bias_attempted': bias_attempted,
               'bias_delta': ([float(bias_delta[p]), float(bias_delta[f])] if bias_delta is not None else None),
               'bias_delta_sha256': _delta_sha(bias_delta) if bias_delta is not None else None,
               'calibration_matches': match_rows,
               'calibration_match_available': bool(bias_available and len(match_rows) == 2 and
                                                   all(row['matched'] for row in match_rows)),
               'weight_delta_sha256': {name: _delta_sha(delta) for name, delta in deltas.items()},
               'control_realized_norm': control_norm}
    return summary, deltas, bias_delta


def run_evaluation(engine, journal, manifest, screen_events, development_events, expected_development):
    summary, deltas, bias_delta = rebuild_development(engine, manifest, screen_events, development_events)
    if (json.dumps(summary, sort_keys=True, separators=(',', ':'), allow_nan=False) !=
            json.dumps(expected_development, sort_keys=True, separators=(',', ':'), allow_nan=False)):
        raise ValueError('public development decision differs from its raw evidence')
    targets, localities = evaluate_rows(manifest, summary['selected_object'])
    conditions = ['original', 'primary', 'opposite', 'random', 'wrong-target']
    if summary['bias_available']:
        conditions.append('bias')
    for row in targets + localities:
        original_raw = None
        for condition in conditions:
            if condition == 'original':
                result = engine.run(journal, row, phase='evaluation', condition=condition)
                original_raw = tensor_bytes(result['logits'])
            elif condition == 'bias':
                result = engine.run(journal, row, phase='evaluation', condition=condition,
                                    delta=bias_delta, parameter='bias')
                require_unchanged_other_logits(original_raw, tensor_bytes(result['logits']),
                                               summary['purple_id'], summary['factual_id'])
            else:
                engine.run(journal, row, phase='evaluation', condition=condition,
                           delta=deltas[condition], parameter='weight')


def rebuild_evaluation(engine, manifest, screen_events, development_events, evaluation_events):
    """Compute the fixed outcomes from all saved full-vocabulary evaluation rows."""
    development, deltas, bias_delta = rebuild_development(engine, manifest, screen_events, development_events)
    targets, localities = evaluate_rows(manifest, development['selected_object'])
    conditions = ['original', 'primary', 'opposite', 'random', 'wrong-target']
    if development['bias_available']:
        conditions.append('bias')
    expected = [(row, condition) for row in targets + localities for condition in conditions]
    actions, extras = _completed_actions(manifest, evaluation_events, 'evaluation', engine.vocabulary)
    if extras or len(actions) != len(expected):
        raise ValueError('evaluation contains a missing or extra action')
    for event, (row, condition) in zip(actions, expected):
        if event['case_id'] != row['id'] or event['condition'] != condition:
            raise ValueError('evaluation order or condition differs')
        if condition == 'original':
            if (not {'parameter', 'parameter_sha256', 'realized_norm'}.issubset(event) or
                    event.get('parameter') is not None or event.get('parameter_sha256') is not None or
                    type(event.get('realized_norm')) is not float or event['realized_norm'] != 0.0):
                raise ValueError('original condition reports a parameter update')
        elif condition == 'bias':
            _check_parameter(event, engine, bias_delta, 'bias')
        else:
            sha, norm = _check_parameter(event, engine, deltas[condition], 'weight',
                                          development['control_realized_norm'])
            if condition == 'primary' and (sha != development['grid'][development['chosen_grid_index']]['parameter_sha256']
                                           or norm != development['control_realized_norm']):
                raise ValueError('evaluation primary differs from the selected update')
    by_key = {(event['case_id'], event['condition']): event for event in actions}
    p, f = development['purple_id'], development['factual_id']
    all_results = []
    for row in targets + localities:
        original = records.decode_raw(by_key[(row['id'], 'original')]['logits'], engine.vocabulary)
        if development['bias_available']:
            require_unchanged_other_logits(records.raw_f32_bytes(by_key[(row['id'], 'original')]['logits'], engine.vocabulary),
                                           records.raw_f32_bytes(by_key[(row['id'], 'bias')]['logits'], engine.vocabulary), p, f)
        accepted = None if row in targets else [row['word_ids'][word] for word in row['accepted']]
        requested = sorted(set([p, f] + ([] if accepted is None else accepted)))
        arms = {}
        for condition in conditions:
            values = records.decode_raw(by_key[(row['id'], condition)]['logits'], engine.vocabulary)
            info = metrics.describe(values, requested, accepted)
            info['top_tie_count'] = len(info.pop('top_ids'))
            info['distance_from_original'] = metrics.distances(original, values)
            if row in targets:
                info['effect'] = metrics.effect(original, values, p, f)
            arms[condition] = info
        all_results.append({'case_id': row['id'], 'role': row['role'], 'object': row['object'],
                            'accepted_ids': accepted, 'arms': arms})
    heldout = []
    for item in all_results[:4]:
        arms = item['arms']
        bias = arms.get('bias')
        matched = bool(bias is not None and metrics.row_matches(arms['primary']['effect'], bias['effect']))
        heldout.append({'case_id': item['case_id'], 'original_factual_top': arms['original']['unique_top'] == f,
                        'matched': matched,
                        'both_purple_top': bool(bias is not None and arms['primary']['unique_top'] == p
                                                and bias['unique_top'] == p)})
    matched_available = bool(development['discovery_top_achieved'] and development['bias_available'] and
                             development['calibration_match_available'] and
                             all(row['original_factual_top'] and row['matched'] for row in heldout) and
                             sum(row['both_purple_top'] for row in heldout) >= 2)
    decision_rows = []
    for item in all_results[4:]:
        arms = item['arms']
        original, primary, bias = arms['original'], arms['primary'], arms.get('bias')
        decision_rows.append({'case_id': item['case_id'], 'role': item['role'],
                              'original_top': original['unique_top'], 'primary_top': primary['unique_top'],
                              'bias_top': None if bias is None else bias['unique_top'],
                              'accepted_ids': item['accepted_ids'],
                              'primary_kl': primary['distance_from_original']['kl'],
                              'bias_kl': None if bias is None else bias['distance_from_original']['kl'],
                              'primary_nll_change': primary['accepted_nll'] - original['accepted_nll']})
    decision = metrics.locality_decision(decision_rows, match_available=matched_available)
    strata = {}
    for role in ('same_object_locality', 'other_color', 'unrelated'):
        subset = [row for row in decision_rows if row['role'] == role]
        strata[role] = {'count': len(subset),
                        'primary_mean_kl': math.fsum(row['primary_kl'] for row in subset) / len(subset),
                        'bias_mean_kl': (math.fsum(row['bias_kl'] for row in subset) / len(subset)
                                         if all(row['bias_kl'] is not None for row in subset) else None)}
    return {'selected_object': development['selected_object'], 'conditions': conditions,
            'heldout': heldout, 'matched_locality_available': matched_available,
            'locality_decision': decision, 'locality_strata': strata, 'rows': all_results}
