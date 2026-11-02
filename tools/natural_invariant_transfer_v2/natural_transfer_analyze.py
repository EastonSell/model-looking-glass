"""Saved-byte checks and fixed natural donor-transfer calculations; no model."""
import hashlib
import json
import math
import re
import struct
from pathlib import Path

from natural_transfer_numeric import ensure_round_to_nearest, plan_candidates
from natural_transfer_prior import _tensor


class SavedEvidenceError(ValueError):
    pass


def _require(value, message):
    if not value:
        raise SavedEvidenceError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _floats(raw):
    return tuple(x[0] for x in struct.iter_unpack('<f', raw))


def _contrast(raw):
    return float(struct.unpack_from('<f', raw, 318 * 4)[0]) - float(struct.unpack_from('<f', raw, 389 * 4)[0])


def _distance(left, right):
    _require(len(left) == len(right) and len(left) % 4 == 0, 'logit lengths differ')
    return math.sqrt(math.fsum((a[0] - b[0]) ** 2 for a, b in
                              zip(struct.iter_unpack('<f', left), struct.iter_unpack('<f', right))))


def _steps(event, final_index, length, width):
    site = event.get('site')
    _require(type(site) is dict and type(site.get('layer')) is int
             and type(site.get('output_selector')) is list
             and all(type(x) is int for x in site['output_selector'])
             and site == {'id':'3:attn:last','module_path':'transformer.h.3.attn','layer':3,
                          'component':'attn','token_role':'last','output_selector':[0]}, 'wrong patch site')
    steps = event.get('plan_metadata', {}).get('steps')
    expected = [('whole_before','capture',None,None), ('before','capture',0,final_index),
                ('action','patch',0,final_index), ('after','capture',0,final_index),
                ('whole_after','capture',None,None)]
    _require(type(steps) is list and len(steps) == len(expected), 'wrong patch steps')
    for i, (step, (name, op, batch, token)) in enumerate(zip(steps, expected)):
        _require(type(step) is dict and step.get('name') == name and step.get('operation') == op
                 and type(step.get('order')) is int and step['order'] == i
                 and step.get('module_path') == 'transformer.h.3.attn'
                 and type(step.get('invocation')) is int and step['invocation'] == 0
                 and type(step.get('output_selector')) is list
                 and all(type(x) is int for x in step['output_selector']) and step['output_selector'] == [0]
                 and (step.get('batch_index') is None if batch is None else
                      type(step.get('batch_index')) is int and step['batch_index'] == batch)
                 and (step.get('token_index') is None if token is None else
                      type(step.get('token_index')) is int and step['token_index'] == token)
                 and step.get('device') == 'cpu' and step.get('dtype') == 'torch.float32'
                 and type(step.get('forward_index')) is int and step['forward_index'] == 0
                 and step.get('call_site_known') is False
                 and step.get('module_aliases') == ['transformer.h.3.attn']
                 and type(step.get('output_shape')) is list
                 and all(type(x) is int for x in step['output_shape'])
                 and step['output_shape'] == [1, length, width]
                 and type(step.get('selected_shape')) is list
                 and all(type(x) is int for x in step['selected_shape'])
                 and step['selected_shape'] == ([1, length, width] if batch is None else [width])
                 and step.get('source_capture') is None
                 and step.get('preceding_same_site_actions') == [x[0] for x in expected[:i]], 'wrong patch hook')


def load_phase(directory, mapping, prior, phase, *, events_sha256, terminal_sha256,
               expected_binding, width, vocabulary):
    ensure_round_to_nearest()
    _require(phase in ('discovery','evaluation'), 'invalid phase')
    binding_keys = {'pre_run_sha256','mapping_sha256','prior_receipt_sha256','latest_exposure_sha256'}
    if phase == 'evaluation':
        binding_keys.add('freeze_sha256')
    _require(type(expected_binding) is dict and set(expected_binding) == binding_keys
             and all(type(x) is str and re.fullmatch('[0-9a-f]{64}', x) for x in expected_binding.values()),
             'invalid external phase binding')
    directory = Path(directory)
    event_bytes, terminal_bytes = (directory/'events.jsonl').read_bytes(), (directory/'terminal.json').read_bytes()
    _require(_sha(event_bytes) == events_sha256 and _sha(terminal_bytes) == terminal_sha256,
             'phase hash mismatch')
    terminal = json.loads(terminal_bytes)
    _require(terminal.get('status') == 'complete_candidate'
             and type(terminal.get('events')) is int and terminal['events'] == 97
             and terminal.get('storage_accounting_complete') is True
             and type(terminal.get('detail',{}).get('forwards')) is int
             and terminal['detail']['forwards'] == 96, 'phase not complete')
    records = [json.loads(line) for line in event_bytes.splitlines()]
    _require(len(records) == 97 and type(records[0].get('index')) is int and records[0]['index'] == 0,
             'phase event count')
    contract = records[0].get('event',{})
    _require(contract.get('kind') == 'run_contract' and contract.get('phase') == phase
             and type(contract.get('expected_forwards')) is int and contract['expected_forwards'] == 96
             and contract.get('model_state_sha256') == terminal['detail'].get('model_state_sha256_before_after')
             and type(expected_binding) is dict and contract.get('binding') == expected_binding,
             'phase contract mismatch')
    rows = [x for x in mapping['receivers'] if x['split'] == phase]
    _require(len(rows) == 24, 'phase receiver count')
    result, sidecars, tensor_refs = [], set(), set()
    for j, row in enumerate(rows):
        receiver = row['receiver_case_id']
        c = row['candidates']
        direct, lexical, other = c['Q2']['donor_case_id'], c['Q1']['donor_case_id'], c['Q4']['donor_case_id']
        base, donor = prior[receiver], prior[direct]
        plans = plan_candidates(base['vector_f32'], donor['vector_f32'],
                                prior[lexical]['vector_f32'], prior[other]['vector_f32'])
        m0, md = _contrast(base['last_logits_f32']), _contrast(donor['last_logits_f32'])
        d0 = _distance(base['last_logits_f32'], donor['last_logits_f32'])
        output = {'case_id': receiver, 'split': phase, 'orientation': row['orientation'],
                  'determiner': row['determiner'], 'frame_index': row['frame_index'],
                  'context_index': row['context_index'], 'noun_index': row['noun_index'],
                  'receiver_M': m0, 'donor_M': md, 'donor_forecast': md-m0, 'D0': d0,
                  'q': plans['q'], 'candidates': {}}
        for offset, key in enumerate(('Q1','Q2','Q3','Q4')):
            index, record = 4*j+offset, records[1+4*j+offset]
            event, refs = record.get('event',{}), record.get('tensors',{})
            donor_id = direct if key in ('Q2','Q3') else lexical if key == 'Q1' else other
            _require(type(record.get('index')) is int and record['index'] == index+1
                     and event.get('kind') == 'forward' and type(event.get('index')) is int and event['index'] == index
                     and event.get('phase') == phase and event.get('case_id') == receiver
                     and event.get('receiver') == receiver and event.get('candidate') == key
                     and event.get('donor') == donor_id and type(event.get('input_ids')) is list
                     and all(type(x) is int for x in event['input_ids'])
                     and event['input_ids'] == row['receiver_token_ids']
                     and type(event.get('attention_mask')) is list
                     and all(type(x) is int for x in event['attention_mask'])
                     and event['attention_mask'] == [1]*len(row['receiver_token_ids'])
                     and event.get('all_sequence_logits_finite') is True, 'phase action mismatch')
            _steps(event, row['final_index'], len(row['receiver_token_ids']), width)
            expected = plans['candidates'][key]
            dose = event.get('dose',{})
            _require(dose.get('dose_valid') is True
                     and json.dumps(dose.get('certificate'), sort_keys=True, separators=(',', ':'), allow_nan=False)
                         == json.dumps(expected['certificate'], sort_keys=True, separators=(',', ':'), allow_nan=False)
                     and expected['certificate']['pass'] is True
                     and type(dose.get('requested_norm')) is float and dose['requested_norm'] == plans['q']
                     and dose.get('receiver_vector_sha256') == base['vector_sha256']
                     and dose.get('donor_vector_sha256') == prior[donor_id]['vector_sha256'], 'dose or donor mismatch')
            name = dose.get('displacement_file')
            _require(name == f'displacements/{index:03d}-{key}.f64le' and name not in sidecars, 'displacement path mismatch')
            h = (directory/name).read_bytes()
            _require(h == expected['displacement_f64'] and _sha(h) == dose.get('displacement_sha256'),
                     'binary64 displacement mismatch')
            sidecars.add(name)
            _require(set(refs) == {'last_logits','before','after','whole_before','whole_after'}, 'tensor set mismatch')
            tensor_refs.update(x['sha256']+'.f32le' for x in refs.values())
            length = len(row['receiver_token_ids'])
            before = _tensor(directory, refs, 'before', (width,))
            after = _tensor(directory, refs, 'after', (width,))
            whole_before = _tensor(directory, refs, 'whole_before', (1,length,width))
            whole_after = _tensor(directory, refs, 'whole_after', (1,length,width))
            logits = _tensor(directory, refs, 'last_logits', (vocabulary,))
            _require(before == base['vector_f32'] and after == expected['replacement_f32']
                     and whole_before[-4*width:] == before and whole_after[-4*width:] == after
                     and whole_before[:-4*width] == whole_after[:-4*width], 'landing or untouched bits mismatch')
            mp = _contrast(logits)
            dq = _distance(logits, donor['last_logits_f32'])
            output['candidates'][key] = {'patched_M': mp, 'delta_M': mp-m0,
                                         'oriented_delta_M': row['orientation']*(mp-m0),
                                         'Dq': dq, 'per_case_R': None if d0 == 0 else 1-dq/d0}
        result.append(output)
    actual_sidecars = {'displacements/'+x.name for x in (directory/'displacements').iterdir() if x.is_file()}
    actual_tensors = {x.name for x in (directory/'tensors').iterdir() if x.is_file()}
    _require(actual_sidecars == sidecars and actual_tensors == tensor_refs, 'phase file inventory differs')
    ensure_round_to_nearest()
    return result


def select_numeric(oriented_rows):
    """Select from 24 anonymous numeric Q columns; no labels or case metadata."""
    ensure_round_to_nearest()
    _require(type(oriented_rows) is list and len(oriented_rows) == 24, 'wrong numeric discovery matrix')
    _require(all(type(row) is list and len(row) == 4 and
                 all(type(value) is float and math.isfinite(value) for value in row)
                 for row in oriented_rows), 'numeric matrix must be finite binary64 scalars only')
    keys = ('Q1','Q2','Q3','Q4')
    means = {key: math.fsum(row[i] for row in oriented_rows)/24 for i,key in enumerate(keys)}
    ranking = sorted(keys, key=lambda key: (-means[key], keys.index(key)))
    selected = ranking[0] if means[ranking[0]] > 0 else None
    ties = [[x for x in keys if means[x] == means[key]] for key in ranking]
    ties = [group for i, group in enumerate(ties) if group not in ties[:i] and len(group)>1]
    ensure_round_to_nearest()
    return {'means_oriented':means, 'ranking':ranking, 'ties':ties, 'selected':selected,
            'selection_abstained':selected is None}


def make_freeze(discovery, mapping, *, mapping_sha256, prior_events_sha256, discovery_events_sha256):
    ensure_round_to_nearest()
    _require(len(discovery) == 24 and all(x['split'] == 'discovery' for x in discovery), 'wrong discovery rows')
    keys = ('Q1','Q2','Q3','Q4')
    matrix = [[row['candidates'][key]['oriented_delta_M'] for key in keys] for row in discovery]
    selection = select_numeric(matrix)
    means = selection['means_oriented']
    evaluation_rows = [x for x in mapping['receivers'] if x['split'] == 'evaluation']
    _require(len(evaluation_rows)==24, 'wrong evaluation mapping')
    predictions = [{'case_id': row['receiver_case_id'], 'orientation': row['orientation'],
                    'predictions': {key: row['orientation']*means[key] for key in keys}}
                   for row in evaluation_rows]
    ensure_round_to_nearest()
    return {'schema':'natural-invariant-donor-transfer-freeze-v1', 'mapping_sha256':mapping_sha256,
            'prior_events_sha256':prior_events_sha256, 'discovery_events_sha256':discovery_events_sha256,
            **selection, 'evaluation_predictions':predictions}


def evaluate(evaluation, freeze):
    ensure_round_to_nearest()
    _require(len(evaluation)==24 and all(x['split']=='evaluation' for x in evaluation), 'wrong evaluation rows')
    predictions = freeze['evaluation_predictions']
    _require(len(predictions)==24 and all(p['case_id']==r['case_id'] and p['orientation']==r['orientation']
                                         for p,r in zip(predictions,evaluation)), 'prediction order mismatch')
    keys = ('Q1','Q2','Q3','Q4')
    mean_d0 = math.fsum(row['D0'] for row in evaluation)/24
    baseline_correct = sum((r['receiver_M']>0 if r['determiner']=='this' else r['receiver_M']<0) for r in evaluation)
    groups = {}
    for row in evaluation:
        groups.setdefault((row['frame_index'],row['context_index'],row['noun_index']),{})[row['determiner']] = row['receiver_M']
    _require(len(groups)==12 and all(set(x)=={'this','these'} for x in groups.values()), 'baseline pairs incomplete')
    ordered_pairs = sum(x['this'] > x['these'] for x in groups.values())
    adequate = baseline_correct>=18 and ordered_pairs>=9
    summary = {}
    for key in keys:
        mean_dq = math.fsum(r['candidates'][key]['Dq'] for r in evaluation)/24
        norm = None if mean_d0 <= 1e-12 else 1-mean_dq/mean_d0
        errors = []
        for row,pred in zip(evaluation,predictions):
            observed = row['candidates'][key]['delta_M']
            expected = pred['predictions'][key]
            errors.append({'case_id':row['case_id'], 'observed_delta_M':observed,
                           'discovery_prediction':expected, 'discovery_error':abs(observed-expected),
                           'zero_error':abs(observed), 'donor_forecast':row['donor_forecast'],
                           'donor_error':abs(observed-row['donor_forecast'])})
        mae = lambda label: math.fsum(x[label] for x in errors)/24
        method, zero, donor = mae('discovery_error'), mae('zero_error'), mae('donor_error')
        summary[key] = {'mean_Dq':mean_dq, 'R':norm, 'prediction_MAE':method, 'zero_MAE':zero,
                        'donor_MAE':donor, 'beats_zero':method<zero, 'beats_donor':method<donor,
                        'recipient_correct':sum((r['candidates'][key]['patched_M']>0 if r['determiner']=='this'
                                                 else r['candidates'][key]['patched_M']<0) for r in evaluation),
                        'prediction_rows':errors}
    eligible = adequate and mean_d0 > 1e-12
    pattern = None if not eligible else (summary['Q2']['R']>0.25 and
                summary['Q2']['R']-max(summary[q]['R'] for q in ('Q1','Q3','Q4'))>0.05)
    ensure_round_to_nearest()
    return {'schema':'natural-invariant-donor-transfer-result-v1', 'mean_D0':mean_d0,
            'baseline_correct':baseline_correct, 'baseline_ordered_pairs':ordered_pairs,
            'baseline_adequate':adequate, 'eligible':eligible, 'primary_pattern_supported':pattern,
            'selected':freeze['selected'], 'summaries':summary, 'cases':evaluation}
