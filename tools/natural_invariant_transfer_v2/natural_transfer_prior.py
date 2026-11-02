"""Read previously closed AgreementJournal no-op witnesses, without a model.

The surrounding operator must independently verify the prior direct-process
receipt, complete saved audit, model/source identity and public bindings.
This reader rechecks the target entries and their content-addressed bytes.
"""
import hashlib
import json
import math
import re
import struct
from pathlib import Path


EXPECTED_SCHEDULE_SHA256 = 'fbf526fd867ed48127c24b3b35004727b1757782e90798d60dcfdec37ac9b2f6'
HEX = re.compile(r'[0-9a-f]{64}\Z')


class PriorEvidenceError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PriorEvidenceError(message)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pinned(path, expected):
    _require(type(expected) is str and HEX.fullmatch(expected), 'invalid external SHA-256 pin')
    data = Path(path).read_bytes()
    _require(_digest(data) == expected, 'prior evidence SHA-256 mismatch')
    return data


def _tensor(directory, refs, name, shape):
    ref = refs.get(name)
    _require(type(ref) is dict and type(ref.get('shape')) is list
             and all(type(x) is int for x in ref['shape']) and ref['shape'] == list(shape)
             and ref.get('dtype') == 'float32-le'
             and type(ref.get('bytes')) is int and ref['bytes'] == 4 * math.prod(shape)
             and type(ref.get('sha256')) is str and HEX.fullmatch(ref['sha256']),
             'invalid prior tensor reference: ' + name)
    raw = (directory / 'tensors' / (ref['sha256'] + '.f32le')).read_bytes()
    _require(len(raw) == ref['bytes'] and _digest(raw) == ref['sha256'],
             'prior tensor bytes mismatch: ' + name)
    _require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw)),
             'nonfinite prior tensor: ' + name)
    return raw


def _check_noop_steps(event, final_index, length, width):
    site = event.get('site')
    _require(type(final_index) is int and type(site) is dict and type(site.get('layer')) is int
             and type(site.get('output_selector')) is list
             and all(type(x) is int for x in site['output_selector'])
             and site == {'id': '3:attn:last', 'module_path': 'transformer.h.3.attn',
                          'layer': 3, 'component': 'attn', 'token_role': 'last',
                          'output_selector': [0]}, 'prior site mismatch')
    steps = event.get('plan_metadata', {}).get('steps')
    expected = [('whole_before', 'capture', None, None), ('before', 'capture', 0, final_index),
                ('action', 'null', 0, final_index), ('after', 'capture', 0, final_index),
                ('whole_after', 'capture', None, None)]
    _require(type(steps) is list and len(steps) == len(expected), 'prior no-op steps')
    for order, (step, (name, operation, batch, token)) in enumerate(zip(steps, expected)):
        _require(type(step) is dict and step.get('name') == name and step.get('operation') == operation
                 and type(step.get('order')) is int and step['order'] == order
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
                 and step.get('preceding_same_site_actions') == [x[0] for x in expected[:order]],
                 'prior no-op hook metadata')


def load_prior(directory, cases, mapping, *, events_sha256, started_sha256, terminal_sha256, expected_schedule_sha256,
               width=768, vocabulary=50257):
    """Return 48 baseline vectors/logits only after exact target checks pass."""
    directory = Path(directory)
    _require(directory.is_absolute(), 'absolute prior journal path required')
    _require(type(expected_schedule_sha256) is str and HEX.fullmatch(expected_schedule_sha256),
             'invalid prior schedule pin')
    _require(type(cases) is list and len(cases) == 68
             and type(width) is int and width > 0 and type(vocabulary) is int and vocabulary > 0,
             'prior case inventory or dimensions')
    rows = mapping.get('receivers')
    _require(type(rows) is list and len(rows) == 48, 'fixed transfer mapping')
    _pinned(directory / 'started.json', started_sha256)
    terminal = json.loads(_pinned(directory / 'terminal.json', terminal_sha256))
    event_bytes = _pinned(directory / 'events.jsonl', events_sha256)
    _require(terminal.get('status') == 'complete_candidate'
             and terminal.get('storage_accounting_complete') is True
             and type(terminal.get('events')) is int and terminal['events'] == 613
             and type(terminal.get('detail', {}).get('forwards')) is int
             and terminal['detail']['forwards'] == 612
             and terminal.get('detail', {}).get('schedule_sha256') == expected_schedule_sha256,
             'prior journal is not complete')
    records = [json.loads(line) for line in event_bytes.splitlines()]
    _require(len(records) == 613, 'prior event inventory')
    contract = records[0].get('event', {})
    _require(type(records[0].get('index')) is int and records[0]['index'] == 0
             and contract.get('kind') == 'run_contract'
             and type(contract.get('expected_forwards')) is int and contract['expected_forwards'] == 612
             and contract.get('schedule_sha256') == expected_schedule_sha256
             and contract.get('model_state_sha256') == terminal['detail'].get('model_state_sha256_before_after'),
             'prior run contract or model state')
    schedule = []
    for i, record in enumerate(records[1:]):
        event = record.get('event', {})
        case_index = i // 9
        _require(type(record.get('index')) is int and record['index'] == i + 1
                 and event.get('kind') == 'forward'
                 and type(event.get('index')) is int and event['index'] == i
                 and type(event.get('case_index')) is int and event['case_index'] == case_index
                 and event.get('case_id') == cases[case_index].get('case_id'),
                 'prior action order or identity')
        schedule.append((event['case_id'], event.get('action')))
    digest = _digest(json.dumps(schedule, separators=(',', ':'), ensure_ascii=True).encode('ascii'))
    _require(digest == expected_schedule_sha256, 'prior action schedule digest')
    case_indices = {x['case_id']: i for i, x in enumerate(cases)}
    _require(len(case_indices) == len(cases), 'duplicate prior case IDs')
    _require(len({x['receiver_case_id'] for x in rows}) == 48, 'duplicate transfer receivers')
    output = {}
    for mapped in rows:
        case_id = mapped['receiver_case_id']
        _require(case_id in case_indices, 'missing prior receiver')
        index = case_indices[case_id]
        case = cases[index]
        _require(case.get('role') == 'target' and case.get('split') == mapped.get('split')
                 and type(case.get('token_ids')) is list and all(type(x) is int for x in case['token_ids'])
                 and type(mapped.get('receiver_token_ids')) is list
                 and all(type(x) is int for x in mapped['receiver_token_ids'])
                 and case['token_ids'] == mapped['receiver_token_ids']
                 and type(case.get('final_index')) is int and type(mapped.get('final_index')) is int
                 and case['final_index'] == mapped['final_index'], 'mapping input identity mismatch')
        clean, noop, repeat = (records[1 + 9 * index + offset] for offset in (0, 1, 8))
        length = len(case['token_ids'])
        for entry, action in ((clean, 'clean'), (noop, 'noop'), (repeat, 'clean_after')):
            event = entry['event']
            _require(event.get('action') == action and type(event.get('input_ids')) is list
                     and all(type(x) is int for x in event['input_ids']) and event['input_ids'] == case['token_ids']
                     and type(event.get('attention_mask')) is list
                     and all(type(x) is int for x in event['attention_mask'])
                     and event['attention_mask'] == [1] * length
                     and event.get('all_sequence_logits_finite') is True
                     and all(event.get(k) == case.get(k) for k in ('split', 'role', 'determiner'))
                     and all(type(event.get(k)) is int and event[k] == case[k]
                             for k in ('frame_index', 'context_index', 'noun_index')),
                     'prior target input, action or coordinates mismatch')
        _check_noop_steps(noop['event'], case['final_index'], length, width)
        _require(set(clean['tensors']) == {'last_logits'} and set(repeat['tensors']) == {'last_logits'}
                 and set(noop['tensors']) == {'last_logits', 'before', 'after', 'whole_before', 'whole_after'},
                 'unexpected prior no-op tensor set')
        clean_logits = _tensor(directory, clean['tensors'], 'last_logits', (vocabulary,))
        noop_logits = _tensor(directory, noop['tensors'], 'last_logits', (vocabulary,))
        repeat_logits = _tensor(directory, repeat['tensors'], 'last_logits', (vocabulary,))
        before = _tensor(directory, noop['tensors'], 'before', (width,))
        after = _tensor(directory, noop['tensors'], 'after', (width,))
        whole_before = _tensor(directory, noop['tensors'], 'whole_before', (1, length, width))
        whole_after = _tensor(directory, noop['tensors'], 'whole_after', (1, length, width))
        _require(clean_logits == noop_logits == repeat_logits, 'prior clean/no-op/repeat logits differ')
        _require(before == after and whole_before == whole_after
                 and whole_before[-4 * width:] == before, 'prior no-op landing or final position differs')
        output[case_id] = {'vector_f32': before, 'last_logits_f32': noop_logits,
                           'vector_sha256': _digest(before), 'logits_sha256': _digest(noop_logits),
                           'noop_event_index': noop['index']}
    return output
