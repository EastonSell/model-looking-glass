"""Convert explicit trusted observations into batched-throughput windows.

This module is an offline adapter, not an event collector, scheduler or
device supervisor. Missing observations are errors; it never estimates a
model-entry or synchronized-finish timestamp from a queue or client event.
"""
from .batched_throughput import _fields, _int, _json_copy, _name, _window, freeze_protocol


class IncompleteEventEvidence(ValueError):
    """The supplied event stream cannot yield a complete window record."""


_EVENT_FIELDS = {
    'window_start': ('kind', 'time_ns'),
    'window_end': ('kind', 'time_ns'),
    'owner_drain': ('kind', 'time_ns', 'drained'),
    'attempt': ('kind', 'time_ns', 'job_id', 'team_id', 'input_sha256', 'request_sha256'),
    'ready': ('kind', 'time_ns', 'job_id', 'team_id', 'broker_id', 'broker_job_id'),
    'selected': ('kind', 'time_ns', 'job_id'),
    'broker_terminal': ('kind', 'time_ns', 'job_id', 'state'),
    'returned': ('kind', 'time_ns', 'job_id', 'state', 'error', 'output_sha256',
                 'cancellation_requested', 'cancellation_requested_ns',
                 'cancellation_accepted', 'cancellation_decision_ns'),
    'forward_entry': ('kind', 'time_ns', 'call_id', 'group_id', 'profile_sha256',
                      'input_shape', 'members'),
    'forward_finished': ('kind', 'time_ns', 'call_id', 'state', 'synchronized', 'error'),
    'forward_uncertain': ('kind', 'time_ns', 'call_id', 'error'),
}


def assemble_window(protocol, *, expected_protocol_sha256, context, events):
    """Build one analyzer window from a bounded, losslessly recorded event set.

    The trusted caller must own the input data while snapshotting it, retain
    the original events, and supply actual instrumentation observations. A
    producer's `events_complete` and drain attestations are not independently
    authenticated by this function. The analyzer still must check the returned
    window and paired outputs; this function alone establishes no experiment
    result or device safety.
    """
    protocol, digest = freeze_protocol(protocol)
    if digest != expected_protocol_sha256:
        raise IncompleteEventEvidence('protocol differs from its frozen digest')
    context, _ = _json_copy(context)
    _fields(context, ('mode', 'window_id', 'owner_session_id', 'model_sha256',
                      'runtime_sha256', 'policy_sha256', 'clock', 'clock_resolution_ns',
                      'events_complete'), 'event context')
    if context['mode'] not in ('serial', 'grouped'):
        raise IncompleteEventEvidence('unknown window mode')
    if context['events_complete'] is not True:
        raise IncompleteEventEvidence('event loss or incomplete recording was reported')
    if context['clock'] != 'perf_counter_ns':
        raise IncompleteEventEvidence('event clock must be perf_counter_ns')
    _int(context['clock_resolution_ns'], 'clock resolution', minimum=1)
    for key in ('window_id', 'owner_session_id'):
        _name(context[key], key)
    if (context['model_sha256'] != protocol['model_sha256']
            or context['runtime_sha256'] != protocol['runtime_sha256']
            or context['policy_sha256'] != protocol[context['mode'] + '_policy_sha256']):
        raise IncompleteEventEvidence('event context differs from the frozen protocol')
    if type(events) is not list or not 3 <= len(events) <= 4096:
        raise IncompleteEventEvidence('event list must be nonempty and bounded')
    event_record, _ = _json_copy({'events': events})
    events = event_record['events']
    jobs = {job['job_id']: job for job in protocol['jobs']}
    by_kind = {name: {} for name in _EVENT_FIELDS}
    for event in events:
        if type(event) is not dict or type(event.get('kind')) is not str:
            raise IncompleteEventEvidence('each event needs a known kind')
        kind = event['kind']
        if kind not in _EVENT_FIELDS:
            raise IncompleteEventEvidence('unknown event kind')
        _fields(event, _EVENT_FIELDS[kind], kind)
        _int(event['time_ns'], 'event timestamp')
        if kind in ('window_start', 'window_end', 'owner_drain'):
            identity = 'window'
        elif kind.startswith('forward_'):
            identity = event['call_id']
            _name(identity, 'call ID')
        else:
            identity = event['job_id']
            _name(identity, 'job ID')
            if identity not in jobs:
                raise IncompleteEventEvidence('event names an unknown job')
            if kind in ('attempt', 'ready') and event['team_id'] != jobs[identity]['team_id']:
                raise IncompleteEventEvidence('observed team differs from frozen job')
        if identity in by_kind[kind]:
            raise IncompleteEventEvidence('duplicate observation')
        by_kind[kind][identity] = event

    try:
        start = by_kind['window_start']['window']['time_ns']
        end = by_kind['window_end']['window']['time_ns']
        drain_event = by_kind['owner_drain']['window']
    except KeyError as exc:
        raise IncompleteEventEvidence('missing window or owner-drain observation') from exc
    if (end <= start or type(drain_event['drained']) is not bool
            or not start <= drain_event['time_ns'] <= end):
        raise IncompleteEventEvidence('invalid window or owner-drain observation')
    if any(not start <= event['time_ns'] <= end for event in events):
        raise IncompleteEventEvidence('an event lies outside the window')
    if drain_event['drained'] and any(
            event['kind'] in ('attempt', 'ready', 'selected', 'forward_entry',
                              'forward_uncertain', 'broker_terminal')
            and event['time_ns'] > drain_event['time_ns'] for event in events):
        raise IncompleteEventEvidence('owner drain precedes observed broker work')

    attempts = by_kind['attempt']
    returns = by_kind['returned']
    if set(attempts) != set(returns):
        raise IncompleteEventEvidence('every observed attempt needs one client-return observation')
    for kind in ('ready', 'selected', 'broker_terminal'):
        if not set(by_kind[kind]).issubset(attempts):
            raise IncompleteEventEvidence('job observation has no corresponding attempt')
    entries = by_kind['forward_entry']
    finished, uncertain = by_kind['forward_finished'], by_kind['forward_uncertain']
    if set(finished) & set(uncertain) or set(entries) != set(finished) | set(uncertain):
        raise IncompleteEventEvidence('each model entry needs exactly one finish or uncertain observation')

    calls, member_call = [], {}
    for call_id, entry in sorted(entries.items(), key=lambda item: (item[1]['time_ns'], item[0])):
        if type(entry['members']) is not list:
            raise IncompleteEventEvidence('call members must be an explicit list')
        members = []
        for member in entry['members']:
            _fields(member, ('job_id', 'team_id', 'broker_id', 'broker_job_id', 'row_offset'), 'observed member')
            _name(member['job_id'], 'member job ID')
            if member['job_id'] not in attempts or member['job_id'] in member_call:
                raise IncompleteEventEvidence('missing or repeated launched job')
            if member['team_id'] != jobs[member['job_id']]['team_id']:
                raise IncompleteEventEvidence('observed member team differs from frozen job')
            member_call[member['job_id']] = call_id
            members.append({key: member[key] for key in ('job_id', 'broker_id', 'broker_job_id', 'row_offset')})
        if call_id in finished:
            observation = finished[call_id]
            if observation['state'] not in ('succeeded', 'failed'):
                raise IncompleteEventEvidence('finished call has invalid state')
            state, sync, finish_ns, error = (observation['state'], observation['synchronized'],
                                              observation['time_ns'], observation['error'])
        else:
            observation = uncertain[call_id]
            state, sync, finish_ns, error = 'unresolved', False, None, observation['error']
        if observation['time_ns'] < entry['time_ns']:
            raise IncompleteEventEvidence('call observation precedes model entry')
        if drain_event['drained'] and finish_ns is not None and finish_ns > drain_event['time_ns']:
            raise IncompleteEventEvidence('owner drain precedes an observed model finish')
        calls.append({'call_id': call_id, 'group_id': entry['group_id'],
                      'profile_sha256': entry['profile_sha256'], 'input_shape': entry['input_shape'],
                      'members': members, 'launch_ns': entry['time_ns'],
                      'finish_ns': finish_ns, 'state': state, 'synchronized': sync, 'error': error})

    rows = []
    calls_by_id = {call['call_id']: call for call in calls}
    for job in protocol['jobs']:
        job_id = job['job_id']
        if job_id not in attempts:
            continue
        attempt, returned = attempts[job_id], returns[job_id]
        ready = by_kind['ready'].get(job_id)
        selected = by_kind['selected'].get(job_id)
        terminal = by_kind['broker_terminal'].get(job_id)
        if terminal is not None:
            if terminal['time_ns'] < attempt['time_ns'] or (ready and terminal['time_ns'] < ready['time_ns']):
                raise IncompleteEventEvidence('broker terminal precedes attempt or admission')
            if selected and terminal['time_ns'] < selected['time_ns']:
                raise IncompleteEventEvidence('broker terminal precedes selection')
            call = calls_by_id.get(member_call.get(job_id))
            if call is not None and call['finish_ns'] is not None and terminal['time_ns'] < call['finish_ns']:
                raise IncompleteEventEvidence('broker terminal precedes model finish')
        if terminal is not None and returned['state'] != 'unresolved' and terminal['time_ns'] > returned['time_ns']:
            raise IncompleteEventEvidence('delivered result precedes broker terminal observation')
        if returned['state'] not in ('rejected', 'submit_error', 'unresolved') and terminal is None:
            raise IncompleteEventEvidence('delivered broker result lacks terminal observation')
        rows.append({'job_id': job_id, 'input_sha256': attempt['input_sha256'],
                     'request_sha256': attempt['request_sha256'], 'state': returned['state'],
                     'attempt_ns': attempt['time_ns'], 'ready_ns': ready['time_ns'] if ready else None,
                     'selected_ns': selected['time_ns'] if selected else None,
                     'returned_ns': returned['time_ns'], 'broker_id': ready['broker_id'] if ready else None,
                     'broker_job_id': ready['broker_job_id'] if ready else None,
                     'call_id': member_call.get(job_id), 'output_sha256': returned['output_sha256'],
                     'error': returned['error'], 'cancellation_requested': returned['cancellation_requested'],
                     'cancellation_requested_ns': returned['cancellation_requested_ns'],
                     'cancellation_accepted': returned['cancellation_accepted'],
                     'cancellation_decision_ns': returned['cancellation_decision_ns'],
                     'broker_terminal_state': terminal['state'] if terminal else None})
    window = {'mode': context['mode'], 'window_id': context['window_id'],
              'owner_session_id': context['owner_session_id'], 'protocol_sha256': digest,
              'model_sha256': context['model_sha256'], 'runtime_sha256': context['runtime_sha256'],
              'policy_sha256': context['policy_sha256'], 'start_ns': start, 'end_ns': end,
              'clock': context['clock'], 'clock_resolution_ns': context['clock_resolution_ns'],
              'drained': drain_event['drained'], 'attempts': rows, 'calls': calls}
    _window(protocol, digest, window, context['mode'])
    return window
