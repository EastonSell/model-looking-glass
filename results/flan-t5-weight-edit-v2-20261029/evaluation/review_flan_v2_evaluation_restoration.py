"""Independently reconcile saved evaluation restoration records and probe bytes."""
import argparse
import hashlib
import json
from pathlib import Path

def require(value, message):
    if not value:
        raise ValueError(message)

def strict(raw):
    def pairs(items):
        value = {}
        for k, v in items:
            require(k not in value, 'duplicate JSON key')
            value[k] = v
        return value
    def bad(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def raw_span(stream, span):
    require(type(span) is dict and span['kind'] == 'forward_complete' and span['stage'] == 'clean-probe'
            and span['shape'] == [32128] and type(span['offset']) is int and span['offset'] >= 0
            and type(span['bytes']) is int and span['bytes'] == 32128 * 4
            and span['metadata'] == {'arm': 'clean-probe', 'decoder_prefix': [0]}, 'probe span differs')
    stream.seek(span['offset'])
    raw = stream.read(span['bytes'])
    require(len(raw) == span['bytes'] and hashlib.sha256(raw).hexdigest() == span['sha256'], 'probe bytes differ')
    return raw

def audit(root, expected_phase, expected_events, expected_logits, development, expected_development_phase):
    root = Path(root)
    phase_path, events_path, logits_path = root/'phase.json', root/'evidence/events.jsonl', root/'evidence/logits.f32'
    require(sha_file(phase_path) == expected_phase and sha_file(events_path) == expected_events
            and sha_file(logits_path) == expected_logits, 'external evidence hash differs')
    phase = strict(phase_path.read_bytes())
    require(phase['events_sha256'] == expected_events and phase['logits_sha256'] == expected_logits
            and phase['phase'] == 'evaluation' and phase['execution_status'] == 'complete', 'phase identity differs')
    expected_arms = [('primary', 1.0), ('same', 1.0), ('broad', 2.0), ('random', 4.0), ('opposite', 4.0)]
    development = Path(development)
    dev_phase_path = development / 'phase.json'
    require(sha_file(dev_phase_path) == expected_development_phase, 'development phase pin differs')
    dev_phase = strict(dev_phase_path.read_bytes())
    dev_logits = development / 'evidence/logits.f32'
    require(sha_file(dev_logits) == dev_phase['logits_sha256'], 'development logits differ')
    starts, installed, completed, forward_records = [], None, [], {}
    active = None
    with logits_path.open('rb') as logits:
        initial = phase['initial_probe']
        reference = raw_span(logits, initial)
        with dev_logits.open('rb') as old_logits:
            require(raw_span(old_logits, dev_phase['initial_probe']) == reference,
                    'evaluation reference differs from development probe bytes')
        with events_path.open('rb') as stream:
            for i, line in enumerate(stream):
                require(line.endswith(b'\n') and len(line) <= 8 * 1024**2, 'invalid journal line')
                event = strict(line)
                require(type(event.get('event_index')) is int and event['event_index'] == i, 'journal index differs')
                kind, m = event['kind'], event.get('metadata', {})
                if kind == 'forward_complete':
                    require(type(event['call']) is int and event['call'] not in forward_records, 'duplicate completed call')
                    forward_records[event['call']] = {k: v for k, v in event.items() if k != 'event_index'}
                elif kind == 'arm_started':
                    require(active is None and len(starts) < len(expected_arms), 'nested or extra arm')
                    require((m['arm'], m['dose']) == expected_arms[len(starts)]
                            and m['state_sha256'] == phase['final_state_sha256'], 'arm order or baseline state differs')
                    active, installed = m, None
                    starts.append((m['arm'], m['dose']))
                elif kind == 'arm_installed':
                    require(active is not None and installed is None
                            and (m['arm'], m['dose']) == (active['arm'], active['dose']), 'arm installation differs')
                    installed = m['sha256']
                elif kind == 'arm_completed':
                    require(active is not None and installed is not None
                            and (m['arm'], m['dose']) == (active['arm'], active['dose'])
                            and m['installed_sha256'] == installed and m['restored_row_sha256'] == active['original_sha256']
                            and m['post_restore_state_sha256'] == phase['final_state_sha256']
                            and m['pre_buffer_sha256'] == active['buffer_sha256'] == m['post_probe_buffer_sha256']
                            and m['topology_unchanged'] is True, 'recorded restoration differs')
                    probe = m['probe_span']
                    require(forward_records.get(probe['call']) == probe and raw_span(logits, probe) == reference,
                            'restored probe differs from saved reference bytes')
                    completed.append((m['arm'], m['dose']))
                    active, installed = None, None
        require(forward_records.get(initial['call']) == initial, 'initial saved probe differs from journal')
    require(active is None and starts == completed == expected_arms
            and type(phase['calls']) is int and len(forward_records) == phase['calls'], 'arm or call count differs')
    require(sha_file(phase_path) == expected_phase and sha_file(events_path) == expected_events
            and sha_file(logits_path) == expected_logits and sha_file(dev_phase_path) == expected_development_phase
            and sha_file(dev_logits) == dev_phase['logits_sha256'], 'saved files changed during audit')
    return {'status': 'recorded_restoration_consistent', 'phase_sha256': expected_phase,
            'events_sha256': expected_events, 'logits_sha256': expected_logits,
            'calls': phase['calls'], 'completed_arms': len(completed), 'restoration_probe_matches': len(completed),
            'development_probe_match': True,
            'scope': 'Saved state/row/buffer hashes and exact saved probe bytes; does not independently authenticate process execution or reconstruct unsaved model-state bytes.'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--development', type=Path, required=True)
    for name in ('phase-sha256', 'events-sha256', 'logits-sha256', 'development-phase-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.attempt, a.phase_sha256, a.events_sha256, a.logits_sha256,
                           a.development, a.development_phase_sha256), sort_keys=True))

if __name__ == '__main__':
    main()
