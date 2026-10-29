"""Independent model-free replay of the prospective color study's screen.

This audits only saved event bytes against a separately pinned manifest. It
does not attest model execution, loaded source, process identity, or history.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_distilbert_target_metrics import require_denormal_support

PUBLIC_CASE_SHA = '4f00000cf205137347d551fc2837e082ac632a515e92c1e5893b3312f1f71623'
MAX_BYTES = 128 * 1024 * 1024
ROLES = ('discovery', 'calibration-01', 'calibration-02')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=_object, parse_constant=bad_constant)


def _exact_int(value, expected):
    return type(value) is int and value == expected


def _exact_ids(value, expected):
    return (type(value) is list and len(value) == len(expected)
            and all(_exact_int(a, b) for a, b in zip(value, expected)))


def _f32(envelope, count):
    if (type(envelope) is not dict or set(envelope) != {'encoding', 'bytes', 'sha256', 'data'}
            or envelope['encoding'] != 'f32le' or not _exact_int(envelope['bytes'], 4 * count)
            or type(envelope['sha256']) is not str or type(envelope['data']) is not str):
        raise ValueError('invalid full-logit envelope')
    try:
        raw = base64.b64decode(envelope['data'], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('invalid base64') from error
    if len(raw) != 4 * count or hashlib.sha256(raw).hexdigest() != envelope['sha256']:
        raise ValueError('full-logit bytes do not match their digest')
    values = struct.unpack('<' + str(count) + 'f', raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('nonfinite full-logit value')
    return values


def _summary(values, ids):
    peak = max(values)
    top_ids = [i for i, value in enumerate(values) if value == peak]
    normalizer = math.fsum(math.exp(value - peak) for value in values)
    log_normalizer = math.log(normalizer)
    def word(token):
        if type(token) is not int or not 0 <= token < len(values):
            raise ValueError('bad target token')
        lp = math.fsum((values[token], -peak, -log_normalizer))
        return {'id': token, 'logit': values[token], 'log_probability': lp,
                'probability': math.exp(lp),
                'rank': 1 + sum(value > values[token] for value in values)}
    return {'top_ids': top_ids, 'unique_top': top_ids[0] if len(top_ids) == 1 else None,
            'top_representative': top_ids[0],
            'top_five_ids': sorted(range(len(values)), key=lambda i: (-values[i], i))[:5],
            'tokens': {name: word(token) for name, token in ids.items()}}


def audit_bytes(manifest_raw, journal_raw, *, expected_manifest_sha, vocabulary):
    require_denormal_support()
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha:
        raise ValueError('manifest digest mismatch')
    if type(vocabulary) is not int or not 2 <= vocabulary <= 30522:
        raise ValueError('invalid vocabulary size')
    if len(journal_raw) > MAX_BYTES or not journal_raw.endswith(b'\n'):
        raise ValueError('incomplete or oversized screen journal')
    manifest = _json(manifest_raw)
    order = manifest['candidate_order']
    if type(order) is not list or len(order) != 3 or len(set(order)) != 3:
        raise ValueError('invalid candidate order')
    events = [_json(line) for line in journal_raw.splitlines()]
    if len(events) != 18:
        raise ValueError('the screen must retain exactly nine start/completion pairs')
    candidates, observed = {}, []
    for candidate in order:
        candidates[candidate] = True
        for role in ROLES:
            matches = [row for row in manifest['cases']
                       if row['object'] == candidate and row['role'] == role]
            if len(matches) != 1:
                raise ValueError('missing or duplicate candidate/role')
            row = matches[0]
            i = len(observed)
            start, done = events[2*i:2*i+2]
            for event, kind, sequence in ((start, 'start', 2*i), (done, 'complete', 2*i+1)):
                if (type(event) is not dict or not _exact_int(event.get('seq'), sequence)
                        or event.get('kind') != kind or not _exact_int(event.get('action'), i)
                        or event.get('phase') != 'screen' or event.get('object') != candidate
                        or event.get('case_id') != row['id'] or event.get('condition') != 'original'
                        or not _exact_ids(event.get('input_ids'), row['input_ids'])
                        or not _exact_ids(event.get('attention_mask'), row['attention_mask'])
                        or not _exact_int(event.get('mask_index'), row['mask_index'])):
                    raise ValueError('screen identity or model-entry binding differs')
            if (not {'parameter', 'parameter_sha256', 'realized_norm'}.issubset(done)
                    or done['parameter'] is not None or done['parameter_sha256'] is not None
                    or type(done['realized_norm']) is not float or done['realized_norm'] != 0.0):
                raise ValueError('screen action must record the unedited zero-norm baseline')
            values = _f32(done.get('logits'), vocabulary)
            names = {name: row['word_ids'][row[name]]
                     for name in ('factual', 'counterfactual', 'wrong_target')}
            details = _summary(values, names)
            eligible = details['unique_top'] == names['factual']
            candidates[candidate] &= eligible
            observed.append({'case_id': row['id'], 'object': candidate, 'role': role,
                             'eligible': eligible, **details})
    selected = next((candidate for candidate in order if candidates[candidate]), None)
    require_denormal_support()
    return {'manifest_sha256': expected_manifest_sha,
            'journal_sha256': hashlib.sha256(journal_raw).hexdigest(),
            'status': 'eligible' if selected is not None else 'baseline-ineligible',
            'selected_object': selected, 'candidate_eligible': candidates, 'rows': observed,
            'scope': 'Saved full-logit and input-record consistency only; execution, source, '
                     'process identity and uncited history require separate evidence.'}


def main():
    require_denormal_support()  # Before parsing even the command's JSON inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit_bytes(args.manifest.read_bytes(), args.journal.read_bytes(),
                         expected_manifest_sha=PUBLIC_CASE_SHA, vocabulary=30522)
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as file:
        if file.write(payload) != len(payload):
            raise OSError('short write')
    print('screen review SHA-256', hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    main()
