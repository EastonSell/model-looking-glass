"""Independent saved-byte arithmetic for the public known-injection study.

This script reads closed records and frozen values, without importing the
producer or its analysis module. Its report is not process attestation.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import audit_public_known_injection_v1 as audit


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def calculate(held, cases, frozen, ids):
    output = {}
    for cell in ('A', 'B'):
        predictions = frozen[cell]['selector']['signed_predictions']
        by_site = {}
        for site in audit.SITES:
            observations = []
            for case in cases:
                cid = case['case_id']
                clean = audit.get(held, cid, 'clean')['logits']
                injected = audit.get(held, cid, 'injected', cell)['logits']
                patched = audit.get(held, cid, 'patch', cell, site)['logits']
                effect = audit.margin(patched, ids) - audit.margin(injected, ids)
                inj_distance = audit.distance(injected, clean)
                patch_distance = audit.distance(patched, clean)
                prediction = predictions[site - 1]
                observations.append({
                    'case_id': cid, 'actual_effect': effect, 'prediction': prediction,
                    'absolute_error': abs(prediction - effect), 'zero_absolute_error': abs(effect),
                    'distance_injected': inj_distance, 'distance_patch': patch_distance,
                    'distance_ratio': patch_distance / inj_distance if inj_distance > 1e-12 else None,
                })
            n = len(observations)
            mean = lambda field: math.fsum(item[field] for item in observations) / n
            mae, zero = mean('absolute_error'), mean('zero_absolute_error')
            di, dp = mean('distance_injected'), mean('distance_patch')
            by_site[str(site)] = {
                'cases': observations, 'prediction_mae': mae, 'zero_mae': zero,
                'prediction_success': mae < zero, 'mean_distance_injected': di,
                'mean_distance_patch': dp,
                'restoration_success': None if di <= 1e-12 else dp < di,
                'undefined_ratio_count': sum(item['distance_ratio'] is None for item in observations),
            }
        cell_freeze = frozen[cell]
        output[cell] = {'by_site': by_site, 'choices': {
            'selector': cell_freeze['selector']['selected_site'],
            'first_changed': cell_freeze['first_changed_site'],
            'magnitude': cell_freeze['magnitude_site'],
            'constant': cell_freeze['constant_site'], 'random': cell_freeze['random_site'],
        }}
    return output


def check_capture_sources(data, cases, hidden):
    for case in cases:
        cid = case['case_id']
        offset = (len(case['token_ids']) - 1) * hidden * 4
        clean = audit.get(data, cid, 'clean')
        null = audit.get(data, cid, 'null-self')
        if null['self_source'] != null['before'][offset:offset + hidden * 4]:
            raise ValueError('null self-patch source differs')
        for cell, site in (('A', 1), ('B', 4)):
            for kind, candidate in [('injected', None)] + [('patch', s) for s in audit.SITES] + [('injected-self', None)]:
                action = audit.get(data, cid, kind, cell, candidate)
                if action['inject_before'][offset:offset + hidden * 4] != clean[f'tap_{site}']:
                    raise ValueError('pre-injection vector differs from the clean reference')
                if kind == 'injected-self' and action['self_source'] != action['self_before'][offset:offset + hidden * 4]:
                    raise ValueError('injected self-patch source differs')


def check_receipt(root, phase, expected_sha, cases):
    path = root / (phase + '.receipt.json')
    if sha_file(path) != expected_sha:
        raise ValueError('receipt digest differs')
    receipt = json.loads(path.read_bytes())
    if (receipt.get('schema') != 'public-known-injection-direct-receipt-v1'
        or receipt.get('phase') != phase or type(receipt.get('exit_code')) is not int
        or receipt['exit_code'] != 0 or receipt.get('interrupted') is not False
        or receipt.get('accepted_complete_phase') is not True
        or receipt.get('failure_file_present') is not False
        or receipt.get('prefix_empty_after') is not True or receipt.get('stage_empty_after') is not True
        or receipt.get('pre_run_sha256') != 'b4f632ed69f0b20ceb82a176ee64047a815771231072ec04bb60dc880c18d6c6'
        or receipt.get('reservation_sha256') != 'e5ae202c1e3449a67e92788a83cd0dc62d404a3fa1e052716a11770ee92cc00c'
        or receipt.get('source_revision') != 'ea3f1e78e9074e3455087919d96c84d7c20db0cc'
        or receipt.get('freeze_sha256') != (None if phase == 'discovery' else
            '5c1d1d0f9db7a5b9daca7cd6784e231f442456a25a8ecac5b99f0bd8d7645152')):
        raise ValueError('receipt identity or status differs')
    start = datetime.fromisoformat(receipt['start_observed_utc'])
    end = datetime.fromisoformat(receipt['reaped_observed_utc'])
    if start >= end:
        raise ValueError('receipt ordering differs')
    if phase == 'held-out' and start <= datetime.fromisoformat('2026-10-29T11:43:31+00:00'):
        raise ValueError('held-out start predates independent public readback')
    for stream in ('stdout', 'stderr'):
        path = root / (phase + '.' + stream + '.txt')
        if path.stat().st_size != receipt[stream]['bytes'] or sha_file(path) != receipt[stream]['sha256']:
            raise ValueError('stream differs from receipt')
    if receipt['stderr']['bytes'] != 0:
        raise ValueError('stderr not empty')
    phase_root = root / ('run-' + phase)
    expected = {'events.jsonl', 'phase-summary.json'} | {c['case_id'] + '.f32' for c in cases}
    if set(receipt['run_files']) != expected or {p.name for p in phase_root.iterdir()} != expected:
        raise ValueError('phase inventory differs')
    for name, meta in receipt['run_files'].items():
        path = phase_root / name
        if not path.is_file() or path.stat().st_size != meta['bytes'] or sha_file(path) != meta['sha256']:
            raise ValueError('phase artifact differs from receipt')
    inv = audit.evidence_inventory(phase_root, cases)
    summary = json.loads((phase_root / 'phase-summary.json').read_bytes())
    if summary != {'phase': phase, 'cases': 8, 'completed_actions': 120,
                   'raw_bytes': inv['raw_bytes'],
                   'metadata_bytes': (phase_root / 'events.jsonl').stat().st_size, 'status': 'complete'}:
        raise ValueError('phase summary differs')
    return receipt, inv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, case_path = args.attempt, args.cases
    freeze_path = root / 'discovery-freeze.json'
    if (sha_file(case_path) != 'f1e1043fc302e701b1224248122efc7f79938665ce12cd48ebb7dd713b353789'
        or sha_file(freeze_path) != '5c1d1d0f9db7a5b9daca7cd6784e231f442456a25a8ecac5b99f0bd8d7645152'):
        raise ValueError('case or freeze differs')
    cases = json.loads(case_path.read_bytes())['cases']
    freeze = json.loads(freeze_path.read_bytes())
    dr, di = check_receipt(root, 'discovery', 'a53fdf018647dc281eddbc78eb8225d437f7156d56745e520d1d944de6fa1a3a', cases[:8])
    hr, hi = check_receipt(root, 'held-out', '577b99d302bce2e56ed1bd3ce4ddc30ab622fc4fa2297796b46a5b125b426e88', cases[8:])
    if freeze.get('evidence') != di:
        raise ValueError('discovery freeze evidence differs')
    discovery = audit.read_phase(root / 'run-discovery', cases[:8], 768, 50257)
    held = audit.read_phase(root / 'run-held-out', cases[8:], 768, 50257)
    audit.audit_controls(discovery, cases[:8], 768)
    audit.audit_controls(held, cases[8:], 768)
    check_capture_sources(discovery, cases[:8], 768)
    check_capture_sources(held, cases[8:], 768)
    frozen = freeze['discovery']
    computed = calculate(held, cases[8:], frozen, (318, 389))
    audit.audit_science(discovery, held, cases, frozen, computed, (318, 389))
    if sha_file(args.result) != '6b901f802853fd6f665129675f75a8f6b7c3dbdfd53ef54a4d1651bbde42113e':
        raise ValueError('reported result digest differs')
    result = json.loads(args.result.read_bytes())
    if (result.get('schema') != 'public-known-injection-result-v1'
        or result.get('discovery') != frozen or result.get('heldout') != computed
        or result.get('heldout_evidence') != hi
        or result.get('discovery_freeze_sha256') != sha_file(freeze_path)
        or result.get('heldout_receipt_sha256') != sha_file(root / 'held-out.receipt.json')
        or result.get('pre_run_sha256') != dr['pre_run_sha256']):
        raise ValueError('reported result differs from independent recomputation')
    report = {
        'schema': 'independent-public-known-injection-heldout-computation-v1',
        'status': 'saved-data-consistent', 'cases': 16, 'actions': len(discovery) + len(held),
        'freeze_sha256': sha_file(freeze_path), 'discovery_receipt_sha256': sha_file(root / 'discovery.receipt.json'),
        'heldout_receipt_sha256': sha_file(root / 'held-out.receipt.json'),
        'reported_result_sha256': sha_file(args.result), 'reported_result_matches': True,
        'discovery_evidence': di, 'heldout_evidence': hi,
        'discovery_start_observed_utc': dr['start_observed_utc'],
        'heldout_start_observed_utc': hr['start_observed_utc'],
        'reader_sha256': sha_file(Path(audit.__file__)),
        'recompute_source_sha256': sha_file(Path(__file__)), 'held-out': computed,
        'scope': 'independent saved bytes and arithmetic; cooperative chronology is reported, not authenticated',
    }
    out = args.output
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    summary = {}
    for cell in ('A', 'B'):
        summary[cell] = {'choices': computed[cell]['choices'], 'by_site': {
            site: {key: value for key, value in data.items() if key != 'cases'}
            for site, data in computed[cell]['by_site'].items()}}
    print(json.dumps({'report_sha256': sha_file(out), 'actions': report['actions'],
                      'summary': summary}, sort_keys=True))


if __name__ == '__main__':
    main()
