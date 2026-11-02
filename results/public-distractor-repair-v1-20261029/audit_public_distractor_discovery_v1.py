"""Audit the public study's discovery bytes before any held-out execution.

This uses the independent saved-byte reader beside this file. It checks local
evidence and the supplied freeze bytes; it does not establish that the freeze
was public at a particular time or attest process execution.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import audit_public_distractor_repair_v1 as audit


def equal(actual, expected):
    if type(expected) is dict:
        return type(actual) is dict and set(actual) == set(expected) and all(equal(actual[k],v) for k,v in expected.items())
    if type(expected) is list:
        return type(actual) is list and len(actual) == len(expected) and all(equal(x,y) for x,y in zip(actual,expected))
    if type(expected) is float:
        return type(actual) is float and math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-15)
    return type(actual) is type(expected) and actual == expected


def verify_binding(directory, cases_path, freeze, pre_run_path, receipt_path):
    pre_run = audit.parse(pre_run_path.read_bytes())
    receipt = audit.parse(receipt_path.read_bytes())
    pre_sha = hashlib.sha256(pre_run_path.read_bytes()).hexdigest()
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    case_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    audit.require(pre_run['schema'] == 'public-distractor-repair-pre-run-v1'
                  and pre_run['case_sha256'] == case_sha
                  and case_sha == '811a1f0251ff21ddaa1869ee96c80dfa47beb87a6b6c2f1e3b6505728fee1825'
                  and pre_run['protocol_sha256'] == '6d87a579eb27f214db3e8769232b5940d263da59490480765118174ef99b5d70',
                  'pre-run case or protocol identity')
    reservation = pre_run['reservation_sha256']
    audit.require(reservation == 'f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9',
                  'reservation identity')
    if freeze is not None:
        audit.require(freeze['case_sha256'] == case_sha and freeze['pre_run_sha256'] == pre_sha
                      and freeze['reservation_sha256'] == reservation
                      and freeze['discovery_receipt_sha256'] == receipt_sha, 'freeze bindings')
    audit.require(receipt['schema'] == 'public-distractor-repair-direct-receipt-v1'
                  and receipt['phase'] == 'discovery' and receipt['pre_run_sha256'] == pre_sha
                  and receipt['reservation_sha256'] == reservation and receipt['freeze_sha256'] is None
                  and receipt['accepted_complete_phase'] is True and receipt['exit_code'] == 0
                  and receipt['interrupted'] is False and receipt['prefix_empty_after'] is True
                  and receipt['stage_empty_after'] is True and receipt['failure_file_present'] is False
                  and receipt['stderr']['bytes'] == 0 and receipt['source_revision'] == pre_run['source_revision']
                  and receipt['source_tree'] == pre_run['source_tree'], 'direct-child receipt')
    files = receipt['run_files']
    audit.require(set(files) == {p.name for p in directory.iterdir()}, 'receipt file inventory')
    for name, item in files.items():
        file = directory/name
        raw = file.read_bytes()
        audit.require(not file.is_symlink() and len(raw) == item['bytes']
                      and hashlib.sha256(raw).hexdigest() == item['sha256'], 'receipt file hash')
    for stream in ('stdout','stderr'):
        log = directory.parent/('discovery.'+stream+'.txt')
        raw = log.read_bytes()
        audit.require(len(raw) == receipt[stream]['bytes']
                      and hashlib.sha256(raw).hexdigest() == receipt[stream]['sha256'], 'receipt log hash')
    return {'pre_run_sha256': pre_sha, 'receipt_sha256': receipt_sha,
            'scope':'saved cooperative direct-child receipt, not independent process attestation'}


def run(directory, cases_path, freeze_path, hidden, vocab, pre_run_path=None, receipt_path=None):
    audit.nearest()
    cases = audit.parse(cases_path.read_bytes())['cases']
    discovery = [c for c in cases if c['split'] == 'discovery']
    audit.require(len(cases) == 16 and len(discovery) == 8, 'discovery case inventory')
    saved = audit.read_phase(directory, discovery, hidden, vocab, 'discovery')
    per_case = {c['case_id']: audit.audit_case(saved,c,hidden,vocab) for c in discovery}
    del saved
    raw_files = {}
    for case in discovery:
        for ext in ('.f32','.f64'):
            name = case['case_id']+ext
            raw = (directory/name).read_bytes()
            raw_files[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    events = (directory/'events.jsonl').read_bytes()
    evidence = {'actions': 224, 'events_sha256': hashlib.sha256(events).hexdigest(),
                'raw_files': raw_files, 'raw_bytes': sum(x['bytes'] for x in raw_files.values())}
    computed = {}
    for layer in (1,2,3):
        rows = [x[layer] for x in per_case.values()]
        absolute = [math.fsum(abs(row['effects'][i]) for row in rows)/8 for i in range(4)]
        signed = [math.fsum(row['effects'][i] for row in rows)/8 for i in range(4)]
        ordering = sorted(range(4), key=lambda i:(-absolute[i],i))
        magnitude = [math.fsum(row['magnitude'][i] for row in rows)/8 for i in range(4)]
        votes = [sum(row['first_changed'] == [block,pos] for row in rows)
                 for block,pos,_ in audit.MAP[layer]]
        q = lambda i: f'Q{i+1}'
        random = int.from_bytes(hashlib.sha256(f'public-distractor-repair-v1|{layer}'.encode()).digest()[:8], 'big')%4
        computed[str(layer)] = {
            'selector': {'selected': q(ordering[0]) if absolute[ordering[0]] > 0 else None,
                         'ranking': [q(i) for i in ordering], 'mean_absolute_effects': absolute,
                         'signed_predictions': signed,
                         'max_tie': [q(i) for i in ordering if absolute[i] == absolute[ordering[0]]],
                         'abstained': absolute[ordering[0]] == 0},
            'first_changed': q(votes.index(max(votes))) if max(votes) else None,
            'first_changed_votes': votes,
            'magnitude': q(magnitude.index(max(magnitude))) if max(magnitude) > 0 else None,
            'magnitude_scores': magnitude, 'constant': 'Q3', 'random': q(random),
            'uniform_site_reference': .25,
            'causal_mask_reference': [q(i) for i,(_,pos,_) in enumerate(audit.MAP[layer]) if pos == 't']}
    freeze_bytes = freeze_path.read_bytes() if freeze_path is not None else None
    freeze = audit.parse(freeze_bytes) if freeze_bytes is not None else None
    if freeze is not None:
        audit.require(freeze['schema'] == 'public-distractor-discovery-freeze-v1', 'freeze schema')
        audit.require(equal(freeze['evidence'], evidence), 'freeze evidence')
        audit.require(equal(freeze['discovery'], computed), 'discovery rules or predictions')
        for layer in ('1','2','3'):
            for recorded, expected in zip(freeze['discovery'][layer]['selector']['signed_predictions'],
                                          computed[layer]['selector']['signed_predictions']):
                audit.require(struct.pack('<d', recorded) == struct.pack('<d', expected), 'exact frozen prediction')
    audit.nearest()
    audit.require((pre_run_path is None) == (receipt_path is None), 'supply pre-run and receipt together')
    binding = verify_binding(directory, cases_path, freeze, pre_run_path, receipt_path) if pre_run_path else None
    return {'scope':'discovery saved-byte review only; publication chronology is separate',
            'actions_checked':224, 'cases_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            'freeze_sha256':hashlib.sha256(freeze_bytes).hexdigest() if freeze_bytes is not None else None,
            'binding':binding, 'evidence':evidence, 'discovery':computed, 'per_case':per_case}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('cases',type=Path)
    parser.add_argument('freeze',type=Path,nargs='?')
    parser.add_argument('--hidden',type=int,default=768)
    parser.add_argument('--vocab',type=int,default=50257)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pre-run',type=Path)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    report=run(args.directory,args.cases,args.freeze,args.hidden,args.vocab,args.pre_run,args.receipt)
    if args.output:
        with args.output.open('xb') as out:
            out.write((json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'actions_checked':report['actions_checked'], 'freeze_sha256':report['freeze_sha256'],
                      'selected':[report['discovery'][str(i)]['selector']['selected'] for i in (1,2,3)]}))


if __name__ == '__main__':
    main()
