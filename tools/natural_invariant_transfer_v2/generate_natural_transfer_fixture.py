"""Generate two natural-transfer phases using only a fabricated 8-wide toy model."""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get('NATURAL_RUNTIME_SOURCE', os.environ['INVARIANT_TEST_OWNER_SOURCE']))

import torch
import natural_transfer_analyze as analyze
import natural_transfer_executor as executor
from looking_glass.agreement_journal import AgreementJournal
from natural_transfer_check_executor import prepared, toy


def save(path, value):
    raw=(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')
    with open(path,'xb') as out: out.write(raw)
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(sys.argv[1])
    root.mkdir()
    mapping, prior, model_sha=prepared()
    mapping_sha=save(root/'toy-mapping.json',mapping)
    prior_events_sha=sha(Path(os.environ['NATURAL_TOY_JOURNAL'])/'events.jsonl')
    torch.set_num_threads(1)
    bindings={x:'1'*64 for x in ('pre_run_sha256','prior_receipt_sha256','latest_exposure_sha256')}
    bindings['mapping_sha256']=mapping_sha
    observations={}
    for phase in ('discovery','evaluation'):
        binding=dict(bindings)
        if phase=='evaluation': binding['freeze_sha256']=freeze_sha
        journal=AgreementJournal(root/phase, {'scope':'fabricated-only'},
                                 max_total_bytes=executor.RAW_LIMIT, max_forward_bytes=4*1024*1024,
                                 max_events=200, max_event_log_bytes=executor.METADATA_LIMIT)
        model=toy.model()
        executor.run_phase(model,journal,mapping,prior,phase,prior_model_sha256=model_sha,
                           binding=binding,custody_check=lambda:None)
        event_sha, terminal_sha=sha(root/phase/'events.jsonl'),sha(root/phase/'terminal.json')
        rows=analyze.load_phase(root/phase,mapping,prior,phase,events_sha256=event_sha,
                                terminal_sha256=terminal_sha,expected_binding=binding,width=8,vocabulary=1500)
        observations[phase]={'events_sha256':event_sha,'terminal_sha256':terminal_sha,'forwards':model.forward_calls}
        if phase=='discovery':
            freeze=analyze.make_freeze(rows,mapping,mapping_sha256=mapping_sha,
                      prior_events_sha256=prior_events_sha,discovery_events_sha256=event_sha)
            freeze_sha=save(root/'freeze.json',freeze)
        else:
            result=analyze.evaluate(rows,freeze)
            result_sha=save(root/'result.json',result)
    meta={'scope':'fabricated toy model only','mapping_sha256':mapping_sha,'prior_events_sha256':prior_events_sha,
          'model_state_sha256':model_sha,'phases':observations,'freeze_sha256':freeze_sha,'result_sha256':result_sha}
    meta_sha=save(root/'fixture.json',meta)
    print(json.dumps({'fixture':str(root),'meta_sha256':meta_sha,'freeze_sha256':freeze_sha,
                      'result_sha256':result_sha,'forwards':192,'selected':freeze['selected'],
                      'eligible':result['eligible'],'primary':result['primary_pattern_supported']},sort_keys=True))


if __name__=='__main__': main()
