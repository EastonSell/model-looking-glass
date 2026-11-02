"""Write discovery freeze or final report from closed saved evidence only."""
import argparse
import hashlib
import json
from pathlib import Path
import distractor_saved as saved

CASE_SHA='811a1f0251ff21ddaa1869ee96c80dfa47beb87a6b6c2f1e3b6505728fee1825'
RESERVATION_SHA='f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9'

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def checked_receipt(attempt,phase,pre_sha,freeze_sha=None):
    path=attempt/(phase+'.receipt.json');receipt=json.loads(path.read_bytes());run=attempt/('run-'+phase)
    if (receipt.get('schema')!='public-distractor-repair-direct-receipt-v1'
        or receipt.get('phase')!=phase or receipt.get('pre_run_sha256')!=pre_sha
        or receipt.get('reservation_sha256')!=RESERVATION_SHA
        or receipt.get('freeze_sha256')!=freeze_sha or receipt.get('accepted_complete_phase') is not True
        or receipt.get('exit_code')!=0 or receipt.get('interrupted') is not False
        or receipt.get('prefix_empty_after') is not True or receipt.get('stage_empty_after') is not True
        or receipt.get('failure_file_present') is not False):
        raise ValueError('direct-child receipt is incomplete')
    files=receipt['run_files']
    if set(files)!={x.name for x in run.iterdir() if x.is_file()}:raise ValueError('run file inventory differs')
    for name,info in files.items():
        file=run/name
        if file.is_symlink() or file.stat().st_size!=info['bytes'] or sha(file)!=info['sha256']:
            raise ValueError('receipt-bound file differs')
    return sha(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('discovery','result'))
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--pre-run-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze-sha256')
    args=parser.parse_args();attempt=args.attempt
    if sha(args.cases)!=CASE_SHA or sha(attempt/'pre-run.json')!=args.pre_run_sha256:
        raise ValueError('case or pre-run pin differs')
    cases=json.loads(args.cases.read_bytes())['cases']
    receipt_sha=checked_receipt(attempt,'discovery',args.pre_run_sha256)
    records,evidence=saved.load_phase(attempt/'run-discovery',cases[:8])
    selections=saved.discovery(records,cases[:8])
    if args.mode=='discovery':
        if args.freeze_sha256 is not None:raise ValueError('discovery cannot consume a freeze')
        output={'schema':'public-distractor-discovery-freeze-v1','case_sha256':CASE_SHA,
            'pre_run_sha256':args.pre_run_sha256,'reservation_sha256':RESERVATION_SHA,
            'discovery_receipt_sha256':receipt_sha,'evidence':evidence,'discovery':selections}
    else:
        freeze=attempt/'discovery-freeze.json'
        if not args.freeze_sha256 or sha(freeze)!=args.freeze_sha256:raise ValueError('freeze pin differs')
        frozen=json.loads(freeze.read_bytes())
        if (frozen.get('discovery_receipt_sha256')!=receipt_sha or frozen.get('evidence')!=evidence
            or frozen.get('discovery')!=selections or frozen.get('case_sha256')!=CASE_SHA
            or frozen.get('pre_run_sha256')!=args.pre_run_sha256):raise ValueError('freeze does not match discovery')
        held_sha=checked_receipt(attempt,'held-out',args.pre_run_sha256,args.freeze_sha256)
        held,held_evidence=saved.load_phase(attempt/'run-held-out',cases[8:])
        output={'schema':'public-distractor-repair-result-v1','case_sha256':CASE_SHA,
            'pre_run_sha256':args.pre_run_sha256,'discovery_freeze_sha256':args.freeze_sha256,
            'discovery_receipt_sha256':receipt_sha,'heldout_receipt_sha256':held_sha,
            'discovery':selections,'heldout_evidence':held_evidence,
            'heldout':saved.heldout(held,cases[8:],selections)}
    data=(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with args.output.open('xb') as stream:stream.write(data)
    print(json.dumps({'sha256':hashlib.sha256(data).hexdigest(),'mode':args.mode},sort_keys=True))

if __name__=='__main__':main()
