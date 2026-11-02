"""Reconstruct a stopped year/label family-dose failure from retained no-op bytes.

The input is a completed attempt, not model execution; Python float32 packing
and binary64 sums reproduce the protocol's recorded calculation on this host.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys

if sys.byteorder!='little' or not __debug__:
    raise RuntimeError('requires little-endian CPU and assertions enabled')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--failure-sha256',required=True)
parser.add_argument('--journal-sha256',required=True)
args=parser.parse_args()
root=args.run
journal=(root/'attempt.jsonl').read_bytes()
failure_bytes=(root/'failure.json').read_bytes()
manifest_bytes=(root/'manifest.json').read_bytes()
assert hashlib.sha256(journal).hexdigest()==args.journal_sha256
assert hashlib.sha256(failure_bytes).hexdigest()==args.failure_sha256
failure_record=json.loads(failure_bytes)
assert failure_record['status']=='incomplete' and failure_record['phase']=='family_dose'
assert failure_record['manifest_sha256']==hashlib.sha256(manifest_bytes).hexdigest()
events=[json.loads(x) for x in journal.splitlines()]
manifest=json.loads(manifest_bytes)
rows=manifest['cases'][12:24]
assert len(rows)==12 and {r['family_index'] for r in rows}=={1}
raw={}
for row in rows:
    event=next(e for e in events if e.get('event')=='complete' and e.get('case_id')==row['case_id'] and e.get('action')=='noop')
    part=event['raw']['block_before']
    with (root/(row['case_id']+'.f32')).open('rb') as f:
        f.seek(part['offset']); data=f.read(part['bytes'])
    assert hashlib.sha256(data).hexdigest()==part['sha256']
    raw[row['condition'],row['label'],row['year_role']]=data
def vec(data):
    result=array.array('f'); result.frombytes(data)
    assert result.itemsize==4 and all(math.isfinite(x) for x in result)
    return result
def norm(values): return math.sqrt(math.fsum(v*v for v in values))
conds=('after','before','near'); labels=('red','blue'); roles=('low','high'); width=768; rowbytes=4*width
vectors={key:vec(data[-rowbytes:]) for key,data in raw.items()}
norms={}
for c in conds:
    assert len({raw[c,l,r][:rowbytes] for l in labels for r in roles})==1
    for l in labels: norms[f'{c}:year:{l}']=norm(b-a for a,b in zip(vectors[c,l,'low'],vectors[c,l,'high']))
    for r in roles: norms[f'{c}:label:{r}']=norm(b-a for a,b in zip(vectors[c,labels[0],r],vectors[c,labels[1],r]))
q=min(norms.values())/2
assert q>2**-20
failure=None
for c in conds:
    for li,l in enumerate(labels):
        for ri,r in enumerate(roles):
            before=raw[c,l,r]; v=vectors[c,l,r]; yr=vectors[c,l,roles[1-ri]]; lex=vectors[c,labels[1-li],r]
            yd=[(q/norms[f'{c}:year:{l}'])*(b-a) for a,b in zip(v,yr)]
            ld=[(q/norms[f'{c}:label:{r}'])*(b-a) for a,b in zip(v,lex)]
            rot=[yd[(i-1)%width] for i in range(width)]
            for action,base,delta in (('year_final',before[-rowbytes:],yd),('label_final',before[-rowbytes:],ld),
                                      ('year_at_zero',before[:rowbytes],yd),('rotated_year_final',before[-rowbytes:],rot)):
                start=vec(base); after=vec(struct.pack('<768f',*(a+d for a,d in zip(start,delta))))
                delivered=norm(a-b for a,b in zip(after,start))
                if abs(delivered-q)>1e-5*q:
                    rounding=[(float(a)-float(b))-d for a,b,d in zip(after,start,delta)]
                    magnitudes=sorted(abs(float(x)) for x in start)
                    def spacing(x):
                        bits=struct.unpack('<I',struct.pack('<f',x))[0]
                        exponent=(bits>>23)&255
                        assert exponent!=255
                        return math.ldexp(1.0,-149 if exponent==0 else exponent-150)
                    spacings=sorted(spacing(float(x)) for x in start)
                    intended=norm(delta)
                    largest_error=max(range(width),key=lambda i:abs(rounding[i]))
                    failure={'condition':c,'label':l,'year_role':r,'action':action,'requested':q,
                             'delivered':delivered,'absolute_error':abs(delivered-q),
                             'relative_error':abs(delivered-q)/q,'relative_tolerance':1e-5,
                             'source_abs_median':statistics.median(magnitudes),
                             'source_abs_p99':magnitudes[int(.99*(width-1))],
                             'source_abs_max':magnitudes[-1],
                             'source_spacing_median':statistics.median(spacings),
                             'source_spacing_p99':spacings[int(.99*(width-1))],
                             'source_spacing_max':spacings[-1],
                             'unrounded_displacement_l2':intended,
                             'rounding_error_l2':norm(rounding),
                             'max_absolute_channel_rounding_error':max(map(abs,rounding)),
                             'largest_error_channel':largest_error,
                             'largest_error_channel_before':float(start[largest_error]),
                             'largest_error_channel_spacing':spacing(float(start[largest_error])),
                             'largest_error_channel_intended_delta':delta[largest_error],
                             'largest_error_channel_actual_delta':float(after[largest_error])-float(start[largest_error]),
                             'rounding_projection_on_direction':math.fsum(d*e for d,e in zip(delta,rounding))/intended}
                    break
            if failure: break
        if failure: break
    if failure: break
assert failure is not None and failure['action']=='year_at_zero'
print(json.dumps({'family_index':1,'norms':norms,'common_dose':q,'first_failure':failure,
                  'scope':'Reconstruction from retained no-op tensors only; no additional forward.'},sort_keys=True))
