"""Extract only decision-relevant saved logits into a compact JSONL journal.

This reads a complete journal but neither proves its execution nor preserves
full-vocabulary logits, intervention vectors, dose or landing witnesses.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

FIELDS = {
    'clean':('kind','case_id','cue','input_ids'),
    'repeat':('kind','case_id','cue','input_ids'),
    'self':('kind','case_id','site','target','input_ids'),
    'transfer':('kind','case_id','site','donor','target','input_ids'),
    'baseline':('kind','case_id','eligible','gaps'),
    'radius':('kind','case_id','radius','dose_eligible'),
    'dose_ineligible':('kind','case_id','radius'),
}
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path); p.add_argument('journal',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--expected-journal-sha256',required=True)
    a=p.parse_args()
    if sha(a.manifest)!=a.expected_manifest_sha256 or sha(a.journal)!=a.expected_journal_sha256:
        raise ValueError('source bytes differ from supplied hashes')
    manifest=json.loads(a.manifest.read_bytes())
    any_id,some_id=manifest['scored_tokens']['any'],manifest['scored_tokens']['some']
    if any_id==some_id or min(any_id,some_id)<0: raise ValueError('invalid target IDs')
    count=0
    with a.journal.open('rb') as raw_in, gzip.GzipFile(fileobj=raw_in,mode='rb') as incoming:
        with a.output.open('xb') as raw_out, gzip.GzipFile(fileobj=raw_out,mode='wb',filename='',mtime=0) as outgoing:
            for line in incoming:
                if len(line)>2*1024*1024: raise ValueError('oversized input event')
                record=json.loads(line); kind=record['kind']
                if kind not in FIELDS: raise ValueError('unexpected event in complete journal')
                reduced={key:record[key] for key in FIELDS[kind]}
                if kind in ('clean','repeat','self','transfer'):
                    logits=base64.b64decode(record['mask_logits_f32le'],validate=True)
                    if len(logits)%4 or len(logits)<4*(1+max(any_id,some_id)):
                        raise ValueError('malformed original logits')
                    pair=logits[4*any_id:4*any_id+4]+logits[4*some_id:4*some_id+4]
                    if any(not math.isfinite(x) for x in struct.unpack('<ff',pair)):
                        raise ValueError('nonfinite target logits')
                    reduced['mask_logits_pair_f32le']=base64.b64encode(pair).decode('ascii')
                outgoing.write((json.dumps(reduced,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
                count+=1
    if sha(a.manifest)!=a.expected_manifest_sha256 or sha(a.journal)!=a.expected_journal_sha256:
        raise ValueError('source changed during extraction')
    print(json.dumps({'status':'passed','events':count,'manifest_sha256':a.expected_manifest_sha256,
                      'original_journal_sha256':a.expected_journal_sha256,
                      'subset_sha256':sha(a.output),'subset_bytes':a.output.stat().st_size,
                      'scope':'operator-derived target-logit subset, not proof of original execution or full witnesses'},sort_keys=True))

if __name__=='__main__':main()
