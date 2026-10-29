"""Write model-free donor-context summary, edge, baseline, effect and TV tables."""
import argparse, csv, hashlib, json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def value(x):
    if isinstance(x,(list,dict)):
        return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
    return x


def write(path, rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for row in rows: writer.writerow({k:value(v) for k,v in row.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('result',type=Path); p.add_argument('geometry',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--expected-result-sha256',required=True); p.add_argument('--expected-geometry-sha256',required=True)
    a=p.parse_args()
    if sha(a.result)!=a.expected_result_sha256 or sha(a.geometry)!=a.expected_geometry_sha256:
        raise ValueError('fixed inputs differ')
    result=json.loads(a.result.read_bytes()); geometry=json.loads(a.geometry.read_bytes())
    if geometry['result_sha256']!=a.expected_result_sha256 or geometry['status']!='passed':
        raise ValueError('geometry report is not bound to result')
    lookup={}
    for g in geometry['rows']:
        key=(g['pair_id'],g['target_context'],g['target_cue'],g['donor_cue'])
        if key in lookup: raise ValueError('duplicate geometry edge')
        lookup[key]=g
    summaries=[]; baselines=[]; edges=[]; effects=[]; tv=[]
    for pair in result['pairs']:
        prefix={'pair_id':pair['id'],'split':pair['split']}
        summary={**prefix,**{k:v for k,v in pair.items() if k not in ('retention','baselines','distances','effects','tv_summary')}}
        summary.update({k:v for k,v in pair['retention'].items() if k!='edges'})
        summaries.append(summary)
        baselines.extend({**prefix,**row} for row in pair['baselines'])
        effects.extend({**prefix,**row} for row in pair['effects'])
        tv.extend({**prefix,**row} for row in pair['tv_summary'])
        for edge in pair['retention']['edges']:
            key=(pair['id'],edge['target_context'],edge['target_cue'],edge['donor_cue'])
            if key not in lookup: raise ValueError('missing geometry edge')
            g=lookup.pop(key)
            if g['within_distance']!=edge['within_distance'] or g['alternate_distance']!=edge['alternate_distance']:
                raise ValueError('geometry distance differs')
            edges.append({**prefix,**edge,'direction_cosine':g['direction_cosine'],
                          'within_transfer':g['within_transfer'],'alternate_transfer':g['alternate_transfer']})
    if lookup or len(summaries)!=4 or len(edges)!=64 or len(baselines)!=32:
        raise ValueError('unexpected pair, baseline or edge count')
    if sha(a.result)!=a.expected_result_sha256 or sha(a.geometry)!=a.expected_geometry_sha256:
        raise ValueError('input changed during read')
    a.output.mkdir(parents=False,exist_ok=False)
    for name,rows in [('pairs.csv',summaries),('baselines.csv',baselines),('edges.csv',edges),('effects.csv',effects),('tv.csv',tv)]:
        write(a.output/name,rows)
    print(json.dumps({'pairs':len(summaries),'baselines':len(baselines),'edges':len(edges),'effects':len(effects),'tv':len(tv)},sort_keys=True))


if __name__=='__main__': main()
