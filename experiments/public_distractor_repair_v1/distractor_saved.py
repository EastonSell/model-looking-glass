"""Model-free replay and fixed discovery/held-out rules for the public study."""
import ctypes
import hashlib
import os
import json
import math
import struct
from collections import Counter
from pathlib import Path
from distractor_analysis import select_candidate, primary_rule, prediction_error

def ensure_round_to_nearest():
    """Model-free thread-mode guard; the interpreter must not change it concurrently."""
    if os.name=='nt':
        control=ctypes.CDLL('msvcrt')._controlfp
        control.argtypes=[ctypes.c_uint,ctypes.c_uint];control.restype=ctypes.c_uint
        if control(0,0)&0x300:raise RuntimeError('floating-point rounding mode is not nearest-even')
    else:
        try:
            getround=ctypes.CDLL(None).fegetround
            getround.argtypes=[];getround.restype=ctypes.c_int
            mode=getround()
        except (AttributeError,OSError) as error:
            raise RuntimeError('cannot inspect floating-point rounding mode') from error
        if mode!=0:raise RuntimeError('floating-point rounding mode is not nearest-even')
    one=1.0;half=float.fromhex('0x1p-53');next_one=float.fromhex('0x1.0000000000001p+0')
    if (one+half!=one or next_one+half!=float.fromhex('0x1.0000000000002p+0')
        or one-float.fromhex('0x1p-54')!=one
        or next_one*1.5!=float.fromhex('0x1.8000000000002p+0')
        or one/10.0!=float.fromhex('0x1.999999999999ap-4')
        or math.sqrt(2.0)!=float.fromhex('0x1.6a09e667f3bcdp+0')):
        raise RuntimeError('binary64 rounding probe failed')


MAP = {1: ((2,'t','downstream'),(0,'f','distractor'),(1,'t','source'),(0,'t','upstream')),
       2: ((2,'t','source'),(1,'t','upstream'),(0,'f','distractor'),(3,'t','downstream')),
       3: ((2,'t','upstream'),(4,'t','downstream'),(3,'t','source'),(0,'f','distractor'))}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _expected(cases):
    result=[]
    for case in cases:
        cid=case['case_id']
        result.extend([(cid,'clean',{}),(cid,'null-self',{}),(cid,'decoy',{})])
        for layer in (1,2,3):
            result += [(cid,'target',{'layer':layer}),(cid,'combined',{'layer':layer})]
            result += [(cid,'patch',{'layer':layer,'candidate':f'Q{i}'}) for i in range(1,5)]
            result += [(cid,'rotation',{'layer':layer}),(cid,'self',{'layer':layer})]
        result.append((cid,'clean-repeat',{}))
    return result


def _plan(kind, details, n):
    t,f=n-2,n-1
    if kind=='null-self':
        return [(name,op,0,pos) for name,op,pos in [('before','capture',None),('null','null',t),
                ('self_source','capture',t),('self_patch','patch',t),('after','capture',None)]]
    layer=details.get('layer',1)
    candidate=MAP[layer][int(details['candidate'][1:])-1] if kind=='patch' else None
    out=[]
    for block in range(5):
        if block==0 and kind!='target':
            out.extend([('decoy_before','capture',block,None),('decoy_add','add',block,f),
                        ('decoy_after','capture',block,None)])
        if block==layer and kind!='decoy':
            out.extend([('target_before','capture',block,None),('target_add','add',block,t),
                        ('target_after','capture',block,None)])
        if candidate is not None and block==candidate[0]:
            out.extend([('patch_before','capture',block,None),
                        ('patch','patch',block,t if candidate[1]=='t' else f),
                        ('patch_after','capture',block,None)])
        if kind=='rotation' and block==layer+1:
            out.extend([('patch_before','capture',block,None),('patch','patch',block,t),
                        ('patch_after','capture',block,None)])
        if kind=='self' and block==layer+1:
            out.extend([('self_source','capture',block,t),('self_before','capture',block,None),
                        ('self_patch','patch',block,t),('self_after','capture',block,None)])
        out.extend([(f'tap_{block}_t','capture',block,t),(f'tap_{block}_f','capture',block,f)])
    return out


def _metadata(record, kind, details, n, hidden, vocab):
    hooks=record['hooks']
    shapes={'logits':[2,vocab]}
    if kind in ('clean','clean-repeat'):
        if len(hooks)!=5: raise ValueError('wrong clean trace event count')
        for block,event in enumerate(hooks):
            if (event.get('sequence')!=block or event.get('layer_index')!=block
                or event.get('module_path')!=f'transformer.h.{block}.mlp'
                or event.get('module_kind')!='mlp' or event.get('invocation')!=0
                or event.get('forward_index')!=0 or event.get('batch_positions')!=[0]
                or event.get('token_positions')!=[n-2,n-1] or event.get('shape')!=[1,n,hidden]
                or event.get('saved_shape')!=[1,2,hidden] or event.get('dtype')!='torch.float32'
                or event.get('generation',{}).get('use_cache') is not False):
                raise ValueError('clean trace metadata differs')
            shapes[f'tap_{block}_t']=[hidden]
            shapes[f'tap_{block}_f']=[hidden]
    else:
        plan=_plan(kind,details,n)
        if len(hooks)!=len(plan): raise ValueError('plan event count differs')
        for order,((name,operation,block,pos),event) in enumerate(zip(plan,hooks)):
            if (event.get('order')!=order or event.get('name')!=name or event.get('operation')!=operation
                or event.get('module_path')!=f'transformer.h.{block}.mlp'
                or event.get('invocation')!=0 or event.get('forward_index')!=0
                or event.get('dtype')!='torch.float32' or event.get('output_shape')!=[1,n,hidden]
                or event.get('batch_index')!=(0 if pos is not None else None)
                or event.get('token_index')!=pos
                or event.get('selected_shape')!=([hidden] if pos is not None else [1,n,hidden])):
                raise ValueError('plan metadata or order differs')
            if name=='self_patch' and event.get('source_capture')!='self_source':
                raise ValueError('self patch source differs')
            if operation=='capture': shapes[name]=[hidden] if pos is not None else [1,n,hidden]
        if kind=='rotation': shapes['rotation_planned']=[hidden]
        if kind=='patch' and MAP[details['layer']][int(details['candidate'][1:])-1][2]=='downstream':
            shapes['realized_displacement']=[hidden]
    if record['shapes']!=shapes:
        raise ValueError('saved tensor names or shapes differ')


def _checks(kind,details):
    if kind=='clean':return {'trace_complete':True}
    if kind=='clean-repeat':return {'logits_equal_clean':True,'vectors_equal_clean':True}
    if kind=='null-self':return {'before_after_equal':True,'self_source_matches':True,'logits_equal_clean':True}
    if kind=='decoy':return {'landing_valid':True,'scored_logits_equal_clean':True}
    if kind=='target':return {'landing_valid':True}
    if kind=='combined':return {'landing_valid':True,'scored_logits_equal_target':True}
    if kind=='rotation':return {'landing_valid':True}
    if kind=='self':return {'landing_valid':True,'self_source_matches':True,'logits_equal_combined':True}
    role=MAP[details['layer']][int(details['candidate'][1:])-1][2]
    out={'landing_valid':True}
    if role=='source':out['scored_logits_equal_clean']=True
    if role in ('upstream','distractor'):out['scored_logits_equal_combined']=True
    if role=='upstream':out['upstream_donor_equal']=True
    if role=='downstream':out['downstream_base_equal_combined']=True
    return out


def load_phase(directory,cases,*,vocab=50257,hidden=768):
    ensure_round_to_nearest()
    root=Path(directory)
    journal=(root/'events.jsonl').read_bytes()
    raw={c['case_id']+suffix:(root/(c['case_id']+suffix)).read_bytes()
         for c in cases for suffix in ('.f32','.f64')}
    if len(raw)!=2*len(cases):raise ValueError('duplicate cases')
    positions={cid:0 for cid in raw}
    case_by_id={c['case_id']:c for c in cases}
    expected=_expected(cases)
    completed={}
    active=record=None
    seen=[]
    for line in journal.splitlines():
        row=json.loads(line)
        event=row['type']
        if event=='start':
            if active is not None or len(completed)>=len(expected):raise ValueError('unexpected start')
            active=(row['case_id'],row['action'],row['details'])
            if active!=expected[len(completed)] or row['ordinal']!=len(completed):
                raise ValueError('fixed action schedule differs')
            record={'tensors':{},'shapes':{},'hooks':None,'rotation_info':None,'displacement_norm':None}
            seen=[]
        elif event=='tensor':
            if (active is None or record['hooks'] is not None or row['ordinal']!=len(completed)
                or (row['case_id'],row['action'])!=active[:2]):
                raise ValueError('unexpected tensor')
            desc=row['tensor'];cid=active[0];shape=desc['shape']
            dtype=desc['dtype'];filename=desc['file']
            if dtype not in ('float32-le','float64-le'):raise ValueError('invalid tensor dtype')
            size=4 if dtype=='float32-le' else 8
            suffix='.f32' if size==4 else '.f64'
            if (filename!=cid+suffix or desc['offset']!=positions[filename]
                or not shape or any(type(x) is not int or x<1 for x in shape)
                or desc['length']!=size*math.prod(shape)):
                raise ValueError('invalid tensor descriptor')
            value=raw[filename][desc['offset']:desc['offset']+desc['length']]
            if len(value)!=desc['length'] or _sha(value)!=desc['sha256']:
                raise ValueError('tensor bytes differ from hash')
            if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f' if size==4 else '<d',value)):
                raise ValueError('nonfinite tensor')
            if desc['name'] in record['tensors']:raise ValueError('duplicate tensor')
            record['tensors'][desc['name']]=value;record['shapes'][desc['name']]=shape
            positions[filename]+=desc['length'];seen.append(desc)
        elif event=='raw-recorded':
            if (active is None or (row.get('case_id'),row.get('action'))!=active[:2]
                or row.get('ordinal')!=len(completed) or row['tensors']!=seen or record['hooks'] is not None):
                raise ValueError('recorded tensor inventory differs')
            record['hooks']=row['hook_events'];record['rotation_info']=row['rotation_info']
            record['displacement_norm']=row['displacement_norm']
        elif event=='complete':
            if (active is None or record['hooks'] is None or row['ordinal']!=len(completed)
                or (row.get('case_id'),row.get('action'))!=active[:2]):
                raise ValueError('unexpected completion')
            cid,kind,details=active
            _metadata(record,kind,details,len(case_by_id[cid]['token_ids']),hidden,vocab)
            if row['checks']!=_checks(kind,details) or any(type(v) is not bool for v in row['checks'].values()):
                raise ValueError('failed or missing required control')
            key=(cid,kind,details.get('layer'),details.get('candidate'))
            if key in completed:raise ValueError('duplicate action')
            completed[key]=record
            active=record=None
        else:raise ValueError('not a complete phase journal')
    if active is not None or len(completed)!=len(expected) or any(positions[k]!=len(v) for k,v in raw.items()):
        raise ValueError('incomplete journal or raw coverage')
    summary=json.loads((root/'phase-summary.json').read_bytes())
    if (summary.get('status')!='complete' or summary.get('cases')!=len(cases)
        or summary.get('completed_actions')!=len(expected)
        or summary.get('raw_bytes')!=sum(map(len,raw.values()))):
        raise ValueError('phase summary differs')
    for case in cases:_audit_case(completed,case,hidden,vocab)
    ensure_round_to_nearest()
    return completed,{'events_sha256':_sha(journal),'actions':len(completed),
        'raw_files':{cid:{'sha256':_sha(data),'bytes':len(data)} for cid,data in raw.items()},
        'raw_bytes':sum(map(len,raw.values()))}


def _rne32(value):
    if not math.isfinite(value):raise ValueError('nonfinite intended value')
    bits=struct.unpack('<Q',struct.pack('<d',value))[0]
    sign=(bits>>32)&0x80000000
    exponent=(bits>>52)&0x7ff;fraction=bits&((1<<52)-1)
    def shifted(v,n):
        if n<=0:return v<<-n
        q,r=divmod(v,1<<n);half=1<<(n-1)
        return q+int(r>half or (r==half and q&1))
    if exponent==0:return sign|shifted(fraction,925)
    power=exponent-1023;sig=(1<<52)|fraction
    if power>=-126:
        rounded=shifted(sig,29)
        if rounded==1<<24:rounded>>=1;power+=1
        if power>127:raise ValueError('float32 overflow')
        return sign|((power+127)<<23)|(rounded&0x7fffff)
    return sign|shifted(sig,-(power+97))


def _values(raw):
    return [x[0] for x in struct.iter_unpack('<f',raw)]


def _distance(a,b):
    if len(a)!=len(b) or len(a)%4:raise ValueError('incompatible vector bytes')
    return math.sqrt(math.fsum((x[0]-y[0])**2 for x,y in zip(struct.iter_unpack('<f',a),struct.iter_unpack('<f',b))))


def _margin(raw):
    return struct.unpack_from('<f',raw,318*4)[0]-struct.unpack_from('<f',raw,389*4)[0]


def _audit_add(tensors,name,pos,hidden):
    before=tensors[name+'_before'];after=tensors[name+'_after']
    offset=pos*hidden*4
    old=struct.unpack_from('<f',before,offset)[0]
    bits=_rne32(old+4.0)
    wanted=struct.pack('<I',bits)
    if wanted==before[offset:offset+4]:raise ValueError('undelivered injection')
    expected=bytearray(before);expected[offset:offset+4]=wanted
    if bytes(expected)!=after:raise ValueError('injection landing or untouched bits differ')


def _audit_case(records,case,hidden,vocab):
    cid=case['case_id'];n=len(case['token_ids']);t=n-2;f=n-1;step=hidden*4;vb=vocab*4
    def rec(kind,layer=None,q=None):return records[(cid,kind,layer,q)]['tensors']
    clean=rec('clean');repeat=rec('clean-repeat');null=rec('null-self');decoy=rec('decoy')
    if (clean['logits']!=repeat['logits'] or clean['logits']!=null['logits']
        or null['before']!=null['after'] or null['self_source']!=null['before'][t*step:(t+1)*step]
        or decoy['logits'][:vb]!=clean['logits'][:vb]):
        raise ValueError('clean, null, repeat or causal decoy control differs')
    if any(clean[f'tap_{k}_{pos}']!=repeat[f'tap_{k}_{pos}'] for k in range(5) for pos in ('t','f')):
        raise ValueError('clean-repeat traces differ')
    _audit_add(decoy,'decoy',f,hidden)
    if decoy['tap_0_f']!=decoy['decoy_after'][f*step:(f+1)*step]:
        raise ValueError('decoy tap is not post-add')
    for layer in (1,2,3):
        target=rec('target',layer);combined=rec('combined',layer)
        if target['logits'][:vb]!=combined['logits'][:vb]:
            raise ValueError('combined scored logits differ from target-only')
        for kind,q in [('target',None),('combined',None)]+[('patch',f'Q{i}') for i in range(1,5)]+[('rotation',None),('self',None)]:
            item=rec(kind,layer,q)
            if kind!='target':_audit_add(item,'decoy',f,hidden)
            _audit_add(item,'target',t,hidden)
            if kind=='combined' and item[f'tap_{layer}_t']!=item['target_after'][t*step:(t+1)*step]:
                raise ValueError('target tap is not post-add')
            if kind=='patch':
                block,pos,role=MAP[layer][int(q[1:])-1];index=t if pos=='t' else f
                donor=clean[f'tap_{block}_{pos}'];expected=bytearray(item['patch_before'])
                expected[index*step:(index+1)*step]=donor
                if bytes(expected)!=item['patch_after']:
                    raise ValueError('candidate patch or untouched bits differ')
                if role=='source' and item['logits'][:vb]!=clean['logits'][:vb]:
                    raise ValueError('source patch scored control differs')
                if role in ('upstream','distractor') and item['logits'][:vb]!=combined['logits'][:vb]:
                    raise ValueError('upstream or future patch scored control differs')
                if role=='upstream' and item['patch_before'][t*step:(t+1)*step]!=donor:
                    raise ValueError('upstream donor differs')
                if role=='downstream':
                    if item['patch_before'][t*step:(t+1)*step]!=combined[f'tap_{layer+1}_t']:
                        raise ValueError('downstream patch base differs from combined state')
                    before=_values(item['patch_before'][t*step:(t+1)*step]);after=_values(item['patch_after'][t*step:(t+1)*step])
                    displacement=[a-b for a,b in zip(after,before)]
                    if item['realized_displacement']!=b''.join(struct.pack('<d',x) for x in displacement):
                        raise ValueError('saved realized displacement differs')
                    norm=math.sqrt(math.fsum(x*x for x in displacement))
                    if records[(cid,kind,layer,q)]['displacement_norm']!=norm:
                        raise ValueError('saved realized displacement norm differs')
            if kind=='self' and (item['self_before']!=item['self_after']
                or item['self_source']!=item['self_before'][t*step:(t+1)*step]
                or item['logits']!=combined['logits']):
                raise ValueError('self-patch control differs')
            if kind=='rotation':
                base=combined[f'tap_{layer+1}_t'];donor=clean[f'tap_{layer+1}_t']
                if item['patch_before'][t*step:(t+1)*step]!=base:
                    raise ValueError('rotation base differs')
                b=_values(base);c=_values(donor);d=[x-y for x,y in zip(c,b)]
                planned=b''.join(struct.pack('<I',_rne32(b[i]+d[(i-1)%hidden])) for i in range(hidden))
                if planned!=item['rotation_planned']:raise ValueError('rotation planned bits differ')
                expected=bytearray(item['patch_before']);expected[t*step:(t+1)*step]=planned
                if bytes(expected)!=item['patch_after']:
                    raise ValueError('rotation landed or untouched bits differ')
                z=_values(planned);delta=[z[i]-b[i] for i in range(hidden)]
                qnorm=math.sqrt(math.fsum(x*x for x in d));rnorm=math.sqrt(math.fsum(x*x for x in delta))
                cosine=math.fsum(x*y for x,y in zip(d,delta))/(qnorm*rnorm) if qnorm>0 and rnorm>0 else None
                info={'clean_norm':qnorm,'rotated_norm':rnorm,'cosine':cosine,
                      'matched':bool(qnorm>0 and rnorm>0 and abs(rnorm-qnorm)/qnorm<=0.001)}
                if records[(cid,kind,layer,q)]['rotation_info']!=info:
                    raise ValueError('rotation norm or cosine report differs')


def _get(records,cid,kind,layer=None,q=None):
    return records[(cid,kind,layer,q)]['tensors']


def discovery(records,cases):
    ensure_round_to_nearest()
    result={}
    for layer in (1,2,3):
        effects=[];magnitudes=[[] for _ in range(4)];votes=Counter()
        for case in cases:
            cid=case['case_id'];clean=_get(records,cid,'clean');combined=_get(records,cid,'combined',layer)
            baseline=_margin(combined['logits']);effects.append([_margin(_get(records,cid,'patch',layer,f'Q{i}')['logits'])-baseline for i in range(1,5)])
            changed=[]
            for block in range(5):
                for position_order,pos in enumerate(('t','f')):
                    name=f'tap_{block}_{pos}'
                    if clean[name]!=combined[name]:changed.append((block,position_order,pos))
            if changed:
                first_block,_,first_pos=min(changed)
                matching=[i for i,(block,pos,_) in enumerate(MAP[layer])
                          if (block,pos)==(first_block,first_pos)]
                if matching:votes[matching[0]]+=1
            for index,(block,pos,_) in enumerate(MAP[layer]):
                name=f'tap_{block}_{pos}'
                magnitudes[index].append(_distance(clean[name],combined[name]))
        selector=select_candidate(effects)
        magnitude=[math.fsum(x)/8 for x in magnitudes]
        order=sorted(range(4),key=lambda i:(-magnitude[i],i))
        most=max(votes.values(),default=0)
        first=next((i for i in range(4) if votes[i]==most),None) if most else None
        digest=hashlib.sha256(f'public-distractor-repair-v1|{layer}'.encode()).digest()
        random_index=int.from_bytes(digest[:8],'big')%4
        result[str(layer)]={'selector':selector,'first_changed':f'Q{first+1}' if first is not None else None,
            'first_changed_votes':[votes[i] for i in range(4)],
            'magnitude':f'Q{order[0]+1}' if magnitude[order[0]]>0 else None,
            'magnitude_scores':magnitude,'constant':'Q3','random':f'Q{random_index+1}',
            'uniform_site_reference':0.25,'causal_mask_reference':[f'Q{i+1}' for i,x in enumerate(MAP[layer]) if x[1]=='t']}
    ensure_round_to_nearest()
    return result


def heldout(records,cases,frozen,*,vocab=50257):
    ensure_round_to_nearest()
    cells={};primary_inputs=[];vb=vocab*4
    for layer in (1,2,3):
        frozen_cell=frozen[str(layer)];per=[];by_q={f'Q{i}':[] for i in range(1,5)}
        combined_dist=[];downstream_dist=[];rotation_dist=[]
        for case in cases:
            cid=case['case_id'];clean=_get(records,cid,'clean');decoy=_get(records,cid,'decoy')
            target=_get(records,cid,'target',layer);combined=_get(records,cid,'combined',layer)
            clean_t,clean_f=clean['logits'][:vb],clean['logits'][vb:]
            combined_t,combined_f=combined['logits'][:vb],combined['logits'][vb:]
            distance=_distance(combined_t,clean_t);combined_dist.append(distance)
            downq=next(f'Q{i+1}' for i,item in enumerate(MAP[layer]) if item[2]=='downstream')
            down=_get(records,cid,'patch',layer,downq);rot=_get(records,cid,'rotation',layer)
            down_d=_distance(down['logits'][:vb],clean_t);rot_d=_distance(rot['logits'][:vb],clean_t)
            downstream_dist.append(down_d);rotation_dist.append(rot_d)
            baseline=_margin(combined_t)
            for i in range(1,5):
                q=f'Q{i}';patched=_get(records,cid,'patch',layer,q)
                effect=_margin(patched['logits'][:vb])-baseline
                by_q[q].append({'case_id':cid,'effect':effect,
                    'distance_to_clean':_distance(patched['logits'][:vb],clean_t)})
            per.append({'case_id':cid,'target_full_logit_distance':_distance(target['logits'][:vb],clean_t),
                'target_contrast_effect':_margin(target['logits'][:vb])-_margin(clean_t),
                'combined_distance':distance,'downstream_distance':down_d,'rotation_distance':rot_d,
                'downstream_reduction':1-down_d/distance if distance>1e-12 else None,
                'rotation_reduction':1-rot_d/distance if distance>1e-12 else None,
                'decoy_final_distance':_distance(decoy['logits'][vb:],clean_f),
                'combined_final_vs_target':_distance(combined_f,target['logits'][vb:]),
                'rotation':records[(cid,'rotation',layer,None)]['rotation_info']})
        primary_inputs.append((combined_dist,downstream_dist))
        mean_combined=math.fsum(combined_dist)/8;mean_rot=math.fsum(rotation_dist)/8
        qsummary={}
        for i in range(1,5):
            q=f'Q{i}';prediction=frozen_cell['selector']['signed_predictions'][i-1]
            qsummary[q]={'prediction':prediction,'accuracy':prediction_error(prediction,[x['effect'] for x in by_q[q]]),
                         'cases':by_q[q]}
        cells[str(layer)]={'cases':per,'candidates':qsummary,
            'rotation_reduction':1-mean_rot/mean_combined if mean_combined>1e-12 else None,
            'target_effect_count':sum(x['target_full_logit_distance']>1e-12 for x in per),
            'selected_source':None if frozen_cell['selector']['selected'] is None else
                MAP[layer][int(frozen_cell['selector']['selected'][1:])-1][2]=='source'}
    output={'primary':primary_rule(primary_inputs),'cells':cells}
    ensure_round_to_nearest()
    return output
