"""Independent post-run scalar/bit spot-check of the closed public v4 intervention.

Reads quiescent saved files only.  Does not attest execution, source provenance,
publication chronology, image provenance, or absence of prior model exposure.
"""
import hashlib
import json
import math
import pathlib
import struct

ROOT = pathlib.Path('work/vision-recognized-v4-intervention-01/data')
SCREEN = pathlib.Path('work/vision-recognized-v4-screen-stage-01/screen-result.json')
SCREEN_DATA = pathlib.Path('work/vision-recognized-v4-screen-01/data')

def require(value, message):
    if not value:
        raise ValueError(message)

def require_nearest():
    # Evaluate nonconstant operands at each boundary; this is a point sample,
    # not a guarantee about unobserved changes during the intervening work.
    one=float.fromhex('0x1p0')
    delta=float.fromhex('0x1.8p-52')
    expected=float.fromhex('0x1.0000000000002p0')
    require(one+delta==expected and -one-delta==-expected, 'binary64 not nearest-even')

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def close(a, b, name, tol=2e-12):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, f'{name}: {a} != {b}')

def load_json(path):
    raw = path.read_bytes()
    return raw, json.loads(raw)

def unpack(raw):
    require(len(raw) % 4 == 0, 'bad float bytes')
    values = struct.unpack('<' + str(len(raw)//4) + 'f', raw)
    require(all(math.isfinite(x) for x in values), 'nonfinite')
    return values

def scores(raw):
    z = unpack(raw)
    require(len(z) == 1000, 'wrong logits')
    def lse(xs):
        maximum = max(xs)
        return maximum + math.log(math.fsum(math.exp(x-maximum) for x in xs))
    dog, cat = lse(z[151:269]), lse(z[281:286])
    total = lse(z)
    off = [z[i] for i in range(1000) if not (151 <= i < 269 or 281 <= i < 286)]
    ot = lse(off)
    return {'margin':dog-cat, 'dog_mass':math.exp(dog-total), 'cat_mass':math.exp(cat-total),
            'off_mass':math.exp(ot-total), 'top1_id':max(range(1000), key=lambda i:z[i]),
            'off':tuple(math.exp(x-ot) for x in off)}

def tensor(row, raw, name):
    d = row['layout'][name]
    a,b = d['offset'], d['offset'] + d['bytes']
    require(type(a) is int and 0 <= a <= b <= len(raw), 'bad segment')
    return raw[a:b]

def block(raw, ids):
    return b''.join(raw[i*768:(i+1)*768] for i in ids)

def dist(a,b):
    return math.sqrt(math.fsum((x-y)**2 for x,y in zip(unpack(a),unpack(b))))

def requested(own, donor, q, distance):
    x,y=unpack(own),unpack(donor)
    scale=q/distance
    return b''.join(struct.pack('<f', a+scale*(b-a)) for a,b in zip(x,y))

def random_requested(own,q,pair,side):
    x=unpack(own)
    unit=q/math.sqrt(len(x))
    def sign(i):
        b=hashlib.sha256(f'vision-coco-small-dose-random-v2:{pair}:{side}:{i}'.encode('ascii')).digest()[0]
        return 1.0 if b & 1 else -1.0
    return b''.join(struct.pack('<f',v+unit*sign(i)) for i,v in enumerate(x))

def main():
    require_nearest()
    terminal_raw,t = load_json(ROOT/'terminal.json')
    screen_raw,s = load_json(SCREEN)
    require(digest(terminal_raw)=='479f1ea67efc001457427c7337313df31fa13f095ea2250ff6bfc9d99331fd1b', 'terminal identity')
    require(digest(screen_raw)=='711894680eac6bed8e60b8acb4413775cce51f5630115e561ecbfe0504efa648', 'screen identity')
    require(t['status']=='complete' and t['source_revision']=='9d90d90176a0903c47bce584c13c839038a88bb8', 'terminal status/source')
    require(t['saved_forwards']==t['submissions']==102, 'forward count')
    require(t['screen_result_sha256']==digest(screen_raw), 'screen linkage')
    require(t['manifest_sha256']=='21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb', 'manifest')
    require(t['model_state_sha256_before_after']=='d1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd', 'state')
    _,start=load_json(ROOT/'started.json')
    _,screen_start=load_json(SCREEN_DATA/'started.json')
    require(start['source']['revision']==t['source_revision'] and start['source']['sources']==screen_start['source']['sources'], 'execution source drift')
    require(start['runtime']==screen_start['runtime']==t['runtime'] and start['model_state_sha256']==screen_start['model_state_sha256']==t['model_state_sha256_before_after'], 'runtime or model state drift')
    require(len(s['pairs']) == len(t['pairs']) == 4, 'pair count')
    expected = {f'forward-{i:03d}.{ext}' for i in range(1,103) for ext in ('json','bin')}
    expected |= {'started.json','terminal.json'} | {f'pair-{i:02d}.json' for i in range(4)}
    require({p.name for p in ROOT.iterdir()} == expected, 'unexpected data file')
    rows={}; maximum_score_error=0.0
    for i in range(1,103):
        _, row=load_json(ROOT/f'forward-{i:03d}.json')
        raw=(ROOT/f'forward-{i:03d}.bin').read_bytes()
        require(row['submission']==i and row['file']==f'forward-{i:03d}.bin' and row['sha256']==digest(raw), f'row hash {i}')
        segments=sorted((v['offset'],v['offset']+v['bytes']) for v in row['layout'].values())
        require(segments[0][0]==0 and segments[-1][1]==len(raw) and all(a[1]==b[0] for a,b in zip(segments,segments[1:])), f'layout {i}')
        sc=scores(tensor(row,raw,'logits'))
        for k in ('margin','dog_mass','cat_mass','off_mass'):
            maximum_score_error=max(maximum_score_error,abs(sc[k]-row['score'][k]))
            close(sc[k],row['score'][k],f'{i} {k}')
        require(sc['top1_id']==row['score']['top1_id'],f'top1 {i}')
        require((row['label'].endswith(':initial')) == (i <= 14), 'initial forwards not complete before patches')
        require(row['label'] not in rows, 'duplicate label')
        rows[row['label']]=(row,raw,sc)
    screens={}
    for i in range(1,65):
        _,row=load_json(SCREEN_DATA/f'forward-{i:03d}.json')
        raw=(SCREEN_DATA/f'forward-{i:03d}.bin').read_bytes()
        require(digest(raw)==row['sha256'] and row['metadata']['image_id'] not in screens, 'screen row')
        screens[row['metadata']['image_id']]=raw
    clean={}
    for label,(row,raw,sc) in rows.items():
        pair,role,condition=label.split(':')
        if condition!='initial': continue
        selected=s['pairs'][int(pair[-2:])]
        image_id=selected[role]['image_id']
        require(image_id not in clean and tensor(row,raw,'logits')==screens[image_id], 'screen-clean bit mismatch')
        clean[image_id]=(row,raw,sc)
    selected_ids={p[role]['image_id'] for p in s['pairs'] for role in ('cat','dog','cat_donor','dog_donor')}
    require(set(clean)==selected_ids and len(clean)==14, 'clean image set')
    report=[]
    for index,selected in enumerate(s['pairs']):
        pair=selected['pair_id']; _,p=load_json(ROOT/f'{pair}.json')
        require(p==t['pairs'][index] and p['pair_id']==pair, 'pair record mismatch')
        ids=selected['box_tokens']; require(len(ids)==4 and len(set(ids))==4 and all(1<=v<197 for v in ids), 'tokens')
        cat=clean[selected['cat']['image_id']][2]; dog=clean[selected['dog']['image_id']][2]
        require(cat['margin']<0 and dog['margin']>0 and cat['cat_mass']>=.2 and dog['dog_mass']>=.2, 'baseline signs/mass')
        require(dog['margin']-cat['margin']>=1 and p['baseline_eligibility']['eligible'] is True and p['baseline_eligibility']['reasons']==[], 'baseline gap')
        close(dog['margin']-cat['margin'],p['baseline_eligibility']['gap'],'baseline gap value')
        side_results={}
        for side,other in [('cat','dog'),('dog','cat')]:
            ident=selected[side]['image_id']; donor=selected[side+'_donor']['image_id']; oident=selected[other]['image_id']
            init,init_raw,baseline=clean[ident]; orow,oraw,_=clean[oident]; drow,draw,_=clean[donor]
            early=tensor(init,init_raw,'early'); other_early=tensor(orow,oraw,'early'); donor_early=tensor(drow,draw,'early')
            own=block(early,ids); other_block=block(other_early,ids); donor_block=block(donor_early,ids)
            norm=dist(own,b'\x00'*len(own)); od=dist(own,other_block); sd=dist(own,donor_block); q=min(.099*norm,od,sd)
            require(min(norm,od,sd,q)>1e-6, 'dose ineligible')
            info=p['sides'][side]; require(info['image_id']==ident and info['same_species_donor_id']==donor, 'identity')
            for name,v in [('receiver_norm',norm),('opposite_distance',od),('same_species_distance',sd),('q',q)]:
                close(v,info['dose'][name],f'{pair} {side} {name}',1e-10)
            require(info['dose']['status']=='matchable','unmatchable')
            wanted={'self_inside':own,'final_inside':block(tensor(orow,oraw,'late'),ids),'natural_inside':other_block,
                    'matched_opposite':requested(own,other_block,q,od),
                    'matched_same_species':requested(own,donor_block,q,sd),
                    'matched_random':random_requested(own,q,pair,side)}
            changes={}; deliveries={}
            for name in ('repeat','null_inside','self_inside','full_self','full_swap','natural_inside','matched_opposite','matched_same_species','matched_random','final_inside','post_clean'):
                row,raw,sc=rows[f'{pair}:{side}:{name}']; cond=info['conditions'][name]
                for k in ('margin','dog_mass','cat_mass','off_mass'):
                    close(sc[k],cond[k],f'{pair} {side} {name} {k}')
                require(sc['top1_id']==cond['top1_id'],'condition top1')
                tv=.5*math.fsum(abs(a-b) for a,b in zip(sc['off'],baseline['off']))
                close(tv,cond['off_conditional_tv'],f'{pair} {side} {name} tv')
                if name in ('repeat','post_clean'):
                    require(tensor(row,raw,'logits')==tensor(init,init_raw,'logits'),f'{name} logits')
                    require(tensor(row,raw,'early')==early and tensor(row,raw,'late')==tensor(init,init_raw,'late'),f'{name} captures')
                    continue
                before=tensor(row,raw,'before_all'); after=tensor(row,raw,'after_all')
                expected_before=tensor(init,init_raw,'late') if name=='final_inside' else early
                require(before==expected_before,f'{pair} {side} {name} before')
                if name in ('full_self','full_swap'):
                    desired=early if name=='full_self' else other_early
                    expected_logits=tensor(init,init_raw,'logits') if name=='full_self' else tensor(orow,oraw,'logits')
                    require(after==tensor(row,raw,'requested')==desired and tensor(row,raw,'logits')==expected_logits, f'{name} full bits')
                else:
                    desired=own if name=='null_inside' else wanted[name]
                    if name!='null_inside': require(tensor(row,raw,'requested')==desired,f'{name} requested bits')
                    require(block(after,ids)==desired,'landing')
                    for j in range(197):
                        if j not in ids: require(after[768*j:768*(j+1)]==before[768*j:768*(j+1)],f'{name} untouched')
                    for k,j in enumerate(ids):
                        require(tensor(row,raw,f'before_{k}')==before[768*j:768*(j+1)] and tensor(row,raw,f'after_{k}')==after[768*j:768*(j+1)],'witness')
                    if name in ('null_inside','self_inside'):
                        require(after==before and tensor(row,raw,'logits')==tensor(init,init_raw,'logits'),f'{name} control')
                    if name=='final_inside':
                        require(tensor(row,raw,'logits')==tensor(init,init_raw,'logits'), 'final control logits')
                    if name.startswith('matched_'):
                        delivered=dist(desired,own)
                        close(delivered,cond['actual_dose'],f'{name} dose',1e-10)
                        require(delivered <= .10*norm and abs(delivered-q)<=info['dose']['tolerance'], 'dose bounds')
                        deliveries[name]=delivered
                        changes[name.removeprefix('matched_')]=(sc['margin']-baseline['margin'])*(1 if side=='cat' else -1)
                delta=(sc['margin']-baseline['margin'])*(1 if side=='cat' else -1)
                close(delta,cond['annotation_oriented_change'],f'{name} direction')
            require(set(changes)=={'opposite','same_species','random'}, 'movements')
            require(all(abs(value-deliveries['matched_opposite']) <= .01*deliveries['matched_opposite'] for value in deliveries.values()), 'matched dose discrepancy')
            passed=changes['opposite']>=.05 and changes['opposite']>=max(abs(changes['same_species']),abs(changes['random']))+.025
            for k,v in changes.items(): close(v,p['descriptive_movements'][side][k],'movement')
            side_results[side]={'image_id':ident,'donor_id':donor,'opposite':changes['opposite'],
                                'same_species':changes['same_species'],'random':changes['random'],
                                'rule_pass':passed,'q_over_receiver_norm':q/norm}
        status='positive' if all(v['rule_pass'] for v in side_results.values()) else 'negative'
        require(p['status']==status,'pair classification')
        report.append({'pair_id':pair,'tokens':ids,'status':status,'sides':side_results})
    positive=sum(p['status']=='positive' for p in report)
    aggregate={'selected':4,'evaluable':4,'positive':positive,'status':'positive' if positive*4>=12 else 'negative'}
    require(t['aggregate']==aggregate,'aggregate')
    require_nearest()
    print(json.dumps({'terminal_sha256':digest(terminal_raw),'screen_result_sha256':digest(screen_raw),
                      'forward_count':len(rows),'distinct_clean_images':len(clean),'aggregate':aggregate,
                      'max_absolute_score_error':maximum_score_error,'pairs':report,
                      'scope':'Closed saved bytes only; no execution, provenance, chronology or population attestation.'},
                     sort_keys=True,indent=2))

if __name__=='__main__': main()
