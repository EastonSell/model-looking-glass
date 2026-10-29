"""Independent selected-logit replay of a closed agreement-subject journal.

This verifies fixed ordering, selected float bits, clean/no-op/after equality,
K/G and predetermined descriptive criteria. It does not attest execution,
source identity, full-vocabulary finiteness, or patch/cleanup correctness.
The separate full saved-byte reader and an independently observed process
receipt remain required. The fixture mode is explicitly synthetic only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

CASE_SHA = '5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e'
REF_SHA = 'bdb31d5b74b6cff17ed2a683415beab22e36b101220257d774909be6f9f804bd'
DIRECTION_SHAS = {'u':'d1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
                  'v1':'4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
                  'v2':'4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}
PAIRS = (('R1','I1'),('R1','I2'),('R2','I1'),('R2','I2'))
CONDITIONS = ('none','mass','attractor_singular','attractor_plural')
OFFSETS = ((0,0,2,2),(2,2,4,4),(2,2,4,4),(4,4,4,4))
TARGETS = (318,389,373,547,262,257,287,319)
LABELS = ('present','past','the_vs_a','in_vs_on')

def digest(data): return hashlib.sha256(data).hexdigest()

def bounded(path, limit):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError('expected an absolute ordinary file')
    if getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('reparse file')
    with path.open('rb') as f: raw = f.read(limit+1)
    if len(raw) > limit: raise ValueError('file exceeds bound')
    return raw

def decode32(word):
    exponent, fraction = (word >> 23) & 255, word & 0x7fffff
    if exponent == 255: raise ValueError('nonfinite selected float')
    significand = fraction if exponent == 0 else (1 << 23) | fraction
    value = math.ldexp(significand, -149 if exponent == 0 else exponent-150)
    return -value if word >> 31 else value

def decode_vector(raw):
    return [decode32(word) for (word,) in struct.iter_unpack('<I', raw)]

def expected_cases(cases):
    if type(cases) is not list or len(cases) != 80: raise ValueError('case count differs')
    seen = set()
    for frame in range(4):
        for condition in range(4):
            block = cases[frame*20+condition*5:frame*20+condition*5+5]
            order = []
            for pair in range(2):
                first = 1 if (frame+frame//2+pair)%2 else 0
                order += [(block[2*pair+first],pair),(block[2*pair+1-first],pair)]
            order.append((block[4],None))
            for row,pair in order:
                if (row['case_id'] in seen or type(row['frame_index']) is not int or row['frame_index'] != frame
                    or type(row['condition_index']) is not int or row['condition_index'] != condition
                    or row['condition'] != CONDITIONS[condition] or row['pair_index'] != pair
                    or (pair is not None and type(row['pair_index']) is not int)):
                    raise ValueError('case identities differ')
                seen.add(row['case_id'])
                if pair is not None and row['subject_key'] != PAIRS[frame][pair]: raise ValueError('pair differs')
                plus = (frame+condition+(0 if pair is None else pair))%2 == 0
                signs = ('plus','minus') if plus else ('minus','plus')
                arms = tuple(name+'_'+sign for name in ('u','v1','v2') for sign in signs)
                offset = OFFSETS[frame][condition] if pair is None else 2*((4*frame+condition)%3)
                arms = arms[offset:]+arms[:offset]
                yield row,('clean','noop_l3_final',*arms,'clean_after')

def tensor(attempt, ref, expected_size):
    if (type(ref) is not dict or ref.get('dtype') != 'float32-le'
        or ref.get('bytes') != expected_size or type(ref.get('sha256')) is not str):
        raise ValueError('tensor reference differs')
    sha = ref['sha256']
    if len(sha) != 64 or any(x not in '0123456789abcdef' for x in sha): raise ValueError('tensor hash differs')
    raw = bounded(attempt/'tensors'/(sha+'.f32le'), expected_size)
    if len(raw) != expected_size or digest(raw) != sha: raise ValueError('tensor bytes differ')
    return raw

def score(attempt,cases,directions,reference,width,vocabulary,expected_events):
    events_raw = bounded(attempt/'events.jsonl',64*1024**2)
    if digest(events_raw) != expected_events: raise ValueError('event hash differs')
    lines = events_raw.splitlines()
    if len(lines) != 721: raise ValueError('event count differs')
    records = [json.loads(line) for line in lines]
    if records[0]['event']['kind'] != 'run_contract': raise ValueError('run contract missing')
    case_results = {}
    canonical_index = {row['case_id']: i for i,row in enumerate(cases)}
    for case_number,(case,actions) in enumerate(expected_cases(cases)):
        rows = records[1+9*case_number:1+9*(case_number+1)]
        margins = {}
        clean = None
        capture = None
        for j,(record,action) in enumerate(zip(rows,actions)):
            event = record['event']
            if (type(record.get('index')) is not int or record['index'] != 1+9*case_number+j
                or type(event.get('index')) is not int or event['index'] != 9*case_number+j
                or event.get('kind') != 'forward'
                or event.get('case_id') != case['case_id'] or event.get('action') != action
                or type(event.get('case_index')) is not int or event['case_index'] != canonical_index[case['case_id']]
                or any(type(event.get(key)) is not type(case[key]) or event.get(key) != case[key]
                       for key in ('frame_index','condition_index','condition','role','pair_index',
                                   'subject_key','subject_number','morphology'))
                or event.get('slot') != (j-2 if 2 <= j < 8 else None)
                or (2 <= j < 8 and type(event.get('slot')) is not int)
                or type(event.get('input_ids')) is not list
                or any(type(token) is not int for token in event['input_ids'])
                or event['input_ids'] != case['token_ids']
                or type(event.get('attention_mask')) is not list
                or any(type(token) is not int for token in event['attention_mask'])
                or event['attention_mask'] != [1]*len(case['token_ids'])
                or event.get('all_sequence_logits_finite') is not True):
                raise ValueError('action or input differs')
            ref = record['tensors']['last_logits']
            if ref['shape'] != [vocabulary]: raise ValueError('logit shape differs')
            raw = tensor(attempt,ref,4*vocabulary)
            selected = [decode32(struct.unpack_from('<I',raw,4*i)[0]) for i in TARGETS]
            margins[action] = {name:selected[2*i]-selected[2*i+1] for i,name in enumerate(LABELS)}
            if action == 'clean': clean = raw
            if action in ('noop_l3_final','clean_after') and raw != clean:
                raise ValueError('clean/no-op/after logits differ')
            if action == 'noop_l3_final':
                ref = record['tensors']['before']
                if ref['shape'] != [width]: raise ValueError('capture shape differs')
                capture = decode_vector(tensor(attempt,ref,4*width))
        odd,even = {},{}
        for name in ('u','v1','v2'):
            odd[name] = {label:(margins[name+'_plus'][label]-margins[name+'_minus'][label])/2.0 for label in LABELS}
            even[name] = {label:(margins[name+'_plus'][label]+margins[name+'_minus'][label])/2.0-margins['clean'][label] for label in LABELS}
        case_results[case['case_id']] = {'case_id':case['case_id'],'role':case['role'],
            'frame':case['frame_index'],'condition':case['condition'],'subject_key':case['subject_key'],
            'number':case['subject_number'],'margins':margins,'odd':odd,'even':even,'capture':capture}
    pairs = []
    for frame in range(4):
        for pair in range(2):
            for condition in range(4):
                position = frame*20+condition*5+pair*2
                s,p = (case_results[cases[position+i]['case_id']] for i in (0,1))
                k = {name:s['odd'][name]['present']-p['odd'][name]['present'] for name in directions}
                g = {name:math.fsum(d*(a-b) for d,a,b in zip(vector,s['capture'],p['capture']))
                     for name,vector in directions.items()}
                pairs.append({'frame':frame,'condition':CONDITIONS[condition],'subject_key':PAIRS[frame][pair],
                    'singular':s['case_id'],'plural':p['case_id'],'k':k,'g':g,
                    'clean_correct':s['margins']['clean']['present']>0 and p['margins']['clean']['present']<0})
    def group(key,value): return [row for row in pairs if row[key]==value]
    baseline_groups = {key:{str(value):sum(row['clean_correct'] for row in group(key,value)) for value in values}
                       for key,values in (('frame',range(4)),('condition',CONDITIONS),('subject_key',('R1','R2','I1','I2')))}
    baseline_count = sum(row['clean_correct'] for row in pairs)
    adequate = baseline_count>=24 and all(count>=5 for groupcounts in baseline_groups.values() for count in groupcounts.values())
    def mean(values):
        items=list(values)
        return math.fsum(items)/len(items)
    def criterion(key,floor,frame_floor):
        main=mean(row[key]['u'] for row in pairs)
        controls={name:mean(abs(row[key][name]) for row in pairs) for name in ('v1','v2')}
        frames=[]
        for frame in range(4):
            rows=group('frame',frame)
            value=mean(row[key]['u'] for row in rows)
            control={name:mean(abs(row[key][name]) for row in rows) for name in ('v1','v2')}
            frames.append({'mean':value,'controls':control,
                           'passes':value>=frame_floor and all(value>x for x in control.values())})
        return {'mean':main,'controls':controls,'frames':frames,
                'passes':main>=floor and all(main>x for x in controls.values()) and sum(x['passes'] for x in frames)>=3}
    consts = [abs(row['k']['u']) for row in pairs]
    constant = {'mean_abs_k':mean(consts),'within_0_10':sum(x<=.10 for x in consts)}
    constant['passes'] = constant['mean_abs_k']<=.10 and constant['within_0_10']>=24
    reference_odd=2.0*(reference[318]-reference[389])
    errors=[abs(case_results[row['case_id']]['odd']['u']['present']-reference_odd) for row in cases]
    specific={'reference_odd':reference_odd,'mean_abs_error':mean(errors),'within_0_20':sum(x<=.20 for x in errors)}
    specific['passes']=specific['mean_abs_error']<=.20 and specific['within_0_20']>=60
    return {'scope':'selected-logit and clean-capture replay only; no execution, full-vocabulary, landing or cleanup attestation',
        'event_sha256':digest(events_raw),'forwards':720,
        'cases':[case_results[row['case_id']] for row in cases],'pairs':pairs,
        'baseline':{'correct_pairs':baseline_count,'groups':baseline_groups,'adequate':adequate},
        'constant_additive':constant,'specific_reference':specific,
        'sensitivity':{**criterion('k',.10,.05),'interpretation':'baseline_adequate' if adequate else 'baseline_limited'},
        'geometry':criterion('g',.5,.5)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--events-sha256',required=True)
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--directions',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--synthetic-fixture',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.synthetic_fixture:
        fixture=json.loads(bounded(args.synthetic_fixture,1024*1024))
        cases,directions,reference,width,vocabulary=(fixture['cases'],fixture['directions'],fixture['reference'],
                                                     fixture['width'],fixture['vocabulary'])
        scope='synthetic_only'
    else:
        if args.cases is None or args.directions is None or args.reference is None: raise ValueError('fixed public inputs required')
        raw=bounded(args.cases,2*1024**2)
        if digest(raw)!=CASE_SHA: raise ValueError('case digest differs')
        cases=json.loads(raw)['cases']
        directions={}
        for name,sha in DIRECTION_SHAS.items():
            raw=bounded(args.directions/(name+'.f64le'),768*8)
            if len(raw)!=768*8 or digest(raw)!=sha: raise ValueError('direction digest differs')
            directions[name]=struct.unpack('<768d',raw)
        raw=bounded(args.reference,50257*8)
        if len(raw)!=50257*8 or digest(raw)!=REF_SHA: raise ValueError('reference digest differs')
        reference=struct.unpack('<50257d',raw)
        width,vocabulary,scope=768,50257,'public_saved_bytes'
    report=score(args.attempt,cases,directions,reference,width,vocabulary,args.events_sha256)
    report['input_scope']=scope
    raw=(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
    if not args.output.is_absolute() or not args.output.parent.is_dir() or args.output.is_relative_to(args.attempt):
        raise ValueError('report path must be outside attempt')
    with args.output.open('xb') as stream:
        if stream.write(raw)!=len(raw): raise OSError('short report write')
    print(json.dumps({'scope':scope,'sha256':digest(raw),'bytes':len(raw)}))

if __name__ == '__main__': main()
