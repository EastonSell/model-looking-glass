"""Independent closed-record taste contrast replay; no model execution."""
import argparse
import base64
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--cases',type=Path,required=True)
parser.add_argument('--tokenizer-json',type=Path)
args=parser.parse_args()
run=args.run
case_raw=args.cases.read_bytes()
result_raw=(run/'result.json').read_bytes()
assert hashlib.sha256(case_raw).hexdigest()=='e59f5e1da0b3ce8c99ea9c231b4998e1c1e685d09c20bff4fdb734dc4910256f'
assert hashlib.sha256(result_raw).hexdigest()=='b7667ad827ff7e9a1380acfcfcad3dd14150be49751a129348f92b5f829276a3'
manifest=json.loads(case_raw); result=json.loads(result_raw)
assert result['status']=='complete' and result['completed_forwards']==336
rows={x['id']:x for x in manifest['rows']}; stems=manifest['selected_stems']
assert len(rows)==24 and len(stems)==6
token_labels=None
if args.tokenizer_json is not None:
    tokenizer_raw=args.tokenizer_json.read_bytes()
    assert hashlib.sha256(tokenizer_raw).hexdigest()=='ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'
    vocabulary=json.loads(tokenizer_raw)['model']['vocab']
    assert len(vocabulary)==30522 and set(vocabulary.values())==set(range(30522))
    token_labels={index:token for token,index in vocabulary.items()}
def f32(bits):
    assert type(bits) is int and 0<=bits<=0xffffffff
    sign=-1. if bits>>31 else 1.; exponent=(bits>>23)&255; fraction=bits&0x7fffff
    assert exponent!=255
    return sign*math.ldexp(float(fraction if exponent==0 else (1<<23)|fraction),-149 if exponent==0 else exponent-150)
def margin(summary):
    sweet,sour=f32(summary['sweet_bits']),f32(summary['sour_bits'])
    assert struct.pack('<d',sweet)==struct.pack('<d',summary['sweet_logit'])
    assert struct.pack('<d',sour)==struct.pack('<d',summary['sour_logit'])
    value=math.fsum((sweet,-sour))
    assert value==summary['sweet_minus_sour']
    return value

events={}; baseline={}; totals=0
for stem in stems:
    name=stem+'.jsonl'; raw=(run/name).read_bytes(); info=result['journals'][name]
    assert {'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==info
    journal=[json.loads(line) for line in raw.splitlines()]
    complete=[e for e in journal if e['event']=='forward_complete']
    assert len(complete)==56
    for e in complete:
        assert e['model_entry']=={'input_ids':rows[e['row_id']]['input_ids'],'attention_mask':rows[e['row_id']]['attention_mask']}
        key=(e['row_id'],e['mode']); assert key not in events; events[key]=e
        margin(e['summary']); totals+=1
        if token_labels is not None:
            assert e['summary']['top_five_tokens']==[token_labels[i] for i in e['summary']['top_five_ids']]
        if e['mode']=='capture_ingredient':
            raw_logits=base64.b64decode(e['baseline_mask_logits_float32_le_base64'],validate=True)
            assert len(raw_logits)==30522*4 and hashlib.sha256(raw_logits).hexdigest()==e['summary']['mask_logits_sha256']
            words=[w[0] for w in struct.iter_unpack('<I',raw_logits)]
            values=[f32(w) for w in words]
            assert words[4086]==e['summary']['sweet_bits'] and words[14768]==e['summary']['sour_bits']
            assert heapq.nsmallest(5,range(len(values)),key=lambda i:(-values[i],i))==e['summary']['top_five_ids']
            baseline[e['row_id']]=e['summary']
assert totals==336 and len(baseline)==24 and result['baselines']==baseline
eligible={}
for stem in stems:
  for template in ('after','with'):
    sugar,lemon=(baseline[f'{stem}-{template}-{x}'] for x in ('sugar','lemon'))
    eligible[f'{stem}-{template}']=(margin(sugar)>0 and margin(lemon)<0 and
       bool({4086,14768}&set(sugar['top_five_ids'])) and bool({4086,14768}&set(lemon['top_five_ids'])))
assert eligible==result['eligibility_descriptive']
effects={}
for effect in result['effects']:
    key=(effect['stem'],effect['template'],effect['direction'],effect['mode']); assert key not in effects
    event=events[(effect['recipient'],effect['mode'])]
    assert effect['summary']==event['summary'] and effect['donor']==event['donor']
    assert effect['baseline_margin']==margin(baseline[effect['recipient']])
    assert effect['patched_margin']==margin(effect['summary'])
    assert effect['effect']==math.fsum((effect['patched_margin'],-effect['baseline_margin']))
    effects[key]=effect['effect']
assert len(effects)==240
comparisons=[]; summaries=[]; position_effects=[]
for stem in stems:
  signed={}; specificity={}
  for template in ('after','with'):
    for direction,sign in (('sugar_to_lemon',1),('lemon_to_sugar',-1)):
      def get(mode):return effects[stem,template,direction,mode]
      ingredient,control=sign*get('ingredient_toward'),sign*get('the_toward')
      contrast=math.fsum((ingredient,-control))
      odd={s:math.fsum((get(s+'_toward'),-get(s+'_away')))/2 for s in ('ingredient','the')}
      even={s:math.fsum((get(s+'_toward'),get(s+'_away')))/2 for s in ('ingredient','the')}
      odd['orth']=math.fsum((get('orth_plus'),-get('orth_minus')))/2
      even['orth']=math.fsum((get('orth_plus'),get('orth_minus')))/2
      specific=math.fsum((sign*odd['ingredient'],-abs(odd['orth'])))
      signed[template,direction]=contrast; specificity[template,direction]=specific
      comparisons.append({'stem':stem,'template':template,'direction':direction,
          'eligible_descriptive':eligible[f'{stem}-{template}'],'signed_matched_position_contrast':contrast,
          'odd':odd,'even':even,'donor_odd_minus_abs_orth':specific})
      position_effects.append({'stem':stem,'template':template,'direction':direction,
                               'signed_ingredient':ingredient,'signed_the':control,'signed_difference':contrast})
  directions=('sugar_to_lemon','lemon_to_sugar')
  after=math.fsum(signed['after',d] for d in directions)/2
  with_=math.fsum(signed['with',d] for d in directions)/2
  diffs={t:math.fsum((signed[t,directions[0]],-signed[t,directions[1]])) for t in ('after','with')}
  sa=math.fsum(specificity['after',d] for d in directions)/2
  sw=math.fsum(specificity['with',d] for d in directions)/2
  summaries.append({'stem':stem,'after':after,'with':with_,'template_difference':math.fsum((after,-with_)),
    'direction_main':math.fsum((diffs['after'],diffs['with']))/2,
    'template_direction_interaction':math.fsum((diffs['after'],-diffs['with'])),
    'direction_specificity_after':sa,'direction_specificity_with':sw,
    'direction_specificity_template_difference':math.fsum((sa,-sw))})
means={k:math.fsum(s[k] for s in summaries)/len(summaries) for k in summaries[0] if k!='stem'}
assert comparisons==result['comparisons'] and summaries==result['stem_summaries'] and means==result['means']
report={'status':'pass','scope':'closed journal/summary arithmetic; no model replay or execution attestation',
 'result_sha256':hashlib.sha256(result_raw).hexdigest(),'forwards':totals,'contrasts':len(comparisons),
 'token_string_mapping_verified':token_labels is not None,
 'eligibility':eligible,'summaries':summaries,'means':means,'position_effects':position_effects,
 'counts':{'positive_contrasts':sum(x['signed_difference']>0 for x in position_effects),
           'negative_contrasts':sum(x['signed_difference']<0 for x in position_effects),
           'positive_template_differences':sum(x['template_difference']>0 for x in summaries),
           'negative_template_differences':sum(x['template_difference']<0 for x in summaries)},
 'limitations':['Patched full logits and untouched whole tensors are owner-observed, not independently retained.',
                'The receipt and external source chronology are separately reviewed.']+
               ([] if token_labels is not None else ['Top-five token strings were not mapped to the pinned vocabulary.'])}
sys.stdout.buffer.write((json.dumps(report,indent=2,allow_nan=False)+'\n').encode('utf-8'))
