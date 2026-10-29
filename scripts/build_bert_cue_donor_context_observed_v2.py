"""Build the public saved-input status from the fixed manifest and score projection."""
import argparse,gzip,hashlib,json
from pathlib import Path

PINS={
 'manifest':'917bad800f72a9613e69d8feae3faf90ffbae5cc446bbeb3e2c8df1d03678dc9',
 'subset':'43088c7fb9763bf80df52a4459b1c1e42d37b5de120582b58b37f59eacea2467',
 'result':'3674c152b3c6e24383f191443ca50d2697baaf0c9334c22fa7766f759ace05c4',
 'exit':'d84f0939d14cc50d4baaf435d95b5c2ce0e0b5ccbf9e8fb62a948b3c23aea2af',
}
JOURNAL='cde003fb88e17fa4b9af3215e8e0ccbef23fb2d5df83e1ef859715b4bc5c17de'
SOURCE='b6667ad5bfcadac6e0f45d4ad6530cca1f4584ca'
RESULT_COMMIT='dd0cb22c25f1c778840b51263dee449143dcdde1'
CUES=('never','rarely','always','often')


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser()
 for n in ('manifest','subset','result','exit','output'):p.add_argument(n,type=Path)
 a=p.parse_args(); paths={n:getattr(a,n) for n in PINS}
 if any(sha(path)!=PINS[name] for name,path in paths.items()):raise ValueError('fixed input hash differs')
 manifest=json.loads(a.manifest.read_bytes()); result=json.loads(a.result.read_bytes()); receipt=json.loads(a.exit.read_bytes())
 if (result.get('source_commit')!=SOURCE or result.get('events_sha256')!=JOURNAL or result.get('status')!='complete' or
     receipt.get('source_commit')!=SOURCE or receipt.get('process_started') is not True or
     receipt.get('process_waited') is not True or type(receipt.get('exit_code')) is not int or receipt['exit_code']!=0):
  raise ValueError('closed source/result/exit differs')
 observed={}; count=0
 with gzip.open(a.subset,'rt',encoding='utf-8') as f:
  header=json.loads(next(f))
  if (header.get('manifest_sha256')!=PINS['manifest'] or header.get('result_sha256')!=PINS['result'] or
      header.get('original_journal_sha256')!=JOURNAL or header.get('source_commit')!=SOURCE):
   raise ValueError('projection header differs')
  for line in f:
   event=json.loads(line);count+=1
   if event['kind']=='clean':
    key=(event['pair_id'],event['context'],event['cue'])
    if key in observed:raise ValueError('duplicate clean identity')
    observed[key]=event['input_ids']
 if count!=1096 or len(observed)!=32:raise ValueError('saved event or clean count differs')
 rows=[]
 for pair in manifest['pairs']:
  for index,context in enumerate(pair['contexts']):
   for cue in CUES:
    row=context['rows'][cue];key=(pair['id'],index,cue)
    if observed.pop(key,None)!=row['input_ids']:raise ValueError('clean input ids differ')
    rows.append({'id':context['id']+'-'+cue,'pair_id':pair['id'],'split':pair['split'],
                 'context':index,'context_id':context['id'],'cue':cue,'text':row['text'],
                 'input_ids':row['input_ids'],'attention_mask':row['attention_mask'],
                 'status':'observed-saved-clean-input-ids'})
 if observed or len(rows)!=32:raise ValueError('manifest coverage differs')
 if any(sha(path)!=PINS[name] for name,path in paths.items()):raise ValueError('input changed during read')
 output={'schema':'bert-cue-donor-context-public-exposures-v2','model':manifest['model'],
         'revision':manifest['model_revision'],'reservation_path':'experiments/bert_cue_donor_context_retention_cases_v1.json',
         'reservation_sha256':PINS['manifest'],'observed_source_commit':SOURCE,'result_public_commit':RESULT_COMMIT,
         'result_sha256':PINS['result'],'journal_sha256':JOURNAL,'score_projection_sha256':PINS['subset'],
         'controller_exit_sha256':PINS['exit'],'saved_clean_input_count':32,'status':'public-nonblind-observed-saved',
         'prior_incomplete_attempt':'https://github.com/EastonSell/model-looking-glass/pull/733',
         'scope':'Observed labels refer to clean input IDs in the public operator-derived projection, independently compared with the local full journal. Text and attention masks are copied from the fixed manifest. Saved records do not attest execution, complete exposure history or pretraining exposure.',
         'rows':rows}
 with a.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(output,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':'pass','rows':len(rows),'source_events':count,'output_sha256':sha(a.output)},sort_keys=True))


if __name__=='__main__':main()
