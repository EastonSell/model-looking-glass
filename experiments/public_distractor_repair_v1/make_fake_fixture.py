"""Generate only a small random-model fixture; no pretrained files are read."""
import json
from pathlib import Path
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from distractor_executor import execute_phase
root=Path(__file__).resolve().parent/'fake-fixture-v3'
root.mkdir(exist_ok=False)
torch.set_num_threads(1)
torch.set_flush_denormal(False)
torch.manual_seed(73)
config=GPT2Config(vocab_size=400,n_positions=16,n_embd=8,n_layer=5,n_head=2,use_cache=False,
                  attn_pdrop=0,resid_pdrop=0,embd_pdrop=0)
config._attn_implementation='eager'
model=GPT2LMHeadModel(config).eval()
cases=[{'case_id':f'distractor-repair-{i:02d}','split':('discovery' if i<=8 else 'held-out'),
        'token_ids':[1,i+3,11,21,31],'candidate_token_ids':[318,389]} for i in range(1,17)]
(root/'cases.json').write_text(json.dumps({'cases':cases},indent=2)+'\n',encoding='utf-8')
for phase,subset in (('discovery',cases[:8]),('held-out',cases[8:])):
    summary=execute_phase(model,subset,root/phase,phase=phase,vocab_size=400,minimum_free_bytes=0)
    print(json.dumps(summary,sort_keys=True),flush=True)
