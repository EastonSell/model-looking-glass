"""Create publication tables and a result note from the closed saved-byte report."""
import csv, hashlib, json, statistics
from pathlib import Path
root=Path(__file__).resolve().parent
raw=(root/'science-replay.json').read_bytes()
if hashlib.sha256(raw).hexdigest()!='9660ba045f12f86f74f2e4b526730cbab14f7caf2ef836ca674210c67ca5225c': raise ValueError('report pin differs')
d=json.loads(raw)
if len(d['cases'])!=68 or d['journal']['events']!=613: raise ValueError('report dimensions differ')
arms=('u','v1','v2'); margins=('present','past','the_vs_a','in_vs_on')
metrics=('s','delta','d_num','d_lex','k_a','k_b','l','i')
with (root/'primary-cells.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.writer(f,lineterminator='\n'); w.writerow(['split','frame','context']+[a+'_'+m for a in arms for m in metrics])
 for split in ('discovery','evaluation'):
  for c in d['partitions'][split]['cells']:
   s=c['statistics']['present']; w.writerow([split,c['frame'],c['context']]+[s[a][m] for a in arms for m in metrics])
with (root/'control-responses.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.writer(f,lineterminator='\n'); w.writerow(['case_id','split','role','frame','context','determiner']+[kind+'_'+a+'_'+m for kind in ('odd','even') for a in arms for m in margins])
 for x in sorted(d['fragments']+d['demonstratives'],key=lambda x:x['case_id']):
  w.writerow([x['case_id'],x['split'],x['role'],x['frame'],x['context'],x['determiner']]+[x[kind][a][m] for kind in ('odd','even') for a in arms for m in margins])
with (root/'case-margins.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.writer(f,lineterminator='\n'); actions=('clean','noop','u_plus','u_minus','v1_plus','v1_minus','v2_plus','v2_minus','clean_after')
 w.writerow(['case_id','split','role','frame','context','noun','determiner']+[a+'_'+m for a in actions for m in margins])
 for x in sorted(d['cases'],key=lambda x:x['case_id']):
  w.writerow([x['case_id'],x['split'],x['role'],x['frame'],x['context'],x['noun'],x['determiner']]+[x['margins'][a][m] for a in actions for m in margins])
def f(x): return f'{x:.6f}'
summary=[]
for split in ('discovery','evaluation'):
 for name,label in (('sensitivity','Sensitivity S'),('number_over_noun','Number-over-noun Delta')):
  z=d['partitions'][split]['criteria'][name]
  summary.append(f"| {split} | {label} | {f(z['u_mean'])} | {f(z['v1_abs_mean'])} | {f(z['v2_abs_mean'])} | {z['positive_cells']}/6 | {z['decision']} |")
cells=[]
for split in ('discovery','evaluation'):
 for c in d['partitions'][split]['cells']:
  s=c['statistics']['present']; vals=[s[a]['s'] for a in arms]+[s[a]['delta'] for a in arms]
  cells.append(f"| {split} | {c['frame']} | {('none','singular','plural')[c['context']]} | "+' | '.join(f(v) for v in vals)+' |')
controls=[]
for x in sorted(d['fragments']+d['demonstratives'],key=lambda x:x['case_id']):
 controls.append(f"| {x['case_id']} | "+' | '.join(f(x['odd'][a]['present']) for a in arms)+' |')
vals=[z for x in d['cases'] for z in x['deliveries'].values()]
if len(vals)!=408 or not all(x['dose_valid'] for x in vals): raise ValueError('unexpected dose verification')
lo=min(x['realized_norm'] for x in vals); hi=max(x['realized_norm'] for x in vals)
text='''# Invariant-noun agreement intervention: public successor v2

The two preregistered descriptive criteria passed in both discovery and evaluation on the saved DistilGPT-2 outputs. Each split had 24/24 correct clean `is` versus `are` signs. In evaluation, mean sensitivity was 0.205178 for the fixed direction versus mean absolute active-control values 0.068777 and 0.071841; mean number-over-noun distance difference was 0.163363 versus 0.055743 and 0.063142. These are fixed, small, public-case comparisons, not evidence of a uniquely abstract grammatical-number representation.

The evaluation number-over-noun result was positive in only the required four of six cells: the frame 0/plural-attractor cell was **−0.008484** and the frame 1/plural-attractor cell was **exactly zero**. In those cells sensitivity was 0.005294 and 0.019948, respectively, below the corresponding v2 active-control responses 0.052645 and 0.052294. The aggregate result is not uniform across contexts. All 12 subjectless fragments and all eight standalone demonstrative controls also had positive u responses; their syntax and context differ from the target prompts and they are not a null distribution.

## What was tested

The fixed intervention adds or subtracts the previously selected direction `u` at the last-token output of layer 3 attention (`transformer.h.3.attn`), with nominal input-space L2 dose 2. The two previously fixed directions `v1` and `v2` are active equal-input-dose controls; equal L2 does not imply equal functional dose. For each prompt, the response `O` is half the difference between the `is − are` logit margins under `+2a` and `−2a`. For each pair of invariant nouns A/B, `S` averages the signed `this − these` response differences within each noun. `D_num` is the average absolute within-noun `this`/`these` difference; `D_lex` is the average absolute between-noun difference at a fixed determiner; `Delta = D_num − D_lex`. The names describe the fixed comparisons, not an isolated mechanism. A zero or negative Delta does not prove number information is absent.

Discovery uses `salmon` and `trout` in the `At sunrise` and `Before sunset` frames; evaluation uses `squid` and `shrimp` in `At midnight` and `During autumn`. Context 0 is `nearby`, 1 uses a singular attractor and 2 its plural form. Each frame has its own attractor pair, so frame and attractor-lemma effects are not separately identified, and the noun sets differ by split. `this` and `these` change the determiner and local syntax even though the noun surface is held fixed; determiner sensitivity, context effects, lexical effects and output bias remain possible. The cases are public and nonblind, and unknown pretraining exposure is not excluded. The six cells reuse forms and frames and are not six independent samples.

The earlier [v1 attempt](../distilgpt2-agreement-invariant-noun-v1/README.md) stopped incomplete after a journal append failure. Its 34 discovery inputs are treated as exposed; none was repeated in v2. The 34 retained v1 evaluation objects appeared beyond the 301 saved events and 302 owner-reported starts/returns under the sequential source contract. That supports, but does not authenticate, their nonexecution in v1. The operator reports that no incomplete-v1 logits or intervention-effect values were used to select the v2 cases.

The inherited v1 protocol called for fresh natural and injection custody observations. Under the explicit no-access instruction those requirements were **not met**. The [public-only departure](https://github.com/EastonSell/model-looking-glass/blob/ac35e97c250aeefa683d9fd4be7a3761d2fee3f9/docs/AGREEMENT_INVARIANT_NOUN_V2_PUBLIC_ONLY_DEPARTURE.md) was published before v2 forwards; it marks separately protected history and target relation unknown. The public support comparison is conditional on its recorded sampler/tokenizer preparation and is not a private membership query. Unknown or separately protected exposure history is unverified.

## Fixed decisions and all primary cells

| Split | Criterion | Mean u | Mean absolute v1 | Mean absolute v2 | Positive u cells | Decision |
|---|---|---:|---:|---:|---:|---|
'''+'\n'.join(summary)+'''

Both baselines were adequate: 24/24 target signs correct in each split, no ties, and all six cells had four correct. The active-control columns above are means of the individual absolute cell values, not the absolute value of a signed mean. The table below gives signed S and Delta values for every fixed cell; [primary-cells.csv](primary-cells.csv) also retains D_num, D_lex, K_A, K_B, L and I for all three directions. Full unrounded values, all margins and additional breakdowns are in [science-replay.json](science-replay.json).

| Split | Frame | Context | u S | v1 S | v2 S | u Delta | v1 Delta | v2 Delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
'''+'\n'.join(cells)+'''

## Subjectless and demonstrative controls

These are signed `is − are` odd responses, not differences from a matched subject-deletion baseline. Discovery fragment u responses range 0.189205–0.423956 (mean 0.284868); evaluation fragment u responses range 0.160398–0.478683 (mean 0.289329). Discovery demonstratives range 0.136627–0.180433 (mean 0.157559); evaluation demonstratives range 0.120953–0.183069 (mean 0.152291). [control-responses.csv](control-responses.csv) also includes all four margins and even responses.

| Case | u odd | v1 odd | v2 odd |
|---|---:|---:|---:|
'''+'\n'.join(controls)+'''

## Execution and reproducibility

The separate E:-backed CPU owner reported 612 forwards in 613 journal events, a complete candidate, and the observer and direct child each exited 0. The full source-first replay checked the saved last-position vocabulary logits, clean/no-op controls, schedule, landing witnesses, and 408/408 valid delivered vectors; realized L2 ranged from '''+f'{lo:.10f}'+' to '+f'{hi:.10f}'+''' (unrounded in the report). A separately implemented selected-logit replay agreed on all 68 cases, baseline counts, cells and decisions. The [full report](science-replay.json), [redacted replay-process observation](science-replay-observation.json), [selected-logit comparison](independent-selected-replay.json), and [independent file/receipt audit](independent-file-review.json) bind the saved data. The full replay cannot independently verify unsaved sequence-position logits; cooperative process observations do not authenticate execution or descendant drain.

Source revision `ac35e97c250aeefa683d9fd4be7a3761d2fee3f9`, fixed cases SHA-256 `188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b`, events SHA-256 `ffe761e9a02338bf6f9afbcb066197ed83a2488d02cf79986585ea8225e3fd10`, terminal SHA-256 `3184ed42ee8171ffc29c56251912df19c03b04c71b48d35522baf649dd52fdfb`, and [owner exit observation](owner-exit-observation.json) are retained. The source reconstructs v43 (`61a80aaf…`); v44 (`78d783dc…`) was separately supplied and checked as the operator's later public inventory. A [France public-input observation](france-observation.json) and [v44 addendum](france-v44-observation.json) were recorded after observer start and before the first saved tensor was observed; their chronology and no-forward statement are cooperative reports. Adapter's v44 cases were later reported forwarded after this observer ended, so their planned inventory labels describe the historical snapshot.

The observer began at 14:03:20 UTC; the pre-model admission marker was created at 14:05:43 and the model-admission marker at 14:05:57. The child ended at 14:08:14 and the observer exited at 14:08:34. These filesystem/operator timestamps indicate roughly 143 seconds of source, inventory and setup, 14 seconds to model admission, and 137 seconds through model work and child cleanup; they are not isolated benchmarks or an authenticated timing trace.

[archive-manifest.json](archive-manifest.json) binds all 1,450 ordinary files and the 11 ordered [archive parts](parts/), independently streamed against the closed originals. Joined size is 88,346,152 bytes, SHA-256 `1111e2630dc755273ff01a4a898b27b75d79171765c26489f47ee4ad513ebcfa`; manifest SHA-256 `da204b88f5c209c5c1151a51ccd600b88087cf232dbb9e3de6b32f4b0f0f4a8b`. From this directory, a fresh output can be reconstructed with:

```sh
python -I -S -B join_evidence.py --parts-dir parts --manifest archive-manifest.json --output ./attempt-v2.tar.gz
```

The joiner verifies the pinned manifest, each part and final archive. A failed write or final verification can leave a partial output; remove or rename that output before a new join. [case-margins.csv](case-margins.csv) provides all 68 cases and nine saved arms, with four margins per arm. The operator's original attempt and one-shot claim remain preserved separately.
'''
with (root/'README.md').open('x',encoding='utf8',newline='') as f: f.write(text)
for name in ('README.md','primary-cells.csv','control-responses.csv','case-margins.csv'):
 p=root/name; print(name,hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size)
