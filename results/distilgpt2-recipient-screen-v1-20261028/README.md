# DistilGPT-2 recipient screen: baseline did not meet the patching gate

On 64 fixed public two-name prompts, the model assigned the intended recipient
a higher next-token logit than the giver in **29 of 64** cases. Only **3 of
32** matched pairs had a positive recipient margin in both directions; the
predeclared threshold was 24. The other gate requirements also failed: the two
templates had 1/16 and 2/16 successful pairs (12 required in each), and 0/8
name pairs met the required three successful pairs (six required). The
planned activation patch was therefore **not run**.

The margin is `logit(recipient) - logit(giver)` for the two space-prefixed,
single-token names. Its mean was −0.129756 across the 64 cases; there were 29
positive and 35 negative margins, with no ties. This is a forced-choice
comparison, not unrestricted next-token accuracy: the intended recipient was
the full-vocabulary top token in 3/64 cases, the giver in 1/64, and another
token in 60/64.

| Name pair | Positive individual margins | Matched pairs positive both ways | Mean margin |
| --- | ---: | ---: | ---: |
| John / Mary | 6/8 | 2/4 | +0.291465 |
| Michael / Sarah | 3/8 | 0/4 | +0.007140 |
| Daniel / Emma | 5/8 | 1/4 | +0.038760 |
| Peter / Lisa | 4/8 | 0/4 | +0.235334 |
| Alice / Claire | 2/8 | 0/4 | −0.447457 |
| Anna / Susan | 2/8 | 0/4 | −0.590071 |
| Paul / Mark | 4/8 | 0/4 | −0.237556 |
| David / Robert | 3/8 | 0/4 | −0.335659 |

The first and second templates had 14/32 and 15/32 positive individual
margins, with means −0.161373 and −0.098138 respectively. The two initial-name
orders had 15/32 and 14/32 positive margins. Every individual prompt, both
answer logits, log probabilities, ranks, and the complete final-position
logit vector are retained. The [raw audit output](raw-audit.json) preserves
per-case results; the [independent replay](independent-audit.txt) verified all
cases and reports aggregate checks.

## Interpretation

This is an exploratory screen on two deterministic public templates and eight
name pairs, not a blind evaluation or a model-wide accuracy estimate. The 64
cases are correlated. The result does not show that DistilGPT-2 lacks all
recipient computation; it establishes that this fixed set did not meet the
criterion for the proposed patch study. No result was selected out after
observing the model. An [IOI circuit study](https://arxiv.org/abs/2211.00593)
examined a different, larger GPT-2 checkpoint. A recent
[DistilGPT-2 preprint](https://arxiv.org/abs/2610.37157) reports weak IOI
behavior on its own prompt distribution, but its exact checkpoint and cases
are not established to match this experiment. Patching metrics and corruption
choices can affect localization conclusions
([Zhang and Nanda](https://arxiv.org/abs/2309.16042)).

## Evidence

The [frozen protocol and cases](https://github.com/EastonSell/model-looking-glass/pull/328)
and source revision `32164cd570dea1f4bc0022c5b7f81fe17f8c2797` were public
before this CPU run. The checkpoint was `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`; its model-weight SHA-256 was
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The direct process returned zero; all 64 forwards and records were observed,
the 12,865,792-byte [raw logits](run/last_logits.f32) are intact, stderr is
empty, no failure record exists, and model state and buffers matched before
and after. CUDA was hidden. The [manifest](run/manifest.json),
[journal](run/attempt.jsonl), [result](run/result.json),
[pre-run record](pre-run.json) and [direct-process receipt](supervisor.json)
are retained. The receipt and journal are process/operator observations, not
independent proof of execution or trusted timestamps.

Two saved-byte replays checked all 64 vectors, scores, ranks, matched-pair
counts, the fixed gate, and all 193 journal events without a model forward.
From a Git checkout containing the frozen source commit, run
`python results/distilgpt2-recipient-screen-v1-20261028/raw-audit.py` and
`python results/distilgpt2-recipient-screen-v1-20261028/independent-audit.py`.
The independent reviewer script was changed only to resolve the package and
repository paths; it verifies all 11 frozen source blobs with `git show`.
These replays verify retained-byte consistency, not an independent model run.

Key SHA-256 values:

* `pre-run.json`: `301fb63824a1697b702edc0b88730a0904c3d36dd26f01c03f2d6c4308301f55`
* `run/manifest.json`: `064bc1a2cf7e0a8201a476582ebde860d9345f67db131f34ec16f9f9cbb077e4`
* `run/attempt.jsonl`: `c43e12ef31dbddb2753fe5d067ff1e95223d62754c34d551572dc00fc477a413`
* `run/last_logits.f32`: `2c5d551f5dfb76db9adc32c2413e3d71d94576e912fff31dea9b94536644e724`
* `run/result.json`: `3848582148a04d027012e862004c783152bc18255d783f028325f7c68be9b344`
* `supervisor.json`: `fbf7e2f93074c502e13ebb2ceac53547985d6952070c98d6d09a457fbe71b9cf`
