# Public repeated-sequence development screen

On eight fixed public token families, DistilGPT-2 assigned the next token a
higher log probability after a repeated five-token suffix than after each of
three matched first-block shuffles: all 24 contrasts were positive. The
eight-family mean contrast was **5.0232 natural-log units**. The next token
was top-ranked on 7 of 8 repeated prompts and 0 of 24 controls. This is a
small descriptive development result, not a population estimate or a blind
test; the public inputs have now been exposed to the model.

Each family uses six distinct single-token words `a,b,c,d,e,f`. The repeated
prompt is `a b c d e f a b c d e` and the target is `f`. Each control keeps
the token multiset, the last five input tokens, the target and its earlier
position, but permutes the first five tokens. Controls also alter local
transitions and the context in which `f` is first encoded. Thus this contrast
does not isolate a unique copying mechanism.

| Family | Mean repeated-minus-control target log probability |
| --- | ---: |
| 1 | 5.5758 |
| 2 | 5.0837 |
| 3 | 5.7005 |
| 4 | 4.2117 |
| 5 | 3.3056 |
| 6 | 6.2852 |
| 7 | 5.7781 |
| 8 | 4.2451 |

The prespecified ranking of last-position attention to the earlier target
found layer 3, head 10 (zero-based) highest, with mean repeated-minus-control
attention **0.8940**. The complete ranking, including negative head contrasts,
is retained. Attention is an observation and only motivates a separate
intervention; it does not establish that this head causes the behavior.

The fixed [protocol](https://github.com/EastonSell/model-looking-glass/blob/55d11a53eb34a13eb6b14e19a5bd0c47172edda5/docs/DISTILGPT2_COPYING_DEV_V1.md) and
[source](https://github.com/EastonSell/model-looking-glass/blob/55d11a53eb34a13eb6b14e19a5bd0c47172edda5/scripts/copying_dev_v1.py) were published at commit
`55d11a53eb34a13eb6b14e19a5bd0c47172edda5` before execution. The 32 planned
rows were checked against known exposures and publicly reserved in the
[v6 inventory](https://github.com/EastonSell/model-looking-glass/blob/4598f5655f622870db7f94c9f966e1adcfdb21a5/docs/PRETRAINED_EXPOSURE_INVENTORY_V6.md) before the run; neither
the local records nor unkeyed Git hashes prove a trusted external timestamp or
exclude unknown prior/pretraining exposures. The source PR merged after the
run. The pinned model was `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, evaluated offline on CPU in
float32 with one intra-op thread. The cooperative Python CUDA/network guard
is not an attestation against native-code bypass.

[Run manifest](run/manifest.json) contains every exact text, token ID,
permutation, target and source hash. [Attempt log](run/attempt.jsonl) has 33
starts and 33 completions: one uncaptured control followed by 32 attention
captures. The uncaptured and captured first-case final-logit bytes were
identical. The full [32 final-logit vectors](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilgpt2-copying-dev-v1-20261028/run/last_logits.f32),
[32 attention arrays](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilgpt2-copying-dev-v1-20261028/run/attentions.f32), and
[uncaptured vector](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilgpt2-copying-dev-v1-20261028/run/uncaptured_logits.f32) are little-endian float32 in
manifest order. Their shapes per case are `[50257]` and `[6,12,11,11]`.
[Result JSON](run/result.json) retains every case score, pairwise difference,
attention summary and all 72 head scores. [Pre-run](pre-run.json) and
[process observation](supervisor.json) record the input reservation and
successful exit; the latter is a local observation, not independent process
supervision or a security attestation.

Two reviewers independently reconciled the saved bytes without a model
forward, including all token rows, 33 event pairs, all scores, attention masks,
and ranking. One retokenized the public prompts offline; the other rehashed
all seven current pinned cache assets. The maximum independently summed
attention-row normalization error was about `3.0e-7`; future-token attention
was exactly zero. The review checks retained consistency, not faithful
execution or independent replication. The portable [audit](audit.py) can be
run from this checkout with `python results/distilgpt2-copying-dev-v1-20261028/audit.py`;
pass `--cache-dir <hub-cache>` to additionally check the current pinned cache.
