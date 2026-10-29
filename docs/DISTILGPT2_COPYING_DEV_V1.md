# DistilGPT-2 repeated-sequence development screen

This is a small, public exploratory screen, not a held-out evaluation. It asks
whether the pinned DistilGPT-2 assigns a previously seen continuation more
probability when a five-token sequence is repeated. It does not establish an
induction circuit or a causal explanation. Use CPU float32, eager attention,
evaluation/inference mode, one intra-op thread and no KV cache. Keep all
results, including negative or mixed effects.

Before any forward, publish this document and the runner. The eight fixed
word groups and exact single-token GPT-2 IDs are in the runner. Each token is
the word with a leading space; tokenize every complete prompt with the pinned
tokenizer and require the expected 11 IDs. All 48 IDs must be distinct. For a
group `[a,b,c,d,e,f]`, evaluate these four fixed inputs in the listed order:

1. `[a,b,c,d,e,f,a,b,c,d,e]`
2. `[e,c,a,d,b,f,a,b,c,d,e]`
3. `[b,d,e,c,a,f,a,b,c,d,e]`
4. `[d,a,e,b,c,f,a,b,c,d,e]`

The next-token target is `f`, scored from logits at the final input position;
its earlier occurrence is at position 5. Within each group, the prompts
have the same multiset of input tokens, the same last five tokens and the same
earlier position of `f`. The controls change the earlier local ordering and
the context in which `f` was encoded, so any difference may reflect those
changes as well as repeated-sequence behavior. Distinct groups are convenient
descriptive examples, not independent model samples or a population sample.

Run one additional uncaptured forward on the first public input and retain its
full final-position logits; require exact float32 bits to match the captured
version. Retain all 50,257 final-position float32 logits and all six layers'
12-head 11-by-11 attention matrices for the 32 recorded cases, in input order.
Check shape, finiteness, probability-row normalization and the causal mask.
Record each target's raw logit, binary64 log probability computed
from the saved float32 logits, one-based rank (`1 +` the number of strictly
greater logits), and top-token ID (ties choose the lowest token ID; all tied
maxima have rank 1). Report each repeated-versus-control
log-probability difference, the mean difference against the three controls
for each group, and the eight-group mean and number with positive mean. Also
report repeated and control top-one counts. Do not infer statistical
significance from these eight examples.

As descriptive leads only, for each layer/head compute the average difference
between repeated and control attention from the final position to position 5,
where the earlier `f` occurs. Record all 72 scores, with ties ordered by lower
layer then head. Attention is not an intervention: after seeing these public
development results, any targeted head/module intervention, its comparison
and disjoint evaluation cases require a separately fixed design. A failed or
negative screen is a result, not a reason to silently change cases or controls.

The runner binds the exact source revision and pinned offline model/tokenizer
asset hashes, records thread/runtime and numerical probes, saves an append-only
attempt log and raw tensor streams, and publishes a result only after complete
forwards and matching before/after model-state hashes. A valid result also
requires a zero process exit and no failure record; an interrupted attempt
remains incomplete with any saved prefix; a new attempt uses a new directory.
Source and local timestamps are operational records, not proof of isolation,
pretraining-data independence, or an externally witnessed precommit.
