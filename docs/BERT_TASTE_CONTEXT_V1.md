# Public DistilBERT taste-context exploration v1

## Question and inputs

For four fixed public sentence frames, does transferring the contextual
representation at the ingredient token from a sugar sentence to its matched
lemon sentence, or in the reverse direction, change the masked-token model's
`sweet` minus `sour` logit margin? This is an exploratory intervention on a
single checkpoint and four convenience frames. It is not a held-out study or
evidence of a general taste concept.

The eight exact texts, token IDs, all-one attention masks and coordinates are
reserved in `experiments/bert_taste_public_cases_v1.json`. Each pair differs
only at WordPiece position 7 (`sugar` 5699 or `lemon` 14380); `[MASK]` is
position 4 (103) and the unchanged word `the` at position 1 (1996) is a
location comparison. `sweet` is 4086 and `sour` is 14768. The runtime checks
the full tokenizer output against every row and confirms each target word is a
single token. It compares the complete IDs and masks, including complete-row
prefixes, against the cited public BERT v5 exposure inventory and matched-case
manifest. This is a scoped public-history check, not a certification of other
local history or pretraining exposure.

The pinned offline model is `distilbert/distilbert-base-uncased`, revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, with safetensors SHA-256
`5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`.
The existing loader verifies all five pinned model/tokenizer files before
loading. Execution uses CPU float32, eager attention, evaluation mode, one
intra-op and one inter-op thread, deterministic algorithms, and seed 20261029.

## Intervention and controls

At the output of each encoder block, capture the full output and the selected
token, replace one selected vector with a matching-layer donor vector from the
opposite ingredient sentence, then capture the selected vector and full output
again. Run both directions at every layer. The directional effect is the
patched recipient's `sweet - sour` margin minus its own clean margin; report
every case and signed effect, without division by the baseline gap or a
success threshold. The clean margin is the log odds between those two words,
not the probability of a correct completion. Full-vocabulary softmax
probabilities and top-five completions make competing outputs visible.

At each layer also perform an ingredient self-patch and a cross-context patch
at the unchanged `the` position. Self-patches and fresh final clean forwards
must reproduce all mask-logit bits. The unchanged-word patch is a comparison,
not a no-effect control: bidirectional context may already carry ingredient
information there. At the final block, an action at either other position must
leave the mask logits bitwise unchanged because the remaining MLM head is
position-wise. The selected post-patch vector must match the donor bits; a
whole-output in-process comparison also checks that every other coordinate is
unchanged. Per-case/layer results are never filtered by sign or magnitude.

The fixed schedule makes 248 forwards: four pairs times six layers, with four
clean captures (two positions for each row) and six interventions (three per
row) per layer, plus two final clean forwards per pair. Model state is hashed before and checked after the schedule,
and hooks must be absent after each observation. These checks assume exclusive,
trusted process ownership; they do not prove absence of transient interference.

## Evidence and reproduction

Run from a committed source checkout using `scripts/bert_taste_parent_v1.py`
with `--cache-dir <offline-hub> --output <new-directory>`. The output and its
separate observer receipt must be new, outside the checkout, with an existing
parent directory. A `--preflight-only` invocation uses a separate new output
name and runs a harmless no-model child. The parent checks selected committed
source bytes before startup, creates a fresh empty bytecode prefix, sets the
CPU/offline environment and launches the child with `-I -B -X pycache_prefix`.
It waits for the direct child and records its exit, source recheck, prefix state
and saved result/journal digests. It is not a whole-process-tree supervisor; a
trusted exclusive environment and unchanged dependencies remain preconditions.
On Windows a virtual-environment executable can be a redirector, so the direct
child can be a launcher while the model interpreter is its descendant. A
timeout, interruption, or otherwise uncertain cleanup requires separate
confirmation that all relevant processes have exited before further work; the
receipt does not itself establish whole-tree cleanup, even on success.
The child also checks selected source bytes after its imports and at the end.
Neither set of checks attests all dependencies or proves execution or
publication chronology. No download or GPU is used.

The four case journals retain each forward start/completion, raw selected
pre/post/source vectors, both raw target-logit bit patterns, and the full
mask-logit bits for the eight clean baselines. A reader can recompute baseline
probabilities/top-five and all signed two-token margins, and compare selected
landing bits. Full pre/post block outputs are retained only as hashes: the
untouched-coordinate equality is an owner-observed check, not independently
reconstructible from the saved data. Patched full-vocabulary probabilities and
top-five outputs are likewise owner-observed. The child binds each journal's
length and hash in its result. Journal records are flushed as they are written;
an interrupted or failed attempt must be retained as incomplete. A readable
child result alone is not completion: any incomplete marker, missing or
invalid observer receipt, uncertain process cleanup, nonzero exit, or journal
mismatch leaves the attempt incomplete or uncertain. No model weights or
private inputs are saved.
