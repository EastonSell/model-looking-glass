# Natural agreement successor v1: prospective scientific protocol

This is a separate evaluation of an already public prediction. Publishing this
protocol, pool and sampler fixes the scientific choices before any successor
private pool comparison or split draw. It does **not** itself authorize either
action: the one-use query and phase custody, execution owner, final runtime
binding and exact production-path rehearsal described below remain required.

## Existing evidence and fixed question

The first natural held-out attempt consumed its claim and sealed one split, but
its owner exited 1 with a sanitized `RuntimeError`; no owner-execution journal
or tensor evidence was retained. The independent no-model startup probes make a
missing site-packages path a strong inference, not an observed original error
message. The [published failure record](../results/distilgpt2-natural-agreement-v1-20261028/heldout-attempt-status-v1.md)
describes that result and its limits. The old claim, split, commitment and
gate markers remain intact and are never reset or replayed. Its forward
announcement is an unchecked possible-exposure ordering point.

Use the existing natural prediction aggregate SHA-256
`35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb`
and prediction-file SHA-256
`eaac12d7229b2ae5615832bc9b58c7e5b2404c5866b5079295e72b0a54a4c545`.
Independently verify their discovery evidence and selection; neither is revised
using successor cases. Retain all eight frozen candidate sites, their
predictions and selected/earliest/random/magnitude labels, including
coincident labels and abstention.

Use four families, each with two distinct noun pairs and singular/plural forms:
16 prompts total. Two families have singular-object contexts and two have
plural-object contexts. Run the original interventions and controls, with
`is`/`are` as the primary contrast and `was`/`were` separately. Retain incorrect
or tied clean preferences. At each site report the four family number-main
effects and their MAE/RMSE against the frozen prediction and zero, together
with per-target clean/patched margins, four cell means, sign flips, control
and landing witnesses. Keep the four-family denominator. These results are
descriptive: a positive number-main effect can arise from erasure, and no
significance or broad population claim follows from four families. The new
lexical distribution is not assumed equivalent to discovery, nor can this
protocol rule out pretraining exposure.

## Fixed public pool and selection

The public pool is `experiments/natural_agreement_successor_pool_v1.json`,
SHA-256 `b3323d296a9f56a4232b4a403f7c8935896634b3d2b6e9fe7eee4aee56ba1126`.
It contains 16 noun pairs and eight balanced contexts; the noun pairs do not
reuse the published original preflight/discovery nouns or the old natural
sampling-frame nouns. That statement does not claim novelty across all model
pretraining or other public studies. The ordered public candidate-input file is
`experiments/natural_agreement_successor_inputs_v1.json`, SHA-256
`6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4`;
it contains every one of the 256 possible prompts and no selected split. The
pinned tokenizer.json SHA-256 is
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.

The allocation helper is `looking_glass/natural_agreement_successor.py`,
SHA-256 `7f89164bbca32febc50ee7bf78fba40ba3b8f8259262f75d61a6f2b6157cf55b`.
For each item, use its unique ASCII ID. Given a single 32-byte OS nonce, sort
by `(SHA256(domain || nonce || uint16be(len(id)) || id), id)`. The four
distinct ASCII domains, each followed by one NUL byte, are:

```
MLG-natural-successor-v1/noun
MLG-natural-successor-v1/singular-context
MLG-natural-successor-v1/plural-context
MLG-natural-successor-v1/family
```

Select the first eight noun pairs and the first two contexts of each number;
re-sort the four contexts by the family domain and assign the ordered noun
pairs two at a time. Within each family order first-lemma singular/plural,
then second-lemma singular/plural. ASCII ID breaks a digest tie. The helper
uses no model and never generates a nonce. The published test uses a fabricated
pool and nonce as an independent reproducibility vector.

Every selectable quartet must have four distinct single subject-token IDs,
identical remaining tokens and length, and `0 < subject_index < len(input_ids)
- 1`. All possible selected prompts must be unique and at most 32 tokens.
For each candidate prompt, the pinned tokenizer must map the prompt followed
by each of ` is`, ` are`, ` was`, ` were` to exactly the prompt IDs followed
by the respective pinned single ID 318, 389, 373 or 547. No model forward is
permitted to qualify or select a candidate.

## One private pool query and one split

Before any private query, publish the exact protocol, pool, candidate-input
bytes, comparator source/command and one-use query identity; independently
verify the frozen prediction and the latest reviewed public exposure snapshot.
Review the cumulative disclosure from earlier public-vs-old-split comparisons
before issuing this query; a retained receipt or claimed clear result does not
by itself establish that one more aggregate disclosure is acceptable.
Only one trusted query compares the 256 fixed public inputs against the old
16-row committed natural split in both complete-row-prefix directions. Bind
the old salted commitment, exact input and comparator hashes, counts and
method; output only a pool-wide `clear` or `collision`, without private rows,
nonce, site or per-row findings. The outcome itself discloses an aggregate
bit. A collision, unavailable check or uncertain query stops this protocol;
there is no adaptive pool revision or another private query under it. Any
later study must disclose the query and review cumulative privacy in advance.

At the query and draw boundaries, also exclude the latest independently
reviewed known input and target-continuation records in both complete-row
prefix directions, including other studies' reservations. A public listing of
this study's unforwarded pool is not evidence of a model forward. If a registry
contains this study's own reservation, authenticate and remove only its exact
still-unforwarded records *before* building the forbidden union; do not
subtract their prefixes from the combined union, where another exposure may
share one. The pure helper's record-hash comparison does not itself prove
ownership, unforwarded status, inventory completeness or currentness.

After the pool-wide clear, separately bound phase claim and all checks, create
one nonce and derive the four families. Re-tokenize the selected 16 prompts,
verify each is a member of the exact cleared pool, recheck continuations and
the then-current public exposure set, and verify the old commitment is
unchanged. Any old-split subset integrity recheck emits no new per-row result;
its only failure outcome is a generic stop. A mismatch, collision, uncertain
claim or preparation failure consumes the fixed attempt without resampling.

The successor record uses canonical sorted-key, compact ASCII JSON under
schema `mlg-natural-successor-heldout-v1`. It binds the protocol, public pool,
pool-clearance receipt, selected exposure snapshot, old salted natural
commitment, unchanged prediction freeze and checkpoint, plus four ordered
families and their private tokenized cases. Its public commitment is SHA-256 of
`b'MLG-natural-successor-v1/heldout-commit\0' || nonce || b'\0' || record`.
The reveal verifier recomputes the allocation and exact record, rather than
accepting supplied rows. Publish and observe the commitment before any model
forward. Never publish a nonce, private rows, unsalted record hash or private
diagnostic before the authorized reveal.

If an injection peer has committed or possibly exposed inputs, a separately
reviewed trusted comparison against that fixed peer must pass before the
affected forward; an absent peer is explicitly unchecked, not clear. The
current transition caller fails closed for a committed peer and does not
authenticate a supplied receipt. A compatible verifier or reviewed custodian
procedure is required; no unrelated, absent track is a reason for an
indefinite hold. Future public inputs are coordinated with the custodian.

Immediately before the first-forward grant, re-observe the official current
exposure inventory and peer-gate state. Verify the actual selected rows against
any newer reviewed inventory and record that check separately; do not replace
the inventory identity already sealed in the commitment. A changed or
unavailable inventory, peer state or comparison must not be treated as clear:
proceed only after the required reviewed state and any affected private
comparison are established, otherwise stop without redraw. Coordinate a short
handoff with owners of inputs that could be forwarded across this boundary,
and perform the bound owner's post-run currentness check as prescribed. These
cooperative observations are point-in-time checks, not a global exposure lock.
If post-run review discovers an overlapping exposure, or a peer state whose
required comparison was not established, before or during affected forwards,
or the required comparison or timing is unavailable or uncertain, retain all
evidence and do not promote the result as an unexposed held-out evaluation. An
exposure demonstrably occurring only after the last relevant forward need not
invalidate that evaluation.

## Runtime, claim and result obligations

This successor needs a new one-use phase and cross-track identity plus a
separate reviewed allocation: no copy, reset, replacement or reuse of the old
ledger claim or markers. Freeze the exact logical and physical ceilings and
available headroom before its claim; the earlier v1 reservation is not a new
allocation. Account for all partial evidence, logs and orphaned temporary
checkpoint staging. The new owner, adapter, ledger/gate and independent auditor
must be published and hash-bound before drawing. The original sampler's
verifier only reconstructs its old fixed pool, so it cannot verify this new
record; a reviewed successor adapter must reproduce the original runner's
case order, controls and arithmetic without silently changing that science.

Before a claim or nonce, rehearse the actual production executable,
source-hash bootstrap, direct base interpreter, isolated flags, approved
site-packages path, source guards, environment and parent/child supervision
lineage with no model and no private inputs. Only predeclared validate-only
and test-configuration arguments may differ. The parent must observe clean
exit/reaping and a marker showing the frozen `_startup()` passed. The current
component bootstrap and startup probe do not by themselves constitute this
end-to-end rehearsal. Run with `MKL_CBWR` and `ATEN_CPU_CAPABILITY` unset; their
unknown original-discovery settings mean complete numerical runtime
equivalence is not established.

Retain the actual one-shot owner exit, all partial evidence and all failed
claims. Independently audit saved event order, landing, controls, tensors,
accounting and scoring before reporting or revealing any result. A failed or
incomplete attempt remains incomplete and is not retried or relabeled a
scientific pass. Hashes and process observations provide cooperative custody,
not independent attestation against the same account or proof that all prior
exposures are known.
