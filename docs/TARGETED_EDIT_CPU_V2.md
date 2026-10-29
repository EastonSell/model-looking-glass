# Development-selected targeted edit: CPU protocol v2

**Status:** prospective draft. CPU scoring, edit construction, public-inventory
and saved/committed-selection verification helpers exist. The complete
execution owner and a frozen manifest are not yet available. The committed
handoff checks saved inputs and attestations but does not authorize a model
forward. No v2 model run or result is claimed.

This is a prospective, exploratory follow-up to the published
[v1 negative result](../results/targeted-edit-cpu-v1-20261028/README.md).
That result showed a large counterfactual change in a London-versus-Paris
next-token contrast but failed several locality limits. V2 asks whether a
smaller, development-selected edit changes that tradeoff on another small set
of public cases. Paris is France's capital; this is a deliberate counterfactual,
not a factual correction or a model intended for deployment. A positive
London-minus-Paris logit contrast does not establish which token is most likely,
what generated text says, or what the model believes. This is not a causal
localization study or evidence of broad capability preservation.

All v1 cases and outputs have been observed and are **development data** in
v2. The new cases below are public and non-blind: they are held out from dose
selection, not from prior human discussion or the model's pretraining. The
v1 failure remains unchanged. V2 deliberately relaxes its target-reversal and
gain thresholds after observing v1; the two success labels are not comparable.
Report the raw common metrics and original v1 scalar thresholds on the new
sample as well as the v2 decision. V1's discovery criterion is evaluated only
on the v1 development discovery row, not on the new evaluation cases.
The v1 scalar-threshold view applies those thresholds to the selected smaller
edit and its new, smaller-dose random control. The 1% gradient edit is a
separate same-input reference; this view does not repeat v1's comparison with
its 1%-matched random control.

## Fixed checkpoint, site and arithmetic

Use the v1 pinned `distilbert/distilgpt2` checkpoint at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, seven verified assets, eager
attention, evaluation mode, no generation cache, float32 CPU and one
intra-op and inter-op thread. Use no GPU. The site is only
`transformer.h.5.mlp.c_proj.weight`, shape `[3072, 768]`; the discovery
objective is final-token `logit( London) - logit( Paris)`, with token IDs
3576 and 6342. No optimizer, bias edit, or additional fitting prompt is used.
Use the v1 protocol's fixed tokenization, runtime identity, denormal probes,
finite checks, gradient/outer-product agreement, restoration and artifact
encoding. Its recorded source and evidence remain unchanged. Check the same
execution-relevant Python, package/build and GPT-2 source identities; record
path fingerprints separately, since a fresh checkout may change a path without
changing execution-relevant bytes. Hashes and probes are attestations with
the limits described in v1; they do not attest every native byte or thread.

Before evaluating the grid, recompute the v1 discovery forward and the
unmodified model-state and site-weight hashes. Reconstruct the v1 1% gradient
candidate and v1 random candidate using its dedicated seed `20261029`, and
require the output, state, weight and both candidate digests to agree with the
published v1 precommit. Any mismatch or incompatible execution-relevant
runtime is `incompatible`, not a reason to retune. The v1 random direction is
used only for this identity check: its weak effect on the development data is
already known, and no random-direction distribution was measured.

Let `D1` be the reconstructed v1 intended float32 gradient update and `W` the
original float32 site weight. For each relative dose `d` below, construct the
intended selected update as
`float32(float64(D1) * (d * ||float64(W)||_F / ||float64(D1)||_F))`.
Require finite nonzero inputs and result. The calculation uses float64 norms
and multiplication followed by one float32 cast, then float32 weight addition.
The original gradient construction is rank one before rounding; neither the
rescaled nor the realized float32 difference is claimed to have exact rank
one. A relative Frobenius norm is not an operator-norm or behavioral bound.

For the v2 random control use one new dedicated CPU PyTorch generator seeded
`20261030`. Draw one float32 standard-normal vector of length 3072 followed by
one of length 768, form their float32 outer product, and scale it in float64
to the selected planned norm before casting once to float32. Do not redraw,
select directions, or reuse the observed v1 random candidate. The opposite
control is the elementwise negation of the selected intended float32 update.
For each edit, measure the actual applied difference as
`float64(float32(W + delta)) - float64(W)`. Require each small-control norm
divided by the selected-edit norm to lie in `[0.999, 1.001]`; otherwise the
attempt is invalid. Require every grid, selected and small-control realized norm
to be finite, nonzero and within `[0.999, 1.001]` of its own planned norm;
also require the reconstructed 1% reference's realized norm to equal the
binary64 value parsed from its published v1 attestation exactly. Record the
planned norm, intended update digest, realized norms,
relative norms and resulting weight digests. A single random draw is not a
distribution of random-edit effects. Model and update bytes are not published.

## Development selection

Publish this protocol, reviewed executable source and a separately committed
manifest before **any v2 model forward**. The manifest binds the source and
runtime requirements, pinned assets, all cases and exact token IDs, v1
compatibility identities, exposure-reservation snapshot, dose grid, control
seed, metrics and decision rules. Before forwarding, verify all 26 complete
development/evaluation rows are distinct and no complete row is a prefix of
another; compare the 13 new rows with known recorded exposure prefixes before
their own reservation. Do not inspect another team's private or reserved
cases. If a custodian has unexposed reserved rows, obtain only its documented
exact/prefix pass/fail result; otherwise disclose that no such comparison was
performed. Unknown prior exposure and pretraining remain unverified.
The manifest records the queried lanes, their reported scope and observation
time as an operator attestation, not independent or exhaustive evidence. Recheck
the applicable custodians before the first forward. Publish a separate receipt
after the manifest and before the development run, binding the manifest, v4
inventory and public row digests to each lane's reported time and either no
protected inputs reported or a private exact/prefix collision pass. Record a
fresh, actually performed comparison against the latest known public inventory
and its reported digest and scope; a syntactically valid digest or zero count
does not perform that comparison. Preserve collision and unavailable reports
as blocked receipts. If protected rows have since been generated, obtain and
preserve a pass/fail collision result without reading the rows. The receipt
stores no protected rows and remains an
operator attestation. A change requiring
different cases or controls requires a new protocol and manifest, not an
unrecorded substitution.

Reconstruct and check v1 as above. Then measure the unedited baseline and a
fixed, ascending grid of relative Frobenius doses
`[0.00025, 0.0005, 0.001, 0.002, 0.004]` on all 13 v1 development inputs.
For each condition restore and verify the same exact original model state
before and after; retain every full final-position float32 logit vector, its
case and condition identity, a checksum, the intended and realized edit norms,
per-case metrics and any failure. Never infer these outcomes from v1's 1%
results. The v1 discovery row in the new baseline must byte-match the
discovery forward used to construct the candidate.

For each dose, use `m = logit(London) - logit(Paris)` from the saved float32
logits promoted to float64. A candidate qualifies only if all these conditions
hold:

* The original discovery margin is negative and at least three of the four
  original development target margins are negative.
* The discovery margin becomes strictly positive, at least one originally
  negative development target margin becomes strictly positive, and the mean
  change across all four development target margins is strictly above 0.5.
* All eight development locality top tokens are unchanged; their mean
  `KL(original || edit)` is at most 0.02 nats, maximum KL at most 0.10 nats,
  and mean expected-token NLL deterioration at most 0.05 nats.

Select the qualifying dose with greatest mean target-margin change; resolve
an exact tie in favor of the smaller dose. If no dose qualifies, select the
smallest dose as a fixed diagnostic fallback and record
`development_qualified=false`; an evaluation still runs, but cannot be called
successful. Preserve all five dose outcomes and failures. No new evaluation
output may influence this selection or any criterion.

Before a fresh evaluation process runs a new case, create and publish a
hash-bound development-selection record and all full development outputs in a
separate descendant commit. It binds the exact selected dose, individual
criteria and tie/fallback path, point and sign predictions, intended edit and
both control digests, realized norms, v1 identity checks, frozen manifest,
checkpoint, runtime and cleanup observations. An independent reviewer fetches
the published record before evaluation begins. Remote observation, local
clocks and Git metadata do not by themselves prove execution chronology.

## New public evaluation cases

The manifest freezes these exact strings and token IDs. Each prompt is encoded
without special tokens, padding or truncation. Check that tokenization of the
prompt plus each listed continuation equals the frozen prompt IDs followed by
the single indicated token; fail rather than substitute or truncate. No case
below has been forwarded to the model as part of v2 before freezing.

| ID | Prompt | Exact prompt IDs | Continuation |
| --- | --- | --- | --- |
| eval-target-1 | `The national capital of France is` | `464, 2260, 3139, 286, 4881, 318` | ` London` / ` Paris` |
| eval-target-2 | `France's capital is the city of` | `28572, 338, 3139, 318, 262, 1748, 286` | ` London` / ` Paris` |
| eval-target-3 | `France's seat of government is in` | `28572, 338, 5852, 286, 1230, 318, 287` | ` London` / ` Paris` |
| eval-target-4 | `The French capital city is` | `464, 4141, 3139, 1748, 318` | ` London` / ` Paris` |
| eval-locality-1 | `The capital of China is` | `464, 3139, 286, 2807, 318` | ` Beijing` (11618) |
| eval-locality-2 | `The capital of Russia is` | `464, 3139, 286, 3284, 318` | ` Moscow` (9070) |
| eval-locality-3 | `The capital of Canada is` | `464, 3139, 286, 3340, 318` | ` Ottawa` (14074) |
| eval-locality-4 | `The capital of Portugal is` | `464, 3139, 286, 19101, 318` | ` Lisbon` (41898) |
| eval-locality-5 | `The opposite of large is` | `464, 6697, 286, 1588, 318` | ` small` (1402) |
| eval-locality-6 | `An hour contains sixty` | `2025, 1711, 4909, 24742` | ` minutes` (2431) |
| eval-locality-7 | `A triangle has three` | `32, 22950, 468, 1115` | ` sides` (5389) |
| eval-locality-8 | `The sun rises in the` | `464, 4252, 16736, 287, 262` | ` east` (7627) |
| eval-locality-9 | `In France, the predominant language is` | `818, 4881, 11, 262, 45718, 3303, 318` | ` French` (4141) |

Four locality probes share the country-capital relation, four cover other
topics, and one retains France while changing the relation. Their predefined
continuations are probes, not claims that the baseline answers correctly.
These few correlated cases do not characterize general capabilities.

## Evaluation, decision and evidence

The separate evaluation process reloads the pinned checkpoint and verifies
the manifest, published development-selection record and reconstructed
candidate and control digests before any new evaluation forward. In the fixed
order `original`, `selected`, `random`, `opposite`, `v1_1_percent`, evaluate
all 13 new cases, restoring and verifying the exact original state between
conditions. The last is the frozen v1 1% gradient edit on the **same** cases,
not a new candidate for selection. Compare its raw per-case effects with the
smaller edit descriptively. Record full logits, norms, state/cleanup checks
and all per-case and aggregate measurements for each condition, including
failed criteria. Record partial, failed and cancelled attempts immutably.

Derive scores using v1's exact float64 log-softmax, NLL, KL and smallest-ID
top-token tie convention. The selected dose has two distinct predeclared
predictions for each target: positive margin change, and point change equal to
its mean development target-margin change. Count a sign prediction as correct
only when observed change is strictly positive; zero is incorrect. Report all
four changes, correct count out of four, and mean absolute error of the four
point predictions, with no censoring. The point prediction may be nonpositive
under the diagnostic fallback.

The v2 success criteria are `development_qualified=true`, at least three of
the four original evaluation target margins negative, at least two of those
originally negative margins becoming strictly positive, mean target-margin
gain strictly above 0.5 and at least 0.5 greater than the random control, all
nine locality top tokens unchanged, mean KL at most 0.02, maximum KL at most
0.10, and mean expected-token NLL deterioration at most 0.05 nats. Report
every locality KL, NLL and top-token change; an aggregate can conceal an
individual regression. The sign-reversed and 1% conditions are reported as
controls/references, not additional selection criteria.

Terminal precedence is explicit: execution, provenance, incomplete artifact
or unverified-cleanup failure is invalid/incomplete; otherwise a baseline with
fewer than three negative evaluation target margins is ineligible; otherwise
an unqualified development fallback or any failed criterion is negative; only
all passing gates give success. Retain each raw criterion and
`development_qualified` separately. If no dose qualifies, a fallback evaluation
cannot become successful even when its evaluation numbers look favorable.
The scoring helper's decision is provisional: the owner and independent auditor
must apply these nonnumerical gates before reporting a final study outcome.

The saved output auditor independently recomputes metrics and verifies
artifact/identity links and the frozen decision. It cannot prove the truth of
execution, public-before-forward timing, exact edits or rollback from logits
alone. Runtime and source hashes do not attest every imported/native byte.
The committed handoff rejects custody observations dated after the recorded
development start; consistent self-reported clocks do not prove chronology.
Where possible, repeat the complete process on the same frozen source and
preserve the original attempt, replay and any failures. This CPU-only study is
not a performance benchmark.
