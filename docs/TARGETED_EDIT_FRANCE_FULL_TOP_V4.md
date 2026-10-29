# France full-vocabulary counterfactual edit: pre-forward protocol v4

This revision preserves the [original proposal](TARGETED_EDIT_FRANCE_FULL_TOP_V2.md)
and replaces its locality L10 before any new study-model forward: the original
`A triangle has three` was found to be an exact public reservation collision
with `planned-parameter-edit-followon-locality-11`. It also incorporates the
separately reviewed public-only exposure-scope amendment. No model outcome
informed either change; all other prompts, controls and decisions are fixed.

**For independent review; no new study model outcome is claimed.** This
exploratory stress test asks whether one persistent parameter edit can change
the unique full-vocabulary next token on held-out France-capital phrasings
while preserving measured behavior on separate locality prompts. The target
is deliberately counterfactual: Paris is the capital of France; London is not.
This is not a factual correction, a general knowledge test, or evidence of
general capability preservation.

The earlier [full-top screen](https://github.com/EastonSell/model-looking-glass/tree/main/results/targeted-edit-full-top-cpu-v1-20261029/screen)
did not qualify any fact under its original rule and performed no edit. Its
published rows did show three France prompts with unique top token ` Paris`.
That observation motivated this new design. The old result is not
reclassified. The three known rows are development data, never held out.

The pinned model is `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. The controlled continuation
tokens at the declared boundary are ` Paris` ID 6342 and ` London` ID 3576.
The site is the persistent CPU float32 parameter
`transformer.h.5.mlp.c_proj.weight`, shape [3072,768]. The edited model is
used without input-dependent hook gating on every evaluation and locality
prompt. This matters: checking unmodified prompts with the intervention
switched off would be a vacuous locality control.

## Fixed development inputs

These exact, already scored full-top-screen rows are used only for fitting
and efficacy calibration, respectively:

| Role | Exact prompt |
| --- | --- |
| fit | `Country: France\nCapital city:` |
| calibration 1 | `Country and capital: Spain - Madrid; France -` |
| calibration 2 | `Capital cities: Spain is Madrid; France is` |

Here and below `\n` denotes one ASCII line feed, not two literal characters.
The backslash before the pipe in T03 is only a Markdown table escape; its
actual prompt contains the literal `|` and no backslash.
The exact prior case identities are
`full-top-france-evaluation-04`, `full-top-france-efficacy-calibration-02`
and `full-top-france-evaluation-05`. Before an edit, freshly measure each
and require its current original-model unique top to be Paris. Save every
full row. If any fails, stop as **development-baseline-ineligible**, retain
all baselines, and do not construct or apply an edit.

The four previously published v1 locality-5 through locality-8 inputs remain
the projection-basis inputs, solely to construct the projection, as in the
[earlier projected protocol at commit 0c5d2bb](https://github.com/EastonSell/model-looking-glass/blob/0c5d2bbf7f0bffceb46e60d282e85f89f8fd8d0c/docs/TARGETED_EDIT_PROJECTED_CPU_V1.md).
They are not evaluation locality cases. Their reuse is exposed development
data, and a null on that basis is not a held-out locality result.

## Fixed target and locality prompts

Freeze all exact strings and answer sets below before any new model forward.
The eight target cases are measured once under the original model and then,
if the baseline gates below pass, under every valid predeclared condition.
Their intended factual and alternative first-token continuations are
` Paris` and ` London` respectively.

| ID | Exact target prompt |
| --- | --- |
| T01 | `Country: France\nNational capital:` |
| T02 | `Nation: France\nCapital:` |
| T03 | `France \| capital city:` |
| T04 | `Complete the geography entry. Country: France; capital:` |
| T05 | `Country / capital: France /` |
| T06 | `Capital of France:` |
| T07 | `Q: What city is the national capital of France?\nA:` |
| T08 | `Capitals: Portugal is Lisbon; France is` |

The 16 fixed locality cases are as follows. Commas in the accepted column
separate distinct exact one-token strings; every string begins with one
ASCII space. All listed answers must be independently reviewed for meaning,
and the pinned tokenizer must verify that each exact string appends one ID
without changing the prompt encoding. For each fit, calibration and target
row also require exactly `encode(prompt + " Paris") = encode(prompt) +
[6342]` and `encode(prompt + " London") = encode(prompt) + [3576]` with
the pinned tokenizer and no special tokens. If a listed answer fails tokenization
or meaning, stop before a model forward and amend this draft with review;
never delete or add accepted variants after seeing model outputs.

| ID | Exact prompt | Accepted one-token continuations |
| --- | --- | --- |
| L01 | `Country: Japan\nCapital city:` | ` Tokyo` |
| L02 | `Country: the United Kingdom\nCapital city:` | ` London` |
| L03 | `Country: Canada\nCapital city:` | ` Ottawa` |
| L04 | `Country: Germany\nCapital city:` | ` Berlin` |
| L05 | `Country: Italy\nCapital city:` | ` Rome` |
| L06 | `Country: Spain\nCapital city:` | ` Madrid` |
| L07 | `City: Paris\nCountry:` | ` France` |
| L08 | `City: London\nSovereign state:` | ` UK` |
| L09 | `The largest planet in the solar system is` | ` Jupiter` |
| L10 | `A square has four` | ` sides`, ` angles`, ` corners` |
| L11 | `In the calendar, Monday is followed by` | ` Tuesday` |
| L12 | `The opposite of winter is` | ` summer` |
| L13 | `A baby dog is a` | ` puppy`, ` pup` |
| L14 | `The plural of dog is` | ` dogs` |
| L15 | `The first month of the year is` | ` January` |
| L16 | `In the base-ten system, 2 plus 2 is` | ` four`, ` 4` |

L01, L02 and L05 were in the earlier full-top baseline screen and are
intentionally exposed sentinel rows; they are not fresh evidence. Before the
first baseline, compare all study and projection-basis rows against the
then-current known public inventories and already-fixed pending public bundles
in both complete-row-prefix directions. The three development and three
sentinel rows are exempt only for their explicitly identified full-top study
lineage and this attempt's own reservation; the four projection-basis rows
are exempt only for their identified earlier v1 lineage and own reservation.
Do not silently exempt another source. Any other public or pending collision,
missing required public source or unexplained newer public exposure stops
before forwarding. If a case must be changed for a pre-forward token,
semantic or public-exposure conflict, publish a revised complete protocol
and manifest and repeat independent review before a new model outcome.

The earlier private-check condition is withdrawn before model execution.
Do not query, open, access, compare against, or indirectly obtain information
from any separately protected or flagged split. Disjointness from private or
unknown history, undocumented uses and pretraining data is unverified.

Use tokenizer-only assembly after this proposal is reviewed and before the
first forward. Publish the exact UTF-8 texts, ordered IDs, prompt IDs,
attention masks, exact factual/counterfactual/accepted continuation IDs,
full appended token IDs, pinned tokenizer hash, selection source and
public/pending exposure reports. No tokenizer-only check proves model behavior.

## Baseline screen and fixed denominator

Before fitting, run the original model on all 27 listed study rows in fixed
order: fit, two calibration, T01–T08, then L01–L16. Save finite 50,257-way
float32 logits for every row and all actual model-entry IDs/masks. A target
is baseline-supported only when Paris is its unique full-vocabulary top.
A locality row is baseline-supported only when its unique top ID is one of
its preregistered accepted IDs. A tie is unsupported, with tie count and
lowest-ID representative reported separately; the representative is not a
unique prediction.

Let E be the number of supported target rows; let H be supported locality
rows; let H_new be the supported rows among L03, L04, L06–L16. Require
E >= 4, H >= 8 and H_new >= 4, in addition to all three development rows
being uniquely Paris-top. If a gate fails, report every baseline and every
reason as **baseline-ineligible**, retain the full raw evidence, and do
not construct an edit. Do not replace, rank by margin or add prompts.
If the gates pass, retain all eight targets and all 16 locality rows under
every valid edit, including baseline-unsupported rows. E, H and H_new are
fixed from the unedited screen; no edited outcome changes them.

The accepted sets are finite operational labels, not exhaustive lists of
correct continuations: other valid words or multi-token answers may exist.
Their top-token and NLL checks do not measure general task correctness or
prove a general capability. The target phrasings ask about the same fact;
they are held out from fitting but correlated, public and not pretraining
blind. Exact-input disjointness is narrower than semantic or pretraining
independence.

## Edit and controls

Use the existing CPU projected-edit construction and guards, adapted to the
fit row and this manifest. Compute the gradient of the final-position
`logit(London) - logit(Paris)` with respect to the site weight, recording
its input/output factors and full gradient. The projected-gradient update
points in the positive direction of that gradient, not its negation. Require finite nonzero factors,
a relative rank-one residual <= 1e-5, and byte-identical gradient-enabled and
no-grad fit site inputs, outputs and full logits, with original state and
buffer hashes. Save the factors and gradient for independent reconstruction.

Capture all site inputs from the four projection-basis prompts. Compute the
same CPU float64 SVD projection and relative singular-value cutoff 1e-10 as
the prior projected study. Require a nonzero projected factor, retained
fraction >= 1e-6 and normalized null residual <= 1e-8. Do not redraw or
alter a cutoff. The saved derivative is a gradient-enabled derivative;
matching forward values does not prove equal downstream Jacobians.

Test projected update fractions of original site-weight Frobenius norm in
this ascending fixed grid: 0.0005, 0.001, 0.002, 0.004, 0.008, 0.01.
At each point restore the exact original model, form the scaled update in
float64, cast the intended delta once to float32, and record post-addition
realized weight and delta hashes/norms. A nonzero realized norm must be
within [0.999,1.001] of its planned norm. Measure and retain every grid
point, even after a success. Select the first point with unique London top
on the fit row. If none succeeds, select 0.01 as a **development-top-miss**
descriptive fallback; the primary positive result is then impossible,
but the fixed evaluation still runs if its baseline gates passed.

At that selected fraction construct five fixed conditions: projected
gradient; equal-planned-norm unprojected gradient; projected random rank-one
direction; the exact opposite of the projected intended delta; and a
rank-matched nuisance-span update. Use the draw order, projection, QR checks
and no-redraw rules of the [screened construction at commit 877257f](https://github.com/EastonSell/model-looking-glass/blob/877257f270f34ccd728a5aea0a4c1e5594ae935b/looking_glass/targeted_edit_screened_math.py),
with its dedicated CPU random seeds 20261103 and 20261104. The new owner
must implement the stated variable-dose grids; the linked helper's fixed
0.001 dose and shorter grid cannot be silently substituted. The random and
nuisance directions are one fixed draw each, not a distribution. Each
same-norm control's realized norm must match the selected projected
realized norm within [0.999,1.001]. Restore and independently check
original weights, all model-state and buffer digests, hooks, gradients,
numeric mode and cleanup between conditions.

Measure both exposed efficacy-calibration prompts under the selected
projected update and the fixed unprojected dose grid
0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006,
0.008, 0.01, 0.012, 0.015. On each row, compute changes from its original
baseline in London log probability and London-minus-Paris logit margin.
A match requires both projected and unprojected gains in both measures on
both rows to be > 0.05; the top categories (Paris, London, other, tied)
must agree on each; and each gain discrepancy must be <=
max(0.005, 0.10*abs(projected gain)). Choose the eligible grid point with
the smallest maximum discrepancy/tolerance ratio; exact ties choose the
smaller dose. If no point matches, record every value and **match unavailable**;
do not interpolate, redraw or inspect evaluation outcomes to tune. An
uncertain or failed forward, nonfinite value or failed restoration is
incomplete, not a normal no-match result. The efficacy-matched comparison
is then ineligible; the other arms remain available. If its update equals
the equal-norm arm, reuse the identical bytes/outputs and disclose shared
evidence.

Before any edited target or locality forward, publish a reviewable
development record with the entire grid, calibration rows, planned and
realized update identities, selected dose, controls, exposure checks and
source identity. Treat chronology and process records as cooperative
observations, not authenticated execution timestamps.

## Output measurements and decisions

Evaluate all 24 target/locality rows with each of the five fixed updates
and, if available, the matched unprojected update. Reuse the verified clean
screen rows only as original baselines. Save all full-vocabulary logits,
unique top IDs and ties, London/Paris logits and probabilities, accepted-set
probabilities, exact per-case hashes, actual dose and all failures. Report
unsupported rows separately but do not omit them. For every row and each
condition also perform deterministic greedy decoding of at most three
tokens, stopping on EOS and using lowest-ID argmax at a tie; save each
step's full logits, exact prefix IDs, ties and generated IDs. The first
decoding row may be reused only when bit-identical to the saved evaluation
row. Prefixes after divergent generated tokens condition on different text,
so do not compare their distributions as if their histories were equal.
Before forwarding any generated continuation prefix, check that exact
prompt-plus-generated-token row against the then-current known public and
fixed pending inputs in both complete-row-prefix directions. Exempt only
this attempt's identity-bound base reservation and its own recorded prefixes;
any other public collision or unavailable required public check stops that
forward and marks the attempt incomplete. Do not query any protected or
flagged split, directly or indirectly. Save the public check and actual input
IDs before each call; a generated token alone is not a model forward.
These short continuations are descriptive; they do not establish a complete
answer or semantic correction.

Let K be the number of E baseline-supported targets whose projected edit
makes London the unique top. Let K_random and K_opposite count the same
event on those same E rows for the projected-random and opposite controls.
The projected behavioral gate requires K >= ceil(3E/4), K > K_random and
K > K_opposite. Report all counts and every row under every arm, including
other top-token changes and cases whose original Paris top was absent.

The locality safeguard requires every one of the H supported locality rows
to remain uniquely topped by an accepted answer under the projected edit;
mean `KL(original || projected)` over **all 16** locality distributions
<= 0.02 nats; maximum KL <= 0.10; and maximum accepted-set NLL increase
over all 16 <= 0.20 nats. Accepted-set NLL uses the sum of probabilities of
the distinct accepted token IDs. Decode saved IEEE binary32 bits exactly to
binary64; reject nonfinite values, probe positive and negative minimum
binary64 subnormal preservation on the scoring thread before and after
calculation, and never reset an unexpected denormal mode. For each row z, compute `m=max(z)`,
`L=log(fsum(exp(z[i]-m) for i in vocabulary order))`; for the sorted
distinct accepted IDs compute `a=max(z[id])` and
`A=log(fsum(exp(z[id]-a) for id in accepted order))`. A reported absolute
accepted NLL is `fsum((m,-a,L,-A))`. Compute its edited-minus-original
change directly as `fsum((m1,-m0,-a1,a0,L1,-L0,-A1,A0))`; do not subtract
separately rounded absolute NLLs. Compute London log-probability gain directly
as `fsum((z1[L],-z0[L],m0,-m1,L0,-L1))`, and London-minus-Paris margin gain
as `fsum((z1[L],-z1[P],-z0[L],z0[P]))`. Here indices L and P denote London
and Paris, not the log-normalizer L.

For KL, compute each original weight as
`p0[i]=exp(z0[i]-m0)/fsum(exp(z0[j]-m0) for j in vocabulary order)`,
and its log-ratio directly as
`d[i]=fsum((z0[i],-z1[i],m1,-m0,L1,-L0))`. Then calculate
`KL=fsum(p0[i]*d[i] for i in vocabulary order)` and the mean as
`fsum(KL)/16.0`. Natural exponential underflow to zero is permitted.
Reject nonfinite results, retain raw finite rounding (including a tiny
negative KL if it occurs), and do not substitute a lowest-ID tie
representative for a unique top. A fabricated regression must include
an unrelated maximum `2**60` and an accepted logit moving 0 to -1:
its accepted-NLL change is +1 even when the two separately rounded
absolute NLL values coincide. Baseline-wrong rows still contribute to
the KL and NLL safeguards. Also report exact top-ID changes separately from
changes between accepted variants and every individual NLL and KL.

The primary result is **positive** only when baseline gates passed, the
development top was achieved, the behavioral gate and the locality
safeguard all pass, and source/exit/evidence checks are complete. This
operational result can be positive even if the separate efficacy match is
unavailable; it does not establish a locality advantage over an
efficacy-matched unprojected edit.
Otherwise a completed, baseline-eligible result is **negative**, with all
failed gates listed. These are chosen operational thresholds, not
statistical significance or a claim of broad capability preservation.

The equal-norm comparison describes projected versus unprojected results at
the same planned/realized norm, even if target effects differ. For a separate
efficacy-matched locality comparison, a match must exist and on **every**
baseline-supported held-out target both projected and matched unprojected
gains in each of the two calibration measures must exceed 0.05, differ by
at most 5% of the larger absolute gain, and have the same top category.
If any fails, label this secondary comparison **ineligible: evaluated
strength imbalance**, show all data and do not retune. If all pass, report
both locality distributions and whether projected mean locality KL is at
most 0.9 times the matched mean and at least 1e-6 lower. This does not
prove exact causal equivalence. A missing development match is ineligible,
not a negative result.

Any nonfinite tensor, wrong bytes, failed state restoration, incomplete
forward, missing saved evidence, unverified source or public exposure, a
mismatch in this study's own fixed case/source custody, nonzero or
unknown direct-child exit, or uncertain cleanup overrides all numerical
labels as **invalid/incomplete**. Preserve partial outcomes and do not
retry an uncertain forward.

## Execution and evidence

Freeze and independently review the complete protocol, case manifest,
tokenizer/exposure checks, fixed scientific reader and owner/source before
model access. The CPU owner uses pinned offline assets, one thread,
deterministic settings and explicit binary32/binary64 denormal probes; E:
holds staging, temporary files, journals and raw evidence. Require a fresh
workload-specific C: and E: capacity check and a protected, initially empty
operator-created bytecode prefix. Do not claim a checked source path proves
all executed bytecode or a point process observation proves descendant
cleanup.

Retain ordered start/return/failure journal events, full raw logits and
generated-step prefixes, fit factors, gradient, intended/realized update
identities, discovery/calibration/selection and result records, source and
asset pins, owner and independently observed exit, and cleanup status.
Independently replay the saved data without a new model forward. The exact
schedule and conservative disk/memory/forward ceilings must be source-bound,
tested with a small untrained CPU fixture and reviewed before execution.
If capacity or ownership cannot be established, do not run the attempt.
