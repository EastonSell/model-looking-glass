# Year versus label donor shifts: exposed-family continuation, protocol v2

This protocol defines a new, public, exploratory continuation of the stopped
[v1 protocol](https://github.com/EastonSell/model-looking-glass/blob/9742d1b6b8858161e4e360f47eacad8db5a8f4b9/docs/DISTILGPT2_YEAR_LABEL_CONTROL_V1.md).
It asks whether the three remaining fixed families show year-donorward
next-token shifts larger than the corresponding label, position and rotated
controls at a closely matched input dose. A passing pattern is specific to
these matched prompts and interventions; it does not identify numerical
comparison, a unique circuit, or eliminate lexical, contextual or frequency
explanations. The label swap and other controls can themselves have effects.

V1 stopped after 168 of 384 planned forwards under its original relative
`1e-5` dose check. All 48 prompts had already been forwarded for clean and
no-op; the first family completed its five interventions and clean-after for
all 12 targets. Independent saved-byte audits found 0/12 of that family's
targets satisfied the full strict control criterion, although all 12 had
positive year-donorward movement. The four-family v1 decision is **incomplete**.
The other three families had no intervention forward. Their clean outputs and
no-op activations are already exposed development data, so this is not blind
or held out, and neither a new result nor this altered tolerance completes,
repairs, or retroactively changes v1. V1's stopped evidence and negative
first-family observations will remain separately published.

## Fixed cases and scope

Use the exact unchanged 48-row v1 case file
`experiments/distilgpt2_year_label_control_v1.json`, SHA-256
`07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08`.
Run only its ordered rows 12 through 47 inclusive: original zero-based family
indices 1, 2 and 3, with 12 cases per family. Their pair/label combinations
are `(09,18)` with `(red,blue)`, `(54,63)` with `(alpha,beta)`, and `(54,63)`
with `(red,blue)`. Preserve the original case IDs, token IDs, metadata, donor
links, fixed candidate bands and original family/wording/label/anchor indices;
do not reindex the rotations or use v1 family 0 in a v2 aggregate. Every
case's reciprocal donor is in the selected family. Recheck the pinned
tokenizer, full prompts and contextual continuation alignment as in v1,
including the distinct `00` next token and its full-string retokenization
exception. A mismatch stops the attempt rather than substituting cases.

Use the same model revision, model and tokenizer hashes, cached CPU/eager
runtime, seeds, metrics, action definitions, donor mappings,
no-op/position-zero checks, action rotation, input order and scientific
controls as v1 unless this document explicitly overrides them. The original
v1 instruction requiring own exposure records to remain planned is superseded:
these 36 rows are already exposed. Verify each unique own record by exact
case ID and token row, record its actual latest public status and v1 saved
evidence, and compare against all other relevant public records and extensions
without treating the already known own exposure as a collision. Bind the
latest reviewed inventory and separately record any known later exposure.
The old sealed natural split was not checked for these rows; public-only
comparisons do not establish private or pretraining disjointness. Respect any
actually applicable frozen cross-split rule, without inventing a private check.

## Revised input-dose fidelity rule

Each family retains the v1 dose `q = 0.5 * min(12 donor-edge norms)` and the
same binary64 evaluation order for differences, `fsum`, square root, scale,
and channel multiplication. A family with nonfinite values or `q <= 2**-20`
stops. Let `q` be the exact rational value of its recorded finite binary64
dose. For one non-self planned action, let `b_i` and `z_i` be the exact
decoded binary32 pre-action and proposed replacement values, and let `d_i`
be the exact rational value of the binary64 displacement produced by the
v1 scale/multiply order, including the prescribed channel rotation. Proposed
replacement bits are the v1 integer-defined round-to-nearest-even float32
conversion of the binary64 `b_i + d_i`; signed zero is preserved.

Using exact rational arithmetic in ascending channel order, form

```text
D2 = sum(d_i * d_i)
R2 = sum((z_i - b_i) * (z_i - b_i))
E2 = sum((z_i - b_i - d_i) * (z_i - b_i - d_i))
```

Every non-self plan must satisfy all three exact comparisons:

```text
(q * (1 - 1/10**12))**2 <= D2 <= (q * (1 + 1/10**12))**2
(q * (1 - 1/5000))**2   <= R2 <= (q * (1 + 1/5000))**2
E2 <= (q/5000)**2
```

The latter two limits bound both the norm error and the complete vector error
by 0.02% of q; comparing only norms could miss a direction error. The first
checks the intended binary64 scaling separately. Independently recompute
these exact comparisons from the actual same-forward before/after bits and
the fixed displacement, not just the plan. Require actual before bits to
equal the passive no-op source, actual after bits to equal all planned site
bits, untouched coordinates to match, and the self action to be bit-identical
to its input. A nonfinite value, overflow, zero delivery, mismatch or failed
comparison stops the attempt; no vector, target or family is excluded and no
tolerance is increased during the run. Record q, all 12 edge norms and
fractions, the exact numerator/denominator for D2, R2 and E2 and their limits,
plus human-readable norm and vector-error summaries for every applicable
action. The full saved before/after tensors, expected site bits and donor
no-op tensors permit independent reconstruction.

The v2 bound was chosen **after** the v1 stop and after inspecting the 144
non-self plans reconstructed from the three families' saved no-op activations.
An independent exact-arithmetic analysis found their maximum relative norm
error about `1.1327426e-5`, maximum vector error about `8.1446254e-5`, and
maximum coordinatewise-rounding certificate about `9.7384557e-5`; `1/5000`
is slightly over twice the last value. This is a numerical engineering choice
informed by already exposed inputs, not an independent calibration or a
claim that future plans must pass. The vector and norm limits bound input
fidelity; even equal L2 does not make the different controls functionally
equivalent. Retain all signs and continuous effects whether or not a target
or family passes.

## Schedule, stopping, and outcome

First run fresh clean and passive no-op forwards for all 36 selected cases.
The v2 clean logits and passive no-op activations define the v2 scores,
donor edges and plans; neither is required to match v1 bits across runs. For
each case, compare the new clean-logit and passive no-op activation bytes
with the retained v1 bytes and report equality or difference descriptively;
that comparison is not an eligibility rule. Within v2, all same-invocation
no-op, self, clean-after, landing and model checks remain mandatory. Recompute
q, all plans and their certificates from the new v2 no-ops; the v1 analysis
does not guarantee that a new plan will pass.
Then plan and run each of the three families in the original case-file order;
for each target rotate v1's five actions using its original stored rotation
and finish with an unhooked clean-after. The complete schedule is 288
forwards: 72 clean/no-op, 180 intervention actions and 36 clean-after.
With fixed 15-token cases, expected raw logits and witnesses total 77,802,624
bytes; the upper bound at 32 tokens is 100,363,392 bytes, plus separately
bounded metadata and partial evidence. Bounds are not RSS guarantees. Save
each full 50,257-element final-logit vector and full hooked before/after
block tensor, plans and journal events. Require no-op/self/clean-after logit
bit identity, exact landing and untouched bits, runtime/model/buffer checks,
hook removal and direct-process observation. Retain any available bounded
payload on failure and stop without automatic retry.

Retain v1's fixed-band `M`, within-anchor `W`, full-vocabulary mass reporting,
and each target's signed year-donorward effect and raw signed controls. Each
target still passes only if its year-donorward effect is strictly greater than
the absolute effect of **each** of the label, position-zero and rotated-year
controls plus 0.1. A family passes only if all 12 of its targets pass. Report
all 36 target outcomes and control magnitudes/signs, then the three individual
family booleans and count. The v2 descriptive pattern is supported only if all
three families pass; fewer than three does not establish no effect or
equivalence. This rule preserves at least the original 3/4 fraction on a
three-family denominator, but the denominator and numerical tolerance were
chosen after v1's partial observations. It is not a population-level test,
blind validation, or a replacement 3/4 v1 decision. The three related families
reuse years, labels and a template and are not independent samples.

Publish the v1 stopped evidence, this protocol, exact executor, case selection
and pre-run record before any v2 pretrained forward. Pin explicit bounded
metadata, journal, and partial-file sizes in the executor and pre-run record.
Freeze the source and have independent
tokenization, arithmetic, scheduling, failure-evidence and saved-data reviews.
Record the observed current inventory and source, cache/model/runtime hashes,
direct child exit and any post-launch changes with their actual chronology.
Public status, direct-process observation and saved-bit consistency are not
authenticated proof of all execution or all outside exposure.
