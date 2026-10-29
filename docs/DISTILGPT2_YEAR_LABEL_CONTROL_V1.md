# Year versus label donor shifts: public protocol v1

This protocol defines a small follow-on to the public explicit-year-direction
study. It asks whether a fixed-band output shift from an opposite-year donor
is larger than the effect of a same-year, opposite-label donor at the same
input L2 dose, across fresh matched contexts. A label swap is an active
lexical/context control, not a presumed null. A passing result would be a
prompt-specific difference between these interventions; it would not show a
numerical comparator, identify a unique circuit or rule out token-frequency
or contextual associations. A failed or reversed comparison is a result.

The earlier [explicit-year result](https://github.com/EastonSell/model-looking-glass/tree/10ddb8961d1d88b65cd3a4b9625f91eb82c5a65f/results/distilgpt2-explicit-year-direction-v1-20261029)
found its strict clean comparator gate failed 0/8, while a separately
declared generic donorward/control pattern passed 8/8 across after, before and
near. This design and its thresholds are informed by that public result.
It adds a donor that changes an earlier label while leaving the year fixed;
it is not a blinded or independent replication of the earlier selection.

The GPT-2-small [greater-than study](https://arxiv.org/abs/2305.00586)
and the [activation-patching methods study](https://arxiv.org/abs/2309.16042)
provide research context. This is a distinct six-layer DistilGPT2 experiment;
different donors, metrics and controls can change an intervention result.

## Frozen cases and provenance

Use `distilbert/distilgpt2`, revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, weight SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`,
and tokenizer SHA-256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.

Cross the ordered year-suffix pairs `(09,18)` and `(54,63)` with the ordered
label pairs `(alpha,beta)` and `(red,blue)`. These two year pairs and two
label pairs form four related families; each pair is used with both members
of the other factor. For each family, cross both labels, both anchor suffixes
and the wordings `after`, `before`, `near` using the exact template:

```text
The event tagged {label} took place {wording} the year 17{anchor} in the year 17
```

This yields 48 fixed public prompts. Enumerate families by year-pair index
then label-pair index, and within each family enumerate wording, label, then
anchor in the orders above. For each target, the **year donor** changes only
the anchor to the other member of its pair, keeping wording and label; the
**label donor** changes only the label to the other member of its pair,
keeping wording and anchor. All four target/donor corners are present.

Component-only checks with the pinned tokenizer found ` alpha`, ` beta`,
` red` and ` blue` each encode as one distinct token. After this protocol is
public, assemble the complete fixed cases without pretrained forwards and
verify the full unpadded prompt and contextual continuations. Each prompt
must be at most 32 tokens; each label and anchor swap must change exactly one
separately identified token, and their positions must be distinct and fixed
across matched prompts. All wordings must have aligned donor positions,
and the final token must be the fixed ` 17` token. Require all `01`-`99`
suffixes to append as distinct single tokens to each unchanged prompt.
Record `00` as a separate next-token mass; canonical `1700` retokenization
need not equal that token continuation. A structural or tokenization mismatch
stops the attempt; do not replace cases in response to model outputs.

Publish this protocol before assembling the full case file; publish the
fixed cases, independently reviewed executor and pre-run record before the
first pretrained forward. Before forwarding, compare fixed public inputs and
the scored input-plus-suffix rows against the latest reviewed public exposure
snapshot available, using both complete-row-prefix directions. If the snapshot
already reserves these cases, exclude an own record only after verifying its
unique case ID, exact token row and still-planned status; a duplicate,
mismatched or already-observed own record stops the attempt. Compare every
other known record and extension. Record the snapshot identity and any other
conflict, and stop rather than silently selecting replacements. Apply any
actually relevant protected-split rule;
otherwise record that private or outside exposure is not independently
checked. No claim of pretraining independence or secret holdout follows.

## Fixed output metrics

For each year pair `a < b`, fix the same output suffix sets for both anchors,
both labels and all three wordings:

```text
middle = {a+3, a+4, a+5}
above  = {b+3, b+4, b+5}
M = logsumexp(logits[middle]) - logsumexp(logits[above])
```

Decode retained little-endian binary32 bits exactly to binary64 first.
For ordered bands `A` and `B`, calculate `u=max(A)`, `v=max(B)`,
`p=fsum(exp(x-u) for x in A)` and `r=fsum(exp(x-v) for x in B)` in listed
suffix order; evaluate `((u-v)+log(p))-log(r)` in that binary64 order. Use
this evaluation for every written LSE difference. All listed six suffix IDs
must be distinct within a comparison and contextual tokenization must align.

For a target with anchor `y`, also report `W` as LSE of `{y+1,y+2,y+3}`
minus LSE of `{y-3,y-2,y-1}`. Hold its six tokens fixed across wordings and
labels. Report all clean `M`, `W`, the low-minus-high `M` contrast by
wording/label, full final logits, full-vocabulary normalized `01`-`99` mass,
separate `00` next-token mass and remaining vocabulary mass. These are
next-token contrasts, not generated complete years. No clean-behavior gate
licenses a comparator claim in this study; the fixed interventions run for
all cases unless a structural, nonfinite, dose or evidence check stops them.

## Matched interventions

Use the zero-based layer-3 block output at the final token. First record an
unhooked clean and passive no-op for every case. For each family compute the
12 unique donor edge norms from saved no-op states: two year-swap edges and
two label-swap edges for each of three wordings. For target `x` and donor `y`,
decode each of 768 float32 channels exactly, form `d_i=y_i-x_i` in binary64
and compute `n=sqrt(fsum(d_i*d_i))` in ascending channel order. Set the
family's common requested input L2 dose to `q=0.5*min(all 12 n)`. Nonfinite
values or `q <= 2**-20` stop the attempt and retain available evidence; do
not exclude an inconvenient family. Record all 12 norms, q, scaling fractions
and achieved doses. A weak label edge can make every family action weak.

Require every passive no-op's saved before and after block tensors to be
bit-identical. For each target use these five actions, all at final position
unless stated:

1. Exact self replacement.
2. The signed year-donor displacement scaled to q.
3. The signed label-donor displacement scaled to q.
4. The same signed year-donor displacement scaled to q at position zero.
5. The same signed year-donor displacement cyclically shifted one channel at
   the final position: output channel `i` takes displacement `(i-1) mod 768`.

For non-self actions form each delivered channel in binary64 as
`target_i + (q/n)*d_i` in that order, then apply the integer-defined
round-to-nearest-even binary64-to-binary32 conversion and preserve signed
zero. Require actual pre-action bits to equal that target's saved no-op at
the site, exact planned/actual site bits, unchanged all other coordinates,
and realized same-invocation L2 error no more than `1e-5*q`. Verify natural
position-zero equality for every same-wording family separately as a
zero-dose architectural check. Require bit-identical clean/no-op/self and
clean-after full logits, stable model/buffer/runtime and hook removal.
Equal input L2 does not imply equal downstream effect, and any control can
have a nonzero effect.

Use the five action order above rotated left by
`(family_index + 2*wording_index + label_index + 3*anchor_index) mod 5`.
For each case save an unhooked clean-after following all five actions.
Rotation is deterministic and not perfectly balanced. The complete schedule
is 384 forwards, independent of clean behavior: 96 clean/no-op, 240 actions
and 48 clean-after. Save each complete 50,257-element final-logit vector and
both full before/after block witnesses for each of the six hooked actions.
Bound raw logits by 77,194,752 bytes and witnesses by 56,623,104 bytes at
32 tokens, for 133,817,856 aggregate raw bytes, plus separate bounded
metadata/partial-evidence limits. These are file bounds, not an RSS bound.

Use cached weights, CPU float32, eager attention, no KV cache, one intra-op
thread and fixed CPU RNG seed 20261029. Capture absence/value of `MKL_CBWR`
and `ATEN_CPU_CAPABILITY` before Torch import, record actual runtime and
numeric probes, and guard CUDA/network entry points cooperatively. A seed
does not guarantee bit identity across runtimes. Preserve every available
bounded returned payload and failure record before later validation; no
automatic retry or outcome-based exclusion is permitted.

## Prespecified comparison and reporting

For each target/action, `E_j=M(patched_j)-M(clean)`. Define signed
year-donorward movement `s_year=-E_year` for a low-anchor target and
`s_year=+E_year` for a high-anchor target. The label donor does not change
year, so its raw `E_label` has no donorward-year interpretation. Position
and rotated controls also retain their signed raw `E`.

An individual target satisfies the matched-control criterion only if all
three strict inequalities hold:

```text
s_year > abs(E_label) + 0.1
s_year > abs(E_position_zero) + 0.1
s_year > abs(E_rotated_year) + 0.1
```

Thus a large opposite-signed control cannot cancel a control effect or pass
merely because its signed average is small. A family passes only if all 12
of its targets pass, covering both donor directions, both labels and every
wording. The predeclared descriptive pattern is supported if at least three
of the four crossed families pass. Thresholds were chosen after seeing the
earlier public study, but before any outcome on these cases; they are not a
statistical significance test. With fewer than three passing families the
prespecified pattern is not supported; that does not establish no effect or
equivalence. Preserve opposite, near-zero and label-dominated results.

Report every target's clean `M` and `W`, every action's raw `E`, signed
`s_year`, all three absolute-control comparisons and margin differences,
each failure, all four family booleans and the aggregate. Report the two
year directions and the two label directions individually before any means;
means use `math.fsum` in fixed family/case order. Include all q/norm and
delivered-bit checks. A successful contrast establishes neither a general
year representation nor arithmetic comparison: label identity, model
nonlinearities, token-level priors and the chosen contexts remain plausible
explanations. The four crossed families share years, labels and template and
are not independent population samples.
