# Protocol: year-conditioned preference transfer

This fixed public, non-blind CPU protocol precedes case construction and
any new pretrained forward. It concerns the pinned
`distilbert/distilgpt2` checkpoint. It separates a narrow test for an
opposite-direction comparison pattern from an exploratory intervention on
year-conditioned next-token preferences. Neither outcome establishes an
arithmetic algorithm or a unique circuit.

## Development evidence and hypotheses

The published [48-prompt screen](../results/distilgpt2-greater-than-screen-v1-20261028/README.md)
found a later-suffix preference under forward, reverse and neutral wording.
A **post-hoc** analysis of its saved float32 logits chose six overlapping
year pairs `(12,32), (21,43), (32,54), (43,65), (54,76), (65,87)` and two
event nouns. Keeping the same three-token middle and
above-both bands across each pair, the mean low-minus-high band contrast was
5.173 forward, 5.120 reverse and 4.498 neutral; all 12 observations were
positive, and forward minus neutral averaged +0.675. These 12 observations
reuse six overlapping pairs and are not independent samples. The token bands,
pairs, thresholds below and this hypothesis were selected after that screen.
No model call was used for this development analysis.

The strong, symmetric comparison-pattern prediction is **opposite**
year-direction changes in forward and reverse sentences on the *same*
output-token bands. The existing reverse data did not show this pattern. Its
absence on these prompts does not rule out weaker comparison effects or a
numerical capability expressed under other conditions. A separate, weaker prediction is that
the final-position layer-3 residual carries a year-conditioned preference:
moving that residual partway toward its opposite-year donor will shift the
fixed-band preference toward the donor in forward and neutral contexts. A
forward-specific increase beyond neutral is a further exploratory contrast.
Contextual associations, lexical effects, position-dependent processing and
other alternatives can produce such shifts; it is not proof of comparison.

## Prospective public inputs

After this protocol is frozen, construct four start-suffix pairs
`(05,14), (27,36), (49,58), (71,80)` and the nouns `war` and `reign` in
century `17`. None of the eight start suffixes appeared as a start suffix in
the development screen. For every pair, noun and low/high suffix, use the
three *unchanged* forward, reverse and neutral templates of that screen.
This yields eight pair/noun observations and 48 different prompts. Case
selection is deterministic and outcome-independent, not random; reusing two
nouns and a template family limits generalization. Publish the 48 full rows
separately and bind them to the latest reviewed public exposure record before
a pretrained call. Consult a protected-split custodian when an applicable
protocol requires a comparison; otherwise record that unknown private or
unrecorded exposure was not independently checked. Unrelated held-out tracks
are not prerequisites for this study.

For a pair `(a,b)` with `a < b`, the common middle band is
`{a+3,a+4,a+5}` and the above-both band is `{b+3,b+4,b+5}`. Their token IDs
are held fixed for all six prompts of that pair/noun. Before any pretrained
forward, verify full prompt tokenization, contextual concatenation of all six
candidate continuations, unique one-token IDs, the final ` 17` token, and
that the low/high inputs in each wording differ only at the first year suffix
token. Also verify the passive capture positions and maximum sequence length
at most 32. Fail the entire attempt on any structural mismatch; do not replace
cases in response to model behavior.

## Clean behavior gates

For final-position logits `z`, let `M(z)` be the stable binary64
log-sum-exp of the three middle logits minus that of the three above-both
logits. Both bands and their order stay fixed when the input year changes.
For each pair/noun and wording `c`, let `D_c=M(z_low,c)-M(z_high,c)`.
The fixed comparison controls a moving output-token set, but fixed token
frequency bias can still interact with contextual reweighting inside a band.
More exactly, promote each saved float32 logit to binary64, let `u` and `v`
be the respective band maxima, compute `A=math.fsum(exp(z-u))` and
`B=math.fsum(exp(z-v))` in ascending suffix order, then evaluate
`M=((u-v)+log(A))-log(B)` in that order. Evaluate each difference in the
order written here. The narrow null of no wording-specific year-dependent
shift says `D_forward-D_neutral = 0`; context-dependent lexical associations
can also violate that null.

Before any intervention, record all 48 clean full-vocabulary logits and
passive no-op captures. A **comparator-pattern gate** passes only if at least
six of eight pair/noun observations, spanning at least three of four year
pairs, have `D_forward > 0.5` and `D_reverse < -0.5`. This is an operational
gate for studying the specified symmetric pattern, not a necessary or
sufficient condition for numerical computation in general.

Independently, an **exploratory preference-transfer gate** passes only if at
least six of eight observations, spanning at least three year pairs, have
`D_forward > 0.5`, `D_neutral > 0.5`, and
`D_forward - D_neutral > 0.1`. Thresholds are fixed operating criteria,
informed by the post-hoc development values, not significance tests. Report
every individual `D`, all signs, and both gate results. If neither gate
passes, do not perform the interventions: publish the complete clean
negative/unstable result and prioritize a distinct, stronger behavior. If
either gate passes, perform all predeclared intervention arms on all cases;
never select a subset by its outcome. Failure of the comparator gate prohibits
a comparison-specific claim even if the exploratory gate passes.

## Matched intervention and controls

Use CPU float32 evaluation mode, eager attention, no cache, one intra-op
thread and the pinned checkpoint. At zero-based transformer block 3, capture
the full block output after the block at all positions. For each wording and
pair/noun, the opposite-year prompt with the same wording supplies the donor;
test low-to-high and high-to-low directions separately. The primary site is
the final ` 17` token, whose token ID is the same in all prompts. This site
and layer are fixed prospectively; the choice is not a discovery result.

Decode float32 donor and target vectors to binary64. For each channel `i`
in ascending order evaluate `delta_i = donor_i - target_i`, then evaluate
`n = sqrt(math.fsum(delta_i * delta_i))` in binary64 without fused
multiply-add. Evaluate `scale = m/n` and each displacement as the single
binary64 multiplication `scale * delta_i`; evaluate each new coordinate as
the single binary64 addition of its decoded target value and that
displacement before rounding to float32.
Within a pair/noun, first capture all six no-op vectors; define a common
absolute input dose `m = 0.5 * min(n_forward,n_reverse,n_neutral)`. Each
direction uses the same `m` and its own signed `delta`. Require finite inputs
and `m > 2**-20`; otherwise stop the whole attempt and retain the evidence.
No outcome is used to choose the dose. This cross-case adaptive dose can be
made small by one condition, so report all norms and fractions `m/n`.

For each target use four hooked actions in fixed left-rotation order based on
`(zero_based_pair_noun + 2*condition_index + target_year_index) % 4`:

1. **Self replacement:** replace the final vector with its own saved bytes.
2. **Year-directed final-site dose:** set each final coordinate to the
   float32 round-to-nearest-even result of `target + (m/n)*delta`, computed
   in binary64 from exact float32 values.
3. **Same-dose position control:** add the same signed `(m/n)*delta` at
   position 0 of the target's own block output, leaving all other positions
   unchanged.
4. **Same-site, same-dose direction control:** add a fixed one-channel cyclic
   permutation of that signed displacement at the final position: output
   channel `i` takes the displacement from channel `(i-1) mod 768`.

Each action's pre-action target tensor must match that prompt's saved no-op
tensor bit-for-bit. Compute the realized L2 displacement from that same
invocation's post-action minus pre-action float32 values promoted to
binary64, using the same ascending-channel `sqrt(math.fsum(d_i*d_i))`
reduction. Each non-self action's realized L2 must be within `1e-5 * m` of
`m`; otherwise stop and preserve the incomplete attempt. No model output may be
used to alter a dose or omit a case. Software-defined float32 bit decoding and
round-to-nearest-even conversion, finite/subnormal checks, and synthetic
boundary tests must be independently reviewed before execution. Record
expected and actual before/after tensor bits, unchanged coordinates, requested
and achieved norms, source/donor identities and all output logits. A natural
donor replacement at position 0 is expected to be zero by causal masking; it
is a zero-dose architectural check, not the matched-dose position control.
Equal input L2 is not equal functional or downstream perturbation. Position 0
can influence later computation, and a cyclic channel permutation need not be
an inert perturbation; report their effects rather than treating either as a
guaranteed null. For each matched low/high pair in the same wording, check
natural position-0 donor/target equality directly from the two saved passive
no-op captures; failure stops the attempt.

For every prompt, run an unhooked clean and passive no-op before family dose
selection, then the four actions, followed by an unhooked clean-after. Exact
full-logit clean/no-op/self/after equality and exact no-op/self landing are
required. Enumerate pairs in the order given above, nouns as `war, reign`,
conditions as `forward, reverse, neutral`, and target years as `low, high`.
The zero-based pair/noun index is `2*pair_index + noun_index`; the other two
indices in the rotation are the positions in those condition and year lists.
If a gate passes, the plan has 336 forwards; if both gates fail, it has 96.
The retained final-position float32 logits occupy 67,545,408 bytes for 336
forwards. At sequence length at most 32, two full block-output snapshots for
each of five hooked actions add at most 47,185,920 bytes. Enforce these raw
limits and a separate bounded metadata/temporary policy; neither bounds total
process RSS. Persist returned bounded bytes and failure records before later
validation, including partial attempts. No automatic rerun is presumed.

## Predeclared effects and falsification

For each action and target, `E=M(patched)-M(clean)`. For the low-year target
define donorward `s_low=-E`; for the high-year target define `s_high=E`.
`S_c = (s_low+s_high)/2` is in the **forward-style** donor direction for
wording `c`. This definition is also used for reverse: a genuine reversal
would produce a negative `S_reverse`. Report both directions, both `E`s,
all control effects, full logits and every individual family; use fixed-order
binary64 arithmetic and `math.fsum` for means. Do not divide by the clean
contrast or normalize by an uncertain intervention effect.

Conditional on the exploratory gate, its prespecified donorward pattern
requires, in at least six of eight pair/noun observations: **both** low/high
directions in **each** of forward and neutral have positive donorward `s`; the primary
`S_forward - S_neutral` exceeds 0.1; and the primary forward `S` exceeds
both same-dose-control forward `S` values. Fewer successes do not support
this pattern; retain null or reversed results. This is a descriptive decision
on eight related public observations, not a population hypothesis test.

Conditional on the comparator-pattern gate, the separate comparator-like
patch prediction requires both forward directions to have positive donorward
`s` and both reverse directions negative in at least six of eight observations.
A control can be as effective as the primary; that weakens a site-specific
reading even if the directional pattern appears. Neither criterion identifies
what information was transferred or excludes contextual token associations.

Freeze and publish the protocol, fixed cases, exact executable source and
pre-run record before new pretrained forwards. Review the arithmetic, case
binding, failure retention and synthetic landing independently. Preserve a
direct-process exit record and independently replay saved bytes and all
controls before publishing any result. Public inputs are exploratory and may
have appeared in pretraining; exposure and held-out checks do not establish
otherwise.
