# Prospective DistilGPT-2 agreement direction study, v1

**Status:** This prospective document fixes the design upon publication. No new
evaluation cases have been assembled and no model has been run for this study.
The design and site
were informed by public discovery, so this is an exploratory public follow-up,
not a blind or preregistered held-out confirmation. Publish the reviewed
protocol before constructing the evaluation rows. Publish the derived direction
bytes and their provenance before constructing those rows as well. The
evaluation inputs, execution source, exposure reservation and custody checks
will be separately published and reviewed before any model forward.

## Question and existing evidence

Does a fixed direction extracted from the exposed layer-3 attention outputs
contribute to the singular–plural agreement contrast on new public, matched
contexts, and do matched donor and control interventions behave as predicted?
The study measures sensitivity of specific interventions, not whether an
abstract grammatical-number feature was uniquely transferred or a circuit was
identified.

The source is the public DistilGPT-2 natural-agreement discovery, source commit
`def07cbb168408a12c76ba30f196af45149e111d`,
[`results/distilgpt2-natural-agreement-v1-20261028/README.md`](../results/distilgpt2-natural-agreement-v1-20261028/README.md).
Its evidence archive SHA-256 is
`9ea7ebf83f0580335cbf35b65b2277d8990090adff5b3eb8d0b3b68409cd3a69`.
The original 16 public inputs in four families selected layer 3 attention at the
last position; D was 3.095347 and 10 of 32 overlapping opposite-number donor
cells reversed, involving six target prompts. Opposite-number donors also
weakened correct margins without necessarily reversing them. These observations
do not distinguish number transfer from attenuation, surface morphology, the
attractor or a context-independent output bias. The historical CPU backend
environment variables `MKL_CBWR` and `ATEN_CPU_CAPABILITY` were not recorded;
derive vectors from the saved bits, without claiming an exact historical model
replay. An independent post-hoc saved-byte analysis finds the eight
singular-minus-plural capture differences aligned with their mean; this is
development evidence on the same exposed families, not evaluation performance.

Layer numbering and token positions are zero-based. The original selected
output is `transformer.h.3.attn`, output tuple element 0, batch 0 and final
input position. In the original examples that position is the attractor, not
the subject. The primary present margin is `m = z[318] - z[389]` for ` is`
versus ` are`. The secondary past margin is `p = z[373] - z[547]` for ` was`
versus ` were`. Tokenizer-only verification also fixed two descriptive
non-number logit differences, `z[262]-z[257]` (` the` versus ` a`) and
`z[287]-z[319]` (` in` versus ` on`); these are imperfect diagnostics and not
standalone language-behavior tests. Save the entire 50,257-logit vector and
report broader vocabulary changes, not only these four differences.

## Fixed direction and control construction

Only the 16 saved clean captures in the pinned public discovery archive train
the direction; no future evaluation or private case contributes. Verify the
archive hash, all capture IDs, selectors, shapes and per-payload SHA-256s.
Decode finite little-endian IEEE-754 binary32 bits directly into binary64,
including subnormals and signed zero; do not depend on an ambient float32 cast.
For the exact ordered pairs, use singular then plural: q1 cat `c00,c01`, dog
`c02,c03`; q2 doctor `c06,c07`, teacher `c04,c05`; q3 artist `c08,c09`,
writer `c10,c11`; q4 farmer `c12,c13`, sailor `c14,c15`. In that order let
`d_k = x_singular - x_plural` coordinatewise. Define `mean_d[j] =
math.fsum(d_k[j] for k in order)/8` and `u = mean_d/norm(mean_d)`, where
`norm(v) = math.sqrt(math.fsum(x*x for x in v))`. A positive u displacement
thus points toward the exposed singular captures.

Construct a nonrandom number-by-lemma contrast control `v1`: average the same eight
differences with signs `+,-,+,-,+,-,+,-` in that order, subtract its projection
on u, and normalize. This contrasts number-difference vectors between the two
lemmas; it is not a direct same-number lexical direction or a direction from
which all number-related information has been removed. Define dot products as `math.fsum(a[j]*b[j] for j in
0..767)`. Construct a fixed pseudorandom control `v2` from, for each `j`, the
first eight bytes of `SHA256(b"mlg-agreement-direction-control-v1" + b"\x00"
+ j.to_bytes(2,"big"))`, interpreted as unsigned big-endian integer `r_j`;
set `r_j/2**63 - 1.0` in binary64. Remove its projection on u, then on v1,
and normalize. Record the seed, every input capture ID and hash, all
intermediate norms, donor projections and pairwise direction dot products.
Abort this construction if an input or norm is nonfinite or a normalization
norm is at most `2**-40`; never pick another seed using outcomes. The three
768-element vectors are persisted as little-endian binary64, with individually
published SHA-256s. Unit norms and pairwise orthogonality must be within 1e-12.
All scalar operations use the stated binary64 order and multiplication for
squares; no blanket claim is made about correctly rounded library `sqrt`.
Publish the deriving source, input manifest, vector bytes, and hashes before
evaluation-case assembly. `v1` and `v2` are two fixed descriptive controls,
not a randomization distribution, and orthogonality does not match semantic
direction, fractional change or downstream functional dose.

## Prospective cases and exposure boundaries

Construct eight four-target families as four base frames crossed with singular
and plural attractors. Within each family, variants 0–3 are regular subject
singular, regular plural, irregular subject singular, irregular plural. Thus
each base frame has the same two subject lemmas under both attractor numbers.
There are 32 target inputs and eight additional no-explicit-subject context
inputs. A family uses a single fixed prefix `The <subject> <relation> the
<attractor>`, with exactly one token for each subject form and for each
attractor form, and exactly five input tokens; the subject is position 1 and
the attractor is the final position. The context-only input is the corresponding
`<Capitalized relation> the <attractor>` fragment; it is not asserted to be a
semantically neutral or subject-free natural sentence. Verify every complete
text with the pinned tokenizer. For every target and context prefix and each
of the four answer strings require `encode(prefix + answer) == prefix_ids +
[answer_id]`, not merely separate one-token encodings.

Select the first four tokenizer-compatible, distinct subject pairs from each
of these fixed ordered pools, without model scores: regular `(chef,chefs)`,
`(baker,bakers)`, `(nurse,nurses)`, `(pilot,pilots)`, `(guard,guards)`,
`(worker,workers)`, `(singer,singers)`, `(dancer,dancers)`,
`(captain,captains)`, `(student,students)`; irregular `(man,men)`,
`(woman,women)`, `(child,children)`, `(person,people)`, `(mouse,mice)`,
`(goose,geese)`, `(foot,feet)`, `(tooth,teeth)`, `(wolf,wolves)`,
`(knife,knives)`. A pair is tokenizer-compatible when each space-prefixed form
is one token and the two forms have distinct IDs. Pair the selected regular and
irregular items by selection order. The resulting eight singular/plural subject
pairs form four regular/irregular pairings; they do not separate morphology
from lexical identity. Fix the four relations, in order, to
`near`, `beside`, `behind`, `outside`. Select the first four compatible,
distinct attractor pairs from `(fence,fences)`, `(tree,trees)`,
`(window,windows)`, `(door,doors)`, `(bag,bags)`, `(basket,baskets)`,
`(room,rooms)`, `(table,tables)`, `(car,cars)`, `(box,boxes)`, and pair them
with the relations in order. Require all selected surface forms and token IDs
to be distinct across subject and attractor pools; skip an incompatible or
already-selected pair deterministically. Require all four subjects within a
family to have identical suffix token IDs. If four pairs per pool cannot be
selected, a complete text/continuation fails tokenization, or any family cannot
meet this structure, stop and publish an incomplete preparation; do not choose
cases based on observed model behavior. Record all eligibility decisions.

Before any forward, freeze a public case file binding all 40 ordered full token
rows, their text, target/number, frame, relation and lexical identities to the
then-current independently reviewed exposure snapshot. Reject equality and
complete-row prefix collisions in either direction with every other proposed
row and with all non-own known public exposure rows. Include all target,
donor and context rows in a later independently reviewed planned reservation;
recheck the latest reviewed inventory at launch. A natural-agreement protected
split exists. Give its custodian the immutable complete 40-row bundle and hash
for an exact and bidirectional complete-row-prefix comparison; record only the
scoped outcome, never the protected rows or nonce. A previous natural held-out
owner failed without a scientific result after a possible-forward gate; its
private split is not this evaluation and is not retried or redrawn here. Ask
the separate injection custodian whether these fixed attention sites conflict
with protected custody and whether it holds protected rows, without revealing
or using the concealed site. If a required comparison is unavailable or fails,
stop before execution. These checks address known histories, not unknown
pretraining or all previous exposures; these public examples are not blind.

## Per-family intervention and fixed dose

For each family, do the clean and no-op passes for all four targets before
computing doses or making any intervention. Let `x_i` be each target's selected
L3-final output from its bit-verified no-op. Its three donors are the
different-lemma same-number row, same-lemma opposite-number row, and
different-lemma opposite-number row within that family. For each target,
compute all three finite donor distances `n=norm(donor-x)` in the order above.
For each same-lemma singular/plural pair, set a common nominal dose `q` to
one half of the minimum of its six donor distances. This is derived from
clean activations only and is shared by both grammatical classes in that pair.
If `q <= 2**-19`, set it to zero and execute the planned arms as bit-identical
no-ops; retain them and the original denominator. Nonfinite captures or
distances make the attempt incomplete. Record all six distances and q.

For nonzero q, each donor arm replaces the selected 768-vector by
`round_f32(x + (q/n)*(donor-x))`, with the operations evaluated in that
order in binary64 and each resulting coordinate rounded independently to
IEEE binary32 nearest-even. The three arms are within the donor segment and
have the same requested input-space L2 per target; record actual delivery and
the original `n`. q=0 copies the original bits, even when n=0. Deliver the
six direction arms `x + q*u`, `x - q*u`, `x + q*v1`, `x - q*v1`,
`x + q*v2`, `x - q*v2`, each rounded from x directly, not from another arm.

Add four site/position comparators: `+/- q*u` on L1 attention final and on
L3 attention subject position. Their source slices come from separate clean
no-op captures. q is still the L3-final donor-based dose, and the same
binary64 and binary32 delivery rule applies. Report local input norms and q
fractions; different sites and positions have different distributions and
routes, so equal input L2 is not equal functional dose or a null assumption.

The separate projection-centering test uses each same-lemma pair's L3-final
no-op vectors. Compute `gap = dot(x_s,u) - dot(x_p,u)` and `t_s = gap/2`,
`t_p = -t_s`. For each target, compare `x - t*u` with `x - t*v1` and
`x - t*v2`, independently rounded from x. This aims to equalize the pair's
u coordinates, rather than zeroing them; record their achieved residual
coordinate gap. If `abs(t) <= 2**-19`, use an exact no-op for all three arms,
retain both targets and the full 16-pair denominator, and flag the zero-dose
pair. Report per-arm changed elements, changed bits, requested and achieved
L2, actual selected bits, source norm and removed fraction. For every
nonzero requested dose, require achieved input L2 within `1e-5` relative;
otherwise retain available payloads and mark the run incomplete. A zero-dose
delivery must preserve the source bit pattern exactly.

Context-only interventions use their own L3-final no-op vector and a q equal
to `math.fsum([q_regular,q_irregular])/2` from the paired family. Run the six
`+/-` direction arms. Context-only q=0 means exact no-op arms; report it and
leave per-unit slopes undefined. These fragments provide a diagnostic of
shared token bias, not a calibrated no-number null.

## Predictions, negative outcomes and reporting

For each vector direction a, define odd response `O_a =
(m(+qa)-m(-qa))/2` and even response `E_a =
(m(+qa)+m(-qa))/2-m(clean)`, with the written binary64 order. Also report
each plus and minus change and, when q is nonzero, `O_a/q`; undefined per-unit
values remain explicitly undefined and are not replaced by zeros. Make the
same calculations for p and the two descriptive non-number margins.

The prospective signed-response prediction is that mean O_u is at least 0.05
logit in both target-number classes and, in each class, strictly exceeds the
mean of the absolute individual O values for each of the two controls. In at
least six of eight target families, mean O_u must also be at least 0.05 logit
and exceed the mean absolute individual response of each control within that
family. The 0.05 floor is a fixed exploratory threshold, not a validated
practical effect size or a significance
test; using mean absolute control values prevents opposite signs from
cancelling. Class means and family means retain all targets,
including zero-dose no-ops; the eight families share four base frames and are
not independent samples. Failure of any clause is an unsupported prediction,
not a rationale to tune the direction, site, cases, radius or thresholds.
Give all family, morphology, attractor-number, class and context-only values,
including opposite and near-zero signs. A positive response can be a general
decoder/token bias or asymmetric disruption; compare context-only responses
and full-vocabulary changes without claiming this removes that alternative.

For each same-lemma pair, let `G_clean=m_s-m_p`, and for each centering arm
`loss_a=G_clean-G_a`, with target order singular then plural. The separate
projection-contribution prediction requires mean `loss_u` at least 0.10 logit
and strictly greater than the mean absolute per-pair loss of each control.
In at least 12 of 16 pairs, `loss_u` must be at least 0.05 logit and strictly
greater than each absolute control loss. These fixed thresholds are descriptive,
not statistical tests. All 16 pairs remain in the
denominator, including zero-dose pairs. Report the individual pairs, achieved
coordinate gaps and actual doses. Even a selective loss can arise from a
scalar correlated with surface morphology or from a global output bias; it
supports only a contribution of this intervention to this matched contrast.

For donor arms, use target-oriented weakening `W=sign(number)*(m_clean-m_arm)`,
where singular is +1 and plural is -1. Report same- versus opposite-number
same-lemma and different-lemma shifts separately, the two differences from
the different-lemma same-number donor, target-number classes, margin
reversals, lexical dependencies and absolute raw changes. A larger weakening
under opposite donors is compatible with generic attenuation. The opposite
same-lemma versus different-lemma same-number contrast changes donor lemma as
well as number; the opposite versus same-number different-lemma contrast keeps
the donor-lemma relation fixed. Report clean
accuracy and all target cases without filtering on correctness, plus original
and altered top-20 tokens, target ranks and `KL(p_clean || p_arm)` from the
full vocabulary. No claim of full selectivity follows from orthogonal or
unrelated-logit controls; see Canby et al.,
https://arxiv.org/abs/2408.15510, for the distinction between intervention
completeness and selectivity.

All class, family, pair, context and aggregate means use `math.fsum` in the
fixed case/pair order divided by their declared full count. For a vector of
finite decoded logits z, let `a=max(z)`, `L=log(math.fsum(exp(z_j-a) for j
in increasing token-ID order))`, and `logp_j=(z_j-a)-L`, in that operation
order. Compute `KL(clean || arm)` as `math.fsum(exp(logp_clean_j) *
(logp_clean_j-logp_arm_j) for j in increasing token-ID order)`, treating
terms with underflowed zero probability as zero. Top-20 sorts by decreasing
logit and then increasing token ID; rank is one plus the number of logits
strictly greater than the target plus the number of equal logits with lower
token ID. Reject a missing, nonfinite or invalid vector as incomplete after
persisting any returned bytes. These deterministic rules do not change the
interpretation or turn the fixed cutoffs into sampling-based inference.

## Fixed execution, failures and evidence

Per target, do genuinely unhooked clean and clean-repeat, then separate bitwise
no-op captures at L3-final, L3-subject and L1-final. All clean/no-op full
final-logit vectors must match exactly. After all four cases' five initial
passes, run the 16 named intervention arms and one unhooked clean-after per
target. The fixed base arm order is `same_diff`, `opp_same`, `opp_diff`,
`u_plus`, `u_minus`, `v1_plus`, `v1_minus`, `v2_plus`, `v2_minus`,
`l1_u_plus`, `l1_u_minus`, `subject_u_plus`, `subject_u_minus`,
`center_u`, `center_v1`, `center_v2`. Rotate left by
`(zero_based_family + 4*variant) % 16`. Each arm occupies each slot twice
overall and once per target-number class; order remains deterministic. Each
context-only input receives unhooked clean, repeat, L3-final no-op, six
`+/-` direction arms rotated by `family % 6`, and unhooked clean-after.
Fixed total: 32*22 + 8*10 = 784 model forwards. No model input or intervention
is selected using its logits. Failure never triggers an automatic retry.

Use a separately reviewed, source-pinned direct CPU owner for checkpoint
`distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`,
with weights SHA-256 `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`
and tokenizer SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
Run offline, CPU float32, eval, eager attention, cache off, with CUDA hidden;
fix and record dependency versions, runtime numeric settings and backend
environment before model import and verify them at each forward. Record an
observed external process exit and all counters. State model/backend and
custodian statements as recorded observations, not authenticated attestations.

Retain each final-position full-vocabulary float32 logit vector, and for each
hooked action the whole selected attention output before and after plus exact
site/position metadata. Verify no-op input bits, selected landing bits,
untouched coordinates, clean-repeat/after logit equality, pre-action inputs,
hook removal, state/buffer hashes and complete action order. Check finiteness
of the complete returned logits during execution; unsaved sequence positions
cannot be independently checked later. Persist bounded returned data before
rejecting a control, numerical or nonfinite failure, and preserve partial
journal/diagnostic files. The retained raw-payload limit is 512 MiB per one
attempt; reserve and check before every forward, use a separate 1 GiB maximum
for this attempt's simultaneous physical evidence copies, require 2 GiB free
disk before launch, and report actual payload/copy sizes. These limits are
host-disk accounting, not a process-RSS or VRAM cap. A source, custody, dose,
control, missing-payload, interrupted or nonzero-exit failure makes the
attempt incomplete, not a negative scientific result.

Publish all 40 cases, input and direction manifests, attempt records, source,
receipt, raw logits/witnesses and per-case/family summaries. Independently
reconstruct raw doses, delivery and controls, direction/erasure and donor
statistics, all vocabulary diagnostics and the actual public source/receipt
links before calling the result scientifically audited. A saved-byte audit
does not independently prove the model execution, custody chronology or
absence of unknown exposure. Preserve every unsupported or heterogeneous
result and the conditions above.
