# Prospective agreement subject-dependence follow-up, v1

**Status:** prospective draft. The fixed source direction and the output-slope
reference below were derived from already exposed data during protocol
development. No new evaluation rows have been assembled or forwarded for this
study. Publish the independently reviewed protocol, reference vector, and
derivation before assembling the evaluation cases. The cases, exposure
reservation, custody checks, owner and prelaunch record require separate
review and publication before a forward.

## Question, development evidence and limits

The public [agreement-direction study](../results/distilgpt2-agreement-direction-study-v1/RESULTS.md)
found large signed responses of `is − are` to a fixed L3-final attention-output
direction `u` in both subject-bearing sentences and eight fragments without an
explicit subject. Those fragments retained number-marked attractors. The
direction and site were selected on earlier exposed discovery data; the study
and this follow-up are exploratory, public and nonblind. A model-free
development comparison of the earlier saved outputs found larger `u` odd
response for the singular member of all 16 paired target comparisons, with
family-dependent doses. This motivated, but does not evaluate, the new test.
The separate [post-hoc context analysis](../results/distilgpt2-agreement-direction-study-v1/posthoc-context/README.md)
also found broad vocabulary changes and a common component across prior
fragments; it is development evidence on those same exposed rows.

The question is whether the response to the same signed, fixed-dose
intervention differs between matched singular and plural subject forms across
new fixed frames and attractor conditions, and how much response also occurs
without an explicit subject. This is a test of conditional sensitivity, not a
proof that an abstract grammatical-number feature has been transferred or
that a unique circuit has been isolated. A subject-associated response can
arise from spelling, morphology, lexical identity, prompt-dependent gain or
downstream nonlinearity. A null response difference does not rule out a
subject-number representation.

Use the existing DistilGPT-2 checkpoint and the exact published `u`, `v1`,
and `v2` little-endian binary64 direction files from
`results/distilgpt2-agreement-direction-training-v1/`. Their SHA-256s are
respectively `d1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0`,
`4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb`, and
`4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02`.
`v1` is a number-by-lemma modulation control and `v2` a fixed pseudorandom
orthogonal control, not a randomization distribution or guaranteed
non-number direction. Do not refit a direction, site, dose, case pool or
threshold using the new outputs.

## Fixed output-space reference

As an analysis-only concrete additive-bias reference, use the eight prior
public context fragments in case-ID order. For vocabulary component `j` and
prior case `i`, decode its saved `u_plus` and `u_minus` last-position f32
logits into binary64 and use its recorded positive nominal dose `q_i`. Define
`b[j] = math.fsum((z_plus_i[j] - z_minus_i[j]) / (2.0*q_i) for i in fixed order) / 8.0`.
The journal SHA-256 is
`7c8be25d480d986b087052685720ba7d2813c53d27509dcc81e67f9e2049eca6`;
the prior science-report SHA-256 is
`8b816905a3009e212ab138c62cbb24f1dd06d53079e228c283bb5d8f6ae1842f`.
The resulting 50,257-element little-endian f64 vector is SHA-256
`bdb31d5b74b6cff17ed2a683415beab22e36b101220257d774909be6f9f804bd`;
its manifest is SHA-256
`69c7bdadb365c83bd4a591da405cc5e1dccd686c204a68f9b03264c604e82ed5`.
The mean `is − are` slope is 0.25118876581346644 per unit q and the
`was − were` slope is 0.23164523187095867. The 16 source logits, their
individual hashes and doses are listed in the manifest. The builder was
independently reconstructed before publication. Its original contexts already
contained numbered attractors, and their doses varied by family.

For a new clean f32 logit vector, decode to f64 and form each component
`z_ref_plus[j] = z_clean[j] + (2.0*b[j])` and
`z_ref_minus[j] = z_clean[j] - (2.0*b[j])`, in that order, without rounding
the reference back to f32 or forwarding it through the model. Its theoretical
odd `is − are` margin is `2.0*(b[318]-b[389])`, approximately 0.502378.
An identical added vector gives theoretical pair interaction zero; finite f64
addition/subtraction can leave a small base-dependent numerical residual, to
be recorded. This reference assumes a fixed first-order response scaled from
the earlier finite doses. It is a hypothetical output vector, not a physical
activation intervention, population null, or unique decomposition of model
effects. A poor fit may reflect nonlinear dose response or context changes.

## Fixed public case construction

After protocol/reference publication, select two regular and two irregular
singular/plural subject pairs using the following fixed order, tokenizer only:
regular `(horse,horses)`, `(bird,birds)`, `(duck,ducks)`, `(cow,cows)`,
`(pig,pigs)`, `(lion,lions)`, `(bear,bears)`, `(frog,frogs)`; irregular
`(mouse,mice)`, `(goose,geese)`, `(wolf,wolves)`, `(ox,oxen)`,
`(calf,calves)`, `(louse,lice)`. The pools use animal nouns to reduce, not
eliminate, the semantic-category confound between regular and irregular
forms. A pair is eligible only if each leading-space form encodes as one
distinct token and neither form/token is among the exposed discovery or prior
evaluation subject forms. Check those exact published case records. Select the first two eligible
pairs per pool, record all eligibility decisions, and call them R1, R2, I1,
I2. If the pool is exhausted, stop with an incomplete preparation.

The four frame prefixes, in order, are `At dawn,`, `At dusk,`, `At noon,`,
and `In summer,`. Assign subject pairs `(R1,I1)`, `(R1,I2)`, `(R2,I1)`,
and `(R2,I2)` to those frames. Thus each frame contains one regular and one
irregular pair and each subject pair occurs in two frames. The four ordered
singular/plural attractor pairs are selected as the first four eligible,
distinct pairs from `(bridge,bridges)`, `(road,roads)`, `(ship,ships)`,
`(hill,hills)`, `(garden,gardens)`, `(box,boxes)`, `(car,cars)`,
`(table,tables)`. Each form must have one distinct leading-space token; reject
overlap with selected subject forms/tokens, the exposed discovery or prior
evaluation subject forms/tokens, or the prior evaluation's attractor forms/tokens. Assign the
four selected attractor pairs to the four frames in order. The mass noun is
the fixed text `water`; it is a lexical comparator, not a guaranteed
number-neutral representation.

For each frame, subject pair and subject number, construct four target inputs,
in condition order:

1. no attractor: `<frame> the <subject> nearby`;
2. mass-noun locative: `<frame> the <subject> near the water`;
3. singular count-noun locative: `<frame> the <subject> near the <singular attractor>`;
4. plural count-noun locative: `<frame> the <subject> near the <plural attractor>`.

For the same frame and condition, add exactly one shared fragment without an
explicit subject: `<frame> nearby`, `<frame> near the water`, or `<frame>
near the <attractor>` as applicable. They are fragments, not controlled
subject deletion or neutral sentences. The no-attractor form changes the
terminal token and potentially the position; `water` is a different lexical
and semantic choice from each count noun. The four-frame partial crossing
improves coverage but does not separate morphology from lexical identity.

This gives 64 target rows plus 16 shared fragments = 80 distinct inputs.
Order by frame, condition, regular then irregular subject pair, singular then
plural, then the one shared fragment for that condition. Verify complete
tokenization with the pinned tokenizer, the subject and final-token positions,
and `encode(text + answer) == input_ids + [answer_id]` for ` is`, ` are`,
` was`, ` were`, ` the`, ` a`, ` in`, and ` on` (IDs 318, 389, 373, 547,
262, 257, 287, 319). Both members of every target pair must have equal input
length and identical token IDs outside the one subject position. Within a
frame/subject pair, mass and count-noun conditions must have equal lengths and
identical IDs outside the last noun token. The singular and plural count-noun
fragments must likewise differ only in that last token. Each input has at most
12 tokens. Record exact token positions and lengths; a failure stops the
preparation without a model call.

No selection uses model scores. Record the then-current reviewed public
exposure version, all candidate/target IDs and ordered input digest. Check each
complete proposed input in both complete-row-prefix directions against every
other proposed row, all non-own current known public rows and any separately
published pending inputs. Include the 128-row original natural sampler support
and the public 256-row successor pool in this check, with their exact source
and digest. Ask the natural custodian for a scoped independent comparison of
the frozen 80-row bundle to applicable protected support or splits; do not
inspect or disclose protected membership, rows, nonce or commitments. Ask the
injection custodian whether the fixed L3-final attention-output site has a
direct protected-target conflict and whether protected token rows exist.
Publish the cases and a reviewed planned exposure reservation, then refresh
the relevant inventory and both custodian reports at launch. If a required
comparison is unavailable or fails, stop before execution; do not adapt cases
to a private comparison or retry after an uncertain forward. These checks do
not establish complete historical exposure. The new cases are withheld from
direction/reference development, but their prospective public publication
does not make them blind or secret held-out data.

## Intervention, schedule and evidence

Use the output tuple element 0 of `transformer.h.3.attn`, batch 0 and the
last input-token position. On every case use the same requested q = 2.0
binary64 input-space L2 for `+u`, `−u`, `+v1`, `−v1`, `+v2`, `−v2`.
This development-derived q is below the earlier target-dose range
2.543762–4.402681; it does not establish local linearity or equal functional
dose for these new inputs. For each 768-coordinate f32 source x, evaluate
`x[j] + (signed_q * direction[j])` in the written binary64 order, rounding
each result directly from the original source to IEEE f32 nearest-even.
Decode every source and delivered f32 from its IEEE bits into binary64,
including zeros and subnormals, independent of ambient flush-to-zero behavior.
Record the requested q, source norm, actual L2, changed elements and bit
count, the complete before/after f32 bits, and the signed projected
displacement `math.fsum((after[j]-before[j])*direction[j] for j in 0..767)`.
For every nonzero arm require both achieved L2 and absolute projected
displacement within 1e-5 relative of 2.0, with the correct sign. Any
nonfinite capture, output, source norm or dose, zero delivered displacement,
or failed tolerance makes the one-shot attempt incomplete; preserve bounded
partial evidence and do not retry or discard a case.

Traverse frames, conditions and regular/irregular pairs in their fixed order,
then the shared fragment for each condition. The case file keeps canonical
singular/plural order, but execute the plural case first for a pair if
`(frame_index+frame_index//2+pair_index)%2==1`, otherwise execute singular
first. This balances first-subject order in 16 pairs each, with one of each
order within every frame and condition and four of each order for each of
R1, R2, I1 and I2. For each execution case, perform clean, a
verified no-op at that site, then six direction arms and clean-after. For target cells let
`j=4*frame_index+condition_index` (both zero-based) and rotate left by
`2*(j%3)`. Let `pair_index` be 0 for regular and 1 for irregular. Use the
plus-first base order `u_plus,u_minus,v1_plus,v1_minus,v2_plus,v2_minus` if
`(frame_index+condition_index+pair_index)%2==0`, and the minus-first base
order `u_minus,u_plus,v1_minus,v1_plus,v2_minus,v2_plus` otherwise. Both
members of a subject pair use exactly the same order. For fragments use rotation offsets
`[[0,0,2,2],[2,2,4,4],[2,2,4,4],[4,4,4,4]]` indexed by frame and condition;
use the plus-first base order if `(frame_index+condition_index)%2==0` and
the minus-first order otherwise. Require full saved last-logit equality across
clean/no-op/clean-after and bit-identical no-op whole/selected captures. This
is 9 calls per case, 720 forwards. Each arm occupies each slot 13 or 14 times
overall; each sign is first in 16/32 singular targets, 16/32 plural targets,
16/32 regular targets, 16/32 irregular targets, and 8/16 fragments. This is deterministic, not randomized; record actual
placement counts and check the fixed schedule with a fabricated regression.
Record the entire f32 last-position vocabulary
vector and whole/selected pre/post attention outputs for each hooked call;
check finiteness of all sequence logits before marking a forward complete.
Full-sequence logits need not be saved on success, so later replay cannot
independently verify their finiteness. Record clean and post-run state hashes,
runtime, source, asset, receipt and cooperative child-exit observations.

With 80 inputs of at most 12 tokens, logical successful payloads are at most
`720*50257*4 + 560*(2*12*768*4 + 2*768*4) = 189,468,480` bytes before
content-addressed deduplication. Reserve separately for bounded failed full
sequence returns, the journal, manifest, receipts and output copies. A worker
may cap retained tensors at 512 MiB and journal at 64 MiB within a stated
physical-attempt limit; these are not RAM or process-tree bounds. Require at
least 2 GiB free before the one-shot claim. Use the reviewed rooted gated CPU
supervisor or an equivalently reviewed custody path; save an independently
observed parent exit and retain all incomplete/failure records. Exact source,
physical copy accounting and failure bounds must be frozen and reviewed before
the run. The scientific replay must be independent of the owner and work from
saved bytes; a process receipt does not authenticate execution or drain of
unobserved descendants.

## Measurements and competing predictions

Let the primary raw logit margin be `m = z[318]-z[389]` (`is − are`). Let
`O_a(x) = (m(x,+2a)-m(x,−2a))/2`, and also record
`E_a(x) = (m(x,+2a)+m(x,−2a))/2-m(x,clean)` and each signed change. For
every one of the 32 fixed frame/subject-pair/condition cells, compute
`K_a = O_a(singular)-O_a(plural)` for `a` in `u,v1,v2` at the same nominal
q. Compute the clean geometry from bit-decoded saved L3-final no-op captures:
`G_a = math.fsum(a[j] * (x_singular[j]-x_plural[j]) for j in 0..767)` in
that order; `G` is a descriptive
clean-coordinate association, not evidence that the model uses that feature.
Report all `O`, `E`, `K`, `G`, actual signed projections, doses and raw
margin values. Report within-pair `K_attr_singular-K_no_attractor`,
`K_attr_plural-K_no_attractor`, `K_attr_plural-K_attr_singular`, and
`K_mass-K_no_attractor`, and the corresponding G differences. They change
surface tokens, final positions or semantics and do not isolate a pure
attractor or subject-removal effect.

As a separate exploratory geometry prediction, the mean G_u across all 32
cells is at least 0.5 activation units and exceeds each mean absolute
individual G for v1 and v2; at least three of the four frame means must be
at least 0.5 and exceed their corresponding mean absolute control values.
This fixed floor is not a validated practical effect or a statistical test.
Report all values and signs whether or not this criterion passes. A geometry
association can reflect surface forms and is not a test that the projection
is used by the model.

The fixed additive-output hypothesis predicts the same odd response for
singular and plural forms: theoretical `K_u = 0`, and the prior-derived
constant reference predicts `O_u = 2*(b[318]-b[389])` on every new row.
Two deliberately coarse checks are frozen before any new output. For the
general constant-additive-bias class, mean `abs(K_u)` must be at most 0.10
and at least 24/32 individual paired values at most 0.10. Separately, to fit
this *particular* earlier-fragment-calibrated reference, across all 80 rows
mean absolute error in that odd margin must be at most 0.20 logit and at
least 60/80 errors at most 0.20. A different constant additive bias can pass
the K check while failing the specific reference check; verify that with a
fabricated example. Report both checks and all discrepancies by role and
condition. These tolerances are descriptive, not calibrated tests; failure
does not prove grammatical specificity, and fit does not prove the reference
is the actual mechanism. Only q=2 is run here; the single dose cannot itself
establish local linearity or a dose-response curve.

The competing subject-form sensitivity prediction is that mean K_u across
the 32 cells is at least 0.10 logit and strictly greater than each mean of
the 32 individual absolute K values for v1 and v2. For each frame, average
the eight K_u values in fixed regular/irregular then condition order and
separately average the eight absolute control K values; at least three of
four frames must have mean K_u at least 0.05 and greater than each control
mean. Evaluate every fixed cell regardless of its baseline or sign. The
thresholds are exploratory, not validated effect sizes or statistical tests;
the 32 cells share four frames and four subject pairs and are not independent
samples. Report positive, zero and negative individual K counts. A pure constant additive bias predicts K=0, but a real number
coordinate with a linear downstream readout can also yield K=0. Conversely,
input-dependent gain, including `m(x,t)=beta*(u dot x+t)^2+b`, can yield
K≠0 even when the signal reflects surface morphology or context rather than
grammatical number. Either prediction may fail; they are not exhaustive or
necessarily mutually exclusive at the stated tolerances.

Before interpreting the behavioral criterion, count clean margins with the
correct sign for both subject forms of each of the 32 pairs: singular must be
strictly positive and plural strictly negative, with a tie counted incorrect. The fixed
baseline adequacy criterion is at least 24/32 correct pairs overall and at
least 5/8 within each condition, each frame, and each subject pair (each has
eight comparisons). If it fails, retain every
result and label the behavioral comparison baseline-limited; do not filter
or replace any case. Report accuracy separately by subject, frame, morphology
and condition. Also report the 16 fragment odd responses at the same q;
they contain no explicit subject but are not a matched subject-removal test.

Use fixed order and `math.fsum/n` for every specified mean; never replace an
undefined value with zero. In addition to the present margin, save and report
`was − were` (373−547), `the − a` (262−257), and `in − on` (287−319), plus
the full 50,257 logits, top-20 token IDs and values, ranks of those eight
tokens and stable full-vocabulary `KL(clean || arm)` for every action. Break
rank/top ties by ascending token ID. Compute the same diagnostics for the
hypothetical f64 output reference, and report per-case reference residuals
and actual projected doses. For each case, form the observed f64 odd logit
slope `d[j]=(z_plus[j]-z_minus[j])/(2.0*2.0)` and center each of d and b by
subtracting its `math.fsum(values)/50257` vocabulary mean. Compute their dot
product and L2 norms with ordered `math.fsum`, then divide the dot by the
product of norms. Record both centered norms. If either centered norm is zero, record a null cosine and
the valid/undefined counts; do not replace it with zero. This cosine is
descriptive, invariant to a common logit shift, and treats low-probability
vocabulary entries equally. Check that shift invariance with fabricated data.
Publish per-case, condition, crossed frame/subject and aggregate results,
including negative and near-zero effects, clean baseline failures, all
reference discrepancies, exact prelaunch and exposure identities, raw
witnesses, incomplete evidence and the independent replay. No new forward is
authorized before the separate case, custody, source and launch reviews.
