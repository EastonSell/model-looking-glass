# DistilBERT agreement: matched-magnitude subject interventions v1

Status: **analysis preregistered before new case assembly.**

This protocol is fixed before constructing the new public evaluation cases.
It follows the earlier [balanced CPU evaluation](../results/bert-agreement-cpu-evaluation-v1-20261029/README.md),
where a block-0 subject replacement had a larger donor-number effect than a
block-2 subject replacement in all eight families. Those unscaled replacement
vectors could have different lengths. Here the question is whether the earlier
site retains a larger *signed donor-number contrast at a matched absolute
replacement magnitude*. This is a comparison of particular interventions, not
proof that a grammatical computation is uniquely localized at either block.
As an exposed-data feasibility check before this freeze, the same six-distance
minimum computed from the 32 prior evaluation targets ranged from 7.2189 to
10.0816 (median 8.3699). Its smallest ratio to base-vector norm was 0.3835 at
block 2 and 0.4193 at block 0; the smallest interpolation fraction among the
six directions was 0.3753. Those magnitudes can be large relative to the base
state. These old records informed feasibility, not validation of new cases.

## Model, sites and new cases

Use `distilbert/distilbert-base-uncased`, revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, with the same five checkpoint and
tokenizer hashes, float32/eager model, evaluation mode, CPU runtime, fixed seeds,
one intra-op and one inter-op thread, and CPU custody contract as the frozen
[BERT agreement v1 protocol](BERT_AGREEMENT_V1.md). Use zero-based encoder
blocks 0 and 2, tuple element 0, invocation 0, batch index 0, and the subject
token index 2. The full hidden state is `[1,10,768]`; the selected vector is
768 float32 values. The exact actual runtime, model and source must be checked
and recorded before and after execution. Any GPU version requires a separately
reviewed workload-specific profile and ownership path; the earlier calibration
does not authorize it.

After this protocol is published and independently reviewed, assemble eight
new public noun-pair families without consulting model outputs. Use the same
ten-token template `The {subject} beside the {distractor} [MASK] ready.` and
cross singular/plural subject with singular/plural distractor. Each family has
four target rows, with subject lemma A. The lexical-control donor subject B is
the next family's A subject cyclically. At fixed distractor and template, each
target has A/same, A/opposite, B/same and B/opposite donor rows, as in the
previous study. The complete inventory contains 32 targets and 64 distinct
target/donor input rows.

Choose 16 distinct common human-role singular lemmas, none used as a subject
or distractor in the cited development and evaluation inventory or its v5
exposure successor. Require each singular and its ordinary plural to be one
distinct WordPiece in the pinned tokenizer. Reject tokenization changes or
duplicate exact full input rows; do not replace a failed case after any new
model outcome is observed. Record the selection procedure, ordered lemmas,
texts, token IDs, masks, and their comparison with the prior inventory in a
separate public manifest. Publish and independently review that manifest and
the exact runner source before any forward on these new rows. The fixtures are
public and nonblind; prior exposure outside cited records and pretraining is
unknown. They are a convenience sample, not a population sample.

## Common displacement and controls

For each target, capture its unpatched subject vector `h_s` and the three
nonidentity donor vectors `h_s,c` at each site `s` in `{0,2}`, where `c` is
`A_opposite`, `B_same` or `B_opposite`. At each site compute the Euclidean
distance `d_s,c = ||h_s,c - h_s||_2` from the stored float32 bits decoded into
binary64. Define the one target-specific radius `M` as the minimum of all six
distances. For each site and donor condition replace the target vector by
`h_s + (M / d_s,c) * (h_s,c - h_s)`, rounded componentwise to float32. No
requested interpolation fraction exceeds 1; rounding can change the delivered
norm slightly. The minimum couples the chosen dose to all
six directions and can make it small; report all six distances and scale
fractions. It is not a dose-response curve.

Compute each component difference as `math.fsum((donor_i, -base_i))` in index
order and each norm as `math.sqrt(math.fsum(d_i * d_i for i in order))`.
Compute each interpolated component as
`math.fsum((base_i, (M / d_s,c) * d_i))`, then convert each binary64 result to
IEEE float32 bits by integer-defined round-to-nearest, ties-to-even. Preserve
those bits directly in little-endian order, without a float32 conversion in
between. The caller checks denormal preservation and representative binary64
rounding operations before and after the run; these calling-thread checks do
not attest every native thread. Reject
nonfinite inputs, distances, fractions,
components, outputs or effects. If any distance or `M` is below `1e-4`, mark
the attempt incomplete: do not drop that case or choose a larger dose. This is
a numerical guard, not a claim that all accepted doses are scientifically
large. Recompute each actual displacement norm from the captured post-action
float32 bits and the pre-action bits. Require its difference from `M` to be at
most `max(1e-6, 1e-5 * M)`; any failure makes the attempt incomplete.
Record `M`, every raw distance, fraction, pre/post bit pattern, realized norm,
absolute and relative norm error, base-vector norm and realized/base norm ratio
(or `null` for zero base norm). At the minimum accepted radius, the absolute
tolerance permits up to 1% relative error; do not describe it as a uniform
`1e-5` relative match.
The achieved absolute norms are matched within this explicit rounding
tolerance. Equal Euclidean lengths at different layers do not imply equal
relative displacement, representational change, downstream effect, or
plausibility. Interpolation and patching may create unnatural hidden states.

The A/same donor is the target itself. Test it separately as a zero-dose
self-patch with exact float32 bit landing and exact full-mask-logit equality;
it is not a matched-dose donor intervention. For each target/site also patch
one fixed artificial sign direction at radius `M`. Construct 768 signs by
concatenating SHA-256 digests for counters 0, 1 and 2 of
`b'bert-matched-v1\0' + row_id_ascii + b'\0' + site_ascii + b'\0' + counter.to_bytes(4, 'big')`,
reading each digest byte from most- to least-significant bit and mapping 1 to
`+1`, 0 to `-1`. Here `site_ascii` is `block0` or `block2`. Compute
`u = M / math.sqrt(768.0)` and each component as
`math.fsum((base_i, u if sign_i == 1 else -u))`, using the same float32
rounding and realized-norm rule. This is a reproducible artificial-direction
control, not a randomization null or a test that natural perturbations are
generally harmless.

Before the first patch, capture all eight family rows in manifest order at
block 0 and then at block 2. Process the four targets in the fixed cell order;
for each target use sites 0 then 2, with self, A/opposite, B/same, B/opposite
and artificial-direction operations in that order. Compare
the complete unpatched mask-logit bits for each row across site captures.
For each target and patch, verify the observed pre-action vector equals that
target's capture and the post-action bits equal the submitted vector. Require
each self-patch and a final clean target forward to reproduce the authoritative
unpatched full-mask-logit bits; require no remaining hooks and matching
registered parameter/buffer identities, storages and values after execution.
Save the actual model input IDs and attention masks, donor identity, ordered
site/action metadata, raw `is` and `are` float32 bits, vector evidence and
full-mask-logit digests. Do not claim unrelated capabilities are preserved
during an intervention. Full mask logits are not recoverable from digests.

## Preregistered analysis

For each target and intervention let `L = logit(is) - logit(are)`, with the
two stored float32 bit patterns decoded independently to binary64, and let
`r = +1` for a singular target and `-1` for a plural target. Compute margins
with `math.fsum((is_logit, -are_logit))`. The **primary matched-dose** effect at
site `s` is

`E_s = -r * math.fsum((L_s(B_opposite), -L_s(B_same)))`.

Both B interventions have the same target-specific radius at both sites. The
different-lemma, same-number donor is a lexical control, not an assumed zero
effect. Within each family average all four target effects in fixed order
`(singular,singular)`, `(singular,plural)`, `(plural,singular)`,
`(plural,plural)` with `math.fsum(values)/4`; average eight family means in
manifest order with `math.fsum(values)/8`. Compute the primary paired site
difference as `math.fsum((E_0, -E_2))` per target and apply those same family
and across-family reductions. Record both the mean of paired differences and
the difference of site means; the former is the decision statistic.

The preregistered early-site persistence criterion passes only if the block-0
mean `E_0 > 0.1`, the mean paired difference `E_0 - E_2 > 0.1`, and at least
six of eight family paired differences are positive. Otherwise report that
the criterion was not met; a negative result does not establish equal sites.
The 0.1-logit thresholds are fixed descriptive cutoffs, not significance
tests. Include all 32 cases, including baseline errors, without post-hoc
filtering or layer, example, dose or threshold selection. Report baseline
accuracy and mean margins by the four subject/distractor-number cells, all
per-case and family effects, distances and realized doses.

Secondary descriptive quantities are the signed A/opposite versus A/same
identity contrast and the previous two-donor average. The A/same side has zero
dose, so neither secondary contrast is a fully matched-dose 2x2 effect. Report
same/opposite raw margins and site effects, opposite-donor sign reversals,
the artificial-direction shifts and per-cell summaries without inference
from dependent cases. The eight cyclic donor families are not independent
population samples. A positive B contrast may still reflect lexical/surface
changes or selective erasure; no unique feature or cross-model generalization
follows from it.

For each artificial direction report raw `L_artificial - L_clean` and the
oriented shift `-r * (L_artificial - L_clean)`, with the subtraction evaluated
by `math.fsum`. Reduce the oriented shift by the same fixed case, family and
across-family order as the primary effect. This is a descriptive perturbation
comparison; it is not a null distribution or a pass/fail criterion.

## Execution, exposure and evidence

Each family uses 16 unpatched source captures, 24 scaled donor patches, eight
self-patches, eight artificial-direction patches and four final clean target
forwards: 60 sequential forwards, or 480 across the eight families. Any
missing case, source mismatch, failed control, invalid norm or uncertain
cleanup makes the attempt incomplete. Record starts and completions as well as
failures; preserve partial attempts and all input rows that may have been
exposed. A retry uses exactly the frozen protocol/source/manifest, is a
separately recorded repeat, and cannot restore first-exposure status. The first
complete attempt is primary, with prior partial or uncertain attempts
disclosed. Use the reviewed supervised CPU process custody and an append-only
ledger binding the complete selected workload before launch. Bound serialized
journal output at 16 MiB and fail incomplete rather than truncate; this cap
does not bound peak process or transient tensor memory. A file/hash inventory
and independent closed-record audit must precede publication;
source hashes and locally recorded process state are not execution attestation.
