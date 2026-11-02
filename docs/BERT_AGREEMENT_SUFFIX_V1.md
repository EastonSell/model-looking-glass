# DistilBERT agreement: terminal mask versus a continuation, protocol v1

**Status:** prospective protocol and tokenizer-only inputs. No model forward
on these inputs has occurred at the time of writing. Freeze the protocol,
input manifest and execution source before the first model forward. Record
negative and incomplete attempts as well as positive findings.

## Motivation and fixed cases

In the [preceding study](../results/cross-model-agreement-cpu-v1-20261029/README.md),
the masked model's unique full-vocabulary top was a period in all 32 terminal
cases; the two candidate agreement verbs held a median 0.141% probability.
Its binary `is`/`are` sign was correct in 26/32 cases. The present follow-up
tests whether a single, number-neutral continuation makes this particular
masked-word task more suitable and whether the intervention contrast persists.
It was designed after observing the previous result; these are fresh public
cases, not a blind sample or a guarantee of absence from pretraining.

The fixed [manifest](../experiments/bert_agreement_suffix_cases_v1.json) has
SHA-256 `1da4f8e4773009fedc750480c21597fc60ecaa0c990dcff0af15a5c7101c3dca`.
It defines 32 sentence bases: two surface frames, four human subject lemmas,
two subject numbers and two attractor numbers. The subjects are scout, monk,
poet and guide; the respective frame attractors are visitor and tourist. Each
base has two tokenized variants. The first ends `[MASK] [SEP]`; the second
ends `[MASK] ready . [SEP]`. Their prefix through `[MASK]` is identical. The
whole changed right suffix includes punctuation and the moved separator, and
BERT may use it at every earlier position. Both `is ready` and `are ready`
are grammatical. The two new surface frames use the same broad
subject–prepositional-attractor syntactic construction as the predecessor;
they are not novel syntax. One suffix, two frames and four reused lemmas do
not support claims about broad linguistic generalization or 32 independent
samples. Differences from the predecessor also include the subject nouns, so
the continuation effect is evaluated only within these paired new cases.

The checkpoint is `distilbert/distilbert-base-uncased`, revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`. Its tokenizer assets are pinned
in the manifest; `is`, `are`, `was` and `were` have token IDs 2003, 2024,
2001 and 2020 respectively. All subjects and attractors are single tokens.
The intervened site is the subject token output of zero-indexed transformer
block 3; the scored position is the mask. Seed is 20261029. Use float32 and
eager attention, evaluation/inference mode and batch size one. Fix the device
before the attempt: CPU uses one PyTorch intra-op and inter-op thread and
preserves subnormals on the calling thread; CUDA uses deterministic algorithms,
disables TF32 and cuDNN benchmarking, fixes the cuBLAS workspace configuration
before importing PyTorch, and synchronizes before saving outputs and sampling
memory. Deterministic algorithms are requested in either case. Check that
clean, self and clean-after selected logits are bitwise equal on the selected
device; do not change devices or relax tolerances after an observed forward.
Saved mask-position logits and selected-site witnesses do not independently
attest unsaved positions. Capture donors only from the **same variant**
as the target, with the same frame and attractor number. The opposite donor
uses the target lemma in the opposite number. The same-number control uses the
other lemma in a fixed adjacent pair: scout ↔ monk and poet ↔ guide. The paired
random control uses the same PCG64 direction for both context variants of a
base case, seeded by `20261029 + case.index`.

## Fixed interventions

Capture clean subject vectors for all 64 inputs. For each base case compute
the four raw L2 distances from each context's clean subject vector to its
opposite-number and same-number donor, and set one common pair dose
`q = min(2.0, all four distances)`. If any input or norm is nonfinite or the
common dose is zero, mark the attempt incomplete. For each context apply
displacements of dose `q` toward its opposite donor, same-number donor and
the shared random direction, plus self and clean-after controls. Construct
directions in float64, add to the float32 base, and round once to float32.
Check the realized displacement's norm and vector against the intended
direction to relative tolerance `1e-4`; fail closed on zero, nonfinite or
excessively distorted landings. Equal input L2 does not imply equal effects.

Add one descriptive continued-context arm using the **terminal-context
opposite-donor direction** at the same common `q`. The terminal opposite arm
already uses this direction. Retain both subject before/after witnesses and
report the realized errors for both; adding the same intended direction to
different float32 bases need not produce bit-identical displacements. This
shared direction can be off-manifold in the continued context. Paired random
directions provide a common-direction control. Because the base hidden state
and right context also change, neither this arm nor the context-specific
donor comparison isolates a downstream gain or identifies a mechanism.
Save all four natural donor distances and the cosine between the two contexts'
opposite-donor directions and between their same-number directions; report
undefined cosines explicitly rather than assigning a value.

There are 64 initial clean captures, then five arms per terminal variant and
six per continued variant: **416 forwards** total. The selected full-vocabulary
float32 logits and two 768-float witnesses per forward occupy 53,344,512 bytes
before journals. Cap this study's retained evidence at 128 MiB, counting all
output files; this is a trace-storage limit, not a GPU VRAM limit. Save a
durable start record before each forward, preserve returned payloads on
failure when possible, and classify an uncertain or partial attempt as
incomplete. No automatic retry after an uncertain forward. Save asset,
protocol, manifest, source and runtime identities, direct-child completion
and exit observations, and per-forward donor, dose and landing witnesses.
Use fresh host, output-volume and GPU admission checks before a GPU attempt;
record actual device memory and coordinate with the shared GPU owner. If a
GPU is unavailable, record the CPU execution and do not report GPU timing.

## Prespecified readouts

For every one of the 32 paired bases, retain `P(is)`, `P(are)`, their sum,
the full-vocabulary unique top token and tie state, and binary margin
`m = logit(is) − logit(are)` in both clean variants. Strictly positive `m`
is subject-correct for singular subjects and strictly negative for plural;
ties are incorrect. Report clean binary correct counts and wrong-to-correct,
correct-to-wrong and unchanged transitions. Report paired **absolute**
changes and ratios in candidate-pair probability, including zero/undefined
ratio handling. Describe per-frame and per-lemma results, full-vocabulary
top changes, all individual pairs, and the dependent structure.

The prospective baseline hypothesis is that adding this suffix increases
`P(is)+P(are)`: the fixed descriptive criterion is positive mean paired
absolute change and positive change in at least 24/32 pairs. The continued
variant is labeled baseline-limited when its clean binary sign is correct on
fewer than 24/32 cases. Report whether the continued variant is at least as
accurate as terminal in the same cases. A failed criterion is a result;
publish all intervention effects, every row and all raw values regardless.

For intervention outcomes let donorward signed change be `−Δm` for a
singular target and `+Δm` for a plural target. In **each** context, the fixed
donor hypothesis requires positive mean opposite-minus-same and
opposite-minus-random effects, and positive values against both in at least
24/32 cases. Also report all intervention binary flips and full-vocabulary
top changes, and the parallel `was`/`were` margin as a secondary outcome.
The baseline-correct subsets are prespecified as descriptive: report the
correct cases separately for each variant and the intersection correct in
both. Ties are excluded from these subsets. None replaces the full 32-case
denominator.

Report the difference between the two contexts' opposite-minus-control
effects as a **context-by-intervention recipe** contrast: donor directions
can rotate with the suffix. Separately report the paired difference for the
fixed terminal-derived direction, alongside the shared-random direction;
do not substitute the rotated continued same-number donor for a fixed-vector
control. No direction or threshold is asserted in advance for these interaction
contrasts. Report each context's full-vocabulary distribution change, paired
direction cosines, dose and natural donor distances. These are descriptive
comparisons on a small dependent fixed set, not significance tests.

## Input scope and review

Before execution, compare both tokenized variants against named public BERT
inventories and all known pending fixed bundles in both complete-token-prefix
directions, both with and without a terminal separator. The only permitted
within-proposal prefix relations are the declared terminal/continued pairs
sharing their mask prefix; do not globally relax the comparison. Also report
scoped subject/attractor lemma and frame-surface overlaps. These checks do not
prove disjointness from unknown exposure history, private material, or
pretraining. Record the exact checked versions and launch-time currentness.
Have an independent reviewer inspect cases, donor mapping, math, controls,
saved-output replay and failure handling without pretrained forwards.
