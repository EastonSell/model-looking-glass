# Causal interventions: matched dose, prediction, and edit locality

Status: methodological review, 2026-10-29. This note interprets published
project results and proposes follow-up experiments. It does not amend any
frozen protocol, establish a new finding, or prescribe a launch gate.

## What the existing results establish

- The [DistilGPT2 copying factorial](https://github.com/EastonSell/model-looking-glass/blob/514459c52ccf50b0b3d07250c4ac9b028afe2da5/results/distilgpt2-copy-head-factorial-v1-20261028/README.md)
  used equal pre-projection L2 head-slice doses within each of eight public
  four-prompt families; the dose varied by family. Mean projection-output
  changes nonetheless differed: 5.803 for H2,
  4.556 for H10, and 5.671 for H7. The joint log-probability interaction
  was descriptive, and its hypothetical additive-logit reference is not
  an exact decomposition of the observed nonlinear response. Equal input
  norm does not imply equal projected perturbation or equal behavioral effect.
- The [DistilGPT2 subject-form study](https://github.com/EastonSell/model-looking-glass/blob/bc4caaace1746358ae6cfcfeca62346174b16c17/results/distilgpt2-agreement-subject-dependence-v1/RESULTS.md)
  found a larger direction response on the singular side of 31/32 fixed
  pairs, but also substantial shifts on subjectless fragments. Shared
  nouns and frames make these correlated, public observations. The result
  supports input-dependent sensitivity; it does not uniquely identify a
  grammatical-number computation.
- The [DistilBERT cue-set by frame factorial](https://github.com/EastonSell/model-looking-glass/blob/d5b23ff25134e6a22084bc0d9f1c8edab1863211/results/bert-cue-factorial-cpu-v1-20261029/README.md)
  found 0/3 cases satisfying its strict specificity composite in each of
  four cells. The comparison changed cues, placement and other linguistic
  details, and used a lower common dose than earlier studies. It did not
  isolate a cause of the difference or show that all transfers had no effect.
- The [ViT CLS/spatial study](https://github.com/EastonSell/model-looking-glass/blob/97305b9b3de805a5d5f97afb1b4cd6ffeeac6f22/results/vision-coco-cls-v5-20261029/README.md)
  found 0/2 model-screened pairs satisfying its composite. There were
  nonzero output changes, including a spatial top-1 change. Two pairs
  reused donors, so neither population conclusions nor spatial equivalence
  follow.

These are the named experiments' outcomes, not independent confirmations of
the proposed explanations. Their linked records define the exact checkpoint,
inputs, interventions, controls and exposure limits.

## Interpreting a matched perturbation

A same-norm control balances one property of an intervention. It need not
balance direction, fraction removed, projection into the residual stream,
local sensitivity or downstream distribution. A donor state is naturally
produced in its source run, but combining it with the recipient's other
states can still create an unusual joint state. In studied settings,
[Zhang and Nanda](https://arxiv.org/abs/2309.16042) found that patching
localizations can depend on corruption and metric choices;
[Grant et al.](https://proceedings.iclr.cc/paper_files/paper/2026/hash/133e588e1429f9f1e25b215da145580e-Abstract-Conference.html)
analyzed representation divergence from interventions. Neither finding
invalidates every patch; each motivates explicit sensitivity checks.

For a head slice x_h and its linear projection block W_h, a proposed separate
comparison can set one family-wide q to the minimum of ||W_h x_h|| over all
heads and repeated/shuffled variants, and attenuate each slice by
f_h = q / ||W_h x_h||. Finite nonzero denominators make f_h at most one.
Check the actual float32 projected changes, include input-norm-matched arms,
and report small-dose exclusions. This matches a single-head projected
magnitude, not direction or downstream effect. For two equal projected
norms q and angle theta, the ideal joint projected norm is
q sqrt(2 + 2 cos(theta)); it may be larger or smaller than either single
dose. Floating-point delivery and cancellation require measured checks.

For a dose response using an adjusted margin A = I - max(|S|, |R|), let
I, S and R be same-recipient opposite-donor, same-class and random effects
signed toward the opposite-class donor, using the recipient-specific sign.
Retain each raw effect and the identity of the largest control at each dose.
A change in A can result from a changing control or switching maximum as
well as a change in the opposite-donor effect. A same-class donor is an
active comparator, not a guaranteed semantic null; one random direction is
not a random-control distribution. Keep the existing prespecified decision
and interpret additional comparisons as descriptive unless frozen in a
separate protocol.

## A distinct next prediction test

The existing cross-model work plans subject-position donor swaps. A later,
separately versioned experiment can ask whether agreement predictions also
depend on the attractor or final-position state. This is a proposal, not a
result or a prerequisite for the existing study.

1. Fix discovery and held-out noun families and at least one held-out frame,
   disjoint from the earlier studies' known cases. Cross subject number with
   attractor number. Use a fixed, grammatical trailing token so the
   attractor and final token positions differ; verify tokenizer alignment
   and one-token answer continuations before model evaluation.
2. Freeze two simple competing predictions. A strong subject-following
   account predicts subject-number swaps produce donor-directed changes
   beyond same-number lexical controls, while attractor-number swaps have
   little effect by the specified criterion. A strong nearest-noun account
   predicts the converse and predicts the nearest noun's number on clean
   incongruent sentences. Permit both, neither, and inconclusive outcomes:
   actual networks need not implement either simplified account.
3. At prespecified, consistently named subject, attractor and final sites,
   compare opposite-number donors, same-number different-lemma donors,
   self patches and a fixed random control. Use both directions and retain
   full donor swaps and common-radius comparisons. If comparing sites by
   magnitude, match the delivered radius across them or state explicitly
   that doses differ. Token or source changes can still carry lexical and
   morphological information.
4. Record raw clean and patched is-minus-are logits, clean correctness on
   congruent and incongruent cases, actual dose and output distributions.
   Retain all failed baselines and report any eligible subset with its
   denominator. Do not use a baseline-correct subset to claim a nearest-noun
   rule was independently refuted. Evaluate held-out families without
   refitting sites, directions, thresholds or case choices.

GPT-2's causal mask prevents an earlier subject position from attending to
a later attractor; DistilBERT's bidirectional states need not have this
property. Accordingly, the same location and swap have different
interpretations across the two models. Even a correct held-out intervention
prediction is evidence for a particular aligned causal abstraction, not
proof of a unique circuit. For the formal framing of aligned variables and
interchange interventions, see [Geiger et al.](https://proceedings.mlr.press/v236/geiger24a.html).

## Editing and capability preservation

An activation-restoration map is not automatically a weight-edit map:
[Hase et al.](https://papers.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)
found that denoising localization did not identify the best layer for
overriding facts in their tested editing setting. Compare independently
chosen edit sites and methods at measured, approximately matched target
efficacy, and retain unmatched outcomes. Report related localities, such as
the same subject with a different relation or neighboring objects, as well
as unrelated tasks and baseline correctness. The neighboring-knowledge
setting of [Ma et al.](https://proceedings.mlr.press/v235/ma24h.html) is a
useful motivation, not evidence that these particular edits have side effects.

The fixed [FLAN-T5 input-row protocol](https://github.com/EastonSell/model-looking-glass/blob/74e525ca039f92006c64ece8cb10a3e89f632cff/docs/FLAN_T5_TARGETED_WEIGHT_EDIT_V1.md)
is an explicit token substitution, and correctly notes that cases without
the edited token in either encoder input or decoder prefix are structural
no-ops, not demonstrations of learned selectivity. A separate extension could
test more non-target tasks containing the edited token. Keep complete
pre/post output distributions and task-level results: in their tested
settings, [Hoelscher-Obermaier et al.](https://aclanthology.org/2023.findings-acl.733/)
found side effects that earlier specificity measurements missed. KL or
total variation describe distributional change, but neither alone proves a
capability was lost or preserved.
