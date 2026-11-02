# Cue-set by clause-placement comparison: prospective CPU protocol

## Question and fixed design

The earlier [cue-class study](https://github.com/EastonSell/model-looking-glass/blob/a13a2eab68d02cbf93f43af9ad01559364d129fd/results/bert-cue-class-cpu-v1-20261029/README.md)
passed its fixed specificity rule in 7/9 evaluation contexts. A later
[held-out-frame follow-up](https://github.com/EastonSell/model-looking-glass/blob/975d55aea042912e88cee86b13738d490fea18b6/results/bert-cue-heldout-frames-cpu-v1-20261029/README.md)
failed in all six contexts, but changed cues, frames and lexical triples at
once. This prospective 2×2 comparison holds the lexical triple and requested
intervention norm fixed while crossing the two cue sets with two clause
placements. It can describe their joint pattern on these inputs, not identify
a single cause of the earlier difference or general negation understanding.

Before this study's first pretrained forward, fix exactly three public
role/verb/object triples: `teacher/graded/essays`, `gardener/watered/plants`,
and `mechanic/repaired/engines`. Cross each with both templates and cue sets:

* Main: `The {role} has {cue} {verb} [MASK] of the {object}.`
* Subject relative: `The {role} who has {cue} {verb} [MASK] of the {object} has signed the form.`
* Prior set: negative/low-frequency `never`, `rarely`; affirmative/high-frequency
  `always`, `often`.
* New set: negative/minimizing `not`, `hardly`; affirmative `really`, `certainly`.

Thus there are 12 cells and 48 variant rows, without new development
forwards. The main-clause placement corresponds to an earlier configuration,
but it is **not** the earlier exact template: the common `has` is added to
make the cross with `not` grammatical, and changes aspect. The subject-relative
placement and suffix follow the later study. Both frames use the same `has`,
cue sets and triples, but prefix, token position, length and discourse meaning
still differ. A frame contrast does not isolate syntax from these changes.

The cue classes are operational labels. `rarely` and `hardly` are scalar and
not interchangeable with `never` or `not`; `always`, `often`, `really`, and
`certainly` differ in frequency, intensity and certainty. `any` can have
free-choice uses. The comparison is a score under interventions, not a human
acceptability judgment. Equal input norms do not equalize functional impact.
Cases are public and chosen after both earlier outcomes; the three lexical
triples are paired repeated contexts, not independent population samples or
known absent from pretraining. Prior work shows that NPI assessments depend on
licensor and method ([Warstadt et al., 2019](https://aclanthology.org/D19-1286/),
[DeCarlo et al., 2023](https://aclanthology.org/2023.blackboxnlp-1.25/)).

Verify all 48 rows, indices, attention masks, cue and scored-token IDs with the
pinned tokenizer before loading model weights. The [public inventory comparison](../experiments/bert_cue_factorial_public_exposure_check_v1.json)
found zero exact or either-way full/one-final-SEP-removed prefix relations with
433 rows in ten named public BERT snapshots, including the pending color
reservation, and zero among the 48 new rows. It does not clear private splits,
unknown history or pretraining exposure. No output from this study is used to
replace cases, choose sites or retune a threshold.

## Runtime and equal-dose controls

Use cached `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, CPU float32 eager attention,
evaluation mode, one intra-op and one inter-op thread, deterministic
algorithms, no autocast or network, and seed `2026102904`. Set
`CUDA_VISIBLE_DEVICES=-1` before Torch import and never initialize CUDA. Reuse
the preceding source-bound owner's exact assets, versions, state digest
including nonpersistent buffers, and before/after no-denormal-flush probes.
Record source, manifest, runtime and state digests.

Capture the cue, object and mask after block 2 and cue after final block 5.
Each cell's 24 distances cover all six unordered pairs of its four cues at all
four sites. For vectors `x`, `y`, compute
`sqrt(fsum((float(y_i)-float(x_i)) ** 2))` in coordinate order. The requested
radius is **0.5 for every cell and triple**. This value was chosen before this
study's outputs using the previous 18 observed minimum radii, all at least
0.7678. It may be below a useful dose or infeasible in new inputs; it does not
reproduce the previous studies' individually chosen radii.

All 24 natural distances must be finite and at least 0.5 for dose eligibility;
a tie is eligible. If a distance is smaller, retain the four clean records,
all distances, a `dose_ineligible` record and the four final clean repeats,
then continue. Never amplify, replace a case or silently drop it. For an
eligible transfer use `alpha = 0.5 / distance`, binary64 operations and one
nearest-even float32 cast. Require `0 < alpha <= 1`, the pre-action state to
bit-match its clean capture, the landed vector to bit-match the submitted
vector, untouched token coordinates to match, and realized displacement's
relative error to be at most 0.001. Record all distances, fractions, vectors,
before/after whole states and complete 30,522-element mask-logit vectors.
Hidden width must be 768. Nonfinite inputs, incomplete forwards, failed
landing, dose or control invalidate the attempt rather than count as a
negative. Retain a bounded returned witness and named failure when feasible.

Cases follow manifest order. For each, run four clean variants; for each site
in `cue2`, `object2`, `mask2`, `cue5` order run four self patches and all 12
distinct directed transfers in target-then-donor cue order; then run four
clean repeats. A dose-eligible cell uses 72 forwards; the maximum is 864.
Clean/repeat forwards use passive capture hooks, not a separate hook-free
baseline. Self and repeat full mask logits must equal clean bits; final-block
cue transfers must also preserve the full mask-logit vector. Object and mask
sites are active comparators, not assumed nulls.

## Fixed screens, outcomes and paired contrasts

Let `L = logit(any) - logit(some)`. In each cell the clean screen passes only
if all four `L(negative)-L(affirmative)` gaps in its cue set exceed 0.5
strictly. Save all clean values, gaps, top-five mask tokens and probabilities.
Baseline-ineligible cells still receive the full patch schedule when
dose-eligible. Dose- or baseline-ineligible cells are retained and not counted
as primary passes.

For target cue `t`, let the sign be −1 if its class is negative/minimizing and
+1 otherwise. The target margin is the weaker of the two signed opposite-class
donor effects minus the absolute effect of the other same-class donor into
that same recipient at the same requested norm. The cue-site cell outcome is
the **minimum of its four target margins**, `M(cue_set, frame, triple)`; a
screen-eligible case passes iff `M > 0.05` strictly. Save every raw effect,
signed effect and margin, including ineligible or failing cases, and report
the limiting target or all exactly tied limiting targets. At each of
the four cue-set/frame combinations, fewer than two of three screen-eligible
triples makes that cell `unevaluable`; otherwise the cell passes if at least
two eligible triples pass, and fails if not. Ties at the clean or specificity
threshold fail.

For each triple, define the signed cue-set contrast at a fixed frame as
`C(frame) = M(new, frame) - M(prior, frame)` and the frame contrast at a fixed
cue set as `F(set) = M(set, subject_relative) - M(set, main)`. Each contrast
uses only triples where its two particular cells pass both screens; report
all eligible per-triple differences and their arithmetic mean when at least
two paired triples qualify, or `unevaluable` otherwise. For each triple where
all four cells pass both screens, define the interaction as
`I = C(subject_relative) - C(main)` and report its mean if at least two
complete triples qualify, otherwise `unevaluable`. This sign convention is
fixed before outputs. Compute each two-cell difference with binary64 `fsum`
on its two terms; compute each interaction with one binary64 `fsum` of the
four terms `M(new, relative), -M(prior, relative), -M(new, main),
+M(prior, main)`, then compute each mean with `fsum(values)/count`. This
rounding order can differ by an ULP from subtracting two rounded `C` values.
Do not impute, substitute or treat an ineligible case as a negative. Also
report the raw `M` values for every dose-eligible cell,
including baseline-ineligible cells, distinctly from the screened contrasts.

These paired contrasts and the four cell classifications are descriptive:
there is no significance test, causal isolation of syntax or population
estimate. `M` is a strict worst-target summary over different cue words; the
limiting target may change between cells, so its differences and interaction
must be read alongside all four target margins, not interpreted as an effect
on one fixed target or an abstract polarity parameter. Screening on model scores
may change the eligible pairs and makes estimates conditional on this screen.
Report denominator and eligibility reason for every cell and contrast.

## Evidence

Freeze and independently review cases, protocol and executable source before
the first new pretrained forward. Save ordered gzip JSONL with sync-flushed
events, complete mask logits and bounded intervention witnesses. Retain
incomplete attempts and failure phase. Independently replay the closed
schedule, target scores, dose, landing, unchanged-token witnesses, controls,
screens and outcomes without another model forward. Direct CPU process exit
is an operator observation; recorded hashes rely on a trusted quiescent
checkout and dependencies and are not execution or pretraining attestations.
