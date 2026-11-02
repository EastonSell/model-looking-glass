# Fixed two-dose cue-specificity comparison: prospective CPU protocol

## Question and fixed cases

The [factorial study](https://github.com/EastonSell/model-looking-glass/blob/d5b23ff25134e6a22084bc0d9f1c8edab1863211/results/bert-cue-factorial-cpu-v1-20261029/README.md)
used a common 0.5 intervention norm and failed the strict same-recipient
specificity rule in all four cue-set/frame cells. Earlier cue studies used
larger, individually chosen radii. Does increasing the fixed norm from 0.5 to
0.7 change that rule or its worst-target margin on new, fixed public cases?
This is a paired two-dose comparison, not a causal analysis of syntax, a
population estimate, or a test of general negation understanding.

Before any new pretrained model forward, fix three public evaluation triples:
`chef/prepared/meals`, `farmer/harvested/crops`, and
`pilot/inspected/instruments`. Include `gardener/watered/plants` as an
**exposed calibration** triple: its eight rows are exactly the factorial
study's already-forwarded main-clause cases and are excluded from all fresh
evaluation denominators. The fixed template is
`The {role} has {cue} {verb} [MASK] of the {object}.` The cue sets are:

* Prior: negative/low-frequency `never`, `rarely`; affirmative/high-frequency
  `always`, `often`.
* New: negative/minimizing `not`, `hardly`; affirmative `really`, `certainly`.

The manifest contains eight cells and 32 variant rows: two cue sets for each
of the three evaluation triples plus two exposed calibration cells. Evaluation
triples are paired across cue sets; all inputs are public, not blinded or known
absent from pretraining. Selection and dose choice followed the observed
factorial outcome. Verify all rows and the calibration identity against the
pinned tokenizer and prior manifest before loading model weights. A check of
named public BERT rows must find no exact or either-way prefix collision for
the 24 fresh rows, both with complete rows and with one final `[SEP]` removed.
The eight declared calibration-row matches to the prior factorial cases are
expected; report them separately. If a fresh collision is found, record it and
stop this fixed study without replacing the input. The check does not establish
private, complete-history or pretraining nonexposure. No model output from this
new study may replace cases or alter its sites, dose values or thresholds.
The [recorded public comparison](../experiments/bert_cue_dose_response_public_exposure_check_v1.json)
finds zero fresh-row relations against 481 rows in 11 named snapshots and the
eight expected calibration matches; it also finds no relations among distinct
new rows. This is a comparison of those pinned public snapshots only.

`rarely` and `hardly` are scalar and not interchangeable with `never` or
`not`; the affirmative cues also differ in frequency, intensity and certainty.
`any` can have free-choice uses, and a score contrast is not human
acceptability. NPI evaluations are sensitive to licensor and method
([Warstadt et al., 2019](https://aclanthology.org/D19-1286/),
[DeCarlo et al., 2023](https://aclanthology.org/2023.blackboxnlp-1.25/)).

## Fixed doses, eligibility and controls

Use cached `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, CPU float32 eager attention,
evaluation mode, one intra-op and one inter-op thread, deterministic
algorithms, no autocast or network, and seed `2026102905`. Set
`CUDA_VISIBLE_DEVICES=-1` before Torch import and never initialize CUDA. Reuse
the prior owner's pinned assets, dependency versions, model-state digest
including nonpersistent buffers, and before/after no-denormal-flush probes.
Record source, manifest, runtime and state hashes.

Capture cue, object and mask after block 2 and cue after final block 5. For
each cell compute all 24 pair/site distances among its four clean cue states,
using `sqrt(fsum((float(y_i)-float(x_i)) ** 2))` in coordinate order. For
each dose independently, require every distance finite and at least the fixed
requested norm; a tie is eligible and no donor is amplified. The fixed doses
are 0.5 and 0.7, selected after the factorial study's 12 observed minimum
distances, 0.6943–1.2629. The earlier two studies' minimum observed radius
was 0.7678, so a negative result at 0.7 cannot exclude an effect at those
larger doses. These choices may be too weak or infeasible in new cells.

If a cell's minimum distance is below 0.5, record both doses ineligible, four
clean baselines, all distances, named ineligibility and four final clean
repeats; do not patch. If it is at least 0.5 but below 0.7, run all 0.5
patches and record 0.7 as ineligible. If it is at least 0.7, run both dose
families. A nonfinite distance, failed control, incomplete forward or
rounding/landing error makes the attempt incomplete, not a negative result.
Baseline-ineligible cases still receive every dose-feasible patch.

For an eligible transfer use binary64 `alpha = dose / distance`, with
`0 < alpha <= 1`; compute separate binary64 subtraction, multiplication and
addition then one nearest-even float32 cast. Require the same-forward
pre-action state to bit-match its clean capture, the selected landed vector to
match the submitted vector, unchanged token coordinates to match, and the
realized displacement's relative error to be at most 0.001. Retain fractions,
all input/donor/submitted/landed vectors, whole before/after states and the
complete 30,522-element mask-logit vector. Hidden width must be 768. Preserve
bounded returned witnesses and named failure phases when feasible.

Process cases in manifest order, and cues in each set's declared order. For
each cell: four clean forwards; if at least one dose is feasible, for each
site in `cue2`, `object2`, `mask2`, `cue5` order run four same-forward self
patches, then all 12 distinct transfers in target-then-donor cue order at 0.5
and, if feasible, the same ordered transfers at 0.7;
finally four clean repeats. Thus a cell uses 8, 72 or 120 forwards depending
on dose eligibility, with a maximum of 960 across eight cells. Each dose's
same-recipient control is delivered at that same dose. Clean and repeat runs
use passive capture hooks, not a separately hook-free baseline. Self and
repeat full-mask logits must match clean bits, and final-block cue transfers
at either dose must preserve the full mask-logit vector. Object and mask
sites are active comparators rather than assumed nulls.

## Fixed outcomes and paired comparison

Let `L = logit(any) - logit(some)`. The clean baseline screen requires all
four `L(negative)-L(affirmative)` gaps to exceed 0.5 strictly. At each dose,
sign opposite-class donor effects −1 for negative/minimizing targets and +1
for affirmative targets. Each donor effect for the prespecified decision is
patched-minus-clean `L` at the block-2 cue site (`cue2`). A target's margin
is the weaker of its two signed
opposite-class donor effects minus the absolute effect of the other
same-class donor into the same recipient at the same dose. For each cell and
dose, `M` is the minimum of the four target margins. A case passes only if its
baseline and that dose are eligible and every target margin is strictly
greater than 0.05. Ties at baseline or specificity thresholds fail; these
are fixed operational thresholds, not calibrated significance cutoffs. Record
all four margins and `M` for every dose-feasible cell even if its baseline
screen fails, plus all exactly tied limiting targets and raw and signed donor
effects, actual dose/fractions, clean probabilities and full-vocabulary top
five. The limiting target can change across doses, so a change in `M` is not
necessarily a change for one fixed target.

For each cue set and dose, the fresh-evaluation cell is `unevaluable` when
fewer than two of three fresh triples pass the baseline and dose screens.
Otherwise it passes if at least two eligible triples pass all four targets,
and fails if not. Keep the three per-triple outcomes and eligibility reasons;
exclude the exposed calibration from these counts.

Within each cue set, a paired contrast uses a fresh triple only when its
baseline and **both** doses are eligible. Record `M(0.7) - M(0.5)` using a
two-term binary64 `fsum`, its limiting targets and its low/high pass/fail
transition. With at least two paired eligible triples, report every difference,
their arithmetic mean `fsum(values)/count`, and the counts of fail→pass,
pass→fail, pass→pass and fail→fail. The descriptive transition status is
`changed` if any paired case switches, otherwise `unchanged`. With fewer than
two paired eligible triples, the aggregate is `unevaluable` and its mean is
null; retain any individual eligible pair and all ineligible cases. No case is
imputed or replaced. No hypothesis test, monotonic or linear dose curve, or
population inference follows from two doses and three public triples. Report
the low-dose, high-dose and paired denominators separately: feasibility of
the larger dose may select a subset of low-dose-eligible cases.

Report calibration separately. After closure, compare its saved clean and
0.5-dose records with the earlier exposed factorial records where available,
including full mask-logit bits and target/landing vectors. Treat equality or
disagreement as a descriptive cross-run check, not as an automatic exclusion
or a basis for retuning. Equal input norms do not equalize functional impact,
and this single placement does not test or isolate syntax.

## Evidence

Freeze and independently review the exact cases, protocol and executable
before the first new pretrained forward. Save ordered, sync-flushed gzip
JSONL with completed logits and intervention witnesses, including all
ineligible or failed cases. Independently replay the closed schedule, both
dose families, landing/untouched witnesses, full-logit controls, screens,
decisions and paired comparisons without a new model forward. The operator's
direct CPU exit observation and recorded hashes rely on a trusted quiescent
checkout, interpreter and dependencies; they are not execution or pretraining
attestations. Since disk headroom has fluctuated in this workspace, require at
least 2 GiB free on the output volume before creating the run directory, and
at least 1 GiB before each evidence event; falling below the latter stops the
attempt as incomplete. These checks are point observations and cannot prevent
another writer from filling the volume between checks.
