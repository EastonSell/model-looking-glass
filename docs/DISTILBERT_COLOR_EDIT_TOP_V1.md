# DistilBERT masked-color targeted change: prospective protocol

**Protocol only.** No new case inventory has been assembled or forwarded for
this study. Publish and independently review this protocol before constructing
the tokenizer-only cases. Freeze the cases, execution source, selection rules,
controls, checkpoint, and CPU runtime before the first pretrained forward.

This small study asks whether a single final-FFN parameter update can change a
masked object's usual-color prediction across new phrasings while disturbing
12 fixed locality prompts less than a deliberately global output-bias control
with comparable measured target effects. It concerns one public masked-token
prediction in one DistilBERT checkpoint. A masked-token top is not a generated
answer, and neither a two-token margin nor a small locality set establishes
broad capability preservation, semantic knowledge, or unique localization.

## Fixed question and chronology

Use the cached `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be` on eager CPU float32 with explicit
full attention masks, evaluation mode, one intra-op and one inter-op thread,
no generation, network access, autocast, training dropout or concurrent model
use. Pin the five checkpoint/tokenizer files and runtime versions in the
executable manifest. The CPU owner must independently verify asset bytes,
input rows, original parameter and buffer state, expected shape, finite
logits, hooks, and clean exit. An uncertain interruption, mutation, failed
cleanup or incomplete evidence is **invalid/incomplete** and takes precedence
over numerical results; save the completed prefix and do not silently retry.

Fix these candidate object/usual-color pairs and order before any new case or
model outcome: ripe banana/yellow, ripe strawberry/red, ripe lemon/yellow.
`purple` is the deliberately counterfactual replacement for each; `blue` is
the fixed wrong-target comparison. This is about conventional prototypes: fruit
colors vary with cultivar, ripeness and context, and the words are not a
complete description of those objects. Verify all four color words are distinct
single vocabulary tokens, with no unknown tokens, before the first forward.

For **each** candidate, construct in order one discovery sentence, two distinct
calibration sentences, four different held-out evaluation phrasings of the same
object/usual-color relation, and two same-object locality sentences asking a
different, clearly specified relation. Freeze ten further locality sentences:
five other-object color relations, including at least one for which `purple`
is an appropriate answer, and five non-color relations. All texts and expected
semantic readings must be independently checked. Construct exactly 37 distinct
input rows: nine screen rows, twelve held-out target rows, six same-object
localities and ten shared localities. Localities are active observations, not
assumed nulls or a representative capability benchmark. Freeze each case ID,
role, object, text, expected and accepted token IDs, and the one mask position;
each locality's accepted set must be nonempty, include its fixed expected ID,
have distinct in-vocabulary non-special IDs, and be independently checked for
the displayed relation. Every accepted item must be a complete single-token
fill, not a continuation fragment. Do
not add cases or alter accepted variants after observing any score.

Use the pinned tokenizer offline, with normal special tokens and one `[MASK]`;
retain its exact unpadded IDs, an all-one attention mask, token strings and
length for each case. Require one mask, no unknown tokens, finite length 2–64,
and single-token factual, purple and blue alternatives in the vocabulary.
For every target alternative and locality accepted token, independently
tokenize its complete filled sentence and require the encoded IDs to equal
the masked input with exactly that mask ID replaced by the accepted token ID;
this is tokenizer-only validation, not a model forward.
Check exact and complete-row-prefix relations in both directions among all 37
rows and against the currently known same-model public histories, including
their input namespaces. Check both the full encoded rows and rows with exactly
one final `[SEP]` removed; otherwise the terminator can hide a shared input
prefix. Preserve and document any genuinely different input namespace rather
than silently treating it as equivalent. Coordinate any applicable protected DistilBERT cases
with their custodian before a forward. Reserve the exact rows publicly before
screening. A missing required comparison or collision stops the affected stage;
no check establishes absence from pretraining or undisclosed exposure.

Publicly freeze the exact case record and reviewed owner before the screen.
Only the nine discovery/calibration rows are forwarded in that screen, once
each under the original model, in candidate/role order. Retain the complete
float32 vocabulary logit row, exact input and mask, source/action identity,
hash, truthful completion status and all top/rank/probability observations.
The factual color is **eligible** on a row only if it is the unique strict
full-vocabulary top; a tie has no unique top, is ineligible, and is recorded
with its count and smallest tied ID as a deterministic representative. Select
the first candidate in the fixed order whose discovery and both calibration
rows are eligible. Report all nine rows and candidate decisions. If none
qualifies, the study is **baseline-ineligible**: publish the full screen and
stop without an intervention, replacing cases, lowering thresholds or drawing
another candidate.

Publish the exact baseline results, decision and selected input IDs before
constructing any edit. The held-out target and locality rows for every candidate
were already fixed, but have not been forwarded in this attempt. For the
selected candidate, the four target phrasings are held out from edit and dose
construction, **not held-out objects or a private test set**. Only the selected
candidate's four target and two same-object locality rows, and all ten shared
localities, are used in evaluation; do not inspect other candidates' evaluation
rows with the model under this protocol.

## Edit, active controls and matching

On the selected discovery row, differentiate the purple-minus-factual masked
logit margin with respect to the final DistilBERT block's FFN `lin2.weight`.
Freeze all other parameters for differentiation. Capture the exact mask-site
linear input and output gradient and the full weight gradient; require finite,
nonzero values and a rank-one reconstruction residual at most `1e-5` relative
to the gradient norm. The gradient is used as an *edit direction*, not proof
that this site stores or uniquely causes the association. Require the original
grad-enabled discovery full logits and site input/output to match the original
no-grad replay; record both. Model state and buffers must match the screen's
original state before construction.

Normalize the gradient in float64 and, from the same restored original model
each time, measure every planned update norm fraction in this fixed ascending
grid: `0.0005, 0.001, 0.002, 0.004, 0.008, 0.016` times the original weight's
Frobenius norm. Cast the intended delta once to float32 before addition, save
the planned and realized norms and hashes, and require the realized update norm
to be within `[0.999, 1.001]` of plan. Save every discovery full-vocabulary
row, including failures; do not stop the grid on an early hit. Choose the first
fraction where purple is the strict unique discovery top. If none achieves
that, use the final fraction as a fixed **descriptive fallback** and record
that the development top was not achieved; do not expand the grid or call it
successful.

At that frozen dose, form three equal-realized-norm active weight controls:
the opposite gradient direction; one rank-one Gaussian direction from a fixed
CPU generator seed `20261121`, drawing the output factor and then the input
factor in float64 with no intervening draws; and the gradient of blue-minus-
factual on the same discovery row. Normalize and round each once; retain
source factors, hashes and realized norms, require `[0.999, 1.001]` matching
to the primary's realized norm, and do not redraw or choose a control by its
observed effect. These individual directions are controls, not a random-control
distribution. Each direction must be finite and nonzero. If a required control
cannot be constructed or rounded within the norm tolerance, record it as
unavailable and stop before held-out execution with **invalid/incomplete**
status; do not redraw, substitute or claim that all controls were tested.
Record their discovery and both calibration full-logit rows.

The second active comparator is a deliberately global output-bias update to
only the purple and factual vocabulary entries, with all other parameters
restored. Choose its two bias values *from the discovery row only* so its
realized discovery purple-minus-factual margin and purple log probability
match the selected FFN edit. Analytically solve against the original row's
other-token partition. Specifically, let `x` and `y` be the original and
primary-edited discovery logits; let `Lx` and `Ly` be their stable log-sum-exp
over every vocabulary entry except purple and factual. Set the bias-control
purple and factual logits to `fsum(y_purple, Lx, -Ly)` and
`fsum(y_factual, Lx, -Ly)`, respectively, leaving all other logits unchanged.
The corresponding changes
from `x` determine the two bias increments. This matches both endpoints in
exact arithmetic without subtracting a near-one probability. Cast the intended
bias update once to float32, then require a finite solution and an actual
no-grad replay within `0.001` nats of both discovery endpoints. Save the
original and realized bias entries, deltas, full logits and source hashes. If the primary discovery
margin gain is below `0.1` or its purple log-probability gain is below `0.025`,
or the solve/match is unavailable, mark the bias match **unavailable**; do not
invent or retune it. This comparison is not equal parameter norm and its global
logit effects are expected to be broad.

For each of the two calibration rows and later each of the four held-out rows,
compare actual FFN and bias gains from the same original full logits for both
purple-minus-factual margin and purple log probability. Each arm must gain at
least `0.1` margin and `0.025` log-probability nats on that row. For each
endpoint the absolute between-arm difference must be at most
`0.10 * max(abs(gains))`. Both arms must have the same unique full-vocabulary
top ID, including when that top is neither purple nor factual. Any tie or failed condition
makes the corresponding match unavailable. A failed calibration cannot be
rescued or retuned using held-out outcomes. If the discovery solve succeeds but
calibration fails, still evaluate that bias arm descriptively; the matched
locality comparison remains unavailable. Publish the complete grid, realized
weights/bias summaries, original/calibration rows and match decision before any
held-out forward. Verify that the decoder's tied weight alias remains unchanged;
the bias control must change only the two intended bias entries. The raw checkpoint weights need not be published; preserve
the pinned checkpoint and enough factors and hashes to reconstruct the edits.

## Evaluation and interpretation

With the frozen conditions, forward the four selected held-out target rows and
12 fixed locality rows under the original model, primary gradient update,
opposite direction, random direction, wrong-target gradient and, only when
its discovery solve is available, bias control, even if calibration matching
failed. Restore and check full parameter/buffer state, hooks,
input bytes and the affected parameter before and after each condition. Record
every finite complete vocabulary row and action status; any failed or
interrupted condition remains in the result and cannot count as completed.
The frozen owner permits at most 256 started forwards across all phases and at
most 128 MiB of uncompressed retained outputs. Count actual serialized bytes
of every journal, full-logit row, gradient/factor or intervention witness,
metadata and failure record, including any base64 expansion, as each is
written. Count no later compressed archival copy twice. Model assets, source
files, interpreter state and transient memory are excluded. Exceeding either
cap stops the attempt as invalid/incomplete; neither cap is a measured process
RSS bound.
Do not filter a target or locality case because its clean or edited top is
unexpected. Report the original and edited unique top/tie state, top gap,
purple and factual ranks, logits, probabilities and log probabilities;
compute full-vocabulary total variation and `KL(original || condition)` in
stable float64. Reduce KL with a compensated sum of `p * (log(p) - log(q))`;
record its raw value, reject a raw value below `-1e-12` as a numerical failure,
and clamp only values in `[-1e-12, 0)` to zero for comparisons. For each
locality also report the NLL of its fixed accepted token set as full-vocabulary
log-sum-exp minus accepted-set log-sum-exp, its NLL change, and
whether the unique top belongs to that set. Accepted sets are fixed semantic
probes, not exhaustive descriptions of every plausible masked answer. Keep
every row, including baseline-wrong and tied localities. A top-change count
includes only rows with unique original and edited tops and reports its
denominator; ties are unknown, never counted as unchanged.

An efficacy-matched locality comparison is **available** only if the discovery
top was achieved, the bias solve and both calibration matches passed, all four
held-out factual colors were strict original tops, and both arms satisfy the
same per-row held-out gain, closeness and unique-top-ID rules. In addition,
both arms must have purple as the unique top on at least two of the **same four**
held-out rows. Otherwise report the actual rows and an **unavailable match**;
do not select favorable rows or infer matched locality from mean gains alone.

When available, the fixed descriptive prediction is that the primary update
has mean locality KL strictly below `0.8` times the bias control's mean and an
absolute advantage of at least `1e-6` nats, while producing no more locality
unique-top changes than that control. It also requires primary mean KL at most
`0.02`, maximum KL at most `0.10`, mean accepted-set NLL deterioration at most
`0.05` and maximum individual deterioration at most `0.20` nats, and every
originally accepted locality's primary unique top still belongs to its accepted
set. The NLL bounds use all 12 rows, including baseline-wrong rows. If there are
no baseline-accepted localities, or any original, primary or bias locality top
is tied, this prediction is **not evaluable**. Report all 12 individual KLs,
total variation, accepted-set NLLs, top changes and their related/unrelated
strata as well as both means; a missing arm, unmatched effects or a tie is
never a success. The
threshold is an exploratory design choice, not a statistical test. Even a
passing comparison only contrasts this particular final-FFN update with a
purposefully global bias at this small fixed set of contexts. It does not prove
global semantic specificity, broad preservation, superiority over other edit
methods, or a distinct knowledge location.

An independent reader must reconstruct original/edited rows, exact eligibility,
matching, tie handling, all metrics and result precedence from saved bytes; an
independent reviewer checks case semantics and scientific interpretation.
Process and clock records are cooperative observations, not authenticated proof
of execution or protection from undisclosed inputs. Motivation includes the
separation of editing specificity and generalization in [ROME](https://arxiv.org/abs/2202.05262)
and the distinction between editing and causal localization in [Hase et al.](https://papers.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html).
