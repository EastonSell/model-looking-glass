# Independent pairwise copying: preliminary saved-record replay

`looking_glass.copy_pairs_dose_audit` independently replays the numerical
parts of the prospective public pair/dose protocol. It imports neither the
experiment runner nor Torch. It does not load a checkpoint, run a model, or
establish source authenticity, process execution or chronology. The cases are
public and nonblind; these calculations do not establish a unique mechanism,
head cooperation or generalization to a population.

`analyze_family` computes scores and the six pair/dose hypothetical additive
full-vocabulary references from four ordered saved logit sets. It retains
per-prompt deltas, pair interactions, raw target-logit diagnostics, low-dose
matched-single comparisons and high-minus-low and between-pair comparisons.
The reference uses `(single_a - clean) + single_b` in binary64 without a
float32 recast. A nonzero log-probability interaction may arise from
normalization even when logits add exactly; the reference does not isolate
the precise cause of an observed interaction.

`validate_delivery_family` recomputes all requested doses from the four
saved no-op projection inputs, checks the integer-defined nearest-even
binary32 delivery, achieved norms and every post-action bit, including all
untouched coordinates. It records changed-element and XOR bit counts.
`audit_saved_family` additionally checks all 72 ordered records, exact raw
slice offsets, lengths and SHA-256 digests, complete scores, no-op/clean
controls, pre-action input equality and full projection-output finiteness. It
reports every output norm and pair residual. `replay_result` checks all eight
families, the fixed schedule, 576 counters and aggregate payload length; its
callback is invoked for only one family of four case files at a time.
`summarize_study` preserves every family and uses `math.fsum` for means. The
default vocabulary size is 50,257; smaller explicit sizes exist only for
fabricated tests.

These functions are **not yet a complete study evidence reader**. In
particular, they do not bind the case file, manifest, exposure inventory,
source revision, process receipt, whole-file hashes, journal or runner's
summary to an independently published identity. The eventual caller must
perform those checks before reporting an execution complete, and must preserve
partial/failure records without silently retrying. `read_case` must itself
enforce a bounded read; the function checks byte length after it returns. The
fixed successful case payload is 5,240,520 bytes and the total is 167,696,640
bytes. Decoded Python floats, temporary vectors and caller-held buffers add to
these payload sizes, so they are not a peak-memory limit.

The frozen protocol currently spells norm squares with `** 2`; the replay
uses that literal operation. On the tested Windows CPython runtime a binary64
halfway example shows `x ** 2` and `x * x` can differ by one ULP, and the
former may choose the odd neighbor. The protocol's blanket nearest-even
wording therefore requires a public clarification before execution. A runtime
or library change can also affect exact transcendental results, so cross-runtime
bit equality must not be assumed.
