# Synthetic-v2 held-out scoring

`looking_glass.causal_v2_scoring.score_heldout` scores a complete, already
revealed synthetic-v2 evaluation from persisted bytes. It neither creates
study cases nor runs a model or accesses a GPU. The current tests use only
fabricated development identifiers, values and outcomes; they are not results
of the proposed study. This helper does not authorize a study or publish files.

The function accepts, in order, canonical pre-manifest, freeze, private
prediction record, public prediction commitment, revealed held-out record,
and revealed schedule bytes; the three 32-byte held-out root, site nonce and
prediction nonce; and an iterable of the 54 saved complete held-out chunk
bytes in frozen-cell order. A chunk is parsed and checked before the next is
requested. The input iterator must read the actual saved files only after the
executor has stopped and cleaned up, and the caller must verify their durable
inventory and externally recorded hashes. The returned compact canonical
JSON records each chunk's name and SHA-256. Bounds on individual JSON records
and chunks are serialization bounds, not a complete host-memory reservation:
parsed chunks, inputs, accumulated results and encoding buffers can overlap.

Six keyword-only arguments default to `False` and must each be exactly `True`:

* `phase_order_verified`: the prediction freeze and commitment were saved
  before any held-out forward and all required phases completed in order.
  No invalidating failure, cancellation, failed control, cap violation or
  retry occurred before scoring, and the attempt obeyed the global fail-stop
  rule. A later post-commit crash is retained separately. Disclosure is
  recorded through the separate exposure classification.
* `input_model_pairing_verified`: the actual model, input, source and base
  references, forward identities, runtime controls and recorded forward counts
  were independently matched to the declared execution. Equal saved values or
  locally distinct run IDs alone do not establish those facts. The executor
  uses one attempt-wide forward counter; caller verification must also catch
  run-ID collisions across discovery and held-out chunks, because this scorer
  only consumes the held-out chunks.
* `private_plan_metadata_verified`: all saved plan and private-capture metadata
  were independently checked against the frozen policy and actual source/base
  ownership; generic nonempty IDs are insufficient.
* `durable_evidence_verified`: before scoring, each required chunk underwent
  atomic publication, file flush, saved readback and digest/inventory checking,
  and executor cleanup completed. An in-memory callback or a claimed path does
  not establish that evidence. Windows directory-flush and power-loss limits
  still apply; this flag does not imply universal durability. It does not require
  terminal publication, which happens only after scoring and later checks.
* `case_generation_verified`: both split constructions were recomputed from
  their proper roots and checked against the complete v1/pilot/development
  exclusion inventory. The scorer checks pairwise split disjointness itself,
  but cannot infer missing exclusions or prove root entropy.
* `discovery_freeze_verified`: discovery selections, measurements, controls
  and fitted means were independently recomputed from the persisted discovery
  chunks and compared to the frozen artifact before scoring. The scorer checks
  concrete predictions against frozen means; that comparison alone cannot show
  that those means were fitted on the promised discovery evidence.

These are cooperative caller attestations, not a security boundary or proof of
blinding, source execution, timestamps, independent sampling, or correct model
behavior. The runner must bind the protocol and all executed source files in
the pre-manifest and perform the checks described by the protocol and its
clarifications. Invalid, incomplete or unverified records raise `ValueError`;
iterator I/O failures and other operational exceptions can propagate as their
original types. The caller must preserve raw evidence and the failure status,
not replace a failed attempt with a partial result.

The caller must also pass `exposure_classification` as `unexposed`,
`unblinded`, or `unknown`, based on retained exposure records. This is a
caller-reported procedural classification, not proof that same-account
processes enforced secrecy. The result's `status: complete` refers only to
the checked evidence and controls. An unblinded attempt is explicitly labeled
exploratory, and unknown exposure cannot be treated as a confirmation.
Development attempts are not study confirmation. `unexposed` study summaries
remain conditional on all external attestations and the complete protocol.

The scorer verifies canonical encodings and all three salted commitments;
model, cell, case and strength order; case and input bit-pattern identities;
the three consecutive committed prediction rows referenced by each condition;
complete sparse traces, bit-preserving repeated-clean controls and saved L2
comparisons; target injection arithmetic; plan capture delivery; null, self,
upstream, downstream, reverse and post-hook clean controls; and the
architecture-specific step map after revealing each injection block. The
required comparison treats signed zero as equal and rejects nonfinite values.
The scalar logit contrast is calculated by exact integer binary32 rounding,
then converted to binary64 before the effect is subtracted. This reproduces
the specified arithmetic without relying on ambient flush-to-zero settings.
The helper validates plan structure and captures, but relies on the separate
metadata and pairing attestations for whether recorded IDs and actions reflect
the true forwards. It validates the exact typed plan and private-capture
metadata against the executor's policy as well as checking capture values;
plan source/comparison IDs must derive from the saved sparse run IDs, and base
IDs must be those run IDs followed by `/logits`. Within held-out evidence, a
clean or repeat ID can be reused only for the same model/input and role; each
perturbed ID is single-use and cannot alias either clean role.
The shared `validate_saved_controls(record, *, target, strength, polarity)`
performs the same checks on a caller-owned decoded discovery or held-out v2
condition without a model forward. The caller must keep the record unchanged
and verify cross-condition identities and actual execution separately. Both
the record and target-bearing validation remain private until reveal. Active
observations with no changed sparse boundary remain
observation misses rather than being dropped or called negatives.

The result retains every case's full baseline, forward, reverse, null, self,
upstream and final clean logits as float32 bits, site effects and predictions,
the corresponding reverse effect relative to the clean base, and
per-predictor residuals. `residual` is the polarity-conditioned prediction
minus the observed effect; `constant_residual` and `zero_residual` use the same
prediction-minus-observation convention. Repair is the absolute difference
between the observed patch effect and `m(clean)-m(perturbed)`; it is reported
per condition and grouped by model, revealed target, strength, input polarity
and site, without division by the task effect. The same groups report the
matched three-predictor MAEs and paired baseline improvements. Chunk evidence separately keeps
the full source/base identities, metadata and captures.

For each cell and site, polarity, constant and zero MAE use all 16 paired
inputs and both strengths, including zero effects. Selecting-policy summaries
average only the frozen non-abstaining cells within each model, reporting
coverage and paired baseline-minus-polarity improvements on the same sites.
An all-abstaining model has an undefined prediction summary; the equal-weight
mean of six models is defined only when all six are defined. Per-strength and
per-polarity summaries retain those same selected cases. Localization counts
an abstention as a miss while showing coverage separately. The random baseline
averages the three site's metrics within each cell, then the nine cells per
model and six model means; a single uniform draw has block-hit probability
one third. It is not a post-hoc best-site choice.

The sign summary pools selected cells within each model, excludes observed
effects with absolute value at most `1e-7`, treats predictions within that
same band as wrong on an eligible effect, and reports included/excluded and
positive/negative counts. An empty eligible set is undefined. Discovery and
held-out no-injection counts are shown separately once per model/input; the
discovery counts come from the independently verified freeze, while the
held-out counts are recomputed from the saved repeated-clean pairs. The six
fixed initializations, shared inputs, and nine configurations are not
independent samples. Results are descriptive and do not establish a model's
reasoning, causal explanation of a natural behavior, confidence intervals, or
statistical significance.
