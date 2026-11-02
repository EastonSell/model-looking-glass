# FLAN-T5 locally held-out dose-response, CPU

The fixed primary replication rule was **inconclusive**. Two of the six lexical
pairs were positive on both templates, and four were unevaluable. The separate
raw-baseline-gate track was also inconclusive with the same pair statuses. The
Δ-supported track, which is descriptive and includes some cells whose target
preferences did not pass the baseline sign gate, was positive under its fixed
rule: four positive pairs, one negative pair and one unevaluable pair. This is
not a primary replication result or a significance test.

The custodian reports that the 24-pair pool and two templates were chosen
locally after the source was published and before tokenizing the pool, without
querying the model. The pool
was committed with a private random salt and independently read back before
the tokenizer-only screen; the first six pairs passing both templates were
selected in pool order. Fourteen pairs passed that screen; the selected pool
indices were 1, 2, 3, 6, 7 and 8. The complete pool, salt, screen, case payload,
case salt, semantic review, public cards and observations are in `custody/`.
These locally recorded publication and draw chronology reports are not trusted
timestamps; withholding from collaborators does not prove independence from
the model's pretraining or make the examples a random population sample.

The selected words and outcomes are in `case-summary.csv`; all signed word,
instruction and punctuation movements and achieved matched doses at zero,
one-third, two-thirds and the full common dose are in `dose-curves.csv`. Both
templates of the bed/room pair were marked semantic-ineligible before the
case commitment, because bare “room” has distinct common interior and space
senses. They received no model forward and were not replaced. Five further
cells failed the fixed primary and raw sign baselines. Ten cells passed the
Δ≥1 descriptive eligibility and received the fixed intervention controls;
nine were descriptive-positive and one descriptive-negative. The negative
cell does not imply a zero word effect, and descriptive-positive cells can
remain baseline-unsupported on the primary track.
The `reasons` field in `case-summary.csv` covers semantic and intervention
failures. For cases with baseline measurements, raw and context baseline-gate
reasons are retained in `baseline_gate` in the original completed and case
JSON files.

For each target, the score is the sum of teacher-forced float32 log
probabilities for its fixed lexical token and EOS, using decoder-start token 0.
The two-target margin is the first target's score minus the second's. This is
not a greedy generated translation. The primary gate required opposite raw
margin signs, a gap Δ of at least one nat and context-reference-relative signs.
For a matchable cell, the operational positive rule required nondecreasing
donor-directed word movement with a 0.0001-nat adjacent tolerance, an endpoint
at least 0.05Δ, and at each nonzero level a word movement at least the larger
absolute matched-control movement plus 0.02 × fraction × Δ. These thresholds
are fixed exploratory decision rules, not calibrated nulls or p-values.
Equal input L2 does not imply equal functional perturbation, and the instruction
and punctuation sites can themselves carry relevant information. The
context-only prompt is a descriptive reference, not a correction for priors.

The source-bound CPU owner reported one successful child exit, reaping, no
timeout and empty bytecode prefixes. It recorded 780 forwards and a 1,018,544,128
byte peak process working set. That peak is an observation, not a future bound.
The runtime recorded CPU float32, deterministic eager attention, one thread,
`MKL_CBWR=COMPATIBLE`, Python 3.11.15, Torch 2.8.0+cu128, Transformers 4.57.3
and tokenizers 0.22.2; the launch requires the offline flags and hides CUDA.
The pinned model is `google/flan-t5-small` revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, with the exact seven-file asset
manifest bound in the preregistered source. No GPU execution is asserted.

The independently frozen saved-byte auditor passed all 780 records, including
exact float32 scores, full-state and logit controls, same-forward landing and
untouched-position witnesses. Its maximum diagnostic binary64-versus-recorded
float32 per-token log-probability difference was 1.4524e-5. The separate
scalar check initially stopped on one exact L2 comparison: squaring by `** 2`
instead of the fixed `d * d` operation differed by one binary64 ULP
(5.551115123125783e-17). The retained `audit/` records include this incomplete
attempt and a separate saved-byte diagnosis. The corrected scalar check uses
the protocol's multiplication, passed and agrees with the full auditor on all
case decisions and aggregate tracks; it was revised after inspecting the
failure. The original scalar source was fixed before its author inspected the
real result, but its publication is not claimed to precede the first forward.
Only the original traceback's local machine paths are redacted in the public
copy; its original hash and retained-local status are recorded in
`audit/scalar-stderr-redaction.json`.

The raw original run is preserved as indexed archive parts in `parts/`; use
`extract-and-verify.py --verify-only` with `source-index.json` to verify its
members. The terminal SHA-256 is
`256e125da4e79aa3d293dc518a40ef7b7c3cdb3d94436b587e15fedc3a959adc`.
The source protocol and runner were published at
[5342de9e](https://github.com/EastonSell/model-looking-glass/commit/5342de9e2df4c0c2ecef731154a369b5f8feae62);
the pre-forward full auditor was published at
[3f7c6c41](https://github.com/EastonSell/model-looking-glass/commit/3f7c6c41b633f9297627200918e093a9513bcafb).
The original and corrected scalar sources are linked by
[3b16ecbb](https://github.com/EastonSell/model-looking-glass/commit/3b16ecbbaf6453de5fc35bcabb41626a6c2a5e72)
and [9dc161f8](https://github.com/EastonSell/model-looking-glass/commit/9dc161f8d03bea29ec0a98c330b4a39fc6a17648).
Saved-byte and locally observed process records support these fixed checks;
they do not independently prove execution, secrecy, chronology, causal word
localization, pretraining independence or generalization beyond these cases.
