# Model-free FLAN multi-subtoken selection

This package implements the case-selection and semantic-handoff stages of
[`FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md`](../../docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md).
It does not load model weights or execute a model. The four discovery slots
and the two `painting` slots follow that protocol's fixed strata, word order,
and no-replacement rule. The pilot words and story/concert contexts are
tokenizer-exposed development material; the painting context is separately
assessed with the same words. The package makes no lexical-holdout,
pretraining-history, model-behavior, or global exposure claim.

The core validates the pinned pilot inventory, builds alphabetical handoffs
without proposed labels, retains each rater's directions, prior-exposure
statement and judgments, and chooses the first compatible unused tuple for
each fixed slot. Both raters receive the same fixed directions. The painting
record must name the same two rater IDs as the sealed discovery record. These
are checks of supplied records; they do not independently establish a person's
identity, blinding, or when the ratings were made.

The stage checks the exact protocol, candidate, pilot, tokenizer, asset
manifest and exposure-inventory hashes; reads named Git sources with ambient
Git overrides disabled; and compares complete EOS-stripped token rows in both
prefix directions. It verifies that the released held-out and sentiment pool
inventories are covered by their saved screens (240 and 96 token rows) and
checks their published commitments. The scan records all source row counts,
including sources with no token rows. Additional Git source descriptors and a
separate custodian receipt must describe the current relevant scope. These
checks bind records to their sources; they do not prove that all historical or
protected exposures are known.

## Running

Use a directly launched base Python 3.11.15 interpreter with `-I -S -B`;
the Windows virtual-environment shim is rejected. Provide the exact reviewed
repository commit containing these source files via `--source-commit`, and a
repository with the pinned public Git objects via `--repo`. The stage verifies
its executing core and stage bytes against that commit. For tokenizing phases,
provide the verified `tokenizers` 0.22.2 package site directory via
`--tokenizer-site`; the stage checks its resolved origin and records its
package and native-library hashes. No model library is imported.

Every phase also takes `--protocol`, `--candidates`, `--pilot` (gzip report),
`--tokenizer`, `--asset-manifest`, `--descriptors`, `--released-pool` and a
new `--output` path. Paths must be to the exact assets whose hashes appear in
the core. Output creation is exclusive: the stage writes, flushes, syncs, and
reads back the bytes. A separate operator must observe the direct child's exit
and retain the output SHA and independent saved-byte readback before accepting
a stage. Preserve partial outputs if execution or cleanup is uncertain; a
source hash or JSON receipt alone does not attest process chronology.

Run these phases in order; seal and retain each completed output before using
it in the next phase:

1. `handoff`: produce the story/concert reviewer handoff. Obtain two
   independent, complete reviews before any new model output. Each review is
   an object in a two-element JSON array with `reviewer_id`, `prior_exposure`,
   `instructions`, `handoff_sha256` and `ratings`. The instructions equal the
   handoff's fixed text; `handoff_sha256` hashes its canonical JSON *without*
   the final newline. Each rating contains `word`, `subject`, `label`
   (`positive`, `negative`, or `ambiguous`) and `reason`. Record truthful
   reviewer exposure, including any relevant prior context.
2. `discovery-screen`: provide `--reviews`, `--additional-sources` (a JSON
   array of Git `{path, revision, sha256}` descriptors, possibly empty),
   `--tokenizer-site`, and a new `--scan-output` path. It saves exact
   tokenization rows and a public prefix scan. A custodian independently checks
   the rows against the applicable protected scope and records one status per
   row as `clear`, `collision`, or `unknown`. The receipt has schema
   `flan-multisubtoken-custody-v1`, a scope description, and `rows` containing
   `word`, `subject`, `instruction`, `input_ids_sha256`, `status` and `reason`.
   The row hash is over canonical JSON for the complete input-ID array.
3. `discovery`: add `--screen`, `--scan`, and `--custody` to the same review,
   source and tokenizer arguments. The stage retokenizes and rescans, requires
   equality to the saved screen and scan, then writes all dispositions,
   review records, and four fixed discovery slots. Independently observe and
   seal the exact saved bytes and SHA before any painting tokenization.
4. `painting-handoff`: provide `--discovery` and the externally sealed
   `--discovery-sha256`. Only words from selected slots 0 and 2 appear; absent
   slots stay absent. Obtain ratings from the same two reviewers and record
   their prior story/concert exposure.
5. `painting-screen`: provide the sealed discovery binding, painting reviews,
   additional descriptors, tokenizer site and new scan output. No painting
   eligibility may alter discovery selection.
6. `painting`: provide the painting screen, public scan and custodian receipt
   as in step 3. The output binds the discovery SHA and keeps the two original
   slots and all missing or ineligible reasons.

The independent execution owner must revalidate the selected manifests,
tokenizer rows, relevant exposure bindings, and execution resources before a
model action. A future run and its scientific outcome are separate artifacts.

## Checks

Run `python -m pytest -q -p no:cacheprovider` on the two test files. The tests
use fabricated rows and a tiny temporary Git repository; they do not select
actual cases or run a model. They cover fixed-slot and underfill behavior,
reviewer and handoff bindings, painting isolation, span/site validation,
prefix matching, prior-pool coverage, source-byte and child-interpreter
checks, and the complete staged flow. Published prior records can be checked
independently without screening any successor input.
