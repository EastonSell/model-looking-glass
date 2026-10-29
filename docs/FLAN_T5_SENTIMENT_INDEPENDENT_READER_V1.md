# FLAN sentiment intervention: independent saved-record reader

This reader is prepared before the prospective intervention. It reads a
**closed** attempt and does not load a model checkpoint or perform a forward.
The prospective selection and its salt remain sealed until the attempt has
closed; no new cases or intervention outcomes are included here.

`scripts/flan_sentiment_intervention_independent_audit.py` checks the released
selection commitment, derives the first four eligible units and reciprocal
donor map, retokenizes the selected rows with the pinned offline tokenizer,
and checks the fixed text templates and the actual subject/adjective offsets.
It compares those IDs and masks with the saved component records. It
independently reconstructs every matched vector using the adjacent
`flan_sentiment_dose_oracle.py` and the pinned Fraction-based
`bert_taste_matched_independent.py` helper. It checks all saved float32
tensors for finiteness and hashes, actual landing and untouched-state bits,
the full decoder logits and greedy sequences, teacher scores and ranks,
first-token total variation, controls, ineligible skips, ordering, both dose
levels, and the fixed four-unit rule. Its report retains all 32 baseline
records, all measured case-level matched actions and the natural references.
The CPU PyTorch log-softmax kernel in the
auditing environment recomputes the recorded float32 teacher values; an
additional binary64 log-sum-exp check constrains their mathematical meaning.

The report binds the started record, event stream and terminal hashes and
checks the recorded owner, selected commitment, source revision, and this
reader's three-file pin for internal agreement. These are **saved-record
checks**, not authentication of execution or publication chronology. A
separate independent custodian must verify the complete pool, tokenizer
selection and public exposures; a separate owner/process review must verify
the source and asset pins and observed direct-process exits. Equal norms do
not equalize downstream effects, and the comparison arms are active controls.
An incomplete or absent terminal returns `incomplete` without a primary result;
that response does not claim to have numerically replayed the partial journal.

After closure, supply the independent custodian's public commitment values,
the released selection record and its exact 32-byte binary salt, the observed
started-record SHA-256, the owner revision, and the pinned local tokenizer
asset directory:

```text
python -I -B scripts/flan_sentiment_intervention_independent_audit.py \
  --attempt <closed-child-directory> \
  --selection <released-selection-json> --selection-salt <released-32-byte-file> \
  --pool-commitment <64-hex> --selection-commitment <64-hex> \
  --runner-revision <40-hex> --started-sha256 <64-hex> --assets <pinned-asset-directory>
```

The tokenizer, PyTorch, and their environment are trusted local dependencies;
the five tokenizer asset files and their manifest are hash-checked. The reader
checks the local binary32/binary64 rounding and denormal mode before parsing
JSON and fails closed if it differs. Exact libm and kernel reductions are
replayed on the auditing runtime and can vary across platforms. The report is
JSON on stdout; do not publish any private selection data or unsalted selected
row digest before the attempt closes.

## Fabricated-only checks

The following generator takes the separately reviewed runner checkout and
creates a new directory containing only invented 2-dimensional states,
8-token logits and fake input rows. It uses no pretrained weights. Set
`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

```text
python -I -B scripts/write_flan_sentiment_fake_journal.py <runner-checkout> <new-fake-attempt-directory> pass
python -I -B scripts/flan_sentiment_intervention_independent_audit.py --synthetic --attempt <new-fake-attempt-directory>
```

Point `FLAN_SENTIMENT_FAKE_ATTEMPT` at that completed fake attempt and run
`python -I -B scripts/test_flan_sentiment_journal_audit.py`. The suite checks
the accepted complete fixture and deliberately rehashed source, untouched
state, nonfinite, score, TV, summary, input-type, mode and selection
corruptions. The fixture producer also accepts `zero` and `rounding` to
exercise respectively dose-ineligible and dose-unmatchable complete schedules.
To test both failures at once, generate another fake attempt with mode
`rounding_unsupported` and point `FLAN_SENTIMENT_FAKE_COMBINED_ATTEMPT` at it;
the combined-failure test otherwise reports a skip.
`python -I -B scripts/review_flan_sentiment_linear_toy.py <runner-checkout>`
additionally checks passing and failing decisions against hand-derived fake
effects and verifies both skip paths.
