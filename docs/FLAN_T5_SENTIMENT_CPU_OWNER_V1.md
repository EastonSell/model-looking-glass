# FLAN sentiment CPU owner and selector-stage observation

This owner supplies the admission and direct-child observation around the
[prospective protocol](FLAN_T5_SENTIMENT_INTERVENTION_V1.md) and
[execution core](FLAN_T5_SENTIMENT_CPU_CORE_V1.md). It does not itself form a
pool, establish an independent reviewer, prove publication timing, or make a
scientific result. Its tests use fabricated cases and model objects. Before a
real pool is formed, publish and independently review the exact selector,
stage observer, owner/operator, execution core and independent readers.

The custodian refreshes the known FLAN exposure and protected-pool reports,
fixes the 12-unit pool and its semantic dispositions, and preserves canonical
UTF-8 JSON without a trailing newline. The pool names the exact reviewed
owner/runner Git commit. No tokenizer screen precedes observation of the first
public pool card. The 12 units are all screened, with no replacement, and the
second public card is independently observed before any model forward. Keep
the pool, salts, selection, process logs and selected cases private until the
fixed attempt closes; follow the protocol's reveal rule on incomplete stages.
Immediately before the study grant, refresh and retain the scoped exposure and
protected-pool observation again. If a new protected pool is known, its
custodian privately checks the fixed proposal; otherwise record that none is
known. This observation does not establish an unknown exposure history.

## Selector stages and independent replay

For each selector stage, prepare a new empty outer bytecode-prefix directory
and new stage-attempt directory. Run the observer from the reviewed checkout,
with `REVISION` the reviewed commit and all paths absolute. On Windows use
the direct base Python executable, not the virtual-environment launcher;
`VENV` identifies the trusted dependency environment whose `pyvenv.cfg`
names that base interpreter and the same Python version:

```sh
BASE_PYTHON -I -S -B -X pycache_prefix=EMPTY_OBSERVER_PREFIX \
  tools/observe_flan_sentiment_selector_stage_v1.py \
  --revision REVISION --venv VENV --attempt NEW_STAGE_ATTEMPT -- \
  commit-pool --repo CHECKOUT --pool PRIVATE_POOL.json \
  --pool-salt NEW_PRIVATE_POOL_SALT.bin --pool-card NEW_POOL_CARD.json
```

Publish the pool card and have a separate observer read its exact Git commit,
path and SHA-256. Then run the `select` stage in a new observer/prefix pair,
using the selector arguments documented in
[FLAN_T5_SENTIMENT_SELECTOR_V1.md](FLAN_T5_SENTIMENT_SELECTOR_V1.md). The
observer's `stage-runtime.json` records Python, Transformers and tokenizers
versions inside the same interpreter that runs the selector; the loaded
libraries, source and empty child prefix are checked again afterward. Its
`stage-exit.json` separately records the direct child's reaping, exit code,
timeout, runtime-receipt hash and bounded log lengths/hashes. Accept a stage
only after actual observer exit zero and a completed runtime/exit record;
preserve partial files and do not retry this fixed attempt on uncertainty.
The child waits for a one-shot stdin grant before loading the selector or
reading private inputs. The observer sends it only after it has the direct
child object, started both bounded log drains and recorded the spawn. A lost
spawn or grant outcome is incomplete: retain the record and resolve child
state before further work. The grant and direct-child record do not prove the
state of unknown native spawns or descendant processes. A child that has
already received its grant can outlive a killed or crashed parent: this gate
is not a parent-death watchdog. Preserve an uncertain attempt and establish
the actual child state before further work; a parent exit alone does not prove
termination.
The granted child verifies the virtual environment and appends only its
site-packages directory, without executing `.pth` or site startup code. Its
first stdout record binds its process ID, parent ID and hashes of the interpreter path,
prefix path, dependency path and venv configuration; the observer requires that ID
and parent to match its direct child and observer. The runtime receipt records the actual Python,
Transformers and tokenizers versions. The interpreter, environment and native
libraries remain trusted inputs. The loaded tokenizer libraries are checked
against that selected site directory; the owner makes the same origin check
for Torch, NumPy, Transformers and tokenizers before model loading.

The two card observations have exactly `schema`, `path`, `revision`, and
`sha256`, with schema `flan-sentiment-pool-observation-v1` or
`flan-sentiment-selection-observation-v1` respectively. The observed card
commits and the auditor commits must be available as Git objects in the
checkout even when its HEAD remains fixed at the owner revision. Git checks
disable replacement objects. They establish local blob consistency only, not
independent public chronology.

A separate model-free reader retokenizes all 96 rows, checks both commitments
and the complete prior-source screen, and records a
`flan-t5-sentiment-independent-tokenizer-replay-v1` receipt. It binds the
observed selector source revision, versions, raw runtime and exit receipt
hashes, both commitments, exact runner and public auditor source. The
independent reader does not assess semantic dispositions, unknown exposures,
physical secrecy or whether the claimed observer was independent. Fewer than
four eligible units is a no-forward outcome, not permission to redraw.

Separately provide the already reviewed full scientific reader as a canonical
sorted UTF-8 JSON file with one final newline:

```json
{"revision":"REVIEWED_40_HEX_COMMIT","schema":"flan-t5-sentiment-scientific-reader-pin-v1","sha256":{"scripts/bert_taste_matched_independent.py":"64_HEX_SHA256","scripts/flan_sentiment_dose_oracle.py":"64_HEX_SHA256","scripts/flan_sentiment_intervention_independent_audit.py":"64_HEX_SHA256"}}
```

The custodian independently observes publication of that reviewed source.
The owner checks the specified three Git blobs against these hashes; the
resulting source pin is not an execution attestation.

## One-shot CPU owner

Use another new empty outer prefix and new attempt directory. Pass the sealed
private inputs, both public cards and observation files, the `select` stage's
`stage-runtime.json` and `stage-exit.json`, the independent tokenizer replay,
the scientific-reader pin, the pinned seven-file asset directory, and the
reviewed owner commit and `--venv VENV` to
`tools/run_flan_t5_sentiment_intervention_v1.py`, launched with the same
direct base interpreter and `-I -S -B` settings.
Its `--help` lists the required path options. It creates the attempt directory
exclusively, observes the child exit and bounds each child log to 64 KiB.
The owner child likewise waits for the grant before loading the owner package,
reading custody files, importing the tokenizer or loading the model.
Log truncation, timeout, missing receipt or a nonzero/unknown exit makes the
attempt incomplete. Retain the raw PID-bearing spawn receipt privately; any
public projection must disclose that it is a projection and preserve the
original digest separately.

Before checkpoint loading, the child checks both salted commitments and
observed cards, canonical private bytes, the exact 12-unit/96-row independent
replay and stage exit, the reviewed scientific-reader blobs, known public
exposure inventories, source revision, at least 2 GiB free space and all seven
checkpoint assets. It retokenizes all 96 rows and requires the complete saved
selection to match. It records the actual CPU/runtime configuration, loads the
pinned model offline in float32/eager/evaluation mode, and calls
`run_cpu_core` once. It checks source, assets and custody again afterward.
The 8 MiB custody allocation includes both stage logs, the bounded child
logs, observation files and other custodian receipts; the custodian accounts
for any external files not supplied to the owner.

`complete_candidate` is provisional. The separate reader must replay the
closed journal, full outputs, landings, controls and fixed four-unit rule,
and the operator/custodian must record the actual process exits. A receipt or
source hash does not authenticate execution, native libraries, secrecy,
independence or the complete exposure history. Preserve every failed screen,
unmeasured group and partial attempt; do not automatically retry.
