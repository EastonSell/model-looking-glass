# Natural agreement held-out adapter

This CPU-only adapter connects an already committed number-study held-out
record to the existing natural agreement runner. Its tests use fabricated
families and a synthetic model; they are not a pretrained discovery or held-out
result. The same reviewed source must be bound before the natural discovery
forwards and used for that track's later evaluation.

The trusted controller publishes a canonical natural prediction file with
exactly `schema` (`mlg-number-natural-predictions-v1`), `protocol_sha256`,
`source_commit`, `checkpoint_revision`, `selection`, `present_site_metrics`,
and `secondary_past_site_metrics`. The last two are the ordered eight event
objects from the audited discovery journal, including all per-family and
target/donor records. `selection` is the discovery summary. JSON uses sorted
keys, ASCII escapes, compact separators, finite numbers and no trailing
newline. The aggregate freeze contains the SHA-256 of those exact bytes.

`run_committed_heldout` verifies the immutable prediction bytes against that
aggregate and the saved selection against the discovery metrics, rechecks the
held-out record's commitment and selected inventory, and maps the four ordered
cells in each committed family to the corresponding runner cases. Before the
first forward, the private run journal records the held-out commitment,
selected inventory digest, natural prediction digest and aggregate digest.
The explicit held-out mode performs the same candidate patches and controls as
discovery but does not compute a held-out selection or magnitude ranking. It
retains each family's effects and compares each of the eight frozen predictions
to four family values, reporting MAE and RMSE alongside the zero predictor.
The selected, earliest, random and magnitude sites are copied from discovery;
abstention and coincident baseline sites remain unchanged. Clean margins and
accuracy, full candidate effects, and secondary was/were measurements remain
in the private evidence.

The controller must independently verify the checkpoint, source and runtime,
tokenization and appended continuation alignment, owner process, disclosure,
current reviewed inventory, and durable publication of both the freeze and
held-out commitment before invoking this adapter. The provided current digest
is an assertion by that trusted controller; no local hash proves it is the
latest reviewed snapshot. A verified commitment also does not establish honest
generation or publication time. Do not redraw or alter failed cases. The
controller must preserve failed attempts and enforce the protocol's cumulative
study storage limit across tracks. A validation failure before runner entry
leaves the freshly created journal incomplete; the controller must record it.

Records, nonces, journals, exception objects, tracebacks, local frames, logs and
other diagnostics are private until the protocol's reveal. This is procedural
blinding in a cooperative same-account process, not a security boundary. The
adapter requires a trusted caller and cannot prevent transient concurrent
mutation, hostile hooks, unintended exposure, or other same-process access.
The low-level runner can be called directly, so the controller must enforce
this verified adapter and full phase order rather than interpreting a candidate
terminal record as authoritative experiment completion.
