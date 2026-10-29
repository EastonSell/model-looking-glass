# Site-blind injection scoring

`looking_glass.pretrained_number_selection` scores the eight **public**
first-lemma singular/plural discovery prompts from
`PRETRAINED_NUMBER_PATCH_V1.md`. It does not run a model, choose or accept the
hidden injection layer, draw a held-out case, publish anything, or authorize a
study. The caller supplies immutable `DiscoveryCase`,
`Observation`, `Boundary`, `Action` and `Donor` records. The executor's public
observation and action descriptors can be copied into these value types; raw
vectors and logits remain immutable little-endian float32 bytes.

Cases must be in the protocol's four-family, singular-then-plural order. Their
input digests must match the eight exact public token sequences. Every
observation needs a distinct opaque request ID and the same input digest for
its case. The selector validates all 13 ordered attention, MLP and final-block
boundaries, coordinates, shapes, device, dtype, finite vectors and full
50,257-element last-token logits. It checks candidate action and donor
identities against the corresponding clean or injected baseline and checks
post-action donor bits. It requires clean repeat and post-family clean traces,
null/self, upstream and downstream controls to pass the fixed tolerance.

The selector has no hidden action metadata, same-forward self-source snapshot,
or full pre/post witnesses. `private_controls_verified=True` is only the
trusted caller's attestation. The custodian must separately verify and retain
the actual injection addition and ordering, bit-exact self landing, untouched
batch/token coordinates, all private upstream checks and plan cleanup. The
selector can compare the self result with the injected reference; it cannot
prove that the self source was captured in that invocation. The caller must
also verify and bind the actual model-entry token IDs and all-one attention
mask, execution device, source, checkpoint, runtime and model state, enforce
single-owner execution and preserve the private site until reveal. An
input-digest field supplied to this selector is not proof of the actual model
input or mask. The caller must establish that each `verify_clean` observation
was actually taken after its family; identities alone do not prove execution
order. No checksum here proves faithful execution or secrecy from another
process using the same account.

The 32-hex request IDs are caller-supplied and become public in the prediction
record. Their syntax alone does not establish that they are opaque: an ID can
encode a hidden layer. Before execution the custodian must assign and verify
IDs from a site-independent precommitted request plan or independent randomness,
and check that public paths, observation ordering, error reports and other
released metadata do not encode the private site. This helper cannot establish
that blinding property from the supplied IDs or metadata.

`freeze_discovery` returns UTF-8 canonical JSON bytes with no newline. Its
per-case records include all observed votes, decoded norms, present- and
past-tense margins, forward and reverse effects, public trace/action metadata,
and SHA-256s of each raw vector and full logit payload. It reports the fixed
earliest and seeded random layers, the magnitude baseline, vote ranking,
abstention when all votes are zero, and the mean forward effect at each layer.
The two prompts from each family remain grouped; they are not independent
experimental units. Reverse effects are reported separately and are never
assumed to negate forward effects. A first observed difference describes these
specific taps; it does not identify the source of natural agreement behavior.

The caller must durably retain the actual raw bytes named by every hash and
publish them with the prediction file before constructing the aggregate
prediction freeze. Each hash addresses an immutable raw little-endian payload;
the role, request ID, boundary position and fixed payload length in this schema
identify its use. The prediction JSON does not embed the raw vectors or
logits. Independently recompute those hashes and verify that every referenced
payload is present. The function does not write an artifact or enforce that the
files, runtime record and aggregate were actually published before held-out
allocation.

## Evaluation after the prediction freeze

The same reviewed source must be published before injection discovery begins.
Only after the complete discovery prediction and aggregate are published may
the custodian draw the held-out split, bind and publish its commitment, and
execute the held-out forwards. `score_evaluation` is a pure scoring function:
it generates no cases and runs no model. Its eight `EvaluationCase` records
carry the custodian-supplied text and token tuple for the first lemma's
singular/plural prompts in each of four committed families. It checks the same
observation identities, action/donor coordinates, landing bits and public
controls as discovery; it records raw-payload hashes and all per-case outcomes.
It validates basic token shape and uniqueness but does not prove tokenizer
alignment, the full four-prompt family, inventory exclusion or a commitment's
preimage. The custodian must verify these against the committed held-out record
and the current exposure inventory, verify actual inputs/masks and all private
action/cleanup controls, and retain every raw payload and incomplete attempt.

The function takes the exact immutable discovery JSON bytes and checks their
SHA-256, canonical form, protocol/device identity, votes, rankings, means and
predictions before reading the evaluation observations. It uses only the
selections and site predictions in that record, including abstention; it never
reselects from held-out outcomes. It reports each forward and reverse effect,
the individual errors against the frozen and zero predictions, per-site MAE
and RMSE, and the four two-prompt family groupings. These are descriptive
measurements, not eight independent families or evidence of a natural circuit.
For each site the full last-token vocabulary restoration uses one `math.fsum`
over all squared decoded-logit differences across all cases and coordinates.
The numerator and denominator are always retained; the value is JSON `null`
when the denominator is at most `1e-8`, and negative values are preserved.

`prediction_sha256`, `prediction_freeze_sha256` and
`heldout_commitment_sha256` identify separate records. The caller must verify
that the exact prediction digest appears in the published aggregate, that the
aggregate predates the split draw, and that the published held-out commitment
predates all held-out forwards. `custody_verified=True` asserts those facts and
the correspondence of the supplied rows to the committed record; it is not
cryptographic proof of publication order or truthful execution.
`private_controls_verified=True` remains the separate attestation described
above. Evaluation results and case material remain private until protocol
reveal. `EvaluationInvalid` means no successful evaluation is produced; retain
the incomplete attempt and evidence according to the protocol.

Clean correctness uses the fixed present-tense sign rule. A zero margin is
incorrect and is also included in the separately reported tie count; the tie
count is therefore a subset of the incorrect count.

## Rank after site reveal

Publish `revealed_site_rank` with the rest of the analysis source before
discovery, but call it only after the protocol permits the site reveal. It
checks the exact immutable prediction bytes against their expected SHA-256 and
the frozen selection invariants, then returns the revealed layer's 1-based
position in that already-frozen ranking. It neither reranks nor changes a
prediction. Ties use the original lower-layer-first rule. An all-zero vote
ranking still has deterministic positions, but the original `first_changed`
remains `null`: report that abstention alongside the rank rather than calling
the position a successful localization.

The custodian must independently verify the site nonce and site commitment,
the published prediction digest and aggregate, and the timing of the reveal.
The lookup accepts a layer supplied by the caller; it cannot itself establish
those facts. Discovery continues to accept no hidden site.

Failures raise `DiscoveryInvalid` rather than yielding a partial successful
freeze. The caller must record an incomplete attempt and retain all available
raw observations, failed controls and cleanup evidence; tracebacks and locals
must remain in custody. This in-process helper does not enforce a global
no-retry policy or prove that caller-supplied attestations are true. Its input
and output bounds are not a host-memory peak or GPU admission limit. The
fabricated tests do not run the model or create any actual held-out input.
