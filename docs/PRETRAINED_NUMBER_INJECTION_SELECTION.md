# Site-blind injection discovery scoring

`looking_glass.pretrained_number_selection` scores the eight **public**
first-lemma singular/plural discovery prompts from
`PRETRAINED_NUMBER_PATCH_V1.md`. It does not run a model, choose or accept the
hidden injection layer, draw or inspect a held-out case, publish anything, or
authorize a study. The caller supplies immutable `DiscoveryCase`,
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

Clean correctness uses the fixed present-tense sign rule. A zero margin is
incorrect and is also included in the separately reported tie count; the tie
count is therefore a subset of the incorrect count.

Failures raise `DiscoveryInvalid` rather than yielding a partial successful
freeze. The caller must record an incomplete attempt and retain all available
raw observations, failed controls and cleanup evidence; tracebacks and locals
must remain in custody. This in-process helper does not enforce a global
no-retry policy or prove that caller-supplied attestations are true. Its input
and output bounds are not a host-memory peak or GPU admission limit. The
fabricated tests do not run the model or create any held-out input.
