# DistilGPT2 number-study sampling and commitments

`looking_glass/pretrained_number_commitments.py` implements the canonical
record encodings and deterministic allocation in
`PRETRAINED_NUMBER_PATCH_V1.md`. It binds the exact protocol SHA-256 and
pinned checkpoint revision. The tests substitute fabricated noun/context
pools and fabricated nonces; they do not create an actual study split or
perform any model forward.

The helper generates **no secrets** and does not publish anything. Before
using a real nonce, the trusted custodian must independently publish each
participating track's complete prediction and runtime files and the exact
canonical aggregate, verify their file hashes, record their publication
order, and then create a fresh OS-random 32-byte split nonce once. For the
injection track, it must separately create and publish the site commitment
before injection discovery; its site nonce must be independently generated.
Never reroll a nonce to avoid a failed check. The cryptographic digest alone
does not prove publication time, faithful execution, confidentiality or
independent randomness.

`make_prediction_freeze` constructs the canonical aggregate from trusted
track records containing the source commit and the SHA-256 values of the
published runtime and prediction bytes. `load_prediction_freeze` validates
previously saved canonical bytes and their digest. The resulting immutable
`PredictionFreeze` records a consistent hash; it is **not** a publication
attestation. `sample_families` requires a valid freeze object and an
already-owned nonce. For an injection track it validates the private site
commitment, source binding and distinctness of nonces before any sampling.
It then applies the protocol's SHA-256 sort domains and byte-order tie break.
This ordering check cannot establish that another
process published the freeze or that the nonce was generated afterward.
The custodian must enforce and document those events independently.

`seal_site` derives the hidden injection layer and its private canonical
record from the site nonce and source revision. `verify_site_reveal`
recomputes the site commitment. Before sealing held-out cases for an
injection track, `seal_heldout` requires that private site material and
checks the source revision against the injection entry in the prediction
freeze. It rejects accidental equality of site and split nonces. Distinct
bytes are necessary but do not prove independent randomness.

After the custodian tokenizes the sampled cases, `seal_heldout` checks the
four-family structure, ordered texts, token bounds, all-one masks, subject
positions, within-family alignment and duplicate rows. It also compares the
exact 16 public discovery and 16 held-out rows with the reviewed v3 exposure
inventory before hashing the protocol's canonical held-out record.
`verify_heldout_reveal` revalidates the allocation, metadata, source and
freeze binding, exposure check, and commitment after disclosure. The current
implementation binds v3's inventory digest; a later inventory with new known
rows must be separately checked and bound before use. The comparison is a
known-prefix check with the limitations in the inventory notes, not proof
against unknown history, semantic overlap or pretraining.

These checks do not independently establish that text was tokenized by the
pinned tokenizer, that the subject index identifies the actual subject, or
that appending the four contrast strings preserves prefix tokenization. The
custodian must perform and record those protocol-required checks on the actual
assets. It must also bind the actual checkpoint, source, runtime and device,
and record partial failures. A source/checkpoint digest supplied to this
helper is not proof that it was executed.

The private record bytes and nonce belong inside the trusted custodian until
the protocol permits reveal. Object representations suppress private record
and family contents, and exception messages are generic, but exception
tracebacks and local variables may retain private data. Do not send those
objects or unfiltered logs to the selector. Caller-owned objects must remain
quiescent during the initial snapshot; this is procedural cooperation, not
isolation from same-account processes. Record sizes are bounded to one
million serialized bytes for this helper; this is not a host-memory or GPU
admission limit.
