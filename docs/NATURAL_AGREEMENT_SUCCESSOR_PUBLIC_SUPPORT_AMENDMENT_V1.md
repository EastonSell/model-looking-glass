# Natural-agreement successor: public-support amendment

This prospective amendment replaces the single private 256-by-16 old-split
query in the [successor protocol](NATURAL_AGREEMENT_SUCCESSOR_V1.md) and its
[query clarification](NATURAL_AGREEMENT_SUCCESSOR_QUERY_CLARIFICATION_V1.md)
with the fixed public-support check below. No successor private pool query,
phase claim, nonce or split was made before this amendment. The previous
natural-heldout attempt remains incomplete: its consumed claim, sealed split,
public commitment and possibly exposed forward announcement are unchanged.

The original published sampler at source commit
`def07cbb168408a12c76ba30f196af45149e111d` has eight noun pairs and eight
contexts. Each selected full prompt is `The {subject} {context}`. The recorded
original preparation tokenized those prompts with the pinned tokenizer, without
padding, truncation or special tokens; if invoked, the published runner would
consume the resulting full input IDs. All 16 noun forms crossed with all eight contexts therefore
give a **conservative 128-row public superset** before the sampler's balancing,
structural and exposure filters. Some combinations may be unreachable; no
membership in the sealed split is identified. The sampler SHA-256 is
`635f25553c20dd437f5cc4a8a9ca3a7a1453e7f8a517875fe75eb01ede2d5f35`, the
pinned tokenizer SHA-256 is
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`, and the
public [support file](../experiments/natural_agreement_original_public_support_v1.json)
has SHA-256 `5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6`.

The proof is conditional on the recorded original preparation using those
published sampler and tokenizer bytes, on the source-bound preparation
records, and on cooperative preservation of the sealed split. The public
support does not independently inspect or authenticate the actual private
record, establish what the model encountered in pretraining, or prove that the
failed old owner made no forwards.

Before a successor phase claim or model exposure, compare all 128 public
support rows with all 256 fixed successor candidate inputs in both complete
row-prefix directions: a pair collides if either entire row is a prefix of
the other. Retokenize both public sets from the pinned assets, verify all fixed
source and input digests, and independently review the method and result. A
collision or unavailable/uncertain verification stops this fixed attempt;
there is no model-dependent choice, adaptive pool revision or redraw. Since
every selected successor row must be a member of the exact 256-row candidate
file, a clear result separates every selectable successor input from the
conservative original support under the stated preparation assumptions.

The reproducible [public auditor](../tools/audit_natural_successor_public_support.py)
has SHA-256 `005a1a3ce967fc7d254b89800f4099dfbf3f966a972ed018f31d7e885a5db554`.
Its [comparison receipt](../results/distilgpt2-natural-agreement-v1-20261028/successor-public-support-comparison-v1.json)
has SHA-256 `262a3889fc4950de4eca39a7705a9028f248f19fdc0dd3b76370ce691638dc91`
and records 128 × 256 rows, zero relations in either direction, and `clear`.
The 256-input file remains SHA-256
`6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4`; the
pool, prediction, selection algorithm, family count and no-redraw conditions
remain fixed. This public computation reads no private split or nonce and
adds no new old-split membership query to the eight earlier recorded
invocations. Their prior aggregate disclosures and timing/failure limits
remain as documented.

Bind the published amendment and public receipt in the separate successor
operation. The successor record's existing `pool_clearance_sha256` field now
binds this public receipt, and its `protocol_sha256` continues to bind the
original frozen successor protocol; the operation also binds this amendment.
The claim and execution components that expect a private-query claim or the
old receipt schema must be updated and independently reviewed before use.
No private old-split subset recheck is required: instead, verify the selected
rows are members of the exact checked 256-row public file. The new phase still
needs its own one-use claim, fresh nonce, public commitment, exact
production-path no-model rehearsal, current reviewed exposure checks,
cross-track gate and independently audited saved evidence as specified in the
original protocol. Latest-inventory and peer-state observations remain
required at the stated boundaries; this fixed support comparison is not a
claim about unknown or later exposures.
