# FLAN-T5 multi-subtoken sentiment: fixed discovery selection

This record freezes the discovery selection for the separately published
[protocol](../../docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md). It contains no
model output. Two independent reviewers assessed the alphabetized, label-free
handoff before selection. All four fixed slots were filled: two with one
lexical subtoken and two with two; 60 of the 160 development-exposed candidates
were eligible after the fixed filter, semantic ratings and scoped checks. These
are tokenizer-exposed words and story/concert contexts, not lexical holdouts.

The [selection](discovery-selection.json) retains all 160 dispositions, both
original reviewer records, exact selected token records, fixed slots, and
source and receipt identities. Its SHA-256 is
`cf50b392c58704e7b673a076a2932991f05fbf6de6ae3e3e7ab01088a6b5c0a3`.
The [public prefix scan](discovery-public-prefix-scan.json), SHA-256
`72afb25fe32622144cc750c5983ec1fa35077e1c41fb6b1f062c9cd1f1df95d9`,
compared the 272 retokenized discovery rows against the explicitly named
public inventories. All 272 were clear within that scope; sources with no
recorded token rows are identified and the two released pools' screen coverage
was checked. The [own-custody receipt](own-custody-schema-v1.json), SHA-256
`4db3e1dda628e34e1b1b93408f88e28d7ebee6590537d8a7c397bcbb331e47c4`,
means only that its reviewer knew of no currently protected FLAN pool in that
reviewer's custody. No protected or flagged input was queried or compared for
that receipt. Unknown and private histories and pretraining exposure remain
unverified; see the [public source observation](public-source-currentness-observation.json).
An unpublished tokenizer-only weight-edit proposal with SHA-256
`af823e4a88f81944f3982cfd5d0af121e3cf2c4a86d9c09c66a61a4388e08ac3`
was reported after the scan and was not compared; its owner reported no model
forward. This is an additional known limitation of the scope observation.

The operator recorded a zero exit and readback for the selector child; the
exit-observation SHA-256 is
`297b000dbef9f6a7efcdad3cee45e8fee4445f1da4ac40eaecbef2589a65fdd6`.
The earlier first handoff attempt stopped before producing output because a
published Git blob was absent locally; its known incomplete record is retained
locally. Process and reviewer records are cooperative observations, not
independent attestations of chronology or identity.

After this discovery digest was sealed, a separate `painting` handoff was
created using only slots 0 and 2. Its ratings and tokenization are a distinct
follow-up and cannot alter discovery selection. Both manifests must be fixed
and published before the first model forward. There is no scientific outcome
in this record.
