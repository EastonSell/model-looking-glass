# DistilGPT2 development exposure inventory v16

This successor reserves the 48 already-public, unpadded year-transfer inputs
before a model forward on those cases. The rows are `planned`; no forward is
asserted for them. It preserves all 359 v15 records and their annotations,
including the greater-than inputs, pairs/dose reservation and projected-edit
reservation. The inherited pairs/dose and projected-edit labels describe the
earlier v15 snapshot: their owners have since reported completed attempts,
which can be reconciled in later evidence-bound status updates. Known input
prefixes grow from 2,959 to 3,247.

The canonical module is `looking_glass.pretrained_exposures_v16`. Its v15
predecessor payload SHA-256 is
`6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34`;
the v16 canonical payload SHA-256 is
`275a16d49a386b1de247bff0f9031e92105d1b76f2493352090bf27c37d58843`.
The resulting 407 records include the 48 new planned cases. The module exports
`YEAR_TRANSFER_CASES` and `YEAR_TRANSFER_ROWS` in the published case order;
each record ID is `planned-` followed by its exact public case ID.

The year-transfer protocol was fixed at
`38ffcab52256e3952315b7bd9b58d6f10d7921c2`, SHA-256
`6f31697b09b2b68bf701494b4ceedde185b9f8501aea61caf50b647edac1fee7`.
The 48-case file was merged at `421f008a03f94da27e737d08040413265c19cbdd`,
SHA-256 `d6177125f8ac835676eebe59c5f65e379cb20d1d782eeeb17edf2e8700ac23d5`.
Its model, tokenizer, protocol and v15 construction pins are retained in the
public file. The reservation records exactly its input IDs and texts, not its
candidate continuation IDs; 16 inputs each have lengths 12, 13 and 15.

The new complete inputs are checked against prior complete inputs and their
recorded target-appended rows in both prefix directions, and against each
other. Reused wording creates shared *short* prefixes; those remain in the
forbidden prefix set. The separately published case inventory and these
checks do not establish that a private held-out split is disjoint. Apply any
required protected-split check through its custodian before a forward.

The predecessor digest check hashes one constructed predecessor payload; it
does not cache mutable inventory data. These source and saved-byte checks do
not attest execution or establish a complete history of prior exposure,
including pretraining and unrecorded private use. Known-prefix checking alone
does not authorize a study seal.
