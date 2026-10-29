# FLAN-T5 input-row change: fixed case scope

The 24-case tokenizer-only bundle has SHA-256
`a118dde671351d21c519b6a61957e9478c9501bac52d55aa34e8394772e933d5` and
binds the prospective protocol SHA-256
`c06fd598e6b5d9cf8d9734c26f6711fbf53b0fbfda08f009ab691ee91dca5478`
published by PR #592. It contains the eight explicitly reused, already
exposed discovery/calibration rows, four new target/donor pairs, and eight
new locality rows. This is the only tokenizer-only proposal in this lane;
there are no discarded rows. No new case was forwarded before this note.

The independent FLAN custodian's recorded-source comparison covers 593
EOS-ended rows from the eight exact public sources named in its receipt. The
16 new rows had no complete-row prefix relation in either direction after
removing one final EOS; the eight declared old rows matched previously
exposed records. This does not establish unknown-pretraining disjointness.
The pending separate sentiment pool is still reported unformed and must
incorporate this exact bundle and all its variable words in its exclusion
ledger before its draw; no private pool was inspected or queried. A fresh
scoped custody report and current relevant exposure sources are required at
each forward boundary.

Independent semantic review found the four positive target sentences and
their `awful` substitutions unambiguous in the intended review context, and
the fixed locality answers plausible. The four sentiment localities reuse
the corresponding target's instruction and subject, changing only the
adjective; they are paired diagnostics, not independent task samples.
Locality 2 is exactly the clean `excellent` text substitution of target 2,
so its relationship to the same-polarity row replacement is mechanical.
The two target-token-present localities test exact repetition and a simple
triangle question; the two remaining unrelated localities ask about square
sides and the opposite of hot. Expected strings are fixed exact-answer
criteria. For example, `three`, `four`, or `cool` can be reasonable answers
outside those criteria, and an exact-token mismatch is not automatically a
semantic or capability failure. Baseline correctness remains unobserved for
all 16 new inputs until the prospective run.

The fixed expected answers, complete token IDs, masks, offsets and EOS are in
the case file. The corresponding builder is model-free and uses the pinned
tokenizer; the complete checkpoint is not loaded. Scientific claims remain
limited to these wordings and controls, with the identity behavior for
target-token-absent inputs and the substitution nature of the edit as stated
in the protocol.
