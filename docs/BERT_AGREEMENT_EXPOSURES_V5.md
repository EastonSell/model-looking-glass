# DistilBERT agreement exposure status v5

This append-only snapshot preserves the pre-evaluation v4 snapshot, SHA-256
`1f9ae67040646341097414f5ef6da1f66d0ccc5f0be47deeee36cfe66550e32e`.
The v5 canonical payload SHA-256 is
`34db577a211d6036e7803b8cdb6aaab35687b56974de8cf1ac6bccbf3272fb3c`.
It adds the completed CPU evaluation to all 64 public evaluation rows whose
exact input IDs and masks appear in the completed-forward records. The 32
development rows and all earlier evidence entries remain unchanged: ten
development rows are observed in cited evidence and 22 remain reserved. The
64 evaluation rows are now **observed in cited evidence**, not unexposed
held-out inputs for a later study.

The evaluation used source `4ee50069464228798071218b58b64221ac5960f4` and
run `ba65abf8bbf7680ba28d2a2ebea67d45`. The published evidence commit is
`ada1b7746a5541af2e6addbc943c08a6b737e0b4`, under
`results/bert-agreement-cpu-evaluation-v1-20261029/`. The complete terminal
SHA-256 is `7637da956e49bbae690cfea45debe7fb89f6bb600ee85a75dbb28223553e48d5`;
the compressed journal SHA-256 is
`ad791c45388d6d8a5ee34c6710dd9968760c7fc17007fa3609a24fe0a00f9c01` and
its uncompressed 9,490,808 bytes have SHA-256
`f30b6ae11e4fbf5276d9f5a5dc3ae5206fd2a3c9f2fdf4252aae0a844fb6f7ee`.
The saved journal records 1,120 started and 1,120 completed forwards plus 384
passing controls. The owner and controller are recorded reaped with zero
exit; the outer launcher exit 0 was observed in-session rather than saved as a
separate original file. The saved records and these local observations are
not independent attestation that execution occurred as described.

The 64 rows are public and nonblind. The pre-evaluation ledger and v4 snapshot
cover cited project records, not every possible local or external exposure or
pretraining. Their prior designation does not certify complete history, and
the result does not establish semantic independence or generalization. This
overlay is evidence accounting, not authorization to rerun the inputs. No
model was loaded or forwarded to create or verify the v5 snapshot.
