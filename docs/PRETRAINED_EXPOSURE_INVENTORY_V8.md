# Known DistilGPT2 development exposures: inventory v8

This immutable successor preserves the v7 predecessor snapshot with canonical
SHA-256 `e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344`.
The v8 canonical payload SHA-256 is
`22b1392ed43568fcac574b55ae6cf0212766771d8439ac39e599f7aa112b298f`.
It retains the same 138 records and 962 known or reserved nonempty prefixes.
The categories are 104 saved observations, three operator-reported
observations, 19 planned, four reference-only and eight synthetic-only.
All 32 public copying evaluation rows reserved by v7 are now recorded as
observed; the other planned records remain planned. No new cases are added.

The [public copying-head evaluation](../results/distilgpt2-copy-head-eval-v1-20261028/README.md)
used the pinned checkpoint on CPU, with 32 distinct rows and five fixed
actions per row. The saved evidence records 160 submitted and returned
actions, a complete result, and a direct-process exit observation of zero.
It is public exploratory evidence, not a blinded held-out study. The immutable
evidence commit is `ebe5caaae8394d210265ab65ab84c295b6463580`:

| Evidence | SHA-256 |
| --- | --- |
| [Manifest](../results/distilgpt2-copy-head-eval-v1-20261028/run/manifest.json) | `0398bd998fae4ad8ef473264fca4c8b5dcf2c63424e086e4f8c989a2f7cb819b` |
| [Result](../results/distilgpt2-copy-head-eval-v1-20261028/run/result.json) | `e36732e0538f7b5a069fed337fd86eb725744663b3801d1d953907d0edb93a02` |
| [Attempt log](../results/distilgpt2-copy-head-eval-v1-20261028/run/attempt.jsonl) | `88026a17a65ddf7d303479bceba120517b456b90da3fefd0b5c6dd1090d26766` |
| [Process observation](../results/distilgpt2-copy-head-eval-v1-20261028/supervisor.json) | `2935b0dfbecf10a8171cd62d05a0856f86994c7cce975d9090022a2d74610c9a` |

The manifest binds the v7 inventory and fixed public case inventory. The
attempt log records start, payload-recorded and completion events for each
action, and the receipt records the process exit; hashes establish saved-byte
consistency, not independent proof of faithful execution. The earlier v7
snapshot remains unchanged and accurately records the status before these
forwards. This successor makes no claim about other planned or unseen cases.

The module `looking_glass.pretrained_exposures_v8` exposes the canonical bytes,
digest and the same bounded known-split checks as v7. Checks reject a candidate
full row that equals a known or reserved nonempty prefix. They do not
automatically reject a longer candidate extending a shorter known row, or rule
out unknown history, semantic overlap or pretraining. A check is not a complete
exposure-history certification and does not authorize a study seal or model
forward. A trusted custodian must bind the actual checkpoint and tokenizer,
the current independently reviewed snapshot and immutable candidate bytes.
Keep private inputs, exceptions and tracebacks private. No private or hidden
held-out case was generated, and no model forward was run, while preparing
this snapshot.
