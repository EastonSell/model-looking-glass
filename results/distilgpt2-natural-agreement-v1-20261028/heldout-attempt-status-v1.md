# Natural agreement held-out attempt: incomplete

The natural agreement held-out attempt stopped on 2026-10-29 without a
scientifically evaluable held-out result. The natural-only prediction freeze
(`35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb`)
and the v15 exposure snapshot
(`6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34`)
were fixed before the one-use natural held-out phase was claimed and its split
was prepared. The resulting salted commitment
`8b475c373034201e5c36186207edbc56f0c8b62221f497d8d09faad1ce182150` was
[published](https://github.com/EastonSell/model-looking-glass/blob/4d4c29683d58ea796c9bd14656d2e637bdfe0913/results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json)
before any held-out model work was attempted. The nonce and case rows remain
private. The phase claim remains consumed; this attempt will not be retried or
replaced by another draw under this protocol.

The retained parent observation records that the separately started owner
process exited with code 1 and was reaped. Its retained sanitized error records
`RuntimeError` and `quarantined: false`; it
does not include the original exception message or traceback. There is no
retained owner directory, owner-start marker, owner-execution supervisor receipt,
event journal or tensor file. The preparation-supervisor receipt is retained.
The reviewed supervisor creates the owner directory before it
spawns its child, which is consistent with a stop before child launch under the
cooperative artifact-retention assumptions. These records are not independent
proof that no model forward occurred. A durable first-forward announcement was
made before the owner started and remains a possible-exposure ordering point.
At that point the other split had not been allocated; the gate recorded it as
unchecked, not as a disjointness result.

The retained owner command used the reviewed source-hash bootstrap but did not
add the approved virtualenv site directory to `sys.path`. The frozen runtime's
startup check requires that directory exactly once. Independent no-model
replays of the source-bound startup functions under the recorded isolated
interpreter settings failed with a site-packages `RuntimeError` when the path
was absent and passed when it was added once. This is a source-supported
diagnosis, not the original recorded exception message. The earlier no-model
validation used a gated bootstrap that did add the site directory and therefore
did not exercise this exact owner invocation.

The 48 public year-transfer rows that were published before the split was
prepared were subsequently compared privately with all 16 sealed natural rows
in both complete-row prefix directions. The source-verified comparator
returned clear in a reaped process; its digest-only receipt has SHA-256
`ac0e30b30f91bd05ac73ef8750c0b7e21df9906ae6738c7e5e18d522c6baa65f` and
binds the published natural commitment and the public input file SHA-256
`d6177125f8ac835676eebe59c5f65e379cb20d1d782eeeb17edf2e8700ac23d5`.
This check does not reveal the protected rows or establish disjointness from
any separate injection split.

At the 2026-10-29 07:30 UTC accounting observation, the retained natural phase
and separately counted support files totalled 861,975 apparent bytes, including
the accounting record itself, with no retained event or tensor bytes. The
metadata-only accounting record has SHA-256
`3e68d587cecf47afb60b6fa11217c8c930dcd2613fb3d3d6530f7ed38118e97b`.
The total was within the existing held-out allowance at that observation.
The custodian observed that later public exposure versions were published after
the owner exit; they do not change the sealed v15 identity. The parent record
does not independently timestamp the exit. The discovery runtime did not record historical `MKL_CBWR` and
`ATEN_CPU_CAPABILITY` values, so equality of the other recorded runtime fields
would not by itself establish complete numerical-runtime equivalence.
