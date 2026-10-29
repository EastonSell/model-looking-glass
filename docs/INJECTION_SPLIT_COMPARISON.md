# Injection and natural split comparison

The separate injection and natural predictions, split commitments and attempts
remain unchanged. The injection custodian may compare the committed original
natural split with a separately committed injection split using
`tools/injection_split_comparison.py`. It verifies the published freeze and
commitment bytes before opening either private record, verifies both complete
records against their commitments, and checks the full token rows in both
prefix directions. Its exclusive, bounded receipt contains only commitment,
freeze, inventory and comparator digests, case counts, and `clear` or
`collision`. A collision stops the affected later evaluation without a redraw;
it does not retroactively invalidate an independent completed attempt.

The study entry fixes the original natural freeze and its published held-out
commitment, the injection freeze and the existing site commitment. The exact
injection commitment URL and digest are bound by the operator after that
separate commitment has been published. The private site nonce, site record,
split nonces and split records are path-only inputs and never appear in the
receipt or normal diagnostics. The original natural split remains in scope
because its first-forward announcement makes it possibly exposed even if its
owner later stopped.

This caller covers that original natural commitment only. Any separately
committed or possibly exposed natural successor must also be accounted for by
the later injection owner. Its state and the injection forward decision must be
inspected under the same shared gate lock so a successor cannot appear between
the check and the forward announcement. An incomplete attempt whose rows cannot
be established stays uncertain; an absent peer or an unchecked result is not a
clear comparison. This caller does not announce a forward or implement that
shared-gate transition.

The source-bound operator must independently establish the protected paths and
ACLs, exact configuration bytes, current peer state and available phase budget.
It launches a direct-base isolated `-I -S -B` interpreter with a fresh empty
bytecode prefix and the gated start token, after verifying this caller's exact
published source before execution. The caller verifies every dependency against
its pinned SHA-256 and immutable public bytes before running project code, then
checks all four public freeze/commitment files before reading private inputs.
Only an independently observed successful child exit and cleanup, plus the
retained exact receipt, may be used by the separately reviewed gate caller.
Receipt existence alone does not establish execution, timing or publication.

The `--validate-sources` command uses the same bounded configuration and
source-loading path but does not open private split or site files, claim a
phase, draw a nonce or load a model. It is a startup check, not a private
comparison. `--compare` writes a new receipt exclusively, up to 8 KiB. Partial
or failed files are retained and charged within the existing injection held-out
physical share; no prior file is overwritten. The operator preserves and
accounts for any incomplete execution and does not infer a clear result from it.

`tools/injection_comparison_observation.py` adapts the reviewed outer observer
to the direct comparator child. It fixes the comparison caller's published
source and constructs the isolated hash-before-execution command, binds the
configuration and operation, and verifies the resulting receipt against both
published public commitments after the child exits and the Job is drained.
Both `clear` and `collision` can be observed; only `clear` can ever support a
later gate decision. Its trusted operator must source-verify the adapter and
its dependencies, protect the actual paths and leases, and independently
observe the adapter process's own clean exit. The observer's saved record is
not self-authenticating and neither record announces a model forward.

`tools/injection_comparison_exit.py` supplies the separate operator terminal
and outer-observation handoff. It binds the operator's native Windows PID and
creation time to the exact configuration, operation, comparator command,
inner observation and receipt. The outer observer must retain and reap that
operator process, check its terminal against the observed identity, and
verify Job cleanup. The configured receipt path must match the fresh path used
by both observations. The source-bound caller still has to establish the
meaning of the operation, public receipt verification and protected source
leases; a saved terminal alone is not an exit observation. Use the direct base
Python interpreter so the observed PID is the process writing the terminal.
Its public receipt verifier checks both commitment readbacks and the complete
receipt identities without opening either private record. A verified collision
remains a negative result and cannot authorize a forward.
