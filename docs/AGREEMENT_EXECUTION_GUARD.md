# Agreement execution configuration guard

The natural-patching runner snapshots selected non-tensor execution settings
under its model lease and compares them before and after each forward and at
cleanup. These include GPT-2 attention scaling, layer index, head dimensions,
causality and eager-attention settings, output/cache configuration, dropout
and layer-normalization parameters, and model-parallel flags. It records the
observed values and their SHA-256 in the run contract. A changed or unreadable
configuration aborts the attempt; if work has started, the model remains
quarantined rather than becoming available for another lease.

The same comparisons observe deterministic-algorithm and warning modes,
intra/inter-op thread counts, float32 matmul precision, and the raw result bits
of a small CPU underflow probe. The probe runs on the calling thread and does
not attest floating-point modes in other worker or BLAS threads. The guard reads
these settings; it does not reset them before comparing.

This complements tensor value, storage identity, module-tree, hook and mode
checks. It is a cooperative guard over explicitly listed settings, not a
complete proof of behavior or protection against hostile model code,
unobserved attributes, or a transient change that is restored between checks.
The controller still has to load the verified checkpoint and freeze source,
runtime and inputs before a study.
