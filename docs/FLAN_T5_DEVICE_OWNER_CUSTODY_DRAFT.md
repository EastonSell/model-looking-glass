# FLAN-T5 device owner custody draft

This is a CPU-tested owner and custody building block, not a controller, source
or execution attestation, resource profile, or authorization to use a GPU. Its
dedicated child requires an existing supervised session and physical-device
lease before importing model dependencies. A separately reviewed controller
must freeze its source and inputs, launch and reap the exact child, preserve all
partial evidence and handle uncertain ownership before this path can be used.

The observation producer binds each consumed lease to the exact custody object.
`RetainedObservation` accepts only that association, an active lease with the
fixed trace claims, and the exact returned host result and bit-view roots. It
keeps the owner thread and result rooted while evidence is persisted. After the
owner drops its roots, retirement requires the returned CPU tensor storages to
have no remaining PyTorch-visible aliases. Only then does it release the trace
claims and lease. An interruption during collection or release leaves the
release uncertain and forbids retry in that owner; the supervisor must retain
the process and its accounting until recovery or replacement.

This is a cooperative reference-lifetime check, not a proof about arbitrary
native pointers, external storage owners, allocator pools, host RSS, or CUDA
work. A caller must separately account for such aliases, other copied data and
model/forward allocations. The caller must keep the original custody and lease
alive throughout an unsuccessful observation as well; this class only manages a
completed host result. Its owner check returns `None` on success and raises on
failure. The bound owner must prevent concurrent mutation and early direct
release of its claims; same-process adversarial code is outside the contract.

The single-use CUDA owner binds the runtime UUID before setting its allocator
cap, stages the seven pinned checkpoint files, checks the eight-layer topology
and model-state fingerprint, and reserves a separate model/forward commitment
plus at most three observation claims. It checks numerical flags and the state
witness around the forwards. Failures leave the owner, exception, model and
possible observation roots retained; missing or partial stage records are not
cleanup evidence. Only a fully retired set of host observations can enter model
cleanup. A successful owner cleanup synchronizes, clears the supported cuBLAS
workspace and PyTorch cache, checks zero current allocator counters, and then
releases the logical model commitment. A failure or interruption keeps the
owner uncertain, with no automatic retry.

The GPU allocator cap and synchronized process-local peaks do not bound CUDA
context or native allocations, other processes, host RSS, or memory released
after process exit. The controller must separately retain exact-child exit,
supervisor and Job-drain evidence and device-wide observations. Rechecked
source and dependency entry-file hashes describe observed files, not a proof of
every executed module or native binary. The dedicated owner and its callbacks
remain trusted and cooperative; arbitrary native-return and process failure
gaps require process-level recovery.
