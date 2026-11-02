# Projected study owner: cooperative scope

The prospective projected-study owner uses a trusted, quiescent local checkout
and checkpoint cache, a pinned pre-forward manifest, byte checks before and
after execution, and a supervised direct CPU child with a persistent study
root and a drained Windows Job. The source finder in `v2_source_bootstrap.py`
is reused only for checked project imports and platform setup. Its separate
guarded-start and inherited-source-handle contract is **not** claimed or used
by this owner. A source hash or a checked import does not authenticate native
libraries, code already executed, or bytes against a concurrent mutator.

Each process claims the same CPU study root before importing Torch, switches
the claimed empty CUDA mask to `-1` before importing model dependencies, and
records the actual one-thread runtime, checkpoint and restoration checks. A
point-in-time available-memory floor is an admission observation, not a memory
reservation or a measured peak. The loader verifies all checkpoint assets,
and the child rehashes them after work. The parent publishes a complete phase
record only after a clean zero-exit owner, whole-job drain, matching identity,
and source/manifest rechecks. A missing parent result or uncertain cleanup is
incomplete. Preserve failed and cancelled attempt directories and the study
root's sticky marker; do not retry automatically.

The manifest is committed and made publicly observable before any study
forward. It binds the source revision, fixed protocol and supplement, public
cases and continuations, seven checkpoint hashes, v1 model and weight hashes,
the current known-public exposure inventory, and scoped recent custodian
reports. The local clock and Git ancestry are operator attestations, not
trusted timestamps or proof of exhaustive prior exposure. Recheck current
inventory and custodian status immediately before each phase.
The current inventory must preserve every complete row from the frozen
snapshot and is compared in both prefix directions; a separately recorded
review identifies its committed source, version and digest. Its currentness
is still an operator observation. The historical Git path check treats Windows
case and trailing-dot/space aliases conservatively, but does not prove the
absence of 8.3 aliases or uncommitted historical files. Physical paths and the
checkout must remain ordinary and quiescent throughout each phase.

Development uses only the previously exposed discovery and calibration rows.
Publish its full candidate inventory and parent phase record, then obtain an
independent review of those exact published bytes before evaluation. The
evaluation child reconstructs and verifies all edits and saved development
measurements before any new case forward. Its numerical decision is
provisional until supervision, source, completeness and independent saved-logit
checks pass; failed or uncertain cleanup always has incomplete precedence.
The public cases are non-blind and the locality sample does not establish
general capability preservation or independence from pretraining.
