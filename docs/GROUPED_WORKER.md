# Opt-in grouped resident execution

Status: an in-process, trusted-owner primitive. CPU tests cover custody,
cancellation and failure behavior. This is not an enabled GPU service, a
measurement of throughput or proof of model batch separability. The existing
serial path remains independent.

`GroupedWorker` is configured with exact immutable `BatchProfile` records,
one `BudgetedWorker`, one authoritative `HostBudget`, and service-owned byte
and item bounds for the finite-check scratch of each profile. The checkpoint,
model/runtime fingerprint, eval/cache/autocast policy, separability evidence,
input-dependent worst case and scratch envelope must be verified by the trusted
loader and an appropriate versioned measurement protocol; labels and this code
alone do not prove them. Exact combined rows, tensor names, dtypes and shapes
must also match a configured whole-batch `WorkloadProfile`. The device budget
samples current telemetry and takes one lease for the full batch.

Before `prepare`, the owner reserves a host claim from the same active request
and ledger, of at least `required_host_capacity(profile_id)` bytes and items.
The declared envelope covers five overlapping logical input copies, the
resident worker's maximum full output plus split outputs, and the separately
configured finite-check scratch bound. This is cooperative accounting: model
outputs/activations and unprofiled runtime temporaries, Python metadata,
allocator overhead, client processes, and OS RSS require separate controls.
The owner must retain separately reserved source/input custody from before
snapshotting. Group inputs must have exact, contiguous CPU storage with a
zero offset, and IDs are bounded; externally held aliases still require trusted
ownership and cannot be discovered or disposed by the ledger.

`prepare` binds the exact claim once, pins it with a host operation, then
copies inputs outside the ledger lock. An interrupted preparation stays in the
ledger-scoped, bounded `active_groups()` recovery registry; its handle must
not be disposed while construction is in flight. `recover_preparation()` is a
trusted attestation that interrupted construction and all related aliases or
tracebacks have stopped, not an automatic proof. It permits subsequent
explicit disposal. A regular caller can abandon a definitely unlaunched group
via `abort_unexecuted(error)`; the broker owns per-member terminalization if
there is no issued launch.

`launch` assembles a plan-owned capability, recording input digests before it
is returned. The owner must prevent mutation until the worker snapshot completes;
PyTorch version checks alone cannot detect every alias mutation. The worker
checks the exact launch object, plan identity, profile and snapshot digests
before its one forward. Digest checks detect accidental changes but are not a
security boundary against concurrent malicious mutation or arbitrary model code.

During execution the host claim stays pinned; the single whole-batch device
lease stays active through synchronization, CPU result validation and fixed
`Microbatch.finish` row splitting. Requests cannot provide callbacks. A
postlaunch failure or invalid/split result quarantines admission and retains the
lease and host group; replace the process and reconcile external custody after
its exit. Private `Microbatch` records may label a member cancelled even when
the same batch failed or ran out of memory. The broker must give the whole-batch
failure precedence for every member and cannot publish that private cancelled
record as evidence of a successful batch. Do not retry a smaller batch
implicitly. Individual cancellation is
recorded on the plan and never passed as the shared worker cancellation event.
The optional execution event is for the whole group only.

A cancelled host request is observed before launch and before dispatch; one
observed after the forward suppresses its rows. These are observations, not an
atomic broker-publication decision: a cancellation accepted after that final
check can return while private successful rows are still being produced. The broker
must arbitrate each member's cancellation versus publication under its own
lock, retain the group claim through every plan, launch, output, transport and
consumer alias, and dispose suppressed results under that custody.

After all covered aliases and retained exceptions are actually gone, a trusted
owner calls `group.dispose()` to finish the operation and release its claim.
This is an explicit cooperative attestation; the code cannot verify discarded
Python aliases. The owner then closes the host request. An uncertain group
cannot be disposed in process. Arbitrary asynchronous interruption can strand
Python locks or obscure an operation's completion; the supervisor must treat an
unresponsive owner as uncertain and replace its process.
