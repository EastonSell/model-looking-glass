# Execution supplement for BERT agreement suffix protocol v1

This supplement specifies the software and evidence contract for the
[frozen protocol](BERT_AGREEMENT_SUFFIX_V1.md); it does not change its cases,
interventions, outcome definitions or descriptive criteria. It must be
source-bound before a model forward.

The owner verifies the source commit and checkpoint/tokenizer asset bytes
before loading model weights. The operator selects a trusted, absolute Git
executable outside the checkout; its file hash is recorded and checked again
afterward. Replacement objects and global/system Git configuration are
disabled for source reads, and the identifier must name a commit. The source
and helper are checked before dependency imports and again before completion.
The checkout root is not added to the import path. Python, installed
dependencies, operating system, Git executable and model framework remain
trusted components; these checks are not a sandbox or independent proof of
which machine code executed.

The loader requires empty **lists** for missing, unexpected and mismatched
keys and load errors; it retains these diagnostics. It enables deterministic
algorithms, disables TF32, requests the one-thread and denormal policy, and
rechecks the calling thread's settings after forwards. Full output finiteness
is reported by the owner, while the saved logits cover only the selected mask
position. Per-forward starts, returned logits, before/after site vectors,
memory samples and control geometry are retained under the protocol's
128 MiB attempt-output cap. The model's weak reference must be dead before
the owner writes its completion record; for CUDA, allocated/reserved and
driver-reported memory observations are sampled with synchronization. These
observations do not prove process-wide or driver-wide cleanup.

The included direct-child observer is a **CPU fallback**, not a shared GPU
owner. It admits a run only when at least 1 GiB is observed free on each
output volume and 2 GiB of physical memory is available. These conservative
point-in-time thresholds cover a 128 MiB capped attempt and leave room for a
single approximately 268 MB checkpoint and runtime overhead; they are not a
reservation or a proven peak bound. Temporary files and retained evidence
are directed to the chosen output volume. A GPU execution needs a separately
coordinated, source-bound owner and a fresh under-lease memory and storage
admission; a free-memory snapshot alone is not a lease. No unrelated GPU work
is blocked by a CPU run.

The direct-child observer records a successful exit only for a reaped child
whose source, metadata and completion record remain consistent. A lost spawn
handle or incomplete attempt does not justify a retry until the child state
has been reconciled. The saved-output reader requires an independently supplied
exit-receipt hash and completion hash, checks all 416 starts, returned events,
memory samples, selected logits, donor vectors, doses and control landings,
and fails closed on a failure marker or unresolved lifecycle. Its numerical
and source checks describe retained bytes and cooperative observations; they
do not establish private exposure history or unsaved model states.
