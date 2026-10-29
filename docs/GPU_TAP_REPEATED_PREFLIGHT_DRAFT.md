# Repeated tap preflight, disabled draft

`tap_repeated_preflight._FixedPreflight` executes the four fixed paths using
the separately disabled owner and timing seams. Its CPU tests use a real tiny
GPT-2 and fake device operations. They do not establish the pinned checkpoint,
physical-device binding, parent grant, GPU behavior or a timing result.

Before the child announces readiness for the parent screen, preparation runs
the guarded synchronization, peak reset, memory check and first current-stream
observation. It makes no further device calls while waiting for the grant. The
first forward attempt checks a trusted, non-device start-gate callback over
an interval of at most 250 ms, rechecks the owner, and records its source
timestamp just before the Python model call. The gate must reject unless the
entire interval satisfies the initial ordering, freshness and expiry bounds. The
owner obtains a fresh CUDA free/allocated/reserved snapshot after the grant and
before that gate, within the enclosing guarded operation; it validates each
counter before the next query. Its source interval is recorded and at most two
seconds old at the actual call. A final owner check follows the gate, so a loss
observable during the snapshot stops the forward. This is a point sample, not
a physical memory reservation or atomic mid-call revocation protection.
future supervised owner must bind that gate to the exact one-use parent grant,
live owner, cross-process clock and fresh process/GPU observations. A timestamp
marks an invocation attempt; it does not prove the model body or a CUDA kernel
began. Failure or interruption requires process replacement, not a retry.
The gate must return `None` on success; a rejection or any other return stops
before the forward. CUDA timing rejects every sample until this exact timing
owner has completed one successful preflight.
The preflight row retains both authorized interval endpoints; the eventual
parent must validate them against the grant and the source-stamped start.
The widened interval is prospective and accommodates the observed cost of the
physical owner checks; the parent still imposes its shorter grant expiry and
fresh activity bounds. Scheduling can still pause the process after any check.

The preflight runs uncaptured, ID-only, direct and local gradient-reporting
paths, validates their actual output and trace metadata, and checks exact
`torch.equal` for all four outputs and for the two captured values. It retains
the four model results, four logits, three sessions and event containers, two
captures and two current-stream wrappers through the completion sync. After
individually checked drops it rejects surviving weak references, synchronizes
again and records allocator peaks, a final memory snapshot and its postcheck
time. A failed check preserves the known roots and admission, subject to the
documented single-fault and native-return limits of the underlying owner.

This is only one component of the real runner. The pinned model/input, source
identity, fixed parent barriers, durable journal, no-device-work screen wait,
direct hard exit, verified disposal and strict supervisor remain to be wired
and reviewed. The 128 KiB GPU trace claim is not a host-memory or transient
workspace bound.
