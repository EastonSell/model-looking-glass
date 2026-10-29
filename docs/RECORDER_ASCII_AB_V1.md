# Paired CPU comparison of recorder string checks

This preregistered experiment compares two exact recorder implementations in
one Python process on synthetic metadata. It performs no neural-network or GPU
work. Its purpose is to test whether the reviewed ASCII string-check change
affects this fixed recorder workload while reducing the session drift present
in two separate earlier runs. It does not measure contended service or GPU
throughput.

The baseline source is `looking_glass/batched_recorder.py` at commit
`f3c6cbb2211f04b46ff5b25dcde9bf251558dc41`, SHA-256
`d557159a7650a5cb4ad5f4a0354570edba04be6c3666f01a6f76421e12ef62f9`.
The fast-path source is the same file at commit
`23a19abb75248474151c0e84af29016d71bd334f`, SHA-256
`8e241c8b6670539ee9d130dea587220fe3132c3452330605e612cc26c1eb53cd`.
The program retrieves each exact Git blob, verifies both hashes, and rejects
any difference beyond the preregistered ASCII predicate and its comment.
It compiles the verified bytes directly into distinct module namespaces,
without replacing the production recorder or sharing recorder instances.
Both use the same loaded event-schema object, whose source must equal the
identical schema blob in the two commits, SHA-256
`79cc10978caca7a026824be35ddf01013be601b94bce5afcd3303a361d83bffc`.
The source and namespace identities are saved. Dependencies and interpreter
state remain trusted; matching source bytes are not in-memory attestation.
The loaded schema mapping's canonical SHA-256 is
`4e68e27b8a9a07bb20ac7dda36a33ecb15d107759eca3ebe924aeb3018ab779f`;
the two timestamp-free templates have canonical SHA-256
`a592f928a20404ec83a88c6e37bea514da0f0245e74ca21a38494885d9994083`.
Both are checked before and after the measured blocks.

Each variant warms on 200 events. Eleven paired blocks then each record 400
events per variant, alternating the same synthetic `selected` and two-member
`forward_entry` templates used in the original CPU protocol. Six blocks are
baseline-first and five fast-first; `Random(41)` shuffles the allocation.
Recorder construction, the independent expected-payload copy, snapshot, payload
equality checks and artifact writing are outside timed windows. Both variants
read `perf_counter_ns`, append each
source timestamp to an in-memory list, and construct identical event payloads
inside the window. That timestamp-list cost is included symmetrically in both
arms. Every stored event is checked after the window against its separately
retained original timestamp and payload. The two timestamp sequences are saved
for independent comparison, and a backwards source sequence invalidates the
window. The full stored payloads are not retained, so
their equality is the runner's check rather than independently replayable
original evidence. A lost event, differing
payload or incomplete buffer invalidates the run; completed windows and a
bounded loss report are preserved on a best-effort failure path.

The saved record retains all 22 raw start/end/duration samples, sample order,
source and recorded timestamp sequences, event counts, GC counter snapshots and
paired baseline-minus-fast differences
per event, including negative differences. Each complete window must be at
least 100 times the nominal clock resolution, and the clock must report
monotonic, nonadjustable behavior. This is a warning screen, not a guarantee of
accuracy or scheduler isolation. GC enabled state and thresholds are recorded,
and snapshots immediately outside each window may include boundary effects.
Interpreter specialization, caches, allocator and GC carryover, host frequency,
affinity and other CPU activity remain uncontrolled. Eleven blocks on one
machine characterize only this workload and do not establish a population
distribution or broad service speedup.

Run from a clean checkout with a newly created empty absolute bytecode prefix,
CUDA visibility set to the empty string, and an output path outside the
checkout, as in [RECORDER_OVERHEAD_V1.md](RECORDER_OVERHEAD_V1.md), substituting
`-m looking_glass.recorder_ab`. The controller checks the clean checkout and
source hashes before and after measurement, explicitly including the A/B runner
when it executes as `__main__`, records the runtime and PyTorch's
uninitialized CUDA process state, and publishes a terminal JSON atomically
without overwrite. A clock-screen failure is retained with nonzero status.
An error or interruption best-effort saves the completed windows and active
phase, but a killed process or failed publication may leave no artifact. The
operator must ensure the prefix really was empty at launch and avoid concurrent
source/dependency mutation; startup hooks and dependency binaries are trusted.

The two recorders are compared within the same process, but this is still a
single-thread synthetic experiment. It does not establish real model, GPU or
multi-client overhead and must not be subtracted from a throughput result. A
future model experiment must measure instrumentation effects separately for
serial and grouped modes and retain actual batch sizes and failures.
