# Two-client synthetic CPU host-memory observation

This provisional observation exercised two real IPC clients against a separate
CPU owner process with a synthetic model. Both requests succeeded, and the two
fake forwards ran serially after both clients were admitted. It is planning
evidence, not a service admission profile or a worst-case memory bound.

In a separate final read after the periodic sample loop stopped, Windows
reported process-lifetime peak working sets of
450,191,360 bytes (429.34 MiB) for the owner and 437,305,344 bytes (417.05 MiB)
for the client process. Across 68 observations, the largest **sequentially
sampled sum** of the two current working sets was 866,799,616 bytes
(826.64 MiB). The largest observed gap between samples was 125 ms. The sum can
double-count shared pages, and sampling can miss transient or simultaneous
peaks; it is not the peak of a continuously monitored aggregate. The last
read reported private commit of 816,033,792 bytes for the owner and
820,494,336 bytes for the client process. Private commit and resident working
set measure different things.

The workload used the real resident/lifetime worker, broker, IPC server and
client paths, with two connections, at most two pending requests and one per
team. The synthetic CPU model returned deterministic finite fp32 outputs of
shape `1 × 2 × 50,257`; there were two fake forwards, no pretrained model or
checkpoint, and no CUDA context. Test-only contract flags enabled this fake
composition. Request frames were at most 233 bytes and successful response
frames at most 1,845,205 bytes, including the four-byte frame header. The
configured caps were 4 KiB per request and 4 MiB per response. A separate full
CPU test suite overlapped the observation; system availability was therefore
not isolated from background activity.

The owner used a direct base Python interpreter. Before requests, its reported
process identity and creation time matched the retained process handle. After
the periodic sampler stopped, the observer checked the creation time again
using that same handle. The observer recorded the owner exit as zero, and the
operator observed the controller exit as zero. The retained source record
names local commit `1417613fa7d7c80608e860b0552a246a624375b9`, which is not a
public commit. Its complete Git tree matches published
[PR #434's head](https://github.com/EastonSell/model-looking-glass/tree/04acc68578f007bc76c826de78287c70ad1e5984),
tree `414c68379f13be484234ad354b9e2fd4adfa0b7f`. The recorded pre/post checks
covered 246 source files; selected installed dependencies and the local
execution environment remain trusted assumptions.

The earlier v3 run is [explicitly superseded](superseded-v3.json): on Windows,
the virtual-environment launcher exposed a redirector process handle rather
than the Python owner handle. Its owner and aggregate working-set figures are
invalid and must not be used. The raw v4 and v3 records remain private because
they include process identifiers. Their hashes in [observation.json](observation.json)
are commitments to retained originals, not publicly reproducible evidence of
execution. The public projection contains the anonymous numeric samples and
can be checked with `python verify.py`.

The observer excludes other processes and transient source-check helpers from
its two-process sum. It does not measure the final service source, a real GPU
owner, model loading, or all possible IPC payloads. Logical host ledgers and
wire-size caps are not operating-system RSS limits or reservations. Any later
admission needs its own reviewed source, workload, runtime and host-headroom
evidence.
