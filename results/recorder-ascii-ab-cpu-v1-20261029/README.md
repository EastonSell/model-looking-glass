# Paired CPU recorder comparison

This is one CPU-only execution of the preregistered
[paired comparison protocol](../../docs/RECORDER_ASCII_AB_V1.md), with the runner recording clean
source commit `15052b4974de17545b3cec622220013de675d6f7`. It compares exact
baseline and ASCII fast-path recorder blobs within one Python process; no model
was loaded and no GPU work was performed. The unmodified [result.json](result.json)
has SHA-256 `fdef9243011f09c831016f83196f12c61a8051fc563e7fa9c1750e517b02d833`.

Eleven seed-41 paired blocks each recorded 400 alternating synthetic selection
and two-member forward-entry events per variant. The median paired
baseline-minus-fast difference was **19,443.75 ns per event**. Ten block
differences favored the fast path; one favored the baseline. The full range
was **−179,669.75 to 24,433.25 ns per event**, including the large negative
outlier. The median baseline and fast window durations were 13.56 ms and
5.64 ms, respectively, per 400 events. This is a measurement of the specified
synthetic workload on one host, not a general service or GPU speedup.

All 8,800 observations were accepted, and the runner reported matching payloads.
The saved source and recorded timestamp sequences match for every event; the
original full payloads were not retained, so their equality cannot be
independently recomputed from this artifact. The Windows QueryPerformanceCounter
reported a nominal 100 ns resolution, and the shortest of 22 measured windows
was 3,712,700 ns, above the protocol's 10,000 ns screen. The JSON contains
monotonic timestamps but no wall-clock timestamp; the directory label is not
independent evidence of the execution date.

GC snapshots showed one collection around each window, with collection
generations differing across the two arms. In particular, the large negative
pair included an 86.57 ms fast window and a generation-2 collection between its
snapshots. The snapshots may include boundary effects and do not establish the
cause of that window's duration. Host CPU activity, frequency and affinity were
not controlled; interpreter specialization, caches, allocator and GC carryover
also remain possible influences. Eleven blocks do not establish a population
distribution or contended-recorder overhead. These timings do not measure a
neural-network or GPU workload and must not be subtracted from GPU results.
