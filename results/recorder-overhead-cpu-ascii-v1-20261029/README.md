# CPU recorder characterization after the ASCII fast path

This is one single-thread, CPU-only execution of the fixed
[recorder overhead protocol](../../docs/RECORDER_OVERHEAD_V1.md), using source
commit `23a19abb75248474151c0e84af29016d71bd334f`. No model was loaded and no
GPU work was performed. The saved [result.json](result.json) has SHA-256
`bbafecb7a25cc8829ca9a1e8d0b233a6c1218d2608f611e0376ce3fd14031ef3`.

Each of seven balanced, seed-17 paired blocks timed 400 alternating synthetic
selection and two-member forward-entry observations. Both modes constructed
the same records and read the same clock; one discarded each record, while the
other copied it into the bounded recorder. The median paired incremental time
was **10,320 ns per mixed event**; the seven block values ranged from 8,887 to
15,155 ns per event. All 2,800 recorder-mode observations were accepted. The
clock was Windows QueryPerformanceCounter with a reported nominal 100 ns
resolution; all 14 measured windows passed the 100-times-resolution screen.
The JSON has monotonic timestamps but no wall-clock timestamp; the directory
label is not independent evidence of the execution date.

GC counter snapshots showed one collection around every recording window and
none around discard windows; these observations can include boundary effects.
External CPU activity, frequency and affinity were not controlled. The earlier
[characterization](../recorder-overhead-cpu-v1-20261029/README.md) recorded a
28,229 ns median on the same fixed workload, but the two sessions were
unpaired, sequential and unisolated. Their difference alone does not establish
a causal speedup or population distribution. Neither run measures lock
contention, model or GPU overhead; neither should be subtracted from a GPU
result. Future performance claims require controlled, mode-specific observer
comparisons and reporting of batch formation and failures.
