# CPU recorder overhead characterization

This is one single-thread, CPU-only execution of the frozen
[recorder overhead protocol](../../docs/RECORDER_OVERHEAD_V1.md), using source
commit `f3c6cbb2211f04b46ff5b25dcde9bf251558dc41`. No model was loaded and no
GPU work was performed. The saved [result.json](result.json) has SHA-256
`f50c843a3aafa716d0259311f3672392064b72bcd73ba5cde33bbdb1bf9698c6`.

Each of seven balanced, seed-17 paired blocks timed 400 alternating synthetic
selection and two-member forward-entry observations. Both modes constructed
the same records and read the same clock; one discarded each record, while the
other copied it into the bounded recorder. The median paired incremental time
was **28,229 ns per mixed event**; the seven block values ranged from 20,838 to
38,320 ns per event. All 2,800 recorder-mode observations were accepted. The clock was
Windows QueryPerformanceCounter with a reported nominal 100 ns resolution;
all 14 measured windows passed the protocol's 100-times-resolution screen.
The JSON has monotonic timestamps but no wall-clock timestamp; the directory
label is not independent evidence of the execution date.

The GC counter snapshots showed one collection around every recording window
and none around discard windows. The snapshots are taken just outside the
windows and may include boundary effects. This single run does not isolate a
fixed lock/copy cost: external CPU activity, frequency and affinity were not
controlled, and seven blocks cannot establish a population distribution. Its
single-thread synthetic numbers do not measure lock contention, model or GPU
overhead, and must not be subtracted from a GPU result. A future performance
study needs mode-specific, controlled recorder-on comparisons and must report
any changes in batch formation and failures.
