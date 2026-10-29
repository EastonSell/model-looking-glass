# Exploratory CPU serializer timing

These are post-change diagnostics, not a preregistered benchmark or a GPU
throughput result. They exercise the float32 IPC encoder from commit
`4990d4b635f77055a8208fc039b8f4711b844d9f` on one fixed 100,514-value CPU
tensor of synthetic finite float32 bit patterns, not model logits. The host
was not isolated. The timed operation ends after producing Python numbers; it
excludes JSON framing, network transport and GPU work.

For each denormal-flush setting, the script interleaves the NumPy encoder,
the standard-library fallback, and the old conversion once per round. It
records three warmup and 12 measured rounds per mode, using seed 6017 and
`perf_counter_ns`. Flush-off and flush-on blocks run sequentially, so their
difference does not identify a causal effect of the flush setting. The
repetitions use one input and are not independent examples.

| Run | Flush-on NumPy median | Flush-on fallback median | Flush-on old conversion median |
| --- | ---: | ---: | ---: |
| Preliminary | 3.14 ms | 86.20 ms | 2.89 ms |
| Pinned | 4.99 ms | 124.51 ms | 3.56 ms |

The old conversion is not numerically equivalent: under flush-on it changed
the four tested positive and negative subnormal edge values to signed zero.
Its timed path contains the former finite check and reshape/`tolist`, not the
full prior encoder's validation or framing.
The exact encoders preserved all eight checked edge values in both modes.
This diagnostic checks only those eight outputs; broader independent bit and
wire checks are separate tests. The two runs illustrate variation on this
non-isolated host, not a general performance range.

`preliminary.json` and `pinned.json` retain the raw ordered rows and all
summary medians. The preliminary file predates recording a script hash and
timestamp; the pinned file records both. The operator reports using the same
timing loop and input generation, but the preliminary file alone cannot verify
that. Three fresh interpreter checks per run record Torch import,
whether it had already imported NumPy, and total child-process time; operating
system caches were not cleared, so these are not cold-storage measurements.

To repeat the diagnostic, install the repository's CPU dependencies and run
the script with the repository path and a **new** output path. The script
uses exclusive creation for the result file and refuses to overwrite one. Its
final CUDA-initialization check occurs after the file is written, so accept a
new result only if the script exits successfully. Use a committed checkout and
record any local changes separately. For example,
from PowerShell on this machine:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python results/ipc-serializer-cpu-exploratory-v1/measure.py . <new-output.json>
```

The exact source-byte hash depends on checkout line endings; the recorded Git
commit identifies the repository revision. No model was loaded or evaluated.
