# Supervised GPU service composition (in development)

The owner composition in `looking_glass/gpu_service.py` is **not a public
launcher or a measured GPU service**. Its supported target is the pinned
DistilGPT2 checkpoint and one exact no-capture forward signature: one batch,
two `int64` input IDs, a 512 KiB tensor output cap, and a 4 MiB JSON response
cap. One seed-17 cold forward on `[15496, 995]` is intended to precede warm
requests. The worker serializes model execution; multiple clients may submit
work, but this is not evidence of concurrent GPU kernels, throughput, or a
safe upper bound on memory use. Trace retention is zero in this initial mode.

The staged owner checks the supervised device lease and the runtime UUID before
telemetry, resolves separate load/cold/warm profiles against independently
constructed live keys, retains the model-load reservation until process exit,
and repeats the lease, runtime, and headroom checks before copying weights.
The actual CPU model must pass an explicit verifier before that copy. At
shutdown, the owner distinguishes workload failure from verified device drain;
unverified state latches process quarantine and retains reachable resources.
These ordering checks are CPU-tested with fakes, including real supervisor
subprocesses and an authenticated two-team CPU service. They do not establish
that the GPU path works on this machine.

Real service launch still requires a reviewed CPU-only preflight, a trusted live
source/runtime/model verifier, a private supervisor-to-parent startup channel,
and separately reviewed measured model-load and forward profiles. None of the
present test profiles is measured or approved. All real owners on this machine
must use the same canonical project workspace root for device coordination.
The runtime UUID representation also requires verification against the actual
device under supervision. A missing prerequisite must fail before launching an
owner; an uncertain owner must exit with its sticky marker intact.

## Development incident: unintended CUDA query

During a failing development regression at approximately 2026-10-29 00:57–00:58
UTC, `test_model_verifier_must_explicitly_return_true` invoked the real backend
with a fake session. The implementation had not yet rejected a verifier's
`False` return, so it reached `torch.cuda.device_count()` and
`torch.cuda.get_device_properties(0)` before fake identity verification failed.
No model was copied to the GPU; no benchmark, calibration, or profile evidence
was collected. No coordinated GPU window was active.

The command was `work\.venv\Scripts\python.exe -m unittest discover -s tests
-p test_gpu_service.py -v` in `work\teams\gpu_service`. The command runner
reported exit code 1 after 7.7577 seconds (output chunk `92ae2a`), but did not
report the Python PID. Attempts to recover a matching Sysmon or Security 4688
event yielded no event, so the PID is **unknown**. A subsequent Windows process
query found no Python command line containing `gpu_service`; a read-only
`nvidia-smi` process query listed no Python process. Windows WDDM reported
per-process memory as `N/A`, so neither observation proves exclusive GPU use.

At 2026-10-29 00:59:03 UTC an independent read-only `nvidia-smi` snapshot
reported 8192 MiB total, 203 MiB reserved, 871 MiB used, 7119 MiB free, and 0%
utilization. It is a point-in-time observation, not an isolation guarantee.
All timings before a fresh quiet window remain non-benchmark data. The model
verifier now must return exactly `True`, and the CPU test module hides CUDA
before importing PyTorch, blocks PyTorch CUDA/driver entry points, CUDA tensor
copies, NVIDIA library loads through `ctypes`, and `nvidia-smi` child commands,
and installs the same call guard in fake supervised children. These test guards
are not a sandbox against arbitrary native code.
