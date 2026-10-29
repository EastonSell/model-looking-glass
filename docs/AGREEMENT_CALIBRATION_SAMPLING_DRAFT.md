# Five-token agreement calibration sampling draft

This is a CPU-tested sampling protocol for the reviewed supervised resource
pilot, not a GPU result or an approved profile. The owner enables the sampler
only within the dedicated one-attempt process. It constructs its workload key from source, checkpoint, observed runtime
and physical device, holds the shared device lease and memory/trace claims,
and loads a model that has not previously performed a forward. The launcher
requires an exact frozen revision and verified process and device ownership.

Once those prerequisites are reviewed, the intended sampler accepts the exact
`CudaAgreementCalibrator` and matching warm workload key. For the cold warmup
and then the 13 fixed warm actions, it checks the owner, writes a private
started record, synchronizes, resets peaks, synchronizes again and records a
baseline. It executes the whole callback, validates its returned fixed case,
seed and actual plan metadata, synchronizes, and records the final snapshot and
allocator peaks. A caller-provided writer must durably store each bounded stage
record in a private run directory. Partial records are not complete samples or
approved profiles. Both the actual baseline and endpoint are saved before the
corresponding fresh budget check; each observation must satisfy the live
free/reserved inequalities and the 3 GiB free threshold. The sampler requires
distinct, active commitments of exactly 2 GiB resident and 1.5 GiB job, with
exactly 2 MiB reserved for each trace domain. The calibrator accepts larger
minimum commitments, but this sampler deliberately fixes these exact amounts
so the recorded admission arithmetic matches its protocol. Those provisional
amounts are policy choices, not measured load
or worst-case bounds. Observed headroom is only a point-in-time check.

The metadata validator checks every expected module, operation, invocation,
coordinate, selector, shape, dtype, device, action order and logical source
and capture byte count. It rejects extra fields, including behavioral payloads.
These checks validate a trusted owner's records; they do not authenticate code,
prove actual device execution, or establish a worst-case memory bound. The
warmup callback returns its separately agreed policy hash, seed, model state
hash and ten-step metadata. The actual adapter-to-sampler callback seam was
exercised on a CPU synthetic model; this is not GPU resource evidence.

The sampler roots a recognized calibrator before checking its key, initial
state or claims, because it may already own CUDA resources. It does not retry
any device synchronization after a validation, callback or measurement
failure. It marks the calibrator unsafe, retains its own known
references, and attempts to record only a bounded failure type. The outer owner
must treat an exception or uncertain return from the sampler as terminal, retain
the physical and memory leases and known model/trace references until process
exit, and leave the supervisor marker in place without a clean attestation.
Repeated asynchronous interruption and device-internal temporaries remain
outside this Python custody guarantee. A watchdog must handle a hung callback.

The controller must freeze a private run manifest with source and asset hashes,
runtime expectations, physical inventory and the fixed public cases before the
owner begins CUDA work. The owner checks the freeze before initialization and
compares live observations before model loading. CUDA context and runtime
initialization precede the later allocator fraction and budget commitments;
they are not bounded by those controls. The calling-thread denormal probe does
not establish the arithmetic mode of other threads or all input handling.
The owner checks the imported Torch/CUDA build before its first explicit CUDA
call after import; native activity during the trusted import is not attested. Its
Windows platform shim avoids a subprocess version probe, and a Python audit
hook rejects Python-created descendants; the external Job still accounts for
native descendants. The driver version is the controller's prelaunch
`nvidia-smi` observation, not an independent child-side query, and the
controller must compare its post-run inventory before accepting evidence.

Profile JSON and stage files written by the child are untrusted candidates.
Even apparently complete files must not be registered or published if cleanup
or attestation fails. Only the external controller can validate and promote a
candidate after the exact supervised owner exits and is reaped cleanly. The
cleanup check reports a PyTorch-specific cuBLAS cache clear and zero current
PyTorch allocator bytes; it does not establish that all native or CUDA-context
memory is gone. Private source/runtime records contain paths and the device UUID.
The tests use CPU-only callbacks and a real CPU intervention plan; changing
its reported device field in a fixture does not establish CUDA behavior.
