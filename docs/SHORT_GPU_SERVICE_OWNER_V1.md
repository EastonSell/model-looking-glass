# DistilGPT2 short-input supervised development owner

This owner prepares a single, source-specific CUDA observation. It is not a
general shared-model service, a capacity guarantee or a throughput benchmark.
The previously published [CPU preparation](SHORT_GPU_SERVICE_PILOT_V1.md)
explains the fixed public inputs and their reservation history. The exact v33
inventory, including token masks, must be supplied, and later public exposure
updates must be checked before forwarding. These are public development inputs,
not held-out examples or proof of pretraining independence.

An independently reviewed attempt requires a fresh coordinated device slot.
Run one separate attempt for bucket 16 and one for bucket 32. Each uses the
synthetic row and the corresponding right-padded science row, one resident
DistilGPT2 model, two serial full-logit references and two concurrently
submitting in-process Broker clients. The model forwards are serial. The
complete float32 logits are retained: 3,216,448 bytes per 16-token result and
6,432,896 bytes per 32-token result. Both clients must match their own complete
serial reference at `1e-6 + 1e-5 * abs(reference)` and their last-active rows;
the reference rows and the wrong-client comparisons must differ at that
tolerance. The model-free reader reconstructs these checks from retained bits.
The owner copies the actual device-side token IDs and masks back to the CPU
at the model's Python forward entry, under the same lease, and checks and saves
them before each of the four executions. The last two client calls may arrive
in either order. These extra synchronized copies are part of the measured
workload. An entry record is an observation under the trusted-process
assumption, not independent execution attestation.
The references share the padded GPU path. They test routing and repeatability,
not pretrained padded-versus-unpadded equivalence, IPC, compact-output
projection, simultaneous model calls or a two-row batch.

Before CUDA-capable imports, the direct child binds the supervisor's canonical
physical-device lease and protected source manifest. Under that lease it checks
the pinned checkpoint, seed 29, float32 eager evaluation, disabled cache and
autocast, deterministic settings and disabled TF32. Its provisional admission
requires at least 5 GiB device-free at preload, the native allocator capped at
25% of device total, a 2 GiB resident claim and a 1 GiB workspace claim, plus
the existing reserve policy. Live device telemetry is rechecked during
admission. These controls are policy commitments and are not predictions of
the actual peak. The 1×2 and 1×4 observations provide no bound for this owner.

The owner also requires point observations of at least 3 GiB available host
physical memory and at least 2 GiB free on the common output/cache/temporary
filesystem. The evidence parent, checkpoint staging cache, `TEMP` and `TMP`
must be plain directories on the same local fixed volume; the output is a new
direct child of that parent. The canonical workspace and physical-device lock
remain on their original volume. A separate 1 GiB workspace-volume free-space
screen covers the small device-coordination and checkout/Git writes expected
there, with a substantial residual margin. It cannot bound unrelated system
writes. Both screens are repeated before device use and at retained-output
boundaries. The private storage binding records directory identities, and the
controller and child recheck them; these cooperative point checks do not make
path replacement or external writes atomic. A
256 MiB cooperative host claim covers the planned output and validation
envelope. None of these is an operating-system reservation or a bound on model
construction, Python objects or RSS. The loader temporarily stages the pinned
checkpoint in the cache, so the free-space check precedes loading; space may
change after any check. Existing evidence and partial files must be preserved.

Synchronized records include device-wide free/total points and this process's
allocated/reserved counters and reset-window peaks. Separate windows cover the
two references and the pair of serial client executions. Device-free samples
are points, and allocator counters and peaks do not include other processes or
all native allocations. On a successful workload the broker and clients are
joined, known model and input/output references must be released, guarded
CUDA cleanup must report zero process-local allocator counters, and the host
and device claims must be released before the owner attests clean. The
physical lease remains held until child exit. These checks do not establish
complete native CUDA teardown or global device idleness. A failed or interrupted
operation remains incomplete without clean attestation; no automatic retry is allowed.

The isolated bootstrap uses the explicit dependency directory and a fresh empty
bytecode prefix. It verifies the committed source files, pinned input manifest,
and checkpoint bytes before the child runs. The supervisor requires a clean,
reaped direct-child exit and the Windows Job to drain. A candidate file alone
is not completion: the controller verifies the owner identity, saved memory
inventory, all four full-output hashes and the reconstructed routing checks.
The outer controller must also be separately observed and reaped with exit 0,
and its result and inventory read back and hashed: an interrupted write may
leave a result file even when the controller did not finish. Partial or
contradictory evidence remains incomplete. Original owner and
supervision records may contain process or physical-device identifiers and
should stay private; a public projection is a saved-byte consistency report,
not independent proof of execution or globally idle hardware.

With a reviewed immutable source, a freshly coordinated slot, a currentness
check, and adequate fresh disk/host observations, the controller command has
this shape (paths must be absolute and the output must be a new directory
directly under an existing plain evidence parent on the common storage volume;
`TEMP` and `TMP` must be set to existing plain directories on that volume):

```text
python -I -S -B -X pycache_prefix=<fresh-empty-prefix> \
  scripts/short_service_gpu_bootstrap.py --site-packages <venv-site> \
  --workspace-root <canonical-workspace> --cache-dir <pinned-hub-cache> \
  --inventory <v33-canonical-json> --output <new-work-directory> \
  --git <absolute-reviewed-git-executable> --bucket 16
```

The owner and controller assume a trusted, quiescent checkout and interpreter;
source hashes and file leases are cooperative checks, not a hostile-process
security boundary. A separately supervised CPU reference is needed for the
unpadded comparison described in the CPU preparation guide.
