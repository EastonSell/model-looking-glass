# Agreement calibration saved-evidence audit v1

This CPU-only checker reads a retained private five-token calibration attempt.
It neither loads the model nor imports Torch or opens CUDA, and it never
registers or approves a resource profile. A successful report has status
`saved_records_consistent_not_approved`.

Select the published producer commit, physical GPU UUID, run-freeze SHA-256,
source checkout, Git executable and dependency site independently. The
default `controller-reported` freeze observation records only the producer's
source-defined ordering, not an independently witnessed prelaunch commitment.
If a separate observer captured the freeze digest before launch, the operator
may declare `--freeze-observation independent-prelaunch`. The tool cannot
verify that declaration. Calculating the hash only from the mutable post-run
file does not establish prelaunch chronology. Keep
the private attempt stable and retain the controller's result and all run
files. The audit expects the original run directory, including its recorded
device/inode identity; copying it elsewhere is not a verified replay.

From this auditor's checkout, using an appropriate Python interpreter:

```text
python -B -m looking_glass.agreement_calibration_saved_audit \
  --run-dir PRIVATE_RUN_DIRECTORY \
  --controller-result CONTROLLER_RESULT_JSON \
  --source-checkout FROZEN_PRODUCER_CHECKOUT \
  --git-executable TRUSTED_GIT_EXECUTABLE \
  --site TRUSTED_DEPENDENCY_SITE \
  --expected-commit PUBLISHED_PRODUCER_COMMIT \
  --expected-device-uuid PRESELECTED_PHYSICAL_GPU_UUID \
  --expected-freeze-sha256 RUN_FREEZE_SHA256
```

The single entry point reads and validates the raw 84 numbered stage files,
the exact warmup and 13 actions, plan coordinates and callback metadata. It
independently recomputes cold and warm profile estimates from their sampled
peaks and checks the profile and evidence bytes against the source manifest.
It then binds all 14 journal callbacks, source/runtime and pinned asset
identities, candidate and final records, reported owner PID and clean reap,
headroom, and allocator cleanup. It compares the selected committed Git blobs
with the checkout and rehashes six recorded dependency entry files inside the
selected site. It rejects missing records, a failure sidecar, mismatches, and
inconsistent or impossible reported counters. A reversed wall-clock timestamp
is rejected even if an external clock adjustment caused it.

The report includes hashes of the examined files and the ordered concatenated
stage stream, with no local path or physical UUID. Save it separately alongside
its auditor source revision; the tool itself does not write a report. Invalid
or incomplete evidence returns a nonzero exit status and is not approved.

This is a retrospective consistency audit of self-reported local evidence,
not cryptographic attestation of execution. The source checkout and the six
dependency entry files are checked at audit time, not proven to be the bytes
executed. Asset pins are checked, but cached checkpoint bytes are not rehashed
by this tool; a separate preflight must verify them. The controller's recorded
NVIDIA snapshots include the controller-reported UUID, but neither those
snapshots nor the runtime UUID check independently attests physical hardware.
Neither sampled
peaks, allocator cleanup nor the reaped owner claim proves worst-case memory,
absence of native allocations, or global GPU quiescence. Separate source,
hardware, lifecycle and safety reviews remain necessary before any promotion.
