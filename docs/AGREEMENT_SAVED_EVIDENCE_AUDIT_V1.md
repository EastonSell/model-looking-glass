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

From a trusted auditor checkout, using an appropriate Python interpreter and
a new, empty private bytecode prefix outside the checkout, run the standalone
file with isolated Python. Do not add the checkout root to `PYTHONPATH`:
untracked files there could shadow standard-library imports. This invocation
does not itself attest the auditor's source bytes or interpreter.

```text
python -I -S -B -X pycache_prefix=EMPTY_PRIVATE_PREFIX \
  looking_glass/agreement_calibration_saved_audit.py \
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

## Incomplete attempts

The full directory command rejects an incomplete attempt. For an independently
bound identity and a checked, gap-free sequence of decoded stage files, the
Python helper `audit_partial_stage_records(records, warm_key)` validates only
the supplied prefix, including available headroom checks and completed callback
metadata. It reports the number of complete samples and the next expected
phase, and preserves an explicitly recorded trailing failure. Even a failure
after all 84 ordinary stage records remains an incomplete attempt. A started
or partly recorded action has unknown outcome; it is not a passed callback.
The next-phase description does not authorize resuming an interrupted owner.
This helper does not read files, establish file continuity or
source identity, inspect a journal or owner, reconstruct a profile, determine
whether the process exited, or approve anything. Retain the partial artifacts
and independently review those other boundaries.

`audit_incomplete_directory(...)` adds bounded, contiguous raw stage-file reads,
the frozen directory and workload identity checks, Git-blob and dependency
entry-file reconciliation, and a fingerprint of any saved journal bytes. The
journal fingerprint is not a callback or terminal validation. This method
does not inspect supervisor status, attest owner exit, establish cleanup, or
read cached model weights. Even if all 84 ordinary stages exist, it reports
that owner completion is unverified and returns no resource profile. The
freeze's chronology remains a controller-reported claim.

## Synthetic tests and frozen-producer compatibility

The default CPU test suite uses a versioned fake-telemetry fixture; it does not
depend on the proposed sampler being present on main. Its synthetic samples,
profiles and callbacks are not GPU measurements. The fixture was generated by
the CPU fake sampler from producer commit
`ff03d3c9739c894b33839e907b633109dba2a7b3`, and its exact bytes are checked by
the tests. The end-to-end fixture separately rebuilds profiles with the
repository's resource-profile implementation, allowing the auditor to check
the result independently.

An optional compatibility test replays that frozen producer's fake sampler
and compares its 84 stages, workload, profiles and evidence with the fixture.
Set `MLG_AGREEMENT_PRODUCER_CHECKOUT` to a trusted checkout at exactly that
commit, then run `pytest tests/test_agreement_calibration_producer_compatibility.py`.
The test requires Git and the producer's Python dependencies, including Torch.
It archives tracked source to a temporary directory, uses isolated Python with
CUDA masked, checks that CUDA was not initialized, and does not write into the
producer checkout. This test checks compatibility with that producer snapshot;
it does not attest a real GPU execution or make a resource profile eligible.
