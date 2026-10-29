# Custody-bound two-token service profile v1

This protocol defines a distinct workload identity for the fixed two-token
service. It describes a measurement to perform, not evidence that it has
already run or a guaranteed allocation bound. The older direct-worker
`trusted-service-runtime-v1` profile measures a different execution path and
must not authorize this one.

Use the pinned offline DistilGPT2 assets and the verified supervised physical
device, source, runtime, cache and offline settings described by
`TRUSTED_SERVICE_PROFILE_V1.md`. The service and collector must independently
derive their live identities from the same frozen, byte-identical source and
runtime; neither may copy a key supplied by a client or by a profile file.
The new runtime schema is `trusted-custody-service-runtime-v1`. Its identity
binds this document's SHA-256 and the exact execution descriptor in
`looking_glass.custody_service_profile`.

The measured callback must invoke the real
`LifetimeWorkspaceWorker.execute`, which calls
`ResidentWorker._execute_with_custody`, `_execute`, `_perform` and
`_perform_with_custody`. The resident worker must use the owner-bound
`_fork_owned_rng` and the authority's device checks. The collector and the
service must verify the actual worker/authority/lease binding and the matching
execution implementation before admitting work; an identity manifest or a
profile pin alone cannot prove those facts. Both retain the distinct 2 GiB
resident and 512 MiB workspace commitments through process exit, with zero
GPU and host trace allowance and at most one forward at a time.

Run one cold forward using seed 17, transfer the exact cold result through
`_finish_cold`, and then run warm forwards using seeds 29, 43 and 71 with the
same fixed int64 `[1, 2]` input IDs `[15496, 995]`. Each warm result must be
rooted and transferred through the same custody path used by the service.
The synchronized sample boundary must include the declared wrapper, result
validation, offload and required custody transfer. Use separately admitted,
prespecified bootstrap commitments; any provisional descriptors used to
construct the worker are not measured profiles or approved profile pins.

Record the model-load sample and the ordered cold/warm samples, telemetry,
reserve decisions, output checks, cleanup, supervisor/owner linkage, source
and runtime manifests, and failure or quarantine evidence. Preserve failed or
incomplete attempts as failures. Derive the three estimates and margins using
the fixed resource-profile protocol and require independent full profile pins
before service use. A sample, its source hashes and supervisor records remain
attestations within a trusted process; they do not prove executed code or
prevent out-of-memory failures from unobserved or external allocation.
