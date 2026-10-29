# Targeted edit v2 CPU owner

**Status: implementation in progress. Live development and evaluation entry
points are disabled.** No new checkpoint forward or result is claimed by this
component. The private execution and writer helpers may be exercised only on
synthetic test models until the committed source, manifest, fresh custody
receipt, and independently witnessed remote-publication gate are in place.

The intended owner uses a fresh supervised CPU-only direct child for each stage.
The child must claim the persistent CPU study root before importing model
dependencies, use the reviewed launch grant, and retain its exact lease until
exit. The parent must keep partial outputs separately, and may publish a
complete terminal only after a clean, zero-exit, reaped child, exact source
and identity checks, verified restoration, and the independent saved-output
reader all pass. Timeout, interrupted return, failed cleanup, missing receipt,
or unverified publication yields an incomplete attempt. A terminal record is
an attestation and does not prove a model or exception graph was disposed.
See [CPU study supervision](CPU_STUDY_SUPERVISION.md) for the cooperative
process/root and CUDA-mask preconditions.

The inherited v1 environment helper calls `platform.platform()`, which can
spawn a short-lived subprocess on Windows. Before enabling the owner, rehearse
the full supervised import and environment path without loading or forwarding
the checkpoint, account for subprocess lifetime, and ensure any spawn-free
replacement exactly matches the frozen v1 environment. Do not weaken process
accounting to make this preflight pass.

The private condition guard checks the original named parameter identity,
registered parameter and module identities, parameter gradient/require-grad
flags, module training flags, forward/backward and state/load-state hook
registries, the existing v1 state-dict hash, and a separate digest for every
named CPU buffer including nonpersistent buffers. The supplemental digest
hashes canonical `targeted-edit-v2-buffers-v1` JSON containing each buffer's
name, dtype, shape and contiguous raw-byte digest and each module's registered
buffer names and nonpersistent set. Object identity is checked in-process and
is not encoded in that digest. The original weight is restored and all checks
are repeated after each condition. A failed check or condition poisons the
guard and prevents subsequent forwards. A later evaluation must reproduce the
development buffer digest as well as the published v1 identities.

After the callback returns and restoration verifies, the condition remains
fenced. The writer flushes and fsyncs its condition-state journal entry before
calling `commit_condition()` to reopen the guard. The guard trusts the caller
to make that record durable; it cannot prove the fsync occurred. A lost condition
return remains fenced. If the acknowledgement itself returns ambiguously,
the caller must stop and reconcile the journal and final publication; it must
never blindly replay the completed condition. Rename/return reconciliation is
best effort and requires the supervising controller to audit the exact saved
destination independently after any lost or failed return.

These checks do not capture arbitrary Python attributes, monkeypatches, global
hooks outside the listed registries, native library state, concurrent writers,
or hidden external model owners. Trusted standard model code and exclusive,
quiescent ownership are preconditions. `state_dict` hooks are rejected because
they could conceal a mutation. Uncertain restoration requires process-level
recovery; a clean saved record cannot itself establish that recovery happened.
The saved selection, including a diagnostic fallback or negative result, must
remain unchanged when evaluation is later implemented.
