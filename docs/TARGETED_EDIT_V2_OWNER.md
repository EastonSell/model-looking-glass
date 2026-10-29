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

For this v2 study, the child must first claim with the supervisor's literal
empty `CUDA_VISIBLE_DEVICES` value and no model-capable dependency imported.
Before importing those dependencies it must change the value to the exact v1
literal `-1`, verify it, and keep checking it through the work and immediately
before clean attestation. The saved supplemental runtime field records `-1`;
the original v1 descriptive `disabled` field is not changed. The two literal
settings produced different `torch.__config__.show()` hashes on the recorded
host, so they are not interchangeable for v1 reproduction. These are
cooperative CPU preconditions, not proof that arbitrary code cannot use CUDA.
Before loading the checkpoint, the owner validates and snapshots all recorded
v2 runtime settings; it rejects any changed setting after loading or tokenizer
verification before running the discovery forward. The condition guard keeps
checking that snapshot through the study. These checks still rely on trusted,
exclusive code and cannot detect a change that is made and reversed between
checks.

## Historical v1 cleanup limitations

The historical v1 candidate and protocol remain byte-for-byte unchanged. Its
cleanup uses `sys.exception()`, which is unavailable in Python 3.10; a failure
there can leave registered hooks and altered gradient flags behind. Even on
Python 3.11 or later, a caller's unrelated handled exception can be mistaken
for a candidate failure and an ordinary hook-removal error can be suppressed.
The recorded v1 run used Python 3.11; these findings do not reclassify its
result. The old direct v1 entry point is not protected by the v2 guard and
should not be used in either affected context.

The prospective v2 owner rejects a missing `sys.exception()` API or an active
handled exception before checkpoint loading and checks again immediately
before calling the unchanged candidate. In an accepted context, a hook-removal
failure after an otherwise successful candidate is surfaced. This is a narrow,
cooperative guard, not a repair of the historical direct entry or proof of
cleanup. An asynchronous interruption during hook registration may lose a
just-created handle, and a failed cleanup may leave a hook installed. On any
uncertain candidate failure or interruption, stop and reap the isolated owner;
do not perform another forward or infer disposal from an exception or record.

## Source and custody

The prospective source inventory includes the remote-witness and evaluation
reader and writer modules and their protocols, as well as the CPU supervision and
supervisor contracts. A static regression checks direct relative imports of
the listed project modules; it cannot discover dynamic imports or establish
where a running interpreter imported a module. Before any study run, the final
controller and bootstrap must themselves be inventoried and an independent
source-origin gate must verify the actual imported bytes.

The inherited v1 environment helper calls `platform.platform()`, which can
spawn a short-lived subprocess on Windows. Before enabling the owner, rehearse
the full supervised import and environment path without loading or forwarding
the checkpoint, account for subprocess lifetime, and ensure any spawn-free
replacement exactly matches the frozen v1 environment. Do not weaken process
accounting to make this preflight pass.

The separate [source-bound rehearsal](TARGETED_EDIT_V2_SOURCE_REHEARSAL.md)
exercises that path without a checkpoint or forward. It uses one CPU
supervisor and its pre-grant Windows Job. It is a diagnostic, not a complete
live-study controller: its parent-held source protections do not survive
parent death, and it does not establish a manifest, custody receipt, witness,
or study readiness.

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
The destination must be lexically absent, including dangling links; an
uninspectable path is rejected. This check relies on the trusted exclusive
namespace and is not an atomic fence against a concurrent filesystem writer.
Before the development terminal is assembled, each staged artifact must be a
regular file within its declared byte bound, and every matrix must have the
fixed size; a later independent reader still audits the saved content.
The publication and reconciliation helpers set the calling thread's Torch
denormal mode to preservation and do not restore its previous mode. Run them
in a designated CPU context and account for this state change.

These checks do not capture arbitrary Python attributes, monkeypatches, global
hooks outside the listed registries, native library state, concurrent writers,
or hidden external model owners. Trusted standard model code and exclusive,
quiescent ownership are preconditions. `state_dict` hooks are rejected because
they could conceal a mutation. Uncertain restoration requires process-level
recovery; a clean saved record cannot itself establish that recovery happened.
The saved selection, including a diagnostic fallback or negative result, must
remain unchanged when evaluation is later implemented.
