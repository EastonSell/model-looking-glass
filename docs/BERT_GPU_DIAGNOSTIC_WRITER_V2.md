# DistilBERT calibration diagnostic publication v2

This is a follow-up to the development calibration in
`results/bert-gpu-calibration-v1-20261029/`. The archived v1 artifacts and
their original source are unchanged. They used direct writes to final names;
their existing truncation and interruption caveat remains in force.

Future invocations of the calibration use `bert_gpu_evidence.write_json_new`
for their diagnostics and result. It serializes trusted in-process values as
sorted, compact UTF-8 JSON with nonfinite numbers rejected, and rejects encoded
payloads above 1 MiB before opening a file. JSON serialization itself and
Python/file-system buffers are outside that logical byte limit; this is not a
bound on process RAM. Diagnostic directories must be owned by the cooperative
caller, with no concurrent modification by another process.

The writer exclusively creates `<name>.partial` in the destination directory,
loops until all bytes are written, flushes the file with `fsync`, and closes
it before creating the final name with a hard link. The final name is
create-only: an existing file is never replaced. This procedure is intended for
the local Windows NTFS workspace where same-directory hard-link creation is
supported; an unsupported filesystem fails closed. After a successful link,
failure to remove the extra partial name does not invalidate the complete final
file. Once partial creation succeeds, subsequent failures before linking leave
the partial for inspection and do not publish the final name. Existing files,
including partials, are not overwritten.

An interruption around link creation has an uncertain acknowledgement: either
name may exist. The interruption is propagated by the writer; the owner must
preserve the original cancellation, classify its attempt accordingly, and
inspect retained evidence without blindly retrying publication. A byte readback
can determine what is present, but neither readback nor file `fsync` proves
directory-entry durability after power loss. The implementation does not make
a portable directory-`fsync` claim, authenticate the filesystem, or protect
against an adversarial process modifying the directory concurrently.

This change concerns evidence publication only. It does not widen the measured
GPU workload, certify a general memory bound, or authorize a new GPU run.
