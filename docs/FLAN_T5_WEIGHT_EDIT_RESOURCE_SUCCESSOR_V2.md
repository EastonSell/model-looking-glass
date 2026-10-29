# Prospective FLAN weight-edit successor: resource admission

This is a separately identified successor to the incomplete development
attempt recorded in `results/flan-t5-weight-edit-cpu-v1-20261029/development-failure/`.
That attempt's child stopped at the combined volume check before the driver
import; no model or case forward is recorded. Its one-shot claim and all
partial evidence remain in place and are not reused. This successor is fixed
before its own first model call and does not use a model outcome from that
attempt to change its cases or methods.

The original `FLAN_T5_TARGETED_WEIGHT_EDIT_V1.md` protocol, its fixed 24-case
bundle SHA-256 `a118dde671351d21c519b6a61957e9478c9501bac52d55aa34e8394772e933d5`,
checkpoint, discovery/calibration order, doses, thresholds, efficacy-matched
controls, target/locality cases, selection, underfill and incomplete handling
remain unchanged. The eight prior discovery/calibration inputs remain exposed
development; the other fixed wordings are not claimed independent of unknown
pretraining. This resource change is not a new scientific comparison.

Use the new fixed root `E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2`,
with separate development/evaluation directories and one-shot phase claims.
The original v1 root and claims are never modified or deleted to make space.
Keep the verified checkpoint and Python environment on C: as read-only
inputs; do not copy the model. All known study evidence, child and parent
temporary paths, disabled Python bytecode prefix, Git object staging and
HF, Torch, XDG and plotting-library caches are directed to this v2 root on E:.
The publication Git repository must be bare; the launcher verifies its actual
Git, common and object directories are inside that root and rejects object
alternates. Parent and child effective temporary/cache paths are checked and
recorded before the study driver is imported.
The exact source and both-phase manifest are published and independently
reviewed before a model call. Public development evidence and an independent
saved-data/process review still precede evaluation.

For this new attempt use a C: admission floor of **1 GiB plus 128 MiB**
(1,207,959,552 bytes), rather than the v1 amendment's 1.5 GiB plus 128 MiB.
Keep the E: floor of 2 GiB beyond the 256 MiB cumulative evidence cap and
1 MiB failure reserve. Check both volumes before child grant, immediately
before the first model call and before every subsequent model invocation.
If a check fails, stop further model calls, restore the current row if
possible, retain partial records and classify the phase incomplete. A failed
or uncertain v2 phase is not retried under the same claim.
Record the actual free bytes and pass/fail at each model-call boundary,
including a rejected observation, and retain a completed phase's number of
checked calls and minimum observed free space on each volume. Abrupt process
termination can leave only a partial record and remains incomplete.

This C: floor is a chosen operational heuristic, not an established safe
minimum or a filesystem reservation. The source/test files total about 200 KiB
and the 307,867,048-byte checkpoint is read in place. A separate no-model
runtime-import probe with the stated E: paths completed before this note;
its system-wide C: readings rose from 1,587,933,184 to 1,628,188,672 bytes
while the host had concurrent activity. Only the 1,518-byte probe record was
present in that E: probe directory afterward. Neither difference measures
process-specific C: writes, bounds paging or native-library behavior, or
guarantees headroom during the study. The C: margin, E: capacity and ongoing
checks are safeguards against observed pressure, not guarantees. The named
probe record SHA-256 is
`85c6e8fcbe5ba060f362b55919e8aca32835c9f4d7e793e17fc33fff9c53d42c`.
