# Targeted-edit v2 source rehearsal

This Windows-only diagnostic launches one trusted direct CPU child without
loading a checkpoint, constructing a model, or performing a model forward. It
does not create a study result, reserve evaluation cases, verify a study
manifest or remote witness, or enable either live phase. A successful report
is a cooperative observation of a selected runtime and source snapshot; it
does not attest every installed interpreter, dependency, native library, or
system byte.

The isolated parent checks committed source bytes and holds source, manifest,
and empty-bytecode-cache protections while the child is active. The source-only
finder checks project modules against the committed inventory before loading
them, with the interpreter's standard library ahead of the checkout. The child
claims the persistent CPU study root under the shared supervisor's launch
grant and pre-grant Windows Job before importing Torch. It records the literal
empty visibility mask at claim, changes it to `-1` before model-capable imports,
and compares selected environment fields and denormal probes with the pinned
v1 precommit. The Windows platform shim uses a parent-observed uname and native
version fields to avoid a Python-visible `cmd.exe` spawn; the reported match is
an observation, not an independent proof of the parent observation.

The child-written report is provisional until the parent verifies its bounded
regular-file contents, exact supervisor run ID and reaped direct-child PID,
clean zero-exit result, confirmed Job drain, source inventory, and cache. A
clean flag depends on trusted code's teardown attestation. Job accounting does
not attest escaped or otherwise out-of-job processes; the Python audit hook
also does not intercept arbitrary native process creation. Source leases do
not protect installed packages or native libraries. The report explicitly says
`study_ready: false` and `parent_death_source_protection: false`.

Each attempt requires a fresh, single-use isolated parent. The caller-known
attempt ID must not be reused within that parent process; the bounded registry
does not retain old-ID tombstones after release. The ID roots a quarantine before
source/cache/job acquisition. While the owner is running or disposal is in
progress, another caller cannot release it. Any exception after launch begins,
including a lost supervisor return, leaves protections rooted while the parent
process remains alive; `get_quarantine` retrieves the exact record. An
interrupted or failed disposal is not blindly retried. The registry assumes
trusted, quiescent callers and does not authenticate capabilities. As with
other Python lock contexts, an asynchronous exception can strand a lock and
require process-level recovery.

The command-line diagnostic reports the attempt ID and known quarantine
directory on failure. `quarantine_rooted_at_report` describes only the instant
the report is written, and `parent_exit_releases_handles` states the lifetime
limit. The diagnostic may then exit with unresolved custody. An interrupted
directory-creation return is retained at its preselected path; an existing
collision is never deleted as if it were owned. A failed attempt may leave a
restricted scratch cache or directory for later inspection. Parent exit or
death releases its handles; job close may kill its
children, but the source leases are not guaranteed to remain held until that
kill finishes. A successful job drain in a normal return does not remove this
limitation. A future live owner must specify and independently review its
source-custody and parent-exit assumptions, including whether it relies on a
trusted, quiescent checkout or protection handed to the child. Before any
study forward, it must bind its final source and dependency origins to the
frozen manifest and validate the independent custody/witness chain. It must
handle ambiguous outcomes without treating them as complete and verify its
own model and process teardown before terminal publication.
