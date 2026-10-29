# Injection public-runtime replay process custody

`tools/injection_runtime_replay_owner.py` composes the published gated child
with the reviewed Windows CPU Job and source/cache leases. It is a component,
not a command-line entry point or authorization to claim a phase or run a
model. Its output is private; `private_replay_pending_independent_audit` is
not scientific completion.

The trusted caller must independently read back the immutable replay plan and
each source, verify the clean frozen discovery checkout and assets, protected
parent/ledger identity and access control, the exact one-use ledger row and a
separately observed successful claimant exit. The caller must keep complete
frozen and operational source leases and the protected empty bytecode prefix
through the child Job's verified exit, drain and close. Any claim or replay
failure is retained and does not permit an outcome-driven retry. A separately
observed successful exit of this owner process is required before accepting
the saved parent terminal. The independent saved numerical and landing audit
and a site-blind public projection remain separate steps.

`verify_preclaim_handoff` re-reads the exact intent and receipt, recomputes the
retained diagnostic inventory, and invokes the separately verified v6 guard on
the independently anchored ledger state and exact claim-row hash. It checks
the protected parent ACL and path identities before and after. The caller must
supply the receipt digest from an independently observed successful claimant
exit; the returned consistency status is deliberately not an exit attestation
or permission to run the replay. Its configuration must be parsed from the
same protected canonical bytes later passed to `launch_child`; the caller must
re-read and match those bytes across the guarded launch and exit.

The private-parent check reads the DACL with a fixed system PowerShell command
and rejects any policy other than protected inheritance, current-user owner,
and two explicit inheritable full-control grants for that user and SYSTEM.
The PowerShell executable is resolved from `GetSystemDirectoryW`, rather than
an ambient `WINDIR`, and its module path is restricted to the system modules.
The fixed ACL reader is a separate synchronous process with a timeout; this
check does not independently inventory possible native helper descendants.
It checks the directory identity before and after the ACL read; the caller
must check again under its protected prelaunch and postlaunch scopes. This is a
cooperative local-process boundary, not protection against a malicious current
user or administrator, who can also change an ACL between observations. The
protected bytecode prefix is created first; the
caller writes its exact path into the configuration, then acquires and roots a
lease on that generated configuration before allowing the child to start.

`run_guarded` masks ordinary Windows console interrupt signals before entering
the lease scope, so a healthy in-progress child is not cancelled by Ctrl+C.
It holds process and lease roots when cleanup is unknown; forced process death
and arbitrary native return-boundary interruption are not covered. The caller
must use a dedicated process, avoid concurrent users of the same supervisor
module, and retain source/Job objects in the rooted `retained` value. The
source-bound bootstrap hashes the child before compiling it and passes only
paths and public digests in the command. It uses the direct base interpreter,
isolated flags and the protected empty prefix; the child consumes its start
token before importing any project module.

The protected precheck must return the singleton `True`; false, missing, or
exceptional outcomes stop before the child is launched. Supervisor logs are
rechecked against the same 256 KiB per-stream cap, and the supervisor output
inventory is incrementally bounded to exactly its three expected files.

The supervisor is cloned into a private namespace with a 256 KiB limit for
each log stream; the original module is not modified. The replay journal's
14 MiB bound, both streams, and a conservative reserve for the child and
supervisor metadata must fit the fixed 16 MiB pilot allowance. The custodian
must inventory the whole disjoint pilot root and all retained partials, keep
the 4 MiB no-model diagnostic reservation separate, and enforce the remaining
phase-wide logical and physical limits. A no-model validation run uses a
distinct bounded output and does not read the site custody or create a claim.

`inspect_child_postcheck` checks the exact private postcheck, terminal and
receipt hashes after clean child cleanup. `check_supervisor_record` binds the
actual command, child PID, zero exit, Job drain/close, logs and postcheck to a
private parent terminal, including the owner's PID and Windows process birth
time. The owner uses its current identity; an independent observer may supply
the identity obtained from its retained process handle when reconstructing the
expected terminal. Both intentionally report the scientific replay as
pending. Neither establishes that the separately observed owner process has
itself exited, that the protected source or row callbacks were truthful, nor
that the numerical replay is correct.
