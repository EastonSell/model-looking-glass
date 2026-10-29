# Development-only source and process probes

These Windows-oriented primitives exercise parts of the proposed v2 source
policy. They do **not** launch a study or establish that a study is ready; no
v2 split, hidden schedule, or held-out case is prepared by them. The
diagnostic explicitly reports `study_ready: false`.

`scripts/v2_source_bootstrap.py` is an isolated, CPU-masked import diagnostic.
It requires `-I -S -B`, a fresh empty `-X pycache_prefix`, and explicit paths
to a checkout, source-digest manifest, virtual environment, and dependency
directories. It installs an allowlisted project-only finder before the package
initializer. For each imported `looking_glass` source, that loader hashes and
compiles the supplied `.py` bytes directly, without loading bytecode caches;
unknown project modules fail closed. Checkout paths are not put on `sys.path`,
and dependency paths are appended after the interpreter's standard-library
paths. It checks selected provenance and source hashes again after import.
Its manifest bounds, path checks and loader are tested using fabricated files,
including stale and sourceless bytecode and attempted import shadows.

The diagnostic accepts *supplied* hashes and checks dependency directory
locations; it does not authenticate a commit, verify dependency bytes,
prevent source mutation for its full lifetime, or establish custody. Its
post-import inventory is an operational check, not proof of all code executed.
The diagnostic has no model or experiment entry point. `-I` ignores
`PYTHONHASHSEED`; output-relevant ordering must be explicit.

`scripts/v2_source_guards.py` supplies separate Windows file-sharing,
disposable empty-cache, inheritable-handle and kill-on-close job primitives.
A lease holds read-sharing handles for the declared source files and ancestor
directories. A fresh disposable cache directory is protected using a
directory handle and a temporary ACL denying child creation, and a separate
process tests that denial. These are ordinary-concurrency protections, not a
security boundary against a process running under the same account. A partial
cleanup is a failure: retained handles must be retried and the process
quarantined until cleanup succeeds. The cache is never recursively deleted; a
nonempty prefix is retained for inspection.

`scripts/v2_source_launcher.py` composes these checks for the **diagnostic
only**. It checks the launcher's, bootstrap's, guard's and project Python
sources against tracked Git blobs, acquires source and manifest leases plus
an empty-cache guard, then verifies them again. The parent starts the actual
base interpreter directly with no inherited protection handles, assigns it to
a kill-on-close job, duplicates noninheritable protections **directly into
that child**, and only then sends their child-local values in a bounded start
frame. The child verifies its handles and rejects EOF, a partial frame or
invalid values before project imports. No temporary inheritable protection
copies are created in the parent. Run the diagnostic in a fresh dedicated
process: a check for Python-managed threads catches routine misuse but cannot
prove that no native or low-level thread exists. On normal and failed returns
the parent terminates and drains the job before releasing protection;
assignment failure kills and reaps the still-unassigned interpreter.
Subprocess tests exercise a fabricated Git checkout, dirty sources, source
locks, parent interruption, assignment failure and unknown bytecode. The
host's versioned interpreter alias is a Windows junction, so this diagnostic
resolves and checks its canonical target while retaining the no-reparse rule
for protected checkout and cache paths. A virtual-environment launcher may
be a process shim: observing only the shim's exit does not prove that its
actual interpreter has exited.

The launcher permits only the trusted committed `looking_glass.trace_schema`
diagnostic import; Python also executes its package initializer, so this is
not a sandbox for untrusted project sources. The diagnostic records the
checked source list, but does not yet retain the
child's complete imported-module inventory, bind study protocols or custody,
or verify dependencies and native code. Windows cleanup after an owner crash
also needs a recovery policy for a leftover disposable ACL. An embedding
caller can retain the `LaunchError.quarantine` object and retry cleanup; if
the CLI cannot clean up after a retry, it reports protection as unknown on
process exit. Owner death can
close the job and parent handles in an unspecified order; for a project that
spawns descendants, this implementation has not established that every
descendant retains protection until it exits. A study launcher must address
that limitation or prohibit descendant processes, and be reviewed separately.
These checks do not authorize a v2 run.

Use the tested direct-script invocation with the canonical base interpreter
and `-I -S -B`, not an imported launcher from an embedding process. The Git
checks authenticate the on-disk launcher bytes against the selected commit;
they are not a cryptographic attestation of the bytes already executing. An
embedding process could import a timestamp-valid cached launcher before this
check, and neither the interpreter nor native dependencies are attested here.
