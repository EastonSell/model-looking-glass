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

Direct-interpreter subprocess tests check that a child with explicitly
inherited duplicate file handles retains protections after the owner dies,
and that a job terminates its child on owner death. A complete launcher must
independently compose these primitives: bind the launcher, bootstrap and
allowlist to the committed source revision, protect them for the entire
child lifetime, restrict inherited handles, block a child from project imports
until job assignment succeeds, kill and reap it on failure, and verify the
actual interpreter process is tracked. In particular, a virtual-environment
launcher may be a process shim: observing only the shim's exit is insufficient
to prove that its actual interpreter has exited. Windows cleanup after a
crash also requires a separate recovery policy for a leftover disposable ACL.
These tests do not establish that the eventual launcher meets those conditions.
