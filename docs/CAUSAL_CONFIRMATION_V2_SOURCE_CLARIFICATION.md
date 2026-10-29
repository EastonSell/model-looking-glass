# Synthetic causal v2: pre-seal Python source-execution clarification

This clarification supplements the proposed, pre-seal
[v2 protocol](CAUSAL_CONFIRMATION_V2.md) and its
[numeric clarification](CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md).
It does not alter the saved v1 or development-pilot records. Commit and review
it before any v2 split or site schedule is sealed. The public pre-manifest must
bind this file's SHA-256 and the source revision, launcher policy, interpreter,
and recorded project-source hashes. It must retain the original exact keys
of the protocol's commitment `context` object; bind the clarification in a
separate top-level `source_clarification_sha256` field instead.

Hashing a module's `.py` file and inspecting `module.__file__` do not by
themselves prove those source bytes were compiled: Python can load a
timestamp-valid cached `.pyc` while reporting the corresponding source path.
Before importing any project module, start each study process using a fresh,
private, empty bytecode-cache prefix and the Python options
`-I -S -B -X pycache_prefix=<absolute-empty-directory>`. Isolated mode ignores
user-site and Python environment path settings; `-S` prevents site startup and
its `.pth` hooks, `-B` prevents bytecode writes, and the prefix directs ordinary
importlib cache lookups away from existing project caches. Create the prefix
exclusively for this process and do not reuse it.

Use a standalone standard-library-only bootstrap outside `looking_glass`;
`python -m looking_glass.runner` would import the package too early. Before any
project import, verify isolated/no-site/no-bytecode flags, the expected
`sys.pycache_prefix`, and that the private directory is empty; reject any
project module already in `sys.modules`. Fail closed if any check fails. The
bootstrap must **not** add the checkout root to `sys.path`: an ignored
top-level sourceless `.pyc` could otherwise shadow a standard-library module.
Use an explicit allowlisted source-file loader for the intended project
package and its committed `.py` modules, fail closed on unknown project
modules, and verify each source path and committed hash before import. Install
the source-only finder before executing the package initializer. Add only
explicitly verified dependency package paths after the standard-library paths,
since `-S` suppresses normal site setup. Test stale package caches, ignored
sourceless top-level shadows, and an unknown/sourceless package module
imported from the package initializer on development fixtures.
Isolated mode ignores `PYTHONHASHSEED`; all output-relevant inventories,
aggregations and serializations must use explicit protocol order rather than
depending on set or hash-table iteration order.
Prevent concurrent modifications to project sources and the cache prefix for
the lifetime of the process. This is an operational requirement rather than a
security boundary against malicious same-account processes.

Record the policy and checks, launcher/interpreter settings and source
revision, plus the resolved checkout-relative path and SHA-256 for each project
source imported during every phase, including lazy imports. Verify actual
resolved paths are within that checkout; do not publish private absolute paths.
The launcher and bootstrap
themselves must be committed and included in the source inventory; the external
launcher checks the bootstrap bytes against the committed blob before invoking
it. Require ordinary source-file imports with source-loader provenance
resolving into the intended checkout. Before model work, require each relevant
file to be tracked at that revision and its disk bytes to match the committed
blob. Record runtime and dependency versions and model-state hashes as
required by the protocol. A
reinitialized model or resumed phase must start under the same checked policy.
Any failed check invalidates preparation or execution; preserve failure
evidence where a run record exists.

This controls ordinary project-bytecode cache reuse; it is not a cryptographic
attestation of the process, a security boundary against a malicious process on
the same account, or proof of the bytes executed by PyTorch native libraries,
the interpreter, or other dependencies. An external witness or stronger
attestation would require separate evidence. The earlier pilot's unchanged
record does not retroactively satisfy this policy.
