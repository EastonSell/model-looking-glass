# CPU injection discovery outer owner v1

This Windows-only, one-shot parent runs one `prepare`, `plan`, `seal`, or `run`
command of the CPU injection launcher. It is a process and evidence-custody
control, not a new scientific protocol or proof of a causal explanation.
Follow the published number-injection protocol and its publication, ledger,
site-custody, model, and held-out gates separately.

## Trusted operating boundary

Use a protected, quiescent study root and a clean, detached checkout of the
published full source revision. The operator must independently verify the
outer script's working bytes against the published Git blob **before** starting
it. Start a fresh direct base-Python process with `-B -E -s -S -P`; do not reuse
the parent interpreter. Trust the base Python, Windows APIs, Git on the chosen
`PATH`, installed dependencies, and the same-account custodian. The outer
script's before/after byte checks and Python executable digest do not prove
which already-started outer source or native dependency bytes executed. The
checkout and paths must remain quiescent throughout the child run; a transient
same-account replacement between checks is outside this control.

The owner checks its own, gate, and launcher checkout bytes against the named
commit before and after the run. The child executes the verified gate bytes
passed in its command and, after the grant, reads the launcher, verifies its
digest, and compiles exactly those bytes. Other project files are subsequently
verified by the inner launcher; these checks are not file leases. The child
gets only the owner's explicit CPU, offline-cache, thread, numerical, and OS
environment values. Its empty bytecode prefix and workspace-contained cache
are private. No site or held-out records should be inspected by the reviewer.
Before a discovery run, the owner also checks at least 1 GiB free on the
actual cache filesystem and rejects preexisting `looking-glass-stage-*`
entries there. The pinned loader temporarily stages about 353 MB there, which
may be on a different volume from study evidence. The owner checks for such
entries again after the child; a remaining stage makes the attempt incomplete
and must be retained and reconciled separately. These are point-in-time
checks, not a filesystem reservation or an enforced disk quota.

For each invocation provide absolute paths to `--repo`, `--source-commit`,
`--base-python`, `--site-packages`, `--workspace`, `--hf-home`, `--cache-dir`,
`--output-dir`, and a bounded `--timeout` in seconds, followed by `--` and the
inner command and its arguments. The parent inserts the inner source-commit
and site-packages options. For example, the final argument tail for discovery
is `-- run --cache-dir <cache> --runtime <runtime> --runtime-url <url> --plan
<original-plan.json> --plan-url <url> --commitment <commitment>
--commitment-url <url> --custody-dir <private-dir> --attempt-dir <fresh-dir>
--phase-ledger <ledger>`. `prepare`, `plan`, and `seal` use the options in the
inner launcher's help. Keep each outer output directory fresh and under the
workspace, outside source, dependencies, and `HF_HOME`, and disjoint in both
directions from declared child inputs and artifacts. The original plan file
path matters: its parent is the one-use plan-attempt directory. Inline
`--option=value` paths are supported; abbreviated child path options and
duplicate run attempt paths are rejected before launch.

## Process, result, and storage boundaries

The gated child cannot run the launcher until both capture threads have
started and its assignment to a kill-on-close Windows Job is verified. The
parent reaps the direct child and checks zero *active* Job members. This is
not a lifetime process inventory: the launcher legitimately creates short
Git subprocesses. Timeout, setup, capture, source, cleanup, prefix, or
completion uncertainty yields an incomplete attempt. An interrupted or
ambiguous native spawn remains ungranted; its exact process identity can be
unknown, and the gate expires after 60 seconds. Exit the one-shot parent,
independently establish that the process tree is quiescent, preserve the
attempt, and reconcile storage before any new attempt. An uncertain Job,
direct child, or reader remains rooted in a preallocated in-process owner
context and blocks another owner in that interpreter. A hard parent death or
inability to create a terminal can leave
no terminal receipt; absence is never success.
Where a terminal can be written, fixed site-independent reason codes identify
the failed owner operation or check without persisting exception text. The
first cancellation is preserved; a fresh cancellation takes precedence over
an ordinary error, and cleanup exception chains are retained where available.
The CLI retains nonzero integer `SystemExit` codes, maps other exit payloads
to failure, and never prints their potentially private contents.

For `run`, a success receipt means zero child exit, confirmed empty Job,
clean local owner checks, and matching child-report, launcher-terminal, and
supervisor-completion prediction/archive digests and 128-record counts. It
does **not** independently hash or audit the actual saved prediction or
archive. Publication still requires the outer process's independently observed
zero exit, the launcher's successful postchecks and terminal, publication
ordering, and a separate raw public archive numerical audit. For `prepare`,
`plan`, and `seal`, the owner reads the local runtime, plan, or commitment
and checks it against the child report. Publication, remote readback, and
protocol checks remain separate. A saved outer `succeeded` receipt alone is
insufficient if the outer process did not itself exit successfully: a write
or close could fail after its bytes reached disk.

The parent's fixed private files are `started.json` (at most 16 KiB), `grant`
(at most 64 bytes), `stdout.bin` and `stderr.bin` (each at most 256 KiB), and
`terminal.json` (at most 16 KiB): at most 557,120 bytes in total, within the
1 MiB reserved by the injection launcher. Captured log overflow is discarded
after draining and makes the attempt incomplete; it is classified after the
child exits rather than stopping an already-running study. Saved byte counts and hashes
describe bounded bytes actually read back from disk; null means unavailable.
Readback after a failed write does not prove `fsync` or close succeeded. Keep
all detailed receipts and logs private: diagnostics and their hashes could
reveal information about the blinded site. Only a separately constructed,
site-independent status may be shared before the prescribed reveal.

This is a cooperative bound on those parent-created files, **not** an OS disk
quota or a bound on arbitrary child/dependency writes. In particular, `-B`
does not prevent an explicit write into the retained `pycache/` directory.
Any nonempty prefix makes the attempt incomplete; retain and conservatively
inventory its bytes, including any surplus, against the overall study physical
allowance before another phase. Before the injection phase claim exists,
prepare/plan/seal files and their failures are charged to the reviewed 8 MiB
preparation carve-out when it applies, or otherwise to reconciled
prior/overhead. They are not silently treated as covered by the run phase.
Console redirections or other
operator-created files also require separate accounting. No unexpected or
partial evidence is deleted automatically.
