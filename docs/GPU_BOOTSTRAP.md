# GPU bootstrap: CPU-tested building blocks

The two-token DistilGPT2 resource bootstrap is **not yet runnable**. The code
in `looking_glass/gpu_bootstrap.py` supplies a private evidence journal, a
direct-child launcher and an owner startup guard. No GPU measurements, approved
resource profiles or service admission evidence are produced by these pieces.
The complete protocol is `docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md` when that
protocol is integrated.

The parent must first obtain a canonical source manifest from a CPU-only
preflight of an immutable, clean detached checkout and the selected virtual
environment. `prepare_owner_launch` checks the manifest structure and base
interpreter bytes, prepares a bounded file-origin index, creates a fresh empty
bytecode prefix, and constructs an explicit environment and controlled PATH.
The trusted controller must itself start from the same verified source and
revalidate the snapshot before launching; this module does not establish that
controller provenance by itself. The parent must use the supervisor's explicit
replacement environment (`inherit_environment=False`), and call
`OwnerLaunch.verify_ready()` immediately before the spawn.

The child starts the base interpreter with `-B -S -P`, no inherited Python
path, and a fresh private bytecode prefix. A pre-import finder checks the
source index, verifies and compiles the exact approved Python bytes, checks
extension origins and hashes, and refuses unapproved module loaders. It also
checks distribution metadata against that index. It preserves built-in and
frozen importer precedence and allows a genuinely missing optional module to be
reported as absent. `assert_import_guard(source_sha256)` rechecks the manifest
binding and the approved path and cache. The launcher and preflight assume
cooperating trusted code and a protected snapshot: they are not a sandbox
against code that rewrites interpreter internals or another process that
changes files while they are being read. Standard-library, native dependency,
OS and driver binaries are not fully covered by the source manifest.

`claim_owner_and_load_dependencies` is explicitly disabled by default. It
requires an identity-aware supervisor
session and its live device lease before importing CUDA-capable dependencies.
It records the child run ID and PID, retains its lease until process exit, and
checks that the complete set of `torch/lib/*.dll` files matches the verified
manifest and hashes before importing Torch. It rejects already-imported Torch,
Transformers and other listed CUDA-capable packages, though it cannot undo any
device activity that occurred before that check. It
rejects unexpected CUDA initialization during Torch import, and verifies the
runtime UUID and device count under that lease before importing Transformers.
Any uncertain failure remains quarantined for the supervisor; this primitive
does not attest teardown or claim that memory has been freed.

The parent journal publishes `terminal.json` only after a supervisor result is
available. A clean terminal must match the child-produced owner record. When no
reaped result exists after a possible launch, the run remains quarantined and the
terminal uses null supervisor and exit fields. Successful publication also
requires the independent prospective artifact audit; a missing auditor or an
incomplete candidate fails closed. Atomic no-replace publication preserves the
first terminal even if the process is interrupted immediately afterward.
An unknown result defaults to quarantine; a trusted parent can explicitly
classify a prelaunch failure only when it knows no child was started.

CPU tests exercise hostile startup paths, inherited environment removal with
a real supervisor, stale and sourceless bytecode, metadata and import guards,
owner-record binding and failure publication. The device tests use fake
dependencies and UUIDs: they do not initialize CUDA, load a model or establish
GPU correctness. Run them with the shared environment and CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m unittest discover -s tests -p test_gpu_bootstrap.py -v
```

Before scheduling a real isolated window, the remaining work includes the
reviewed worker admission and output-storage fixes, exact-source live-key and
model verification, measured calls through `ResidentWorker.execute`, bounded
output and wire checks, exception-safe synchronized disposal, all five
observations and admission records, profile assembly, prospective audit, and
independent review of their integration. Profiles can authorize only the same
verified runtime and source snapshot; a later source change requires a new
measurement.
