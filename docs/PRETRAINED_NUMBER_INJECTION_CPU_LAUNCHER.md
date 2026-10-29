# CPU number-injection discovery launcher

This launcher prepares and runs the public discovery phase of the pretrained
number-injection study. It does not draw a held-out split, establish a hidden
site for a selector, or authorize any held-out execution. The source must also
contain the independently reviewed supervisor and `agreement_phase_ledger`.
Use a clean detached checkout of the exact published commit containing all
three components. Keep the checkpoint cache, the source checkout, the phase
ledger, and all study artifacts under trusted single-owner custody.

The procedure is deliberately noninteractive. Each invocation must start a new
process using the direct base interpreter with `-B -E -s -S -P` and an explicit,
fresh empty `-X pycache_prefix=<absolute-directory>` outside the checkout and
site-packages. A separately trusted outer supervisor must exclusively create
each fresh private prefix, check that it is empty immediately before starting
the child, and keep it quiescent through child startup. The child's later check
cannot attest standard-library code that Python may already have imported;
`-B` prevents bytecode writes but does not prevent reads. Pass the absolute
site-packages directory with `--site-packages`.
Set `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_HUB_DISABLE_TELEMETRY=1`, `MKL_CBWR=COMPATIBLE`, and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` in the process
environment. The trusted supervisor must construct a child environment from
only those controls, the four cache/workspace variables below, and the optional
OS startup variables `PATH`, `SYSTEMROOT`, `WINDIR`, `TEMP`, `TMP`, `COMSPEC`,
`PATHEXT`, `USERPROFILE`, `HOMEDRIVE`, and `HOMEPATH`. `PATH` is required.
Every other variable, including unreviewed ATen, NumPy and numerical-library
overrides, is rejected before importing Torch. The manifest records the fixed
controls and a SHA-256 of the OS variable names and values without publishing
their paths; the prepare and run child environments must match. Set
`MLG_STUDY_WORKSPACE` to an absolute workspace
directory, `HF_HOME` to an existing directory inside it, and both
`HF_HUB_CACHE` and `TRANSFORMERS_CACHE` to the absolute offline cache directory
inside `HF_HOME`. The launcher checks the loaded libraries' effective cache and
offline settings; conflicting legacy cache overrides are rejected. The public
runtime manifest records the cache policy without publishing private paths.
The launcher fetches public source and artifact bytes for exact
readback before it installs its runtime network guard; model loading and study
forwards are offline, on CPU. It checks all seven pinned checkpoint files and
the loaded model, all 16 fixed public tokenizer rows and their four continuation
alignments, one-thread deterministic float32 behavior, denormal handling,
and the source again after the forward phase. Source verification uses
bounded raw Git blobs and index checks with Git replacement objects and
fsmonitor disabled; it does not invoke filter-aware Git hashing. The numerical
readback includes the default dtype/device, MKLDNN flags, CPU capability and
the PyTorch configuration digest. The CPU floating-point probe is
finite and does not certify every possible operation.

All subcommands take `--source-commit <full-commit> --site-packages <absolute>`
before the subcommand. Place output files and directories outside the checkout.
The four commands and their ordering are:

1. `prepare --cache-dir <absolute-offline-cache> --runtime-output <new-file>`
   creates a canonical, path-free runtime manifest without loading the model
   or executing a model forward. Publish that exact file at an immutable raw
   GitHub URL under the specified repository and a full commit hash.
2. `plan --runtime <runtime-file> --runtime-url <published-url> --plan-dir
   <study-root/new-directory>` reads back the published runtime and creates a
   fixed exclusive attempt marker in that existing study root before drawing
   32 public random bytes once and writing the version-2 request plan. A
   second plan directory in the same study root cannot redraw the entropy.
   Publish the exact `plan.json` at another immutable raw GitHub URL before
   continuing. The directory's `started.json` records the locally observed
   source/runtime binding and must be retained along with the root marker even
   if the attempt fails. The custodian uses one trusted root for this study.
3. `seal --runtime <runtime-file> --runtime-url <published-url> --plan
   <plan-file> --plan-url <published-url> --custody-dir <new-private-directory>`
   reads back both public files before drawing a separate private 32-byte site
   nonce once. It first creates an exclusive site-draw marker in the original
   local plan directory; a later call with a different custody directory cannot
   draw again from that plan attempt. Keep that original plan directory and
   marker intact through execution. It retains `nonce.bin` and `site-record.json` privately and
   writes the public `commitment.txt`. Publish only that commitment at an
   immutable raw GitHub URL; do not publish the nonce or site record before
   the protocol permits reveal. If the nonce equals the public plan entropy,
   or any other step fails, keep the attempt and do not reroll it.
4. `run --cache-dir <absolute-offline-cache> --runtime <runtime-file>
   --runtime-url <published-url> --plan <plan-file> --plan-url <published-url>
   --commitment <commitment-file> --commitment-url <published-url>
   --custody-dir <private-directory> --attempt-dir <new-directory>
   --phase-ledger <initialized-ledger>` reads back all three public artifacts,
   validates the private custody binding, compares the live runtime to the
   published bytes, then consumes the one-use injection-discovery phase grant
   before loading the model. The ledger initializer must independently verify
   prior owner drainage and account for earlier and orphaned study evidence.
   It may run before or after preparation. The launcher reserves 8 MiB of the
   injection physical share for preparation files created after initialization,
   including runtime, plan, custody, bridge, invoker, failed and orphaned files.
   Before the phase claim, the trusted custodian must inventory these files, verify
   their aggregate is at most 8 MiB and avoid double-counting other reserved
   or previously inventoried paths. When preparation was already inventoried,
   this reservation remains conservative unused capacity.

   The launcher also reserves its start and terminal markers and 1 MiB for the
   run owner's and invoker's receipts and logs inside the physical share. The
   outer owner must enforce that aggregate bound across all of those files,
   including partial and failed files. The fresh bytecode prefix must remain
   empty; the bound is not a filesystem quota on unexpected child-created files.
   If it is nonempty, the attempt is incomplete: retain and charge those files and
   reconcile them before later work. The remainder is available to the
   supervisor for all complete, partial, and failed evidence. A consumed
   grant is never reused, including when the process stops before a marker is
   saved. An attempt that never receives a claim still retains its evidence;
   charge and reconcile it within the appropriate preparation reservation or
   prior/overhead inventory before later work.

The command emits only a public status and hashes. A successful supervisor
result is `complete-local`: the independent numerical and custody audit,
publication of frozen predictions, and any later held-out work remain separate
steps. Before publishing a prediction, the trusted outer owner must independently
confirm a zero child exit, the supervisor's completed record, the matching
`complete-local` launcher terminal, and the independent audit. A journal
inspector's `complete-prefix` result alone does not certify a successful child
or supervisor completion; a failed or ambiguous attempt cannot be promoted.
Keep the private custody directory, private supervisor records, local
exception objects, and tracebacks away from the investigator and from public
logs. Command-line errors are sanitized; the underlying Python API is a trusted
custodian interface and its exceptions can retain private frame locals. Cleanup
and marker writes after failure are best effort. If publication or interruption
leaves the outcome uncertain, retain the evidence and reconcile it; do not
silently retry or infer completion from an individual file.

The public readbacks establish byte equality at the time observed by this
process. They do not independently prove publication time, random-source
independence, source execution, or security against another process running as
the same user. The custodian must keep the model quiescent and exclusive for
the case and independently document chronology and publication. The interpreter,
site-packages, native libraries and trusted executable path must remain trusted
and unchanged across preparation, discovery and any later held-out run. Package
versions and the selected runtime hashes do not attest every dependency or
native-library byte. Hashes do not
make this cooperative boundary an access-control mechanism. The one-use draw
markers assume the original study root and plan-attempt directory are retained
by that trusted custodian: copying or recreating an entire root defeats this
cooperative safeguard and is not permitted by the procedure.
