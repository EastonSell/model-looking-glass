# Natural agreement: bounded CPU discovery

This launcher is limited to the 16 public, exposed discovery inputs from the
reviewed preflight. It does not create a held-out nonce, held-out cases, or
injection. Before a study forward it verifies the exact protocol, the separately
published natural prediction clarification, public
manifest, passed preflight result and archive, offline checkpoint assets, and
fresh tokenizer alignment including each appended contrast. It binds the whole
committed project Python source, runtime and case mapping to a published file.
The executor captures all donors afresh in the actual CPU owner process.

Use a clean, reviewed, publicly accessible source commit that already includes
all analysis and held-out scoring code required by the protocol. Prepare the
binding with the actual base Python interpreter (not the Windows venv
redirector), bytecode and site startup disabled, a new nonexistent absolute
pycache prefix, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=1`, and
`MLG_NATURAL_SITE` set to the virtualenv's `Lib/site-packages`. The explicit
bootstrap adds that directory without executing `.pth` files. Its interpreter,
site-path fingerprint, virtualenv configuration and dependency versions are bound:
The same invocation with `check-runtime` in place of `prepare` checks the
direct interpreter and virtualenv without loading a model.

```
BASE_PYTHON -I -S -B -X pycache_prefix=ABSENT_ABSOLUTE_PATH -c "import os,runpy,sys; sys.path.append(os.environ['MLG_NATURAL_SITE']); script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script,run_name='__main__')" tools/natural_agreement_cpu.py prepare --cache CACHE --binding ABSOLUTE_BINDING_PATH
```

Publish the exact binding bytes under `results/` at a GitHub commit before any
study forward. Run from the same source commit and runtime using a new output
directory outside the checkout. Publish the binding from a separate worktree or
commit; keep the execution checkout on the original source commit, because the
binding records and rechecks that exact `HEAD`:

```
BASE_PYTHON -I -S -B -X pycache_prefix=ANOTHER_ABSENT_ABSOLUTE_PATH -c "import os,runpy,sys; sys.path.append(os.environ['MLG_NATURAL_SITE']); script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script,run_name='__main__')" tools/natural_agreement_cpu.py run --cache CACHE --binding ABSOLUTE_BINDING_PATH --published-url COMMIT_PINNED_RAW_GITHUB_URL --output NEW_ABSOLUTE_OUTPUT_DIRECTORY
```

The launcher checks that the published bytes and core source files are
retrievable at their commit IDs. The parent and owner hold read-only Windows
source-file leases, recheck source, and each independently bind runtime/assets.
The parent verifies committed bytes using Git before the owner starts; the
owner verifies the exact frozen source inventory under its own lease without
starting Git or other study subprocesses. A Python audit hook rejects subprocess
creation before importing Torch; native code is outside that hook and the job
still fails closed if its lifetime process inventory cannot be verified.
The Windows platform-version probe uses the process-observed Win32 version
instead of starting a shell, including when Torch queries the architecture.
The binding separately records the Win32 platform version; these process-visible
values can reflect Windows compatibility mode and are not an OS attestation.
In the cooperating supervisor path, the gated owner waits until the supervisor
has a process handle and assigns it to a kill-on-close Windows job before
loading the model. Do not invoke `child` directly. Startup, execution,
logging and storage are bounded. The process has a two-hour deadline and each
log retains at most 1 MiB; log overflow is incomplete, with truncation reported.
The gated `validate-child` mode uses that same runtime, source lease, public
binding, checkpoint asset and tokenizer checks, then exits without model loading
or a study journal. A supervised validation-only rehearsal is not discovery.
Source/asset checks and leases are cooperative operational controls, not
protection against a hostile actor with the same account. Third-party Python
and native dependency code is identified by runtime version and build metadata,
but is not byte-attested by the project source lease. A digest or local clock
does not independently attest publication order or faithful execution.
Project imports are resolved from source without adding the checkout root to
the general Python import path. If job cleanup cannot be verified, the parent
lease is released when the launcher exits and protection of any surviving
descendant is unverified; stop and resolve the process state before retrying.

Every attempt uses a new directory. Failed, timed-out or uncertain attempts,
their available raw journal, logs, and supervisor record are preserved. If
run validation fails before starting an owner, `launcher-terminal.json` records
the phase and `launcher-error.txt` retains the bounded traceback; owner evidence
is under `owner/`. If
`cleanup_verified` or `retry_safe` is false, stop and independently verify
that no previous owner or job remains before retrying. A lost process handle is
reported unreaped even though the bounded gate prevents it from commencing a
study. A crashed supervisor may leave no terminal record; that is incomplete.
The launcher terminal records the SHA-256 of its fsynced supervisor record;
the independent consumer must check that digest before using both artifacts.
The supervisor counts all actual tensor-directory files, including unreferenced
and partially published files, after verified cleanup. With uncertain cleanup it
does not read live files; accounting stays pending. Preserve and reconcile them
before any later study attempt.

`evidence_complete` means that a zero-exit owner was reaped and its retained
event order, case/donor coverage, witness shapes and references, storage hashes,
reported controls and prediction inventory passed the controller's structural
checks. It is not numerical validation of landing, untouched bytes, tolerance,
effects or selection; those require a separate independent raw-evidence audit.
The protocol's 512 MiB study-wide retained-tensor limit includes the 3,216,448
unique bytes of public preflight logits. Identical checkout or archive copies
of that one logical payload are counted once; toy fixtures and unrelated prior
mechanics tests are not pretrained agreement-study evidence. The checked
discovery owner is capped at 128 MiB. A conservative fixed-five-token estimate
is 66,663,296 bytes without deduplication. The binding reserves prospectively
160 MiB for natural held-out, 48 MiB for injection discovery and 72 MiB for
injection held-out, leaving 105,835,456 bytes for failures or extra evidence.
Those future caps require enforcement and reconciliation by their respective
owners; the binding does not perform that work. The 160 MiB estimate covers
the natural held-out 32-token protocol limit, and injection reservations cover
the current no-deduplication estimates. Separately, a prospective 1 GiB physical
host-disk allowance reserves 224/224/128/192 MiB for those four phases and leaves
256 MiB for preflight copies, failures and overhead. It includes serialized
copies and metadata, and requires custodian reconciliation and enforcement; it
is not a GPU memory reservation. Before launch the owner checks at least 1 GiB
free disk and exclusively writes a local one-shot claim adjacent to
the binding. The claim prevents reuse of that local binding, but copies can
bypass it: the custodian must coordinate a single study owner externally, and
source read leases are not a study-wide lock. This first-attempt launcher binds
the known preflight as the only prior payload and cannot itself build a
reconciled retry binding after an attempt retains tensors. A retry requires a
separately reviewed accounting/source update, reconciliation of every retained
file including orphans and partial files, and a new published binding and
reservation. Publish independent
discovery validation and prediction freeze before any held-out allocation.
Discovery inputs are exposed, and passing this small study does not establish
a general agreement circuit.
