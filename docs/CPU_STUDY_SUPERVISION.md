# Cooperative CPU study supervision

This helper extends the reviewed direct-child supervisor to one trusted,
CPU-only experiment process. It uses an existing persistent **CPU study
custody root**, distinct from GPU ownership state. It coordinates only
cooperating CPU study launchers using that same root. It is not a sandbox,
does not control other applications, and makes no claim about GPU state, GPU
availability, GPU isolation, or GPU teardown.

The root must be an absolute, existing directory on a trusted local filesystem.
All processes sharing a study must use the same canonical root and preserve
the fixed cpu-study.lock, cpu-study.unsafe.json, controller lock and
attestation files. Never delete or replace the persistent lock files; a
remaining marker or attestation blocks the next owner. The root and all its
ancestors must remain stable and exclusively trusted throughout the run:
replacing or renaming a directory can split the lock and marker namespace.
This helper does not authenticate or pin directory identity. Different study
roots are independent and do not coordinate with each other or the GPU lease.
Choose a root dedicated to this study, separate from the physical GPU ownership
root; the API cannot discover every alias or enforce organization-wide root
selection. The underlying supervisor checks the lock identity and refuses
hard-link aliases.

The compatibility marker CPU_STUDY_SUPERVISION_CONTRACT is
cooperative-cpu-study-v1. The helper requires the reviewed
SUPERVISOR_LAUNCH_CUSTODY_CONTRACT value gated-rooted-child-v1 before it
starts or admits a child. These markers identify cooperative compatibility
contracts, not security proofs.

## Parent and child

The trusted parent supplies a direct owner command and a separately reviewed
source/environment binding. With inherit_environment=False (the default),
env is an explicit replacement environment; supply required operating-system
variables and the trusted module search path as needed. Do not include
untrusted PYTHONPATH, PYTHONHOME, preload hooks, or site startup code.
The owner must be a fresh direct child. These helpers do not validate a model
checkpoint, study protocol, interpreter, installed dependencies, or the
honesty of the command or environment.

On this Windows host, launching the shared virtual environment's python.exe
can start an intermediate launcher, so the executing interpreter is not the
supervisor's direct child and the owner check rejects it. Use a verified direct
base interpreter and add a source-bound trusted dependency directory only
after the CPU claim; verify the process relationship on the actual host.

    from looking_glass.cpu_study_supervision import run_cpu_study

    result = run_cpu_study(
        owner_command,               # a direct, trusted CPU-only child
        study_root=study_root,       # existing absolute persistent CPU root
        timeout=600,
        env=trusted_environment,
    )
    if not result.clean or result.exit_code != 0:
        raise RuntimeError("CPU study did not finish with verified teardown")

At the very start of the trusted owner, before importing Torch, Transformers,
or any other CUDA-capable model dependency, call:

    from looking_glass.cpu_study_supervision import claim_cpu_study

    session, lease = claim_cpu_study(study_root, lease_timeout=30)
    # Only now import and use model dependencies. Keep model, inputs and
    # outputs on CPU and do not initialize or invoke CUDA.
    # ... perform the source- and protocol-bound experiment ...
    # Stop work and drop all study-owned resources before attesting.
    session.attest_clean(lease)
    # Retain both session and lease until this direct owner exits.

The claim waits for the private launch grant, then requires the exact
identity-less session, the derived canonical lock and marker paths, an empty
CUDA_VISIBLE_DEVICES value, and no already-loaded module in its known list of
CUDA-capable dependency names (including submodules). It acquires and checks
the CPU study lease before returning. A missing/mismatched mask, different root,
identity-aware GPU session, or early dependency causes failure before study
admission. The parent supervisor supplies the empty mask. The list and mask
are cooperative preflight checks: they are neither exhaustive import guards
nor proof that later code cannot change the environment or use a GPU. Trusted
study code must ensure all work remains on CPU. The parent does not prove the
child used this helper; source binding and independent review of the actual
owner entrypoint are required before a study gate is enabled.

After successful work the child must stop and drain CPU study work, dispose
relevant resources and attest clean while retaining the exact lease through
process exit. A workload exception is not a clean teardown by itself. A parent
result marked clean means the cooperative owner attested, the direct child
exited, and the study lease was reacquired; a nonzero exit still means the
workload failed. On any uncertain startup, acquisition, teardown, cancellation,
or lost return, terminate the child or supervise/reconcile it and retain the
marker. In particular, a lost return from claim_cpu_study can leave an
acquired lease in the child; that child must not continue work and should exit
without attesting. Timeout, an available lock, or garbage collection is not
proof of teardown.

On Windows, the parent may additionally pass a fresh, empty, independently
reviewed `process_job` capability to `run_cpu_study`. The supervisor assigns
the exact direct child to it before the launch grant and requires confirmed
whole-job drain before clearing the study marker. A live descendant, uncertain
assignment or drain, or unavailable accounting leaves the marker sticky.
These checks are cooperative and do not attest processes that escaped or were
never in the job. Without a job, `clean` still covers the direct child only.
The caller must keep the job and associated source protections in trusted
custody: never close or reuse a job retained by a pending supervisor recovery
record. Errors before supervisor adoption may leave the caller responsible for
the unused job; inspect and reconcile any uncertain outcome under the exact
supervisor recovery contract before releasing protections.

The launch gate, exact-child recovery registry, startup-channel cancellation
ordering, bounded gate deadline, unknown native-spawn outcomes and
asynchronous-interruption limitations are described in
[SUPERVISOR.md](SUPERVISOR.md). A run_cpu_study timeout covers the post-launch
child wait, not a stalled process creation or grant write.
