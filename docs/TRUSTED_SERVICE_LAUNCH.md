# Fixed launch settings for the two-token service pilot

`looking_glass.trusted_service_launch.prepare_trusted_launch` prepares a
direct-child command and a replacement environment for either the fixed
two-token service owner or its matching resource collector. It does not start
the child, acquire a GPU, validate a profile, or authorize a workload. The
service, collector and measured profiles must still be integrated and
independently reviewed before use.

The helper selects the base Python interpreter with `-B -S -P`, appends the
same absolute checkout and virtual-environment site-packages paths in the same
order, and passes an absolute private config path plus its file identity and
SHA-256 to an allowlisted owner module. Each owner must call
`read_trusted_config` and parse the returned captured bytes before GPU activity;
this detects a config replacement or byte change after the parent preflight.
It fixes the selected full physical GPU UUID, native allocator,
cuBLAS, thread and offline settings, and requires an explicit existing
`HF_HOME` directory rather than inheriting the user's default cache location.
The service and collector must put their offline checkpoint cache beneath that
directory and use the same path. It omits inherited Python paths and
unreviewed CUDA/numerical knobs. When `PROGRAMFILES` is present, it preserves
only a canonical existing directory so the Windows driver inventory utility can
locate its dependencies. Its value, including absence, is part of the observed
runtime identity; the collector and service must match. This does not establish
the cause of any previous driver failure. Use the returned mapping as a **replacement**
environment with the identity-aware `run_supervised`, and call
`verify_ready()` immediately before spawning. The config is an ordinary,
bounded private file containing paths and protocol settings, not bearer
tokens; readiness credentials travel through the separate startup channel.

Both collection and service must use the same frozen checkout path, runtime
paths, interpreter and settings to produce matching `trusted-service-*` keys.
Hashing these inputs records their identity under a trusted-process assumption;
it is not a proof of all native libraries or interpreter state. The source,
environment and private paths must remain under exclusive trusted control
throughout the run. This helper is not a sandbox against a hostile same-user
process or code that mutates interpreter internals.

In particular, `-B` prevents writing bytecode; it does **not** prevent
reading a `.pyc` that appears in the fresh prefix after the prelaunch check.
The empty-prefix check cannot attest which bytecode executed. Protect the
prefix and source through child imports and record bytecode execution as
unattested. A source-only loader or OS-enforced custody would require its own
review. Matching hashes and runtime versions alone do not prove universal
reproducibility or bound GPU memory.

CPU tests check the collector and service receive identical interpreter paths,
that unexpected environment variables are excluded, that an altered config or
populated prefix fails prelaunch verification, and that the child read rejects
a post-check config change. They do not initialize CUDA or establish source
integrity after the check.
