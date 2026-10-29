# Synthetic v2 runtime and model preflight

`looking_glass.causal_v2_runtime_preflight.inspect_runtime_and_models()`
measures the proposed study's fixed CPU runtime and the state SHA-256 of each
of its six freshly initialized `TinyResidual` models. It uses no input cases,
site schedule or model forwards. It requires the prescribed Python and PyTorch
versions, `CUDA_VISIBLE_DEVICES=-1`, an uninitialized CUDA runtime, float32
defaults and the checked single-thread denormal-preservation policy. It
constructs models explicitly on CPU and restores the caller's CPU RNG state.
The returned numeric policy belongs to the invoking process and thread; the
study callback must recheck model state and numeric policy during execution.
`inspect_backend_settings()` separately measures the inter-op thread count,
float32 matmul precision, reported CPU dispatch capability, MKLDNN and deterministic-algorithm settings, and six
bounded OMP/MKL/OpenBLAS/NumExpr environment values. The preflight compares
these snapshots before and after its model initializations. A launcher must bind
the separately returned settings into the pre-manifest and recheck them
throughout the run; the eight-field runtime record alone is incomplete for
reproducibility. A transient change restored between checks is not detected.
The future launcher should take the settings snapshot before calling
`inspect_runtime_and_models()`, compare it with a new snapshot immediately
afterward, then merge the verified snapshot into its public runtime record.

The runtime record contains the platform, versions, byte order, intra-op
thread count, preservation result and a SHA-256 of the textual PyTorch build
configuration. That digest identifies the reported configuration; it does not
authenticate interpreter, dependency or native-library bytes. Source and path
verification, private/public custody, exclusion checks, evidence storage and
the decision to prepare cases remain responsibilities of the independently
reviewed launcher and custodian. This function is not a study entry point and
does not establish that a study is ready.
