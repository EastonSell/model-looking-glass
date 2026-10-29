# Version-bound cuBLAS workspace cleanup

`clear_cublas_after_drain(session=..., lease=..., torch=...)` provides the
cleanup sequence observed in the supervised two-token GPU pilot. The owner
must first stop concurrent work and dispose model, input and output references.
The helper requires the actual live `OwnerSession` and matching `DeviceLease`,
plus a verified `cuda:0` runtime identity. It accepts only PyTorch
`2.8.0+cu128` and calls its private `_cuda_clearCublasWorkspaces` API between
CUDA synchronizations, returning allocated/reserved counters before and after.
PyTorch's own test cleanup uses this private API, but it is not a stable
public interface and this helper rejects other versions.

The caller decides whether the counters satisfy its teardown policy, retains
the device lease through process exit, and attests clean only after verifying
its own shutdown requirements. An exception means cleanup was not verified;
do not attest clean. The helper neither proves that all tensor aliases are
gone nor measures every CUDA-context, native-library or external allocation.
See [the observed pilot](GPU_ONESHOT_RESULTS_V2.md) for its specific scope.
