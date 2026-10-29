"""Version-bound cuBLAS workspace cleanup for a drained GPU owner."""

from .device_lease import DeviceLease
from .supervisor import OwnerSession


def clear_cublas_after_drain(*, session, lease, torch):
    """Synchronize and clear PyTorch 2.8's cuBLAS cache under the live lease.

    Call only after stopping concurrent work and disposing model/input/output
    references. This does not attest clean, release the lease, prove that all
    CUDA or native allocations are gone, or promise that counters reach zero.
    A caller must inspect the returned counters and apply its own teardown
    policy while retaining ownership through process exit.
    """
    if type(session) is not OwnerSession or type(lease) is not DeviceLease:
        raise ValueError("an actual supervised owner and device lease are required")
    OwnerSession.check(session, lease)
    if (session.identity is None or session.identity.cuda_device_index != 0
            or getattr(session, "_runtime_verified", False) is not True):
        raise RuntimeError("a verified runtime identity for cuda:0 is required")
    if (torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8"
            or not callable(getattr(torch._C, "_cuda_clearCublasWorkspaces", None))):
        raise RuntimeError("unsupported PyTorch cuBLAS workspace cleanup API")
    if torch.cuda.is_initialized() is not True:
        raise RuntimeError("CUDA must already be initialized by the verified owner")

    def snapshot():
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        if (type(allocated) is not int or type(reserved) is not int
                or allocated < 0 or reserved < allocated):
            raise RuntimeError("invalid PyTorch allocator counters")
        return {"allocated_bytes": allocated, "reserved_bytes": reserved}

    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()
    before = snapshot()
    OwnerSession.check(session, lease)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()
    after = snapshot()
    OwnerSession.check(session, lease)
    return {"before": before, "after": after,
            "api": "PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces"}
