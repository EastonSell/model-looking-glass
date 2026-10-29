# Canonical physical GPU identity

`looking_glass.device_identity` maps a full physical NVIDIA GPU UUID to the
same device lock, unsafe marker, controller lock, and attestation paths for
all processes that use the same configured workspace root. It makes no CUDA
calls and does not query the GPU. The input inventory and post-lease runtime
UUID must come from trusted, current sources; tests use CPU-only fakes.

## Shared root and physical identity

Every service, supervisor, and isolated benchmark must be configured with the
**same existing absolute local workspace root**, outside individual worktrees.
For example, worktrees under one task can all use its common `work/` directory.
The resolver canonicalizes root spelling, case and existing symlink aliases,
then uses the following paths beneath that root:

```text
.looking-glass-devices/GPU-<full-uuid>.lock
.looking-glass-devices/GPU-<full-uuid>.unsafe.json
.looking-glass-devices/GPU-<full-uuid>.unsafe.json.controller.lock
.looking-glass-devices/GPU-<full-uuid>.unsafe.json.attestation
```

The marker and its siblings match the supervisor protocol. Only complete
`GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` physical UUIDs are accepted. Input
case is normalized to uppercase `GPU-` and lowercase hexadecimal. Short UUID
prefixes, indices, MIG identities, malformed strings and duplicate inventory
entries fail closed. Different physical UUIDs get different path bundles.

The workspace root is a deployment setting: this module cannot force another
process to use the same root or detect a separately configured root. The root
and its state directory must be trusted against modification by unrelated
software. On Windows the root must resolve to a local fixed drive; on other
platforms the operator must establish that it is a reliable local filesystem.
Existing state files with multiple hard links, symbolic links, reparse points
or nonregular types are rejected, and root/state directory identities and all
derived names are checked again by `identity.validate_paths()`. This is not
protection against an untrusted actor replacing files between checks.

## Visibility and runtime verification

The supplied inventory is an ordered snapshot of *all* physical GPU UUIDs.
`parse_nvidia_smi_inventory` parses bounded text from
`nvidia-smi --query-gpu=uuid --format=csv,noheader`; it cannot establish the
executable's identity, query success, source trust, completeness, or freshness.
The supervisor must use a trusted executable, check its exit status and bound
its runtime/output. Do not initialize CUDA merely to discover the lock path.

Preflight allows either an unset `CUDA_VISIBLE_DEVICES` on a machine with
exactly one physical GPU, or a mask naming exactly the selected full GPU UUID.
Only CUDA ordinal zero is accepted. Numeric masks, multiple visible devices,
partial UUIDs and MIG are rejected even where CUDA itself might accept them.
The CUDA visibility/order environment is captured and must remain unchanged
through runtime verification. Use the actual child environment, not a fake
mapping, in production.

```python
from looking_glass.device_identity import resolve_device_identity

# Inventory and UUID come from a trusted pre-CUDA physical inventory query.
identity = resolve_device_identity(
    common_workspace_root,
    physical_uuid,
    physical_uuids=physical_inventory,
)

# The supervisor uses identity.controller_path, identity.lock_path, and
# identity.marker_path. Its GPU-owning child acquires the device lease and
# checks the unsafe marker before initializing CUDA. Under that live lease,
# obtain the actual CUDA device UUID and visible device count, then call:
identity.verify_runtime_binding(
    lease=child_device_lease,
    runtime_uuid=actual_runtime_gpu_uuid,
    device_count=actual_cuda_device_count,
)
# Only after successful verification should the owner load the model.
```

`verify_runtime_binding` requires a live `DeviceLease` for this exact derived
lock path, verifies all path invariants, and checks the caller-supplied
runtime UUID, visible count, ordinal, and unchanged environment. A copied
preflight UUID or a model name is not runtime identification. If a reliable
runtime UUID cannot be obtained, stop without loading a model. The resolver
does not track the lifetime of later GPU work and does not substitute for the
supervisor's marker and process-exit protocol.

The identity module alone does not coordinate a GPU. Broker and benchmark
entry points must actually use the same bundle, controller and device leases,
and unsafe marker throughout their relevant lifetimes. External GPU users
remain outside this cooperative protocol. No hardware end-to-end validation
is claimed by these CPU-only tests. This is not a security boundary against
arbitrary code executing inside a trusted participant process.

See NVIDIA's [CUDA visibility and ordering documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
and [CUDA device UUID API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html).
