# Five-token GPU trace: retained-byte successor

This prospective, separately admitted successor repeats the fixed workload in
`PRETRAINED_GPU_TRACE_PILOT_V1.md`. It is a correctness experiment, not a
latency benchmark, model-loss gradient claim, or reusable capacity profile.
The previously published hash-only pilot and its failed first attempt remain
unchanged. Publish this amendment and the exact source before a new forward.

The prior pilot's prohibition on retaining raw outputs is superseded only for
this fixed successor's private E: attempt directory. The public result remains
redacted and hash-based; do not publish raw logits, activations or weights.
The same public five-token input, pinned checkpoint, seed, evaluation/runtime
settings, ten-forward order and trace sites apply. Each of two reference,
two direct and two auxiliary-gradient forwards captures nine sites: attention,
MLP and block outputs at layer indices 0, 2 and 5, for token positions 0 and 1.
ID mode has metadata only. The auxiliary-gradient probe is not a gradient of
the model output or loss.

After the guarded device operation and CPU offload have synchronized, save the
actual CPU float32 bits in little-endian order, without numeric conversion of
their values. Save ten full-logit files of shape `[1, 5, 50257]` (1,005,140
bytes each). Save two files per value-bearing mode, each concatenating its nine
ordered `[1, 2, 768]` event values (55,296 bytes per file, 6,144 per event).
The fixed raw payload is exactly 10,383,176 bytes across 16 files. The manifest
is limited to 65,536 bytes and binds the file hashes, byte lengths, forward and
mode order, event offsets, layer/module/selector and token positions. Every
tensor's shape, dtype, CPU location and finiteness are checked before it is
serialized. A new manifest is written only after the complete fixed inventory;
the candidate binds its hash and payload size.

Each file uses an exclusive partial name, flushes through `fsync`, and is
published with a create-only link. A failed or interrupted write may leave a
partial file; retain it, stop the owner, and never treat it as complete. A
forward or capture already saved before a later failure is unverified partial
evidence, not a successful candidate. File sync is not a directory or
power-loss durability guarantee. The normal candidate validator reads bounded
saved bytes, checks each file and event binding against the existing forward
and trace records, rejects nonfinite or inconsistent records, and compares the
ten full-logit files and all 36 direct/reference and gradient/reference event
pairs byte for byte. This includes signed zero and subnormal bit differences.
An independently implemented saved-byte reader must repeat these checks after
the owner and direct controller have exited before any result is claimed. Its
replay checks retained-byte consistency; it does not authenticate execution or
independently rerun the model. The private attempt is assumed quiescent through
replay; hash checks detect inconsistent retained bytes but do not lease them
against an external writer.

The device commitment, 1.5 GiB PyTorch allocator ceiling, 128 KiB logical GPU
trace charge, device reserve, phase-boundary synchronization and free/peak
observations remain as in the prior pilot. The logical host trace/evidence
charge increases to 4 MiB. For the fixed geometry, the three retained capture
sets total 331,776 bytes; their pieces plus a joined capture buffer can overlap
by up to 110,592 bytes, and two full-logit serialization buffers can account
for another 2,010,280 bytes. Up to six retained full-logit tensors total
6,030,840 additional bytes outside that charge. These counts exclude other
Python/Torch objects, temporary or native allocations and process RSS; the
logical charge is not a physical memory reservation or a bound on them.

Create the raw directory inside the new E: attempt along with the existing E:
temporary/cache and checkpoint-staging directories. The 3 GiB output-volume
and 1 GiB workspace-volume checks are fresh free-space points, not reservations;
the 10,383,176-byte payload and 65,536-byte manifest are in addition to the
verified checkpoint staging copy, diagnostics, filesystem overhead and uncapped
trusted stdout/stderr. The canonical physical lease and unsafe-marker namespace
stay in the existing shared workspace. The prior direct-child/source binding,
owner/lease checks, synchronized disposal and failure quarantine apply. No
automatic retry follows an uncertain operation. A future attempt needs its own
reviewed source, fresh host/volume/device admission and coordinated device slot.
