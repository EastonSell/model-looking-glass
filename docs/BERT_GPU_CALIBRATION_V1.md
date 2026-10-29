# DistilBERT ten-token CUDA calibration, v1

This is a **development-only resource and intervention-delivery calibration**.
It uses two already exposed public rows, not the agreement study's evaluation
cases. It is neither a latency/throughput benchmark nor a bound for the full
140-forward-per-family study. It does not establish a causal explanation of
agreement or generalize an observed peak to other shapes, sites or schedules.

## Fixed workload

Use `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, with its five pinned local files,
verified before launch and again while copied to a private staging directory.
Load only from that directory, with remote code and downloads disabled. The
exact fast tokenizer must reproduce the two rows and the single-token output
IDs. Load one float32, eager, evaluation-mode `DistilBertForMaskedLM`; require
the input embedding and vocabulary projector to remain the same `Parameter`
after transfer. The explicit PyTorch seed is 17. Isolated Python startup
ignores `PYTHONHASHSEED`; no fixed Python hash-seed claim is made.
The record includes readbacks of deterministic algorithms, thread counts,
TF32/cuDNN settings, conversion flags and the selected execution device, as
well as Python, PyTorch, Transformers, tokenizers and safetensors versions.

The target `The farmer near the nurses [MASK] ready.` is
`[101,1996,7500,2379,1996,11500,103,3201,1012,102]`; the donor changes the
subject to `farmers`, yielding
`[101,1996,6617,2379,1996,11500,103,3201,1012,102]`. The mask is at index 6;
the selected output IDs are `is=2003`, `are=2024`. Each input is one row of
ten tokens with an all-ones attention mask. The intervention site is layer 2
(`distilbert.transformer.layer.2`, tuple output element 0, invocation 0,
batch 0, token 2), a 768-element vector. Four sequential forwards run on one
resident model:

1. Capture the target vector and mask-position vocabulary logits.
2. Capture the donor vector and mask-position vocabulary logits.
3. Capture, self-patch from the same forward, and recapture the target vector.
4. Capture, patch from the donor capture, and recapture the target vector.

All four observations and their captures remain live through the final stage
snapshot. Require a donor vector different from the target, exact raw-bit
landing for the self and cross patches, and a repeated target before-vector
equal to its baseline. Require the self-patch mask-position **full vocabulary**
logits to match the baseline by float32 bits. The complete logits are checked
finite during each forward, but only the mask-position full-vocabulary digest,
the two selected logit bit patterns and decoded values, capture/landing vector
bytes and intervention metadata are saved. A reader cannot reconstruct the
unsaved complete mask logits, independently re-run the raw-byte comparison,
or independently link the two saved selected values to a mask digest. This is
a delivery/control check, not evidence that
the selected site uniquely performs a linguistic computation.

## Device, memory, and ownership

The parent obtains one physical-device identity using `nvidia-smi`; the child
is pinned to that UUID and must verify the runtime UUID and exactly one
visible CUDA device. The common physical-device lease is held from before
CUDA-capable imports until child exit. The CUDA context must be initialized to
read its identity and telemetry; that bootstrap happens under the physical
lease but before the software memory commitment, so it is not a measured
pre-context allocation bound.

Verify the active PyTorch allocator backend is `native` and record its
readback, then set its limit to 25% of reported device total. Require
the device-property and telemetry totals to agree. Before model work, hold a
`MemoryBudget` commitment of the full allocator cap plus 256 MiB outside
allowance, a 64 KiB aggregate GPU trace claim, and a 1 MiB logical host trace
claim. At admission and synchronized stage boundaries, require free memory and
unreserved capacity to leave the commitment plus `max(1 GiB, 15% of total)`;
the commitment is conservatively counted again after allocations materialize.
Require at least 256 MiB of allocator headroom and at least 3 GiB initially
free. Recheck the bound total and lease around CUDA telemetry and operations.
These are conservative cooperative checks, not a guarantee against OOM or
unrelated applications changing device use between samples. The allocator cap
does not bound CUDA context, driver, external process or all non-PyTorch use.

The 64 KiB trace claim covers six retained selected device vectors (18,432
logical bytes). The 1 MiB host trace claim covers retained raw selected-vector
and mask-vocabulary bytes and the encoded selected-vector evidence through
candidate writing. The four 30,522-float mask byte strings alone total
488,352 bytes; full masks are hashed, not base64-serialized. Temporary CPU
tensors/copies, Python object and metadata overhead, JSON buffers, staging,
model state, full logits, finite-check scratch, hidden-state copies and native
workspace are outside that logical claim: it is **not a total host RAM cap**.
The synchronized CUDA snapshots and the held whole-owner commitment include
the actual model/forward path, subject to allocator and telemetry limits.

Peak allocator counters are reset **once**, before model loading. Every later
peak is cumulative since that reset, not a per-stage peak. Synchronized
endpoints are retained before/after model transfer and before/after each
forward; the final cumulative peak covers this exact observed sequence and
allocator history. No per-forward latency is measured. The complete output is
kept live through a synchronization, then discarded before a second
synchronization. Device captures and input/model references are discarded;
model and input objects are checked with weak references. Disposal is followed
by synchronization, and allocator/cache and the version-bound cuBLAS workspace
are cleared. Success requires zero reported
PyTorch allocated/reserved bytes at the final snapshot.

The candidate is provisional until it is serialized, its hash computed while
the trace claim is held, the in-memory result discarded, both claims and the
memory commitment released, a separate final cleanup record written, the
owner attests clean, and the supervisor reaps it. A failure, cancellation,
uncertain synchronization, cleanup or ownership state remains quarantined;
the device is not declared clean from a missing result. Partial stage records
are retained on failure when available, but incomplete observations are not
reconstructed. The Python process roots its first failure and ownership
objects until exit. The supervisor's unsafe marker protects against treating
an unverified child exit as a successful cleanup.

## Source and execution

The controller requires a clean committed checkout and records the Git commit
and hashes of the bootstrap, protocol, and all imported local project modules.
Both controller and child use the source bootstrap with `-I -S -B` and a fresh
empty `-X pycache_prefix`: it loads the checkout package explicitly, allows
only the listed project modules, and puts the supplied dependency directory on
the import path without adding the checkout root. This guards against stale
bytecode and a same-name installed project package; it does not authenticate
native libraries, the full dependency environment, transient same-account
edits, filesystem/hardware behavior or every possible network/CUDA API.

From this worktree, with the isolated device window coordinated, the command
below uses the existing workspace virtual environment and an existing offline
cache. The output path must be new. The bootstrap prefix must be a new empty
plain directory and must stay empty throughout the controller run.

```powershell
$workspace = (Resolve-Path '..\..\..').Path
$site = (Resolve-Path '..\..\.venv\Lib\site-packages').Path
$prefix = Join-Path "$workspace\work" ('mlg-bert-calibration-' + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $prefix | Out-Null
& '..\..\.venv\Scripts\python.exe' -I -S -B -X "pycache_prefix=$prefix" `
  scripts/bert_gpu_bootstrap.py --bootstrap-site $site --workspace-root $workspace `
  --cache-dir "$workspace\work\hf-cache\hub" --output "$workspace\work\bert-gpu-calibration-v1.json"
```

The `work/` result is a reviewed development record only after the owner and
supervisor both complete and the output reports success. The saved before/after
device-wide memory observations are point samples; Windows WDDM may not show
all clients. No other team should use these observations as an admission
profile for a different workload without a separately reviewed calibration.
The compact public outcome reports the controller's verification of the
private candidate, final cleanup record and reaped-child identity; it does not
independently expose the private PID/hash linkage for external reconstruction.
