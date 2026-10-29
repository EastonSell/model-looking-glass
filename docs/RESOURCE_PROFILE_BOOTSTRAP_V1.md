# GPU resource-profile bootstrap, protocol v1 (proposed)

Status: **draft; do not run yet**. No GPU resource profile or throughput result
has been produced by this protocol. The manifest builder, owner integration,
independent protocol review and exclusive device window are prerequisites.
This is a bounded pilot to obtain observations for exact workload profiles;
its initial reservations are operator-selected estimates, not measured bounds
or guarantees against out-of-memory (OOM) failures.

## Fixed workload

- The offline loader `looking_glass.pretrained.load_distilgpt2` must verify all
  seven pinned assets with `allow_download=False`. The model is
  `distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`;
  `model.safetensors` is 352,824,413 bytes with SHA-256
  `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`;
  `config.json` SHA-256 is
  `4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a`.
  Preserve the complete verified asset manifest. Disable remote code and network
  fetching; load on CPU before copying the model to the GPU. Before the copy,
  require both PyTorch overwrite- and swap-on-conversion future flags to be
  false and verify that the language-model head and input embedding are the
  same `Parameter` object. Recheck that identity and their device storage after
  transfer; sharing CPU storage between distinct `Parameter` wrappers is not
  sufficient to establish a tied GPU model.
- Exactly one CUDA-visible physical GPU, canonical UUID mapping at `cuda:0`;
  batch 1, sequence length 2, `input_ids` only, dtype `torch.int64`, token IDs
  `[15496,995]`; FP32 model/compute, eager attention,
  eval and `torch.inference_mode()`, no cache, autocast, compile, intervention,
  generation, gradients or capture. Execute the actual
  `looking_glass.resident_worker.ResidentWorker.execute` path with a
  `ForwardRequest`; this includes request snapshot, forked RNG state and
  CPU/CUDA seeding, device input copies, model state checks, output offload and
  the worker's synchronization. A raw model forward has a different resource
  footprint and cannot authorize service admission. The complete output must fit a 512 KiB tensor cap and a 4 MiB encoded-response cap;
  verify the config's vocabulary size is 50,257 before forward, giving expected
  float32 logits of shape `(1, 2, 50257)` and exactly 402,056 bytes. The worker
  checks rank, input dimensions and tensor byte cap before CPU offload. After
  offload, validate the exact CPU shape, dtype and byte count; check the logits are
  finite. The current IPC preflight budgets 32 bytes per value: 3,216,448 bytes
  is below its 4 MiB response cap. Enforce the actual serialized response cap
  as well. Dispose of request-owned GPU input/output references before
  returning. This post-forward check cannot prevent an earlier allocation.
  Do not silently change shape, tokens, dtype, operation, output cap or settings
  to make a run fit.
- The canonical case manifest is
  `{"schema":"resource-bootstrap-case-v1","input_ids":[15496,995],"shape":[1,2],"dtype":"torch.int64"}`;
  its canonical JSON SHA-256 is
  `a137759c4f2a40c4da15d1a4625086e160edc75f161748db91b07d83dae751e4`.
  The trusted owner calculates this from the actual CPU input and shape, rather
  than copying the constant. Set and record Python, NumPy and PyTorch seeds.
  One cold forward uses seed 17. Only after it completes successfully, collect
  three warm forwards for this **same signature**, using seeds 29, 43 and 71.
  Repeated measurements of one input are not independent task cases.
- The canonical warmup manifest is
  `{"schema":"resource-bootstrap-warmup-v1","procedure":"one_completed_cold_forward_same_case","cold_seed":17,"warm_seeds":[29,43,71]}`;
  its SHA-256 is
  `84b21a9ba63413bb9394dbac6d31dbc08ce3171508a25477f44e596be579760c`.
  The capture manifest is
  `{"schema":"resource-bootstrap-capture-v1","mode":"none","max_gpu_trace_bytes":0,"max_host_trace_bytes":0}`;
  its SHA-256 is
  `1ec2c8752f73cab7403fa9012e0796de77fdd87eb395a7cbc796044bf1643250`.
  These are hashes over `canonical_fingerprint` JSON, not raw prose.
- The load sample uses seed 17 and the canonical manifest
  `{"schema":"resource-bootstrap-load-case-v1","operation":"model_load","checkpoint_sha256":"e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b","config_sha256":"4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a"}`,
  SHA-256 `1c46b2cda5fe1f93648d80135b202816303009835bb5bf050b1e6e676110021c`.
  Verify these digests from the actual staged assets. Loading is one completed
  sample, not a claim of a distribution of load peaks.

## Reservation and hard limits

Use binary GiB and MiB throughout. The machine inspection on 2026-10-28 found
an NVIDIA GeForce RTX 2070 SUPER with 8,192 MiB total; this is identification,
not an admission snapshot. Sample live device-wide free and total immediately
before each admission. Reject if total is outside 7–9 GiB, telemetry is
unavailable/inconsistent, the actual physical identity cannot be verified, or
headroom is insufficient. Require reserve
`R = max(1 GiB, ceil(0.15 * device_total_bytes))`, as in `MemoryBudget`.

Before any model copy to CUDA, commit **2 GiB** as a separately held model-load
lease; retain it throughout all forwards and shutdown unless a separately
reviewed handoff exists. Before each forward, commit another **512 MiB** job
lease, held until output offload, synchronization and disposal are complete.
Both use zero GPU/host trace caps. `MemoryBudget` must enforce its current
process-reserved and device-free checks including all outstanding commitments
and `R`, and recheck after initial synchronization before the operation. The
held load estimate deliberately double-counts memory already realized by the
resident model; do not demote or release it based solely on a snapshot.

The 2 GiB load commitment is more than five times the 352,824,413-byte on-disk
weight file, with 512 MiB separately committed for the short forward. That
arithmetic motivates a cautious starting point; file size does **not** bound
runtime memory, model copies, CUDA context, operator workspace or future model
behavior. There is no automatic retry with a larger limit. If admission or
execution fails, save a failed attempt and stop.

After the supervised session/lease and runtime binding are verified, but before
any GPU model allocation, require the native PyTorch CUDA caching allocator and
call `torch.cuda.set_per_process_memory_fraction(0.20, device=0)`, read it back
using `torch.cuda.get_per_process_memory_fraction(0)`, and verify the unchanged
CUDA total and the effective fraction computed as `int(0.20 * total) / total`.
PyTorch 2.8 truncates to an integer byte limit, so comparing the readback with
exact `0.20` would spuriously fail. Record requested fraction, effective integer
limit, readback and total. At 8 GiB, the nominal `0.20 * total` allocator ceiling
is 1.6 GiB, below the 2 GiB initial load commitment. Treat the total-range
check above as mandatory. The cap constrains PyTorch's caching allocator, not
all CUDA allocations, driver/context memory, other processes or the physical
GPU as a whole; free memory can fall between checks. See the official
[PyTorch allocator cap](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
and [CUDA memory semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html).

## Identity and independently reconstructed manifests

All manifests below are canonical JSON data accepted by `canonical_fingerprint`
(sorted object keys, no floats); literal fractions are decimal strings. The
supervisor constructs them from actual loaded assets and live settings, never
by copying fields from the profile being selected. A service may use a profile
only when its independently reconstructed complete `WorkloadKey` matches the
pinned record. Bootstrap v1 additionally requires
`key.source_sha256 == provenance.source_manifest_sha256 == SHA256(canonical_source_manifest_bytes)`;
the generic profile schema permits these hashes to be distinct. A changed or
unavailable field is a hard failure. Profile hashes
check consistency of supplied bytes; they do not authenticate a supervisor or
prove its observations.

The **proposed**, not yet implemented, source schema is:

```
{
  "schema": "resource-bootstrap-source-v1",
  "git_commit": "<40 lowercase hex>",
  "git_tree": "<40 lowercase hex>",
  "repository_files": [{"path": "<repo-relative POSIX path>", "size": 0, "sha256": "<64 lowercase hex>"}],
  "python_executable": {"size": 0, "sha256": "<64 lowercase hex>"},
  "distributions": [{"name": "<normalized name>", "version": "<exact version>",
                     "files": [{"path": "<environment-relative POSIX path>", "size": 0, "sha256": "<64 lowercase hex>"}]}]
}
```

The example zero sizes denote field types, not actual expected sizes. Require a
clean source checkout at one exact detached commit; the runner and service must use
that same frozen source snapshot, so other teams may continue merging on main.
Keep that checkout protected against ordinary writes for its lifetime; hashing
at startup cannot prove an external process never changed or replaced a file.
Any changed source/runtime requires a new profile. Hash the real bytes of all tracked files under
`looking_glass/`, including the actual executed
`looking_glass/gpu_service.py` entrypoint, this protocol document at
`docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md`, and all installed files recorded for
all installed venv distributions, including Torch, Transformers, safetensors,
tokenizers, Hugging Face Hub and NumPy, sorted by normalized distribution name
and normalized path. Include transitive packages so the owner can construct a
pre-import allowlist from their hashed executable files. Reject duplicate, missing, non-regular, or
outside-environment files. Compute hashes from file contents, not just package
metadata. Hash the actual base Python executable. Verify that the loaded module
origins for the owner, loader and these distributions resolve to the hashed
files; install a pre-import guard that rejects `SourcelessFileLoader` and
refuses any repo/environment executable origin absent from that set before it
executes. Preserve built-in and frozen importer precedence, wrapping only the
path finder so a repository file cannot shadow a genuine builtin. Append the
verified checkout and environment after trusted standard library paths; reject
unknown loaders or origins rather than only checking after execution. Hash this complete
manifest as `source_sha256` and revalidate before profile-backed service startup. This
coverage is deliberately explicit: Python's standard library and OS/driver
components are not all file-hashed, so record
those limitations and do not claim exact binary reproducibility. Extend and
version the schema before relying on untracked code. Preserve the manifest
privately because paths or hardware identifiers can reveal machine details.

The **proposed**, not yet implemented, runtime settings object has exactly
these keys: `schema` (`resource-bootstrap-runtime-v1`), `device`, `device_uuid`,
`device_name`, `device_total_bytes`, `compute_capability` (two integers),
`python_version`, `torch_version`, `transformers_version`,
`cuda_runtime_version`, `driver_version`, `cudnn_version`,
`cuda_visible_devices`, `cuda_device_order`, `allocator_backend`,
`allocator_fraction` (string `"0.20"`), `allocator_effective_limit_bytes`
(integer), `model_dtype`, `compute_dtype`,
`attention_backend`, `model_mode`, `grad_mode`, `autocast`, `use_cache`,
`compiled`, `float32_matmul_precision`, `cuda_matmul_allow_tf32`,
`cudnn_allow_tf32`, `cudnn_enabled`, `cudnn_benchmark`,
`cudnn_deterministic`, `deterministic_algorithms`, `execution_path`, `torch_num_threads`,
`torch_num_interop_threads`, `input_limit_bytes`, `output_tensor_limit_bytes`,
`response_wire_limit_bytes`, `python_import_paths` (the actual ordered,
validated post-bootstrap path strings), and `environment` (the explicitly selected
nonsecret CUDA/PyTorch/backend variables, including absent values as null).
The owner sets and reads back the device, allocator and backend flags and
fingerprints the resulting object as `runtime_settings_sha256`. The model,
forward mode, capture and execution-path fields are the planned contract at
key construction, not observations of a model or execution thread; verify the
loaded model and actual worker's per-call inference/autocast/cache behavior
before treating a sample or profile as eligible. `execution_path` is the exact string
`looking_glass.resident_worker.ResidentWorker.execute`; use `eager`, `eval`,
`inference`, disabled autocast,
cache and compile, `torch.float32`, native allocator, highest float32 matmul
precision, both TF32 flags false, cuDNN enabled, cuDNN benchmark false, cuDNN
deterministic true, deterministic algorithms true, and one intra-op and one
interop CPU thread. Read back and record the actual cuDNN version, failing if it
is unavailable. Worker input, output tensor and response wire caps are 64 KiB,
512 KiB and 4 MiB respectively. The fixed input tensor itself is 16 bytes. Validate the actual serialized response,
including framing, is within the wire cap; the value-only estimate is not a
complete frame bound. Record unsupported or unverified native dependencies; do
not claim the manifest covers every component that can affect memory.

The approved owner must set these exact environment controls before importing
or initializing CUDA: `CUDA_VISIBLE_DEVICES` to the full physical UUID,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTORCH_CUDA_ALLOC_CONF=backend:native`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `PYTHONHASHSEED=17`,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`. Launch the direct base interpreter with `-S -P` so startup
does not import `site`/`sitecustomize` or prepend an untrusted working directory.
The parent **removes `PYTHONPATH` before interpreter startup**, then the
trusted bootstrap imports its standard-library dependencies and inserts only the
preverified frozen checkout and selected venv site-packages programmatically.
The child verifies actual module origins and `sys.path` against the preflight
inventory before model work. Under `-S`, Python 3.11 does
not infer the venv prefix, so pass the selected environment path explicitly.
Before starting the direct interpreter, the trusted parent creates a new private
empty directory and sets `PYTHONPYCACHEPREFIX` to its
absolute path; the child verifies `sys.pycache_prefix` matches it,
`sys.dont_write_bytecode` is true, and the directory remains empty. Merely
setting `PYTHONDONTWRITEBYTECODE` does not prevent reading old bytecode. The
parent must set the hash seed and cache variables before interpreter startup;
a later environment read alone cannot prove the active hash seed or that a
cache was initially empty. Module origins and allowed import paths must match
the hashed repo and selected environment; reject user-site or unknown shadows.
The random unique cache path is recorded privately as a policy attestation, not
made part of the reusable runtime fingerprint. Require
`PYTORCH_ALLOC_CONF`, `PYTORCH_NO_CUDA_MEMORY_CACHING`, `PYTHONPATH`, and
`CUDA_LAUNCH_BLOCKING`, `PYTHONHOME`, and inherited `LD_*`/`DYLD_*` loader
controls to be absent. The trusted parent supplies only explicitly allowed,
existing, canonical absolute directories in Windows `PATH`, without empty,
relative, current-directory, or duplicate entries. The initial Windows pilot
uses only the resolved `SystemRoot/System32` directory after its CPU preflight;
the exact value is recorded privately and must match at service startup. This
does not authenticate OS or driver binaries. Reject any other inherited `CUDA_`,
`PYTORCH_`, `TORCH_`, `CUBLAS_`, `HF_`, `TRANSFORMERS_`, `OMP_`, `MKL_`,
`OPENBLAS_`, or `PYTHON` prefixed setting
unless it is added to a newly reviewed protocol; remove secrets from the child
environment rather than recording them. Record the selected values and absent
ones in `environment`. Verify the actual mode/flags before each profile-backed
service start; do not derive them from a profile's asserted values.

Construct distinct exact `WorkloadKey`s for cold model load, cold forward and
warm forward. The load key has no request shape or output/capture; the forward
keys have the exact signature and limits above. All bind the checkpoint, model
config, source/runtime manifest hashes, physical device, operation, warmup and
capture hashes. The resource-profile code verifies equality and evidence
integrity; it does not itself implement these manifest collectors or verify
that a manifest describes the actual running code. Until that implementation
is independently reviewed, profile-backed service startup must fail closed.

## Supervised execution and observation

1. A coordinator reserves an isolated window, drains shared work and uses the
   shared task workspace root (the parent of `work/`, not a team checkout or
   `work/` itself) for the UUID-derived identity and `run_supervised`. All GPU
   owners must use this same root. The parent uses
   `run_supervised(..., inherit_environment=False, env=sanitized_mapping)` so
   no unlisted parent environment values leak into the child, and writes the
   persistent unsafe marker before starting a **direct** owner
   child. No other cooperative GPU workload overlaps. The lease does not stop
   unrelated applications; record any visible competing activity.
2. The child acquires the canonical `DeviceLease` and calls
   `OwnerSession.check` before **any** CUDA call, including a telemetry-backed
   `MemoryBudget.acquire`. Only then obtain the CUDA runtime device count and
   `torch.cuda.get_device_properties(0).uuid`. PyTorch 2.8 source specifies
   `torch._C._CUuuid`, whose string representation is lowercase full UUID hex
   without `GPU-`; require that exact type and `8-4-4-4-12` format, prepend
   `GPU-`, and call `OwnerSession.verify_runtime_binding`. This representation
   is source-verified but not yet observed on this machine's runtime instance.
   An unsupported type/format, mismatch or multiple visible devices fails closed.
   See the [PyTorch 2.8 UUID binding](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/cuda/Module.cpp#L951-L994).
   Verify the budget/probe telemetry targets the same physical device. A
   preflight UUID or copied profile field is not runtime verification.
3. Verify offline files and construct the CPU model; set/verify the allocator
   cap and exact runtime flags. Sample telemetry, acquire the 2 GiB load lease,
   and use `collect_sample` to synchronize, read baseline, recheck headroom,
   reset PyTorch peaks, copy the CPU model to `cuda:0`, synchronize and record
   allocator peaks and device-free/total endpoints. Keep the model in a
   controlled holder; its owned GPU references remain covered by the load
   lease. The callback returns `None`, while preserving the intended resident
   model. Do not interpret end free memory as a device-wide peak.
4. Construct the actual `ResidentWorker` around the loaded model under the
   held load lease. Acquire a 512 MiB job lease and collect the worker's cold
   `execute(ForwardRequest(...))` with seed 17. Within the collector callback,
   validate the returned CPU result's exact shape, dtype and finite values,
   encode and bound the full response frame, and dispose of request references
   before returning `None`; then release the job
   lease. If successful, separately collect the three warm forwards with the
   same verified signature and seeds 29, 43 and 71, acquiring and releasing a
   job lease for each and performing the same validation inside its callback.
   A failure yields no completed sample. No traces are captured. Save all before/after snapshots,
   allocated/reserved peaks and identities as individual samples.
5. Build separate profiles only from successfully completed samples. For this
   pilot, predeclare `margin_bytes=512 MiB, margin_bps=2500` for load and
   `margin_bytes=256 MiB, margin_bps=2500` for forward. The formula in
   [RESOURCE_PROFILE.md](RESOURCE_PROFILE.md) adds declared input/output bounds
   to observed deltas and then applies the margin. Margins are heuristics and
   cannot turn a finite sample into a guaranteed upper bound. The owning team pins
   each complete profile ID separately; no request supplies estimates or pins.
   A service warmup must use the approved **cold** profile for the exact same
   signature before accepting requests under the corresponding warm profile.
   Cold failure means no warm admission. Different signatures need their own
   validated warmup and profiles.
6. Stop submissions; release and dispose all request, model, state-dict and
   loader aliases; synchronize; verify commitments/trace claims are settled.
   If safe teardown is verified, attest clean with the same device lease and
   retain the device lease until the supervised child exits. Otherwise preserve
   the unsafe marker and quarantine; process disappearance alone does not prove
   cleanup. A `worker.close()` call alone is not proof that other aliases died.

Use a new, private, non-overwriting run directory. Preserve protocol version,
start and terminal/failure records, source/runtime and asset manifests, actual
case hash/seeds, exact per-stage telemetry, budget/allocator settings,
synchronization outcomes, profile/evidence bytes, source revision, and any
exception or cancellation. Write artifacts atomically and retain partial or
failed records. An OOM, cancellation, non-finite output, output-limit failure,
or synchronization failure is a failure, never a successful sample. On OOM,
poison/stop that owner; only a verified synchronized teardown can attest clean.
On failed or indeterminate drain leave commitments and the marker quarantined.
Do not store raw physical UUIDs in public repository results; private profile
evidence contains the full identity needed for exact matching. A public summary
may report hardware model, aggregate observations and limitations without the
unique identifier.

## Private artifact layout and verification

The parent creates one new run directory and writes canonical UTF-8 JSON bytes
without a trailing newline for each ordinary JSON file. Private files are
`started.json`, `owner.json`, `source.json`, `runtime.json`, `assets.json`, `load-case.json`,
`forward-case.json`, `warmup.json`, `capture.json`, `checks.json`,
`admissions.jsonl`, `observations.jsonl`, `load.profile.json`, `load.evidence.json`, `cold.profile.json`,
`cold.evidence.json`, `warm.profile.json`, `warm.evidence.json` and
`terminal.json`. A failure may leave only a subset. The asset manifest binds the
model ID, revision and sorted list of verified `{name, size, sha256}` records.
The canonical source, runtime, case, warmup and capture manifests follow the
schemas above; the source digest must agree with each profile's provenance and
key. No raw UUID or local path goes into a public summary.

`assets.json` has exactly `schema` (`resource-bootstrap-assets-v1`), `model_id`,
`revision` and `assets`. The latter is a name-sorted array of exactly the
loader's seven `{name, size, sha256}` objects, with integer byte sizes.
`started.json` has exactly `schema` (`resource-bootstrap-start-v1`), `run_id`
(32 lowercase hex), `started_utc` (UTC ISO-8601 string), `protocol_sha256`,
`git_commit`, `git_tree`, and `requested` (an object with exactly
`load_commitment_bytes`, `job_commitment_bytes`, `reserve_min_bytes`,
`reserve_percent`, `allocator_fraction` string, `input_limit_bytes`,
`output_tensor_limit_bytes`, `response_wire_limit_bytes`, `max_gpu_trace_bytes`,
`max_host_trace_bytes`). All values are integers except `allocator_fraction`;
`reserve_percent` is integer 15. Values must equal the protocol limits above,
and `output_tensor_limit_bytes` must equal each forward key's
`output_limit_bytes`. Its protocol hash must match the actual tracked protocol
file in `source.json`.
Write `started.json` before device work. This `run_id` is the artifact run ID;
it is separate from the supervisor's generated run ID.

The supervised child writes `owner.json` at startup before CUDA access, using
exactly `schema` (`resource-bootstrap-owner-v1`), `parent_run_id` (the artifact
run ID), `supervisor_run_id` (the 32-hex `OwnerSession.run_id`) and `pid` (the
positive integer `os.getpid()`). The parent never invents child identity on a
pre-launch failure. A clean terminal requires `owner.json` to match the actual
reaped `SupervisionResult` run ID and child PID, including for clean workload
failures. Unclean/incomplete failures may lack `owner.json`.

`checks.json` has exactly `schema` (`resource-bootstrap-checks-v1`),
`device_total_bytes` (integer), `allocator_requested_fraction` (`"0.20"`),
`allocator_effective_limit_bytes` (integer), `allocator_readback` (round-trip
Python decimal string), `model_aliases_disposed` (true),
`device_synchronized_after_disposal` (true), `owner` (object exactly
`{run_id,pid}` matching `owner.json`'s supervisor run ID and child PID),
`load_lease_disposition`
(`"released_after_verified_disposal"` or `"held_until_exit"`), `import_policy`
(object specified below), and `forwards`
(a sequence-sorted array). Record the
actual total and allocator readback after setting the cap; the effective limit
must equal `int(0.20 * device_total_bytes)` and the parsed readback must equal
that integer divided by the unchanged total. Each forward object has exactly
`sequence` (2, 3, 4 or 5), `output_shape` (`[1,2,50257]`), `output_dtype`
(`"torch.float32"`), `output_tensor_bytes` (402056), `finite` (true),
`framed_response_bytes` (integer between 201,031 bytes and 4 MiB inclusive),
`offload_complete` (true), and `disposal_complete` (true). Encode the actual
CPU result using the same production response serializer and count the complete
framed message; do not assume the 32-bytes-per-value preflight includes framing.
The lower bound is conservative: 100,514 JSON numeric values require at least
one byte each, 100,513 separators and the four-byte frame header, even before
JSON object and array syntax. A zero-length successful frame is invalid.
The callback performs this serializer check before returning, so a failed
check never produces a completed sample. The `import_policy` object has exactly `private_cache_prefix` (absolute private
path string), `parent_created_empty` (true), `runtime_prefix_matches` (true),
`cache_empty_after_imports` (true), `dont_write_bytecode` (true),
`safe_path_enabled` (true from `sys.flags.safe_path`), `site_import_disabled`
(true from `sys.flags.no_site`), `user_site_enabled` (false: no user-site path is active; `site.ENABLE_USER_SITE`
may be null under `-S`), `hash_seed_set_before_exec` (true),
`module_origins_verified` (true), and `sys_path_verified` (true). The parent
records its pre-exec creation/seed attestations; the child checks its actual
runtime prefix, remaining cache, module origins and path policy. The private
prefix is not included in the reusable runtime fingerprint. These fields are
recorded attestations, not independent proof that output disposal or a
particular code path occurred.

`admissions.jsonl` contains one canonical, LF-terminated record per completed
stage, sorted by the same sequence as the observation journal. Each has exactly
`schema` (`resource-bootstrap-admission-v1`), `sequence`, `stage`, `telemetry`
(the four `CudaMemorySnapshot` fields as integer bytes), `reserve_bytes`,
`committed_incremental_bytes`, `load_commitment_bytes`, `job_commitment_bytes`,
`owner_headroom_check` (true), `collector_post_sync_check_attested` (true), and
`job_lease_released_after` (null for load, true for each forward). The load
commitment is 2 GiB throughout; the job commitment is zero for load and 512 MiB
for each forward; the total committed must equal their sum. The verifier
recomputes `reserve_bytes = max(1 GiB, ceil(15 * total / 100))`, and checks both
`process_reserved + committed + reserve <= total` and
`committed + reserve <= device_free` against this adjacent snapshot. These are
owner-observed admission snapshots; the collector's own post-sync snapshot is
not returned by its current API, so its check is an attestation, not a saved
independent observation. A failed/incomplete run may retain a partial journal.
Append and durably flush one canonical JSON line with a single LF to
`observations.jsonl` per completed synchronized sample. Each line has exactly
`schema` (`resource-bootstrap-observation-v1`), `sequence`, `stage`, `key` and
`sample`; `key` and `sample` are the full canonical dataclass objects. In order,
the expected `(sequence, stage, seed)` values are `(1, load, 17)`,
`(2, cold, 17)`, `(3, warm, 29)`, `(4, warm, 43)`, `(5, warm, 71)`.
Each sample's case digest and workload fingerprint must match the corresponding
manifest and exact key. The three profile documents and evidence blobs use
`MeasuredResourceProfile.to_json()` and `encode_evidence`, respectively.
Candidate profile IDs may be recorded for review, but the owning team stores
the three full reviewed pins independently of this run directory.

The supervising parent writes `terminal.json` atomically only after the child
has exited and the supervisor reports its marker/teardown outcome. It uses
`resource-bootstrap-terminal-v1` and has exactly `schema`, `run_id`, `ended_utc`
(UTC ISO-8601 string), `status` (`success`, `failed`, `cancelled` or
`quarantined`), `child_exit_code` (integer or null), `clean_teardown` (boolean),
`failure_stage` (string or null), `reason` (string or null), `supervisor`
(null if no reaped supervisor result is available, otherwise exactly
`{run_id,pid,exit_code,clean,timed_out}` from the actual `SupervisionResult`),
and `inventory`
(a path-sorted array of `{path, size, sha256}` for every other run file). A
successful terminal has exit code 0, `clean_teardown=true`, and null failure
stage/reason; its supervisor must be non-null, not timed out, and match
`owner.json`, `checks.json.owner`, `child_exit_code` and `clean_teardown`.
Any non-null supervisor must match the terminal exit code and teardown flag;
if `clean_teardown=true`, it must also match `owner.json`. A null supervisor may
indicate pre-launch failure, or an interrupted/failed supervisor without a reaped
result; in either case `clean_teardown=false` and `child_exit_code=null`, and an
unverified post-launch state is quarantined. Do not fabricate an owner ID. Every non-success
terminal has nonempty string failure stage and
reason (`unknown` is permitted when the parent cannot determine the stage);
failed/cancelled runs may have either teardown value, whereas quarantined runs
have `clean_teardown=false`. Inventory paths are single relative filenames with
no traversal. A successful terminal requires a clean supervised
exit, all five exact ordered samples and admission records, complete valid `checks.json`, valid
source/identity bindings and all three profile/evidence pairs. A failed operation or unverified teardown can
never produce a success terminal; a started run with no terminal is incomplete.
Independent verification must recompute each evidence document and profile ID,
the estimate from raw samples, each inventory hash and all case/key/profile
bindings. Success must reject truncated or extra JSONL records in either journal, and the inventory
must exactly match the complete actual file set, not just the files it lists.
Failure inventories must retain partial and temporary artifacts rather than
silently omitting them. The terminal inventory detects changes only when the terminal and
approved pins are protected separately; it is not an authenticated seal. No
artifact writer or verifier implementing this layout is claimed to exist yet.

This pilot is not a latency or shared-throughput benchmark. Timing comparisons
and multiple-client throughput require their separate approved protocols and
resource coordination.
