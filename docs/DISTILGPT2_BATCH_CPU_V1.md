# DistilGPT-2 sampled CPU batch check, v1

This protocol is fixed before looking at the model's outputs. It checks the
CPU-only serial-versus-mixed agreement of the pinned pretrained checkpoint on
four public synthetic token sequences. Its scope is correctness on these
specific cases, not a throughput benchmark or proof of general batch
separability, privacy, or GPU behavior. It does not use causal-study prompts or
any reserved causal inputs.

## Inputs and binding

Use `distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, loaded offline by
`looking_glass.pretrained.load_distilgpt2`. The loader verifies all pinned
assets; the safetensors SHA-256 is
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The model is evaluated in float32 on CPU, with eager attention, no cache,
autocast, hooks, intervention, training, or model RNG. Run an isolated Python
process with CUDA hidden, one intraop CPU thread, deterministic algorithms,
and seed 2026102808. Record the complete measured runtime and source/checkpoint
identity in the frozen binding and report; the profile's separability field is
a trusted declaration under examination, not evidence for itself.
The bound model fingerprint is the SHA-256 of a canonical descriptor of the
pinned model ID, revision and asset checksums; it is not a digest of loaded
weights. The separately recorded before/after state digest characterizes the
loaded tensors and tests for change during the run, without proving that they
match the checkpoint.

Every input contains `input_ids` and an all-one `attention_mask`, both int64
with per-request shape `(1, 4)`. Two public discovery rows are used only to
check exact row-content disjointness and are not forwarded:

* `[50256, 101, 201, 301]`
* `[50256, 102, 202, 302]`

The mixed cases are fixed in this order, with one request per owner in each
group:

| Group | Owner | `input_ids` |
| --- | --- | --- |
| 1 | team-a | `[50256, 103, 203, 303]` |
| 1 | team-b | `[50256, 104, 204, 304]` |
| 2 | team-a | `[50256, 105, 205, 305]` |
| 2 | team-b | `[50256, 106, 206, 306]` |

These are synthetic token IDs, not natural-language prompts. ID 50256 is
`<|endoftext|>` in the pinned tokenizer. The public v1 causal grammar contains
ordinary prose and no `<`, `>` or `|` characters; its byte-level tokenizer
cannot emit this special token from those strings when called with
`add_special_tokens=False`. This is a grammar-based
disjointness argument, not an inspection or exhaustive comparison of reserved
inputs. A separate custodian must make an aggregate overlap check if a later
causal study uses other input-generation rules.

The approved combined batch contains exactly two rows. Each serial and mixed
output must be finite CPU float32 logits with shape `(rows, 4, 50257)`. Compare
each mixed row against its serial reference elementwise using the frozen rule
`abs(mixed - serial) <= 1e-6 + 1e-5 * abs(serial)`. This is a prespecified
correctness tolerance, not one chosen from results. For each group run the two
serial requests followed by the mixed request through the resident worker;
the planned model-entry shapes are `(1, 4), (1, 4), (2, 4)` per group.

## Controls, accounting, and evidence

Freeze the validator protocol, exact runtime and source identities, input
hashes, and tolerances before any pretrained forward. Publish the source and
frozen manifest first. The trusted CPU executor must authenticate request IDs;
verify model evaluation mode, no cache, and the resident worker's bounds; and
record actual model-forward input shapes using a temporary calling-thread
Python profiler that is restored in `finally`. This observer adds overhead, so
do not report its execution as a speed or latency measurement. A missing,
extra, or wrong-shaped model entry makes the run incomplete. Bind each observed
entry to the worker request ID and record its exact small public token and mask
rows, verifying they match the frozen ordered cases. Record a streaming digest
of named model state before and after execution; reject a changed digest. This is additional
integrity evidence, not proof that the loaded model is the declared checkpoint.

Save all four serial logits and both mixed logits, raw little-endian float32
in a non-pickle ZIP archive, with their shapes, call identities, and SHA-256
digests. The declared raw-output cap is 8 MiB, and exactly 6,432,896 bytes are
planned; this cap does not cover model memory, temporary copies, or the ZIP's
peak storage. Save the frozen protocol, comparison report, source/checkpoint and
runtime provenance, observed calls, raw archive digest, and negative-control
results. External checksums and source history make errors detectable; they do
not authenticate execution. A failure or interruption is retained as
incomplete rather than replaced with a passing run.

The source-only bootstrap `tools/run_distilgpt2_batch_cpu_v1.py` requires
`-I -S -B` and an absent `-X pycache_prefix` path. It places the standard
library before the explicitly selected virtual environment's dependency
directory, rejects sourceless dependency loaders, and restricts project
imports to listed `.py` files. It checks `CUDA_VISIBLE_DEVICES=-1` before
importing dependencies, records that `torch.cuda.is_initialized()` is false
before the run, and rechecks it afterward. This is PyTorch's process state,
not an assertion about all native CUDA activity. Both freeze and execution
verify that the recorded source commit is an ancestor of the clean checkout
and that the listed `SOURCE_PATHS` match that commit's ordinary file blobs.
Publish the source commit, then freeze and publish the manifest as a subsequent
commit before any model forward; retain its `manifest_sha256` separately. The
runner requires that exact tracked manifest file to match a declared later
ancestor commit, but cannot independently verify public publication. Put the run directory
outside the clean checkout and copy its evidence into the repository only
afterward. Runtime information is checked again after the model calls.
The interpreter, native libraries, dependency installation, filesystem and
process isolation are trusted; these measures are not an execution
attestation. In particular, project source paths and any filesystem link
targets remain trusted. Python hash randomization is not fixed or claimed to
be fixed. Runtime records preserve only SHA-256 fingerprints of the ordered
interpreter paths, cache prefix and Torch build configuration; those hashes
can be guessed and do not authenticate execution.

Use a separately identified deterministic batch-mean toy model as a negative
control: its serial outputs are zero while mixed outputs depend on their
co-batch, so the validator must reject them. Independently use the archived
pretrained outputs in positive playback to reproduce the primary comparison
metrics, then in a deliberately row-reversed mixed-output playback control with
a different adapter identity; it must expose cross-owner misdelivery. Playback
is a control over stored outputs, not another pretrained execution. No evidence
from either control certifies the real model.

Pass means that all four sampled cases met the frozen tolerance in the stated
CPU runtime and the declared two-row shape. Exact row-hash exclusion addresses
only the supplied discovery inputs, not semantic independence, all possible
cases, or a future GPU/broker configuration. Save discrepancies, invalid
outputs, or incomplete outcomes without adjusting the sample or tolerance.
