# Private synthetic-v2 cell evidence adapter

This development-only adapter connects caller-supplied JSON records to the
bounded [`v2_evidence` writer](V2_EVIDENCE.md). It runs no model and generates
no cases or schedule. Its local checks do not establish correct model execution,
the phase gate, causal interpretation, blind custody, or the authenticity of
caller-supplied identities. The custodian/executor must verify those properties,
validate the controls, and keep the target-identifying private chunk outside the
investigator's view until the protocol permits reveal. Same-account processes
are not a security boundary.

Reserve `CELL_CHUNK_LIMIT` (2,166,784 bytes) for each declared cell/phase
chunk. All 108 such declarations total 234,012,672 artifact bytes, leaving
room under the registry's separate 512 MiB and 128-name limits for the other
declared evidence. The `v2_evidence` constructor must additionally reserve
manifest, outcome, staging, and externally retained files within the 1 GiB
host-disk allowance, and check free space. This is not a VRAM allowance. The
adapter buffers at most 32 complete 64 KiB records and one bounded partial
record, but Python objects, parsing, joining, writer verification and other
simultaneous host-memory copies need an independently measured bound.

Create `CellEvidence(writer, name=..., cell_id=..., phase=..., model_seed=...,
model_state_sha256=...)` for one frozen chunk, then call `append(bytes)` in
case-index order, with strengths 0.25 then 0.75 for each case. Each record is
compact sorted-key UTF-8 JSON, canonicalized using the registry's JSON encoder
(`ensure_ascii=False`, finite numbers, no newline). `publish_complete()`
requires all 32 records. This completeness check is relative to that expected
inventory; it does not prove that the promised forwards were run. A held-out
record carries a caller-supplied reference to its already committed prediction;
the adapter does not enforce that commitment's timing or correspondence.

Every full record has exactly these top-level keys:

| Field | Required value |
| --- | --- |
| `schema` | `causal-v2-condition-record-v1` |
| `cell_id`, `phase`, `model_seed`, `model_state_sha256` | The buffer's fixed opaque identity, `discovery` or `heldout`, seed and lowercase SHA-256. |
| `case_id`, `case_index`, `condition_id`, `input_sha256`, `strength` | Caller-bound identities. The two strengths must share the case and input digest; case IDs, input digests and condition IDs cannot be reused across cases. |
| `prediction_ref` | Null for discovery; a nonempty opaque reference for held-out records. |
| `traces` | `clean`, `repeat_clean`, and `perturbed`, each an ordered four-event array. Each event has `metadata` from `TraceEvent.metadata()` and a `value` tensor. |
| `logits` | `clean`, `repeat_clean`, `perturbed`, `upstream`, `post_hook_clean`, and arrays `forward`, `reverse`, `null`, `self` with three site outputs each. This includes all 13 plan outputs and the separate final clean output. |
| `plans` | Thirteen records in `forward`, `reverse`, `null`, `self` site order 1–3, then `upstream` site 0. Each has exactly `kind`, `site`, full `metadata`, `captures`, `source_id`, and `base_id`. |
| `private_integrity` | Exactly `target`, `delta`, `clean_target`, `perturbed_target`, `clean_capture_metadata`, and `perturbed_capture_metadata`. Both target tensors and both metadata objects are retained. |
| `comparison` | A JSON object retaining the executor's complete comparison. |

A tensor is exactly `{"dtype":"torch.float32","device":"cpu","shape":[1,N],
"bits":[...]}`, with `N` equal to 8 for trace/target/capture values and 2 for
logits. Each bit is an unsigned 32-bit integer encoding a finite float32; the
exact bits, including signed zero and subnormals, are saved. Plan captures are
`post_action` for each plan and additionally `self_source` for self-patches.
The null plan has a null source ID; self-patches use
`same_forward_self_source`; other plans and all bases have nonempty references.
Full plan and private capture metadata are retained unchanged, but the adapter
does not check their semantic contents or calculate the referenced digests.
The executor must bind the digest to its actual model state and input bytes,
and verify the source/base references and all plan/control metadata.

After a failed condition or cancellation, stop model work, perform required
cleanup, and call `publish_incomplete()` to retain the validated prefix. An
optional canonical partial record may be supplied with exactly `schema`
(`causal-v2-partial-record-v1`), `cell_id`, `phase`, `model_seed`, `stage`,
and `evidence` (an object of whatever partial facts the caller has retained).
Then publish the bounded private failure envelope and terminal outcome as
specified by `v2_evidence`; do not expose private diagnostics as public status.
If the partial record is invalid or too large before any writer call, a single
salvage call without that partial preserves the accepted prefix and adds a
`causal-v2-partial-unavailable-v1` marker. Record the complete original error
separately; this marker does not claim the missing content was preserved. If
that error cannot be stored, leave the run incomplete.

The buffer and writer belong to one process and thread and cannot be copied or
serialized. Once publication is attempted, any error or uncertain return is
fail-stop: inspect the persisted chunk and hashes, and do not replay the same
attempt. Asynchronous exceptions at Python return boundaries require caller
cooperation; a local buffer cannot attest that a caller received a result. No
helper can recover a current condition or partial record that remained only in
memory when the process died.
