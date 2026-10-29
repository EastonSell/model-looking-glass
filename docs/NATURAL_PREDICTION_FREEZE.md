# Natural discovery prediction freeze

`build_natural_prediction_freeze` is a pure postprocessor for the public natural
agreement track. It consumes immutable bytes from a completed discovery attempt
and its separate independent saved-evidence audit, then returns immutable,
canonical natural prediction and natural-only aggregate bytes. It does not load
or run a model, generate a nonce, draw held-out cases, write files, contact a
publication service, or establish that a claim of execution, audit, publication,
or chronology is authentic. Its tests use fabricated records over the already
public preflight rows; they perform no model forwards.

The trusted custodian must supply the exact public binding/runtime-manifest
bytes, `owner/attempt/{started.json,events.jsonl,terminal.json}`,
`owner/supervisor.json`, `launcher-terminal.json`, and the canonical independent
audit acknowledgment, plus the exact frozen public preflight manifest bytes.
The binding bytes are the already-published discovery
binding containing the runtime, source, and checkpoint configuration; their
SHA-256 becomes the natural track's `runtime_sha256`. The controller's
`evidence_complete` status is a structural candidate, so the separate
independent numerical and landing audit is mandatory. The caller supplies the
expected full source commit, the independently established audit-source digest,
the independently established builder-source digest, and the separately
published natural-prediction clarification digest. The expected auditor source
and clarification digests must also match the pre-discovery binding.

Before invoking the function, the caller must establish that the audit was
actually produced by the reviewed auditor over the complete retained journal,
and that the execution and publication acknowledgments are authentic. It must
also use a clean checkout at the bound source commit, verify the executed
builder and its dependencies against the binding, protect those files through
execution, and use a source-first process with a fresh bytecode-cache prefix.
The function compares the supplied expected builder digest with the binding;
it cannot attest its own loaded code or prevent a dishonest caller from
supplying mutually consistent fabricated records. Externally establish process
and file quiescence before reading any artifacts, and retain the original
files and any failures. The auditor checks the fixed seed, thread count and
determinism flag, and the summary's unchanged-state assertion, but does not
attest the actual model-state digest or runtime/source binding. The custodian
must establish those provenance facts and authenticate the auditor's execution
and acknowledgment separately.

The function hashes the exact public preflight bytes and compares every ordered
bound case ID, label, input row, mask and subject coordinate with that manifest.
It also checks the fixed protocol and checkpoint identity; exact
binding/start correspondence; successful launcher and supervisor cleanup;
controller inventory and independent-audit digests for the same journal; the
608-forward event order, token-row uniqueness and quartet alignment; every expected witness
reference, shape and retained-file entry; the bound natural-discovery storage
reservation; and the independent audit's
predictions, family effects, sign counts, baselines and selected sites against
the extracted events. It then validates the output using the reviewed held-out
adapter, which independently checks the prediction file's internal arithmetic
and schema. These checks do not recompute the model or independently inspect
the tensor files: that is the auditor's separate responsibility. Input byte and
line limits do not bound Python parsing overhead or process RSS.
The bound budget must include the recorded 3,216,448-byte public preflight
archive and all four fixed phase reservations. The custodian must additionally
reconcile any other attempts, orphaned files and shared study allocations; the
builder cannot discover unreported storage from supplied records.

The v2 natural prediction file includes the SHA-256 of the exact complete
event log and the independent audit acknowledgment. The returned bundle
contains `prediction_bytes`, `aggregate_bytes`, their
SHA-256s, the binding/runtime SHA-256, source commit, builder and auditor source
digests, the clarification digest, and the started, events, terminal,
supervisor, launcher-terminal and independent-audit digests. Publish or otherwise retain the supporting digest
inventory with references to the exact audited artifacts, and preserve the
full journal, controls, tensor witnesses, audit and operational evidence. The
prediction file contains the selection and ordered present/past metric events;
it does not contain the raw per-forward controls or tensor witnesses.

For this natural-only aggregate, publish the exact runtime binding and natural
prediction file, then publish the exact aggregate and its digest. A trusted
custodian must observe and retain those acknowledgments before generating any
held-out split nonce or revealing cases. The digest binds bytes but does not
prove publication time, source execution or confidentiality. If the injection
track is also to use the same held-out cases, independently validate and
publish that track's completed prediction/runtime first and construct a new
joint aggregate with the protocol's reviewed `make_prediction_freeze` helper;
publish that joint aggregate before the draw. Never silently add a track after
the shared held-out inputs are revealed.
