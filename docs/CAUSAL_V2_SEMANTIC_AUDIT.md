# Synthetic-v2 saved-data semantic audit

`looking_glass.causal_v2_semantic_audit` checks a **completed, revealed**
synthetic-v2 attempt using the retained public and private registries. It does
not run a model, generate cases, draw a nonce, or inspect a live attempt. A
trusted custodian must establish that the original process has exited, the
entire run is quiescent and stable, and access to private artifacts is
authorized before using it. No part of this tool changes the study's frozen
protocol or permits an earlier reveal.

After that handoff, run from a trusted checkout with the study's dependencies:

```text
python -m looking_glass.causal_v2_semantic_audit ABSOLUTE_RUN_ROOT
```

The command emits one canonical JSON report on stdout on success and a generic
error on stderr with a nonzero exit status otherwise. Store any redirected
report outside the run root. The program does not durably publish its output;
publication, process-exit recording, and provenance of the auditor's own source
and dependencies remain the caller's responsibility. Set
`PYTHONDONTWRITEBYTECODE=1` if bytecode writes are undesired. Existing registry
lock files are opened for exclusive locking with `r+b`; no artifact bytes are
intentionally changed, but write access to those locks is required. The API
`audit_run(absolute_path)` returns the same report, while `audit_evidence(...)`
accepts already retained canonical bytes and ordered chunk iterables.

The reader independently bounds and hashes both registries, requires their
successful matching terminal records, their exact artifact inventories, all
four ordered source-inventory snapshots, and all 54 discovery plus 54 held-out
chunks. It rejects missing or unexpected entries, symlinks, reparse points,
multiply linked regular files, pending writes, changed files, incomplete chunks
and failed controls. It then rechecks the inventories after replay. These
filesystem checks are cooperative checks under external quiescence, not a
sandbox against a concurrent process or filesystem changes after the audit.

The replay binds the disclosed private records to their commitments and the
frozen manifest; checks each saved condition's coordinates, input and run IDs,
target landing, trace comparison, captures, forward/reverse patches, null,
self, upstream and repeated-clean controls; and reconstructs the discovery
selector, prediction functions and held-out cell, per-model, equal-model and
uniform-site summaries. It uses the existing discovery binder and structural
metadata validator as additional checks, while independently deriving effects,
controls and summaries from the retained float32 bit patterns. Integer
binary32 arithmetic preserves subnormals even when the host flushes denormals;
signed zero is normalized for the protocol's equality checks. Failed or
incomplete evidence is never reported as a successful semantic audit.

The report means **saved-byte consistency only**. It does not establish that
the declared execution, model state, source, runtime, timestamps, publication
order, fail-stop history, or exposure/secrecy claims are true. The checker
validates case identities, split uniqueness and commitment links, but does not
regenerate the sampled cases or independently verify the completeness of the
historical exclusion inventory. It does not search for or account for earlier
attempts elsewhere. The exposure label is a retained custodian attestation,
not an independent secrecy verdict. The source-inventory snapshots are also
retained claims, not proof that those bytes executed. External provenance,
case-generation/exclusion, resource, publication, and exposure reviews remain
necessary before interpreting any study result.

This API is not supplied with independently pinned manifest or freeze bytes:
a wholly self-consistent replacement corpus could pass. Before relying on a
report, the custodian must compare its manifest, freeze, prediction,
commitment, reveal and terminal-inventory hashes with separately retained
pre-run, publication, and completion records. Missing independent pins must be
reported as a limitation rather than treated as an authentication success.

The auditor is a post-hoc tool: its own source revision and dependencies must
be recorded separately, rather than represented as part of a study source
freeze that preceded its creation. Its tests construct fabricated records with
non-study model IDs and never forward a model or generate reserved inputs.
