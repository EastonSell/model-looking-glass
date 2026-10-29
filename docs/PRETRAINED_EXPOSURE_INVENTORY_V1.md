# Known DistilGPT2 development exposures: inventory v1

This inventory lists saved or operator-reported exposures and conservative
reservations; it is not a certified complete history of model access or a
held-out dataset. It does not generate, inspect,
tokenize or execute any real study split. A successful known-row comparison
does not authorize a recorded study or establish a blind evaluation. The
pretrained study remains blocked until a separately reviewed pre-seal
clarification binds the applicable inventory, source/runtime and other required
controls to the public manifest.

`looking_glass/pretrained_exposures.py` fixes the canonical UTF-8 JSON payload
under version `distilgpt2-development-exposures-v1`. Its SHA-256 is
`6f21bffeacbe2323c21e53378161443eae8ba28b7210f30e70fd353497e87c8f`.
Canonicalization sorts object keys, uses compact separators and ASCII escapes,
and preserves array order. The payload binds the target model and tokenizer
`distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, model-safetensors SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`,
tokenizer JSON SHA-256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`,
and a default reference source snapshot at commit
`740692931da8deb5dc5c39a9692a2183cea32898`. Each referenced record names its
own snapshot commit; the later saved batch-validation records use
`c67a7ed036a9744ac6e67c94c73127d50f8eae27`, and the capture-preflight source
uses `9d289b8715f6ffe2cde52ccd13924c79873e065c`. A path and digest do not
authenticate an unsaved operator report or prove that all other work was
recorded.

The 13 observed rows include ten supported by saved repository records:

- `[464, 6766, 318, 4171, 13]`: pinned-model trace smoke and same-host replays,
  using the public text `The sky is blue.`; earlier unsaved loader/adapter
  checks used the same row.
- `[464, 1402, 6512, 11406, 319]` and `[32, 6016, 1657, 4120, 287]`:
  successful CPU throughput-smoke jobs. Their token IDs were independently
  reconstructed from the saved public prompts with the verified offline
  pinned tokenizer, with no model forward during reconstruction.
- `[262]` and `[257]`: saved CPU service and CLI reference/client checks.
- `[15496, 995]`: saved CPU bootstrap preflight and independent replay;
  an unsaved verifier challenge also used this row.
- `[50256, 103, 203, 303]` through `[50256, 106, 206, 306]`, with each
  second value paired with values 100 and 200 larger: saved pinned-model CPU
  batch-validation runs and an independent fresh-process replay. These are
  fixed synthetic development inputs, not causal-study held-out prompts.

Three additional observed rows have operator-reported or local-only evidence:
`[464, 2068, 318, 257]` and `[262, 257]` were unsaved diagnostics using the
pinned weights before and after temporary model-state mutations;
`[15496, 11, 616, 1438, 318, 257, 1332, 13]` was reconstructed from the
source bound to a locally saved CPU capture preflight; its record does not
separately contain raw token IDs. The source-file hash was independently
checked. The file `work/benchmark-preflight-9d289b-20261028-01/cpu_preflight.json`
has SHA-256 `becf33ace3373c4f880c03e6c8b6f993b6ab7222c7c291c50dbaa631b3a12b27`;
two earlier CPU preflights were also reported.
The preflights stopped at the closed GPU gate and no GPU timing is claimed.
They are included conservatively; the first two are not saved experimental
outcomes. A previously uncertain early socket-guard input was recovered from
the reviewer's retained session transcript and
matched the existing `The sky is blue.` row. That audit covered one person's
calls and does not establish a complete account of all agents or processes.

The payload separately lists three **planned** target-model input rows, two
**reference-only** batch discovery rows that were checked for disjointness and
not forwarded, and eight **synthetic-only** software-test rows. These statuses
describe the inventory snapshot: they are not claims of target-model execution. All are
reserved conservatively for collision checking. If an additional input is
used, a planned input actually runs, or a missing historical input is
recovered, preserve this version and publish a new inventory version with its
own digest and provenance before any future split is sealed. In particular,
`[50256, 50256, 50256, 50256]` was proposed for a real-model smoke but was
not reported executed when recorded here; `[1, 2, 3, 4]` was used with a
random model, not the target pretrained weights.

`check_known_splits` accepts two private candidate lists of eight token rows,
requires the expected inventory digest and pinned model/tokenizer identity,
and rejects malformed rows, duplicate token sequences within or across the
splits, and any candidate equal to a **nonempty prefix** of an inventoried
observed, planned, reference-only or synthetic row. Prefix exclusion is
conservative because an autoregressive forward can compute a context at each input position. The
check returns only counts and the inventory digest, never candidate rows or
their positions; collision errors omit them as well. Input containers must be
quiescent during copying. Python exception tracebacks can retain private input
locals even when their messages are sanitized, so the custodian must not send
exception objects or tracebacks to an investigator or log their locals. The
caller must independently establish that the actual pinned tokenizer, with no
special tokens or truncation, produced the
supplied IDs.

The check concerns exact token sequences only. Longer inputs may share an
exposed prefix, prompts may be semantically related without sharing tokens,
and unknown historical forwards may be missing from this list. The function
does not inspect model behavior, prove tokenizer identity or preserve a private
seal. Its result explicitly says `known-rows-only` and `preseal_ready=False`;
`require_preseal_readiness` remains fail-closed because this check alone is not
a seal authorization. `prepare_preseal_disclosure` can instead associate the
checked inventory with a caller-supplied protocol SHA-256 after explicit
acknowledgement that prior exposure history is unverified. Its result is a
trusted-caller report, not proof that the protocol was reviewed, a fresh split
was sampled, a manifest was committed, or execution is authorized. A separate
reviewed study protocol may proceed with this bounded check and transparently
disclose remaining exposure uncertainty; exhaustive proof of every historical
model access is not an automatic prerequisite. Neither helper replaces a
trusted, procedural-blinding custodian or provides same-account isolation.

The existing v1 protocol and previous results remain unchanged. Before using a
later inventory for actual discovery and held-out inputs, the custodian must
verify the pinned assets, privately check both tokenized splits, preserve the
exact inventory version and bytes, and bind its digest in a new reviewed
pre-seal protocol/manifest. Any disclosure, incomplete exposure history or
post-commit change must be recorded rather than silently presented as a clean,
independent holdout.
