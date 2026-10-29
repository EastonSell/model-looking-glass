# DistilGPT2 intervention correctness: CPU development smoke

This records one development-only check of the frozen
[v1 protocol](../../docs/DISTILGPT2_INTERVENTION_DEV_V1.md) at source commit
`3f8de5d7fb299e4878dbf75f86bb58f2249d6a37`. It used the verified offline
`distilbert/distilgpt2` checkpoint and tokenizer at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, with the literal artificial input
`[50256, 50256, 50256, 50256]`, an all-one attention mask, and CPU execution.
The input is public development data. Before a future study seals its actual
discovery and held-out splits, its custodian must check and exclude this row and
the synthetic test row `[1, 2, 3, 4]`, failing or versioning the protocol on a
collision.

The known injection at `transformer.h.2`, invocation 0, batch 0, token 2,
hidden coordinate 0 had a stored-value increase of `0.9999999403953552`.
Its maximum recorded change across all `1 × 4 × 50257` logits was
`0.0446319580078125`. On this execution, the manual-reference and plan logits
matched bit-for-bit for clean, injected, and patched outputs; the clean-source
patch, null and self-patch activation controls, earlier-token checks, hook
registry checks, and model-state fingerprint checks passed. A separate run with
an empty offline cache correctly failed before loading the model and is retained
alongside the successful attempt.

Independent review included a read-only audit of the saved source bindings,
metadata, hashes and result consistency. A reviewer also repeated the same
protocol in a fresh CPU process and separate checkout on the same machine and
software environment; the recorded output and activation digests, comparisons,
and plan-event metadata matched. Full activation and logit tensors were not
saved, so the original numerical maxima and finiteness checks cannot be
recomputed from its hashes alone. The manual and plan paths share one loaded
model; same-site clean repair is expected by construction. This one input and
known injection provide no held-out causal localization, behavioral explanation,
generalization, or GPU-performance claim.

The original local records remain private. These publication copies replace
local workspace and interpreter path prefixes, and [publication.json](publication.json)
links their raw and redacted SHA-256 digests. The raw successful result digest is
`8e5594edf340646b20498106c698bc5a25e80048fa6b564c1e1af0dab74f6d91`. Raw
digests are integrity fingerprints, not anonymization of guessable path values.
