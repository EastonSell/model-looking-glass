# Development-only analysis for the pretrained causal protocol

`looking_glass/pretrained_causal_experiment.py` contains a **development-only**
exercise of the public analysis and publication sequence in
`PRETRAINED_CAUSAL_PROTOCOL_V1.md`. It does not load a model, create a genuine
secret split or site, run a pretrained discovery trial, or run reserved held-out
inputs. A successful outcome from its synthetic callbacks means only that the
development exercise completed; it is not a recorded protocol result.

The public analysis checks the fixed manifest schema, exact trace coordinates
for six MLP outputs and the final block output, float32 CPU snapshots, source
shape, dtype and device, repeated clean runs, the upstream, null and self-patch
controls, late rescue, and a fresh clean observation after every intervention.
It computes first-difference and magnitude selectors, the fixed random and
earliest baselines, all four paired patch-effect predictions, reverse-patch
effects and full-logit residuals. Float32 saved values are evaluated using
float64 differences and summaries. A first observed difference at layer 5 is
reported, but does not vote for an eligible injection site. With no eligible
votes, selection abstains; revealed rank is missing, top-1 is false, and
reciprocal rank is zero. This prevents the arbitrary order of zero-score ties
from being reported as a discovery. A baseline can still happen to select the
injected site.

The development runner publishes the canonical public manifest before it
invokes a callback and writes separately hashed predictions after all eight
discovery cases. Its synthetic evaluation checks externally retained manifest
and prediction digests before invoking its callback, saves complete condition
and post-clean evidence, and only then permits a synthetic commitment reveal.
Before any evaluation callback, it creates a one-shot claim in the registry
under its process-safe lock. Simultaneous contenders do not run callbacks or
finalize the winning run. A claim left by a crashed evaluator remains visibly
incomplete and rejects automatic replay; preserve that attempt and explicitly
terminate it before starting a separate development run.
Discovery and evaluation each publish a completion marker under a cross-process
phase lock after their data artifact is written. Publishing this marker is the
phase's commit point; later phases require its exact predecessor hashes. If a
marker write or lock cleanup reports an error after publication, the writer
verifies the marker and all predecessor bytes and treats a match as committed,
returning a warning and best-effort writing a sanitized warning artifact. A
crash or another storage failure can prevent the warning artifact from being
saved. If the marker's state cannot be verified, the caller receives an
unresolved-publication error and the run is not marked failed or successful.
If cancellation arrives after a verified phase commit, a distinct
`StudyCommittedCancellation` reports the committed stage and artifact digest;
the marker remains valid, but the caller is not silently advanced to another
phase. Another process may proceed independently after the lock is released.
Cancellation before commitment records a cancelled attempt when possible; if
publication cannot be verified, `StudyCancellationUncertain` preserves both
the cancellation signal and the unresolved state.
`StudyCancelled` also reports the verified terminal state, or `unverified`, so
an interrupt during cleanup of an already-failed attempt is not misreported as
the reason that attempt failed.
The same rule applies if terminal reveal finalization reports an error: verify
the terminal outcome and all reveal/predecessor hashes before returning a
committed warning. As terminal artifacts cannot be changed afterward, its
sanitized best-effort warning is a `.phase/reveal_warning.json` sidecar outside
the terminal artifact digest; it is not an independently authenticated record.
A reveal artifact without a verifiable terminal outcome remains incomplete.
An invalid or interrupted callback produces a failed or cancelled registry
outcome and best-effort partial evidence, not a successful result. Error
messages from callbacks are intentionally omitted from public evidence.
Failures of storage or host memory can prevent even this best-effort write;
the incomplete directory must then be retained and inspected.

Every public observation consists of seven selected vectors and the full
last-token vocabulary logits. Artifact tensor payloads are deduplicated by
SHA-256 of their little-endian float32 bytes and base64 encoded; each event
references its byte hash. The cap charges every forward even when two tensors
deduplicate, including fixed private pre/post/realized injection vectors and
source/self-patch snapshots. Fixed site-independent counts prevent those
public byte fields from encoding the hidden location. The 1 MiB per-forward
and 256 MiB all-phase limits constrain recorded tensor payload, not peak CUDA
VRAM, temporary Python allocations, total process memory, or arbitrary work
performed by a callback. The registry additionally places individual and
aggregate limits on serialized artifacts.

The runner validates the callback's tensor shapes and limits before copying,
then analyzes and serializes the same private CPU snapshot. The backend must
keep its submitted data quiescent during this copy: cloning does not produce an
atomic snapshot if another thread writes concurrently. Later backend mutations
cannot alter the owned copy. This can temporarily retain both the submitted
tensors and up to one case's owned copies, alongside serialization buffers;
the payload cap is not a cap on those transient allocations or Python overhead.

`verify_injection_integrity` checks a private pre/post pair against the exact
specified float32 addition, returning the realized delta and norm. The caller
must retain these values privately and independently establish that only the
intended site, invocation, batch and token were modified. A zero downstream
model effect is valid; an unchanged injection vector is not.
The exact addition check compares float32 bits, so positive and negative zero
are distinguished. CPU and accelerator floating-point-policy differences can
therefore reject a trial; such a mismatch is not evidence of a model behavior.

These functions trust the caller's synthetic `Observation` records. They do
not prove how a tensor was obtained, that a purported integrity check occurred,
that hooks were removed, that source tensors came from the same input, or that
an exclusive GPU lease was held. Python callbacks can perform arbitrary work:
the development API cannot enforce custodian isolation or prevent a caller
from accessing same-account files. Never pass a private study reveal or a real
reserved held-out prompt to the development callback. Digests provide consistency
with independently retained values; they do not attest a timestamp, randomness,
or secrecy. All agents share the same OS account, so a future study can claim
procedural blinding only, with any disclosure reported.

A recorded study remains blocked until a reviewed ordered composite plan,
an exclusive shared-model lease, source/runtime/checkpoint/tokenizer binding,
private custodian with per-prompt exact injection evidence, and safe failure
and cancellation handling are connected and independently reviewed. The source
checkout must be clean and match the manifest, the site/split must be committed
before discovery, and all predictions must be frozen before the first held-out
forward. The original protocol controls these requirements. Discovery inputs
are incomplete clauses and the held-out grammar produces complete templated
sentences: any eventual effect comparison measures transfer to those fixed,
shifted cases, not IID population generalization. Intervention effects and
first observed changes are not evidence of a unique natural reasoning mechanism.
