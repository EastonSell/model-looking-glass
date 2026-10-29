# DistilGPT-2 subject–verb agreement discovery preflight, v1

This is a fixed, exploratory CPU baseline on 16 public development prompts. It
measures whether the pinned pretrained model assigns a higher next-token logit
to the grammatical member of ` is` and ` are` in these prompts. It is not a
causal experiment, a held-out evaluation, a GPU measurement, or evidence that
an activation explains a behavior. Do not tune the prompts, threshold or
analysis after seeing the model outputs. Preserve an incomplete or negative
result rather than replacing it with a more favorable run.

## Fixed inputs and measure

Use `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, with its pinned tokenizer JSON
SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`
and safetensors SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The repository's offline `load_distilgpt2` verifies its complete asset list.
The exact ordered texts are `The {subject} {context}`, with no trailing space:

| Group | Subjects in order | Context | Attractor number |
| --- | --- | --- | --- |
| 1 | cat, cats, dog, dogs | near the windows | plural |
| 2 | teacher, teachers, doctor, doctors | in the room | singular |
| 3 | artist, artists, writer, writers | beside the rivers | plural |
| 4 | farmer, farmers, sailor, sailors | behind the house | singular |

Singular subjects have grammatical sign `+1`, plural subjects `-1`.
Congruence means that subject and context-noun numbers match. For each text,
encode with the pinned tokenizer and `add_special_tokens=False`; use one
batch-one int64 input row, its all-one attention mask, and the final prefix
position. Before any forward, require that appending ` is`, ` are`, ` was` and
` were` each preserves the complete original token row and adds exactly one
token, IDs 318, 389, 373 and 547 respectively. Record the text, all IDs and mask values, token count,
final position, zero-based subject position and token ID, subject number,
attractor number and congruence. Require `The` plus the subject to append one
token and the full token row to be exactly that prefix followed by the
tokenized context. For each quartet,
record whether all four prefix lengths match and whether the tokenized
`" " + context` is an identical suffix of all four rows. Any failed
quartet or next-token alignment aborts the run before any model forward.
Before this protocol was frozen or any model output was observed, a
tokenizer-only structural check showed that the initially proposed `bakers`
subject split into two tokens. `writer, writers` replaced `baker, bakers`
solely to restore single-token subject alignment; all other prompts remained
unchanged. Both the original proposal and this amendment remain recorded here.

The margin is the float64 arithmetic difference of the two extracted float32
logits, `logit[318] - logit[389]`. A prompt is correct only when the margin
times the grammatical sign is strictly positive; a zero margin is incorrect.
Also save the ` was` and ` were` logits and their float64 difference as a
descriptive secondary measure; this does not change the present-tense success
criterion. Decode each target from its raw little-endian IEEE float32 bits
using integer sign, exponent and fraction into binary64, including subnormals
under float32 flush-to-zero, and save both bits and the resulting Python float.
Before seeing any logits, label the baseline **informative under this fixed
criterion** only if at least 12 of 16 prompts are correct overall, and at least
5 of 8 are correct in each of the four overlapping strata: singular subjects,
plural subjects, congruent attractors and incongruent attractors. Otherwise the
outcome is a retained negative baseline. The small, deliberately constructed
sample does not estimate a population accuracy or establish generalization.

## Prepublication and exposure

Commit the source, tests and this protocol before generating the frozen
manifest and before the first model forward. The tokenizer-only structural
check above was performed before this version was frozen. Offline preparation
produces a versioned manifest of all 16
exact text, token, mask and alignment rows. Commit and publish that frozen
manifest and its source revision before the first model forward. Reserve the
16 complete rows and every nonempty prefix in the versioned pretrained exposure
inventory before the run. The later study custodian must bind the complete
exposure-union digest and check its own discovery and held-out policy; these
public development rows are already exposed and cannot subsequently be
presented as untouched held-out examples. No held-out inputs are opened here.

## Execution and retained evidence

Run in a fresh CPU-only Python process with `CUDA_VISIBLE_DEVICES=-1`, PyTorch
float32 eager attention, model evaluation mode, no cache or custom hooks, one
intraop CPU thread, deterministic algorithms and seed `2026102817`. The seed,
versions, platform, source revision, frozen-manifest digest, verified asset
hashes, device, dtype and actual input rows must be saved. Load the pinned
checkpoint offline. Refuse a dirty checkout or source/manifest bytes that do
not match their declared Git commits. These checks do not attest to every
dependency, native library, filesystem operation or runtime property.
Start both the manifest preparation and execution with `python -B -X
pycache_prefix=<absolute, nonexistent path>`; the entry point rejects other
settings before importing project code. This prevents timestamp-valid stale
bytecode reads in this trusted process. It does not attest to the Python
interpreter or external dependencies.
Request float32 denormal flushing disabled in the execution thread, and check
both subnormal-input and subnormal-output arithmetic before and after the model
work. Abort if either probe fails. These checks do not attest to transient
changes during kernels or to other threads; replay may differ with a different
runtime or arithmetic environment.

Forward the 16 rows in the order above, once each, without batching or
interventions. Preserve an append-only attempt log before each forward, and
save all four target logits and both margins for every completed row. Save each full
last-token vocabulary logit vector as raw little-endian float32, in row order,
with shape, byte offsets and SHA-256 digests. Exactly 16 vectors of 50,257
elements are planned (3,216,448 raw bytes), below a 4 MiB artifact cap. The
cap does not bound model memory or intermediate tensors. Verify output shape,
dtype, finite values, disabled cache and unchanged registered model state.
Record the full-archive digest, all per-row metrics, stratum counts and
criterion. A completed output is one fixed-run observation, not a performance
benchmark or causal finding.
Run the forward process under the versioned supervisor, which separately saves
its exit status, stdout and stderr. Accept `result.json` only with supervisor
exit status zero and a terminal `complete` event whose digest matches the
exact saved result bytes. A result file without both conditions is incomplete.
The event and digest establish saved-byte consistency, not successful fsync
or a clean exit on their own; the separate status captures observed process
failure. Filesystem and supervisor reliability remain trusted assumptions.

An independent reviewer checks the frozen inputs, tokenizer alignment,
archived float32 target values, margins, strata and threshold computation.
Do not recapture or save activation donors during this preflight; a later
intervention study must capture its own controls on its bound execution
device and runtime.
