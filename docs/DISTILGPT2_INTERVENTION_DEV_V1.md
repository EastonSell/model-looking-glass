# DistilGPT2 ordered-intervention development smoke v1

This is a **development-only CPU correctness check** for the ordered
activation-plan implementation on one pretrained model and one public synthetic
input. It tests a known intervention against an independently assembled forward
path. It does not locate an unknown change, evaluate a held-out case, establish
a model's reasoning, measure GPU execution or support a behavioral
generalization claim. A same-site clean patch is expected to reverse this
particular constructed injection; that recovery is not evidence of a discovered
mechanism.

## Frozen setup

- Load `distilbert/distilgpt2` revision
  `2290a62682d06624634c1f46a6ad5be0f47f38aa` through the offline verified
  loader, with `allow_download=False`. It verifies the full pinned asset
  inventory, including `model.safetensors` SHA-256
  `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
- Use the literal token-ID row `[50256, 50256, 50256, 50256]` with attention
  mask `[1, 1, 1, 1]`. ID 50256 is the checkpoint tokenizer's EOS token. This
  is an intentionally synthetic public development input, not ordinary text.
  No study discovery or held-out inputs are run or inspected. The future study
  custodian must exclude and check this exact row against both actual splits
  before commitment and fail or version the study protocol on any collision;
  disjointness is not asserted by this smoke.
  The associated random-model software tests forward the separate public row
  `[1, 2, 3, 4]`; include that row in the same future exclusion inventory.
- CPU float32, evaluation mode, eager attention, no KV cache, one intra-op and
  one inter-op thread, fixed seed 1729, deterministic algorithms, and CPU
  flush-to-zero enabled. The runner records the actual environment and checks
  CPU placement and explicit CUDA-entry guards. It verifies flush-to-zero with
  a raw-bit subnormal multiply probe. It performs no download.
- Inject at the output of `transformer.h.2`, invocation 0, batch index 0,
  token index 2: add exactly +1.0 to hidden coordinate 0 and zero elsewhere.
  The patch source is the clean vector for the same input, module, invocation,
  batch and token, captured independently by the manual reference. All steps
  use the declared output-completion order and bounded activation storage.
  Record the actual post-add minus pre-add coordinate by decoding the two
  stored float32 values to binary64 before subtraction; rounding may make it
  differ from the nominal real-number increment. Since a vector add
  under flush-to-zero can change subnormal values even where the delta is zero,
  require raw-bit equality for every other hidden coordinate before and after
  injection, and require the selected coordinate's bits to match its own
  prescribed float32 `before + 1.0` result, and require the decoded realized
  delta to be strictly positive. Fail if any of these checks fails. Require
  raw-bit equality between the independent clean donor and the plan's
  pre-injection vector, and again between the post-patch vector and that donor.
- Construct the reference without `ActivationPlan` or modification hooks:
  add token and position embeddings, explicitly construct the causal mask,
  run the six model blocks in order, modify the chosen block's returned tensor
  directly, then apply the final layer norm and language-model head. The same
  immutable model weights are used by both paths; this is an independent
  execution pathway, not an independently trained or independently loaded
  model. Check the manual clean forward against an untouched model forward.

## Comparisons and controls

All comparisons cover every float32 logit at every input position; all compared
values and captured vectors must be finite. A comparison passes only if its
maximum absolute difference is at most `1e-6`, with relative tolerance zero.
The absolute tolerance accommodates CPU floating-point operation ordering and
is not a claim of equality across arbitrary hardware or software versions.
To avoid flush-to-zero corrupting the calculation of a difference, decode each
stored IEEE float32 bit pattern to binary64 using integer sign, exponent and
fraction fields: for exponent zero use `sign * fraction * 2**-149`, and for
exponents 1 through 254 use `sign * (2**23 + fraction) * 2**(exponent-150)`.
Reject exponent 255. Subtract the decoded values in binary64 and take the
maximum absolute value across the flattened row-major tensors. This records
differences between stored values even when float32 arithmetic would flush a
subnormal to zero. Raw-bit checks remain separate from this numeric criterion.

1. Untouched clean model logits agree with the manual clean forward. Repeat a
   clean forward to detect state changes.
2. Plan injection agrees with manual injection, both in the selected
   post-injection activation and in all logits. The injection must change at
   least one logit from clean by more than `1e-5`; otherwise fail instead of
   reporting a masked or inactive test as success. Repeat with a fresh plan.
3. Inject then patch the same site with the manual clean vector. Check the
   pre- and post-patch vectors and all logits against the manual inject-then-
   patch forward and the untouched clean run. This restoration is expected by
   construction at the modified boundary.
4. A `null` plan on the clean path agrees with the untouched clean run. A
   capture followed by a same-site patch from that capture agrees with clean;
   an injection, capture and self-patch of the post-injection value agrees with
   the injection-only result. These are forward-value controls, not gradient
   equivalence claims. Capture immediately after each control and require
   bit-exact equality with its own pre-action or source activation. Also record
   whether the output comparisons are bit-exact. Also require the logits at
   pre-intervention positions 0 and 1, for both injection and patch, to agree
   with clean logits within the same `1e-6` tolerance; their bit equality is
   recorded separately from that numeric criterion.
5. Reject preexisting local or global forward, pre-forward and backward hooks.
   Confirm declared step metadata, invocation, shape, source and selector; check hook
   registries after every forward and require plans to restore the empty
   baseline. This cooperative check cannot detect a transient hostile change.
   Hash all model parameters and buffers in bounded chunks before and after;
   require equal state hashes. A final untouched clean forward agrees with the
   first one. Reject nonfinite, malformed, missing or failed observations
   instead of silently omitting them.

## Record and limits

Commit and publish the protocol and runner before any checkpoint forward and identify
their exact source revision and raw byte hashes. The recorded time is not a trusted
independent timestamp. Each project process must use Python 3.11 or later and the direct base
interpreter with `-B -E -s -S -P -X pycache_prefix=<fresh-empty-dir>`,
establishing the initially empty private prefix at interpreter startup,
with standard-library paths before the explicitly selected package directory. Load
the project package explicitly by source path without exposing the checkout root
as a general import path. Check source origins and imported raw project bytes against the recorded Git
revision, record the actual flags and cache state, and make no network request.
The operator creates and verifies the empty private prefix before launch; the child
rechecks it only after interpreter and standard-library startup, so these checks
do not protect against a hostile host or a pre-start cache race.
`-E` ignores `PYTHONHASHSEED`, so Python hash randomization remains enabled;
the CPU model generator is explicitly seeded and output mappings are sorted.
These checks do not attest to every native library or defend against a hostile
host. CUDA and network guards cover selected Python entry points after the initial
Torch import; they do not instrument every native call. Use a unique output
directory; write the attempt manifest before model loading and preserve a failure
record when storage permits. Bootstrap failures before the manifest or disk-write
failures must also be retained in the launcher's exit status and stderr. A successful record includes checkpoint/asset hashes, exact input IDs,
seed and determinism settings, package versions, CPU-only guard status, full
output shapes/dtypes and SHA-256 hashes of contiguous little-endian float32
logit bytes for **each** reference, plan result and control, selected activation
hashes, all maximum errors, injection magnitude, hook-cleanup and completion
status. Hashes enable exact digest comparison, but a reader cannot recompute
the original full-tensor maximum errors or finiteness checks from hashes alone;
that requires retained tensors or an independent replay. Do not overwrite
previous attempts. Before publishing, review paths and failure tracebacks for
private information and redact sensitive fields without concealing the outcome.

One input, one fixed injection and one pretrained model are insufficient to
evaluate generalization, assess a candidate localization policy, or establish
causal explanations. Any future study must use its own frozen, disjoint splits,
controls and intervention predictions.
