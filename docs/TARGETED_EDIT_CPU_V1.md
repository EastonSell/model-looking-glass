# Exploratory targeted edit on a pretrained transformer: CPU protocol v1

This version defines the procedure independently of any outcome. Publish its
source and frozen manifest before the recorded model run. This is
one counterfactual, not a correction of a fact: Paris is the capital of
France, and we deliberately test whether a bounded edit can increase the model's
preference for the single token ` London` over ` Paris` after prompts about the
capital of France. The model is not to be deployed with the edit. We measure
one-token logit contrasts, not whether generated text answers a question or
whether the model has acquired a belief.

This is a minimal parameter-edit experiment inspired by the separation of
generalization and specificity in [ROME](https://arxiv.org/abs/2202.05262) and
the broader failure modes in [RippleEdits](https://aclanthology.org/2024.tacl-1.16/).
It is **not** the ROME algorithm or a causal localization result. A useful
one-site intervention does not establish that the site uniquely stores or
computes the association.

## Model and candidate

Use the repository's offline, hash-verified `distilbert/distilgpt2` checkpoint
at revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, float32 CPU, eager
attention, no generation cache, evaluation mode, one intra-op and one inter-op
thread. Record the pinned asset hashes and runtime/architecture, imported code
hashes and repository revision. Use no GPU. Keep all checkpoint and model
parameter bytes private; store only hashes, scalar diagnostics and output
logits. Tokenize without special tokens, padding or truncation. Confirm both
` London` and ` Paris` are exactly one token; otherwise fail the run.
Require the actual model to be `GPT2LMHeadModel` with 81,912,576 unique
parameters, the tokenizer to be `GPT2TokenizerFast`, all parameters on CPU in
float32, all modules in evaluation mode, and configured attention to be eager.
Record and validate Python, PyTorch, NumPy, Transformers, CPU build and GPT-2
source identities. Hashed local paths are guessable fingerprints, not
anonymization or proof of package contents.

In the fresh study process explicitly request denormal preservation with
`torch.set_flush_denormal(False)`. On the execution thread, verify before and
after discovery and each forward that the minimum float32 subnormal multiplied
by one has raw bits `0x00000001` and the minimum normal multiplied by one-half
has raw bits `0x00400000`; also require NumPy float32-to-float64 conversion of
the minimum subnormal to produce exactly `2**-149`. Record these probes and
abort on disagreement. Set and verify the same mode for saved-output analysis.
These checks detect the stated denormal cases; they do not attest every
floating-point mode or native operation.

The single discovery prompt is `The capital city of France is`. With target
token ` London` and competitor ` Paris`, the discovery objective is the last
token's unnormalized logit margin `m = logit(London) - logit(Paris)`. At the fixed
site `transformer.h.5.mlp.c_proj.weight`, compute its gradient on that prompt
only. Use the gradient direction, rescaled to a planned Frobenius norm of
0.01 times the original tensor's Frobenius norm. This is one bounded,
single-tensor update, without an optimizer, iterative tuning, additional
examples or bias edits. A 1% Frobenius bound alone does not establish a small
operator-norm perturbation or a small behavioral effect. At this GPT-2 `Conv1D` projection, with weight shape
`[3072, 768]`, capture the final-position input `h` and the objective gradient
`g` of its final-position output; construct `outer(h, g)` in float32. Require
the full autodifferentiated weight gradient to agree with this outer product
using `rtol=1e-5, atol=1e-7` and the float64 global relative Frobenius residual
`||gradient - outer|| / ||gradient|| <= 1e-5`; record the residual and require
finite, nonzero factors and gradient.
Compute factor and original-weight norms in float64. Scale the outer product
in float64 to the planned norm and cast to float32 for addition. Record a
digest of the intended float32 delta and the realized difference between
stored float32 weights, measured by subtracting their float64 casts. The update
is constructed as rank one before finite-precision rounding; no exact rank is
claimed for the rounded realized difference.

The control is one independently seeded random rank-one update of the same
shape, at the same site and with the same planned Frobenius norm. Draw two
independent float32 standard-normal vectors on CPU from a dedicated PyTorch
generator seeded `20261029`, form their float32 outer product in the same
orientation, and scale in float64 then cast as above. It is a single draw,
not an estimate of a random-edit distribution. Report both
realized norms, relative norms, and their relative difference after float32
rounding; require the realized ratio to lie in `[0.999, 1.001]` and both
relative norms to be at most `0.01001`. If either condition fails, terminate as
failed rather than describe the edits as equal-magnitude. Freeze the seed in
the manifest. Preserve a private copy of the original tensor, restore it after
each condition and verify exact byte hashes, including on exception. Hash the
whole model state before and after all conditions; a mismatch invalidates the
run. Record whether parameter flags, gradients and hooks were restored; an
unverified cleanup invalidates the run. Gradients must be cleared, and no
edited checkpoint is saved.

## Fixed cases and chronology

Before any model forward, encode and register every prompt and all of its
known autoregressive prefixes in the project's exposure inventory; retain the
report and disclose that the inventory is incomplete. The tokenizer-only step
is allowed before the model study. Publish the frozen protocol, exact token IDs,
cases, targets, metrics, thresholds, code identity and fixed random seed before
the recorded run. Source bytes, pinned assets and the frozen manifest must be
verified at execution. Public prompts are fixed; this is a held-out split from
parameter selection, not a protected blind benchmark.

All 13 public rows were checked against the prior known inventory before
reservation; none was identical to a known complete row or to any recorded
row's nonempty prefix. The [public reservation snapshot](https://github.com/EastonSell/model-looking-glass/tree/41158c397a60febb4c7464aa393261a2346e7b9a)
is commit `41158c397a60febb4c7464aa393261a2346e7b9a`, with canonical payload
SHA-256 `e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1`.
This rule does not reject longer rows merely for sharing a shorter opening
prefix and cannot establish anything about unknown history or pretraining.

Only the discovery prompt may be forwarded before the gradient and random
candidate are fixed. Before forwarding any target or locality case, atomically
save a precommit record containing the discovery objective, original tensor
digest, both candidate digests, shapes, norms, seed and the frozen-manifest
digest. If precommit fails, no held-out forward is allowed. Evaluate conditions
in the fixed order original, gradient edit, random edit, restoring the identical
original tensor between conditions. For each condition evaluate the discovery
prompt followed by all target and locality cases in the listed order. No
selection, retuning, or omission may depend on held-out outputs.

Target held-out prompts:

1. `The capital of France is`
2. `France has its capital in`
3. `In France, the capital city is`
4. `France's capital city is`

Separate locality prompts and their expected single-token continuations:

| Prompt | Continuation |
| --- | --- |
| `The capital of Italy is` | ` Rome` |
| `The capital city of Spain is` | ` Madrid` |
| `Japan's capital is` | ` Tokyo` |
| `The capital of Germany is` | ` Berlin` |
| `The largest ocean is the` | ` Pacific` |
| `Two plus two equals` | ` four` |
| `The opposite of hot is` | ` cold` |
| `A week has seven` | ` days` |

The expected labels are predefined probes, not claims that the unedited model
answers correctly. Verify each is exactly one token and every prompt is within
a fixed 32-token limit; also verify that tokenizing each prompt plus its
continuation yields precisely the frozen prompt IDs followed by the chosen
single continuation ID (for both contrast tokens on discovery and target
cases). Fail instead of truncating or changing cases. The
protocol uses only these 13 public inputs and does not inspect another team's
reserved split. Future custodian prefix checks must establish known disjointness
against these exposures; prior exposure and model pretraining remain unknown.
The scope is four closely related paraphrases and eight hand-selected locality
probes; it does not measure general capabilities or establish preservation
outside them.

## Measures, decision, and evidence

For every condition and prompt retain the entire final-position float32 logit
vector in a bounded, checksummed raw output artifact, plus case ID and token
digest. Record per-case margins on the discovery and target prompts, local
expected-token log probabilities, top token IDs and `KL(original || edited)`
on all locality prompts. Calculate and save raw paired values and declared
aggregates for both edited conditions. An independent reader must be able to
recompute the metrics from these logits without running the model. For each
saved float32 vector `z`, cast to float64 and compute `logp = z - (max(z) +
log(sum(exp(z - max(z)))))`, with the sum in float64. Expected-token NLL is
`-logp[y]`. KL is `sum(exp(logp_original) * (logp_original - logp_condition))`,
also summed in float64; report the raw value without clamping. The top token is
the smallest token ID attaining the maximum. Margins are float64 subtractions
of the saved float32 logits. The logits are model outputs, not saved model
weights; no activation traces are needed.

Predeclare this demanding success rule before running: the original margin
`m` is negative on discovery and on at least three target cases; the gradient
edit makes `m` positive on discovery and on at least three of those originally
negative target cases; its mean held-out target margin improvement exceeds 1.0 logit
and exceeds the random control's mean improvement by at least 0.5 logit;
all eight locality top tokens remain unchanged, mean locality KL is at most
0.02 nats, maximum KL at most 0.10 nats, and mean expected-token negative-log-
probability deterioration at most 0.05 nats. A failed criterion is a result,
not a reason to tune or discard the run. If the original model does not show
the prerequisite Paris preference on enough cases, classify the case as
ineligible for this success rule and still report every measurement. Report
the same target and locality metrics for the random edit, even when it fails.
No statistical confidence claim is justified by these few correlated prompts.

Create a new output directory and never overwrite a previous attempt. Save a
start record before model work, the precommit before held-out work, per-case
raw outputs and phase records as they complete, and one success, failure or
cancelled terminal with cleanup status. Partial/incomplete attempts are never
promoted to success. Verify finite values, exact case and shape counts, checksums,
source/model identities and restoration before reporting a success terminal.
Runtime and source hashes support repeatability but do not prove the execution
or every imported/native byte. Because weight and delta bytes are not saved,
the claimed edit, norms and rollback remain run attestations until independently
replayed; logits alone only support recomputation of output metrics. Save an
independently executed replay if possible. This study is CPU-only and is not a
latency benchmark.

The operator publishes source and manifest separately before launch. A later
reader can check the Git ancestry and the reported wall-clock sequence, but
neither local clocks nor Git commit timestamps independently prove when a
forward happened or when a remote object became available.

## Commands

Use the same recorded interpreter and offline checkpoint cache throughout.
The source commit must already contain this protocol, the runner and both
inventory modules. Create the manifest at a new tracked path, then commit and
publish it as a descendant before executing the run:

```text
python -B -m looking_glass.targeted_edit_cpu_v1 freeze --source-commit <source-commit> --output <repo-relative-new-frozen.json>
python -B -m looking_glass.targeted_edit_cpu_v1 run --manifest <committed-frozen.json> --manifest-sha256 <frozen-sha256> --cache-dir <offline-hub-cache> --output <new-absent-result-directory>
python -B -m looking_glass.targeted_edit_cpu_v1 audit --manifest <committed-frozen.json> --manifest-sha256 <frozen-sha256> --output <result-directory>
```

For the study and a later fresh-process replay, set `CUDA_VISIBLE_DEVICES=-1`,
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before Python starts. Use a
fresh externally created empty bytecode-cache prefix for each process and
disable bytecode writes. This reduces ordinary stale-cache reuse; it does not
attest every startup hook, sourceless import or native byte. The read-only audit
recomputes metrics and checks links between saved records, not the truth of
their execution or rollback assertions.
