# CPU targeted edit v1: target shift with locality failures

**The prespecified result is negative.** A single 1%-relative parameter edit
substantially shifted the chosen next-token contrast, but failed all three
distribution/NLL locality limits and changed the top token on two of eight
locality probes. Four locality prompts concern other countries' capitals and
four concern separate topics. This deliberately counterfactual experiment does
not correct a fact or identify a unique mechanism. The original fact is that
Paris is the capital of France.

The [frozen protocol](../../docs/TARGETED_EDIT_CPU_V1.md) and
[case manifest](frozen.json) specify one DistilGPT2 CPU discovery prompt, four
public target paraphrases held out from fitting, eight separate public locality
prompts, the edit, one norm-matched random control, and the decision rule. The
contrast is the final-position logit margin `London - Paris`; a positive margin
does not establish that either token is the top prediction or that a generated
answer would contain it.

| Prespecified measure | Gradient edit | One random edit | Gradient success requirement |
| --- | ---: | ---: | --- |
| Discovery margin, original −1.2681 | 14.1209 | −1.2688 | Greater than 0 |
| Reversals among the 3 initially negative target cases | 3 / 3 | 0 / 3 | At least 3 |
| Mean margin change over all 4 target cases | +11.7549 | −0.00048 | Greater than 1.0 and at least 0.5 above random |
| Locality top tokens unchanged | 6 / 8 | 8 / 8 | 8 / 8 |
| Mean locality KL(original \|\| edit), nats | 0.18287 | 0.0000365 | At most 0.02 |
| Maximum locality KL, nats | 0.74500 | 0.0000809 | At most 0.10 |
| Mean expected-token NLL deterioration, nats | +0.28551 | −0.00911 | At most 0.05 |

One of the four target paraphrases already preferred London to Paris in the
original model; it contributes to the mean change but was not counted as a
reversal. The two locality prompts with changed top tokens were the Japan and
Germany capital prompts, both shifting to token ` London`. Original and
edited values for every case, including all failed criteria, are in the saved
reports and raw logits. The random result is one fixed draw, not a distribution
of controls.

The selected tensor was `transformer.h.5.mlp.c_proj.weight`. Its recorded
original Frobenius norm was 306.7527609682. The planned norm of each update
was 3.0675276097; the reported realized relative norms were
0.0100000000204 for the gradient edit and 0.0099999999887 for the random edit.
The recorded full-gradient versus constructed outer-product relative residual
was zero in this run. These quantities, the parameter and model-state hashes,
and successful hook, gradient, flag, and weight rollback are run attestations:
weight and update bytes are deliberately not published. A 1% Frobenius bound
does not imply a small operator norm or small behavioral effect.

## Records and replay

The source commit is `1eee5d8589386fbf8256afac19b99518805f4bfc`; the
descendant manifest commit is `ce025e994d28969384bd7fbd07b9e9e8e220c8f7`,
and the manifest SHA-256 is
`93d953494cca7f2fcf0e7cd8baec35fa9ca1b68945e1e44f183e75e10ed03d1d`.
The manifest and its source ancestry were independently fetched from the
published branch before the reference launch was announced. The reviewer's
local clock showed 2026-10-29 02:27:59 UTC; the reference reported a start at
02:28:33 UTC and the replay reported a start at 02:29:46 UTC. These local
clocks and Git timestamps are not trusted proof of remote publication or
execution timing.

[Reference](reference/) and [replay](replay/) are separate fresh processes on
the same machine, interpreter, offline checkpoint and frozen source. Both
reported complete cleanup and a `negative` decision. Their 39 ordered
observation records and all three raw logit files are byte-identical; their
precommit and report fields also match after excluding the distinct run IDs
`f97470d3ff334b85828590b8a6e94c3f` and
`194a0e8cd40d47b1a7f9c2129707a583`. This is a same-host repeat, not a new
statistical sample or proof of execution. Independent reviewers recomputed the
paired metrics, controls, thresholds and artifact links from the saved files.

Each `original.f32`, `gradient.f32`, and `random.f32` is a C-order little-endian
float32 matrix of shape `[13, 50257]`, ordered as in the manifest. Both run
directories retain all files and their independent terminal inventories. The
matching file SHA-256s are:

| File in each run | SHA-256 |
| --- | --- |
| `original.f32` | `947518214fd7c8bdfebd347a1944cc4ce5cb97ac9e622846cc7e21308f96c309` |
| `gradient.f32` | `43e94ab1a840df8c0d4fb65e56966befb06110a05edae9094e3817515cf339f7` |
| `random.f32` | `1d7fd5030e7d54bbab5ce3796f1a2e719e82c1e136de64294e061babf1c9fef7` |
| `observations.jsonl` | `9fec61b36d704e38946f7528769d5817f0e18dd2f39ba11de97ed4698f8aaff2` |

From a checkout with the frozen runner, recompute the saved-output decision for
each run without loading the model:

```text
python -B -m looking_glass.targeted_edit_cpu_v1 audit --manifest results/targeted-edit-cpu-v1-20261028/frozen.json --manifest-sha256 93d953494cca7f2fcf0e7cd8baec35fa9ca1b68945e1e44f183e75e10ed03d1d --output results/targeted-edit-cpu-v1-20261028/reference
python -B -m looking_glass.targeted_edit_cpu_v1 audit --manifest results/targeted-edit-cpu-v1-20261028/frozen.json --manifest-sha256 93d953494cca7f2fcf0e7cd8baec35fa9ca1b68945e1e44f183e75e10ed03d1d --output results/targeted-edit-cpu-v1-20261028/replay
```

Both runs hid CUDA, used one CPU thread and an offline, hash-verified checkpoint.
The operator-created bytecode prefixes were empty before and after both runs;
this mitigates ordinary stale-cache reuse but does not attest all startup hooks,
imports, native code or floating-point modes. The denormal probes cover the
calling thread only. The cases are public and non-blind, are held out only from
fitting the edit, and have unknown pretraining and historical exposure. Four
related paraphrases and eight hand-selected probes do not establish general
capability preservation or justify claims beyond this particular edit.
