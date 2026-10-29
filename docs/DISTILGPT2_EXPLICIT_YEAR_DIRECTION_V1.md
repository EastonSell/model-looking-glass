# Protocol: explicit year-direction behavior and intervention

This is a prospective, public, non-blind CPU study of `distilbert/distilgpt2`
revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, with weight SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`
and tokenizer SHA-256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
It asks whether explicit `before` and `after` wording produces opposite local
suffix preferences consistently across these fixed cases. Independently of that clean outcome, a
bounded intervention asks whether an opposite-year displacement at the final
layer-3 block output produces a generic donorward preference shift across all
three wordings. The clean gate controls eligibility for the separate
comparator-like intervention interpretation. A passing pattern would be
prompt-specific evidence of an output effect, not proof of arithmetic
comparison, a unique circuit or general numerical ability. A failed gate or
intervention criterion is a result.

## Development basis and chronology

The published [year-transfer result](https://github.com/EastonSell/model-looking-glass/tree/d350a3613a9f6626dddc3e37c0e4a728b8fde10a/results/distilgpt2-year-transfer-v1-20261029)
found its clean symmetric comparator gate failed 0/8, while a distinct
preference gate passed 6/8. The final-position year-directed displacement
shifted the fixed-band preference donorward under forward, reverse and neutral
wording, yet only 1/8 met the forward-specific intervention criterion. The
new wording and thresholds are informed by that public result and are not an
independent preregistration of the earlier experiment. The four year pairs,
templates, metrics and stopping rules below are fixed before any full new
study cases or model outputs are generated.

Before this protocol, the pinned tokenizer was used without a model to check
the individual temporal words and candidate year/suffix components below.
This was a structural check, not a pretrained forward or a behavior screen.
The complete 48 prospective prompts have not yet been assembled. The
earlier GPT-2-small study by [Hanna, Liu and Variengien](https://arxiv.org/abs/2305.00586)
motivates studying year continuations, but this is a distinct smaller-model
experiment, not a replication of its circuit.

## Fixed inputs and tokenization

Use the four ordered start-suffix pairs `(07,16), (29,38), (51,60),
(73,82)` in century `17`, and the ordered nouns `war, reign`. Within each
pair/noun, use both anchor years and the exact templates below, with the
wording order `after, before, near`:

```text
The {noun} took place after the year 17{suffix} in the year 17
The {noun} took place before the year 17{suffix} in the year 17
The {noun} took place near the year 17{suffix} in the year 17
```

Thus the output is intended as a year after, before or near the anchor,
respectively. `Near` is a matched contextual comparison, not an assumed null.
The design contains 48 different prompts. No outcome may be used to replace
or omit a prompt. These deterministic related examples are not random draws.

For each pair `(a,b)` with `a < b`, hold the middle band `{a+3,a+4,a+5}`
and the above-both band `{b+3,b+4,b+5}` fixed for both anchor years and all
wordings. For each individual anchor `y`, also define a local below band
`{y-3,y-2,y-1}` and above band `{y+1,y+2,y+3}`. Hold those same six local
suffixes fixed across the three wordings for that anchor. They all lie within
01–99 for the selected anchors. Pair-specific bands may overlap the local
bands; they serve different declared comparisons.

After this protocol is published, construct and publish the full ordered case
manifest. With the exact pinned tokenizer, verify each full prompt; require
all three wordings of a given pair/noun/anchor to have identical token lengths
and differ only at the single temporal-word token. Require low/high inputs
within each wording to differ only at the single anchor-suffix token. Verify
the final token is ` 17`, every 01–99 suffix appends as one distinct token to
the unchanged complete prompt. Within each local below/above comparison and
each pair-specific middle/above-both comparison, require six distinct IDs;
overlap between those two comparisons is allowed.
`00` is a separately measured next-token ID: canonical retokenization of the
full `1700` text need not equal that token continuation. Require unpadded
length at most 32. Structural failure stops the attempt; it does not select a
replacement after seeing any model outcome.

Compare the fixed public rows and candidate continuations with the latest
reviewed public exposure inventory, reserve them before a forward, and record
any separately reported new exposure. Only a genuinely applicable shared
protocol requires a protected-split comparison; do not make unrelated
studies a prerequisite. Exposure outside these records, including pretraining,
is unknown. Publish the exact cases, reviewed executor and a pre-run record
before the first pretrained forward.

## Clean metrics and deliberately strict comparator gate

Decode each saved final-position float32 vocabulary vector exactly to
binary64. For a contrast of ordered bands `A` and `B`, let `u=max(A)`,
`v=max(B)`, `p=math.fsum(exp(z-u) for z in A)` and
`q=math.fsum(exp(z-v) for z in B)`, with suffixes in ascending order; compute
`((u-v)+log(p))-log(q)` in that binary64 order. Evaluate subsequent written
subtractions in the order shown. For each prompt define the within-anchor
local directional score

```text
W = logsumexp(above_y) - logsumexp(below_y).
```

For each pair/noun and wording define the patch-aligned fixed-band score

```text
M = logsumexp(middle) - logsumexp(above_both)
D_wording = M(low anchor) - M(high anchor).
```

The comparator eligibility gate passes only if at least six of the eight pair/noun
observations, spanning at least three year pairs, each satisfy all six strict
inequalities:

```text
W_after(low) > 0.5       W_after(high) > 0.5
W_before(low) < -0.5    W_before(high) < -0.5
D_after > 0.5           D_before < -0.5
```

Report every `W`, every `D`, neutral/near values and their wording
differences, all six inequalities and counts, and all clean full-vocabulary
logits. Also report full-vocabulary-normalized mass on contextual 01–99
suffix tokens, the separate next-token `00` mass and all remaining vocabulary
mass. The gate compares relative logits in small candidate sets; it is not a
claim that numeric continuations dominate the full vocabulary.

This gate is a deliberately stringent operational condition for interpreting
the comparator-like intervention pattern, not a necessary or sufficient test
for numerical ability. In
particular, at the low anchor under `before`, both fixed middle and above-both
bands lie above the anchor, so their relative score is not determined by
correct temporal ordering alone. Same-anchor `W` avoids that particular
ambiguity, but context-dependent token and calendar associations can satisfy
either pattern without an explicit numerical comparator. The two nouns and
four pairs are related observations, not independent population samples.
Regardless of the comparator gate, continue the fixed, all-case intervention
to evaluate the separately declared descriptive generic-shift question.
Record a gate failure and do not treat the comparator interpretation as
eligible when it fails. Structural, nonfinite, dose or evidence failures
still stop the attempt and preserve available evidence.

## Intervention and controls

Run every fixed case in both year directions, including families that
individually fail either behavioral check. Use the zero-based layer-3 block output
at the final token; donor and target share noun and wording and differ only
in anchor suffix. For each pair/noun compute the three full 768-channel
opposite-year displacement norms from the six saved no-op captures, using
binary64 decoded values and ascending-channel `sqrt(math.fsum(d*d))`. Set the
common input L2 dose to half the smallest of the three. Nonfinite inputs or
dose at most `2**-20` stop the attempt. This choice is based on activations,
not model outputs, and can make all three interventions weak when one norm is
small. Record each norm, dose, fraction and achieved norm.

For every target use four fixed actions: exact self replacement at the final
position; year-directed displacement at the final position; the identical
signed displacement at position zero; and a one-channel cyclic permutation
of that displacement at the final position, where output channel `i` takes
the displacement from `(i-1) mod 768`. Use the same binary64 addition and
software round-to-nearest-even float32 delivery as the prior
[year-transfer protocol](https://github.com/EastonSell/model-looking-glass/blob/38ffcab52256e3952315b7bd9b58d6f10d7921c2/docs/DISTILGPT2_YEAR_TRANSFER_V1.md).
Require every action's actual pre-action bits to match that case's saved
no-op, exact planned versus actual site bits, all untouched coordinates, and
realized L2 error at most `1e-5 * dose` from that same invocation. Verify
natural position-zero low/high no-op equality separately; it is a zero-dose
architectural check, not the nonzero position control. Equal input L2 does not
imply equal downstream effects, and neither nonzero control is assumed inert.

Enumerate pairs, nouns, wordings and years in the orders above; rotate the
four actions left by `(pair_noun_index + 2*wording_index + year_index) % 4`.
For each prompt retain an unhooked clean and a passive no-op before dose
selection; after the four actions retain an unhooked clean-after. Require
bit-identical clean/no-op/self/clean-after full logits and exact hook removal.
Use cached weights, CPU float32, eager attention, no KV cache, one intra-op
thread, the fixed CPU RNG seed `20261029` and the prior runtime/rounding
probes. Keep `MKL_CBWR` and `ATEN_CPU_CAPABILITY` absent before the Torch
import, record their actual presence/value, and guard CUDA and network entry
points cooperatively. Record the actual runtime and model/buffer states; a
fixed seed does not guarantee bit equality across different runtimes.
The complete plan has 336 forwards, irrespective of the comparator gate. Enforce at
most 67,545,408 bytes of final logits and 47,185,920 bytes of block snapshots,
plus independent bounded metadata/partial-evidence limits; these are file
bounds, not an RSS guarantee. Persist available bounded returned evidence and
failure metadata before later validation. Do not automatically retry.

For action `j`, let `E_j = M(patched) - M(clean)`. Use the previous
forward-style donor direction: `s_low = -E_low` and `s_high = E_high`, and
`S_j = (s_low + s_high)/2`. The separately declared generic-shift pattern
requires, in at least six of eight observations, both main-action `s` values
to be positive in **each** of after, before and near, and in **each** wording
`S_main > S_position + 0.1` and `S_main > S_rotated + 0.1`. It is a
descriptive fixed-band output pattern and may occur through lexical or other
contextual associations; it does not establish a generic circuit or exclude
a comparator.

The distinct comparator-like directional pattern is eligible for interpretation
only if the clean comparator gate passed. It requires, in at least six of eight
observations, both main-action `s_after` values positive and both `s_before`
values negative. Define `T_j = S_after,j - S_before,j`; the stricter
control-aware pattern additionally requires `T_main > T_position + 0.1` and
`T_main > T_rotated + 0.1` for each counted observation. If the clean gate
fails, save all directional counts and effects as descriptive observations
but mark both comparator-pattern decisions ineligible. These operating
thresholds are not significance tests. Report every `E`, `s`, `S`, `T`,
near-wording effect, control and nonpositive or reversed outcome; means use
`math.fsum` in fixed pair/noun order divided by eight. No ratio normalized by
the clean contrast is used.

An independently reviewed exact executor and synthetic landing/rounding,
gate and failure-retention tests are required before any pretrained forward.
Retain a direct-process exit observation and independently replay all saved
bytes, controls and gates before reporting. Neither a clean gate nor an
intervention identifies which information is moved by a residual-vector
displacement; lexical and context-dependent alternatives remain live.
