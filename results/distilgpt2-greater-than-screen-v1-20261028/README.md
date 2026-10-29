# DistilGPT-2 year comparison: wording shifts a persistent later-year preference

The fixed 48-prompt screen found a positive forward-minus-reverse contrast in
**all 16 matched triples**, with a mean of **+0.615031** log units. However,
the reverse wording did **not** reverse the preference: the model favored the
eight later suffixes over the eight earlier ones in all 16 reverse prompts,
and also in all 16 neutral prompts. The outcome supports a wording-dependent
shift atop a persistent later-suffix preference on these cases; it does not
establish a general temporal comparator or reproduce a circuit mechanism.

For each prompt, `M = log(sum probability of eight later suffixes / sum
probability of eight earlier suffixes)`. The sets have equal size and are
equally distant from the given year; the exact-year token is excluded from
both. Mean `M` was +2.176175 for forward wording, +1.561144 for reverse, and
+1.517678 for neutral. Mean forward-minus-neutral was +0.658497, while mean
reverse-minus-neutral was +0.043466 (10 positive, 6 negative). The neutral
wording is a contextual control, not a perfect subtraction of historical or
language priors.

| Start year | Event | Forward M | Reverse M | Neutral M | Forward − reverse |
| --- | --- | ---: | ---: | ---: | ---: |
| 1712 | war | +3.121621 | +1.822999 | +1.832896 | +1.298622 |
| 1712 | reign | +2.955705 | +1.526814 | +1.779268 | +1.428891 |
| 1721 | war | +2.289747 | +1.995525 | +1.746821 | +0.294221 |
| 1721 | reign | +2.341487 | +1.828580 | +1.803952 | +0.512907 |
| 1732 | war | +1.721716 | +1.478265 | +1.135350 | +0.243451 |
| 1732 | reign | +1.596982 | +1.213890 | +1.068557 | +0.383093 |
| 1743 | war | +2.171470 | +1.373185 | +1.406249 | +0.798286 |
| 1743 | reign | +2.080194 | +1.221487 | +1.351621 | +0.858706 |
| 1754 | war | +2.180430 | +1.699037 | +1.660458 | +0.481393 |
| 1754 | reign | +2.372625 | +1.744381 | +1.778937 | +0.628244 |
| 1765 | war | +1.738154 | +1.559473 | +1.415435 | +0.178681 |
| 1765 | reign | +1.703863 | +1.417110 | +1.329171 | +0.286753 |
| 1776 | war | +1.934042 | +1.226756 | +1.165216 | +0.707286 |
| 1776 | reign | +2.074825 | +1.098893 | +1.154217 | +0.975932 |
| 1787 | war | +2.278833 | +1.952691 | +1.848913 | +0.326142 |
| 1787 | reign | +2.257099 | +1.819215 | +1.805783 | +0.437884 |

The full-vocabulary probability mass assigned to context-aligned `01`–`99`
next tokens averaged 99.911% in forward, 99.833% in reverse and 99.858% in
neutral prompts. Averaged by condition, approximately 35–44% of the full
vocabulary mass lay on other aligned suffixes outside the 16 local candidates
and the exact year; the reported comparison is specific to the selected sets.
The `00` next-token choice is reported separately: a tokenizer
can emit that token after the fixed prefix, but canonical re-encoding of the
resulting full year merges the boundary. Individual candidate logits,
set-level probability masses, all raw final-position vectors and all 16
contrasts are saved; individual candidate probabilities can be reconstructed
from the raw vectors.

## Scope and evidence

These are two event nouns, eight fixed suffixes, one century and three public
deterministic templates. Different words and positions occur in the forward,
reverse and neutral sentences, and suffix frequency priors need not cancel.
The cases are correlated, non-blind and not a model-wide accuracy estimate.
The task is related to [Hanna, Liu and Variengien's GPT-2-small study](https://arxiv.org/abs/2305.00586);
a recent [DistilGPT-2 preprint](https://arxiv.org/abs/2610.37157) reports its
own results on a different or insufficiently specified case/checkpoint set.
We performed no activation intervention and make no claim about its proposed
mechanism.

The [fixed protocol, cases and source](https://github.com/EastonSell/model-looking-glass/pull/345)
were public before the CPU run. It used source revision
`47d42b8137203fa8e5ffdc60942a1adece6c0feb` and the verified
`distilbert/distilgpt2` checkpoint revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`; weight SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The direct process exited zero with 48 observed returns and records, no failure
record, empty stderr, and unchanged model state and buffers. CUDA was hidden.
The [manifest](run/manifest.json), [journal](run/attempt.jsonl),
[result](run/result.json), [raw logits](run/last_logits.f32),
[pre-run record](pre-run.json), and [direct-process receipt](supervisor.json)
are retained. The receipt and journal are operator/process observations,
not independent execution proof or trusted timestamps.

Two independent saved-byte replays verified all 48 score vectors, candidate
masses, 145 ordered journal events and 16 triple contrasts without another
model run. The [raw audit](raw-audit.py) and [output](raw-audit.json) retain
per-case results; the [independent audit](independent-audit.py) and
[output](independent-audit.txt) recompute the scores with a separate direct
sum. From a Git checkout containing the frozen source commit, both scripts can
be run with Python; the independent script's path setup was adapted for the
package, and it verifies the source blobs with `git show`.

Key SHA-256 values:

* `pre-run.json`: `b666715ebc84e601c645cd1d88eb4bed8e711bba9bd7f2f6502321a0b8265a3b`
* `run/manifest.json`: `5590f02a6becd05daffb0915088bbe66e41b3cc8dce2310030fb87162d63ee1c`
* `run/attempt.jsonl`: `f280ef79bdd6b0d6aea66d8c35e0868d4a18193bc5ebb7654f20b25f2b82ac3a`
* `run/last_logits.f32`: `e2cd020bc219d1e042fea90ecd098d95c0ccbf653e3526cfab1d6c39cb9b4737`
* `run/result.json`: `65e48fdd8bfdb2a69455037e5ad059731106793014c366b06896252bb35970ef`
* `supervisor.json`: `469b61cf70916434d2e96779ba842ba1dd9e2c50be9fd511a76ac8d3894bcd48`
