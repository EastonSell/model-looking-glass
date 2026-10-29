# Public DistilGPT-2 copying-head ablation

Zeroing layer 3, head 10 at the final input position reduced the
repeated-versus-shuffled target log-probability contrast by **0.6381 nats on
average** across eight new public families. The same operation on a fixed
layer-matched control head 7 increased that contrast by **0.1298 nats**;
the mean candidate-minus-control difference was **−0.7679 nats**. Both the
candidate contrast change and the paired difference were negative in all
eight families. The unmodified mean contrast was +5.8816 nats and remained
+5.2435 after the candidate ablation: the intervention reduced rather than
eliminated this measured behavior.

The contrast has two components. Under head 10 zeroing, the target log
probability fell by an average of 0.3528 nats on repeated examples and rose
by 0.2854 nats on their matched shuffles. Under head 7 zeroing the respective
changes were −0.1544 and −0.2841 nats. The target remained top-ranked on all
eight repeated examples after head 10 zeroing, as in the clean condition;
head 7 zeroing left it top-ranked on seven of eight. Thus the detected
probability change did not change repeated-case top-one accuracy for the
candidate intervention.

Each table entry is an ablated-minus-clean change in target log probability;
the shuffle column averages the three fixed shuffled inputs in that family.
`D` is repeated minus mean shuffle change, and `D10 − D7` compares the heads.

| Family | H10 repeat | H10 shuffle | D10 | H7 repeat | H7 shuffle | D7 | D10 − D7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | −0.1382 | +0.6958 | −0.8340 | −0.1542 | −0.6395 | +0.4852 | −1.3192 |
| 2 | −0.1946 | +0.5757 | −0.7704 | −0.1134 | +0.0380 | −0.1514 | −0.6189 |
| 3 | −0.9184 | +0.1085 | −1.0269 | −0.4923 | −0.6485 | +0.1562 | −1.1831 |
| 4 | −0.5853 | +0.1890 | −0.7742 | +0.0031 | −0.5884 | +0.5915 | −1.3657 |
| 5 | −0.6067 | +0.0811 | −0.6878 | −0.0722 | −0.1540 | +0.0818 | −0.7696 |
| 6 | −0.0438 | +0.1564 | −0.2002 | −0.0060 | +0.0622 | −0.0682 | −0.1320 |
| 7 | −0.1334 | +0.1357 | −0.2691 | −0.3240 | −0.1061 | −0.2179 | −0.0513 |
| 8 | −0.2017 | +0.3406 | −0.5423 | −0.0759 | −0.2369 | +0.1610 | −0.7033 |
| Mean | −0.3528 | +0.2854 | −0.6381 | −0.1544 | −0.2841 | +0.1298 | −0.7679 |

The head was selected from a separate [public development screen](../distilgpt2-copying-dev-v1-20261028/README.md)
using its attention contrast, not a causal test. The [fixed follow-up protocol](https://github.com/EastonSell/model-looking-glass/blob/947332a52aa9db197cd8624f839ed8c50c68154a/docs/DISTILGPT2_COPY_HEAD_EVAL_V1.md)
and [exact cases](https://github.com/EastonSell/model-looking-glass/blob/947332a52aa9db197cd8624f839ed8c50c68154a/experiments/distilgpt2_copying_head_eval_v1.json)
were published before the evaluation forwards. The 32 evaluation inputs are
disjoint from the screen and were reserved in the [v7 known-exposure inventory](../../docs/PRETRAINED_EXPOSURE_INVENTORY_V7.md).
The head and control were chosen before the new rows were assembled, but that
ordering was not committed at the time. These are public exploratory examples,
not a blinded or independently preregistered holdout, and known-exposure
checks cannot rule out unknown prior or pretraining exposure.

The CPU run used the pinned `distilbert/distilgpt2` checkpoint at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. The [runner source](https://github.com/EastonSell/model-looking-glass/blob/5458c5d54ddba5ca024e495e236519d376971583/scripts/copying_head_eval_v1.py)
was frozen at `5458c5d54ddba5ca024e495e236519d376971583`. Each case ran in
the fixed order clean, no-op, head 10 zero, head 7 zero, and clean after
hook removal. There was no randomization or counterbalancing of action order.
The intervention sets one 64-component head slice to zero at the final
position immediately before the layer's attention output projection; it
leaves the other components and projection bias in place. Every zero action
changed all 64 selected float32 bit patterns. All no-op and post-clean full
final-logit vectors matched the clean vectors bit-for-bit; delivered slices
and untouched components passed their recorded checks.

Zeroing can be out of distribution. The result supports sensitivity to these
specific final-position interventions on these examples; it does not identify
a unique natural mechanism, establish a complete induction circuit, or rule
out compensation and order effects. Head 7 is one layer-matched control,
not a sample of a population of random heads and not magnitude-matched.

[Manifest](run/manifest.json), [event log](run/attempt.jsonl), and
[results](run/result.json) retain all 160 actions, scores, controls and
per-family comparisons. The 32 `run/copy-eval-*.f32` files retain all
41,896,576 raw bytes in little-endian float32; each completed event gives
offsets, lengths and SHA-256s for the full 50,257-element final-logit vector
and, for hooked actions, `[1,11,768]` pre-action, delivered and projection
output witnesses. [Pre-run record](pre-run.json) and [process observation](supervisor.json)
record the reviewed source, inputs, direct app process exit 0 and absence of
a failure file. This is an operational observation, not proof of descendant
supervision or a trusted external timestamp. Standard output and empty
standard error are retained alongside it.

Two independent reviewers reconstructed every saved final-logit score and
family contrast, and checked the delivered slices, untouched bits and
bitwise no-op/clean controls. A separate audit also reconciled the event log,
all saved witness hashes and fixed source/model/runtime bindings.
The before/after model-state hash matched the earlier screen. These audits do
not rerun the model or prove faithful execution. Run the portable [audit](audit.py)
from a checkout with the source revision available using:

```
python results/distilgpt2-copy-head-eval-v1-20261028/audit.py --attempt results/distilgpt2-copy-head-eval-v1-20261028 --repository . --source-revision 5458c5d54ddba5ca024e495e236519d376971583
```

A useful next test would freeze several additional layer-3 heads, including
the second-ranked attention head and several independently selected controls,
and compare final-position versus earlier-position ablations on new disjoint
families. That would test the specificity of this dependence and possible
position effects; it would still not establish a complete circuit.
