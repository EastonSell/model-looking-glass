# DistilGPT2 public copying: six head/position interventions

**Result: the prespecified H10-specificity comparison did not pass.** Zeroing layer 3 head 10 at the final input position reduced the repeated-versus-shuffled target log-probability contrast by an average of 0.478150 nats, but zeroing head 2 at the same position reduced it by 1.036665 nats. Head 2 produced the more negative contrast change in seven of eight families. The eight families are public exploratory examples; these differences are not estimates of a population effect or evidence of a unique copying circuit.

Each family has one repeated-sequence prompt and three matched shuffles. For each intervention, define Δ as ablated minus clean target log probability and **D = Δ(repeated) − mean(Δ(shuffles))**. Negative D means the intervention decreased the repeated-versus-shuffled contrast. The table retains both components: H10’s D combines a −0.201049-nat repeated change with a +0.277101-nat shuffle change. H2 combines −0.761527 and +0.275138 nats. The full differential is not solely the repeated-input change.

| Intervention | Mean D | Mean repeated Δ | Mean shuffled Δ | Repeated top-1 | Shuffle top-1 | Mean slice L2 (range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L3 H10, final position 10 | -0.478150 | -0.201049 | +0.277101 | 6/8 | 1/24 | 4.858 (0.601–7.194) |
| L3 H2, final position 10 | -1.036665 | -0.761527 | +0.275138 | 5/8 | 1/24 | 4.075 (0.455–7.251) |
| L3 H7, final position 10 | -0.007277 | +0.007324 | +0.014601 | 6/8 | 1/24 | 5.348 (2.699–5.976) |
| L3 H3, final position 10 | +0.158051 | +0.111203 | -0.046848 | 6/8 | 1/24 | 1.313 (0.541–2.191) |
| L3 H1, final position 10 | -0.032525 | -0.014905 | +0.017620 | 6/8 | 1/24 | 1.307 (0.385–3.198) |
| L3 H10, earlier position 5 | +0.007061 | +0.020337 | +0.013276 | 6/8 | 1/24 | 0.871 (0.517–3.485) |

Clean target top-1 was 6/8 for repeated prompts and 1/24 for shuffles. H10 zeroing left the repeated count at 6/8; H2 zeroing yielded 5/8. Every ablation changed the intended 64 float32 bit patterns, with all other recorded projection-input coordinates unchanged. Pre-ablation slice norms varied across heads and positions, so these zeroing interventions are not magnitude matched; their contrast differences cannot isolate head identity from intervention magnitude.

The following table preserves all eight family contrasts. H10 minus H2 is the paired difference between their D values: positive values mean H2’s reduction was larger. Family 7 slightly favored H10; family 3 was close to zero.

| Family | H10 final | H2 final | H7 final | H3 final | H1 final | H10 pos. 5 | H10 − H2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.110271 | -0.247448 | -0.071528 | +0.068512 | -0.089123 | -0.006134 | +0.137177 |
| 2 | -0.388460 | -1.437213 | -0.023171 | +0.179389 | -0.052801 | -0.015497 | +1.048753 |
| 3 | -0.721498 | -0.740544 | -0.025329 | +0.270863 | +0.014807 | -0.031726 | +0.019047 |
| 4 | -0.859934 | -2.742085 | -0.283682 | +0.374380 | -0.057645 | +0.069618 | +1.882151 |
| 5 | -0.228730 | -0.603810 | +0.524585 | +0.085374 | -0.094406 | -0.002915 | +0.375080 |
| 6 | -0.036455 | -0.464603 | -0.055710 | +0.115816 | +0.048081 | +0.051865 | +0.428148 |
| 7 | -0.773725 | -0.763139 | +0.100397 | +0.002619 | +0.065482 | -0.006588 | -0.010586 |
| 8 | -0.706129 | -1.294476 | -0.223777 | +0.167454 | -0.094600 | -0.002135 | +0.588347 |

## Design and controls

The candidate H10 was selected from the earlier public attention screen; H2 was the second-ranked same-layer head. H7 is the prior fixed layer-matched comparator, while H3 and H1 were selected by fixed hashes before assembling the new cases. The H10 position-5 arm tests a different position and downstream route. The new rows and their token IDs were distinct from the two earlier copying datasets and reserved in exposure inventory v9. The protocol was published before the case file; the cases are nevertheless public, non-blind and informed by earlier results. Known-inventory checks do not establish independence from pretraining or unknown prior exposure.

At each of 32 prompts, the runner performed unhooked clean, a no-op, six zeroing arms and unhooked clean-after: 288 CPU forwards total. The six arms rotate by `(zero_based_family + 2 * variant) % 6`; the order is deterministic and not perfectly balanced. Full final-position vocabulary vectors and pre-action, post-action and projection-output witnesses are retained. All clean/no-op/clean-after full vectors matched exactly, all pre-action inputs matched the no-op witness, and each ablation landed on the intended head slice with other input bits unchanged. There were no failed controls. Zeroing can be out of distribution; attention weight and ablation sensitivity alone do not establish a natural mechanism.

## Evidence and provenance

- [Frozen protocol](https://github.com/EastonSell/model-looking-glass/blob/a9f29a5421ba40404708d48913c7d599be69a429/docs/DISTILGPT2_COPY_HEAD_FOLLOWUP_V2.md), SHA-256 `70616bb01aa7ccb8440da9dc503c274a3d5cb7b24dd3965fd985de6c645ca68c`.
- [Fixed case inventory](https://github.com/EastonSell/model-looking-glass/blob/939d59540b642c83bc999bcf45220ff2155f41b2/experiments/distilgpt2_copy_head_followup_v2.json), SHA-256 `3831bfd1a48aaa3fcb67fbdb3cdc10f4422149b1fed90fd5267d3c73525255a8`; v9 exposure digest `e0c394786b316ea61b9f3697063eb86c7cf290d898c3ed84549880df8af91c4d`.
- [Frozen executable source](https://github.com/EastonSell/model-looking-glass/tree/40e98bc5e851a7be95035f8a60b266042422fff9), reviewed tree `b179c1e0610b9559720881aa249cfe76fc525cd8`. The earlier [screen](../distilgpt2-copying-dev-v1-20261028/README.md) and [single-comparator intervention](../distilgpt2-copy-head-eval-v1-20261028/README.md) explain the candidate selection.
- [Result](run/result.json), SHA-256 `4a6c9251a5752714525836df2c29c9e1a8c8659430720a79da47e5f4f3fc18f1`; [manifest](run/manifest.json), [ordered attempt log](run/attempt.jsonl), and 32 adjacent `.f32` files preserve 80,604,288 raw bytes. The manifest and receipt contain all file hashes and raw offsets.
- [Pre-run record](pre-run.json), SHA-256 `c5a16a605e2e270ddf10d4387856a082bac95673e822d36ac1da50bcacbab360`; [observed-process receipt](supervisor.json), SHA-256 `2ffc6adc91d51b3b6b91db587ad011d11eb72efc8d1eb614d0868fc1cff38ea9`; [stdout](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilgpt2-copy-head-followup-v2-20261028/stdout.bin) and empty [stderr](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilgpt2-copy-head-followup-v2-20261028/stderr.bin). The direct CPU process exited 0; all 288 starts have matching raw-recorded and complete events, and no failure file was present. The receipt is an operator observation, not independent descendant or timestamp attestation. No automatic retry was made.

The run used the pinned `distilbert/distilgpt2` checkpoint on CPU, float32 eager attention, one intra-op thread, and no cache. The manifest records the checkpoint and selected Transformers source-file hashes, library versions, and the absent `MKL_CBWR` and `ATEN_CPU_CAPABILITY` variables before PyTorch import; runtime checks rechecked them. These settings were not recorded in the earlier run, so exact cross-run runtime identity is not claimed. Source publication timing relative to the local launch is recorded by the operator rather than independently timestamped by this receipt.

The [saved-byte audit](audit_saved_bytes.py) and [its output](audit.json) reconstruct the journal, all logits/scores, witness delivery, controls, source bindings and family summaries. Two independent reviewers separately replayed the saved vectors and comparisons; their [raw replay](independent_raw_replay.py) and [science replay](independent_science_replay.py) are included with only their local path setup adapted to this result directory. They check the retained records, not an independent model re-execution or the faithful execution of native code. The science replay requires NumPy; the raw replay and primary audit use the Python standard library.

From a checkout containing the frozen source commit, reproduce the main audit with:

```sh
python results/distilgpt2-copy-head-followup-v2-20261028/audit_saved_bytes.py --attempt results/distilgpt2-copy-head-followup-v2-20261028 --repository . --source-revision 40e98bc5e851a7be95035f8a60b266042422fff9 --exposure-v9 e0c394786b316ea61b9f3697063eb86c7cf290d898c3ed84549880df8af91c4d
python results/distilgpt2-copy-head-followup-v2-20261028/independent_raw_replay.py
python results/distilgpt2-copy-head-followup-v2-20261028/independent_science_replay.py
```
