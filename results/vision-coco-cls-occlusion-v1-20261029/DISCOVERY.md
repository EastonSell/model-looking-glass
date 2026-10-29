# Exposed-image CLS versus spatial discovery

All four exposed discovery cases were **limited-dose**: their shared absolute dose was less than 2% of at least one receiver norm, so none is eligible for the fixed primary denominator. The following effects are descriptive; they are not a 0/4 primary negative result. These cases were chosen after observing their earlier occlusion outcomes. The preregistered fresh-image screen remains to be run.

Each entry is a signed drop in log probability of the annotated species group. A positive value means the change decreased that group's probability. The seven arms share one planned absolute dose per image, with realized doses matched within 1%; `C` is the CLS token and `A` and `B` are the animal and background spatial blocks. Random effects are individual controls, not a distribution.

| Image | Species | Animal mask | Background mask | C from animal | C from background | A transfer | B transfer | C random | A random | B random |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 404350 | cat | +0.001558 | −0.098187 | −0.003019 | +0.000480 | −0.000363 | −0.000301 | −0.000244 | +0.000263 | +0.000079 |
| 151497 | cat | −0.009216 | −0.013605 | −0.000393 | −0.000228 | −0.000011 | +0.000005 | +0.000388 | −0.000052 | +0.000019 |
| 470798 | dog | +0.224702 | −0.035809 | −0.002422 | −0.002450 | −0.001118 | +0.000218 | −0.002392 | +0.000426 | +0.000050 |
| 135116 | dog | +0.010196 | −0.002290 | −0.000039 | −0.000076 | −0.000026 | −0.000002 | +0.000025 | +0.000088 | +0.000004 |

`B` is the animal-mask drop minus the background-mask drop. `T_C` and `T_S` are the corresponding CLS and spatial transfer contrasts. `D = |B − T_S| − |B − T_C|` is positive when the CLS contrast is numerically closer to B. The final columns give the planned common dose as a fraction of each receiver norm.

| Image | B | T_C | T_S | D | q/n_C | q/n_A | q/n_B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 404350 | +0.099745 | −0.003498 | −0.000062 | −0.003436 | 0.005499 | 0.003451 | 0.003370 |
| 151497 | +0.004389 | −0.000166 | −0.000016 | −0.000150 | 0.007055 | 0.004464 | 0.004283 |
| 470798 | +0.260510 | +0.000027 | −0.001336 | +0.001363 | 0.006174 | 0.003694 | 0.003854 |
| 135116 | +0.012486 | +0.000037 | −0.000024 | +0.000061 | 0.008168 | 0.004547 | 0.005214 |

Across the 28 matched transfer/random arms, realized dose/receiver fractions were 0.003370–0.008168; the largest relative difference between a realized dose and its q was below 0.00000201. Whole-vocabulary TV was 0.000034–0.002591, KL(clean‖changed) was 0.000000012–0.00001859, and off-group conditional TV was 0.000082–0.003462. None changed top-1. Including the eight input-mask conditions, maximum whole-vocabulary TV was 0.454666 and two of 36 displayed conditions changed top-1. Thus the limited-dose transfer comparison does not imply the model was unaffected by masking.

The full 92-forward saved evidence, including all logits, inputs, controls, requested and landed token states, is indexed in [discovery-evidence-index.json](discovery-evidence-index.json). The [independent saved-byte replay](discovery-independent-audit.json) checks those controls and the decision arithmetic; a separate [60-digit logit review](discovery-decimal-review.json) checks the signed drops. Their scope is saved-byte consistency, not proof of execution or publication timing. The [direct observer](discovery-observer.json) records a reaped, zero-exit child without timeout or forced stop; it does not attest the process tree. The forward-forbidden [preflight](preflight-terminal.json) recorded zero submissions and zero saved forwards.

To reproduce the primary replay, reconstruct the discovery archive from the published parts and extract it to a fresh directory. Reconstruct the earlier [occlusion evaluation evidence](../vision-coco-occlusion-v1-20261029/README.md) separately; its extracted `data` directory is needed to bind the exposed input and logit bytes. Extract [discovery-replays.zip](discovery-replays.zip), then run:

```sh
python restore-discovery.py --index discovery-evidence-index.json --output /tmp/cls-occlusion-discovery.zip
python -m zipfile -e /tmp/cls-occlusion-discovery.zip /tmp/cls-occlusion-discovery
python -m zipfile -e discovery-replays.zip /tmp/cls-occlusion-replays
python /tmp/cls-occlusion-replays/vision_cls_occlusion_intervention_independent.py \
  --manifest ../../experiments/vision_coco_cls_occlusion_v1.json \
  --prior-manifest ../../experiments/vision_coco_occlusion_v1.json \
  --prior-audit ../vision-coco-occlusion-v1-20261029/evaluation-independent-audit.json \
  --prior-data /path/to/restored-occlusion-evaluation-data \
  --data /tmp/cls-occlusion-discovery --report /tmp/new-discovery-audit.json
python /tmp/cls-occlusion-replays/vision_cls_occlusion_discovery_decimal_portable.py \
  /tmp/cls-occlusion-discovery discovery-independent-audit.json /tmp/new-decimal-review.json
```

The Decimal helper pins the published independent-audit bytes; its separate arithmetic checks the saved logits. The ZIP also retains the original local-layout review source. The new screen is conditional on both recognition and an observed mask contrast; it will preserve exclusions and negative outcomes. Masking introduces artificial pixels, and transferring mixed token states can be off-manifold. These observations do not establish CLS mediation, exclusive localization, population effects or pretraining independence.
