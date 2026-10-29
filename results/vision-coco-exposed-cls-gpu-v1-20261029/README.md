# Exposed-pair ViT CLS versus spatial CUDA exploration

One previously exposed cat/dog pair was run through the pinned pretrained ViT-tiny on an NVIDIA GeForce RTX 2070 SUPER. Both sides had matchable intervention doses, but neither met the prespecified descriptive CLS criterion; the pair result was **negative**. The interventions nevertheless changed the output distributions. This is one selected, previously observed pair, not a held-out v5 evaluation, a population estimate, or evidence that the intervention has no effect.

The frozen protocol was published before the first forward in source commit `8a2ebc71747a3c01b46e124efbbbbf67b7a09eb3`; the corresponding source is merged in PR #607. The selected v4 targets were cat 405058 and dog 196065, with same-species donors 69231 and 142769, at encoder block index 2. The earlier v4 outcomes and donor selection were already known. That earlier spatial-only CPU intervention used a different dose rule, so this GPU comparison is not an exact CPU replication.

The table reports annotation-oriented changes in the dog-minus-cat log-sum-exp score. Positive means movement toward the other species group; each opposite-donor action has same-species and fixed-random controls at the same realized absolute dose.

| Target | Site | Opposite donor | Same species | Random |
| --- | --- | ---: | ---: | ---: |
| Cat | CLS | -0.211378 | -0.200570 | +0.233697 |
| Cat | Four spatial tokens | +0.001295 | +0.007989 | +0.003682 |
| Dog | CLS | +0.039302 | +0.015944 | +0.075447 |
| Dog | Four spatial tokens | +0.012217 | -0.091979 | +0.026915 |

The control-adjusted CLS and spatial advantages were respectively -0.445075 and -0.006695 for the cat, and -0.036146 and -0.079762 for the dog. Opposite-donor CLS total variation was 0.04077 for the cat and 0.08002 for the dog. Cat q was 0.853603 and dog q was 1.042770; all six realized doses per side were within 1% of the corresponding opposite-CLS dose. The complete per-action score, KL(clean || patched), total-variation, landing and control records are in the evidence archive.

Fresh same-GPU clean baselines classified the target cat and dog as top-one classes 282 and 209. Their cat/dog group masses were 0.550738/0.004105 and 0.000341/0.925984. The same-species donors had top-one classes 281 and 173 and cat/dog masses 0.763340/0.003394 and 0.000271/0.987665. Donor recognition was established by the prior CPU screen; it was not an additional GPU recognition gate. Across the four full 1,000-logit clean vectors, the maximum absolute CPU/GPU difference ranged from 1.14e-5 to 1.30e-5; their mean absolute differences ranged from 2.44e-6 to 3.23e-6. Exact CPU/GPU bit equality was not assumed.

## Evidence and resource scope

The direct owner and retained controller each exited 0. The saved outcome is a complete candidate with 34 forwards. Two independent saved-byte executions of the reviewed reader produced the same report SHA-256 `4801de7e66d7c8e679a5a1c2603328f48af12c2ce8727654f4c418ebcada00a0`; a separate 60-digit Decimal calculation reconciled all 12 signed movements within 8.33e-16. The reader checked action ordering, all retained logits, the requested/landed and untouched float32 bits, the clean controls, dose matching and score arithmetic. Its conclusion concerns the saved bytes; the source, execution and process records are separate evidence.

The 40 saved resource stages reported cumulative PyTorch peaks of 60,224,512 B allocated and 62,914,560 B reserved, and minimum sampled device free memory of 7,362,052,096 B. The maximum recorded process peak working set was 902,807,552 B and maximum sampled private bytes were 1,583,718,400 B. The saved local post-cleanup allocated and reserved counters were both zero; model and parameter weakrefs were dead and the recorded active logical lease count was zero. These are observations for this one fixed execution, not general memory or RSS bounds. The independent memory audit SHA-256 is `3d1e847456fc8e8227f5babdf56d12e0bd19bfc0f9e9de3e5035f6bbaf225888`. The post-exit OS point found no marker, attestation or recorded numeric process IDs; it does not establish native CUDA cleanup, a Job drain, or the absence of unobserved descendants.

The owner used float32/eager inference, one host thread, deterministic algorithms, nearest-even boundary checks, preserved denormals and disabled TF32/autocast. The saved runtime identifies PyTorch 2.8.0+cu128, CUDA build 12.8, Transformers 4.57.3, NumPy 2.4.6 and NVIDIA driver 616.92. Ownership, memory and numerical checks are cooperative points, not atomic reservations or continuous state guarantees.

## Reproducible saved-byte package

`manifest.json` lists every member and SHA-256. Concatenate `evidence.zip.part-000`, `-001`, and `-002` in that order. The resulting 11,280,397-byte ZIP has SHA-256 `b3b9cea57dddb6a57a21adfd6ceddb7fb9019312ecbaf9e944fd23672b17443a`. All 193 member hashes and a fresh replay of the reassembled package were checked. The archive contains all 34 forward and return records, the 40 resource stages, the fixed reference payloads, the independent reader and its two source dependencies, and the science and memory audit reports. The public candidate omits local run identifiers, process ID and physical GPU UUID. Its SHA-256 is `a0b0fc61e69c7b2375e64d954dab5703ebe5ac9264fe65693e045acb3ec79da0`; the original private candidate SHA-256 is `f60c2eabff62294f414fee38cbb2ffc31de0a398cf94f1e62cb001a7103fb285`. Replaying the public projection gives audit SHA-256 `bc810a36b6c7f40632d9b1f0ad57393cc04aa6195380ed21eaff68364bc73abd`; apart from the candidate hash, it matches the original independent report.

For example, from this directory use Python to join the pieces, verify the archive digest, extract it, and run `reader/vision_cls_exposed_gpu_independent.py` with `--data data --candidate candidate-public.json --reference reference --report <new-file>`. The independent reader is a saved-data audit and makes no execution or population claim.
