# Copying-head comparison at matched projection-output dose

The fixed output-dose specificity criterion was **not met**. All eight families were evaluable, but H10 minus H2 was positive in **5/8** families, below the preregistered 75% requirement; its mean was +0.07808. The separate input-dose comparison was also **not met**: 5/8 positive, mean +0.35527. The earlier [input-dose study](../distilgpt2-copy-head-prompt-dose-v1-20261029/README.md) and [zeroing comparison](../distilgpt2-copy-head-followup-v2-20261028/README.md) remain available.

The public [protocol](../../docs/DISTILGPT2_COPY_OUTPUT_DOSE_V1.md) fixed eight new word families, each with one repeated and three shuffled prompts. For each family, all four no-op captures set a single projected-output dose: the minimum of the 12 H2, H10 and H7 projected norms. Individual head slices were attenuated to that common dose, with exact float32 landing and unchanged other positions checked. H7 is one layer-matched active comparator; no-op and clean-after controls also ran. The primary outcome is `C = D_H10 - D_H2`, where `D_h` is the head's target next-token log-probability change on the repeated prompt minus the mean change on its three shuffles. A positive result required at least six evaluable families, positive mean C, and C > 0 in at least 75% of evaluable families. This is a fixed descriptive rule, not a significance test.

| Family | Output dose | D H2 | D H10 | D H7 | C: H10 − H2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.7880 | -0.23457 | -0.10104 | -0.06536 | +0.13352 |
| 2 | 7.9112 | -0.55244 | -0.37752 | -0.03730 | +0.17492 |
| 3 | 2.4264 | -0.00588 | -0.02528 | +0.02813 | -0.01940 |
| 4 | 1.8683 | -0.09457 | -0.08216 | -0.00569 | +0.01241 |
| 5 | 7.1994 | -0.23780 | +0.03228 | +0.06147 | +0.27008 |
| 6 | 2.0996 | -0.06413 | -0.07948 | -0.01849 | -0.01535 |
| 7 | 11.4975 | -0.22652 | -0.09446 | +0.04562 | +0.13206 |
| 8 | 2.4524 | -0.00815 | -0.07175 | +0.04757 | -0.06360 |

Mean output-matched D was -0.17801 for H2, -0.09993 for H10 and +0.00699 for H7. Both H2 and H10 had lower D than H7 in all eight families; that comparison is descriptive and does not change the failed H2-versus-H10 criterion or establish uniqueness. The complete per-family and per-prompt values, including the secondary input-dose arms, are in [family-metrics.csv](family-metrics.csv) and [prompt-metrics.csv](prompt-metrics.csv).

| Output-matched arm | Mean target log-probability change | Mean KL(clean ∥ arm) | Mean total variation | Full-vocabulary top changed | Mean input L2 removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2 | -0.07029 | 0.01433 | 0.05243 | 2/32 | 1.4153 |
| H10 | +0.01295 | 0.00813 | 0.04105 | 1/32 | 1.7857 |
| H7 | +0.00780 | 0.00584 | 0.03270 | 1/32 | 1.4347 |

The requested output dose ranged from 1.8683 to 11.4975 across families, a 6.15-fold range. Across the 32 prompts, the actual output-dose means were approximately 4.90534 for each head; the largest relative per-arm error, vector residual and within-family range were 6.84e-7, 2.74e-6 and 1.04e-6, each below the fixed 1e-4 tolerance. The first shuffled variant set the family minimum every time: H2 in 3 families and H10 in 5. Actual H2/H10 output-vector cosines ranged from 0.1924 to 0.3667. Matching a norm therefore leaves different output directions and downstream effects, and the required input removals differ.

In the secondary input-matched arms, mean input removals were approximately 3.84831 for all three heads, but mean projection-output changes were 13.3682 for H2, 10.6461 for H10 and 13.1063 for H7. The two schemes use different intervention fractions; their descriptive contrast does not establish an equal functional dose or a unique copying mechanism. On the clean prompts, the target was the unique full-vocabulary top token in 4/32 cases: 3/8 repeated and 1/24 shuffled. These measurements concern next-token probabilities and do not establish successful generation. The families are small, public and selected after earlier study results; held out from those analyses does not mean blind selection, population representativeness or absence from pretraining.

The 288-action CPU run retained 80,604,288 bytes of raw tensors. The independent [saved-byte reader](../../audits/copy_output_dose/saved_independent_audit.py) reconstructed the scores, full input/landing witnesses, both decisions and actual projected vectors, and verified the retained projection rows against the pinned checkpoint. Its full report is [independent-audit.json.gz](independent-audit.json.gz). The compressed [result](result.json.gz), [journal](attempt.jsonl.gz), [run manifest](run-manifest.json) and [process projection](process-projection.json) are provided with [prelaunch boundary](prelaunch-boundary.json) and [hashes](sha256s.json). The 32 full raw float32 streams and retained projection-weight bytes remain local; the public package alone cannot reproduce the full saved-byte replay. The process record reports direct-child exit 0, reaping, empty stderr and a clean checkout; it is a cooperative observation, not hardware, interpreter, clock or descendant attestation. Public exposure checks are point-in-time and do not cover unknown or private inputs.

To replay with the preserved local raw streams and pinned checkpoint, use the linked independent reader with `--run <closed-attempt> --cases experiments/distilgpt2_copy_output_dose_cases_v1.json --checkpoint <pinned-safetensors> --journal-sha256 b4f4b1a74dfb372d0bf3ea5c44f8d6b163e3d4c9e4c214374344c8eba0fbe6d0 --result-sha256 c6347a9ff6ca606387c4fd799bf4bdd5455d6019b9494b59c226198aa7130e23`. The linked [table builder](build_result_tables.py) accepts the cases, uncompressed result and uncompressed independent audit. Original raw streams, process records and their hashes remain undisturbed locally.
