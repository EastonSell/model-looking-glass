# Copying factorial: additive-logit reference, pre-forward clarification

This analysis-only clarification is published after the public factorial
case file was assembled but before any of its model forwards. The original
[factorial protocol](DISTILGPT2_COPY_HEAD_FACTORIAL_V1.md), its cases,
interventions, family dose, order and primary outcomes remain unchanged. The
case file being fixed at this point has SHA-256
`69f8463da5f919fb99aef05aa4401de439248100e99ca90899f8d9f98af90d24`.

The common dose is computed from all four variants and all three heads in a
family. It is therefore an intervention rule for that fixed prompt bundle:
replacing a shuffle could change the repeated prompt's dose even if the
repeated prompt itself stayed unchanged. It is not a context-independent
per-prompt dose.

The joint intervention's requested input-slice L2 is approximately `sqrt(2)`
times a single intervention's requested L2. Moreover, log-softmax can create
an output-scale interaction even if the full vocabulary logits are additive.
To describe that effect without additional model forwards, use each prompt's
saved full-vocabulary clean, H2 and H10 logits to construct a reference:

```
z_add[j] = (z_H2[j] - z_clean[j]) + z_H10[j]
```

Decode each saved binary32 logit bit pattern into binary64 as specified in the
original protocol and evaluate the subtraction and addition in the written
binary64 order. Calculate the reference target log probability using the
same maximum-subtracted, `math.fsum` log-sum-exp rule without rounding the
constructed vector back to binary32. This vector is a
constructed reference and is never used as model input or a predicted target
chosen after viewing outcomes.

For each variant let `delta_add = logp_add - logp_clean`; for each family use
the original protocol's fixed three-shuffle `math.fsum(...)/3` order to obtain
`D_add`. Report all per-prompt reference deltas and the following per-family
values alongside the original arm components and observed `J_family`:

```
J_add = (D_add - D_H2) - D_H10
joint_minus_additive = D_joint - D_add
```

Report the eight family values and their `math.fsum(...)/8` means for each.
In exact arithmetic, `joint_minus_additive` equals `J_family - J_add`; the
separate floating-point evaluation orders need not produce identical bits.
If reconstruction or its finite-value checks fail, preserve the original
measurements and mark this reference incomplete, rather than silently
substituting a value. Independently rederive the reference from saved logits
before publication.

`J_add` is a **descriptive additive-logit reference**, not the exact
normalization component of the observed `J`, a randomization distribution,
significance test, or empirical H7-pair intervention. The
observed difference from it can include nonlinear downstream effects,
rounding and other changes induced by the interventions; it does not establish direct
H2–H10 cooperation, mediation, or a unique circuit. Fixed public cases and
three fixed shuffle orders retain the original protocol's non-blind limits.
