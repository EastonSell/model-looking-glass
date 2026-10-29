# Exploratory causal intervention study: v1

This is a CPU instrumentation sanity check on one untrained residual network,
not an explanation of a pretrained model. The [protocol](CAUSAL_PROTOCOL_V1.md)
was committed before this run. A [pre-run clarification](CAUSAL_PROTOCOL_V1_NOTES.md)
records that a development regression had previously executed one public
held-out input with a different model seed. The recorded v1 results are
therefore **exploratory**, not an untouched blind confirmation.

The [run directory](../results/synthetic-causal-v1-20261028) contains the
manifest, frozen predictions, per-case results, and terminal outcome. The
manifest records source revision `6d0131ee3eb6c2bb7dbef9433f0a8733c759331f`, published on
`research/causal` before the run; it was not on main at that time. Registry
verification checks saved hashes and consistency, while the implementation
and its tests establish the intended precommit-before-held-out execution order.
Neither a hash nor a filesystem timestamp proves that order externally.

## What was tested

The protocol uses one CPU float32 model (width 8, depth 3, seed 2301), eight
fixed discovery vectors, and 16 distinct fixed held-out vectors. A known
`+1.0` change was placed immediately before the `block.1.activation` tap. The
scalar metric was output logit 1 minus output logit 0; the target correction
was its clean value minus its corrupted value. This arbitrary scalar is not a
language task or a preserved-capability benchmark.

The first-discrepancy method and magnitude ranking both chose
`block.1.activation` on discovery inputs. The single seeded random draw chose
`block.2.output`, the full state immediately before the head. Thus the first
two methods selected the **same** intervention, and the final-state patch is
expected to restore the downstream scalar regardless of the earlier injection
site. These are not three independent discoveries or evidence of a unique
mechanism.

Before any v1 held-out forward, the runner published a frozen mean prediction
of `0.0065164510160684586` for each site. On the 16 fixed held-out cases,
the three site-specific effect vectors were exactly identical. That equality
is expected here: patching the same-input clean state at the injection tap or
at the final state removes or bypasses the sole injected perturbation in this
deterministic model.

| Descriptive quantity | Value |
| --- | ---: |
| Prediction MAE against each site's actual patch effect | 0.00200006715 |
| Same-site zero-effect predictor MAE | 0.00670493534 |
| Correct predicted signs among eligible cases | 16 / 16 |
| Absolute patch residual from the clean-minus-corrupt scalar target, mean | 0.0 |
| Target correction range | 0.00356531143 to 0.01280942559 |

The sign tolerance was `1e-7`; all 16 observed effects and the frozen
prediction were positive, so these cases did not test sign variation. The
discovery minimum and maximum in the precommit artifact are descriptive, not
a calibrated prediction interval. These values describe this fixed suite and
model seed; no inferential confidence interval or population claim is made.
The metrics object's bootstrap unit field is dormant because the saved run has
zero bootstrap samples and method `none`.

## Controls and limits

In the recorded execution, null, identity, pre-injection upstream, and clean
null controls passed the runner's full-logit floating-point equality checks. The clean-versus-
clean trace control also passed. Their saved scalar effects are zero; the
individual control logits were not saved, so an independent reader cannot
reconstruct the full-logit control comparisons from the artifact alone. The
reverse patch produced the opposite scalar effect on all 16 cases. These are
conditional results for the particular injection and source/base inputs, not
general necessity or sufficiency claims. The code tests check full-logit
controls, but a future protocol should save their raw logits too.

Patching the candidate restored **the chosen logit difference** on these
cases. The artifact does not establish full-logit restoration by the candidate
patch, and matching the known injected tap in a deliberately dense
instrumentation setup does not establish localization accuracy for unobserved
boundaries or other model families. One model seed, one public split and one
random site cannot establish generality. A fresh, independently unexposed
suite and protocol would be needed before a confirmatory claim.

An independent reviewer checked the registry, source/protocol/model/case
hashes, split identities, precommit hash, discovery summaries, held-out rows,
and reported metrics. A separate reviewer challenged the scientific
interpretation and highlighted the final-state control and unsaved control
logits. Source revision and data identities are retained in the manifest.
The recorded source hashes do not establish the exact executed Python bytecode:
ordinary Python imports can load a timestamp-valid cached `.pyc` while still
reporting the corresponding source path. The recorded run did not bind its
executed bytecode. A [proposed future-v2 launch policy](CAUSAL_CONFIRMATION_V2_SOURCE_CLARIFICATION.md)
addresses ordinary cache reuse without retroactively changing this evidence.
The v1 trace comparator and full-logit controls used floating-point equality
without recording the CPU denormal-flushing mode. In a flushing environment
these comparisons can miss a stored subnormal difference; signed zeros also
compare equal. Flushing can also erase a subnormal contribution to the
discovery magnitude ranking. Full discovery trace values and control logits were not saved,
so this artifact cannot determine whether that edge occurred in this run;
there is no basis here to claim that the recorded result changed. The proposed
v2 protocol specifies a checked numeric mode and bit-level comparisons.

## Recheck the saved artifact

From a checkout containing the recorded run:

```bash
python -c "from looking_glass.registry import verify_run; print(verify_run('results/synthetic-causal-v1-20261028')['state'])"
```

The saved precommit SHA-256 is
`f5a3719b0b95b1436f3831956af25d4a14723d7ffd9d186453689ad616f288f0`;
the saved result SHA-256 is
`298e35840226e21cb36beff2f0092c64ed9502c88f3c758bc5f9b3336c77d4a2`.
The suite SHA-256 is
`883b28112ccee7605125959a4c8209c9e20bb438b73938b323e3fc80994f1e8e`.
