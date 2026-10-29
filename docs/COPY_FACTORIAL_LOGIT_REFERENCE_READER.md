# Saved factorial additive-logit reference

`scripts/copy_factorial_logit_reference.py` reconstructs the pre-forward
[additive-logit clarification](DISTILGPT2_COPY_FACTORIAL_LOGIT_REFERENCE_V1.md)
from a completed, quiescent factorial attempt. It fixes the 32-case file,
protocol, clarification, v11 and v12 digests, 50,257-element vocabulary and
61,250,432-byte raw-payload total. It checks the direct-process receipt and
file hashes, 224-action order, every recorded logit-slice hash, clean/no-op/
clean-after logit equality, and the observed H2/H10/joint scores and family
interactions. Observed score and summary comparisons use an absolute `1e-12`
tolerance; the new reference itself uses the specified binary64 order without
casting the constructed logits back to float32.

The JSON output contains each prompt's original and reference scores and
deltas, all eight `D_add`, `J_add` and `D_joint - D_add` values, and their
means. It hashes the saved attempt journal through the operator's receipt
but **does not parse or reconcile its events**, validate intervention landing
or all primary controls, authenticate the receipt/source, or rerun the model.
The separate full saved-data audit must check those boundaries. A failure here
leaves the additive reference incomplete; retain the primary measurements and
do not substitute a value.

For a completed attempt whose operator recorded the exact public runner SHA:

```sh
python scripts/copy_factorial_logit_reference.py \
  --attempt PATH_TO_ATTEMPT \
  --case-file experiments/distilgpt2_copy_head_factorial_v1.json \
  --source-revision EXACT_PUBLIC_RUNNER_SHA
```

The supplied source SHA must agree with the pre-run record, manifest and
receipt; that consistency is not proof of when the bytes were published or
executed. The pure calculator's tests include a fabricated additive-logit
example with a nonzero log-probability interaction, a non-additive joint,
subnormal/signed-zero decoding, float64 reference construction, incomplete
receipt and corrupted control/summary cases. No model is needed by the reader
or its tests.
