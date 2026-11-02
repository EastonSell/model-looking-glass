# FLAN-T5 saved-logit float64 cross-check

This supplemental audit independently checks the previously published
[FLAN-T5 CPU feasibility evidence](../flan-t5-causal-cpu-v1-20261029/README.md).
It reads the 44 saved raw logit arrays, verifies each metadata/data hash
against the published manifest, checks target EOS and decoder shifts, and
recomputes selected token log probabilities using the Python standard
library's float64 log-sum-exp. No model is loaded or executed.

The baseline eligibility decisions and the book/house intervention decision
match the original record. The maximum absolute difference between a
float64 recomputation and a recorded float32 per-token score was
`8.207355641332015e-06`, below this check's explicit `1e-4` per-token
tolerance. This numerical cross-check does not independently establish model
execution, intervention landing, or the correctness of the control checks;
the original evidence bundle contains a separate bit-level control audit.

To repeat, run `python high_precision_score_check.py RAW_DIRECTORY MANIFEST_JSON`,
where `RAW_DIRECTORY` is the original `raw/` folder and `MANIFEST_JSON` is
the original `manifest.json`. The script pins that manifest's SHA-256
`a62822eb557657d9220cd4e35a34662b1d726657afe6a1968e3a5121b0004e8a`
and the original terminal SHA-256
`57cbbc319fe4dcc84f3f3a4c50a5bceb633cf653886e8fe16d379cccd9f09afc`.
The saved output is `result.json`. Float64 library behavior may vary by
platform; the tolerance and observed difference are reported explicitly.
