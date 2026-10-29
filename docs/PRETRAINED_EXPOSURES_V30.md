# DistilGPT-2 development exposures: v30 status reconciliation

This is a status-only successor to the reviewed v29 public inventory. It
preserves all 915 input records' token rows and the same 5,391 input prefixes.
The canonical v30 payload SHA-256 is
`921025a7f9c1246c5f10e97f2b987ca4971462241b2d05ebbc525d4aa0736c79`;
the v29 predecessor is
`f2dc3ff6cc8befc600cc62de68c9c804f8962c7261b87cad3c94ea7744bb7df6`.

The 48 year/label records change from planned to observed-saved because the
public v1 attempt saved clean and no-op evidence for all 48 prompts. Its
first family also completed interventions; the four-family v1 result remains
incomplete. The separately frozen v2 continuation repeated the other 36
prompts and completed its own intervention schedule. The v1 and v2 saved
records, direct-process receipts and scientific limits are retained at
`results/distilgpt2-year-label-control-v1-20261029/` and
`results/distilgpt2-year-label-control-v2-20261029/` respectively.

The 36 full-top target records change to observed-saved because the public
baseline screen retained one full-logit record for each target. The screen was
ineligible; no edit or locality forward was run in that attempt. Its 12
full-top locality records remain planned. Evidence is retained at
`results/targeted-edit-full-top-cpu-v1-20261029/screen/attempt/` and
`experiments/targeted_edit_full_top_screen_manifest_v1.json`.

The status overlay validates the exact planned IDs, text, token rows and
candidate groups before changing them, retains other records unchanged, and
rejects any changed input prefix. `tools/audit_exposure_v30_evidence.py`
checks the pinned public case files, manifests, journals and saved raw-segment
hashes without a model. Its retained report is
`experiments/pretrained_exposure_v30_evidence_audit.json`. Process and owner
records are cooperative observations, not authenticated execution or proof
of complete exposure history. The v1 private path-bearing stderr is neither
published nor read by this audit. The old sealed split and outside/pretraining
exposure for year/label cases remain unknown.

Other inherited planned labels can also become historical. In particular,
the 128 two-attractor inputs have a separately published completed baseline at
`results/public-two-attractor-agreement-v1-20261029/` (merge
`e82736f003e7e14e9c0470acfb606c6d0c3ef2f1`), supported by cooperative parent
and saved-data reports. Their v29 labels remain unchanged here pending a
separate evidence-bound reconciliation. The agreement-subject owner also
reports its 80 already-reserved inputs have run; that result was still under
independent review when this update was prepared. Neither observation adds a
new input prefix.

From a repository checkout containing the two public result directories:

```sh
python -B tools/audit_exposure_v30_evidence.py \
  --year-cases experiments/distilgpt2_year_label_control_v1.json \
  --year-v1 results/distilgpt2-year-label-control-v1-20261029 \
  --year-v2 results/distilgpt2-year-label-control-v2-20261029 \
  --full-top-cases experiments/targeted_edit_full_top_cases_v1.json \
  --full-top-manifest experiments/targeted_edit_full_top_screen_manifest_v1.json \
  --full-top-attempt results/targeted-edit-full-top-cpu-v1-20261029/screen/attempt
```
