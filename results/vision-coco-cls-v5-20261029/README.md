# ViT CLS versus spatial comparison: screened sample

The [fixed protocol](../../docs/VISION_COCO_CLS_COMPARE_V5.md) and
[64-image manifest](../../experiments/vision_coco_cls_compare_v5.json) were
published before this clean screen. The CPU screen used the pinned ViT model
at source commit `f5435d6f6636fbcc0697cfd8b0d4e6d766f20024` and retained
all 64 full logit vectors in `screen-evidence.zip`. It recognized 31 images:
2/24 cat targets, 4/8 cat donors, 19/24 dog targets and 6/8 dog donors. The
fixed selection yielded two pairs and four distinct targets. One recognized
cat donor and one recognized dog donor are each reused in both pairs.

These pairs are selected using the model's outputs, and the small sample is
conditional on that screen; they are fresh only relative to the documented
local exposures. The one-per-species donor reuse makes pair outcomes
dependent. Near-duplicates, unrecorded prior exposure, pretraining overlap
and the upstream authenticity of mirrored images are not established by the
recorded hashes. The inherited `heldout` field means this conditional
evaluation phase, not model-blind or pretraining-disjoint selection.

The complete screen, deterministic selected pairs, direct-controller
observation and independent saved-byte audit are preserved in this directory.
The audit checks consistency of the saved bytes and decision rules; it does
not attest execution, process-tree cleanup, publication timing or training
history. The intervention outcome is pending and will be reported with all
controls and exclusions. Two selected pairs are sufficient to run the fixed
comparison: if both remain dose-matchable, both must pass for a positive
aggregate; fewer than two evaluable pairs is inconclusive.
