# Screened spatial ViT dose response: clean screen

The [prospective protocol](../../docs/VISION_COCO_SPATIAL_DOSE_V6.md),
[runtime clarification](../../docs/VISION_COCO_SPATIAL_DOSE_V6_RUNTIME_NOTE.md)
and [fixed 64-image manifest](../../experiments/vision_coco_spatial_dose_v6.json)
were public before the clean screen. The CPU screen recognized 31/64 images:
7/24 cat targets, 1/8 cat donors, 17/24 dog targets and 6/8 dog donors.
The fixed rule selected four pairs with eight distinct targets and four
distinct donors. The sole recognized cat donor is reused in all four pairs;
the three selected dog donors are reused two, one and one times.

The full 64-vector screen and its independently reviewed saved-byte audit are
retained here. Screening makes the selected pairs model-conditioned. Their
freshness is relative to documented local exposures; exact ID/JPEG/pixel
exclusion does not prove independence from near-duplicates, unknown previous
exposures or pretraining. Reused donors create dependence. The intervention
outcome is pending and will include all nine matched conditions per side,
controls, exclusions and negative results.
