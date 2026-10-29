# Occlusion CPU storage amendment

This prospective execution amendment is fixed after annotation-only preparation
and a forward-forbidden checkpoint preflight, and before the four discovery
model forwards. It changes no image, selection, comparison, threshold, control,
or reporting rule in the [study protocol](VISION_COCO_OCCLUSION_V1.md).

The checkpoint and source checkout remain on C: and are read only by the
study. Evidence, observer logs, temporary files, Python bytecode prefix and
explicit Hugging Face, XDG, Torch, TorchInductor and Triton cache directories
are placed on E:. Network model downloads remain disabled. The CPU-only owner
requires the output and these cache/temporary paths to be on the same volume,
separate from the source volume. It checks at least 1.5 GiB free on the source
volume and 1 GiB on the output volume immediately before a phase; before each
model forward it stops if either volume has fallen below 1 GiB. This replaces
the original owner's 2 GiB source-volume prephase check, without changing a
running process. The lower floor reflects the absence of planned study writes
to C:, not a reservation: unrelated system or user processes can consume space
between checks. Any failure, timeout, or forced stop remains incomplete and is
retained, with no automatic retry. The amended owner receives a new forward-
forbidden checkpoint preflight before discovery.
