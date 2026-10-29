# Independent saved-output audit for the projected edit

This model-free reader checks the closed, quiescent public development and evaluation records for the fixed projected CPU study. It requires independently obtained SHA-256 pins for the manifest, candidate and both phase records. The public case document and source/checkpoint identities are fixed by the reader. Its bounded snapshots reject missing, altered, extra or redirected artifacts, including a failure marker beside a complete-looking phase.

It verifies the recorded parent/child and reaping identities, the matching runtime and public-inventory attestations, the six per-condition restoration and numerical-mode attestations, the 78 ordered evaluation rows and their hashes, the saved discovery and all 18 calibration-token shifts. It independently decodes the float32 logits, recalculates each target and locality metric, and applies the frozen eligibility and decision rules. A failed or incomplete record stops the audit before a numerical decision is emitted. It compares the recorded control digests and dose ratios and checks the recorded planned dose against 0.1% of the recorded weight norm. It does not independently reconstruct the gradient, SVD, projection, control factors or actual applied updates; those have a separate development review.

Run from a checkout containing this module, with the committed manifest, fixed public case document and the two closed phase directories available as ordinary files. Supply the independently checked hashes rather than accepting them from those same directories. The command writes one canonical JSON object to stdout and exits nonzero with an `incomplete` object when verification fails; save stdout to a new file outside the evidence directories if a persistent report is needed.

```text
python -m looking_glass.targeted_edit_projected_independent_audit \
  --manifest MANIFEST.json --cases PUBLIC_CASES.json \
  --development DEVELOPMENT_DIRECTORY --evaluation EVALUATION_DIRECTORY \
  --manifest-sha256 SHA256 --candidate-sha256 SHA256 \
  --development-sha256 SHA256 --evaluation-sha256 SHA256
```

For the recorded attempt, the manifest, candidate, development phase and evaluation phase hashes are respectively `813149fca657f015cf1c70821f4bcd7d5a154e89f84369ca56af25973aff8961`, `2764be1bbcc0525f834845e1e6178151159eea4f063ef9967c69c06acca120b7`, `1e9826de2117c51de043e6a5bccb5b181df33a36002a488c19781ed11877fdc1`, and `3a91f9e5c1783b5e94b9dd0b18ff8059894aa84962e10315c6ef3840960d630a`.

The reader does not establish executed-code authenticity, real process lifetimes, source or weight custody, Git ancestry, remote publication timing, prior exposure, or concurrent-file safety. Recorded source, process, weight and cleanup hashes are cooperative attestations; the actual original and edited weight bytes were not retained. Use separate provenance and development-math reviews for those boundaries.

The four public target cases include two direct capital prompts and two government-location or seat prompts. This recorded protocol deviation matters even if numerical criteria are met; the cases are not four strictly equivalent capital paraphrases. The public, non-blind examples do not establish general capability preservation.
