# Fresh evaluation screen

After the fixed [discovery evidence](discovery-README.md) was published, the
model scored all 64 fresh inventory images in the preregistered order. The
complete [screen result](screen-result.json), [131-file evidence archive](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-occlusion-v1-20261029/screen-evidence.zip),
[direct observer](screen-observer.json), and [independent saved-byte audit](screen-independent-audit.json)
are retained here. The observer reports exit 0, reaped, without timeout or
forced stop; it is a cooperative direct-process record.

| Species | Candidates | Paired geometry | Recognized | Both | Selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cat | 32 | 17 | 10 | 6 | 4 |
| Dog | 32 | 15 | 26 | 12 | 4 |

The first four recognized and geometry-qualified cats were 122196, 480720,
404350 and 151497; the corresponding dogs were 135116, 426342, 470798 and
468792. Selection used only fixed annotation geometry and clean recognition;
the sample is therefore conditional on this model-based screen. All 64 logits,
including unrecognized and geometry-ineligible cases, remain in the archive.
Exact ID and byte disjointness from prior recorded images does not establish
pretraining or near-duplicate independence.

The independent screen replay source was published before screening in
[this archive](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-occlusion-v1-20261029/independent-screen-replay.zip). The separately reviewed
[discovery replay bundle](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-occlusion-v1-20261029/discovery-independent-replay.zip) contains its
source, required helpers and the discovery audit. These assess saved-bit and
arithmetic consistency, not independent execution or chronology attestation.
