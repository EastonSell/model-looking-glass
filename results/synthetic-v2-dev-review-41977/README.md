# Development-only intervention review inputs

An independent reviewer reported using these two fabricated CPU float32
vectors with untrained model seed 41977 for development checks of the v2
intervention mechanics. This record was written **after** those checks and is
an input-exclusion inventory, not precommitted evidence or a v2 study result.
The reviewer reported that the positive vector was used for a 16-forward
same-site/step-map check, and that both vectors were used for 72 separately
bounded integrity-capture forwards across nine sites and two strengths. Four
additional forwards on the positive vector checked one perturbed sparse trace
and three same-forward plans with metadata and source/trace agreement. The
reviewer then used the same two vectors for 576 further forwards across all
nine target locations, both strengths and polarities, including forward,
reverse, null, same-forward self-patches, upstream and post-hook controls.
Another 324 forwards, including 72 trace baselines, checked 252 post-patch
captures over all 36 location/strength/polarity conditions against the intended
paired source. An additional 254 forwards checked three-step self-patches,
two-step nulls, metadata and two expected failure cases. The reviewer reported
two further forwards checking patch delivery of signed zeros and minimum
subnormals. The reviewer reported 1,248 ad hoc CPU root forwards in total at
the time of this inventory.
These checks are not the proposed recorded-run schedule or held-out evidence.

The first coordinates, ±2.0, lie outside the proposed v2 input support; retain
and explicitly exclude the full tuples from both v2 splits anyway. The exact
canonical UTF-8 bytes in [`input_exclusions.json`](input_exclusions.json) have
SHA-256 `e64290d259acbeb9dd845263dc74241f9a5712d276c4dc70714290398709c28c`.
Float32 big-endian bit patterns are:

- Negative: `c0000000 be800000 3f000000 3e000000 bf400000 3e4ccccd be99999a 3ecccccd`
- Positive: `40000000 be800000 3f000000 3e000000 bf400000 3e4ccccd be99999a 3ecccccd`

This inventory is separate from the sealed
[202-forward pilot](../synthetic-v2-resource-pilot-20261028), whose exclusion
digest and files remain unchanged. No v2 split or site schedule was generated.
