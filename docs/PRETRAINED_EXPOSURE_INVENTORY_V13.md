# DistilGPT2 development exposure inventory v13

This snapshot records the 32 public copying-head factorial rows reserved by
v12 as `observed-saved`. It is a status update on the reviewed v12 inventory.
It adds no inputs and preserves every other record, including the 13 planned
public projected-edit cases. Prior snapshots remain unchanged.

The canonical inventory is `looking_glass.pretrained_exposures_v13`. Its
predecessor is v12, SHA-256
`af374aaceb708b3b38e06b3955ca975467b048162ee105cbfcc9452a8a600e49`;
v13's canonical payload SHA-256 is `27ffe8d2ab312ccf2a62a07dab835d3cafce7364ca8abadbb1e73ec835e9aca8`.
There are 215 records: 168 observed-saved, 3 observed-reported, 32 planned,
4 reference-only, and 8 synthetic-only. Its 1,709 recorded token prefixes are
identical to v12's; the 32 observed rows were already excluded as reservations.

The public, non-blind CPU factorial evidence is at commit `cab7ce24057692f0611a37e799cfa3b57e842ce5` under
`results/distilgpt2-copy-head-factorial-v1-20261028/`. The manifest identifies
the 32 reserved case IDs, inputs and targets for the pinned DistilGPT2 model.
The result reports 224 submitted and returned actions, seven per case; the
attempt journal contains a start, raw-recorded event and completion for each.
The direct-process observation reports exit code 0, no interruption and no
failure file. It also describes a reported resource-sampler announcement
overlap, without establishing process-time overlap. These records support an
exposure status for the rows; they do not establish blindness or a general
model-behavior conclusion.

| Evidence file | SHA-256 |
| --- | --- |
| `run/manifest.json` | `3fe5f868917b89912b96200ce01029812192c9703c5c39244d6b78485b5605bb` |
| `run/result.json` | `7f62ed7a014978f71c09ed2e5973d1ce38f544eff0f2a1652cc9fd75e2fb125b` |
| `run/attempt.jsonl` | `4e56988a89dddc9aeddcc67c69957e473402140a7a4ed6244cdaa75b5776adfa` |
| `supervisor.json` | `35626526739498d468c9e91d1c4f39e3f79603cd9015fc7fbd386ba3917f086d` |

The case inventory was public before the factorial forward and has SHA-256
`69f8463da5f919fb99aef05aa4401de439248100e99ca90899f8d9f98af90d24`.
The focused tests reconcile the committed manifest, result, journal and
process observation with the frozen rows and each other; they check the raw
file names in the receipt but do not reread all raw tensor files. Hashes
establish consistency of saved bytes, not independent execution attestation.
The process observation is limited to the direct process.

The v12 known-row checks remain in force: a proposed row is rejected when it
is a recorded prefix or extends a recorded complete row. These checks cover
only known history. They do not establish full prior exposure, pretraining or
private held-out disjointness, and do not authorize a study seal. Future public
inputs need a separately reviewed reservation before forwarding.
