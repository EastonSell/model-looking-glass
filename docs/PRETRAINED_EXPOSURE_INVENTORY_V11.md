# DistilGPT2 development exposure inventory v11

This snapshot records the 32 public copying-follow-up rows reserved by v9 as
`observed-saved`. It is a status update on the reviewed v10 inventory. It adds
no inputs and preserves every other record, including v10's 13 planned public
projected-edit cases. The prior snapshots remain unchanged.

The canonical inventory is `looking_glass.pretrained_exposures_v11`. Its
predecessor is v10, SHA-256
`390550772b895b7ec2b1945f2df7683b3eb23ff831de5df2f55f8bb540f87588`;
v11's canonical payload SHA-256 is
`84b1a95a41670798ff55c9f08272890c00a939a74a4cad7ad9b95fbf4bcdaa25`.
There are 183 records: 136 observed-saved, 3 observed-reported, 32 planned,
4 reference-only, and 8 synthetic-only. Its 1,357 recorded token prefixes are
identical to v10's; the 32 observed rows were already excluded as reservations.

The public, non-blind CPU follow-up evidence is at commit
`0712ec0bdb9b1b9490bee48919f86d11dbf10869` under
`results/distilgpt2-copy-head-followup-v2-20261028/`. The manifest identifies
the 32 reserved case IDs, inputs and targets for the pinned DistilGPT2 model.
The result reports 288 submitted and returned actions, nine per case; the
attempt journal contains a start, raw-recorded event and completion for each.
The direct-process observation reports exit code 0, no interruption and no
failure file. The saved records support an exposure status for these rows; they
do not make a blind-study or general model-behavior claim.

| Evidence file | SHA-256 |
| --- | --- |
| `run/manifest.json` | `39758a7d1670e31d0e159a0bf738a9094646efe8b638ffdee6519f28d50385fb` |
| `run/result.json` | `4a6c9251a5752714525836df2c29c9e1a8c8659430720a79da47e5f4f3fc18f1` |
| `run/attempt.jsonl` | `fa28d4921fcf91734c451d2bb3a5e31e27310fd3058d694389f436444398912b` |
| `supervisor.json` | `2ffc6adc91d51b3b6b91db587ad011d11eb72efc8d1eb614d0868fc1cff38ea9` |

The case inventory was public before the follow-up forward and has SHA-256
`3831bfd1a48aaa3fcb67fbdb3cdc10f4422149b1fed90fd5267d3c73525255a8`.
The tests reconcile the committed manifest, result, journal and process
observation with the frozen rows and each other. Hashes establish consistency
of the saved bytes, not independent attestation of execution. The process
observation is limited to the direct process.

The v10 known-row checks remain in force: a proposed row is rejected when it
is a recorded prefix or extends a recorded complete row. These checks cover
only known history. They do not establish full prior exposure, pretraining or
private held-out disjointness, and do not authorize a study seal. Future public
inputs need a separately reviewed reservation before forwarding.
