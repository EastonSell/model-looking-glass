# DistilGPT2 development exposure inventory v17

This status-only successor reconciles two completed public CPU studies with
their previously planned rows: 32 pairs/dose inputs and 13 projected-edit
inputs. It retains all 407 v16 input rows and all 3,247 known input prefixes.
The 48 year-transfer rows added by v16 remain planned; this update makes no
claim that they were forwarded.

The canonical module is `looking_glass.pretrained_exposures_v17`. Its v16
predecessor payload SHA-256 is
`275a16d49a386b1de247bff0f9031e92105d1b76f2493352090bf27c37d58843`;
the v17 canonical payload SHA-256 is
`e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a`.
For the 45 reconciled records, the `planned-` ID prefix becomes `observed-`,
category becomes `observed-saved`, and provenance and references point to the
saved evidence. Input IDs, text, targets and the other records are preserved.

The pairs/dose evidence was merged at
`28d7c0321cda85d1139c93b25f8605b874feae96` under
`results/distilgpt2-copy-head-pairs-dose-v1-20261028/`. Its manifest contains
the exact 32 reserved rows; the saved result and journal report 576 returned
actions, 18 per case, and the process observation reports exit 0 with no
interruption or failure file. The manifest, result, journal and process
observation SHA-256 values are respectively
`93ee4d22d0c4d040e50c25804cd36db60492698e601736e04282909e1dcafa35`,
`b382cf60edd79c9b9eccbd992dd1fffe33d11a3c202728a7a44127ef256946cf`,
`2e05be2c6f5cba20f8e62d3f0e03d189cdc5fe4555763c5ce5733a60bbc9a1e3` and
`60423637baf78178b0e4fa23438f41e5f6310423fae4ce7437ee10e3d1508d9d`.

The projected-edit evidence was merged at
`e704cb82ed6078d76ba5ab5bd32378eff106924a` under
`results/targeted-edit-projected-cpu-v1-20261028/evaluation/`. Its fixed
manifest names the exact 13 reserved inputs; the observation journal contains
all 13 in each of six conditions. The saved phase record reports completion,
drain, a reaped owner and supervisor exit 0. The manifest, phase record,
observation journal and child result SHA-256 values are respectively
`813149fca657f015cf1c70821f4bcd7d5a154e89f84369ca56af25973aff8961`,
`3a91f9e5c1783b5e94b9dd0b18ff8059894aa84962e10315c6ef3840960d630a`,
`e84965fae4be92552af676c247ac5ceda08f2a510cfb48f26bac02d72431a439` and
`d5de75fb1f64879c21a8896d5fcf9fd415becf2010e400e67299c921c321d7ca`.

The checks compare the cited saved bytes and identities; they do not
independently attest execution, establish all prior exposure or pretraining,
or prove private held-out disjointness. A known-prefix check alone does not
authorize a study seal.
