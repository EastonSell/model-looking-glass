# Exact multisuffix execution source

The CPU attempt records local Git commit
`42519c13a5b33ce3601b89ba65d67b07ebd2f78d`. The adjacent Git bundle preserves
that commit and its two local ancestors; its SHA-256 is
`c4a15c5e60bfde3a1f1e98ec5c5cf1c96f69739c8a2f46c6984227b08a3b19be`.
The bundle requires public ancestor `2a492669b2ebb16d2612366fbefdf80c4265faca`.

In a clone containing that ancestor, verify and import the Git objects with
`git bundle verify audits/bert_multisuffix_agreement_v1/executed-source-42519.bundle`
and `git bundle unbundle audits/bert_multisuffix_agreement_v1/executed-source-42519.bundle`.
Then `git cat-file -t 42519c13a5b33ce3601b89ba65d67b07ebd2f78d` should report
`commit`; a detached worktree at that commit can provide the exact source
required by the saved-evidence reader. Importing the objects does not change a
branch. The source snapshot establishes recorded bytes, not proof of model
execution or complete process-tree custody.
