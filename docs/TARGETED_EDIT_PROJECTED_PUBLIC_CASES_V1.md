# Projected edit: public case reservation v1

The accompanying `experiments/targeted_edit_projected_public_cases_v1.json`
records four target and nine locality proposals for the previously published
projected-edit protocol. These are **public, non-blind** cases. The target
rows are excluded from fitting this edit, but their exposure during model
pretraining and other unknown history is not known. Four locality cases use
other capitals, four concern other topics, and one concerns another relation
about France. They do not establish broad capability preservation.

The record contains exact prompts, tokenizer IDs, continuation IDs, full
prompt-plus-continuation encodings and the tokenizer-only check. It pins the
offline DistilGPT2 tokenizer asset and its software versions. The operator
reports that no model forward for this projected experiment occurred before
this record; that report and the case publication order are not trusted
timestamps or independent proof of execution.

The known-public comparison checked all 13 complete proposed rows against
all 170 records in exposure inventory v9 in both complete-row prefix
directions, and checked the 78 pairs among the proposals. No relation was
found in those comparisons. That is a check of the stated public snapshot,
not of an exhaustive history, pretraining, or any private split. A successor
inventory must reserve these cases before projected-experiment forwards.

This document and its JSON are a case reservation, **not** an execution
manifest. Before any projected-experiment model forward, separately commit
the exact cases, execution sources, checkpoint asset hashes and applicable
custody checks as required by `TARGETED_EDIT_PROJECTED_CPU_V1.md`. Preserve
the v1 negative result and the separate v2 dose-grid experiment unchanged.
