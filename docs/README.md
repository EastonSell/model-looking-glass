# Documentation

Use the short path below. Most remaining files in this directory are **frozen study contracts**, not an onboarding checklist or supported API catalog.

## Start here

1. [Repository quick start](../README.md#quick-start): direct capture without downloading a model.
2. [Known-corruption → localization → restoration tutorial](GPT2_TRACE_WORKFLOW.md): one end-to-end CPU workflow using an explicitly cached checkpoint.
3. [Findings, support scope and limits](PROJECT_STATUS.md): what the research actually supports, with representative positive and negative evidence.
4. [Roadmap and execution efficiency](RESEARCH_ROADMAP.md): the single living plan, not a promise of implemented features.

## Reference shelf

| Need | Read |
|---|---|
| Observe and compare | [Trace schema](TRACE_SCHEMA.md), [bounded store](TRACE_STORE.md), [localization](LOCALIZATION.md), [transformer comparison](TRANSFORMER_LOCALIZATION.md) |
| Apply controlled changes | [Intervention primitives](INTERVENTIONS.md), [ordered plans](INTERVENTION_PLANS.md), [causal metrics](CAUSAL_METRICS.md), [behavioral evaluation](BEHAVIORAL_EDIT_EVALUATION.md) |
| Load a pinned model | [Pretrained loader and asset checks](PRETRAINED_MODEL.md) |
| GPT-2 | [Trace reference](TRANSFORMER_TRACE.md), [head slices](GPT2_HEAD_SLICE.md) |
| DistilBERT | [Trace reference](DISTILBERT_TRACE.md), [intervention workflow](DISTILBERT_INTERVENTION_WORKFLOW.md) |
| T5 | [Trace reference](FLAN_T5_TRACE.md), [comparison](FLAN_T5_TRACE_COMPARISON.md), [intervention workflow](FLAN_T5_INTERVENTION_WORKFLOW.md) |
| ViT | [Trace reference](VIT_TRACE.md), [comparison](VIT_TRACE_COMPARISON.md), [donor patches](VIT_DONOR_PATCH.md) |
| Interpret measurements | [Mathematics](MATHEMATICS.md), [denormal counterexample](GRADIENT_DENORMAL_V1.md), [stored-float arithmetic](AGREEMENT_NUMERIC.md), [factorial metrics](AGREEMENT_METRICS.md) |
| Design a study | [Methods synthesis](CAUSAL_INTERVENTION_METHODS_NOTE_20261029.md), [literature and experimental choices](CAUSAL_LITERATURE.md), [original protocol](../PROTOCOL.md) |
| Check validation | [Actual CPU validation and limitations](MAINTENANCE_2026-10-29.md), [experimental opcode boundary](EXPERIMENTAL_OPCODE_TESTS.md) |
| Investigate service prototypes | [Consolidated experimental infrastructure](EXPERIMENTAL_INFRASTRUCTURE.md); not a supported general GPU service |

## Historical research, without hundreds of competing guides

The [findings map](PROJECT_STATUS.md#research-evidence) links representative study outcomes. Their original reports, negative controls, source-bound protocols, case records and tests remain. A dated protocol is evidence of that study's rules, **not current setup advice**; an exposure inventory records known prior use, not proof that all other inputs are unexposed.

Unpinned historical runner manuals, incremental exposure summaries and original implementation plans have moved out of the active tree. Their exact originals remain at an immutable Git commit, indexed by path, title, hash and size in the [documentation archive](../archives/README.md#historical-documentation). They were not discarded as scientifically false. Use the historical snapshot for its full linked context; do not rewrite provenance hashes to make an old study run against new code.

[Result-payload restoration](../archives/README.md) and [retained CPU fixtures](../archives/CPU_TEST_FIXTURES.md) are separate from documentation archival. A shallow checkout may need the explicitly named historical commit and selected objects before replay.
