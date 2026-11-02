# Code map

Start here when changing the code. To use the app, follow the [app guide](../docs/LOCAL_APP.md).

## Browser workbench

| Change | Start with |
|---|---|
| HTTP routes and session state | [app.py](app.py) |
| Local model loading and inference | [app_runtime.py](app_runtime.py) |
| Captures, coordinates and save/reopen | [app_storage.py](app_storage.py) |
| Compare a reference vector | [app_reference.py](app_reference.py) |
| Controlled activation edits | [app_interventions.py](app_interventions.py) |
| Browse local models or download a catalog model | [app_discovery.py](app_discovery.py), [app_downloads.py](app_downloads.py) |
| GPU acceptance check | [app_selftest.py](app_selftest.py) |
| Layout, interaction and captured-site scene | [app_static/](app_static/) |

Install and restart commands live at the repository root: [install.py](../install.py) and [start.py](../start.py). Nearby regression tests are in [tests/](../tests/), usually named `test_app_*`.

## Shared capture tools

- Model adapters: [GPT-2](transformer_trace.py), [Llama](llama_trace.py), [DistilBERT](distilbert_trace.py), [T5](flan_t5_trace.py), [ViT](vit_trace.py).
- Recorded data: [trace_schema.py](trace_schema.py) and [trace_store.py](trace_store.py).
- Temporary edits and model access: [interventions.py](interventions.py), [intervention_plan.py](intervention_plan.py), [model_ownership.py](model_ownership.py).

See the [Python tutorial](../docs/GPT2_TRACE_WORKFLOW.md) for a complete workflow. The app supports specific model configurations; a module's presence is not a compatibility promise.

## Research and history

Many other modules belong to named studies or versioned inventories. An older suffix does **not** mean unused: later studies can import it or verify its exact source bytes. Import paths stay in place for those contracts.

[experiments/](../experiments/), [studies/](../studies/), [scripts/](../scripts/) and [tools/](../tools/) contain research runners and maintenance utilities. [results/](../results/), [evidence/](../evidence/), [audits/](../audits/) and [commitments/](../commitments/) retain their records. [REVIEW.md](../REVIEW.md) is an October 2026 historical review, not the current release checklist.

Use the [findings index](../docs/PROJECT_STATUS.md) and [archive guide](../archives/README.md) instead of reading every study file. Before moving or removing a research family, trace its imports, tests and source-hash references together.
