# Code map

Start here when changing the code. To use the app, follow the [app guide](../docs/LOCAL_APP.md).

## Browser workbench

| Change | Start with |
|---|---|
| HTTP routes and session state | [app.py](app.py) |
| Local model loading and inference | [app_runtime.py](app_runtime.py) |
| Captures, coordinates and save/reopen | [app_storage.py](app_storage.py) |
| Compare a reference vector | [app_reference.py](app_reference.py) |
| Lens read-out (logit lens) | [app_lens.py](app_lens.py), [app_static/app_lens.js](app_static/app_lens.js) |
| Field display of the lens | [app_static/app_field.js](app_static/app_field.js) |
| Jacobian lens: estimator, file format | [jacobian_lens.py](jacobian_lens.py) |
| Jacobian lens: fit session, routes, terminal command | [app_jlens.py](app_jlens.py), [jlens_cli.py](jlens_cli.py), [jlens_corpus.txt](jlens_corpus.txt) |
| Controlled activation edits | [app_interventions.py](app_interventions.py) |
| Browse local models or download a catalog model | [app_discovery.py](app_discovery.py), [app_downloads.py](app_downloads.py) |
| GPU acceptance check | [app_selftest.py](app_selftest.py) |
| Layout, interaction and captured-site scene | [app_static/](app_static/) |

Install and restart commands live at the repository root: [install.py](../install.py) and [start.py](../start.py). App tests are in [tests/](../tests/), named `test_app_*`.

## Shared capture tools

- Model adapters: [GPT-2](transformer_trace.py), [Llama](llama_trace.py), [DistilBERT](distilbert_trace.py), [T5](flan_t5_trace.py), [ViT](vit_trace.py).
- Recorded data: [trace_schema.py](trace_schema.py).
- Temporary edits and model access: [interventions.py](interventions.py), [intervention_plan.py](intervention_plan.py), [model_ownership.py](model_ownership.py).
- Hook sessions: [probes.py](probes.py).

The app supports specific model configurations; a module's presence is not a compatibility promise. Reference pages for these modules are in [docs/](../docs/README.md).

## Tests

Everything in [tests/](../tests/) runs in CI on each pull request:

```sh
python -m pytest tests -q
node tests/app_ui_contract.cjs && node tests/app_scene_contract.cjs && node tests/app_lens_contract.cjs && node tests/app_field_contract.cjs
```

## Research archive

The research studies that used to live in this package are on the [`research-archive` branch](https://github.com/EastonSell/model-looking-glass/tree/research-archive). See the [documentation index](../docs/README.md#research-archive).
