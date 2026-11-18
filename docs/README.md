# Documentation

Start with the [app guide](LOCAL_APP.md): install, load a model, capture, read the lens, and try edits.

Changing the code? See the [code map](../looking_glass/README.md) and the [contribution guide](../CONTRIBUTING.md).

## Reference

These describe the Python modules the app is built on.

| Need | Read |
|---|---|
| What a recorded capture contains | [Trace schema](TRACE_SCHEMA.md) |
| GPT-2 capture | [Transformer trace](TRANSFORMER_TRACE.md) |
| DistilBERT capture | [DistilBERT trace](DISTILBERT_TRACE.md) |
| T5 capture | [T5 trace](FLAN_T5_TRACE.md) |
| ViT capture | [ViT trace](VIT_TRACE.md) |
| Temporary activation edits | [Intervention primitives](INTERVENTIONS.md), [ordered plans](INTERVENTION_PLANS.md) |
| Comparing two captures | [Localization](LOCALIZATION.md), [stored-float arithmetic](AGREEMENT_NUMERIC.md) |
| Capture budgets and dispatch | [Capture dispatcher](CAPTURE_DISPATCHER.md) |

Some of these pages link to study records. Those links point at the archive below.

## Research archive

Until November 2026 this repository also held the project's research studies: about 430 study modules, their frozen protocols, raw results and tests. They were moved out of `main` so the app is easy to find and install. Nothing was deleted from history.

- Browse them on the [`research-archive` branch](https://github.com/EastonSell/model-looking-glass/tree/research-archive).
- Start with its [findings index](https://github.com/EastonSell/model-looking-glass/blob/research-archive/docs/PROJECT_STATUS.md) and [initial results](https://github.com/EastonSell/model-looking-glass/blob/research-archive/RESULTS.md).
- The studies were written against the dependency versions of their time. Some no longer run under the versions pinned here.
