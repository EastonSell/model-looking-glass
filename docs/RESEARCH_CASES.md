# Versioned research input fixtures

Status: input fixtures and integrity checks, not experimental results or a complete causal protocol. These fixtures do not alter the [frozen initial protocol](../PROTOCOL.md) or its [results](../RESULTS.md).

## Synthetic injection inputs v1

`synthetic-injection-inputs-v1` contains 8 discovery and 16 evaluation cases. Each case has a stable ID, a split label, and eight finite numbers, intended as one model input of shape `[1, 8]` with **float32** dtype. The literals are exactly representable in float32. IDs identify cases separately from their numeric content; `synthetic_content_sha256` is independent of the ID and split.

The numerical inputs are artificial controls. They do not establish that a pretrained model performs any particular computation. A future experiment must specify its model and weights, seed, injected change, discovery procedure, negative and baseline controls, predictions, intervention, metric, and evaluation rule separately. A location showing a difference is not, by itself, a causal explanation.

The two sets are intended to support a development/evaluation division: use discovery inputs to select candidate locations and predictions, freeze those choices, and evaluate them on the evaluation cases. The full evaluation set is **public in this repository**, not a protected or blind set. Iterative inspection or tuning against these cases can contaminate it; after such tuning, a fresh separately controlled evaluation set is needed to assess generalization. Recording the split and digest does not itself enforce a freeze or prevent contamination.

The values were generated once under CPython 3.11.15 with separate fixed `random.Random` instances seeded with `2026102801` for discovery and `2026102802` for evaluation. For each case, the procedure drew eight `randint(-16, 16)` values and divided each by 16. The checked-in literals and pinned digest, rather than a runtime RNG implementation, define v1. These are pseudorandom draws, not evidence of statistically independent datasets or representative real-world inputs. Seeds for model initialization and experiment selection are separate and must be recorded by each protocol.

For custom test suites, construct `SyntheticCase` records with distinct IDs and validate both splits. Custom records are **not** the frozen v1 input set; identify and version them separately and save their content digest. `synthetic_suite_sha256(custom_cases)` uses the v1 format identifier but returns a distinct digest; it does not certify that custom cases are the frozen set. If the content or split of the published fixture changes, create a new version and retain v1.

## Pretrained smoke inputs v1

`pretrained-smoke-inputs-v1` contains six short human-readable prompts, with stable IDs and no expected answer labels. It is a separate smoke-only suite for exercising tokenization and model execution. These public prompts are not a held-out capability benchmark and may have appeared in model pretraining. No claim about a model's reasoning, correctness, or intervention response follows from running them. Record the checkpoint and tokenizer revision, tokenization settings and resulting token IDs with any run; text hashes alone do not fix tokenization. The model adapter must still enforce its token limit.

## Validation and hashing

`validate_case_sets()` checks each suite and rejects reused IDs across them. `validate_synthetic_cases()` requires both split labels, unique IDs, immutable length-eight tuples of finite values, and no duplicate inputs anywhere in either split. Duplicate comparisons happen **after conversion to IEEE 754 float32**, with signed zeros treated as equal; this conservatively rejects distinct Python numbers that collapse to the same model input. `validate_pretrained_smoke_cases()` rejects empty or overlong texts, duplicate IDs, and exact duplicate texts. These checks cannot detect semantic similarity, related model-internal representations, prior training exposure, or human selection based on evaluation results.

These content hashes identify the prescribed float32 model inputs, not arbitrary float64 executions. `float32_values(case.values)` exposes the same conversion used for hashing; pass those values to the model as float32. A hash computed for rounded float32 values must not be used as the identity of a float64 run. For v1 the stored numeric values already have exact float32 representations. The hashes preserve the sign of zero because computations can distinguish it; the disjointness check conservatively treats positive and negative zero as equal.

Hash inputs are UTF-8 bytes of JSON serialized with Python's `json.dumps(..., sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)`. The numeric content object is `{"kind":"numeric-vector-v1","values":[...]}`, after float32 conversion, preserving signed zero. The text content object is `{"kind":"prompt-text-v1","text":...}` with the exact text. A suite object has `kind` equal to its version and `cases` as the ordered array of records. Synthetic records contain `case_id`, `split`, and converted `values`; smoke records contain `case_id` and exact `text`. Record order is part of the suite identity. This describes a specific Python serialization, not a claim that an arbitrary JSON serializer emits identical bytes.

Pinned published suite SHA-256 values:

| Suite | SHA-256 |
| --- | --- |
| `synthetic-injection-inputs-v1` | `883b28112ccee7605125959a4c8209c9e20bb438b73938b323e3fc80994f1e8e` |
| `pretrained-smoke-inputs-v1` | `9ce08d4a321c9e3e1cd6c165a35ec290986a9548bb960804573733f46e89dd4f` |

Example usage, after installing the project's PyTorch dependency:

```python
import torch
from looking_glass.research_cases import (
    DISCOVERY_CASES, HELDOUT_CASES, PRETRAINED_SMOKE_CASES,
    float32_values, synthetic_content_sha256, synthetic_suite_sha256, pretrained_smoke_sha256,
    validate_case_sets,
)

validate_case_sets()
case = DISCOVERY_CASES[0]
x = torch.tensor([float32_values(case.values)], dtype=torch.float32, device='cpu')  # [1, 8]
print(case.case_id, synthetic_content_sha256(case.values), synthetic_suite_sha256())
print(HELDOUT_CASES[0].case_id)  # reserve evaluation until choices are frozen
print(PRETRAINED_SMOKE_CASES[0].text, pretrained_smoke_sha256())
```

Hashing and validation require only the Python standard library; running a neural model requires its own dependencies. To check the fixtures without a model download, run `python -m unittest discover -s tests -p test_research_cases.py -v`.
