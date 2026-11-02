# Independent saved-byte replay

From the repository root, with the two audit Python files in this directory adjacent, run with a standard-library Python interpreter and a fresh output path:

```text
python -B results/public-known-injection-v1-20261029/audit/recompute_public_known_injection_v1.py --attempt results/public-known-injection-v1-20261029 --cases experiments/public_known_injection_cases_v1.json --result results/public-known-injection-v1-20261029/result.json --output <fresh-report.json>
```

The reader loads no model. It checks the two receipt-bound file inventories, journals and raw tensor hashes, ordered hooks, injection and patch landing, unchanged bits, null/self/repeat/upstream/true-site controls, discovery freeze and all held-out per-case and aggregate metrics. Its expected case, pre-run, receipt, freeze and result digests are fixed in the source. The published report is `audit/independent-report.json` (SHA-256 `4d02da026c1b2a9e12a94f61303a8b1cb102820c9774ac376b4cb080d526d4c9`). Floating-point library differences can affect exact output on another host; the original result and both receipt-bound raw phases remain available for inspection.

The audit verifies saved-byte and cooperative direct-receipt consistency, not authenticated execution, unknown model exposure or protected-split disjointness. The source and pre-run pins identify the intended program; the receipt alone does not independently prove which instructions executed.
