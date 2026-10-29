# Current synthetic v2 development exclusion union

[`inputs.json`](inputs.json) is canonical compact UTF-8 JSON, sorted by case
ID, containing the two retained
[resource-pilot inputs](../synthetic-v2-resource-pilot-20261028/artifacts/pilot_inputs.json)
and the two separately reported
[intervention-review inputs](../synthetic-v2-dev-review-41977/input_exclusions.json).
Its SHA-256 is
`f518a2a4e7e0aa3e03ebf60170395c5fb49ce0cd2e8acd98a96ba3427b7f3347`.
Its four float32 tuples and IDs are unique, treating signed zeros as equal.
Both future v2 splits must exclude all tuples in this union as well as all v1
inputs. The original pilot record and digest remain unchanged.

This is the inventory known at this revision, not a claim that no other
development inputs will be used. If more are used before preparation, publish
a new complete union and bind its digest before preparing either split; do not
overwrite this record. This file does not contain v2 discovery or held-out
cases and is not evidence of a study result.
