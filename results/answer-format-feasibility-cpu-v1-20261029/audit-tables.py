"""Check every public CSV cell against frozen cases and independent scores."""
import csv
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


work = Path("work")
case_bytes = (work / "answer_format_cases_proposal_v2.json").read_bytes()
score_bytes = (work / "reviews/answer-format-full-independent-308c30.json").read_bytes()
assert digest(case_bytes) == "cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a"
assert digest(score_bytes) == "4a2e17cfab8ae990c983dd8e94ae5fb496efef4781f5b57536d4f90653e6a833"
cases = {row["case_id"]: row for row in json.loads(case_bytes)["cases"]}
report = json.loads(score_bytes)
labels = ("factual", "counterfactual", "madrid", "stockholm")


def load(name):
    path = work / "answer_format_full_tables" / name
    raw = path.read_bytes()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows, digest(raw)


def same(actual, expected, context):
    if isinstance(expected, bool):
        assert actual == str(expected), (context, actual, expected)
    elif isinstance(expected, int):
        assert int(actual) == expected, (context, actual, expected)
    elif isinstance(expected, float):
        assert float(actual) == expected, (context, actual, expected)
    elif expected is None:
        assert actual == "", (context, actual, expected)
    else:
        assert actual == expected, (context, actual, expected)


rows, case_sha = load("cases.csv")
assert len(rows) == len(cases) == 64
observations = {o["case_id"]: (stage, o) for stage in ("development", "heldout")
                for o in report[stage]["observations"]}
assert len(observations) == 64 and {r["case_id"] for r in rows} == set(cases) == set(observations)
for row in rows:
    case, (stage, obs) = cases[row["case_id"]], observations[row["case_id"]]
    expected = {"stage": stage, "canonical_index": obs["canonical_index"],
                "execution_index": obs["execution_index"], "case_id": obs["case_id"],
                "fact": obs["fact_id"], "family": obs["family"], "prefix": obs["prefix"],
                "prompt": case["prompt"], "unique_top_id": obs["unique_top_id"],
                "top_token": obs["representative_decoded_token"], "top_tie_count": obs["top_tie_count"],
                "top_gap": obs["top_gap"], "factual_top": obs["factual_top"],
                "factual_answer": case["labels"]["factual"]["text"], "row_sha256": obs["row_sha256"]}
    for label in labels:
        for field in ("token_id", "logit", "logp"):
            expected[f"{label}_{field}"] = obs["scores"][label][field]
    assert set(row) == set(expected), (row["case_id"], set(row) ^ set(expected))
    for key, value in expected.items():
        same(row[key], value, (row["case_id"], key))

pairs, pair_sha = load("prefix-pairs.csv")
expected_pairs = {(stage, p["reference_case_id"], p["comparison_case_id"]): p
                  for stage in ("development", "heldout") for p in report[stage]["pairs"]}
assert len(pairs) == len(expected_pairs) == 96
seen = set()
for row in pairs:
    key = (row["stage"], row["reference_case_id"], row["comparison_case_id"])
    assert key not in seen and key in expected_pairs
    seen.add(key)
    p = expected_pairs[key]
    expected = {field: p[field] for field in ("fact_id", "family", "reference", "comparison",
                "reference_case_id", "comparison_case_id", "top_relation", "kl", "total_variation")}
    expected["stage"] = key[0]
    expected.update({f"{label}_logp_delta": p["logp_delta"][label] for label in labels})
    assert set(row) == set(expected)
    for field, value in expected.items(): same(row[field], value, (key, field))

contrasts, contrast_sha = load("example-contrasts.csv")
expected_contrasts = {(stage, p["fact_id"], p["family"]): p
                      for stage in ("development", "heldout") for p in report[stage]["contrasts"]}
assert len(contrasts) == len(expected_contrasts) == 16
seen = set()
for row in contrasts:
    key = (row["stage"], row["fact_id"], row["family"])
    assert key not in seen and key in expected_contrasts
    seen.add(key)
    expected = {"stage": key[0], **expected_contrasts[key]}
    assert set(row) == set(expected)
    for field, value in expected.items(): same(row[field], value, (key, field))

result = {"status": "passed", "scope": "all CSV cells compared with frozen cases and independent numerical report",
          "cases_sha256": case_sha, "prefix_pairs_sha256": pair_sha, "contrasts_sha256": contrast_sha,
          "rows": len(rows), "pairs": len(pairs), "contrasts": len(contrasts),
          "model_forward_asserted": False}
data = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
with (work / "reviews/answer-format-full-table-audit-308c30.json").open("xb") as output:
    output.write(data)
print(digest(data), result)
