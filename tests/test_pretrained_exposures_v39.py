"""Public agreement-head reservation checks; no model or private input."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from looking_glass import pretrained_exposures_v38 as prior
from looking_glass import pretrained_exposures_v39 as v39


def predecessor(records=None):
    return {"records": records or [], "version": "fabricated", "update": {}, "predecessor": {}}


def test_case_binding_and_append_only(monkeypatch):
    raw = (Path(__file__).resolve().parents[1] / v39.CASE_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == v39.CASE_SHA256
    manifest = json.loads(raw)
    expected = [(f"{i:02d}", case["text"], tuple(case["token_ids"]))
                for i, case in enumerate(manifest["cases"], 1)]
    assert list(v39.AGREEMENT_HEAD_CASES) == expected
    old = {"record_id": "old", "token_ids": [50256, 1], "category": "planned"}
    source = predecessor([old])
    original = copy.deepcopy(source)
    monkeypatch.setattr(v39, "_prefixes", lambda _records: range(v39.FORBIDDEN_PREFIX_COUNT))
    payload = v39._build_payload(source)
    assert source == original and payload["records"][0] == old
    assert payload["update"] == {"planned_agreement_head_v2_rows": 16}
    assert len(payload["records"]) == 17
    for record, (ident, text, row) in zip(payload["records"][1:], expected, strict=True):
        assert record["record_id"] == "planned-agreement-head-v2-" + ident
        assert record["category"] == "planned" and record["token_ids"] == list(row)
        assert record["public_text"] == text and record["candidate_token_ids"] == [318, 389, 373, 547]


@pytest.mark.parametrize("kind", ["equal", "shorter", "longer", "candidate", "grouped", "active", "scored"])
def test_known_prefix_relations_are_rejected(kind, monkeypatch):
    row = list(v39.AGREEMENT_HEAD_CASES[0][2])
    foreign = {"record_id": "foreign", "category": "planned"}
    if kind == "equal":
        foreign["token_ids"] = row
    elif kind == "shorter":
        foreign["token_ids"] = row[:-1]
    elif kind == "longer":
        foreign["token_ids"] = row + [318, 123]
    elif kind == "candidate":
        foreign.update(token_ids=row[:-1], candidate_token_ids=[row[-1]])
    elif kind == "grouped":
        foreign.update(token_ids=row[:-1], candidate_token_groups={"group": [row[-1]]})
    elif kind == "active":
        foreign.update(token_ids=[50256] + row, attention_mask=[0] + [1] * len(row))
    else:
        foreign.update(token_ids=[50256, 2], scored_prefix_token_ids=row[:-1],
                       scored_candidate_token_ids=[row[-1]])
    monkeypatch.setattr(v39, "_prefixes", lambda _records: range(v39.FORBIDDEN_PREFIX_COUNT))
    with pytest.raises(ValueError, match="known prefix"):
        v39._build_payload(predecessor([foreign]))


def test_distinct_proposals_must_not_share_a_complete_prefix(monkeypatch):
    cases = list(v39.AGREEMENT_HEAD_CASES)
    cases[1] = ("other", "other", cases[0][2] + (318,))
    monkeypatch.setattr(v39, "AGREEMENT_HEAD_CASES", tuple(cases))
    monkeypatch.setattr(v39, "_prefixes", lambda _records: range(v39.FORBIDDEN_PREFIX_COUNT))
    with pytest.raises(ValueError, match="distinct agreement-head cases"):
        v39._build_payload(predecessor())


def test_v39_reconstructs_the_pinned_v38_and_adds_only_planned_rows(monkeypatch):
    before = prior.inventory_payload()
    assert hashlib.sha256(prior._canonical(before)).hexdigest() == v39.PREDECESSOR_SHA256
    monkeypatch.setattr(prior, "inventory_payload", lambda: before)
    after = v39.inventory_payload()
    assert v39.inventory_digest() == v39.INVENTORY_SHA256
    assert after["records"][:len(before["records"])] == before["records"]
    assert len(after["records"]) == len(before["records"]) + 16
    assert all(row["category"] == "planned" for row in after["records"][-16:])
