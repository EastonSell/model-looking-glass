"""Model-free checks on the fixed FLAN weight-change proposal."""

from pathlib import Path
import os
import pytest

import flan_weight_edit_cases as cases


def test_fixed_proposal_has_complete_ordered_pairs_and_locality_strata():
    entries = cases.fixed_texts()
    assert len(entries) == 24
    assert [x["role"] for x in entries[:8]] == [
        "discovery-target", "discovery-donor", "discovery-target", "discovery-donor",
        "calibration-target", "calibration-donor", "calibration-target", "calibration-donor",
    ]
    assert [x["role"] for x in entries[8:16]] == [
        "evaluation-target", "evaluation-donor",
    ] * 4
    assert [x["role"] for x in entries[16:]] == ["locality-sentiment"] * 4 + [
        "locality-target-token", "locality-target-token", "locality-unrelated", "locality-unrelated",
    ]
    assert len({x["id"] for x in entries}) == 24
    assert len({x["text"] for x in entries}) == 24
    for index in (0, 2, 4, 6, 8, 10, 12, 14):
        a, b = entries[index : index + 2]
        assert a["text"].count("wonderful") == 1
        assert b["text"] == a["text"].replace("wonderful", "awful")
        assert (a["expected"], b["expected"]) == ("positive", "negative")


def test_pinned_tokenizer_has_fixed_pair_alignment_and_eos():
    work = os.environ.get("FLAN_WEIGHT_TEST_WORK")
    asset = (Path(work) / "t5-cache/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
             if work else Path(cases.ASSET_DIR))
    if not (asset / "asset-manifest.json").is_file():
        pytest.skip("set FLAN_WEIGHT_TEST_WORK to the existing pinned tokenizer cache")
    tokenizer = cases.load_tokenizer(asset)
    bundle = cases.build_bundle(tokenizer)
    assert bundle["protocol_sha256"] == cases.PROTOCOL_SHA256
    assert len(bundle["cases"]) == 24
    for index in (0, 2, 4, 6, 8, 10, 12, 14):
        a, b = bundle["cases"][index : index + 2]
        assert a["input_ids"][-1] == b["input_ids"][-1] == 1
        assert a["attention_mask"] == b["attention_mask"]
        difference = [i for i, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y]
        assert len(a["input_ids"]) == len(b["input_ids"])
        assert difference == [a["adjective_index"]] == [b["adjective_index"]]
        assert a["input_ids"][difference[0]] == 1627
        assert b["input_ids"][difference[0]] == 16242
    assert all(len(row["input_ids"]) <= 64 for row in bundle["cases"])
    assert all(len(row["expected_ids"]) <= 4 for row in bundle["cases"])


def test_bidirectional_prefix_check_excludes_only_one_terminal_eos():
    assert cases.conflicts([10, 11, 1], [10, 11, 12, 1])
    assert cases.conflicts([10, 11, 12, 1], [10, 11, 1])
    assert not cases.conflicts([10, 11, 1], [10, 12, 1])
