"""Fabricated-only tests for the prospective multi-subtoken selector."""

import copy
import gzip
import json

import pytest

import flan_multitoken_selector_v1 as m


def synthetic_provenance():
    return {"protocol_sha256": m.PROTOCOL_SHA256, "candidates_sha256": m.CANDIDATES_SHA256,
            "pilot_sha256": m.PILOT_SHA256, "tokenizer_sha256": m.TOKENIZER_SHA256,
            "asset_manifest_sha256": m.ASSET_MANIFEST_SHA256,
            "prior_descriptors_sha256": m.PRIOR_DESCRIPTORS_SHA256,
            "released_pool_sha256": m.RELEASED_POOL_SHA256,
            "prefix_method": "one-terminal-eos-stripped-bidirectional-prefix",
            "public_scope_sha256": "a" * 64, "custodian_receipt_sha256": "b" * 64,
            "public_scope_description": "fabricated test, no public scan",
            "custodian_scope_description": "fabricated test, no custodian check",
            "tokenizer_runtime": "fabricated tokens"}


def synthetic_receipt(ids):
    provenance = synthetic_provenance()
    return {"input_ids_sha256": m.rows_digest(ids),
            "public_scope_sha256": provenance["public_scope_sha256"],
            "custodian_receipt_sha256": provenance["custodian_receipt_sha256"],
            "public_prefix_status": "clear", "custodian_status": "clear", "reason": "fabricated"}


def synthetic_case(word, subject, instruction, token_count, token_id):
    text = m.expected_text(word, subject, instruction)
    start = len(m.INSTRUCTIONS[instruction] + f" The {subject} was ")
    subject_start = text.index(f"The {subject} was ") + len("The ")
    subject_end = subject_start + len(subject)
    end = start + len(word)
    ids = [10, 11, 12]
    offsets = [[0, subject_start], [subject_start, subject_end], [subject_end, start - 1]]
    if token_count == 1:
        ids.append(token_id)
        offsets.append([start - 1, end])
    else:
        ids.extend([token_id, token_id + 100])
        offsets.extend([[start - 1, start + 1], [start + 1, end]])
    ids.extend([13, 1])
    offsets.extend([[end, end + 1], [0, 0]])
    return {"text": text, "input_ids": ids, "attention_mask": [1] * len(ids),
            "special_tokens_mask": [0] * (len(ids) - 1) + [1], "offset_mapping": offsets}


def fixture():
    positive = ["paaa", "pbbb", "pccc", "pddd", "peee", "pfff", "pggg", "phhh"]
    negative = ["naaa", "nbbb", "nccc", "nddd", "neee", "nfff", "nggg", "nhhh"]
    candidates = {"positive": positive, "negative": negative}
    cases, scope, pilot_rows = {}, {}, []
    for polarity in ("positive", "negative"):
        for index, word in enumerate(candidates[polarity]):
            token_count = 1 if index < 4 else 2
            token_id = 1000 + 300 * (polarity == "negative") + 2 * index
            templates, signature = [], []
            for subject in m.SUBJECTS:
                for instruction in (0, 1):
                    key = (word, subject, instruction)
                    case = synthetic_case(word, subject, instruction, token_count, token_id)
                    parsed = m.inspect_case(case, *key)
                    saved = dict(case, adjective_char_span=parsed["adjective_char_span"])
                    templates.append(saved)
                    signature.append(parsed["signature"])
                    cases[key] = case
                    scope[key] = synthetic_receipt(case["input_ids"])
            pilot_rows.append({"word": word, "polarity_proposal": polarity, "order": index,
                               "structural_rejections": [], "public_overlap_reasons": [],
                               "fresh_for_scoped_public_sources": True,
                               "signature": signature,
                               "signature_sha256": m.sha256_bytes(m.canonical(signature)),
                               "templates": templates})
    pilot = {"all_candidates": pilot_rows}
    prefiltered = m.prefilter(candidates, pilot)
    handoff = m.discovery_handoff(prefiltered)
    lookup = {word: polarity for polarity in ("positive", "negative")
              for word in candidates[polarity]}
    reviews = synthetic_reviews(handoff, lookup)
    return candidates, pilot, prefiltered, handoff, reviews, cases, scope


def synthetic_reviews(handoff, lookup):
    reviews = []
    for person in ("synthetic-rater-a", "synthetic-rater-b"):
        ratings = [{"word": item["word"], "subject": subject,
                    "label": lookup[item["word"]], "reason": "fabricated"}
                   for item in handoff["items"] for subject in handoff["subjects"]]
        reviews.append({"reviewer_id": person, "prior_exposure": "synthetic fixture",
                        "instructions": handoff["instructions"],
                        "handoff_sha256": m.sha256_bytes(m.canonical(handoff)),
                        "ratings": ratings})
    return reviews


def test_handoff_hides_labels_and_selection_uses_stable_disjoint_slots():
    candidates, _, prefiltered, handoff, reviews, cases, scope = fixture()
    assert [item["word"] for item in handoff["items"]] == sorted(candidates["positive"] + candidates["negative"])
    assert all(set(item) == {"word", "sentences"} for item in handoff["items"])
    assert "polarity" not in json.dumps(handoff) and "signature" not in json.dumps(handoff)
    result = m.select_slots(candidates, prefiltered, reviews, cases, scope, synthetic_provenance())
    assert [slot["status"] for slot in result["slots"]] == ["selected"] * 4
    assert result["slots"][0]["words"] == {"P1": "paaa", "N1": "naaa", "P2": "pbbb", "N2": "nbbb"}
    assert result["slots"][2]["words"] == {"P1": "peee", "N1": "neee", "P2": "pfff", "N2": "nfff"}
    words = [word for slot in result["slots"] for word in slot["words"].values()]
    assert len(words) == len(set(words)) == 16


def test_stale_review_and_duplicate_rater_fail_closed():
    _, _, prefiltered, handoff, reviews, _, _ = fixture()
    broken = copy.deepcopy(reviews)
    broken[0]["handoff_sha256"] = "0" * 64
    with pytest.raises(m.SelectionError, match="handoff hash"):
        m.validate_reviews(handoff, broken)
    broken = copy.deepcopy(reviews)
    broken[1]["reviewer_id"] = broken[0]["reviewer_id"]
    with pytest.raises(m.SelectionError, match="reviewer identity"):
        m.validate_reviews(handoff, broken)
    broken = copy.deepcopy(reviews)
    broken[0].pop("instructions")
    with pytest.raises(m.SelectionError, match="reviewer instructions"):
        m.validate_reviews(handoff, broken)
    broken = copy.deepcopy(reviews)
    broken[0]["ratings"][0]["reason"] = "  "
    with pytest.raises(m.SelectionError, match="rating reason"):
        m.validate_reviews(handoff, broken)
    broken = copy.deepcopy(reviews)
    broken[0]["prior_exposure"] = " "
    with pytest.raises(m.SelectionError, match="reviewer exposure"):
        m.validate_reviews(handoff, broken)
    broken = copy.deepcopy(reviews)
    broken[0]["instructions"] = "different directions"
    with pytest.raises(m.SelectionError, match="reviewer instructions"):
        m.validate_reviews(handoff, broken)


def test_bounded_read_duplicate_keys_and_unbound_scope_fail_closed(tmp_path):
    compressed = tmp_path / "large.json.gz"
    compressed.write_bytes(gzip.compress(b"x" * (m.MAX_JSON_BYTES + 1)))
    with pytest.raises(m.SelectionError, match="decompressed input"):
        m.read_bounded(compressed, compressed=True)
    with pytest.raises(m.SelectionError, match="duplicate JSON key"):
        m.parse_json(b'{"key":1,"key":2}')
    with pytest.raises(m.SelectionError, match="candidate inventory hash"):
        m.load_pinned_inventory(b"{}", b"{}")
    provenance = synthetic_provenance()
    receipt = synthetic_receipt([10, 1])
    receipt["public_scope_sha256"] = "f" * 64
    with pytest.raises(m.SelectionError, match="unbound scope"):
        m._check_receipt(receipt, m.rows_digest([10, 1]), provenance)


def test_underfill_keeps_original_slot_and_retains_rejected_rows():
    candidates, _, prefiltered, _, reviews, cases, scope = fixture()
    scope[("phhh", "story", 0)]["public_prefix_status"] = "collision"
    scope[("phhh", "story", 0)]["reason"] = "fabricated-prefix-collision"
    result = m.select_slots(candidates, prefiltered, reviews, cases, scope, synthetic_provenance())
    assert [slot["status"] for slot in result["slots"]] == ["selected", "selected", "selected", "missing"]
    failed = next(row for row in result["candidate_dispositions"] if row["word"] == "phhh")
    assert any("prefix-collision" in reason for reason in failed["reasons"])
    assert len(failed["attempted_rows"]) == 4
    assert result["slots"][2]["words"]["P1"] == "peee"
    assert result["slots"][3]["slot"] == 3


def test_subject_site_mismatch_does_not_seal_an_unusable_discovery_tuple():
    candidates, pilot, _, _, _, cases, scope = fixture()
    key = ("paaa", "story", 0)
    case = cases[key]
    start, end = case["offset_mapping"][1]
    case["offset_mapping"][1] = [start, start]
    case["offset_mapping"][2] = [start, end]
    assert m.inspect_case(case, *key)["subject_site"] == 2
    pilot_row = next(row for row in pilot["all_candidates"] if row["word"] == "paaa")
    pilot_row["templates"][0]["offset_mapping"] = copy.deepcopy(case["offset_mapping"])
    prefiltered = m.prefilter(candidates, pilot)
    handoff = m.discovery_handoff(prefiltered)
    lookup = {word: polarity for polarity in ("positive", "negative")
              for word in candidates[polarity]}
    result = m.select_slots(candidates, prefiltered, synthetic_reviews(handoff, lookup), cases, scope,
                            synthetic_provenance())
    assert result["slots"][0]["words"]["P1"] == "pbbb"
    assert result["slots"][1]["status"] == "missing"


def test_span_rejects_gap_punctuation_and_unattended_word_token():
    case = synthetic_case("paaa", "story", 0, 2, 1000)
    m.inspect_case(case, "paaa", "story", 0)
    gap = copy.deepcopy(case)
    gap["offset_mapping"][4][0] += 1
    with pytest.raises(m.SelectionError, match="gap"):
        m.inspect_case(gap, "paaa", "story", 0)
    punctuation = copy.deepcopy(case)
    punctuation["offset_mapping"][4][1] += 1
    with pytest.raises(m.SelectionError, match="punctuation"):
        m.inspect_case(punctuation, "paaa", "story", 0)
    unattended = copy.deepcopy(case)
    unattended["attention_mask"][3] = 0
    with pytest.raises(m.SelectionError, match="attention mask"):
        m.inspect_case(unattended, "paaa", "story", 0)


def test_subject_site_ignores_empty_offsets_and_checks_coverage_and_special_ids():
    case = synthetic_case("paaa", "story", 0, 1, 1000)
    start, end = case["offset_mapping"][1]
    extra = copy.deepcopy(case)
    for field, value in (("input_ids", 99), ("attention_mask", 1),
                         ("special_tokens_mask", 0), ("offset_mapping", [start + 1, start + 1])):
        extra[field].insert(2, value)
    assert m.inspect_case(extra, "paaa", "story", 0)["subject_site"] == 1
    empty = copy.deepcopy(case)
    empty["offset_mapping"][1] = [start, start]
    with pytest.raises(m.SelectionError, match="missing subject"):
        m.inspect_case(empty, "paaa", "story", 0)
    partial = copy.deepcopy(case)
    partial["offset_mapping"][1][1] = end - 1
    with pytest.raises(m.SelectionError, match="subject coverage"):
        m.inspect_case(partial, "paaa", "story", 0)
    special = copy.deepcopy(case)
    special["input_ids"][1] = 32099
    with pytest.raises(m.SelectionError, match="invalid subject token"):
        m.inspect_case(special, "paaa", "story", 0)


def test_holdout_binds_sealed_discovery_and_never_replaces_words():
    candidates, _, prefiltered, _, reviews, cases, scope = fixture()
    discovery = m.select_slots(candidates, prefiltered, reviews, cases, scope, synthetic_provenance())
    sealed = m.canonical(discovery) + b"\n"
    handoff = m.holdout_handoff(discovery)
    lookup, holdout_cases, holdout_scope = {}, {}, {}
    for slot_index in (0, 2):
        for role, word in discovery["slots"][slot_index]["words"].items():
            lookup[word] = "positive" if role.startswith("P") else "negative"
            for instruction in (0, 1):
                token_id = cases[(word, "story", instruction)]["input_ids"][3]
                case = synthetic_case(word, "painting", instruction, m.SLOTS[slot_index][0], token_id)
                key = (word, "painting", instruction)
                holdout_cases[key] = case
                holdout_scope[key] = synthetic_receipt(case["input_ids"])
    paint_reviews = synthetic_reviews(handoff, lookup)
    different_raters = copy.deepcopy(paint_reviews)
    different_raters[0]["reviewer_id"] = "replacement-rater"
    with pytest.raises(m.SelectionError, match="painting reviewer identities"):
        m.validate_painting_reviews(discovery, handoff, different_raters)
    original = copy.deepcopy(discovery)
    first_word = discovery["slots"][0]["words"]["P1"]
    holdout_scope[(first_word, "painting", 0)]["public_prefix_status"] = "collision"
    holdout_scope[(first_word, "painting", 0)]["reason"] = "fabricated-collision"
    result = m.assess_holdout(sealed, m.sha256_bytes(sealed), paint_reviews, holdout_cases, holdout_scope,
                              synthetic_provenance())
    assert result["discovery_sha256"] == m.sha256_bytes(sealed)
    assert [slot["status"] for slot in result["slots"]] == ["ineligible", "eligible"]
    assert result["slots"][0]["words"] == original["slots"][0]["words"]
    assert discovery == original
    with pytest.raises(m.SelectionError, match="discovery digest"):
        m.assess_holdout(sealed + b" ", m.sha256_bytes(sealed), paint_reviews,
                         holdout_cases, holdout_scope, synthetic_provenance())


def test_subject_site_mismatch_makes_only_painting_slot_ineligible():
    candidates, _, prefiltered, _, reviews, cases, scope = fixture()
    discovery = m.select_slots(candidates, prefiltered, reviews, cases, scope, synthetic_provenance())
    sealed = m.canonical(discovery) + b"\n"
    handoff = m.holdout_handoff(discovery)
    lookup, holdout_cases, holdout_scope = {}, {}, {}
    for slot_index in (0, 2):
        for role, word in discovery["slots"][slot_index]["words"].items():
            lookup[word] = "positive" if role.startswith("P") else "negative"
            for instruction in (0, 1):
                token_id = cases[(word, "story", instruction)]["input_ids"][3]
                case = synthetic_case(word, "painting", instruction, m.SLOTS[slot_index][0], token_id)
                key = (word, "painting", instruction)
                holdout_cases[key] = case
                holdout_scope[key] = synthetic_receipt(case["input_ids"])
    word = discovery["slots"][0]["words"]["P1"]
    case = holdout_cases[(word, "painting", 0)]
    start, end = case["offset_mapping"][1]
    case["offset_mapping"][1] = [start, start]
    case["offset_mapping"][2] = [start, end]
    result = m.assess_holdout(sealed, m.sha256_bytes(sealed), synthetic_reviews(handoff, lookup),
                              holdout_cases, holdout_scope, synthetic_provenance())
    assert [slot["status"] for slot in result["slots"]] == ["ineligible", "eligible"]
    assert "stratum-or-signature:0" in result["slots"][0]["reasons"]
