"""Fabricated selector-to-executor bindings and fresh tokenizer checks."""

import copy
import hashlib
import json
import re
import zlib

import pytest

from looking_glass.flan_multitoken_manifest_v2 import project_manifests


ROLES = ("P1", "N1", "P2", "N2")
INSTRUCTIONS = ("Classify the sentiment of this review as positive or negative:",
                "Is this review positive or negative?")


class FakeTokenizer:
    all_special_ids = [0, 1, 2]

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": True, "return_attention_mask": True,
                          "return_special_tokens_mask": True, "return_offsets_mapping": True,
                          "truncation": False}
        ids, offsets = [], []
        for match in re.finditer(r"[A-Za-z]+|[^\w\s]", text):
            word, start, end = match.group(), match.start(), match.end()
            left = start - 1 if start and text[start - 1] == " " else start
            token = 3 + zlib.crc32(word.encode()) % 30000
            if word in {"grand", "awful", "loved", "hated"}:
                ids.extend([token, 32000])
                offsets.extend([[left, start + 2], [start + 2, end]])
            else:
                ids.append(token)
                offsets.append([left, end])
        return {"input_ids": ids + [1], "attention_mask": [1] * (len(ids) + 1),
                "special_tokens_mask": [0] * len(ids) + [1],
                "offset_mapping": [tuple(pair) for pair in offsets] + [(0, 0)]}


def case(tokenizer, word, subject, instruction):
    text = f"{INSTRUCTIONS[instruction]} The {subject} was {word}."
    tokenized = tokenizer(text, add_special_tokens=True, return_attention_mask=True,
                          return_special_tokens_mask=True, return_offsets_mapping=True, truncation=False)
    tokenized["offset_mapping"] = [list(pair) for pair in tokenized["offset_mapping"]]
    start, end = text.rindex(word), text.rindex(word) + len(word)
    subject_start = text.index(f"The {subject} was") + len("The ")
    subject_end = subject_start + len(subject)
    offsets = tokenized["offset_mapping"]
    subject_site = max(i for i, (a, b) in enumerate(offsets) if a < b and a < subject_end and b > subject_start)
    adjective_site = max(i for i, (a, b) in enumerate(offsets) if a < b and a < end and b > start)
    return {"word": word, "subject": subject, "instruction": instruction, "text": text,
            **tokenized, "subject_char_span": [subject_start, subject_end],
            "adjective_char_span": [start, end], "subject_site": subject_site,
            "adjective_site": adjective_site}


def fixture():
    tokenizer = FakeTokenizer()
    words = {"P1": "good", "N1": "bad", "P2": "nice", "N2": "poor"}
    groups = [[case(tokenizer, words[role], subject, instruction) for role in ROLES]
              for subject in ("story", "concert") for instruction in (0, 1)]
    discovery = {"schema": "flan-multisubtoken-discovery-selection-v1", "slots": [
        {"slot": 0, "stratum": [1, 1], "status": "selected", "words": words, "groups": groups},
        {"slot": 1, "stratum": [1, 1], "status": "missing", "reason": "synthetic"},
        {"slot": 2, "stratum": [2, 2], "status": "missing", "reason": "synthetic"},
        {"slot": 3, "stratum": [2, 2], "status": "missing", "reason": "synthetic"}],
        "candidate_dispositions": [], "reviewer_records": [], "provenance": {}, "scope": "fixture"}
    discovery_raw = canonical(discovery)
    holdout = {"schema": "flan-multisubtoken-painting-selection-v1",
        "discovery_sha256": hashlib.sha256(discovery_raw).hexdigest(),
        "slots": [{"slot": 0, "status": "eligible", "words": words,
                   "groups": [[case(tokenizer, words[role], "painting", i) for role in ROLES]
                              for i in (0, 1)]},
                  {"slot": 2, "status": "missing", "reason": "discovery-slot-missing"}],
        "reviewer_records": [], "provenance": {}, "scope": "fixture"}
    return discovery, holdout, tokenizer


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def test_projects_stable_slots_and_retokenizes_all_three_subjects():
    discovery, holdout, tokenizer = fixture()
    result = project_manifests(canonical(discovery), canonical(holdout), tokenizer,
                               vocab_size=32128, special_ids=set(tokenizer.all_special_ids))
    assert [(g["unit"], g["instruction"]) for g in result["story"]] == [(0, 0), (0, 1)]
    assert [(g["unit"], g["instruction"]) for g in result["concert"]] == [(0, 0), (0, 1)]
    assert [(g["unit"], g["instruction"]) for g in result["painting"]] == [(0, 0), (0, 1)]
    assert result["story"][0]["cases"]["P1"]["adjective"] == "good"


@pytest.mark.parametrize("mutation", ["row_ids", "offset", "subject_site", "word_role", "subject", "digest"])
def test_rejects_selected_record_or_manifest_binding_changes(mutation):
    discovery, holdout, tokenizer = fixture()
    if mutation == "digest":
        holdout["discovery_sha256"] = "0" * 64
    else:
        row = discovery["slots"][0]["groups"][0][0]
        if mutation == "row_ids":
            row["input_ids"][1] += 1
        elif mutation == "offset":
            row["offset_mapping"][1] = [0, 0]
        elif mutation == "subject_site":
            row["subject_site"] = 0
        elif mutation == "word_role":
            discovery["slots"][0]["words"]["P1"] = "bad"
        else:
            row["subject"] = "concert"
        holdout["discovery_sha256"] = hashlib.sha256(canonical(discovery)).hexdigest()
    with pytest.raises(ValueError):
        project_manifests(canonical(discovery), canonical(holdout), tokenizer,
                          vocab_size=32128, special_ids=set(tokenizer.all_special_ids))


def test_rejects_noncanonical_and_duplicate_key_json():
    discovery, holdout, tokenizer = fixture()
    with pytest.raises(ValueError):
        project_manifests(json.dumps(discovery).encode(), canonical(holdout), tokenizer,
                          vocab_size=32128, special_ids=set(tokenizer.all_special_ids))
    with pytest.raises(ValueError):
        project_manifests(b'{"schema":"one","schema":"two"}\n', canonical(holdout), tokenizer,
                          vocab_size=32128, special_ids=set(tokenizer.all_special_ids))


def test_two_subtoken_slot_may_share_all_final_ids():
    discovery, holdout, tokenizer = fixture()
    words = {"P1": "grand", "N1": "awful", "P2": "loved", "N2": "hated"}
    discovery["slots"][2] = {"slot": 2, "stratum": [2, 2], "status": "selected", "words": words,
        "groups": [[case(tokenizer, words[role], subject, instruction) for role in ROLES]
                   for subject in ("story", "concert") for instruction in (0, 1)]}
    holdout["slots"][1] = {"slot": 2, "status": "eligible", "words": words,
        "groups": [[case(tokenizer, words[role], "painting", instruction) for role in ROLES]
                   for instruction in (0, 1)]}
    discovery_raw = canonical(discovery)
    holdout["discovery_sha256"] = hashlib.sha256(discovery_raw).hexdigest()
    result = project_manifests(discovery_raw, canonical(holdout), tokenizer,
                               vocab_size=32128, special_ids=set(tokenizer.all_special_ids))
    group = result["painting"][2]
    assert group["unit"] == 2
    assert {case["input_ids"][case["adjective_index"]] for case in group["cases"].values()} == {32000}


def test_empty_selection_projects_no_model_groups():
    discovery, holdout, tokenizer = fixture()
    discovery["slots"][0] = {"slot": 0, "stratum": [1, 1], "status": "missing", "reason": "synthetic"}
    holdout["slots"][0] = {"slot": 0, "status": "missing", "reason": "discovery-slot-missing"}
    discovery_raw = canonical(discovery)
    holdout["discovery_sha256"] = hashlib.sha256(discovery_raw).hexdigest()
    result = project_manifests(discovery_raw, canonical(holdout), tokenizer,
                               vocab_size=32128, special_ids=set(tokenizer.all_special_ids))
    assert result == {"story": [], "concert": [], "painting": []}


def test_ineligible_holdout_still_binds_the_exact_discovery_words():
    discovery, holdout, tokenizer = fixture()
    holdout["slots"][0] = {"slot": 0, "status": "ineligible", "words": {
        "P1": "other", "N1": "bad", "P2": "nice", "N2": "poor"}, "reasons": ["synthetic"]}
    with pytest.raises(ValueError):
        project_manifests(canonical(discovery), canonical(holdout), tokenizer,
                          vocab_size=32128, special_ids=set(tokenizer.all_special_ids))


def test_boolean_stratum_is_not_an_integer_stratum():
    discovery, holdout, tokenizer = fixture()
    discovery["slots"][0]["stratum"] = [True, True]
    raw = canonical(discovery)
    holdout["discovery_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        project_manifests(raw, canonical(holdout), tokenizer,
                          vocab_size=32128, special_ids=set(tokenizer.all_special_ids))
