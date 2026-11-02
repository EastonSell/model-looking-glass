"""Fabricated checks for the proposed multi-token span admission.

These cases contain no tokenizer or model output.
"""

import copy

import pytest

from looking_glass.flan_multitoken_span_v2 import validate_aligned_group


SIDES = ("P1", "N1", "P2", "N2")


def fake_group(*, lexical_count=2):
    words = ("good", "evil", "rosy", "grim")
    first_tokens = (100, 101, 102, 103)
    last_tokens = (200, 200, 201, 201)
    group = {}
    for side, word, first, last in zip(SIDES, words, first_tokens, last_tokens):
        text = "Rate the story was " + word + "."
        start = text.index(word)
        end = start + len(word)
        if lexical_count == 2:
            word_ids = [first, last]
            word_offsets = [[start - 1, start + 2], [start + 2, end]]
        else:
            word_ids = [first]
            word_offsets = [[start - 1, end]]
        ids = [10, 11, 12, 13, *word_ids, 14, 1]
        offsets = [[0, 4], [4, 8], [8, 14], [14, 18],
                   *word_offsets, [end, end + 1], [0, 0]]
        group[side] = {
            "side": side,
            "text": text,
            "subject": "story",
            "adjective": word,
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "special_tokens_mask": [0] * (len(ids) - 1) + [1],
            "offset_mapping": offsets,
            "subject_char_span": [9, 14],
            "adjective_char_span": [start, end],
            "subject_index": 2,
            "adjective_index": 3 + lexical_count,
        }
    return group


def check(group, count=2):
    return validate_aligned_group(
        group, lexical_count=count, boundary_count=count,
        vocab_size=32128, special_ids={0, 1, 2},
    )


def test_distinct_multitoken_words_can_share_final_token_ids():
    result = check(fake_group())
    assert result["adjective_index"] == 5
    assert result["subject_index"] == 2
    assert result["boundary_positions"] == (4, 5)
    assert result["lexical_positions"] == (4, 5)


def test_single_token_has_the_same_span_and_final_site():
    result = check(fake_group(lexical_count=1), count=1)
    assert result["boundary_positions"] == (4,)
    assert result["lexical_positions"] == (4,)
    assert result["adjective_index"] == 4


@pytest.mark.parametrize("failure", [
    "outside_id", "same_word_tokens", "site", "gap", "punctuation",
    "unattended", "special", "mask", "subject", "stratum", "text",
])
def test_rejects_a_bad_group_before_model_work(failure):
    group = fake_group()
    case = group["N1"]
    if failure == "outside_id":
        case["input_ids"][1] = 15
    elif failure == "same_word_tokens":
        case["input_ids"][4:6] = group["P1"]["input_ids"][4:6]
    elif failure == "site":
        case["adjective_index"] = 4
    elif failure == "gap":
        case["offset_mapping"][5][0] += 1
    elif failure == "punctuation":
        case["offset_mapping"][5][1] += 1
    elif failure == "unattended":
        case["attention_mask"][4] = 0
    elif failure == "special":
        case["special_tokens_mask"][4] = 1
    elif failure == "mask":
        case["special_tokens_mask"][1] = 1
    elif failure == "subject":
        case["subject_index"] = 1
    elif failure == "text":
        case["text"] = case["text"].replace("story", "books")
    with pytest.raises(ValueError):
        check(group, count=1 if failure == "stratum" else 2)


def test_no_input_mutation():
    group = fake_group()
    before = copy.deepcopy(group)
    check(group)
    assert group == before


def test_empty_offset_cannot_become_a_subject_intervention_site():
    group = fake_group()
    for case in group.values():
        case["offset_mapping"][3] = [10, 10]
        case["subject_index"] = 3
    with pytest.raises(ValueError):
        check(group)
