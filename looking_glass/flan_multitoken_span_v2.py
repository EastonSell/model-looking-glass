"""Model-free admission checks for a matched FLAN adjective-token group.

These checks validate the supplied saved token records; they do not retokenize
text, verify source/selection custody, or authorize a model forward.  A
separate caller must compare every record with the pinned tokenizer first.
"""

from __future__ import annotations

import unicodedata


SIDES = ("P1", "N1", "P2", "N2")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _span(value, text):
    _require(type(value) is list and len(value) == 2
             and all(type(x) is int for x in value)
             and 0 <= value[0] < value[1] <= len(text), "invalid character span")
    return value


def validate_aligned_group(group, *, lexical_count, boundary_count, vocab_size, special_ids):
    """Validate one four-case group and return its common intervention sites.

    The boundary span includes the one space before the adjective.  The
    lexical span includes tokens overlapping adjective characters.  Four
    full word-token sequences must differ, but the final token IDs may agree.
    """
    _require(type(group) is dict and set(group) == set(SIDES), "four sides are required")
    _require(type(lexical_count) is int and lexical_count in (1, 2)
             and type(boundary_count) is int and boundary_count >= lexical_count,
             "invalid declared token counts")
    _require(type(vocab_size) is int and vocab_size > 2
             and type(special_ids) in (set, frozenset)
             and all(type(x) is int and 0 <= x < vocab_size for x in special_ids),
             "invalid tokenizer vocabulary")

    expected = None
    words = set()
    word_sequences = set()
    result = None
    for side in SIDES:
        case = group[side]
        _require(type(case) is dict and case.get("side") == side, "side differs")
        text = case.get("text")
        _require(type(text) is str and unicodedata.normalize("NFC", text) == text,
                 "invalid case text")
        subject, adjective = case.get("subject"), case.get("adjective")
        _require(type(subject) is str and subject and type(adjective) is str and adjective,
                 "subject and adjective are required")
        subject_start, subject_end = _span(case.get("subject_char_span"), text)
        word_start, word_end = _span(case.get("adjective_char_span"), text)
        _require(text[subject_start:subject_end] == subject
                 and text[word_start:word_end] == adjective
                 and word_start > 0 and text[word_start - 1] == " "
                 and subject_end <= word_start - 1, "text spans differ")

        ids = case.get("input_ids")
        attention = case.get("attention_mask")
        special = case.get("special_tokens_mask")
        offsets = case.get("offset_mapping")
        _require(type(ids) is list and 3 <= len(ids) <= 64
                 and all(type(x) is int and 0 <= x < vocab_size for x in ids)
                 and ids[-1] == 1 and ids.count(1) == 1, "invalid input IDs")
        _require(type(attention) is list and len(attention) == len(ids)
                 and all(type(x) is int and x == 1 for x in attention), "invalid attention mask")
        _require(type(special) is list and len(special) == len(ids)
                 and all(type(x) is int and x in (0, 1) for x in special)
                 and special[-1] == 1, "invalid special-token mask")
        _require(type(offsets) is list and len(offsets) == len(ids)
                 and all(type(pair) is list and len(pair) == 2
                         and all(type(x) is int for x in pair)
                         and 0 <= pair[0] <= pair[1] <= len(text) for pair in offsets),
                 "invalid offsets")

        boundary = tuple(i for i, (a, b) in enumerate(offsets)
                         if a < b and a < word_end and b > word_start - 1)
        lexical = tuple(i for i, (a, b) in enumerate(offsets)
                        if a < b and a < word_end and b > word_start)
        _require(len(boundary) == boundary_count and len(lexical) == lexical_count
                 and boundary == tuple(range(boundary[0], boundary[-1] + 1))
                 and lexical[-1] == boundary[-1], "invalid adjective token count or position")
        _require(offsets[boundary[0]][0] == word_start - 1
                 and offsets[boundary[-1]][1] == word_end
                 and all(word_start - 1 <= offsets[i][0] < offsets[i][1] <= word_end
                         for i in boundary)
                 and all(offsets[a][1] == offsets[b][0] for a, b in zip(boundary, boundary[1:])),
                 "adjective offsets must cover exactly the space and word")
        _require(all(attention[i] == 1 and special[i] == 0 and ids[i] not in special_ids
                     for i in boundary), "invalid adjective token")

        subject_tokens = tuple(i for i, (a, b) in enumerate(offsets)
                               if a < b and a < subject_end and b > subject_start)
        _require(subject_tokens and subject_tokens == tuple(range(subject_tokens[0], subject_tokens[-1] + 1))
                 and all(attention[i] == 1 and special[i] == 0 and ids[i] not in special_ids
                         for i in subject_tokens), "invalid subject tokens")
        covered = {x for i in subject_tokens for x in range(max(subject_start, offsets[i][0]),
                                                             min(subject_end, offsets[i][1]))}
        _require(covered == set(range(subject_start, subject_end)), "subject token coverage differs")
        subject_index, adjective_index = case.get("subject_index"), case.get("adjective_index")
        _require(type(subject_index) is int and subject_index == subject_tokens[-1]
                 and type(adjective_index) is int and adjective_index == lexical[-1]
                 and special[0] == 0 and ids[0] not in special_ids
                 and 0 not in boundary and 0 not in subject_tokens
                 and not set(boundary).intersection(subject_tokens), "invalid intervention sites")

        lo, hi = boundary[0], boundary[-1] + 1
        fixed = (text[:word_start], text[word_end:], subject, tuple(ids[:lo]), tuple(ids[hi:]),
                 tuple(attention), tuple(special), boundary, lexical, subject_index, adjective_index)
        if expected is not None:
            _require(fixed == expected, "cases differ outside the adjective span")
        expected = fixed
        _require(adjective not in words, "repeated adjective")
        words.add(adjective)
        word_ids = tuple(ids[lo:hi])
        _require(word_ids not in word_sequences, "repeated full word token sequence")
        word_sequences.add(word_ids)
        result = {"boundary_positions": boundary, "lexical_positions": lexical,
                  "subject_index": subject_index, "adjective_index": adjective_index,
                  "instruction_index": 0, "sequence_length": len(ids)}
    return result
