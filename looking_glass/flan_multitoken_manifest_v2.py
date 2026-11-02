"""Project frozen selector records into retokenized per-subject CPU groups.

This checks canonical manifest bytes, fixed text/slot linkage and selected
tokenizer records. It does not authenticate the selection algorithm, source,
reviewers, exposure scope or model execution; the owner and independent
reader must check those separately.
"""

import hashlib
import json
import re

from .flan_multitoken_span_v2 import validate_aligned_group


ROLES = ("P1", "N1", "P2", "N2")
SUBJECTS = ("story", "concert")
INSTRUCTIONS = ("Classify the sentiment of this review as positive or negative:",
                "Is this review positive or negative?")
_WORD = re.compile(r"[a-z]{1,24}\Z")
_MAX_BYTES = 8 * 1024**2


def _pairs(items):
    output = {}
    for key, value in items:
        if key in output:
            raise ValueError("duplicate JSON key")
        output[key] = value
    return output


def _manifest(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_BYTES:
        raise ValueError("manifest size differs")
    try:
        value = json.loads(raw.decode("utf8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        expected = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n").encode("utf8")
    except (ValueError, TypeError, UnicodeError, OverflowError) as error:
        raise ValueError("invalid manifest JSON") from error
    if raw != expected or type(value) is not dict:
        raise ValueError("manifest is not canonical")
    return value


def _case(record, *, word, role, subject, instruction, tokenizer):
    if type(record) is not dict or type(word) is not str or not _WORD.fullmatch(word):
        raise ValueError("invalid selected word")
    text = f"{INSTRUCTIONS[instruction]} The {subject} was {word}."
    if (len(text.encode("utf8")) > 256 or record.get("word") != word
            or record.get("subject") != subject or type(record.get("instruction")) is not int
            or record["instruction"] != instruction or record.get("text") != text):
        raise ValueError("selected case identity differs")
    actual = tokenizer(text, add_special_tokens=True, return_attention_mask=True,
                       return_special_tokens_mask=True, return_offsets_mapping=True, truncation=False)
    names = ("input_ids", "attention_mask", "special_tokens_mask", "offset_mapping")
    try:
        normalized = {name: list(actual[name]) for name in names}
        normalized["offset_mapping"] = [list(pair) for pair in normalized["offset_mapping"]]
    except (KeyError, TypeError) as error:
        raise ValueError("fresh tokenizer output is malformed") from error
    if any(type(record.get(name)) is not list or normalized[name] != record[name] for name in names):
        raise ValueError("fresh tokenizer output differs")
    subject_start = text.index(f"The {subject} was ") + len("The ")
    word_start = len(INSTRUCTIONS[instruction] + f" The {subject} was ")
    expected_subject = [subject_start, subject_start + len(subject)]
    expected_word = [word_start, word_start + len(word)]
    if record.get("subject_char_span") != expected_subject or record.get("adjective_char_span") != expected_word:
        raise ValueError("selected character spans differ")
    return {"side": role, "text": text, "subject": subject, "adjective": word,
            **{name: record[name] for name in names},
            "subject_char_span": expected_subject, "adjective_char_span": expected_word,
            "subject_index": record.get("subject_site"), "adjective_index": record.get("adjective_site")}


def _group(records, words, *, subject, instruction, unit, tokenizer, vocab_size, special_ids):
    if type(records) is not list or len(records) != 4:
        raise ValueError("selected group requires four rows")
    cases = {role: _case(record, word=words[role], role=role, subject=subject,
                         instruction=instruction, tokenizer=tokenizer)
             for role, record in zip(ROLES, records)}
    count = 1 if unit < 2 else 2
    validate_aligned_group(cases, lexical_count=count, boundary_count=count,
                           vocab_size=vocab_size, special_ids=special_ids)
    return {"unit": unit, "instruction": instruction, "cases": cases}


def project_manifests(discovery_raw, painting_raw, tokenizer, *, vocab_size, special_ids):
    """Verify and project selected cases; preserve missing original slot IDs."""
    if (type(vocab_size) is not int or not 2 < vocab_size <= 32128
            or type(special_ids) not in (set, frozenset)
            or set(getattr(tokenizer, "all_special_ids", ())) != set(special_ids)):
        raise ValueError("tokenizer vocabulary or special IDs differ")
    discovery, painting = _manifest(discovery_raw), _manifest(painting_raw)
    if (discovery.get("schema") != "flan-multisubtoken-discovery-selection-v1"
            or painting.get("schema") != "flan-multisubtoken-painting-selection-v1"
            or painting.get("discovery_sha256") != hashlib.sha256(discovery_raw).hexdigest()):
        raise ValueError("manifest identity or discovery binding differs")
    slots, holdout = discovery.get("slots"), painting.get("slots")
    if (type(slots) is not list or len(slots) != 4 or type(holdout) is not list or len(holdout) != 2
            or any(type(slot) is not dict or type(slot.get("slot")) is not int
                   or slot["slot"] != i for i, slot in enumerate(slots))
            or any(type(slot) is not dict or type(slot.get("slot")) is not int
                   or slot["slot"] != i for i, slot in zip((0, 2), holdout))):
        raise ValueError("planned slot identities differ")
    result = {"story": [], "concert": [], "painting": []}
    used, selected_words = set(), {}
    for unit, slot in enumerate(slots):
        count = 1 if unit < 2 else 2
        stratum = slot.get("stratum")
        if (type(stratum) is not list or len(stratum) != 2
                or any(type(value) is not int for value in stratum)
                or stratum != [count, count]):
            raise ValueError("slot stratum differs")
        if slot.get("status") == "missing":
            continue
        words, groups = slot.get("words"), slot.get("groups")
        if (slot.get("status") != "selected" or type(words) is not dict or set(words) != set(ROLES)
                or any(type(word) is not str or not _WORD.fullmatch(word) for word in words.values())
                or len(set(words.values())) != 4 or used.intersection(words.values())
                or type(groups) is not list or len(groups) != 4):
            raise ValueError("invalid selected slot")
        used.update(words.values())
        selected_words[unit] = words
        for subject_index, subject in enumerate(SUBJECTS):
            for instruction in (0, 1):
                result[subject].append(_group(groups[2 * subject_index + instruction], words,
                    subject=subject, instruction=instruction, unit=unit, tokenizer=tokenizer,
                    vocab_size=vocab_size, special_ids=special_ids))
    for slot in holdout:
        unit, status = slot["slot"], slot.get("status")
        if status == "missing":
            if unit in selected_words:
                raise ValueError("missing holdout conflicts with discovery")
            continue
        if status == "ineligible":
            if unit not in selected_words or slot.get("words") != selected_words[unit]:
                raise ValueError("ineligible holdout differs from discovery slot")
            continue
        words, groups = slot.get("words"), slot.get("groups")
        if (status != "eligible" or unit not in selected_words or words != selected_words[unit]
                or type(groups) is not list or len(groups) != 2):
            raise ValueError("invalid holdout selection")
        for instruction in (0, 1):
            result["painting"].append(_group(groups[instruction], words, subject="painting",
                instruction=instruction, unit=unit, tokenizer=tokenizer,
                vocab_size=vocab_size, special_ids=special_ids))
    return result
