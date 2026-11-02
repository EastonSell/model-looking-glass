"""Model-free prefilter, blinded handoff and fixed-slot selector.

This module does not tokenize or run a model.  The caller supplies bounded
tokenizer records and scoped comparison receipts; it cannot turn those
cooperative receipts into proof of unrecorded exposures.  The public protocol
is docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md, SHA-256 bdb24967... .
"""

from collections import Counter, defaultdict
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import re


SUBJECTS = ("story", "concert")
INSTRUCTIONS = (
    "Classify the sentiment of this review as positive or negative:",
    "Is this review positive or negative?",
)
REVIEW_INSTRUCTIONS = ("For each sentence, classify the sentiment conveyed by the adjective "
                       "about the named subject as positive, negative, or ambiguous in ordinary "
                       "usage. Record one label and a brief reason for each sentence. "
                       "Judge each sentence independently.")
SLOTS = ((1, 1), (1, 1), (2, 2), (2, 2))
ROLES = ("P1", "N1", "P2", "N2")
MAX_TOKEN_COUNT = 64
MAX_TEXT_BYTES = 256
MAX_JSON_BYTES = 16 * 1024 * 1024
WORD = re.compile(r"[a-z]{1,24}\Z")
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
SPECIAL_IDS = frozenset((0, 1, 2, *range(32000, 32100)))
PROTOCOL_SHA256 = "bdb249673dc9045b8e83f4e5f525cc04e530f5f78f7a68fe72d26447e01fdb12"
CANDIDATES_SHA256 = "2b32da31aab398d19ef80e013c377d2ed3d1a06912a431ee918bfeccb64107d1"
PILOT_SHA256 = "c1426f1ea4690f7c5185580167001d946425dcf0e3d08664a1f993fc7ac0a62f"
TOKENIZER_SHA256 = "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a"
ASSET_MANIFEST_SHA256 = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
PRIOR_DESCRIPTORS_SHA256 = "536987b063b002b2ed5a4dcdf8c6d9e105cb0e9648da0eb85a89cf95db6180a8"
RELEASED_POOL_SHA256 = "de4c13ea29ecb0f58eb36a6a6c171abf1488e5036fd0c2fe567b13d34692b93e"


class SelectionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SelectionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_bounded(path, *, compressed=False):
    """Read one bounded raw JSON file, or bounded gzip and decompressed bytes."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "compressed or raw input exceeds limit")
    if not compressed:
        return raw
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
        result = stream.read(MAX_JSON_BYTES + 1)
    require(len(result) <= MAX_JSON_BYTES, "decompressed input exceeds limit")
    return result


def parse_json(raw):
    require(isinstance(raw, bytes) and len(raw) <= MAX_JSON_BYTES, "JSON byte limit")
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject_constant(value):
        raise SelectionError(f"invalid JSON number: {value}")
    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=reject_constant)


def load_pinned_inventory(candidate_bytes, pilot_bytes):
    """Bind the released candidate list and uncompressed report before use."""
    require(sha256_bytes(candidate_bytes) == CANDIDATES_SHA256, "candidate inventory hash")
    require(sha256_bytes(pilot_bytes) == PILOT_SHA256, "pilot report hash")
    candidates, pilot = parse_json(candidate_bytes), parse_json(pilot_bytes)
    require(candidates.get("schema") == "flan-multitoken-tokenizer-feasibility-candidates-v1" and
            len(candidates.get("positive", [])) == len(candidates.get("negative", [])) == 80,
            "candidate inventory shape")
    templates = [f"{instruction} The {subject} was {{word}}."
                 for subject in SUBJECTS for instruction in INSTRUCTIONS]
    require(candidates.get("templates") == templates, "candidate templates")
    require(pilot.get("schema") == "flan-multitoken-tokenizer-feasibility-report-v1" and
            pilot.get("candidate_sha256") == CANDIDATES_SHA256 and
            pilot.get("tokenizer_sha256") == TOKENIZER_SHA256 and
            pilot.get("asset_manifest_sha256") == ASSET_MANIFEST_SHA256 and
            pilot.get("candidates") == 160 and pilot.get("tokenized_rows") == 640,
            "pilot report identity")
    return candidates, pilot


def rows_digest(ids):
    return sha256_bytes(canonical(ids))


def validate_provenance(provenance):
    """Validate declared identities, not the authenticity of their receipts."""
    require(isinstance(provenance, dict), "provenance object")
    fixed = {"protocol_sha256": PROTOCOL_SHA256, "candidates_sha256": CANDIDATES_SHA256,
             "pilot_sha256": PILOT_SHA256, "tokenizer_sha256": TOKENIZER_SHA256,
             "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
             "prior_descriptors_sha256": PRIOR_DESCRIPTORS_SHA256,
             "released_pool_sha256": RELEASED_POOL_SHA256,
             "prefix_method": "one-terminal-eos-stripped-bidirectional-prefix"}
    for key, value in fixed.items():
        require(provenance.get(key) == value, f"provenance mismatch: {key}")
    for key in ("public_scope_sha256", "custodian_receipt_sha256"):
        value = provenance.get(key)
        require(isinstance(value, str) and HEX256.fullmatch(value) is not None,
                f"missing provenance digest: {key}")
    for key in ("public_scope_description", "custodian_scope_description", "tokenizer_runtime"):
        value = provenance.get(key)
        require(isinstance(value, str) and 0 < len(value) <= 4096,
                f"missing provenance description: {key}")


def _check_word(word):
    require(isinstance(word, str) and WORD.fullmatch(word) is not None,
            "word must be 1-24 lower-case ASCII letters")


def _candidate_index(candidates):
    require(isinstance(candidates, dict), "candidate inventory must be an object")
    out = {}
    for polarity in ("positive", "negative"):
        words = candidates.get(polarity)
        require(isinstance(words, list), "candidate polarity list missing")
        for index, word in enumerate(words):
            _check_word(word)
            require(word not in out, "duplicate candidate word")
            out[word] = {"polarity": polarity, "index": index}
    return out


def prefilter(candidates, pilot):
    """Reconstruct capable pilot signatures and return all dispositions.

    A production caller must first verify the fixed candidate and full pilot
    report hashes.  Returned records contain the proposed labels and must not
    be included in a blinded reviewer handoff.
    """
    inventory = _candidate_index(candidates)
    rows = pilot.get("all_candidates") if isinstance(pilot, dict) else None
    require(isinstance(rows, list) and len(rows) == len(inventory), "pilot row count")
    by_word = {}
    counts = defaultdict(Counter)
    for row in rows:
        require(isinstance(row, dict), "malformed pilot row")
        word = row.get("word")
        require(word in inventory and word not in by_word, "unknown or duplicate pilot word")
        expected = inventory[word]
        require(row.get("polarity_proposal") == expected["polarity"] and
                row.get("order") == expected["index"], "pilot candidate order")
        rejected = row.get("structural_rejections")
        overlap = row.get("public_overlap_reasons")
        require(isinstance(rejected, list) and isinstance(overlap, list), "pilot dispositions")
        require(row.get("fresh_for_scoped_public_sources") is (not bool(overlap)),
                "pilot overlap flag")
        signature = row.get("signature")
        signature_sha = row.get("signature_sha256")
        if not rejected:
            require(isinstance(signature, list) and len(signature) == 4,
                    "pilot signature shape")
            require(sha256_bytes(canonical(signature)) == signature_sha, "pilot signature digest")
            require(all(isinstance(s, dict) for s in signature), "pilot signature fields")
            require(isinstance(row.get("templates"), list) and len(row["templates"]) == 4,
                    "pilot template shape")
            if not overlap:
                counts[signature_sha][expected["polarity"]] += 1
        else:
            require(signature_sha is None, "rejected pilot signature")
        by_word[word] = row

    out = []
    for word, identity in inventory.items():
        row = by_word[word]
        reasons = []
        if row["structural_rejections"]:
            reasons.append("pilot-structural-rejection")
        if row["public_overlap_reasons"]:
            reasons.append("pilot-scoped-source-overlap")
        if not reasons:
            signature = row["signature"]
            lengths = [(s.get("lexical_count"), s.get("word_plus_space_count"))
                       for s in signature]
            if lengths not in ([(1, 1)] * 4, [(2, 2)] * 4):
                reasons.append("outside-fixed-strata")
            bucket = counts[row["signature_sha256"]]
            if bucket["positive"] < 2 or bucket["negative"] < 2:
                reasons.append("pilot-bucket-underfilled")
        out.append({"word": word, "polarity": identity["polarity"],
                    "index": identity["index"], "eligible_for_review": not reasons,
                    "reasons": reasons, "pilot_row": row})
    return out


def discovery_handoff(prefiltered):
    """Return only alphabetically ordered words and review sentences."""
    words = sorted(r["word"] for r in prefiltered if r["eligible_for_review"])
    return {"schema": "flan-multisubtoken-semantic-handoff-v1",
            "subjects": list(SUBJECTS),
            "instructions": REVIEW_INSTRUCTIONS,
            "items": [{"word": word,
                       "sentences": [f"The {subject} was {word}." for subject in SUBJECTS]}
                      for word in words]}


def holdout_handoff(discovery):
    """Build a label-free painting handoff from sealed slots 0 and 2 only."""
    slots = discovery.get("slots") if isinstance(discovery, dict) else None
    require(isinstance(slots, list) and len(slots) == 4, "discovery slots")
    words = []
    for index in (0, 2):
        slot = slots[index]
        require(slot.get("slot") == index, "discovery slot order")
        if slot.get("status") == "selected":
            words.extend(slot["words"][role] for role in ROLES)
        else:
            require(slot.get("status") == "missing", "discovery slot status")
    require(len(words) == len(set(words)), "reused discovery words")
    for word in words:
        _check_word(word)
    return {"schema": "flan-multisubtoken-painting-handoff-v1", "subjects": ["painting"],
            "instructions": REVIEW_INSTRUCTIONS,
            "items": [{"word": word, "sentences": [f"The painting was {word}."]}
                      for word in sorted(words)]}


def validate_reviews(handoff, reviews):
    """Require two complete, distinct reviews; return verdicts by word/site."""
    require(isinstance(reviews, list) and len(reviews) == 2, "two reviews required")
    words = [item["word"] for item in handoff["items"]]
    subjects = handoff["subjects"]
    require(words == sorted(set(words)), "handoff order or duplicates")
    require(handoff.get("instructions") == REVIEW_INSTRUCTIONS, "handoff instructions")
    expected = {(word, subject) for word in words for subject in subjects}
    output = {}
    identities = set()
    handoff_sha = sha256_bytes(canonical(handoff))
    for review in reviews:
        require(isinstance(review, dict), "review object required")
        identity = review.get("reviewer_id")
        exposure = review.get("prior_exposure")
        instructions = review.get("instructions")
        require(review.get("handoff_sha256") == handoff_sha, "review handoff hash")
        require(isinstance(identity, str) and 0 < len(identity) <= 128 and identity == identity.strip() and
                identity not in identities, "reviewer identity")
        require(isinstance(exposure, str) and 0 < len(exposure.strip()) and len(exposure) <= 2048,
                "reviewer exposure statement")
        require(instructions == handoff["instructions"], "reviewer instructions")
        identities.add(identity)
        ratings = review.get("ratings")
        require(isinstance(ratings, list) and len(ratings) == len(expected),
                "incomplete review")
        seen = {}
        for rating in ratings:
            require(isinstance(rating, dict), "rating object")
            key = (rating.get("word"), rating.get("subject"))
            require(key in expected and key not in seen, "duplicate or unexpected rating")
            label, reason = rating.get("label"), rating.get("reason")
            require(label in ("positive", "negative", "ambiguous"), "invalid semantic label")
            require(isinstance(reason, str) and 0 < len(reason.strip()) and len(reason) <= 2048,
                    "invalid rating reason")
            seen[key] = {"label": label, "reason": reason}
        require(set(seen) == expected, "missing rating")
        output[identity] = seen
    return output


def validate_painting_reviews(discovery, handoff, reviews):
    """Require the same cooperatively identified raters as the sealed discovery."""
    checked = validate_reviews(handoff, reviews)
    prior = discovery.get("reviewer_records")
    require(isinstance(prior, list) and len(prior) == 2 and
            all(isinstance(item, dict) and isinstance(item.get("reviewer_id"), str)
                for item in prior), "discovery reviewer identities")
    prior_ids = {item["reviewer_id"] for item in prior}
    require(len(prior_ids) == 2 and prior_ids == set(checked), "painting reviewer identities")
    return checked


def expected_text(word, subject, instruction):
    _check_word(word)
    require(subject in ("story", "concert", "painting"), "unknown subject")
    require(type(instruction) is int and instruction in (0, 1), "unknown instruction")
    text = f"{INSTRUCTIONS[instruction]} The {subject} was {word}."
    require(len(text.encode("utf-8")) <= MAX_TEXT_BYTES, "case text too long")
    return text


def inspect_case(case, word, subject, instruction):
    """Check actual IDs, masks, character boundaries, EOS, and distinct sites."""
    require(isinstance(case, dict), "case must be an object")
    text = expected_text(word, subject, instruction)
    require(case.get("text") == text, "case text mismatch")
    ids, mask = case.get("input_ids"), case.get("attention_mask")
    special, offsets = case.get("special_tokens_mask"), case.get("offset_mapping")
    require(all(isinstance(x, list) for x in (ids, mask, special, offsets)), "case arrays")
    n = len(ids)
    require(1 < n <= MAX_TOKEN_COUNT and len(mask) == len(special) == len(offsets) == n,
            "case array lengths")
    require(all(type(v) is int and 0 <= v < 32128 for v in ids), "token ID")
    require(all(type(v) is int and v == 1 for v in mask), "attention mask")
    require(all(type(v) is int and v in (0, 1) for v in special), "special mask")
    require(ids[-1] == 1 and ids.count(1) == 1 and special[-1] == 1, "terminal EOS")
    require(all(isinstance(pair, list) and len(pair) == 2 and
                all(type(x) is int for x in pair) and 0 <= pair[0] <= pair[1] <= len(text)
                for pair in offsets), "offsets")
    word_start = len(INSTRUCTIONS[instruction] + f" The {subject} was ")
    word_end = word_start + len(word)
    require(text[word_start - 1] == " " and text[word_end:] == ".", "word boundary")
    boundary = [i for i, (a, b) in enumerate(offsets)
                if a < b and a < word_end and b > word_start - 1]
    lexical = [i for i, (a, b) in enumerate(offsets)
               if a < b and a < word_end and b > word_start]
    require(bool(boundary) and bool(lexical), "empty adjective span")
    require(boundary == list(range(boundary[0], boundary[-1] + 1)) and
            lexical[-1] == boundary[-1], "noncontiguous adjective span")
    require(all(mask[i] == 1 and special[i] == 0 and ids[i] not in SPECIAL_IDS for i in boundary),
            "unattended, special, or unknown adjective token")
    interval = [offsets[i] for i in boundary]
    require(interval[0][0] == word_start - 1 and interval[-1][1] == word_end and
            all(a < b for a, b in interval) and
            all(interval[i][1] == interval[i + 1][0] for i in range(len(interval) - 1)),
            "adjective offsets have gap, overlap, or punctuation")
    subject_start = text.index(f"The {subject} was ") + len("The ")
    subject_end = subject_start + len(subject)
    subject_tokens = [i for i, (a, b) in enumerate(offsets)
                      if a < b and a < subject_end and b > subject_start]
    require(bool(subject_tokens), "missing subject token")
    require(subject_tokens == list(range(subject_tokens[0], subject_tokens[-1] + 1)),
            "noncontiguous subject tokens")
    require(all(mask[i] == 1 and special[i] == 0 and ids[i] not in SPECIAL_IDS
                for i in subject_tokens), "invalid subject token")
    subject_intervals = [[max(offsets[i][0], subject_start), min(offsets[i][1], subject_end)]
                         for i in subject_tokens]
    require(subject_intervals[0][0] == subject_start and subject_intervals[-1][1] == subject_end
            and all(a < b for a, b in subject_intervals)
            and all(subject_intervals[i][1] == subject_intervals[i + 1][0]
                    for i in range(len(subject_intervals) - 1)), "incomplete subject coverage")
    subject_site, adjective_site = subject_tokens[-1], lexical[-1]
    require(mask[subject_site] == 1 and special[subject_site] == 0 and ids[subject_site] not in SPECIAL_IDS,
            "invalid subject site")
    require(mask[0] == 1 and special[0] == 0 and ids[0] not in SPECIAL_IDS, "invalid instruction site")
    require(subject_site != 0 and all(i not in (0, subject_site) for i in lexical),
            "site overlap")
    left, right = boundary[0], boundary[-1] + 1
    signature = {"lexical_count": len(lexical), "word_plus_space_count": len(boundary),
                 "last_lexical_index": adjective_site, "prefix_ids": ids[:left],
                 "suffix_ids": ids[right:], "prefix_attention": mask[:left],
                 "suffix_attention": mask[right:], "prefix_special": special[:left],
                 "suffix_special": special[right:]}
    return {"word": word, "subject": subject, "instruction": instruction,
            "text": text, "input_ids": ids, "attention_mask": mask,
            "special_tokens_mask": special, "offset_mapping": offsets,
            "adjective_char_span": [word_start, word_end],
            "subject_char_span": [subject_start, subject_end], "boundary_indices": boundary,
            "lexical_indices": lexical, "adjective_site": adjective_site,
            "subject_site": subject_site, "instruction_site": 0,
            "full_word_ids": [ids[i] for i in boundary], "signature": signature,
            "input_ids_sha256": rows_digest(ids)}


def _review_accepts(reviews, word, subject, expected):
    labels = [review[(word, subject)]["label"] for review in reviews.values()]
    return labels == [expected, expected]


def _group_compatible(cases, stratum):
    signatures = [case["signature"] for case in cases]
    if any((s["lexical_count"], s["word_plus_space_count"]) != stratum for s in signatures):
        return False
    if any(s != signatures[0] for s in signatures[1:]):
        return False
    if len({case["subject_site"] for case in cases}) != 1:
        return False
    return len({tuple(case["full_word_ids"]) for case in cases}) == 4


def _check_receipt(receipt, input_ids_sha256, provenance):
    require(isinstance(receipt, dict) and
            receipt.get("input_ids_sha256") == input_ids_sha256 and
            receipt.get("public_scope_sha256") == provenance["public_scope_sha256"] and
            receipt.get("custodian_receipt_sha256") == provenance["custodian_receipt_sha256"] and
            isinstance(receipt.get("reason"), str) and len(receipt["reason"]) <= 2048,
            "missing or unbound scope")
    statuses = (receipt.get("public_prefix_status"), receipt.get("custodian_status"))
    require(all(status in ("clear", "collision", "unknown") for status in statuses),
            "invalid scope status")
    if any(status != "clear" for status in statuses):
        require(bool(receipt["reason"].strip()), "missing scope failure reason")
        return False
    return True


def select_slots(candidates, prefiltered, reviews, cases, scope, provenance):
    """Select the first compatible word tuples without consulting model data.

    `cases[(word,subject,instruction)]` are re-tokenized records. `scope` binds
    each row's input-ID hash to independently produced prefix/custodian
    receipts; non-clear status is retained as ineligibility, not a model result.
    The caller is responsible for verifying all referenced public-source
    bytes and the scope receipt before invoking this pure selection function.
    """
    inventory = _candidate_index(candidates)
    rows = {row["word"]: row for row in prefiltered}
    require(set(rows) == set(inventory), "prefilter inventory mismatch")
    handoff = discovery_handoff(prefiltered)
    checked_reviews = validate_reviews(handoff, reviews)
    require(isinstance(cases, dict) and isinstance(scope, dict), "cases/scope mapping")
    validate_provenance(provenance)
    requested = {(item["word"], subject, instruction) for item in handoff["items"]
                 for subject in SUBJECTS for instruction in (0, 1)}
    require(set(cases).issubset(requested) and set(scope).issubset(requested),
            "unexpected discovery case or scope row")
    eligible = []
    dispositions = []
    checked = {}
    for word, identity in inventory.items():
        reasons = list(rows[word]["reasons"])
        attempts = []
        if not reasons:
            for subject in SUBJECTS:
                if not _review_accepts(checked_reviews, word, subject, identity["polarity"]):
                    reasons.append(f"semantic:{subject}")
                for instruction in (0, 1):
                    key = (word, subject, instruction)
                    attempts.append({"subject": subject, "instruction": instruction,
                                     "case": cases.get(key), "scope_receipt": scope.get(key)})
                    try:
                        case = inspect_case(cases.get(key), word, subject, instruction)
                        pilot_index = 2 * SUBJECTS.index(subject) + instruction
                        saved = rows[word]["pilot_row"]["templates"][pilot_index]
                        for field in ("text", "input_ids", "attention_mask",
                                      "special_tokens_mask", "offset_mapping",
                                      "adjective_char_span"):
                            require(case[field] == saved.get(field), f"pilot replay mismatch: {field}")
                        receipt = scope.get(key)
                        if not _check_receipt(receipt, case["input_ids_sha256"], provenance):
                            reasons.append(f"scope:{subject}:{instruction}:{receipt['reason']}")
                        checked[key] = case
                    except SelectionError as exc:
                        reasons.append(f"case:{subject}:{instruction}:{exc}")
        if not reasons:
            eligible.append(word)
        dispositions.append({"word": word, "polarity": identity["polarity"],
                             "index": identity["index"], "reasons": reasons,
                             "pilot_structural_rejections": rows[word]["pilot_row"]["structural_rejections"],
                             "pilot_public_overlap_reasons": rows[word]["pilot_row"]["public_overlap_reasons"],
                             "pilot_signature_sha256": rows[word]["pilot_row"]["signature_sha256"],
                             "attempted_rows": attempts})

    used = set()
    slots = []
    for slot_index, stratum in enumerate(SLOTS):
        selected = None
        positives = [w for w in candidates["positive"] if w in eligible and w not in used]
        negatives = [w for w in candidates["negative"] if w in eligible and w not in used]
        for pp in itertools.combinations(positives, 2):
            for nn in itertools.combinations(negatives, 2):
                words = {"P1": pp[0], "N1": nn[0], "P2": pp[1], "N2": nn[1]}
                groups = [[checked[(words[role], subject, instruction)] for role in ROLES]
                          for subject in SUBJECTS for instruction in (0, 1)]
                if all(_group_compatible(group, stratum) for group in groups):
                    selected = {"slot": slot_index, "stratum": list(stratum),
                                "status": "selected", "words": words, "groups": groups}
                    break
            if selected:
                break
        if selected is None:
            selected = {"slot": slot_index, "stratum": list(stratum),
                        "status": "missing", "reason": "no-compatible-unused-tuple"}
        else:
            used.update(selected["words"].values())
        slots.append(selected)
    return {"schema": "flan-multisubtoken-discovery-selection-v1",
            "slots": slots, "candidate_dispositions": dispositions,
            "reviewer_records": reviews, "provenance": provenance,
            "scope": "cooperative row-bound receipts; not global exposure proof"}


def load_sealed_discovery(discovery_bytes, expected_sha256):
    """Bind painting work to a separately recorded discovery digest.

    Matching bytes do not themselves establish when or how the digest was
    recorded; the caller must retain the independent sealing observation.
    """
    require(isinstance(expected_sha256, str) and HEX256.fullmatch(expected_sha256) is not None,
            "sealed discovery digest format")
    require(sha256_bytes(discovery_bytes) == expected_sha256, "sealed discovery digest mismatch")
    discovery = parse_json(discovery_bytes)
    require(discovery_bytes == canonical(discovery) + b"\n" and
            discovery.get("schema") == "flan-multisubtoken-discovery-selection-v1",
            "sealed discovery encoding")
    validate_provenance(discovery.get("provenance"))
    slots = discovery.get("slots")
    require(isinstance(slots, list) and len(slots) == 4, "sealed discovery slots")
    used = set()
    for index, slot in enumerate(slots):
        require(isinstance(slot, dict) and slot.get("slot") == index and
                slot.get("stratum") == list(SLOTS[index]), "sealed discovery slot identity")
        if slot.get("status") == "selected":
            words = slot.get("words")
            require(isinstance(words, dict) and set(words) == set(ROLES), "sealed discovery roles")
            for word in words.values():
                _check_word(word)
                require(word not in used, "sealed discovery reused word")
                used.add(word)
        else:
            require(slot.get("status") == "missing", "sealed discovery slot status")
    return discovery


def assess_holdout(discovery_bytes, expected_discovery_sha256, reviews, cases, scope, provenance):
    """Validate painting only for selected slots 0 and 2, without replacement."""
    discovery = load_sealed_discovery(discovery_bytes, expected_discovery_sha256)
    validate_provenance(provenance)
    handoff = holdout_handoff(discovery)
    checked_reviews = validate_painting_reviews(discovery, handoff, reviews)
    require(isinstance(cases, dict) and isinstance(scope, dict), "holdout cases/scope mapping")
    requested = {(item["word"], "painting", instruction) for item in handoff["items"]
                 for instruction in (0, 1)}
    require(set(cases).issubset(requested) and set(scope).issubset(requested),
            "unexpected painting case or scope row")
    output = []
    for index in (0, 2):
        slot = discovery["slots"][index]
        if slot["status"] == "missing":
            output.append({"slot": index, "status": "missing", "reason": "discovery-slot-missing"})
            continue
        require(slot["status"] == "selected", "invalid discovery slot")
        reasons, groups, attempts = [], [], []
        for instruction in (0, 1):
            group = []
            for role in ROLES:
                word = slot["words"][role]
                expected = "positive" if role.startswith("P") else "negative"
                if not _review_accepts(checked_reviews, word, "painting", expected):
                    reasons.append(f"semantic:{role}")
                key = (word, "painting", instruction)
                attempts.append({"role": role, "instruction": instruction,
                                 "case": cases.get(key), "scope_receipt": scope.get(key)})
                try:
                    case = inspect_case(cases.get(key), word, "painting", instruction)
                    receipt = scope.get(key)
                    if not _check_receipt(receipt, case["input_ids_sha256"], provenance):
                        reasons.append(f"scope:{role}:{instruction}:{receipt['reason']}")
                    group.append(case)
                except SelectionError as exc:
                    reasons.append(f"case:{role}:{instruction}:{exc}")
            if len(group) == 4 and not _group_compatible(group, SLOTS[index]):
                reasons.append(f"stratum-or-signature:{instruction}")
            groups.append(group)
        output.append({"slot": index, "status": "ineligible" if reasons else "eligible",
                       "reasons": sorted(set(reasons)), "groups": groups,
                       "words": slot["words"], "attempted_rows": attempts})
    return {"schema": "flan-multisubtoken-painting-selection-v1",
            "discovery_sha256": sha256_bytes(discovery_bytes), "slots": output,
            "reviewer_records": reviews, "provenance": provenance,
            "scope": "cooperative row-bound receipts; not global exposure proof"}
