"""Frozen public DistilBERT agreement inputs and exact exposure comparison.

This module constructs token rows from a pinned, independently checked
WordPiece table. It does not load a model or prove its historical exposure.
"""
from dataclasses import dataclass
import hashlib
import json


SCHEMA = "distilbert-agreement-public-cases-v1"
MANIFEST_SHA256 = "20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5"
MODEL_ID = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
ASSET_SHA256 = {
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
PILOT_RESULT_SHA256 = "2602232c7b0d5382f578b455b83496721ce6a68ff4a9fef398cdb033862387e0"
PILOT_REPLAY_SHA256 = "d0c4acfec26b80fd8609b3a005e3970c73eff76f6686ddc30823d1df1f89076a"
PILOT_ROWS = frozenset({
    (101, 1996, 7500, 2379, 1996, 11500, 103, 3201, 1012, 102),
    (101, 1996, 6617, 2379, 1996, 11500, 103, 3201, 1012, 102),
    (101, 1996, 4405, 2379, 1996, 3548, 103, 3201, 1012, 102),
    (101, 1996, 8221, 2379, 1996, 3548, 103, 3201, 1012, 102),
})

# Each value is (singular surface, plural surface, singular ID, plural ID).
# The pinned tokenizer must independently confirm both the isolated words and
# complete prompts; these constants alone do not verify the runtime tokenizer.
LEXEMES = {
    "farmer": ("farmer", "farmers", 7500, 6617),
    "nurse": ("nurse", "nurses", 6821, 11500),
    "pilot": ("pilot", "pilots", 4405, 8221),
    "soldier": ("soldier", "soldiers", 5268, 3548),
    "driver": ("driver", "drivers", 4062, 6853),
    "singer": ("singer", "singers", 3220, 8453),
    "doctor": ("doctor", "doctors", 3460, 7435),
    "dancer": ("dancer", "dancers", 8033, 10487),
    "painter": ("painter", "painters", 5276, 12499),
    "artist": ("artist", "artists", 3063, 3324),
    "actor": ("actor", "actors", 3364, 5889),
    "guard": ("guard", "guards", 3457, 4932),
    "teacher": ("teacher", "teachers", 3836, 5089),
    "worker": ("worker", "workers", 7309, 3667),
    "chef": ("chef", "chefs", 10026, 27828),
    "clerk": ("clerk", "clerks", 7805, 26706),
    "student": ("student", "students", 3076, 2493),
    "player": ("player", "players", 2447, 2867),
    "lawyer": ("lawyer", "lawyers", 5160, 9559),
    "coach": ("coach", "coaches", 2873, 7850),
    "writer": ("writer", "writers", 3213, 4898),
    "sailor": ("sailor", "sailors", 11803, 11279),
    "leader": ("leader", "leaders", 3003, 4177),
    "cook": ("cook", "cooks", 5660, 26929),
}
FAMILIES = {
    "development": (("farmer", "nurse"), ("pilot", "soldier"),
                    ("driver", "singer"), ("doctor", "dancer")),
    "evaluation": (("painter", "artist"), ("actor", "guard"),
                   ("teacher", "worker"), ("chef", "clerk"),
                   ("student", "player"), ("lawyer", "coach"),
                   ("writer", "sailor"), ("leader", "cook")),
}
NUMBERS = ("singular", "plural")
OUTPUT_TOKENS = {"is": 2003, "are": 2024}


def _surface_and_id(lemma, number):
    singular, plural, singular_id, plural_id = LEXEMES[lemma]
    return (singular, singular_id) if number == "singular" else (plural, plural_id)


def _row_id(split, family, source, subject_number, distractor_number):
    return f"{split}-f{family:02}-{source}-{subject_number[0]}-{distractor_number[0]}"


def _row(split, family, source, subject, subject_number, distractor, distractor_number):
    subject_text, subject_id = _surface_and_id(subject, subject_number)
    distractor_text, distractor_id = _surface_and_id(distractor, distractor_number)
    context, context_id = ("near", 2379) if split == "development" else ("beside", 3875)
    ids = [101, 1996, subject_id, context_id, 1996, distractor_id, 103, 3201, 1012, 102]
    if split == "development":
        status = "observed-pilot-and-replay" if tuple(ids) in PILOT_ROWS else "reserved-development"
    else:
        status = "public-nonblind-evaluation"
    return {
        "row_id": _row_id(split, family, source, subject_number, distractor_number),
        "split": split,
        "family": family,
        "subject_source": source,
        "subject_lemma": subject,
        "subject_number": subject_number,
        "distractor_lemma": distractor,
        "distractor_number": distractor_number,
        "text": f"The {subject_text} {context} the {distractor_text} [MASK] ready.",
        "input_ids": ids,
        "attention_mask": [1] * 10,
        "exposure_status": status,
    }


def build_manifest():
    """Build a fresh public manifest; no tokenizer or model is loaded."""
    rows, targets = [], []
    for split, families in FAMILIES.items():
        for index, (subject, distractor) in enumerate(families):
            family = index + 1
            next_subject = families[(index + 1) % len(families)][0]
            for source, source_subject in (("A", subject), ("B", next_subject)):
                for subject_number in NUMBERS:
                    for distractor_number in NUMBERS:
                        rows.append(_row(split, family, source, source_subject, subject_number,
                                         distractor, distractor_number))
            for subject_number in NUMBERS:
                opposite = NUMBERS[1 - NUMBERS.index(subject_number)]
                for distractor_number in NUMBERS:
                    targets.append({
                        "case_id": f"{split}-f{family:02}-{subject_number[0]}-{distractor_number[0]}",
                        "split": split,
                        "family": family,
                        "target_row_id": _row_id(split, family, "A", subject_number, distractor_number),
                        "donor_row_ids": {
                            "A_same": _row_id(split, family, "A", subject_number, distractor_number),
                            "A_opposite": _row_id(split, family, "A", opposite, distractor_number),
                            "B_same": _row_id(split, family, "B", subject_number, distractor_number),
                            "B_opposite": _row_id(split, family, "B", opposite, distractor_number),
                        },
                    })
    return {
        "schema": SCHEMA,
        "scope": "public non-blind evaluation; unknown pretraining and historical exposure",
        "identity": {"model": MODEL_ID, "revision": MODEL_REVISION,
                     "asset_sha256": dict(ASSET_SHA256)},
        "coordinates": {"subject": 2, "distractor": 5, "mask": 6,
                        "sequence_length": 10, "output_token_ids": dict(OUTPUT_TOKENS)},
        "pilot_evidence": {"observed_unique_rows": 4, "original_result_sha256": PILOT_RESULT_SHA256,
                           "independent_replay_sha256": PILOT_REPLAY_SHA256},
        "rows": rows,
        "targets": targets,
    }


def canonical_manifest_bytes():
    """The v1 pinned encoding includes sorted keys, indentation, and one LF."""
    return (json.dumps(build_manifest(), sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def manifest_digest():
    digest = hashlib.sha256(canonical_manifest_bytes()).hexdigest()
    if digest != MANIFEST_SHA256:
        raise ValueError("public agreement manifest differs from its pinned digest")
    return digest


@dataclass(frozen=True)
class CaseCheck:
    manifest_sha256: str
    development_target_count: int
    evaluation_target_count: int
    development_row_count: int
    evaluation_row_count: int
    observed_pilot_count: int
    exact_token_overlap_count: int
    scope: str = "known-full-token-rows-only"
    historical_exposure_complete: bool = False
    execution_authorized: bool = False


def verify_manifest_bytes(data):
    """Check exact frozen content and development/evaluation row disjointness.

    The caller owns and keeps the bytes quiescent during this call. This is a
    cooperative content check, not runtime model or tokenizer authentication.
    """
    if type(data) is not bytes or len(data) > 1_000_000:
        raise ValueError("manifest must be bounded immutable bytes")
    digest = manifest_digest()
    if data != canonical_manifest_bytes():
        raise ValueError("agreement case manifest does not match the pinned version")
    manifest = build_manifest()
    development = [tuple(r["input_ids"]) for r in manifest["rows"] if r["split"] == "development"]
    evaluation = [tuple(r["input_ids"]) for r in manifest["rows"] if r["split"] == "evaluation"]
    if len(set(development)) != 32 or len(set(evaluation)) != 64:
        raise ValueError("case manifest contains duplicate input rows")
    overlap = len(set(development) & set(evaluation))
    if overlap or not PILOT_ROWS.issubset(set(development)):
        raise ValueError("case manifest overlaps known development inputs")
    return CaseCheck(digest, 16, 32, 32, 64, len(PILOT_ROWS), overlap)
