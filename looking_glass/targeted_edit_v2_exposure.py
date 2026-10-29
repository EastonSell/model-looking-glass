"""Check the fixed public v2 input reservations without model execution."""

from pathlib import Path

from . import pretrained_exposures as inventory_v1
from . import pretrained_exposures_v2 as inventory_v2
from . import pretrained_exposures_v3 as inventory_v3
from . import pretrained_exposures_v4 as inventory_v4
from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2


INVENTORY_COMMIT = "d77481d6eb605cdc863a9676d3320c0b59cc84cd"
INVENTORY_SHA256 = "6c065926d08cf08aa98d2c2725299d6bdc25e919246851c34bd9c7d63e8a4af1"
PREDECESSOR_SHA256 = "76baf0c8996bf2872a3a1b8fd0fe2734e15525ce0f7014e54811145cb98ce912"


def verify_public_reservations(repo):
    """Check the published v4 snapshot and the exact 13 public rows.

    The commit is a provenance label, not a checked Git ancestry or a claim
    that this is the latest snapshot. This cannot establish complete exposure
    history, prior pretraining, or a private-split collision check. The owner
    must separately document any applicable custodian result before a forward.
    """
    root = Path(repo).resolve()
    modules = ((inventory_v1, "pretrained_exposures.py"), (inventory_v2, "pretrained_exposures_v2.py"),
               (inventory_v3, "pretrained_exposures_v3.py"), (inventory_v4, "pretrained_exposures_v4.py"))
    for module, filename in modules:
        if Path(module.__file__).resolve() != root / "looking_glass" / filename:
            raise v1.StudyError("exposure inventory was imported from another checkout")
    try:
        if inventory_v3.inventory_digest() != PREDECESSOR_SHA256 or inventory_v4.inventory_digest() != INVENTORY_SHA256:
            raise v1.StudyError("published exposure inventory identity differs")
        previous = inventory_v3.inventory_payload()["records"]
        records = inventory_v4.inventory_payload()["records"]
        prefix = "planned-parameter-edit-followon-"
        reserved = {row["record_id"]: row for row in records if row["record_id"].startswith(prefix)}
        if len(reserved) != len(v2.EVALUATION_CASES):
            raise v1.StudyError("public follow-on reservations are incomplete")
        for index, (_name, role, _text, ids, _label) in enumerate(v2.EVALUATION_CASES, 1):
            row = reserved.get(f"{prefix}{role}-{index:02}")
            if row is None or row["category"] != "planned" or row["token_ids"] != list(ids):
                raise v1.StudyError("a public follow-on reservation differs")
        prior_full = [tuple(row["token_ids"]) for row in previous]
        forbidden = {row[:length] for row in prior_full for length in range(1, len(row) + 1)}
        for _name, _role, _text, ids, _label in v2.EVALUATION_CASES:
            if ids in forbidden or any(ids[:len(old)] == old for old in prior_full):
                raise v1.StudyError("a public follow-on row collides with a prior known row")
        checked = v2.check_split_rows()
    except (KeyError, TypeError, ValueError) as exc:
        raise v1.StudyError("invalid exposure inventory") from exc
    return {"inventory_commit_label": INVENTORY_COMMIT, "canonical_payload_sha256": INVENTORY_SHA256,
            "predecessor_sha256": PREDECESSOR_SHA256, "public_evaluation_rows": len(reserved),
            "combined_public_rows_checked": checked, "known_row_collisions": 0,
            "complete_history_certified": False, "private_split_check_performed": False}
