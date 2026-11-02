"""Single-snapshot validation must preserve immutable inventory pins and isolation."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
from unittest import mock

import pytest


def module(version):
    suffix = "" if version == 1 else f"_v{version}"
    return importlib.import_module(f"looking_glass.pretrained_exposures{suffix}")


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("version", range(1, 46))
def test_every_version_preserves_canonical_pin(version):
    subject = module(version)
    assert Path(subject.__file__).resolve().parent == Path(__file__).resolve().parents[1] / "looking_glass"
    payload = subject.inventory_payload()
    assert digest(payload) == subject.INVENTORY_SHA256
    assert hashlib.sha256(subject.canonical_inventory_bytes()).hexdigest() == subject.INVENTORY_SHA256
    assert subject.inventory_digest() == subject.INVENTORY_SHA256


@pytest.mark.parametrize("version", range(3, 15))
def test_single_predecessor_and_no_shared_mutable_rows(version):
    subject, predecessor = module(version), module(version - 1)
    original = predecessor.inventory_payload()
    snapshot = copy.deepcopy(original)
    with mock.patch.object(predecessor, "inventory_payload", return_value=original) as build:
        result = subject.inventory_payload()
        build.assert_called_once_with()
    assert original == snapshot
    result["records"][0]["token_ids"].append(-123)
    assert original == snapshot
    assert subject.inventory_digest() == subject.INVENTORY_SHA256


@pytest.mark.parametrize("version", range(3, 15))
def test_changed_payload_rejected_even_when_separate_digest_claims_valid(version):
    subject, predecessor = module(version), module(version - 1)
    payload = predecessor.inventory_payload()
    payload["records"][0]["token_ids"].append(-123)
    with mock.patch.object(predecessor, "inventory_payload", return_value=payload), \
            mock.patch.object(predecessor, "inventory_digest", return_value=subject.PREDECESSOR_SHA256):
        with pytest.raises(ValueError, match="predecessor"):
            subject.inventory_payload()


@pytest.mark.parametrize("version", range(3, 15))
def test_predecessor_pin_mismatch_rejected(version):
    subject, predecessor = module(version), module(version - 1)
    with mock.patch.object(predecessor, "INVENTORY_SHA256", "0" * 64):
        with pytest.raises(ValueError, match="predecessor"):
            subject.inventory_payload()
