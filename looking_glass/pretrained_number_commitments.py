"""Pure commitment helpers for the DistilGPT2 number-study protocol.

No secrets are generated here. A trusted custodian must publish a complete
prediction freeze before creating its split nonce, and must retain failures.
"""
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import re

from . import pretrained_exposures_v3 as exposures


PROTOCOL_SHA256 = "95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee"
CHECKPOINT_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
NOUNS = ("chef/chefs", "pilot/pilots", "lawyer/lawyers", "student/students",
         "nurse/nurses", "musician/musicians", "driver/drivers", "singer/singers")
PLURAL_CONTEXTS = ("under the bridges", "outside the houses", "by the gardens", "inside the buildings")
SINGULAR_CONTEXTS = ("under the tree", "outside the station", "by the river", "inside the kitchen")
_NOUN_DOMAIN = b"MLG-number-v1-nouns"
_PLURAL_DOMAIN = b"MLG-number-v1-plural-contexts"
_SINGULAR_DOMAIN = b"MLG-number-v1-singular-contexts"
_FAMILY_DOMAIN = b"MLG-number-v1-family-order"
_SITE_DOMAIN = b"MLG-number-v1-site\x00"
_SITE_COMMIT_DOMAIN = b"MLG-number-v1-site-commit\x00"
_HELDOUT_COMMIT_DOMAIN = b"MLG-number-v1-heldout-commit\x00"


def _json_value(value, depth=0):
    if depth > 12:
        raise ValueError("commitment record exceeds its nesting bound")
    if type(value) is str:
        if len(value) > 65536:
            raise ValueError("commitment string exceeds its bound")
        return
    if type(value) is int:
        return
    if type(value) is list:
        if len(value) > 256:
            raise ValueError("commitment array exceeds its bound")
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict:
        if len(value) > 256 or any(type(key) is not str for key in value):
            raise ValueError("commitment object has invalid keys or size")
        for item in value.values():
            _json_value(item, depth + 1)
        return
    raise ValueError("commitment records contain only strings, integers, arrays and objects")


def canonical_json(value):
    """Exact protocol JSON: sorted keys, ASCII escapes, compact, no newline."""
    _json_value(value)
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    if len(data) > 1_000_000:
        raise ValueError("commitment record exceeds its byte bound")
    return data


def _decode_canonical(data):
    if type(data) is not bytes or len(data) > 1_000_000:
        raise ValueError("a bounded canonical byte record is required")
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid canonical byte record") from error
    if canonical_json(value) != data:
        raise ValueError("commitment record is not canonical JSON")
    return value


def _hex(value, length):
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError("digest or revision is not lowercase canonical hexadecimal")
    return value


def _nonce(value):
    if type(value) is not bytes or len(value) != 32:
        raise ValueError("nonce must be exactly 32 immutable bytes")
    return value


def _exact_keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("commitment record has the wrong schema")


def _aggregate_record(record):
    _exact_keys(record, ("schema", "protocol_sha256", "checkpoint_revision", "tracks"))
    if (record["schema"] != "mlg-number-prediction-freezes-v1" or
            record["protocol_sha256"] != PROTOCOL_SHA256 or
            record["checkpoint_revision"] != CHECKPOINT_REVISION):
        raise ValueError("prediction freeze has an incompatible protocol or checkpoint")
    tracks = record["tracks"]
    if type(tracks) is not dict or not tracks or not set(tracks).issubset({"natural", "injection"}):
        raise ValueError("prediction freeze has invalid participating tracks")
    for item in tracks.values():
        _exact_keys(item, ("source_commit", "runtime_sha256", "prediction_sha256"))
        _hex(item["source_commit"], 40)
        _hex(item["runtime_sha256"], 64)
        _hex(item["prediction_sha256"], 64)
    return record


@dataclass(frozen=True)
class PredictionFreeze:
    """Validated bytes and digest, not proof of publication or execution."""
    record_bytes: bytes = field(repr=False)
    sha256: str


def load_prediction_freeze(record_bytes, expected_sha256):
    """Validate already prepared aggregate bytes; publication is caller-owned."""
    _hex(expected_sha256, 64)
    record = _aggregate_record(_decode_canonical(record_bytes))
    digest = hashlib.sha256(record_bytes).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256):
        raise ValueError("prediction freeze digest does not match")
    return PredictionFreeze(record_bytes, digest)


def make_prediction_freeze(tracks):
    """Construct a canonical aggregate from trusted, externally published digests."""
    record = {"schema": "mlg-number-prediction-freezes-v1",
              "protocol_sha256": PROTOCOL_SHA256, "checkpoint_revision": CHECKPOINT_REVISION,
              "tracks": tracks}
    raw = canonical_json(record)
    _aggregate_record(_decode_canonical(raw))
    return load_prediction_freeze(raw, hashlib.sha256(raw).hexdigest())


def _freeze_record(freeze):
    if type(freeze) is not PredictionFreeze:
        raise ValueError("a validated prediction freeze is required")
    checked = load_prediction_freeze(freeze.record_bytes, freeze.sha256)
    return _aggregate_record(_decode_canonical(checked.record_bytes))


@dataclass(frozen=True)
class SealedRecord:
    """The commitment is public; the record remains private until reveal."""
    commitment_sha256: str
    record_bytes: bytes = field(repr=False)


def derive_site_layer(site_nonce):
    nonce = _nonce(site_nonce)
    return 1 + (hashlib.sha256(_SITE_DOMAIN + nonce).digest()[0] >> 6)


def _site_record(record, site_nonce):
    _exact_keys(record, ("schema", "protocol_sha256", "source_commit", "checkpoint_revision", "layer"))
    if (record["schema"] != "mlg-number-site-v1" or record["protocol_sha256"] != PROTOCOL_SHA256 or
            record["checkpoint_revision"] != CHECKPOINT_REVISION or type(record["layer"]) is not int or
            record["layer"] != derive_site_layer(site_nonce)):
        raise ValueError("site record does not match its nonce and protocol")
    _hex(record["source_commit"], 40)
    return record


def seal_site(site_nonce, source_commit):
    nonce = _nonce(site_nonce)
    _hex(source_commit, 40)
    record = {"schema": "mlg-number-site-v1", "protocol_sha256": PROTOCOL_SHA256,
              "source_commit": source_commit, "checkpoint_revision": CHECKPOINT_REVISION,
              "layer": derive_site_layer(nonce)}
    raw = canonical_json(_site_record(record, nonce))
    return SealedRecord(hashlib.sha256(_SITE_COMMIT_DOMAIN + nonce + b"\x00" + raw).hexdigest(), raw)


def verify_site_reveal(site_nonce, record_bytes, expected_commitment, *, expected_source_commit):
    nonce = _nonce(site_nonce)
    _hex(expected_commitment, 64)
    _hex(expected_source_commit, 40)
    record = _site_record(_decode_canonical(record_bytes), nonce)
    if record["source_commit"] != expected_source_commit:
        raise ValueError("site source revision does not match")
    actual = hashlib.sha256(_SITE_COMMIT_DOMAIN + nonce + b"\x00" + record_bytes).hexdigest()
    if not hmac.compare_digest(actual, expected_commitment):
        raise ValueError("site commitment does not match")
    return record["layer"]


def _sort_pool(domain, nonce, values):
    return sorted(values, key=lambda item: (
        hashlib.sha256(domain + b"\x00" + nonce + b"\x00" + item.encode("utf-8")).digest(),
        item.encode("utf-8")))


@dataclass(frozen=True)
class FamilyAssignment:
    family_index: int
    context: str = field(repr=False)
    subjects: tuple = field(repr=False)
    texts: tuple = field(repr=False)


def sample_families(freeze, split_nonce, *, site_nonce=None, site_record_bytes=None, site_commitment=None):
    """Derive private assignments only from an existing valid freeze and nonce.

    This does not generate the nonce or prove that the freeze was published.
    The trusted custodian must establish publication *before* creating it.
    """
    aggregate = _freeze_record(freeze)
    nonce = _nonce(split_nonce)
    _verify_site_custody(aggregate, nonce, site_nonce, site_record_bytes, site_commitment)
    nouns = _sort_pool(_NOUN_DOMAIN, nonce, NOUNS)
    plural = _sort_pool(_PLURAL_DOMAIN, nonce, PLURAL_CONTEXTS)[:2]
    singular = _sort_pool(_SINGULAR_DOMAIN, nonce, SINGULAR_CONTEXTS)[:2]
    contexts = _sort_pool(_FAMILY_DOMAIN, nonce, plural + singular)
    result = []
    for index, context in enumerate(contexts):
        first, second = (nouns[index * 2].split("/"), nouns[index * 2 + 1].split("/"))
        subjects = tuple(first + second)
        texts = tuple(f"The {subject} {context}" for subject in subjects)
        result.append(FamilyAssignment(index, context, subjects, texts))
    return tuple(result)


def _checked_cases(assignments, tokenized_cases):
    if type(tokenized_cases) is not list or len(tokenized_cases) != 4:
        raise ValueError("four tokenized families are required")
    families, all_rows = [], []
    for assignment, supplied in zip(assignments, tokenized_cases):
        if type(supplied) is not list or len(supplied) != 4:
            raise ValueError("four tokenized cases per family are required")
        cases, width, subject_index, reference = [], None, None, None
        for text, item in zip(assignment.texts, supplied):
            _exact_keys(item, ("text", "input_ids", "attention_mask", "subject_index"))
            ids, mask, index = item["input_ids"], item["attention_mask"], item["subject_index"]
            if (type(item["text"]) is not str or item["text"] != text or type(ids) is not list or
                    not 1 <= len(ids) <= 32 or any(type(x) is not int or not 0 <= x <= 50256 for x in ids) or
                    type(mask) is not list or len(mask) != len(ids) or
                    any(type(x) is not int or x != 1 for x in mask) or
                    type(index) is not int or not 0 < index < len(ids) - 1):
                raise ValueError("held-out case has invalid token metadata")
            if width is None:
                width, subject_index = len(ids), index
                reference = ids[:index] + ids[index + 1:]
            elif (len(ids) != width or index != subject_index or
                  ids[:index] + ids[index + 1:] != reference):
                raise ValueError("family token alignment does not match")
            cases.append({"text": text, "input_ids": list(ids),
                          "attention_mask": list(mask), "subject_index": index})
            all_rows.append(tuple(ids))
        if len({case["input_ids"][subject_index] for case in cases}) != 4:
            raise ValueError("family subject tokens are not unique")
        families.append({"family_index": assignment.family_index, "context": assignment.context,
                         "subjects": list(assignment.subjects), "cases": cases})
    if len(set(all_rows)) != 16:
        raise ValueError("held-out token rows contain duplicates")
    return families


def _verify_site_custody(aggregate, split_nonce, site_nonce, site_record_bytes, site_commitment):
    site_values = (site_nonce, site_record_bytes, site_commitment)
    if "injection" not in aggregate["tracks"]:
        if any(value is not None for value in site_values):
            raise ValueError("site custody is only valid for an injection track")
        return
    if any(value is None for value in site_values):
        raise ValueError("injection track requires its private site commitment")
    _nonce(site_nonce)
    if hmac.compare_digest(site_nonce, split_nonce):
        raise ValueError("site and split nonces must be distinct")
    verify_site_reveal(site_nonce, site_record_bytes, site_commitment,
                       expected_source_commit=aggregate["tracks"]["injection"]["source_commit"])


def seal_heldout(freeze, split_nonce, tokenized_cases, *, exposure_inventory_sha256,
                 site_nonce=None, site_record_bytes=None, site_commitment=None):
    """Seal tokenized cases after a freeze and the v3 known-exposure check.

    The caller must ensure input objects are quiescent during copying, verify
    actual tokenization/subject positions, and establish publication order.
    """
    aggregate = _freeze_record(freeze)
    _nonce(split_nonce)
    _verify_site_custody(aggregate, split_nonce, site_nonce, site_record_bytes, site_commitment)
    if type(exposure_inventory_sha256) is not str or exposure_inventory_sha256 != exposures.inventory_digest():
        raise ValueError("the checked exposure inventory digest does not match")
    # Freeze caller data before it is checked and before hashing. A cooperative
    # caller must not mutate it while this initial copy is in progress.
    snapshot = _decode_canonical(canonical_json(tokenized_cases))
    assignments = sample_families(freeze, split_nonce, site_nonce=site_nonce,
                                  site_record_bytes=site_record_bytes, site_commitment=site_commitment)
    families = _checked_cases(assignments, snapshot)
    discovery = [list(ids) for _, ids in exposures.PUBLIC_DISCOVERY]
    heldout = [case["input_ids"] for family in families for case in family["cases"]]
    exposures.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=exposure_inventory_sha256,
                                     expected_model_identity=exposures.TARGET)
    record = {"schema": "mlg-number-heldout-v1", "protocol_sha256": PROTOCOL_SHA256,
              "source_commits": {name: item["source_commit"] for name, item in aggregate["tracks"].items()},
              "checkpoint_revision": CHECKPOINT_REVISION, "prediction_freeze_sha256": freeze.sha256,
              "exposure_inventory_sha256": exposure_inventory_sha256, "families": families}
    raw = canonical_json(record)
    commitment = hashlib.sha256(_HELDOUT_COMMIT_DOMAIN + split_nonce + b"\x00" + raw).hexdigest()
    return SealedRecord(commitment, raw)


def verify_heldout_reveal(freeze, split_nonce, record_bytes, expected_commitment, *,
                          site_nonce=None, site_record_bytes=None, site_commitment=None):
    """Revalidate a disclosed record, its freeze, allocation and commitment."""
    aggregate = _freeze_record(freeze)
    _nonce(split_nonce)
    _hex(expected_commitment, 64)
    record = _decode_canonical(record_bytes)
    _exact_keys(record, ("schema", "protocol_sha256", "source_commits", "checkpoint_revision",
                         "prediction_freeze_sha256", "exposure_inventory_sha256", "families"))
    sources = {name: item["source_commit"] for name, item in aggregate["tracks"].items()}
    if (record["schema"] != "mlg-number-heldout-v1" or record["protocol_sha256"] != PROTOCOL_SHA256 or
            record["checkpoint_revision"] != CHECKPOINT_REVISION or record["source_commits"] != sources or
            record["prediction_freeze_sha256"] != freeze.sha256):
        raise ValueError("held-out record does not match its prediction freeze")
    families = record["families"]
    if type(families) is not list or len(families) != 4:
        raise ValueError("held-out record has invalid families")
    for family in families:
        _exact_keys(family, ("family_index", "context", "subjects", "cases"))
    expected = seal_heldout(freeze, split_nonce, [f["cases"] for f in families],
                            exposure_inventory_sha256=record["exposure_inventory_sha256"],
                            site_nonce=site_nonce, site_record_bytes=site_record_bytes,
                            site_commitment=site_commitment)
    if record_bytes != expected.record_bytes or not hmac.compare_digest(
            expected.commitment_sha256, expected_commitment):
        raise ValueError("held-out record or commitment does not match")
    return True
