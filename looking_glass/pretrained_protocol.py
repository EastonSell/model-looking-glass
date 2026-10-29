"""Seal the public/private records for the DistilGPT2 causal protocol v1.

This does not run a model, validate tokenizer mappings, establish access control,
or prove when a manifest was created. The reveal must remain with the custodian
until discovery predictions are frozen.
"""
from datetime import datetime
import hashlib
import json


PROTOCOL_ID = "distilgpt2-injected-change-v1"
MODEL_ID = "distilbert/distilgpt2"
REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
SEED = 20261028
ELIGIBLE_SITES = (1, 2, 3, 4)
ASSETS = {
    "config.json": "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a",
    "generation_config.json": "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
}
DISCOVERY_PROMPTS = (
    "The museum opened early because",
    "After the storm, the road",
    "The chef tasted the soup and",
    "A visitor asked for directions to",
    "The quiet library was filled with",
    "A train arrived at the platform and",
    "For tomorrow's lesson, the teacher",
    "The winter sky looked",
)
_SUBJECTS = (
    "The artist", "The gardener", "A sailor", "The baker", "A dancer", "My neighbor",
    "The mechanic", "A student", "The painter", "The librarian", "A musician", "The captain",
    "A photographer", "The tailor", "The cyclist", "The farmer",
)
_ACTIONS = (
    "found a ticket", "opened a box", "wrote a letter", "carried a basket", "read a note",
    "checked a map", "washed a cup", "held a lantern", "packed a bag", "shared a story",
    "closed a window", "moved a chair", "drew a picture", "fixed a clock", "folded a blanket",
    "painted a sign",
)
_CONTEXTS = (
    "before noon.", "after breakfast.", "near the river.", "beside the door.", "at the station.",
    "during lunch.", "at the market.", "outside the house.", "under the tree.", "in the hallway.",
    "before sunset.", "on the porch.", "at the corner.", "in the garden.", "near the bridge.",
    "after the meeting.",
)
_RUNTIME_KEYS = {"python", "torch", "transformers", "device", "driver", "deterministic_settings"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def canonical_sha256(value):
    """Hash sorted, compact UTF-8 JSON; no trailing newline is included."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _nonce(value, name):
    if type(value) is not bytes or len(value) != 32:
        raise ValueError(f"{name} must be exactly 32 bytes")
    return value


def _digest(value, length, name):
    if (type(value) is not str or len(value) != length or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def heldout_prompts(split_nonce):
    """Derive eight unique prompts; the nonce must stay private until reveal."""
    _nonce(split_nonce, "split_nonce")
    prompts = []
    for counter in range(1024):
        digest = hashlib.sha256(b"MLG-v1-heldout\0" + split_nonce + counter.to_bytes(8, "big")).digest()
        text = f"{_SUBJECTS[digest[0] % 16]} {_ACTIONS[digest[1] % 16]} {_CONTEXTS[digest[2] % 16]}"
        if text not in prompts and text not in DISCOVERY_PROMPTS:
            prompts.append(text)
            if len(prompts) == 8:
                return prompts
    raise ValueError("could not produce eight distinct held-out prompts")


def _token_ids(value, name):
    if type(value) is not list or len(value) != 8:
        raise ValueError(f"{name} must contain eight token-id lists")
    result = []
    for ids in value:
        if (type(ids) is not list or not 1 <= len(ids) <= 32 or
                any(type(token) is not int or not 0 <= token < 50257 for token in ids)):
            raise ValueError(f"{name} has invalid token ids")
        result.append(list(ids))
    return result


def _runtime(value):
    if (type(value) is not dict or set(value) != _RUNTIME_KEYS or
            any(type(item) is not str or not item.strip() for item in value.values())):
        raise ValueError("runtime must contain the six nonempty version/device fields")
    return dict(value)


def seal_manifest(discovery_token_ids, heldout_token_ids, *, site, site_nonce, split_nonce,
                  source_commit, protocol_sha256, runtime, created_utc):
    """Return public and private records without writing either to disk.

    The caller must independently tokenize with the pinned tokenizer and verify
    those mappings and all asset hashes; this helper only validates their shape.
    `created_utc` is caller-reported and is not a trusted timestamp.
    """
    if type(site) is not int or site not in ELIGIBLE_SITES:
        raise ValueError("site must be one of the four eligible layers")
    _nonce(site_nonce, "site_nonce")
    _nonce(split_nonce, "split_nonce")
    if site_nonce == split_nonce:
        raise ValueError("site and split nonces must be distinct")
    _digest(source_commit, 40, "source_commit")
    _digest(protocol_sha256, 64, "protocol_sha256")
    runtime = _runtime(runtime)
    if type(created_utc) is not str:
        raise ValueError("created_utc must be an ISO UTC timestamp")
    try:
        parsed = datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("created_utc must be YYYY-MM-DDTHH:MM:SSZ") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != created_utc:
        raise ValueError("created_utc must be YYYY-MM-DDTHH:MM:SSZ")
    discovery_ids = _token_ids(discovery_token_ids, "discovery_token_ids")
    heldout_ids = _token_ids(heldout_token_ids, "heldout_token_ids")
    if len({tuple(row) for row in discovery_ids + heldout_ids}) != 16:
        raise ValueError("discovery and held-out token sequences must be distinct")
    heldout = heldout_prompts(split_nonce)
    discovery_rows = [{"text": text, "token_ids": ids} for text, ids in zip(DISCOVERY_PROMPTS, discovery_ids)]
    heldout_rows = [{"text": text, "token_ids": ids} for text, ids in zip(heldout, heldout_ids)]
    heldout_commitment = hashlib.sha256(b"MLG-v1-heldout-data\0" + split_nonce + _canonical(heldout_rows)).hexdigest()
    random_layer = 1 + hashlib.sha256(b"MLG-v1-random-baseline\0" + SEED.to_bytes(8, "big")).digest()[0] % 4
    public = {
        "protocol": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "source_commit": source_commit,
        "created_utc_self_reported": created_utc,
        "model": {"id": MODEL_ID, "revision": REVISION, "sha256": dict(ASSETS)},
        "runtime": runtime,
        "seed": SEED,
        "discovery": discovery_rows,
        "discovery_data_sha256": canonical_sha256(discovery_rows),
        "heldout_count": 8,
        "heldout_data_commitment": heldout_commitment,
        "injection": {"eligible_layers": list(ELIGIBLE_SITES), "module": "mlp_output",
                      "token": "last_input_token", "invocation": 0, "delta_l2_nominal": 0.5},
        "metric": {"logit_token_ids": [262, 257], "absolute_tolerance": 1e-5,
                   "relative_tolerance": 1e-5},
        "random_baseline_layer": random_layer,
    }
    context = bytes.fromhex(canonical_sha256(public))
    public["site_commitment"] = hashlib.sha256(b"MLG-v1-site\0" + site_nonce + context + bytes([site])).hexdigest()
    reveal = {"site": site, "site_nonce": site_nonce.hex(), "split_nonce": split_nonce.hex(),
              "heldout_prompts": heldout, "heldout_token_ids": heldout_ids}
    return public, reveal


def verify_reveal(public, reveal):
    """Return whether a reveal regenerates the entire canonical public record."""
    try:
        if type(public) is not dict or type(reveal) is not dict:
            return False
        expected, expected_reveal = seal_manifest(
            [row["token_ids"] for row in public["discovery"]], reveal["heldout_token_ids"],
            site=reveal["site"], site_nonce=bytes.fromhex(reveal["site_nonce"]),
            split_nonce=bytes.fromhex(reveal["split_nonce"]), source_commit=public["source_commit"],
            protocol_sha256=public["protocol_sha256"], runtime=public["runtime"],
            created_utc=public["created_utc_self_reported"])
        return _canonical(public) == _canonical(expected) and _canonical(reveal) == _canonical(expected_reveal)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
