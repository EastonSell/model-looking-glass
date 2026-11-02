"""Draft independent FLAN saved-journal replay; freeze before real evidence.

Real mode requires independently pinned public source, manifests and direct
receipts. It does not load model weights or attest model execution, source
publication timing, selection chronology or pretraining exposure.
"""

from __future__ import annotations

import hashlib
import argparse
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import struct
import subprocess
import sys

CORE = Path(__file__).with_name("flan_multisubtoken_independent_core.py")
CORE_SHA256 = "e0dcf05c684a88ff07b3a3984ddc52d5d588f3edb523f02a6b1c09783cd63dcf"
EXECUTION_SOURCE_REVISION = "a5d959dbac8dab1176baebb346836a6f41459c47"
PROTOCOL_SHA256 = "bdb249673dc9045b8e83f4e5f525cc04e530f5f78f7a68fe72d26447e01fdb12"
ASSET_MANIFEST_SHA256 = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
SOURCE_PATHS = (
    "docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md", "docs/FLAN_T5_MULTISUBTOKEN_CPU_OWNER_V2.md",
    "looking_glass/__init__.py", "looking_glass/bert_matched_numeric.py",
    "looking_glass/flan_t5_feasibility.py", "looking_glass/flan_t5_sentiment_journal.py",
    "looking_glass/flan_t5_sentiment_numeric.py", "looking_glass/flan_t5_sentiment_runner.py",
    "looking_glass/flan_multitoken_geometry_v2.py", "looking_glass/flan_multitoken_journal_v2.py",
    "looking_glass/flan_multitoken_manifest_v2.py", "looking_glass/flan_multitoken_orthogonal_v2.py",
    "looking_glass/flan_multitoken_owner_v2.py", "looking_glass/flan_multitoken_runner_v2.py",
    "looking_glass/flan_multitoken_schedule_v2.py", "looking_glass/flan_multitoken_scoring_v2.py",
    "looking_glass/flan_multitoken_span_v2.py", "tools/run_flan_sentiment_selector_stage_v1.py",
    "tools/run_flan_multitoken_sentiment_v2.py",
)
ASSET_FILES = frozenset(("config.json", "generation_config.json", "model.safetensors",
                         "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"))
if hashlib.sha256(CORE.read_bytes()).hexdigest() != CORE_SHA256:
    raise ValueError("independent numeric core differs before import")
spec = importlib.util.spec_from_file_location("independent_flan_numeric", CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def fail(message):
    raise ValueError(message)


def strict_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                fail("duplicate JSON key")
            result[key] = value
        return result
    def bad(value):
        fail("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad)


def same(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if type(a) is float:
        return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


def ordinary_path(path, *, parents=True):
    """Reject symlinks and Windows reparse points before resolving a path."""
    raw = Path(path)
    if ".." in raw.parts:
        fail("parent traversal in evidence path")
    path = Path(os.path.abspath(raw))
    nodes = list(reversed((path, *path.parents))) if parents else [path]
    for node in nodes:
        try:
            info = os.lstat(node)
        except OSError as error:
            raise ValueError("required evidence path is unavailable") from error
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            fail("evidence path contains a link or reparse point")
    return path


def digest(path):
    h = hashlib.sha256()
    with ordinary_path(path, parents=False).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounded_file(path, limit=8 * 1024 * 1024):
    path = ordinary_path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        fail("required bounded input is absent, aliased or too large")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        fail("required bounded input changed or exceeds its limit")
    return raw


def canonical_object(raw):
    value = strict_json(raw)
    if (type(value) is not dict or raw != (json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")):
        fail("expected canonical JSON object")
    return value


def git_blob(repo, git, revision, path):
    """Read a bounded local committed blob without network or replace refs."""
    if (type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or type(path) is not str or not path or len(path.encode("utf8")) > 256
            or "\\" in path or ":" in path or "\0" in path
            or path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/"))):
        fail("invalid committed source descriptor")
    repo, git = ordinary_path(repo), ordinary_path(git)
    if not repo.is_absolute() or not repo.is_dir() or not git.is_absolute() or not git.is_file():
        fail("Git repository or executable is unavailable")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_REPLACE_OBJECTS="1",
               GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    def command(*args):
        try:
            return subprocess.run([str(git), "--no-replace-objects", "-C", str(repo), *args],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  check=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("local committed source is unavailable") from error
    if command("cat-file", "-t", revision).strip() != b"commit":
        fail("source revision is not a commit")
    descriptor = revision + ":" + path
    size = command("cat-file", "-s", descriptor).strip()
    if not size.isdigit() or int(size) > 8 * 1024 * 1024:
        fail("committed source exceeds its bound")
    raw = command("cat-file", "blob", descriptor)
    if len(raw) != int(size):
        fail("committed source size differs")
    return raw


def source_snapshot(repo, git, revision):
    sources = {path: hashlib.sha256(git_blob(repo, git, revision, path)).hexdigest() for path in SOURCE_PATHS}
    if sources[SOURCE_PATHS[0]] != PROTOCOL_SHA256:
        fail("fixed scientific protocol differs")
    return {"revision": revision, "source_sha256": sources}


def verify_reader_sources(repo, git, revision, reader_path, core_path, expected_reader_sha256):
    """Compare the running reader and its prechecked core to committed blobs.

    This is a consistency check; independent publication/readback before a
    study and trusted invocation remain separate observations.
    """
    if (type(expected_reader_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_reader_sha256) is None
            or reader_path == core_path):
        fail("independent reader identity is malformed")
    own, numeric = bounded_file(__file__), bounded_file(CORE)
    if (hashlib.sha256(own).hexdigest() != expected_reader_sha256
            or hashlib.sha256(numeric).hexdigest() != CORE_SHA256
            or git_blob(repo, git, revision, reader_path) != own
            or git_blob(repo, git, revision, core_path) != numeric):
        fail("independent reader source or committed bytes differ")
    return {"revision": revision, "reader_path": reader_path, "reader_sha256": expected_reader_sha256,
            "core_path": core_path, "core_sha256": CORE_SHA256}


def verify_bound_inputs(repo, git, revision, binding_raw, discovery_raw, painting_raw, review_raw, binding_sha256):
    if hashlib.sha256(binding_raw).hexdigest() != binding_sha256:
        fail("independently supplied binding digest differs")
    binding = canonical_object(binding_raw)
    keys = {"schema", "source_revision", "protocol_sha256", "asset_manifest_sha256",
            "discovery", "painting", "selection_review", "reader_sources"}
    if (set(binding) != keys or binding["schema"] != "flan-multisubtoken-owner-binding-v1"
            or binding["source_revision"] != revision or binding["protocol_sha256"] != PROTOCOL_SHA256
            or binding["asset_manifest_sha256"] != ASSET_MANIFEST_SHA256):
        fail("owner binding identity differs")
    def descriptor(value, expected=None):
        if (type(value) is not dict or set(value) != {"revision", "path", "sha256"}
                or type(value["sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
            fail("invalid public file descriptor")
        raw = git_blob(repo, git, value["revision"], value["path"])
        if hashlib.sha256(raw).hexdigest() != value["sha256"] or (expected is not None and raw != expected):
            fail("public committed bytes differ")
    for name, raw in (("discovery", discovery_raw), ("painting", painting_raw), ("selection_review", review_raw)):
        descriptor(binding[name], raw)
    review = canonical_object(review_raw)
    if (set(review) != {"schema", "status", "discovery_sha256", "painting_sha256", "scope"}
            or review["schema"] != "flan-multisubtoken-selection-review-v1" or review["status"] != "consistent"
            or review["discovery_sha256"] != hashlib.sha256(discovery_raw).hexdigest()
            or review["painting_sha256"] != hashlib.sha256(painting_raw).hexdigest()
            or type(review["scope"]) is not str or not 1 <= len(review["scope"].encode("utf8")) <= 4096):
        fail("selection review identity differs")
    readers = binding["reader_sources"]
    if type(readers) is not list or len(readers) > 8:
        fail("reader descriptor inventory differs")
    seen = set()
    for row in readers:
        descriptor(row)
        identity = (row["revision"], row["path"])
        if identity in seen:
            fail("duplicate reader descriptor")
        seen.add(identity)
    return {"source_revision": revision, "binding_sha256": binding_sha256,
            "discovery_sha256": hashlib.sha256(discovery_raw).hexdigest(),
            "painting_sha256": hashlib.sha256(painting_raw).hexdigest(),
            "selection_review_sha256": hashlib.sha256(review_raw).hexdigest(),
            "reader_sources": readers,
            "scope": "Local Git and saved-byte consistency only; publication and review are separate observations."}


def verify_assets(directory):
    root = ordinary_path(directory)
    raw = bounded_file(root / "asset-manifest.json")
    if hashlib.sha256(raw).hexdigest() != ASSET_MANIFEST_SHA256:
        fail("fixed FLAN asset manifest differs")
    manifest = strict_json(raw)
    if (type(manifest) is not dict or manifest.get("schema") != "mlg-flant5-small-assets-v1"
            or manifest.get("repo") != "google/flan-t5-small"
            or manifest.get("revision") != "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
            or type(manifest.get("files")) is not dict or set(manifest["files"]) != ASSET_FILES):
        fail("asset manifest contents differ")
    for name, reference in manifest["files"].items():
        path = ordinary_path(root / name, parents=False)
        if (type(reference) is not dict or type(reference.get("bytes")) is not int
                or type(reference.get("sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
                or path.is_symlink() or not path.is_file() or path.stat().st_size != reference["bytes"]
                or digest(path) != reference["sha256"]):
            fail("pinned FLAN asset differs: " + name)
    if digest(root / "asset-manifest.json") != ASSET_MANIFEST_SHA256:
        fail("asset manifest changed during verification")
    return manifest


class Evidence:
    def __init__(self, directory):
        self.root = ordinary_path(directory)
        if not self.root.is_dir():
            fail("evidence root is not a directory")
        for name in ("tensors", "vectors"):
            if not ordinary_path(self.root / name, parents=False).is_dir():
                fail("evidence payload directory is unavailable")
        self.used = set()
        self.identities = {}

    def read(self, ref, *, shape, dtype="float32-le"):
        if type(ref) is not dict or set(ref) != {"path", "shape", "bytes", "sha256", "dtype"}:
            fail("invalid evidence reference fields")
        prefix, suffix, itemsize = (("tensors", "f32le", 4) if dtype == "float32-le"
                                    else ("vectors", "f64le", 8))
        path = ref["path"]
        if (type(path) is not str or not re.fullmatch(prefix + r"/\d{6}\." + suffix, path)
                or ref["dtype"] != dtype or not same(ref["shape"], list(shape))
                or any(type(x) is not int or x <= 0 for x in shape)
                or type(ref["bytes"]) is not int or ref["bytes"] != itemsize * math.prod(shape)
                or type(ref["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
            fail("invalid evidence reference shape or identity")
        file = self.root / path
        ordinary_path(file.parent, parents=False)
        ordinary_path(file, parents=False)
        if (not file.is_file()
                or file.stat().st_size != ref["bytes"]):
            fail("missing or aliased evidence")
        with file.open("rb") as stream:
            raw = stream.read(ref["bytes"] + 1)
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            fail("saved evidence bytes differ")
        identity = (ref["bytes"], ref["sha256"])
        if path in self.identities and self.identities[path] != identity:
            fail("conflicting identities for one evidence path")
        self.identities[path] = identity
        self.used.add(path)
        return raw

    def verify_unchanged(self):
        """Rehash every referenced file after the scientific replay."""
        for path, (size, expected_sha256) in self.identities.items():
            file = self.root / path
            ordinary_path(file.parent, parents=False)
            ordinary_path(file, parents=False)
            if (not file.is_file()
                    or file.stat().st_size != size or digest(file) != expected_sha256):
                fail("saved evidence changed during replay")


class Events:
    def __init__(self, path):
        self.stream = ordinary_path(path).open("rb")
        self.count = 0
        self.components = 0

    def take(self, kind):
        line = self.stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024 or not line.endswith(b"\n"):
            fail("missing, oversized, or partial journal event")
        event = strict_json(line)
        if type(event) is not dict or set(event) != {"event", "detail"} or event["event"] != kind or type(event["detail"]) is not dict:
            fail("journal event order differs: expected " + kind)
        self.count += 1
        return event["detail"]

    def done(self):
        if self.stream.read(1):
            fail("extra journal events")
        self.stream.close()


def expected_fields(row, fields, where):
    if any(key not in row or not same(row[key], value) for key, value in fields.items()):
        fail(where + " differs")


def rows_of_state(raw, width, length):
    core.floats32(raw, width * length)
    return [raw[i * width * 4:(i + 1) * width * 4] for i in range(length)]


class ScientificReplay:
    def __init__(self, directory, width, vocab, torch, *, targets=None, special_ids=None, decode=None):
        self.evidence = Evidence(directory)
        self.events = Events(Path(directory) / "events.jsonl")
        self.width, self.vocab, self.torch = width, vocab, torch
        self.expected_numeric = None
        self.targets = targets if targets is not None else {"positive": [3, 1], "negative": [4, 1]}
        self.special = set(special_ids) if special_ids is not None else {0, 1, 2}
        self.decode = decode if decode is not None else self.toy_decode
        self.evaluations = 0
        self.observations = []
        self.skips = []
        self.geometry_records = []

    def numeric(self, value):
        if not same(value, self.expected_numeric):
            fail("runtime numeric probe differs")

    def component(self, kind, side, *, inputs=None, mask=None, phase=None, prefix=None):
        start = self.events.take("component_start")
        expected_fields(start, {"index": self.events.components, "kind": kind, "side": side}, "component start")
        self.numeric(start["numeric_before"])
        if kind == "encoder":
            expected_fields(start, {"input_ids": inputs, "attention_mask": mask}, "encoder input")
        else:
            expected_fields(start, {"phase": phase, "decoder_ids": prefix,
                                    "decoder_mask": [1] * len(prefix)}, "decoder input")
        end = self.events.take("component_return")
        expected_fields(end, {"index": self.events.components}, "component return")
        self.events.components += 1
        self.numeric(end["numeric_after"])
        if kind == "encoder":
            ref = end["state"]
            return ref, self.evidence.read(ref, shape=(1, len(inputs), self.width))
        expected_fields(end, {"phase": phase, "prefix": prefix}, "decoder return")
        ref = end["logits"]
        shape = (1, self.vocab) if phase == "greedy" else (1, len(prefix), self.vocab)
        return ref, self.evidence.read(ref, shape=shape)

    @staticmethod
    def toy_decode(ids):
        return "".join({3: "positive", 4: "negative"}.get(i, "<special>") for i in ids)

    def evaluation(self, case, required, before, after, mask):
        start = self.events.take("evaluation_start")
        action = start["action"]
        expected_fields(action, required, "evaluation action")
        expected_fields(start, {"side": required["receiver"], "input_ids": case["input_ids"],
                                "attention_mask": mask}, "evaluation source")
        shape = (1, len(case["input_ids"]), self.width)
        if self.evidence.read(start["before"], shape=shape) != before or self.evidence.read(start["after"], shape=shape) != after:
            fail("evaluation state differs")
        generated, greedy, logits_raw, refs, unique = [], [], [], [], []
        for _ in range(4):
            prefix = [0] + generated
            ref, raw = self.component("decoder", required["receiver"], phase="greedy", prefix=prefix)
            token, is_unique = core.greedy_step(raw, self.vocab)
            greedy.append({"prefix": prefix, "emitted": token, "logits": ref})
            refs.append(ref)
            logits_raw.append(raw)
            unique.append(is_unique)
            generated.append(token)
            if token == 1:
                break
        teacher, scores = {}, {}
        for label in ("positive", "negative"):
            target = self.targets[label]
            prefix = [0] + target[:-1]
            ref, raw = self.component("decoder", required["receiver"], phase="teacher_" + label, prefix=prefix)
            measured = core.teacher_score(raw, target, self.vocab, self.torch)
            refs.append(ref)
            logits_raw.append(raw)
            teacher[label] = (target, prefix, ref, measured)
            scores[label] = measured["score"]
        end = self.events.take("evaluation_return")
        expected_fields(end, {"action": action, "side": required["receiver"], "before": start["before"],
                              "after": start["after"], "greedy": greedy, "generated_ids": generated,
                              "all_logit_refs": refs}, "evaluation return")
        self.numeric(end["margin_numeric_before"])
        self.numeric(end["margin_numeric_after"])
        for label, (target, prefix, ref, measured) in teacher.items():
            row = end["teacher"][label]
            expected_fields(row, {"target_ids": target, "decoder_ids": prefix, "decoder_mask": [1] * len(prefix),
                                  "logits": ref, "selected_logprobs": list(measured["selected"]),
                                  "ranks": list(measured["ranks"]), "score": measured["score"]}, "teacher score")
            self.numeric(row["numeric_before"])
            self.numeric(row["numeric_after"])
        margin = scores["positive"] - scores["negative"]
        strict = core.strict_label(generated, "positive" if required["receiver"].startswith("P") else "negative",
                                   [True] * len(generated), self.special, self.decode)
        expected_fields(end, {"margin": margin, "strict_correct": strict}, "selected output")
        diagnostic = self.events.take("greedy_diagnostics")
        expected_fields(diagnostic, {"action": action, "side": required["receiver"],
                                    "ties": [not flag for flag in unique],
                                    "special_positions": [i for i, token in enumerate(generated) if token in self.special],
                                    "terminated": generated[-1] == 1, "truncated": generated[-1] != 1}, "greedy diagnostics")
        self.evaluations += 1
        summary = {"action": action, "generated_ids": generated, "unique_maxima": unique,
                   "strict_correct": strict, "tie_free_strict_correct": strict and all(unique),
                   "margin": margin, "first_step_tv": None,
                   "before_sha256": start["before"]["sha256"], "after_sha256": start["after"]["sha256"],
                   "teacher": {label: {"target_ids": entry[0], "decoder_ids": entry[1],
                                       "selected_logprobs": list(entry[3]["selected"]),
                                       "ranks": list(entry[3]["ranks"]), "score": entry[3]["score"]}
                               for label, entry in teacher.items()}}
        self.observations.append(summary)
        return {"action": action, "before": before, "after": after, "generated": generated,
                "unique": unique, "margin": margin, "strict": strict, "teacher": teacher,
                "logits_raw": logits_raw, "refs": refs, "summary": summary}

    def geometry(self, group, states, subject):
        unit, instruction = group["unit"], group["instruction"]
        cases = group["cases"]
        opposite = {"P1": "N1", "N1": "P1", "P2": "N2", "N2": "P2"}
        same_class = {"P1": "P2", "P2": "P1", "N1": "N2", "N2": "N1"}
        sites = {"opposite_adjective": "adjective", "same_adjective": "adjective",
                 "opposite_subject": "subject", "opposite_instruction": "instruction"}
        arms = tuple(sites) + ("orthogonal",)
        def position(receiver, site):
            case = cases[receiver]
            return 0 if site == "instruction" else case[site + "_index"]
        def row(receiver, site):
            case = cases[receiver]
            return rows_of_state(states[receiver], self.width, len(case["input_ids"]))[position(receiver, site)]
        distances = []
        for receiver in core.RECEIVERS:
            for arm, site in sites.items():
                donor = same_class[receiver] if arm == "same_adjective" else opposite[receiver]
                distances.append({"receiver": receiver, "arm": arm,
                                  "distance": core.donor_distance(row(receiver, site), row(donor, site))})
        q = min(r["distance"] for r in distances)
        record = self.events.take("group_geometry")
        self.numeric(record["numeric_before"])
        self.numeric(record["numeric_after"])
        status = "dose-ineligible" if q / 2 <= 1e-6 else "planned"
        expected_fields(record, {"unit": unit, "instruction": instruction, "subject": subject,
                                 "status": status, "q": q, "distances": distances}, "group geometry")
        self.geometry_records.append(record)
        if status == "dose-ineligible":
            expected_fields(record, {"orthogonal": [], "landings": []}, "ineligible geometry")
            return q, {}
        orthogonal = record["orthogonal"]
        landings = record["landings"]
        if len(orthogonal) != 4 or len(landings) != 40:
            fail("incomplete group geometry")
        result = {}
        index = 0
        subject_id = {"story": 0, "concert": 1, "painting": 2}[subject]
        for j, receiver in enumerate(core.RECEIVERS):
            donor = opposite[receiver]
            seed = 20261029 + 24 * unit + 8 * subject_id + 4 * instruction + j
            prepared = core.orthogonal_landing(row(receiver, "adjective"), row(donor, "adjective"), q / 2, seed)
            item = orthogonal[j]
            expected_fields(item, {"receiver": receiver, "seed": seed}, "orthogonal source identity")
            source = self.evidence.read(item["source_ref"], shape=(self.width,), dtype="float64-le")
            if source != prepared["source"]:
                fail("seeded Gaussian bytes differ")
            available = prepared.get("status") != "unavailable"
            if available:
                residual = self.evidence.read(item["residual_ref"], shape=(self.width,), dtype="float64-le")
                if residual != prepared["residual"]:
                    fail("orthogonal residual bytes differ")
                expected_fields(item, {"status": "ready", "failure": None,
                                       "coefficient": prepared["coefficient"],
                                       "residual_norm": prepared["residual_norm"],
                                       "donor_norm": prepared["donor_norm"]}, "orthogonal construction")
            else:
                expected_fields(item, {"status": "unavailable", "failure": prepared["failure"],
                                       "residual_ref": None, "coefficient": None,
                                       "residual_norm": None, "donor_norm": None},
                                "zero orthogonal direction")
            for dose, radius in (("low", q / 2), ("high", q)):
                candidates = {}
                for arm in arms:
                    saved = landings[index]
                    index += 1
                    site = "adjective" if arm == "orthogonal" else sites[arm]
                    target = None if arm == "orthogonal" else (same_class[receiver] if arm == "same_adjective" else donor)
                    expected_fields(saved, {"receiver": receiver, "dose": dose, "arm": arm,
                                            "site": site, "donor": target, "radius": radius}, "landing identity")
                    if arm == "orthogonal" and not available:
                        expected_fields(saved, {"status": "unavailable", "value_ref": None,
                                                "achieved_norm": None, "intended_cosine": None,
                                                "donor_cosine": None}, "unavailable orthogonal landing")
                        candidates[arm] = (saved, False, prepared["failure"])
                        continue
                    calculated = (core.orthogonal_landing(row(receiver, site), row(donor, site), radius, seed)
                                  if arm == "orthogonal" else
                                  core.donor_landing(row(receiver, site), row(target, site), radius))
                    raw = self.evidence.read(saved["value_ref"], shape=(self.width,))
                    if raw != calculated["bytes"]:
                        fail("planned rounded landing differs")
                    names = ("achieved_norm", "intended_cosine", "tolerance")
                    expected_fields(saved, {key: calculated[key] for key in names}, "landing measurement")
                    if arm == "orthogonal":
                        expected_fields(saved, {"donor_cosine": calculated["donor_cosine"],
                                               "coefficient": calculated["coefficient"],
                                               "residual_norm": calculated["residual_norm"]}, "orthogonal landing")
                    else:
                        expected_fields(saved, {"fraction": calculated["fraction"]}, "donor interpolation")
                    achieved, cos = calculated["achieved_norm"], calculated["intended_cosine"]
                    failure = None
                    if achieved == 0:
                        failure = "zero achieved displacement"
                    elif abs(achieved - radius) > calculated["tolerance"]:
                        failure = "landing norm outside tolerance"
                    elif cos is None or cos < 0.99:
                        failure = "landing cosine below threshold"
                    elif arm == "orthogonal" and (calculated["donor_cosine"] is None
                                                   or abs(calculated["donor_cosine"]) > 1e-4):
                        failure = "landing is not orthogonal within tolerance"
                    candidates[arm] = (saved, failure is None, failure)
                reference = candidates["opposite_adjective"]
                ref_valid, ref_norm = reference[1], reference[0]["achieved_norm"]
                for arm, (saved, valid, failure) in candidates.items():
                    if arm == "opposite_adjective":
                        relative = "reference" if ref_valid else "reference-unavailable"
                    elif not ref_valid:
                        relative = "reference-unavailable"
                    elif not valid:
                        relative = "not-applicable"
                    elif abs(saved["achieved_norm"] - ref_norm) > 0.01 * ref_norm:
                        valid, relative, failure = False, "failed", "control norm differs from reference"
                    else:
                        relative = "matched"
                    expected_fields(saved, {"status": "matchable" if valid else "unavailable",
                                            "failure": failure, "relative_status": relative},
                                    "landing disposition")
                    result[(receiver, dose, arm)] = saved
        return q, result

    def comparison(self, action, clean, observed):
        record = self.events.take("output_comparison")
        def teacher(output):
            return {label: {"score": entry[3]["score"], "ranks": list(entry[3]["ranks"])}
                    for label, entry in output["teacher"].items()}
        tv = core.first_step_tv(clean["logits_raw"][0], observed["logits_raw"][0], self.vocab)
        expected_fields(record, {"action": action,
                                 "clean_generated_ids": clean["generated"],
                                 "action_generated_ids": observed["generated"],
                                 "clean_margin": clean["margin"], "action_margin": observed["margin"],
                                 "clean_strict_correct": clean["strict"],
                                 "action_strict_correct": observed["strict"],
                                 "clean_teacher": teacher(clean), "action_teacher": teacher(observed),
                                 "clean_top_id": clean["generated"][0],
                                 "action_top_id": observed["generated"][0],
                                 "tv": tv},
                        "first-step output comparison")
        observed["summary"]["first_step_tv"] = tv
        self.numeric(record["numeric_before"])
        self.numeric(record["numeric_after"])

    def control(self, step, target, observed, expected_key):
        if observed["generated"] != target["generated"] or observed["logits_raw"] != target["logits_raw"]:
            fail("full-logit control does not match")
        record = self.events.take("control_verified")
        expected_fields(record, {"action": step, "expected": list(expected_key)}, "control verification")

    def group(self, group, subject):
        unit, instruction, cases = group["unit"], group["instruction"], group["cases"]
        if type(unit) is not int or type(instruction) is not int or set(cases) != set(core.RECEIVERS):
            fail("invalid group identity")
        baseline, states, state_refs = {}, {}, {}
        for receiver in core.RECEIVERS:
            case = cases[receiver]
            if (case["side"] != receiver or case["subject"] != subject or
                    len(case["input_ids"]) != len(case["attention_mask"]) or
                    case["attention_mask"] != [1] * len(case["input_ids"])):
                fail("case input or mask differs")
            ref, state = self.component("encoder", receiver, inputs=case["input_ids"], mask=case["attention_mask"])
            state_refs[receiver], states[receiver] = ref, state
            step = {"kind": "clean", "subject": subject, "unit": unit,
                    "instruction": instruction, "receiver": receiver}
            observed = self.evaluation(case, step, state, state, case["attention_mask"])
            expected_fields(observed["action"], {"encoder_state": ref}, "clean encoder reference")
            baseline[receiver] = observed
        q, landings = self.geometry(group, states, subject)
        support = all(baseline[r]["strict"] and all(baseline[r]["unique"]) and
                      (baseline[r]["margin"] > 0 if r.startswith("P") else baseline[r]["margin"] < 0)
                      for r in core.RECEIVERS)
        reported = self.events.take("baseline_support")
        expected_fields(reported, {"subject": subject, "unit": unit, "instruction": instruction,
                                   "supported": support}, "baseline support")
        opposite = {"P1": "N1", "N1": "P1", "P2": "N2", "N2": "P2"}
        same_class = {"P1": "P2", "P2": "P1", "N1": "N2", "N2": "N1"}
        arms = ("opposite_adjective", "same_adjective", "opposite_subject", "opposite_instruction", "orthogonal")
        measured = {dose: {} for dose in ("low", "high")}
        for receiver in core.RECEIVERS:
            case, state, clean = cases[receiver], states[receiver], baseline[receiver]
            prefix = {"subject": subject, "unit": unit, "instruction": instruction, "receiver": receiver}
            donor = opposite[receiver]
            for kind, site in (("passthrough", None), ("self", "adjective"), ("self", "subject"),
                               ("self", "instruction"), ("full_swap", None)):
                step = {**prefix, "kind": kind}
                if site is not None:
                    step["site"] = site
                if kind == "full_swap":
                    step["donor"] = donor
                    if case["attention_mask"] != cases[donor]["attention_mask"]:
                        fail("full-state donor mask differs")
                    after, mask, target, expected_key = states[donor], cases[donor]["attention_mask"], baseline[donor], (unit, instruction, donor)
                else:
                    after, mask, target, expected_key = state, case["attention_mask"], clean, (unit, instruction, receiver)
                observed = self.evaluation(case, step, state, after, mask)
                expected_fields(observed["action"], {"source_state": state_refs[receiver]}, "control source")
                if kind == "self":
                    expected_fields(observed["action"], {"position": 0 if site == "instruction" else case[site + "_index"]}, "self position")
                if kind == "full_swap":
                    expected_fields(observed["action"], {"donor_state": state_refs[donor], "source_mask_origin": "donor"}, "full-state source")
                self.control(step, target, observed, expected_key)
                self.comparison(observed["action"], clean, observed)
            for dose in ("low", "high"):
                for arm in arms:
                    donor_name = None if arm == "orthogonal" else (same_class[receiver] if arm == "same_adjective" else donor)
                    step = {**prefix, "kind": "matched", "dose": dose, "action": arm, "donor": donor_name}
                    landing = landings.get((receiver, dose, arm))
                    if landing is None or landing["status"] != "matchable":
                        skipped = self.events.take("action_skipped")
                        expected_fields(skipped, {"action": step,
                                                  "reason": "dose-ineligible" if landing is None else landing["failure"]}, "skipped action")
                        self.skips.append(skipped)
                        continue
                    site = landing["site"]
                    pos = 0 if site == "instruction" else case[site + "_index"]
                    value = self.evidence.read(landing["value_ref"], shape=(self.width,))
                    rows = rows_of_state(state, self.width, len(case["input_ids"]))
                    rows[pos] = value
                    observed = self.evaluation(case, step, state, b"".join(rows), case["attention_mask"])
                    extra = {key: value for key, value in landing.items() if key not in ("receiver", "dose", "arm")}
                    extra.update(position=pos, source_state=state_refs[receiver])
                    if donor_name is not None:
                        extra["donor_state"] = state_refs[donor_name]
                    expected_fields(observed["action"], extra, "matched landing action")
                    self.comparison(observed["action"], clean, observed)
                    if landing.get("relative_status") == ("reference" if arm == "opposite_adjective" else "matched"):
                        measured[dose].setdefault(arm, {})[receiver] = observed["margin"]
            natural = None
            offsets = case["offset_mapping"]
            begin, end = case["adjective_char_span"]
            lexical = [i for i, (a, b) in enumerate(offsets) if a < end and b > begin and b > a]
            for kind in ("natural_final", "natural_span"):
                step = {**prefix, "kind": kind, "donor": donor}
                positions = [case["adjective_index"]] if kind == "natural_final" else lexical
                own = rows_of_state(state, self.width, len(case["input_ids"]))
                other = rows_of_state(states[donor], self.width, len(case["input_ids"]))
                differences = []
                receiver_norm = None
                for pos in positions:
                    x, y = core.floats32(own[pos], self.width), core.floats32(other[pos], self.width)
                    if kind == "natural_final":
                        receiver_norm = core.norm64(x)
                    differences.extend(v - u for u, v in zip(x, y, strict=True))
                    own[pos] = other[pos]
                observed = self.evaluation(case, step, state, b"".join(own), case["attention_mask"])
                expected_fields(observed["action"], {"source_state": state_refs[receiver], "donor_state": state_refs[donor],
                                                      "positions": positions, "composite_dose": core.norm64(differences)}, "natural action")
                if kind == "natural_final":
                    amount = core.norm64(differences)
                    expected_fields(observed["action"], {"target_dose": amount, "achieved_norm": amount,
                                                        "cosine": core.cosine64(differences, differences),
                                                        "receiver_norm": receiver_norm,
                                                        "achieved_over_receiver_norm": None if receiver_norm == 0 else amount / receiver_norm,
                                                        "ratio_reason": "zero receiver norm" if receiver_norm == 0 else None},
                                    "natural final geometry")
                self.comparison(observed["action"], clean, observed)
                if kind == "natural_final":
                    natural = observed
                elif len(lexical) == 1:
                    if observed["after"] != natural["after"] or observed["generated"] != natural["generated"] or observed["logits_raw"] != natural["logits_raw"]:
                        fail("single-token natural anchors differ")
                    expected_fields(self.events.take("natural_anchor_verified"), {"action": step}, "natural anchor")
        if instruction == 0:
            receiver, case = "P1", cases["P1"]
            step = {"subject": subject, "unit": unit, "instruction": instruction, "receiver": receiver, "kind": "repeat"}
            ref, repeated = self.component("encoder", receiver, inputs=case["input_ids"], mask=case["attention_mask"])
            observed = self.evaluation(case, step, repeated, repeated, case["attention_mask"])
            expected_fields(observed["action"], {"source_state": state_refs[receiver], "encoder_state": ref}, "repeat source")
            self.control(step, baseline[receiver], observed, (unit, instruction, receiver))
            self.comparison(observed["action"], baseline[receiver], observed)
        outcomes = {}
        clean_margins = {r: baseline[r]["margin"] for r in core.RECEIVERS}
        for dose in ("low", "high"):
            rows = {arm: values for arm, values in measured[dose].items() if set(values) == set(core.RECEIVERS)}
            if not all(arm in rows for arm in arms[:-1]):
                outcomes[dose] = {"status": "unmeasured", "B3": None, "B4": None,
                                  "passes_B3": None, "passes_B4": None}
            else:
                result = core.group_rule(clean_margins, rows)
                outcomes[dose] = {"status": "measured" if result["B4"] is not None else "B3-only", **result}
        return {"unit": unit, "subject": subject, "instruction": instruction, "baseline_supported": support,
                "geometry_status": "dose-ineligible" if q / 2 <= 1e-6 else "planned", "doses": outcomes}


def validate_groups(groups, subject, vocab, *, special_ids=None, allow_empty=False):
    special_ids = {0, 1, 2} if special_ids is None else special_ids
    if (type(groups) is not list or (not groups and not allow_empty)
            or subject not in ("story", "concert", "painting")
            or type(special_ids) not in (set, frozenset)
            or not {0, 1} <= special_ids
            or any(type(i) is not int or not 0 <= i < vocab for i in special_ids)):
        fail("invalid fabricated group manifest")
    keys, slots = [], set()
    for group in groups:
        unit, instruction, cases = group["unit"], group["instruction"], group["cases"]
        if type(unit) is not int or unit not in range(4) or type(instruction) is not int or instruction not in (0, 1):
            fail("invalid unit or instruction")
        if subject == "painting" and unit not in (0, 2):
            fail("invalid painting slot")
        if type(cases) is not dict or set(cases) != set(core.RECEIVERS):
            fail("incorrect receiver identities")
        keys.append((unit, instruction))
        slots.add(unit)
        outside = None
        word_spans = set()
        for receiver in core.RECEIVERS:
            case = cases[receiver]
            ids, mask, offsets = case["input_ids"], case["attention_mask"], case["offset_mapping"]
            if (case["side"] != receiver or case["subject"] != subject or type(ids) is not list
                    or not 1 <= len(ids) <= 64 or any(type(v) is not int or not 0 <= v < vocab for v in ids)
                    or not same(mask, [1] * len(ids)) or type(offsets) is not list or len(offsets) != len(ids)
                    or type(case["special_tokens_mask"]) is not list or len(case["special_tokens_mask"]) != len(ids)
                    or any(type(v) is not int or v not in (0, 1) for v in case["special_tokens_mask"])
                    or any(type(pair) is not list or len(pair) != 2 or any(type(v) is not int for v in pair)
                           or not 0 <= pair[0] <= pair[1] <= len(case["text"]) for pair in offsets)
                    or ids[-1] != 1 or ids.count(1) != 1):
                fail("invalid case shape or mask")
            text, word = case["text"], case["adjective"]
            begin, end = case["adjective_char_span"]
            sbegin, send = case["subject_char_span"]
            if (text[begin:end] != word or begin < 1 or text[begin - 1] != " "
                    or text[sbegin:send] != subject):
                fail("case character spans differ")
            boundary = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > begin - 1]
            lexical = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > begin]
            subject_pos = [i for i, (a, b) in enumerate(offsets) if b > a and a < send and b > sbegin]
            required_count = 1 if unit < 2 else 2
            if (len(boundary) != required_count or len(lexical) != required_count or not subject_pos
                    or boundary != list(range(boundary[0], boundary[-1] + 1))
                    or offsets[boundary[0]][0] != begin - 1 or offsets[boundary[-1]][1] != end
                    or any(offsets[a][1] != offsets[b][0] for a, b in zip(boundary, boundary[1:]))
                    or lexical[-1] != case["adjective_index"] or subject_pos[-1] != case["subject_index"]
                    or len({0, lexical[-1], subject_pos[-1]}) != 3
                    or any(case["special_tokens_mask"][i] != 0 or ids[i] in special_ids
                           for i in set(boundary + subject_pos + [0]))):
                fail("adjective boundary or intervention sites differ")
            word_spans.add(tuple(ids[i] for i in boundary))
            context = (tuple(ids[:boundary[0]] + ids[boundary[-1] + 1:]), tuple(mask),
                       tuple(case["special_tokens_mask"]), tuple(boundary), tuple(lexical), tuple(subject_pos))
            if outside is None:
                outside = context
            elif context != outside:
                fail("group sequences outside word span differ")
        if len(word_spans) != 4:
            fail("word spans are not distinct")
    if keys != sorted(keys) or len(set(keys)) != len(keys) or set(keys) != {(u, i) for u in slots for i in (0, 1)}:
        fail("group order or instruction coverage differs")
    for unit in slots:
        both = [g for g in groups if g["unit"] == unit]
        if any(both[0]["cases"][r]["adjective"] != both[1]["cases"][r]["adjective"] for r in core.RECEIVERS):
            fail("unit words differ across instructions")
    return sorted(slots)


def project_bound_selection(discovery_raw, painting_raw, tokenizer, *, expected_discovery,
                            expected_painting, vocabulary, special_ids):
    """Independently reconstruct fixed case rows from canonical selected manifests.

    This checks input identities and fresh tokenization, not the selector's
    chronology, semantic review, pretraining exposure or model execution.
    """
    if (type(discovery_raw) is not bytes or type(painting_raw) is not bytes
            or not 1 <= len(discovery_raw) <= 8 * 1024 * 1024
            or not 1 <= len(painting_raw) <= 8 * 1024 * 1024
            or hashlib.sha256(discovery_raw).hexdigest() != expected_discovery
            or hashlib.sha256(painting_raw).hexdigest() != expected_painting
            or type(vocabulary) is not int or not 2 < vocabulary <= 32128
            or type(special_ids) not in (set, frozenset)
            or set(getattr(tokenizer, "all_special_ids", ())) != special_ids):
        fail("selected manifest or tokenizer identity differs")
    def manifest(raw):
        value = strict_json(raw)
        canonical = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
        if type(value) is not dict or raw != canonical:
            fail("selection manifest is not canonical")
        return value
    discovery, painting = manifest(discovery_raw), manifest(painting_raw)
    if (discovery.get("schema") != "flan-multisubtoken-discovery-selection-v1"
            or painting.get("schema") != "flan-multisubtoken-painting-selection-v1"
            or painting.get("discovery_sha256") != expected_discovery):
        fail("selection schemas or discovery binding differ")
    original, transfer = discovery.get("slots"), painting.get("slots")
    if (type(original) is not list or len(original) != 4 or type(transfer) is not list or len(transfer) != 2
            or any(type(row) is not dict or type(row.get("slot")) is not int or row["slot"] != i
                   for i, row in enumerate(original))
            or any(type(row) is not dict or type(row.get("slot")) is not int or row["slot"] != i
                   for i, row in zip((0, 2), transfer, strict=True))):
        fail("selection slot order differs")
    instructions = ("Classify the sentiment of this review as positive or negative:",
                    "Is this review positive or negative?")
    projected = {subject: [] for subject in ("story", "concert", "painting")}
    dispositions = {"discovery": {}, "painting": {}}
    selected, used = {}, set()

    def append_group(unit, subject, instruction, words, rows):
        if type(rows) is not list or len(rows) != 4:
            fail("selection group must have four rows")
        cases = {}
        for receiver, row in zip(core.RECEIVERS, rows, strict=True):
            word = words[receiver]
            text = instructions[instruction] + f" The {subject} was {word}."
            if (type(row) is not dict or len(text.encode("utf8")) > 256
                    or row.get("word") != word or row.get("subject") != subject
                    or type(row.get("instruction")) is not int or row["instruction"] != instruction
                    or row.get("text") != text):
                fail("selected row identity differs")
            fresh = tokenizer(text, add_special_tokens=True, return_attention_mask=True,
                              return_special_tokens_mask=True, return_offsets_mapping=True,
                              truncation=False)
            names = ("input_ids", "attention_mask", "special_tokens_mask", "offset_mapping")
            try:
                fresh = {key: list(fresh[key]) for key in names}
                fresh["offset_mapping"] = [list(pair) for pair in fresh["offset_mapping"]]
            except (KeyError, TypeError) as error:
                raise ValueError("fresh tokenizer row is malformed") from error
            if any(not same(row.get(key), fresh[key]) for key in names):
                fail("selected row does not match fresh tokenization")
            subject_begin = len(instructions[instruction] + " The ")
            word_begin = len(instructions[instruction] + f" The {subject} was ")
            subject_span = [subject_begin, subject_begin + len(subject)]
            word_span = [word_begin, word_begin + len(word)]
            if (not same(row.get("subject_char_span"), subject_span)
                    or not same(row.get("adjective_char_span"), word_span)
                    or type(row.get("subject_site")) is not int or type(row.get("adjective_site")) is not int):
                fail("selected row character or site identity differs")
            cases[receiver] = {"side": receiver, "text": text, "subject": subject, "adjective": word,
                               **{key: row[key] for key in names}, "subject_char_span": subject_span,
                               "adjective_char_span": word_span, "subject_index": row["subject_site"],
                               "adjective_index": row["adjective_site"]}
        projected[subject].append({"unit": unit, "instruction": instruction, "cases": cases})

    for unit, slot in enumerate(original):
        count = 1 if unit < 2 else 2
        if not same(slot.get("stratum"), [count, count]):
            fail("selection stratum differs")
        status = slot.get("status")
        dispositions["discovery"][str(unit)] = status
        if status == "missing":
            continue
        words, rows = slot.get("words"), slot.get("groups")
        if (status != "selected" or type(words) is not dict or set(words) != set(core.RECEIVERS)
                or any(type(word) is not str or re.fullmatch(r"[a-z]{1,24}", word) is None
                       for word in words.values())
                or len(set(words.values())) != 4 or used.intersection(words.values())
                or type(rows) is not list or len(rows) != 4):
            fail("selected discovery slot differs")
        selected[unit] = words
        used.update(words.values())
        for index, subject in enumerate(("story", "concert")):
            for instruction in (0, 1):
                append_group(unit, subject, instruction, words, rows[2 * index + instruction])

    for slot in transfer:
        unit, status = slot["slot"], slot.get("status")
        dispositions["painting"][str(unit)] = status
        if status == "missing" and unit not in selected:
            continue
        if unit not in selected or not same(slot.get("words"), selected[unit]):
            fail("painting words or discovery linkage differ")
        if status == "ineligible":
            continue
        rows = slot.get("groups")
        if status != "eligible" or type(rows) is not list or len(rows) != 2:
            fail("painting slot eligibility differs")
        for instruction in (0, 1):
            append_group(unit, "painting", instruction, selected[unit], rows[instruction])
    for subject, groups in projected.items():
        validate_groups(groups, subject, vocabulary, special_ids=special_ids, allow_empty=True)
    return {"groups": projected, "dispositions": dispositions}


def assemble_study_decisions(selection, reports):
    """Pool only independently verified pass summaries into the fixed slots.

    Callers must first verify the selected manifests, complete pass evidence,
    and direct-child observations. This reducer does not authenticate those
    observations or admit partial prefixes as complete scientific evidence.
    """
    subjects = ("story", "concert", "painting")
    if (type(selection) is not dict or type(selection.get("groups")) is not dict
            or set(selection["groups"]) != set(subjects)
            or type(reports) is not dict or set(reports) != set(subjects)):
        fail("study selection or pass set differs")
    verified, complete = {}, {}
    for subject in subjects:
        projected, report = selection["groups"][subject], reports[subject]
        if type(projected) is not list or type(report) is not dict or report.get("subject") != subject:
            fail("study pass identity differs")
        status, rows = report.get("status"), report.get("groups")
        if type(rows) is not list or status not in ("verified-complete", "no-selected-groups", "omitted-empty", "incomplete", "not-run"):
            fail("study pass disposition differs")
        def identity(row):
            if (type(row) is not dict or type(row.get("unit")) is not int
                    or type(row.get("instruction")) is not int):
                fail("group identity is malformed")
            return (row["unit"], row["instruction"])
        planned = [identity(row) for row in projected]
        allowed_units = (0, 2) if subject == "painting" else range(4)
        if (any(unit not in allowed_units or instruction not in (0, 1) for unit, instruction in planned)
                or planned != sorted(set(planned))
                or set(planned) != {(unit, i) for unit, _ in planned for i in (0, 1)}):
            fail("planned group identities differ")
        if status == "verified-complete":
            if not planned or [identity(row) for row in rows] != planned:
                fail("verified group coverage differs")
            for row in rows:
                doses = row.get("doses")
                high = doses.get("high") if type(doses) is dict else None
                if (row.get("subject") != subject or type(row.get("baseline_supported")) is not bool
                        or type(high) is not dict or high.get("status") not in ("measured", "B3-only", "unmeasured")
                        or (high["status"] == "measured" and type(high.get("passes_B4")) is not bool)
                        or (high["status"] != "measured" and high.get("passes_B4") is not None)):
                    fail("verified group outcome is malformed")
            verified[subject], complete[subject] = rows, True
        elif status == "no-selected-groups":
            if planned or rows:
                fail("empty pass contradicts selection")
            verified[subject], complete[subject] = [], True
        elif status == "omitted-empty":
            # The frozen wrapper stops after a directly observed empty story
            # pass. The other empty projections were deliberately not run.
            if (subject == "story" or planned or rows
                    or reports["story"].get("status") != "no-selected-groups"):
                fail("omitted empty pass lacks its observed story disposition")
            verified[subject], complete[subject] = [], True
        else:
            if rows:
                fail("unverified partial groups cannot enter the pooled result")
            verified[subject], complete[subject] = [], False

    discovery_units = {}
    for unit in range(4):
        rows = [row for subject in ("story", "concert") for row in verified[subject] if row["unit"] == unit]
        discovery_units[str(unit)] = core.classify_unit(rows, ("story", "concert"))
    painting_units = {}
    for unit in (0, 2):
        rows = [row for row in verified["painting"] if row["unit"] == unit]
        painting_units[str(unit)] = core.classify_unit(rows, ("painting",))
    discovery_interval = core.planned_interval([discovery_units[str(unit)]["status"] for unit in range(4)],
                                               4, 3, complete["story"] and complete["concert"])
    painting_interval = core.planned_interval([painting_units[str(unit)]["status"] for unit in (0, 2)],
                                              2, 2, complete["painting"])
    return {"discovery_units": discovery_units, "discovery": discovery_interval,
            "painting_units": painting_units, "painting": painting_interval,
            "pass_complete": complete,
            "scope": "Fixed-slot summary of verified saved records; not proof of execution or selection chronology."}


def check_owner_runtime(record, binding, torch):
    expected_environment = {"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE"}
    numeric_environment = {"MKL_CBWR": "COMPATIBLE", "ATEN_CPU_CAPABILITY": None,
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    expected_fields(record, {"python": "3.11.15", "torch": str(torch.__version__),
                             "numpy": importlib.metadata.version("numpy"),
                             "transformers": importlib.metadata.version("transformers"),
                             "tokenizers": importlib.metadata.version("tokenizers"),
                             "threads": 1, "interop_threads": 1, "deterministic": True,
                             "deterministic_warn_only": False, "float32_matmul_precision": "highest",
                             "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode("utf8")).hexdigest(),
                             "byte_order": "little", "python_float_rounds": 1,
                             "numeric_environment": numeric_environment,
                             "environment": expected_environment, "binding": binding}, "owner runtime contract")
    if (type(record.get("cpu_count")) is not int or record["cpu_count"] < 1
            or type(record.get("cpu_architecture")) is not str or not record["cpu_architecture"]):
        fail("owner CPU metadata differs")


def pass_runtime_settings(torch):
    return {"model_class": "transformers.models.t5.modeling_t5.T5ForConditionalGeneration",
            "python_version": "3.11.15", "torch_version": str(torch.__version__),
            "deterministic": True, "deterministic_warn_only": False, "default_dtype": "torch.float32",
            "float32_matmul_precision": "highest", "cpu_autocast": False,
            "attention_implementation": "eager", "config_use_cache": False,
            "decoder_start_id": 0, "eos_id": 1,
            "environment": {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}}


def verify_pass_receipt(directory, subject, revision, binding_sha256, expected_exit_sha256):
    """Check cooperative owner and directly waited child receipts and logs."""
    directory = ordinary_path(directory)
    raw_exit = bounded_file(directory / "operator-exit.json", 1024 * 1024)
    if hashlib.sha256(raw_exit).hexdigest() != expected_exit_sha256:
        fail("operator pass receipt differs")
    receipt = canonical_object(raw_exit)
    expected_fields(receipt, {"schema": "flan-multisubtoken-operator-exit-v1", "subject": subject,
                              "direct_child_exit_code": 0, "child_reaped": True, "timed_out": False,
                              "operator_exception_type": None, "terminal_consistent": True,
                              "child_prefix_empty_after": True, "log_drain_complete": True}, "operator pass receipt")
    owner_raw = bounded_file(directory / "owner-receipt.json", 1024 * 1024)
    if hashlib.sha256(owner_raw).hexdigest() != receipt.get("owner_receipt_sha256"):
        fail("owner receipt digest differs")
    owner = canonical_object(owner_raw)
    status = owner.get("status")
    if status not in ("complete_candidate", "no-selected-groups") or receipt.get("owner_status") != status:
        fail("owner and operator pass dispositions differ")
    expected_fields(owner, {"schema": "flan-multisubtoken-owner-receipt-v1", "subject": subject,
                            "source_revision": revision, "binding_sha256": binding_sha256,
                            "model_loaded": status == "complete_candidate"}, "owner pass receipt")
    logs = receipt.get("logs")
    if type(logs) is not dict or set(logs) != {"stdout", "stderr"}:
        fail("operator log inventory differs")
    for name in ("stdout", "stderr"):
        path, detail = ordinary_path(directory / (name + ".log"), parents=False), logs[name]
        if (not path.is_file() or path.stat().st_size > 64 * 1024
                or type(detail) is not dict):
            fail("operator log is unavailable")
        expected_fields(detail, {"retained_bytes": path.stat().st_size, "truncated": False,
                                 "error_type": None, "finished": True}, "operator log status")
        if receipt.get(name + "_sha256") != digest(path):
            fail("operator log hash differs")
    spawn = canonical_object(bounded_file(directory / "operator-spawn.json", 4096))
    if (type(spawn.get("pid")) is not int or spawn["pid"] <= 0
            or type(spawn.get("parent_pid")) is not int or spawn["parent_pid"] <= 0):
        fail("direct child identity differs")
    with (directory / "stdout.log").open("rb") as stream:
        first = stream.readline(4097)
    if not first.endswith(b"\n") or len(first) > 4096:
        fail("direct child identity log is absent")
    identity = strict_json(first)
    expected_fields(identity, {"schema": "flan-t5-sentiment-direct-child-v1",
                               "pid": spawn["pid"], "parent_pid": spawn["parent_pid"]}, "direct child identity")
    for key in ("executable_path_sha256", "prefix_path_sha256", "dependency_site_path_sha256", "venv_config_sha256"):
        if type(identity.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None:
            fail("direct child runtime identity shape differs")
    prefix = directory / "child-pycache"
    if not ordinary_path(prefix, parents=False).is_dir() or any(prefix.iterdir()):
        fail("child bytecode prefix is not empty")
    child = directory / "child"
    if status == "no-selected-groups":
        if child.exists() or receipt.get("terminal_sha256") is not None or receipt.get("events_sha256") is not None:
            fail("empty pass has unexpected model evidence")
    else:
        ordinary_path(child, parents=False)
        for name in ("terminal", "events"):
            path = ordinary_path(child / ("terminal.json" if name == "terminal" else "events.jsonl"), parents=False)
            if (not path.is_file() or owner.get(name + "_sha256") != digest(path)
                    or receipt.get(name + "_sha256") != owner[name + "_sha256"]):
                fail("owner evidence digest differs")
    if (digest(directory / "operator-exit.json") != expected_exit_sha256
            or digest(directory / "owner-receipt.json") != receipt["owner_receipt_sha256"]
            or any(digest(directory / (name + ".log")) != receipt[name + "_sha256"] for name in ("stdout", "stderr"))):
        fail("operator pass records changed during review")
    ordinary_path(directory)
    return {"subject": subject, "status": status, "owner": owner,
            "exit_sha256": expected_exit_sha256, "spawn": spawn, "identity": identity}


def initialize_numeric():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.set_flush_denormal(False) is not True:
        fail("cannot preserve float32 denormals")
    if (sys.byteorder != "little" or math.ldexp(1.0, -1074) * 2.0 != math.ldexp(1.0, -1073)
            or 1.0 + 2 ** -53 != 1.0
            or struct.unpack("<f", struct.pack("<I", 1))[0] != math.ldexp(1.0, -149)
            or struct.unpack("<I", struct.pack("<f", 1.0 + 2 ** -24))[0] != 0x3f800000
            or float(torch.tensor([math.ldexp(1.0, -149)], dtype=torch.float32)[0].item()) != math.ldexp(1.0, -149)):
        fail("numeric mode differs")
    return torch


def replay_payload(groups, directory, subject, width, vocab, torch, *, testing, targets=None,
                   special_ids=None, decode=None, owner_check=None, runtime_settings=None):
    """Replay one completed payload; caller independently binds its inputs."""
    directory = ordinary_path(directory)
    slots = validate_groups(groups, subject, vocab, special_ids=special_ids)
    end = strict_json(bounded_file(directory / "terminal.json", 4 * 1024 * 1024))
    reader = ScientificReplay(directory, width, vocab, torch, targets=targets, special_ids=special_ids, decode=decode)
    try:
        if owner_check is not None:
            owner_check(reader.events.take("owner_contract"))
        runtime = reader.events.take("runtime_observation")
        expected_fields(runtime, {"subject": subject, "testing_only": testing}, "runtime observation")
        settings = runtime["settings"]
        expected_fields(settings, {"device": "cpu", "hidden_width": width, "vocabulary": vocab,
                                   "num_threads": 1, "num_interop_threads": 1}, "runtime shape")
        if runtime_settings is not None:
            expected_fields(settings, runtime_settings, "fixed runtime")
        reader.expected_numeric = runtime["numeric"]
        expected_numeric = {"binary32_ties": [1065353216, 1065353218],
                            "float32_denormals": [2, 4194304, -2147483646, -2143289344],
                            "float64_denormals": [2, 2251799813685248, -9223372036854775806, -9221120237041090560]}
        if not same(reader.expected_numeric, expected_numeric):
            fail("saved numeric-mode probes differ")
        labels = reader.events.take("label_targets")
        expected_fields(labels, {"targets": reader.targets, "decoder_start_id": 0, "eos_id": 1}, "label tokenization")
        summaries = [reader.group(group, subject) for group in groups]
        for row in summaries:
            if not same(reader.events.take("group_outcome"), row):
                fail("reported group decision differs")
        reader.events.done()
    finally:
        reader.events.stream.close()
    expected_fields(end, {"status": "complete_candidate", "active_component": None,
                          "components": reader.events.components, "storage_accounting_complete": True,
                          "detail": {"subject": subject, "slots": slots, "groups": summaries, "testing_only": testing}},
                    "terminal")
    all_files = set()
    for root, dirs, files in os.walk(directory, topdown=True, followlinks=False):
        root = Path(root)
        for name in dirs:
            subdir = ordinary_path(root / name, parents=False)
            if subdir.relative_to(directory).as_posix() not in ("tensors", "vectors"):
                fail("unexpected saved evidence directory")
        for name in files:
            file = ordinary_path(root / name, parents=False)
            if not file.is_file():
                fail("unexpected non-file evidence")
            all_files.add(file.relative_to(directory).as_posix())
    expected_files = reader.evidence.used | {"started.json", "events.jsonl", "terminal.json"}
    if all_files != expected_files:
        fail("unexpected or unreferenced saved files")
    size = sum((directory / path).stat().st_size for path in all_files if path != "terminal.json")
    if not same(end["retained_bytes"], size):
        fail("saved-byte accounting differs")
    reader.evidence.verify_unchanged()
    ordinary_path(directory)
    return {"subject": subject, "slots": slots, "events": reader.events.count,
            "components": reader.events.components, "evaluations": reader.evaluations,
            "retained_bytes": size, "groups": summaries, "observations": reader.observations,
            "skips": reader.skips, "geometry": reader.geometry_records}


def replay_fabricated(manifest, directory, subject, width, vocab, expected_manifest, expected_events, expected_terminal):
    if width != 2 or vocab != 256:
        fail("this draft is limited to the fixed fabricated fixture")
    manifest, directory = ordinary_path(manifest), ordinary_path(directory)
    events, terminal = directory / "events.jsonl", directory / "terminal.json"
    if manifest.stat().st_size > 16 * 1024 * 1024 or terminal.stat().st_size > 4 * 1024 * 1024:
        fail("manifest or terminal exceeds the review bound")
    before = (digest(manifest), digest(events), digest(terminal))
    if before != (expected_manifest, expected_events, expected_terminal):
        fail("fixture input hash differs")
    groups = strict_json(manifest.read_bytes())
    output = replay_payload(groups, directory, subject, width, vocab, initialize_numeric(), testing=True)
    if (digest(manifest), digest(events), digest(terminal)) != before:
        fail("input changed during replay")
    if digest(CORE) != CORE_SHA256:
        fail("independent numeric core changed during replay")
    return {"status": "passed-fabricated-only", **output,
            "manifest_sha256": before[0], "events_sha256": before[1], "terminal_sha256": before[2],
            "scope": "fabricated saved-byte replay; not model execution, source identity or a completed study"}


def replay_real_candidate(*, repo, git, revision, asset_dir, binding_path, discovery_path, painting_path,
                          review_path, binding_sha256, attempt, outer_path, outer_sha256):
    """WIP entry point; do not call on real evidence before review and freeze."""
    paths = {"binding": Path(binding_path), "discovery": Path(discovery_path),
             "painting": Path(painting_path), "review": Path(review_path)}
    attempt, outer_path = ordinary_path(attempt), ordinary_path(outer_path)
    if (type(revision) is not str or revision != EXECUTION_SOURCE_REVISION
            or type(binding_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", binding_sha256) is None
            or type(outer_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", outer_sha256) is None):
        fail("external evidence pins are malformed")
    inputs = {name: bounded_file(path) for name, path in paths.items()}
    source = source_snapshot(repo, git, revision)
    binding = verify_bound_inputs(repo, git, revision, inputs["binding"], inputs["discovery"], inputs["painting"],
                                  inputs["review"], binding_sha256)
    verify_assets(asset_dir)
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MKL_CBWR="COMPATIBLE",
                      OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    torch = initialize_numeric()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    if platform.python_version() != "3.11.15" or torch.cuda.is_initialized():
        fail("independent CPU runtime differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True,
                                               use_fast=True, trust_remote_code=False)
    if (type(tokenizer).__name__ != "T5TokenizerFast" or tokenizer.pad_token_id != 0
            or tokenizer.eos_token_id != 1 or tokenizer.encode("positive", add_special_tokens=True) != [1465, 1]
            or tokenizer.encode("negative", add_special_tokens=True) != [2841, 1]):
        fail("fixed label tokenizer differs")
    special = set(tokenizer.all_special_ids)
    selection = project_bound_selection(inputs["discovery"], inputs["painting"], tokenizer,
        expected_discovery=binding["discovery_sha256"], expected_painting=binding["painting_sha256"],
        vocabulary=32128, special_ids=special)
    outer_raw = bounded_file(outer_path, 1024 * 1024)
    if hashlib.sha256(outer_raw).hexdigest() != outer_sha256:
        fail("direct outer observation digest differs")
    outer = canonical_object(outer_raw)
    if (set(outer) != {"schema", "source_revision", "binding_sha256", "exit_code", "directly_waited",
                      "operator_summary_sha256", "scope"}
            or outer["schema"] != "flan-multisubtoken-outer-observation-v1"
            or outer["source_revision"] != revision or outer["binding_sha256"] != binding_sha256
            or type(outer["directly_waited"]) is not bool
            or (outer["exit_code"] is not None and type(outer["exit_code"]) is not int)
            or type(outer["scope"]) is not str or not outer["scope"]):
        fail("outer observation identity differs")
    def recheck_bound_inputs():
        if (source_snapshot(repo, git, revision) != source or verify_assets(asset_dir) is None
                or any(bounded_file(path) != inputs[name] for name, path in paths.items())
                or digest(outer_path) != outer_sha256 or digest(CORE) != CORE_SHA256):
            fail("bound study inputs changed during replay")
    if not outer["directly_waited"] or outer["exit_code"] != 0:
        recheck_bound_inputs()
        return {"status": "incomplete", "reason": "outer exit is nonzero or not directly observed",
                "source_revision": revision, "binding_sha256": binding_sha256,
                "outer_sha256": outer_sha256, "selection_dispositions": selection["dispositions"],
                "scope": "No complete-study decision; partial evidence requires a separately scoped review."}
    summary_raw = bounded_file(attempt / "operator-summary.json", 1024 * 1024)
    if hashlib.sha256(summary_raw).hexdigest() != outer["operator_summary_sha256"]:
        fail("outer observation does not bind operator summary")
    summary = canonical_object(summary_raw)
    expected_fields(summary, {"schema": "flan-multisubtoken-operator-summary-v1",
                              "source_revision": revision, "binding_sha256": binding_sha256}, "operator summary")
    start_raw = bounded_file(attempt / "operator-start.json", 1024 * 1024)
    start = canonical_object(start_raw)
    expected_fields(start, {"schema": "flan-multisubtoken-operator-v1", "source": source,
                            "binding_sha256": binding_sha256,
                            "subjects": ["story", "concert", "painting"]}, "operator start")
    passes = summary.get("passes")
    expected_subjects = ["story"] if not selection["groups"]["story"] else ["story", "concert", "painting"]
    if (type(passes) is not list or len(passes) != len(expected_subjects)
            or [row.get("subject") if type(row) is dict else None for row in passes] != expected_subjects
            or summary.get("status") != ("no-selected-groups" if len(expected_subjects) == 1 else "complete_candidate")):
        fail("operator subject order or disposition differs")
    reports, details, child_identity = {}, {}, None
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for item, subject in zip(passes, expected_subjects, strict=True):
        directory = attempt / subject
        saved = verify_pass_receipt(directory, subject, revision, binding_sha256, item.get("operator_exit_sha256"))
        if item.get("status") != saved["status"]:
            fail("operator summary and pass disposition differ")
        fingerprint = {key: value for key, value in saved["identity"].items() if key not in ("pid", "parent_pid")}
        if child_identity is None:
            child_identity = (saved["spawn"]["parent_pid"], fingerprint)
        elif not same(child_identity, (saved["spawn"]["parent_pid"], fingerprint)):
            fail("direct children report different parent or runtime identities")
        groups = selection["groups"][subject]
        if saved["status"] == "no-selected-groups":
            if groups:
                fail("owner omitted selected groups")
            reports[subject] = {"subject": subject, "status": "no-selected-groups", "groups": []}
            details[subject] = {"status": "no-selected-groups", "operator_exit_sha256": saved["exit_sha256"]}
            continue
        if not groups:
            fail("owner ran an empty selection")
        child = directory / "child"
        child_start = canonical_object(bounded_file(child / "started.json", 4 * 1024 * 1024))
        expected_fields(child_start, {"schema": "flan-multisubtoken-owner-start-v1", "subject": subject,
                                      "source": source, "binding": binding,
                                      "asset_manifest_sha256": ASSET_MANIFEST_SHA256}, "child start")
        start_sha = digest(child / "started.json")
        events_sha, terminal_sha = saved["owner"]["events_sha256"], saved["owner"]["terminal_sha256"]
        if (child / "events.jsonl").stat().st_size > 64 * 1024**2:
            fail("per-pass metadata exceeds the frozen cap")
        payload = replay_payload(groups, child, subject, 512, 32128, torch, testing=False,
            targets={"positive": [1465, 1], "negative": [2841, 1]}, special_ids=special, decode=decode,
            owner_check=lambda record: check_owner_runtime(record, binding, torch),
            runtime_settings=pass_runtime_settings(torch))
        cap, forwards, evaluations = ((5 * 1024**3 // 4, 3516, 580) if subject != "painting"
                                       else (3 * 1024**3 // 4, 1758, 290))
        if (payload["retained_bytes"] + (child / "terminal.json").stat().st_size > cap
                or payload["components"] > forwards or payload["evaluations"] > evaluations):
            fail("saved pass exceeds the frozen resource cap")
        if (digest(child / "started.json") != start_sha or digest(child / "events.jsonl") != events_sha
                or digest(child / "terminal.json") != terminal_sha):
            fail("pass records changed during scientific replay")
        reports[subject] = {"subject": subject, "status": "verified-complete", "groups": payload["groups"]}
        details[subject] = {"status": "verified-complete", "operator_exit_sha256": saved["exit_sha256"],
                            "events_sha256": events_sha, "terminal_sha256": terminal_sha,
                            "started_sha256": start_sha, "scientific_replay": payload}
    if expected_subjects == ["story"]:
        for subject in ("concert", "painting"):
            reports[subject] = {"subject": subject, "status": "omitted-empty", "groups": []}
            details[subject] = {"status": "omitted-empty"}
    decisions = assemble_study_decisions(selection, reports)
    recheck_bound_inputs()
    if (digest(attempt / "operator-summary.json") != outer["operator_summary_sha256"]
            or (attempt / "operator-start.json").read_bytes() != start_raw):
        fail("bound study inputs changed during replay")
    return {"status": "verified-saved-candidate", "source_revision": revision, "source_sha256": source["source_sha256"],
            "binding_sha256": binding_sha256, "outer_sha256": outer_sha256,
            "selection_dispositions": selection["dispositions"], "decisions": decisions, "passes": details,
            "scope": "Independent saved-byte and numeric consistency only; not authentication of execution, selection chronology, publication or pretraining exposure."}


def write_verified_report(output, raw):
    if len(raw) > 64 * 1024 * 1024:
        fail("independent report exceeds its bound")
    with output.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short independent report write")
        stream.flush()
        os.fsync(stream.fileno())
    saved = bounded_file(output, 64 * 1024 * 1024)
    if saved != raw:
        fail("saved independent report differs on readback")
    return hashlib.sha256(saved).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--subject", choices=("story", "concert", "painting"))
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-events-sha256")
    parser.add_argument("--expected-terminal-sha256")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--real", action="store_true")
    for name in ("repo", "git", "asset-dir", "binding", "discovery", "painting", "selection-review",
                 "attempt", "outer-observation", "output", "expected-binding-sha256", "expected-outer-sha256",
                 "reader-revision", "reader-path", "core-path", "expected-reader-sha256"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.synthetic:
        if not all((args.manifest, args.directory, args.subject, args.expected_manifest_sha256,
                    args.expected_events_sha256, args.expected_terminal_sha256)):
            parser.error("synthetic mode needs a manifest, directory, subject and all three expected digests")
        result = replay_fabricated(args.manifest, args.directory, args.subject, 2, 256,
                                   args.expected_manifest_sha256, args.expected_events_sha256, args.expected_terminal_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    required = (args.repo, args.git, args.asset_dir, args.binding, args.discovery, args.painting,
                args.selection_review, args.attempt, args.outer_observation, args.output,
                args.expected_binding_sha256, args.expected_outer_sha256, args.reader_revision,
                args.reader_path, args.core_path, args.expected_reader_sha256)
    if not all(required) or args.manifest is not None or args.directory is not None:
        parser.error("real mode needs all named source, evidence, output and reader identity arguments")
    path_arguments = (args.repo, args.git, args.asset_dir, args.binding, args.discovery, args.painting,
                      args.selection_review, args.attempt, args.outer_observation, args.output)
    if any(not Path(value).is_absolute() for value in path_arguments):
        fail("real-mode filesystem paths must be absolute")
    output = Path(args.output)
    output_parent = ordinary_path(output.parent)
    protected = (ordinary_path(args.attempt), ordinary_path(args.repo), ordinary_path(args.asset_dir))
    if ((os.name == "nt" and output.drive.upper() != "E:") or output.exists()
            or not output_parent.is_dir() or any(output_parent.is_relative_to(root) for root in protected)):
        fail("real report must be a new E: file outside the evidence attempt")
    reader_identity = verify_reader_sources(args.repo, args.git, args.reader_revision, args.reader_path,
                                            args.core_path, args.expected_reader_sha256)
    result = replay_real_candidate(repo=args.repo, git=args.git, revision=EXECUTION_SOURCE_REVISION,
        asset_dir=args.asset_dir, binding_path=args.binding, discovery_path=args.discovery,
        painting_path=args.painting, review_path=args.selection_review,
        binding_sha256=args.expected_binding_sha256, attempt=args.attempt,
        outer_path=args.outer_observation, outer_sha256=args.expected_outer_sha256)
    if verify_reader_sources(args.repo, args.git, args.reader_revision, args.reader_path,
                             args.core_path, args.expected_reader_sha256) != reader_identity:
        fail("independent reader source changed during replay")
    result["independent_reader"] = reader_identity
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
    report_sha256 = write_verified_report(output, raw)
    print(json.dumps({"status": result["status"], "report_sha256": report_sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
