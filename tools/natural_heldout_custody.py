"""Custody-only orchestration for the frozen natural agreement study.

The bound sampler, runner and scorer remain in the discovery source checkout.
No nonce or model is created by importing this module.
"""
from dataclasses import dataclass
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request


SOURCE_COMMIT = 'def07cbb168408a12c76ba30f196af45149e111d'
BINDING_SHA256 = '23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0'
PREDICTION_SHA256 = 'eaac12d7229b2ae5615832bc9b58c7e5b2404c5866b5079295e72b0a54a4c545'
AGGREGATE_SHA256 = '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb'
V7_INVENTORY_SHA256 = 'e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344'
# Reviewed v12 reserves the public factorial inputs and preserves v11.
CURRENT_INVENTORY_VERSION = 12
CURRENT_INVENTORY_SHA256 = 'af374aaceb708b3b38e06b3955ca975467b048162ee105cbfcc9452a8a600e49'
CURRENT_INVENTORY_SOURCE_SHA256 = '6f8d3058fba976f21e70e519be563f61285b5035415373bafd3a496564acaac5'
CHECKPOINT = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
TARGETS = {' is': 318, ' are': 389, ' was': 373, ' were': 547}
LOGICAL_LIMIT = 160 * 1024 * 1024
DISK_LIMIT = 224 * 1024 * 1024
EVENT_LIMIT = 48 * 1024 * 1024
SIDE_LIMIT = 16 * 1024 * 1024  # includes both bounded supervisor logs
SOURCE_SUFFIXES = {
    'custody': 'tools/natural_heldout_custody.py',
    'deviation': 'docs/NATURAL_AGREEMENT_HELDOUT_DEVIATION.md',
    'child': 'tools/natural_agreement_heldout_child.py',
    'owner': 'tools/natural_agreement_heldout_owner.py',
    'supervisor': 'looking_glass/natural_heldout_supervisor.py',
    'ledger': 'looking_glass/agreement_phase_ledger.py',
    'auditor': 'tools/audit_natural_agreement_heldout.py',
}
REQUIRED_FREE_BYTES = 3 * 1024 * 1024 * 1024
BOOTSTRAP = ("import os,runpy,sys; sys.path.append(os.environ['MLG_NATURAL_SITE']); "
             "script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script,run_name='__main__')")
OBSERVER_ENTRY_BOOTSTRAP = ("import hashlib,sys\n"
                            "p=sys.argv.pop(1); h=sys.argv.pop(1); data=open(p,'rb').read(8388609)\n"
                            "if len(data)>8388608 or hashlib.sha256(data).hexdigest()!=h:\n"
                            "    raise SystemExit(1)\n"
                            "sys.argv[0]=p; exec(compile(data,p,'exec',dont_inherit=True),"
                            "{'__name__':'__main__','__file__':p})")
OUTER_BOOTSTRAP = '''import ctypes, msvcrt, os, runpy, sys, time
from ctypes import wintypes
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.PeekNamedPipe.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                  wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
kernel.PeekNamedPipe.restype = wintypes.BOOL
fd = sys.stdin.fileno()
handle = msvcrt.get_osfhandle(fd)
start = b"MLG-natural-discovery-start-v1\\n"
data = bytearray()
deadline = time.monotonic() + 10
while time.monotonic() < deadline and len(data) < len(start):
    available = wintypes.DWORD()
    if not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
        break
    if available.value:
        data.extend(os.read(fd, min(available.value, len(start) - len(data))))
    else:
        time.sleep(0.01)
if bytes(data) != start:
    raise SystemExit(1)
sys.path.append(os.environ["MLG_NATURAL_SITE"])
script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
'''


class CustodyError(RuntimeError):
    """A custody phase could not be completed safely."""


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(',', ':')).encode('utf-8')


def _write_new(path, data):
    with Path(path).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short custody write')
        stream.flush()
        os.fsync(stream.fileno())


def _checked_path(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise CustodyError('private evidence path must be absolute')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise CustodyError('private evidence path traverses a reparse point')
    return path


def _read_private(path, maximum):
    path = _checked_path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise CustodyError('private evidence is absent, nonregular or oversized')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) != info.st_size or len(raw) > maximum:
        raise CustodyError('private evidence changed while reading')
    return raw


def account_phase(attempt, *, reserve_side_bytes=0, extra_side_roots=()):
    """Count all apparent file lengths, including unknown and partial files.

    The journal and supervisor must enforce their separate write limits; this
    scan does not measure allocated filesystem clusters or control other users.
    """
    if type(reserve_side_bytes) is not int or not 0 <= reserve_side_bytes <= SIDE_LIMIT:
        raise CustodyError('invalid side-file reserve')
    root = _checked_path(attempt)
    if not root.is_dir():
        raise CustodyError('phase directory is invalid')
    if type(extra_side_roots) not in (tuple, list) or len(extra_side_roots) > 4:
        raise CustodyError('invalid additional side-file roots')
    extra_roots = [_checked_path(item) for item in extra_side_roots]
    roots = [root] + extra_roots
    if not root.is_dir() or any(not item.is_dir() and not item.is_file() for item in extra_roots):
        raise CustodyError('phase evidence root is not an ordinary file or directory')
    if any(a == b or a in b.parents or b in a.parents
           for index, a in enumerate(roots) for b in roots[index + 1:]):
        raise CustodyError('phase evidence roots overlap')
    sizes = {'tensor_bytes': 0, 'event_bytes': 0, 'side_bytes': 0}
    def enumeration_error(_error):
        raise CustodyError('phase evidence could not be completely enumerated')
    for evidence_root in roots:
        if evidence_root.is_file():
            info = evidence_root.lstat()
            if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise CustodyError('phase side evidence is not an ordinary file')
            sizes['side_bytes'] += info.st_size
            continue
        for directory, names, files in os.walk(evidence_root, followlinks=False, onerror=enumeration_error):
            for name in names:
                info = (Path(directory) / name).lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise CustodyError('phase evidence contains a linked directory')
            for name in files:
                path = Path(directory) / name
                info = path.lstat()
                if (not stat.S_ISREG(info.st_mode)
                        or getattr(info, 'st_file_attributes', 0) & 0x400):
                    raise CustodyError('phase evidence contains a nonregular file')
                rel = path.relative_to(root).as_posix() if evidence_root == root else ''
                category = ('tensor_bytes' if rel.startswith('owner/attempt/tensors/') else
                            'event_bytes' if rel == 'owner/attempt/events.jsonl' else 'side_bytes')
                sizes[category] += info.st_size
    if (sizes['tensor_bytes'] > LOGICAL_LIMIT or sizes['event_bytes'] > EVENT_LIMIT
            or sizes['side_bytes'] + reserve_side_bytes > SIDE_LIMIT
            or sum(sizes.values()) + reserve_side_bytes > DISK_LIMIT):
        raise CustodyError('phase evidence exceeds its reserved apparent-byte limits')
    return sizes


def inspect_cache_staging(cache):
    """Record visible staging files and point-in-time free space; delete nothing."""
    cache = _checked_path(cache)
    if not cache.is_dir():
        raise CustodyError('checkpoint cache is unavailable')
    free = shutil.disk_usage(cache).free
    if free < REQUIRED_FREE_BYTES:
        raise CustodyError('checkpoint cache filesystem has insufficient free space')
    stages = []
    for child in sorted(cache.iterdir(), key=lambda item: item.name):
        if not child.name.startswith(('looking-glass-stage-', 'agreement-tokenizer-')):
            continue
        _checked_path(child)
        if not child.is_dir():
            raise CustodyError('checkpoint staging entry is not a directory')
        entries, apparent = 0, 0
        def enumeration_error(_error):
            raise CustodyError('checkpoint staging could not be fully enumerated')
        for directory, names, files in os.walk(child, followlinks=False, onerror=enumeration_error):
            for name in names:
                _checked_path(Path(directory) / name)
            for name in files:
                path = _checked_path(Path(directory) / name)
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise CustodyError('checkpoint staging contains nonregular evidence')
                entries += 1
                apparent += info.st_size
                if entries > 10000:
                    raise CustodyError('checkpoint staging inventory exceeds its bound')
        stages.append({'name': child.name, 'files': entries, 'apparent_bytes': apparent})
    return {'free_bytes': free, 'stages': stages}


def inspect_evidence_destination(directory):
    directory = _checked_path(directory)
    if not directory.is_dir():
        raise CustodyError('phase evidence destination is unavailable')
    free = shutil.disk_usage(directory).free
    if free < REQUIRED_FREE_BYTES:
        raise CustodyError('phase evidence filesystem has insufficient free space')
    return {'free_bytes': free}


def _official_url(url, suffix):
    if type(url) is not str:
        raise CustodyError('publication URL is invalid')
    parts = urllib.parse.urlsplit(url)
    if (parts.scheme != 'https' or parts.netloc != 'raw.githubusercontent.com'
            or parts.query or parts.fragment
            or re.fullmatch(r'/EastonSell/model-looking-glass/[0-9a-f]{40}/[A-Za-z0-9_./-]+', parts.path) is None
            or not parts.path.endswith('/' + suffix)
            or '..' in parts.path.split('/')):
        raise CustodyError('publication URL is not the required immutable GitHub path')
    return url


def read_public(url, suffix, *, maximum=8 * 1024 * 1024):
    """Read an immutable official path without accepting redirect substitution."""
    _official_url(url, suffix)
    with urllib.request.urlopen(url, timeout=20) as response:
        if response.geturl() != url:
            raise CustodyError('publication redirected')
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise CustodyError('published file exceeds its size bound')
    return data


def _github_json(url, *, maximum=2 * 1024 * 1024):
    request = urllib.request.Request(url, headers={'User-Agent': 'Model-Looking-Glass-heldout-custodian',
                                                  'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.geturl() != url:
            raise CustodyError('official-main observation redirected')
        raw = response.read(maximum + 1)
    if len(raw) > maximum:
        raise CustodyError('official-main observation exceeds its bound')
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise CustodyError('official-main observation is invalid JSON') from error


def observe_current_inventory(*, api_reader=_github_json, source_reader=read_public):
    """Observe official main at one instant; this is not a global exposure lock."""
    if type(CURRENT_INVENTORY_VERSION) is not int or CURRENT_INVENTORY_VERSION < 7:
        raise CustodyError('reviewed exposure version is not configured')
    api = 'https://api.github.com/repos/EastonSell/model-looking-glass'
    head_record = api_reader(api + '/commits/main')
    head = head_record.get('sha') if type(head_record) is dict else None
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise CustodyError('official main has no valid commit identity')
    listing = api_reader(api + '/contents/looking_glass?ref=' + head)
    if type(listing) is not list or len(listing) >= 1000:
        raise CustodyError('official exposure directory listing is unavailable')
    versions = []
    for entry in listing:
        name = entry.get('name') if type(entry) is dict else None
        match = re.fullmatch(r'pretrained_exposures_v([1-9][0-9]*)\.py', name or '')
        if match:
            versions.append(int(match.group(1)))
    if not versions or max(versions) != CURRENT_INVENTORY_VERSION:
        raise CustodyError('official main has a different latest exposure version')
    suffix = f'looking_glass/pretrained_exposures_v{CURRENT_INVENTORY_VERSION}.py'
    url = f'https://raw.githubusercontent.com/EastonSell/model-looking-glass/{head}/{suffix}'
    source_sha = _digest(source_reader(url, suffix))
    if source_sha != CURRENT_INVENTORY_SOURCE_SHA256:
        raise CustodyError('official main exposure source differs from the reviewed source')
    return {'head': head, 'version': CURRENT_INVENTORY_VERSION,
            'source_sha256': source_sha, 'inventory_sha256': CURRENT_INVENTORY_SHA256}


def require_published(local, url, suffix, *, reader=read_public):
    if type(local) is not bytes or not local:
        raise CustodyError('published input must be nonempty immutable bytes')
    remote = reader(_official_url(url, suffix), suffix)
    if type(remote) is not bytes or remote != local:
        raise CustodyError('published bytes differ from the retained input')
    return _digest(local)


def tokenize_assignment(allocation, tokenizer):
    """Validate every full text, single subject and continuation without a model."""
    prefix = tokenizer.encode('The', add_special_tokens=False)
    def valid_tokens(tokens):
        return (type(tokens) is list and bool(tokens) and
                all(type(token) is int and 0 <= token <= 50256 for token in tokens))

    if not valid_tokens(prefix):
        raise CustodyError('subject prefix is not tokenizable')
    output = []
    for family in allocation.families:
        suffix = tokenizer.encode(' ' + family.context, add_special_tokens=False)
        if not valid_tokens(suffix) or len(family.subjects) != 4 or len(family.texts) != 4:
            raise CustodyError('family context is not tokenizable')
        members = []
        for subject, text in zip(family.subjects, family.texts):
            subject_prefix = tokenizer.encode('The ' + subject, add_special_tokens=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if (not valid_tokens(subject_prefix) or len(subject_prefix) != len(prefix) + 1
                    or subject_prefix[:-1] != prefix or type(ids) is not list
                    or ids != subject_prefix + suffix or not 1 <= len(ids) <= 32
                    or any(type(token) is not int or not 0 <= token <= 50256 for token in ids)):
                raise CustodyError('held-out subject or context alignment failed')
            for continuation, target in TARGETS.items():
                continued = tokenizer.encode(text + continuation, add_special_tokens=False)
                if not valid_tokens(continued) or continued != ids + [target]:
                    raise CustodyError('held-out continuation alignment failed')
            members.append({'text': text, 'input_ids': list(ids),
                            'attention_mask': [1] * len(ids), 'subject_index': len(prefix)})
        if len(members) != 4:
            raise CustodyError('held-out family is incomplete')
        output.append(members)
    if len(output) != 4:
        raise CustodyError('held-out allocation is incomplete')
    return output


def reject_known_row_prefixes(cases, inventory_bytes):
    """Prospectively reject either complete-row prefix direction."""
    try:
        records = json.loads(inventory_bytes)['records']
        known = [tuple(row['token_ids']) for row in records]
        candidates = [tuple(item['input_ids']) for family in cases for item in family]
    except (ValueError, KeyError, TypeError, RecursionError) as error:
        raise CustodyError('cannot check known complete-row prefixes') from error
    if len(candidates) != 16 or not known:
        raise CustodyError('known-prefix check has incomplete rows')
    for candidate in candidates:
        for row in known:
            shared = min(len(candidate), len(row))
            if candidate[:shared] == row[:shared]:
                raise CustodyError('held-out complete-row prefix collision')


@dataclass(frozen=True)
class PublicInputs:
    binding: bytes
    prediction: bytes
    aggregate: bytes
    inventory: bytes
    v7_inventory: bytes
    inventory_source: bytes
    operational_binding: bytes
    binding_url: str
    prediction_url: str
    aggregate_url: str
    inventory_sha256: str
    inventory_source_url: str
    operational_binding_url: str
    wrapper_url: str
    deviation_url: str
    deviation_bytes: bytes


def check_public_inputs(public, *, commitments, heldout, wrapper_bytes, reader=read_public):
    """Check frozen bytes and observed publication before any secret is drawn."""
    if type(public) is not PublicInputs or type(wrapper_bytes) is not bytes:
        raise CustodyError('immutable public inputs are required')
    binding_sha = require_published(public.binding, public.binding_url,
                                    'results/distilgpt2-natural-agreement-v1-20261028/binding-v2.json',
                                    reader=reader)
    if binding_sha != BINDING_SHA256:
        raise CustodyError('discovery binding differs')
    require_published(wrapper_bytes, public.wrapper_url, 'tools/natural_heldout_custody.py', reader=reader)
    require_published(public.deviation_bytes, public.deviation_url,
                      'docs/NATURAL_AGREEMENT_HELDOUT_DEVIATION.md', reader=reader)
    prediction_sha = require_published(public.prediction, public.prediction_url,
                                       'natural-prediction-v2.json', reader=reader)
    aggregate_sha = require_published(public.aggregate, public.aggregate_url,
                                      'natural-prediction-freeze.json', reader=reader)
    if prediction_sha != PREDICTION_SHA256 or aggregate_sha != AGGREGATE_SHA256:
        raise CustodyError('published prediction or aggregate differs from the observed freeze')
    if (type(CURRENT_INVENTORY_VERSION) is not int or CURRENT_INVENTORY_VERSION < 7
            or public.inventory_sha256 != CURRENT_INVENTORY_SHA256
            or _digest(public.inventory) != CURRENT_INVENTORY_SHA256
            or _digest(public.v7_inventory) != V7_INVENTORY_SHA256):
        raise CustodyError('reviewed current exposure inventory is not bound')
    inventory_source_sha = require_published(
        public.inventory_source, public.inventory_source_url,
        f'looking_glass/pretrained_exposures_v{CURRENT_INVENTORY_VERSION}.py', reader=reader)
    if inventory_source_sha != CURRENT_INVENTORY_SOURCE_SHA256:
        raise CustodyError('published exposure source differs from the reviewed source')
    require_published(public.operational_binding, public.operational_binding_url,
                      'results/distilgpt2-natural-agreement-v1-20261028/heldout-operational-binding-v1.json',
                      reader=reader)
    try:
        operation = json.loads(public.operational_binding)
        if (type(operation) is not dict or _json_bytes(operation) != public.operational_binding
                or set(operation) != {'schema', 'core_source_commit', 'binding_sha256',
                                      'prediction_sha256', 'aggregate_sha256', 'inventory_version',
                                      'inventory_sha256', 'inventory_source_sha256', 'runtime_overrides',
                                      'limits', 'sources'}
                or operation['schema'] != 'mlg-natural-heldout-operational-binding-v1'
                or operation['core_source_commit'] != SOURCE_COMMIT
                or operation['binding_sha256'] != BINDING_SHA256
                or operation['prediction_sha256'] != PREDICTION_SHA256
                or operation['aggregate_sha256'] != AGGREGATE_SHA256
                or type(operation['inventory_version']) is not int
                or operation['inventory_version'] != CURRENT_INVENTORY_VERSION
                or operation['inventory_sha256'] != CURRENT_INVENTORY_SHA256
                or operation['inventory_source_sha256'] != CURRENT_INVENTORY_SOURCE_SHA256
                or operation['runtime_overrides'] != {'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None}
                or operation['limits'] != {'tensor_bytes': LOGICAL_LIMIT, 'event_bytes': EVENT_LIMIT,
                                           'side_bytes': SIDE_LIMIT}
                or type(operation['sources']) is not dict
                or set(operation['sources']) != set(SOURCE_SUFFIXES)):
            raise CustodyError('operational binding differs from the reviewed plan')
        for name, suffix in SOURCE_SUFFIXES.items():
            entry = operation['sources'][name]
            if (type(entry) is not dict or set(entry) != {'url', 'sha256'}
                    or type(entry['sha256']) is not str
                    or re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) is None):
                raise CustodyError('operational source identity is invalid')
            remote = reader(_official_url(entry['url'], suffix), suffix)
            if type(remote) is not bytes or _digest(remote) != entry['sha256']:
                raise CustodyError('operational source is not published as bound')
        if (operation['sources']['custody'] != {'url': public.wrapper_url, 'sha256': _digest(wrapper_bytes)}
                or operation['sources']['deviation'] != {'url': public.deviation_url,
                                                         'sha256': _digest(public.deviation_bytes)}):
            raise CustodyError('operational binding does not name this custody source and deviation')
        binding = json.loads(public.binding)
        freeze = commitments.load_prediction_freeze(public.aggregate, aggregate_sha)
        aggregate = json.loads(public.aggregate)
        track = aggregate['tracks']['natural']
        if (set(aggregate['tracks']) != {'natural'} or track['source_commit'] != SOURCE_COMMIT
                or track['runtime_sha256'] != binding_sha or track['prediction_sha256'] != prediction_sha
                or binding['source_commit'] != SOURCE_COMMIT
                or binding['checkpoint_revision'] != CHECKPOINT):
            raise CustodyError('natural freeze does not match its discovery binding')
        heldout.load_frozen_natural_predictions(public.prediction, freeze)
        snapshot = commitments.load_inventory_snapshot(public.inventory, public.inventory_sha256)
        current_records = json.loads(public.inventory)['records']
        v7_records = json.loads(public.v7_inventory)['records']
        def prefixes(records):
            return {tuple(row['token_ids'][:length]) for row in records
                    for length in range(1, len(row['token_ids']) + 1)}
        if not prefixes(v7_records).issubset(prefixes(current_records)):
            raise CustodyError('current inventory omits a v7 known prefix')
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        raise CustodyError('public freeze or inventory is invalid') from error
    return binding, freeze, snapshot


def prepare_once(public, private_root, *, commitments, heldout, tokenizer,
                 wrapper_bytes, claim_phase, observe_inventory, config_bytes, config_path, reader=read_public,
                 nonce_source=secrets.token_bytes):
    """Claim once and privately seal one random split; never print its contents.

    Currentness of the independently reviewed inventory and actual published
    chronology remain trusted operator duties. A failed claim is never reused.
    """
    if _read_private(Path(__file__), 8 * 1024 * 1024) != wrapper_bytes:
        raise CustodyError('executed custody source differs from the published bytes')
    config_path = _checked_path(config_path)
    if (type(config_bytes) is not bytes or len(config_bytes) > 64 * 1024
            or _read_private(config_path, 64 * 1024) != config_bytes):
        raise CustodyError('preparation configuration is not bound to its protected file')
    binding, freeze, snapshot = check_public_inputs(public, commitments=commitments,
                                                    heldout=heldout, wrapper_bytes=wrapper_bytes,
                                                    reader=reader)
    first_observation = observe_inventory()
    if (type(first_observation) is not dict or first_observation.get('inventory_sha256') != snapshot.sha256
            or first_observation.get('source_sha256') != CURRENT_INVENTORY_SOURCE_SHA256):
        raise CustodyError('latest official exposure observation differs')
    root = _checked_path(private_root)
    if not root.is_dir():
        raise CustodyError('private custody root must already exist')
    attempt = root / ('natural-heldout-' + freeze.sha256)
    grant = claim_phase('natural_heldout', output_path=attempt)
    if (grant.phase != 'natural_heldout' or Path(grant.output_path) != attempt
            or grant.logical_limit_bytes != LOGICAL_LIMIT or grant.disk_limit_bytes != DISK_LIMIT):
        raise CustodyError('phase reservation differs from the frozen limits')
    attempt.mkdir()
    claim = {'schema': 'mlg-natural-heldout-custody-claim-v1',
             'prediction_freeze_sha256': freeze.sha256, 'binding_sha256': BINDING_SHA256,
             'prediction_sha256': _digest(public.prediction), 'inventory_sha256': snapshot.sha256,
             'wrapper_sha256': _digest(wrapper_bytes), 'deviation_sha256': _digest(public.deviation_bytes),
             'mkl_cbwr': None, 'aten_cpu_capability': None,
             'operational_binding_sha256': _digest(public.operational_binding),
             'preparation_config_sha256': _digest(config_bytes), 'preparation_config_path': str(config_path),
             'event_log_limit_bytes': EVENT_LIMIT, 'side_file_limit_bytes': SIDE_LIMIT}
    _write_new(attempt / 'claim.json', _json_bytes(claim))
    try:
        _write_new(attempt / 'preparation-config.json', config_bytes)
        second_observation = observe_inventory()
        if (type(second_observation) is not dict
                or second_observation.get('inventory_sha256') != snapshot.sha256
                or second_observation.get('source_sha256') != CURRENT_INVENTORY_SOURCE_SHA256):
            raise CustodyError('official exposure changed after the phase claim')
        _write_new(attempt / 'exposure-observations.json',
                   _json_bytes({'before_claim': first_observation, 'after_claim': second_observation}))
        nonce = nonce_source(32)
        if type(nonce) is not bytes or len(nonce) != 32:
            raise CustodyError('split nonce generation failed')
        _write_new(attempt / 'nonce.bin', nonce)
        allocation = commitments.sample_families(
            freeze, nonce, inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256)
        cases = tokenize_assignment(allocation, tokenizer)
        reject_known_row_prefixes(cases, public.inventory)
        sealed = commitments.seal_heldout(freeze, nonce, cases, allocation=allocation,
                                          inventory_snapshot=snapshot,
                                          current_inventory_sha256=snapshot.sha256)
        if len(sealed.record_bytes) > 1024 * 1024 or len(public.inventory) > 1024 * 1024:
            raise CustodyError('private record exceeds its side-file reserve')
        _write_new(attempt / 'heldout-record.json', sealed.record_bytes)
        _write_new(attempt / 'inventory.json', public.inventory)
        commitment = {'schema': 'mlg-natural-heldout-public-commitment-v1',
                      'prediction_freeze_sha256': freeze.sha256,
                      'exposure_inventory_sha256': snapshot.sha256,
                      'heldout_commitment': sealed.commitment_sha256,
                      'wrapper_sha256': _digest(wrapper_bytes),
                      'deviation_sha256': _digest(public.deviation_bytes),
                      'operational_binding_sha256': _digest(public.operational_binding)}
        _write_new(attempt / 'commitment.json', _json_bytes(commitment))
        account_phase(attempt, reserve_side_bytes=8 * 1024 * 1024)
        _write_new(attempt / 'preparation-complete.json', _json_bytes({
            'schema': 'mlg-natural-heldout-preparation-complete-v1',
            'commitment_file_sha256': _digest(_json_bytes(commitment)),
            'operational_binding_sha256': _digest(public.operational_binding)}))
        return attempt, commitment
    except BaseException as error:
        try:
            _write_new(attempt / 'preparation-failure.json',
                       _json_bytes({'status': 'incomplete', 'error_type': type(error).__name__}))
        except BaseException:
            pass
        raise


def audit_preparation(attempt, public, wrapper_bytes, *, allow_owner=False):
    """Hash a completed private preparation after its child has exited."""
    attempt = _checked_path(attempt)
    expected = {'claim.json', 'preparation-config.json', 'exposure-observations.json', 'nonce.bin', 'heldout-record.json',
                'inventory.json', 'commitment.json', 'preparation-complete.json'}
    entries = {path.name for path in attempt.iterdir()}
    if entries != expected and not (allow_owner and entries == expected | {'owner'}):
        raise CustodyError('preparation is incomplete, failed or already started')
    limits = {'claim.json': 8192, 'preparation-config.json': 65536,
              'exposure-observations.json': 65536, 'nonce.bin': 32,
              'heldout-record.json': 1024 * 1024, 'inventory.json': 1024 * 1024,
              'commitment.json': 8192, 'preparation-complete.json': 8192}
    raw = {name: _read_private(attempt / name, limit) for name, limit in limits.items()}
    if len(raw['nonce.bin']) != 32 or raw['inventory.json'] != public.inventory:
        raise CustodyError('preparation private inputs differ')
    try:
        claim = json.loads(raw['claim.json'])
        config_path = claim['preparation_config_path']
    except (ValueError, KeyError, TypeError) as error:
        raise CustodyError('preparation configuration identity is invalid') from error
    if type(config_path) is not str or not Path(config_path).is_absolute() or len(config_path) > 4096:
        raise CustodyError('preparation configuration path is invalid')
    expected_claim = {'schema': 'mlg-natural-heldout-custody-claim-v1',
                      'prediction_freeze_sha256': AGGREGATE_SHA256, 'binding_sha256': BINDING_SHA256,
                      'prediction_sha256': PREDICTION_SHA256, 'inventory_sha256': public.inventory_sha256,
                      'wrapper_sha256': _digest(wrapper_bytes), 'deviation_sha256': _digest(public.deviation_bytes),
                      'mkl_cbwr': None, 'aten_cpu_capability': None,
                      'operational_binding_sha256': _digest(public.operational_binding),
                      'preparation_config_sha256': _digest(raw['preparation-config.json']),
                      'preparation_config_path': config_path,
                      'event_log_limit_bytes': EVENT_LIMIT, 'side_file_limit_bytes': SIDE_LIMIT}
    if raw['claim.json'] != _json_bytes(expected_claim):
        raise CustodyError('preparation claim differs from its frozen inputs')
    expected_complete = {'schema': 'mlg-natural-heldout-preparation-complete-v1',
                         'commitment_file_sha256': _digest(raw['commitment.json']),
                         'operational_binding_sha256': _digest(public.operational_binding)}
    if raw['preparation-complete.json'] != _json_bytes(expected_complete):
        raise CustodyError('preparation completion does not bind its commitment')
    sizes = account_phase(attempt)
    if not allow_owner and (sizes['tensor_bytes'] or sizes['event_bytes']):
        raise CustodyError('preparation unexpectedly contains model evidence')
    return {'status': 'evidence_complete', 'inventory': {'retained_tensor_bytes': 0},
            'preparation': {name: _digest(data) for name, data in sorted(raw.items())}}


def verify_preparation_supervisor(attempt, supervisor_dir, public, wrapper_bytes, *, require_outer_observation=True):
    """Require a drained supervisor receipt over the exact private files."""
    supervisor_dir = _checked_path(supervisor_dir)
    expected = audit_preparation(attempt, public, wrapper_bytes, allow_owner=True)
    try:
        receipt_raw = _read_private(supervisor_dir / 'supervisor.json', 2 * 1024 * 1024)
        receipt = json.loads(receipt_raw)
        claim = json.loads(_read_private(Path(attempt) / 'claim.json', 8192))
        if (_digest(_read_private(Path(claim['preparation_config_path']), 65536))
                != claim['preparation_config_sha256']):
            raise CustodyError('the bound preparation configuration changed')
        command = receipt.get('command') if type(receipt) is dict else None
        if (type(command) is not list or len(command) != 14
                or command[0] != sys.executable or command[1:5] != ['-I', '-S', '-B', '-X']
                or type(command[5]) is not str or not command[5].startswith('pycache_prefix=')
                or not Path(command[5][len('pycache_prefix='):]).is_absolute()
                or command[6:8] != ['-c', BOOTSTRAP] or command[8] != str(Path(__file__))
                or command[9:11] != ['prepare', '--config']
                or command[11] != claim['preparation_config_path']
                or command[12:] != ['--config-sha', claim['preparation_config_sha256']]):
            raise CustodyError('preparation supervisor command differs from the bound entrypoint')
        if (type(receipt) is not dict or receipt.get('status') != 'evidence_complete'
                or type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0
                or type(receipt.get('child_pid')) is not int or not 0 < receipt['child_pid'] <= 0xffffffff
                or any(receipt.get(field) is not True for field in
                       ('child_reaped', 'job_drained', 'job_closed', 'logs_complete', 'cleanup_verified'))
                or any(receipt.get(field) is not False for field in
                       ('retry_safe', 'cancellation_requested', 'quarantined'))
                or receipt.get('supervisor_error') is not None or receipt.get('log_overflow') is not False
                or _json_bytes(receipt.get('audit')) != _json_bytes(expected)
                or receipt.get('retained_tensor_files', {}).get('accounting_complete') is not True
                or type(receipt['retained_tensor_files'].get('total_bytes')) is not int
                or receipt['retained_tensor_files']['total_bytes'] != 0):
            raise CustodyError('preparation supervisor did not complete cleanly')
        for stream in ('stdout', 'stderr'):
            raw = _read_private(supervisor_dir / (stream + '.txt'), 1024 * 1024)
            if receipt.get(stream + '_sha256') != _digest(raw):
                raise CustodyError('preparation supervisor log differs from its receipt')
        if require_outer_observation:
            observed = _read_private(supervisor_dir / 'supervisor-observation.json', 8192)
            observation = json.loads(observed)
            supervisor_pid = observation.get('supervisor_pid')
            process_birth = observation.get('supervisor_birth_100ns')
            if (type(supervisor_pid) is not int or not 0 < supervisor_pid <= 0xffffffff
                    or supervisor_pid == receipt['child_pid'] or type(process_birth) is not int
                    or not 0 < process_birth < 2 ** 64):
                raise CustodyError('outer supervisor process identity is invalid')
            expected_observation = {'schema': 'mlg-natural-heldout-preparation-outer-observation-v1',
                                    'status': 'completed', 'exit_code': 0, 'supervisor_reaped': True,
                                    'outer_job_drained': True, 'outer_job_closed': True,
                                    'supervisor_pid': supervisor_pid, 'supervisor_birth_100ns': process_birth,
                                    'command': command[:7] + [OUTER_BOOTSTRAP] + command[8:9]
                                               + ['supervise-prepare'] + command[10:],
                                    'supervisor_record_sha256': _digest(receipt_raw),
                                    'preparation_config_sha256': claim['preparation_config_sha256'],
                                    'operational_binding_sha256': _digest(public.operational_binding)}
            if observed != _json_bytes(expected_observation):
                raise CustodyError('preparation supervisor exit was not observed')
        account_phase(attempt, reserve_side_bytes=4 * 1024 * 1024,
                      extra_side_roots=(supervisor_dir, Path(claim['preparation_config_path'])))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise CustodyError('preparation supervision is invalid') from error
    return expected


def _process_birth_100ns(process):
    """Read the Windows creation time from the retained process handle."""
    if os.name != 'nt':
        raise CustodyError('outer process observation requires Windows')
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                       ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                       ctypes.POINTER(wintypes.FILETIME))
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(int(process._handle), ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        raise CustodyError('cannot establish the supervisor process birth identity')
    birth = (created.dwHighDateTime << 32) | created.dwLowDateTime
    if not 0 < birth < 2 ** 64:
        raise CustodyError('invalid supervisor process birth identity')
    return birth


def _outer_supervisor_job(guards):
    """Contain the supervisor while its separately gated child has its own Job."""
    import ctypes
    job = guards.WindowsJob()
    try:
        limits = guards._JobExtendedLimits()
        # The nested child starts outside this outer job but cannot proceed
        # before the supervisor assigns it to its own kill-on-close Job.
        limits.basic.flags = 0x00002000 | 0x00001000
        if not job._kernel.SetInformationJobObject(job._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise CustodyError('cannot configure the outer supervisor Job')
        return job
    except BaseException:
        job.close()
        raise


def observe_supervisor_process(command, attempt, supervisor_dir, public, wrapper_bytes, *,
                               config_path, config_sha256, job_factory, custody_state=None, timeout=7200):
    """Observe and reap the actual preparation supervisor before recording success.

    The process owns its own Windows Job and private child logs. On an
    interrupted or uncertain observation, preserve all files and stop; this
    function never retries the one-shot preparation or draws a nonce itself.
    """
    if (os.name != 'nt' or type(command) is not list or not command
            or any(type(item) is not str or not item for item in command)
            or type(timeout) is not int or not 1 <= timeout <= 7200):
        raise CustodyError('invalid outer process observation')
    config_path = _checked_path(config_path)
    if (type(config_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', config_sha256) is None
            or _digest(_read_private(config_path, 65536)) != config_sha256
            or len(command) != 14 or command[0] != sys.executable or command[1:5] != ['-I', '-S', '-B', '-X']
            or not command[5].startswith('pycache_prefix=')
            or not Path(command[5][len('pycache_prefix='):]).is_absolute()
            or command[6:8] != ['-c', OUTER_BOOTSTRAP] or command[8] != str(Path(__file__))
            or command[9:11] != ['supervise-prepare', '--config']
            or command[11] != str(config_path) or command[12:] != ['--config-sha', config_sha256]):
        raise CustodyError('outer supervisor command or configuration differs before spawn')
    if custody_state is None:
        custody_state = [None, None, False, False]
    if type(custody_state) is not list or len(custody_state) != 4 or custody_state != [None, None, False, False]:
        raise CustodyError('outer process custody state is invalid')
    process = None
    job = None
    reaped = False
    drained = False
    closed = False
    try:
        job = job_factory()
        custody_state[0] = job
        process = subprocess.Popen(command, env=os.environ.copy(), stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.DETACHED_PROCESS)
        custody_state[1] = process
        if type(process.pid) is not int or not 0 < process.pid <= 0xffffffff:
            raise CustodyError('invalid outer supervisor PID')
        job.assign(process)
        birth = _process_birth_100ns(process)
        start = b'MLG-natural-discovery-start-v1\n'
        custody_state[2] = True
        if process.stdin.write(start + start) != len(start) * 2:
            raise CustodyError('outer supervisor start gate failed')
        process.stdin.close()
        code = process.wait(timeout=timeout)
        reaped = True
        if type(code) is not int or code != 0:
            raise CustodyError('outer supervisor did not exit successfully')
        job.terminate_and_wait()
        drained = True
        job.close()
        closed = True
        verify_preparation_supervisor(attempt, supervisor_dir, public, wrapper_bytes,
                                      require_outer_observation=False)
        receipt_raw = _read_private(Path(supervisor_dir) / 'supervisor.json', 2 * 1024 * 1024)
        receipt = json.loads(receipt_raw)
        claim = json.loads(_read_private(Path(attempt) / 'claim.json', 8192))
        expected_command = (receipt['command'][:7] + [OUTER_BOOTSTRAP] + receipt['command'][8:9]
                            + ['supervise-prepare'] + receipt['command'][10:])
        if command != expected_command or process.pid == receipt['child_pid']:
            raise CustodyError('observed process differs from the bound preparation supervisor')
        custody_state[3] = True
        observation = {'schema': 'mlg-natural-heldout-preparation-outer-observation-v1',
                       'status': 'completed', 'exit_code': 0, 'supervisor_reaped': True,
                       'outer_job_drained': True, 'outer_job_closed': True,
                       'supervisor_pid': process.pid, 'supervisor_birth_100ns': birth, 'command': command,
                       'supervisor_record_sha256': _digest(receipt_raw),
                       'preparation_config_sha256': claim['preparation_config_sha256'],
                       'operational_binding_sha256': _digest(public.operational_binding)}
        _write_new(Path(supervisor_dir) / 'supervisor-observation.json', _json_bytes(observation))
        verify_preparation_supervisor(attempt, supervisor_dir, public, wrapper_bytes)
        return {'status': 'observed_preparation_exit',
                'supervisor_observation_sha256': _digest(_json_bytes(observation)),
                'supervisor_record_sha256': _digest(receipt_raw),
                'preparation_config_sha256': claim['preparation_config_sha256'],
                'operational_binding_sha256': _digest(public.operational_binding)}
    except BaseException as primary_error:
        if custody_state[2] and not custody_state[3]:
            # Do not kill an outer supervisor that may own the only live
            # handles for its breakaway inner Job. The enclosing source-lease
            # scope sees this preallocated state and holds it for recovery.
            raise
        cleanup_errors = []
        if job is not None and not drained:
            try:
                job.terminate_and_wait()
                drained = True
            except BaseException as error:
                cleanup_errors.append(error)
        if job is not None and not closed:
            try:
                job.close()
                closed = True
            except BaseException as error:
                cleanup_errors.append(error)
        if process is not None and not reaped:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            cleanup_error = next((error for error in cleanup_errors
                                  if isinstance(error, (KeyboardInterrupt, SystemExit))), cleanup_errors[0])
            selected = (primary_error if isinstance(primary_error, (KeyboardInterrupt, SystemExit))
                        else cleanup_error if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit))
                        else primary_error)
            if hasattr(selected, 'add_note'):
                selected.add_note('outer supervisor cleanup is unverified; stop and reconcile')
            if selected is primary_error:
                raise primary_error from cleanup_errors[0]
            raise selected from primary_error
        if process is None and job is not None and not isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
            raise CustodyError('outer supervisor spawn return is unverified; stop and reconcile') from primary_error
        raise


def _hold_unverified_process(custody, *, _test_release=None):
    """Keep source leases and unknown process/job handles rooted for recovery."""
    handlers = []
    for name in ('SIGINT', 'SIGBREAK'):
        try:
            kind = getattr(signal, name)
            handlers.append((kind, signal.getsignal(kind)))
            signal.signal(kind, signal.SIG_IGN)
        except BaseException:
            pass
    # The enclosing production stack keeps both SourceFileLease instances and
    # these strong references alive. A trusted operator must independently
    # establish process-tree termination before explicitly ending quarantine.
    while True:
        try:
            try:
                print(json.dumps({'status': 'quarantined', 'reason': 'process cleanup unverified'}),
                      file=sys.stderr, flush=True)
            except BaseException:
                pass
            while True:
                try:
                    time.sleep(30)
                    if _test_release is not None and _test_release(custody):
                        for kind, previous in handlers:
                            signal.signal(kind, previous)
                        return
                except BaseException:
                    continue
        except BaseException:
            continue


def _hold_persistently(custody):
    while True:
        try:
            _hold_unverified_process(custody)
        except BaseException:
            continue


def _quarantine_state_if_needed(custody_state):
    while True:
        try:
            if custody_state[2] and not custody_state[3]:
                _hold_persistently(custody_state)
            return
        except BaseException:
            continue


def _quarantine_supervisor_if_needed(supervisor):
    while True:
        try:
            if supervisor._QUARANTINED:
                _hold_persistently(supervisor)
            return
        except BaseException:
            continue


def _ignore_console_interrupts():
    """Mask console interruption before starting either gated process."""
    for name in ('SIGINT', 'SIGBREAK'):
        kind = getattr(signal, name, None)
        if kind is not None:
            signal.signal(kind, signal.SIG_IGN)


def _supervise_preparation(supervisor, command, directory, audit_result):
    """Keep the inner supervisor rooted when its own cleanup is uncertain."""
    try:
        verdict = supervisor.supervise_heldout(command, directory, os.environ.copy(), audit_result=audit_result)
    finally:
        _quarantine_supervisor_if_needed(supervisor)
    if type(verdict) is not dict or verdict.get('cleanup_verified') is not True:
        _hold_persistently(supervisor)
    return verdict


def verify_prepared(attempt, public, commitment_url, *, commitments, heldout,
                    tokenizer, wrapper_bytes, observe_inventory, preparation_supervisor, reader=read_public):
    """Recheck a privately sealed split and its publicly observed commitment.

    Returned records remain private. This function does not run a model.
    """
    if _read_private(Path(__file__), 8 * 1024 * 1024) != wrapper_bytes:
        raise CustodyError('executed custody source differs from the published bytes')
    _, freeze, snapshot = check_public_inputs(public, commitments=commitments,
                                              heldout=heldout, wrapper_bytes=wrapper_bytes,
                                              reader=reader)
    current = observe_inventory()
    if (type(current) is not dict or current.get('inventory_sha256') != snapshot.sha256
            or current.get('source_sha256') != CURRENT_INVENTORY_SOURCE_SHA256):
        raise CustodyError('official exposure changed before execution')
    attempt = Path(attempt)
    _checked_path(attempt)
    verify_preparation_supervisor(attempt, preparation_supervisor, public, wrapper_bytes)
    raw_claim = _read_private(attempt / 'claim.json', 8192)
    claim = json.loads(raw_claim)
    expected_claim = {'schema': 'mlg-natural-heldout-custody-claim-v1',
                      'prediction_freeze_sha256': freeze.sha256, 'binding_sha256': BINDING_SHA256,
                      'prediction_sha256': _digest(public.prediction), 'inventory_sha256': snapshot.sha256,
                      'wrapper_sha256': _digest(wrapper_bytes), 'deviation_sha256': _digest(public.deviation_bytes),
                      'mkl_cbwr': None, 'aten_cpu_capability': None,
                      'operational_binding_sha256': _digest(public.operational_binding),
                      'preparation_config_sha256': _digest(_read_private(attempt / 'preparation-config.json', 65536)),
                      'preparation_config_path': claim['preparation_config_path'],
                      'event_log_limit_bytes': EVENT_LIMIT, 'side_file_limit_bytes': SIDE_LIMIT}
    if raw_claim != _json_bytes(expected_claim):
        raise CustodyError('retained claim differs from the frozen inputs')
    nonce = _read_private(attempt / 'nonce.bin', 32)
    record_bytes = _read_private(attempt / 'heldout-record.json', 1024 * 1024)
    if _read_private(attempt / 'inventory.json', 1024 * 1024) != public.inventory:
        raise CustodyError('retained exposure bytes differ')
    raw_commitment = _read_private(attempt / 'commitment.json', 8192)
    completed = _read_private(attempt / 'preparation-complete.json', 8192)
    if completed != _json_bytes({'schema': 'mlg-natural-heldout-preparation-complete-v1',
                                 'commitment_file_sha256': _digest(raw_commitment),
                                 'operational_binding_sha256': _digest(public.operational_binding)}):
        raise CustodyError('preparation completion does not bind its commitment')
    try:
        commitment = json.loads(raw_commitment)
        expected = {'schema': 'mlg-natural-heldout-public-commitment-v1',
                    'prediction_freeze_sha256': freeze.sha256,
                    'exposure_inventory_sha256': snapshot.sha256,
                    'heldout_commitment': commitment['heldout_commitment'],
                    'wrapper_sha256': _digest(wrapper_bytes),
                    'deviation_sha256': _digest(public.deviation_bytes),
                    'operational_binding_sha256': _digest(public.operational_binding)}
        if (_json_bytes(commitment) != raw_commitment or commitment != expected
                or type(commitment['heldout_commitment']) is not str
                or re.fullmatch(r'[0-9a-f]{64}', commitment['heldout_commitment']) is None):
            raise CustodyError('private held-out commitment is malformed')
        require_published(raw_commitment, commitment_url,
                          'results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json',
                          reader=reader)
        allocation = commitments.sample_families(freeze, nonce, inventory_snapshot=snapshot,
                                                   current_inventory_sha256=snapshot.sha256)
        cases = tokenize_assignment(allocation, tokenizer)
        reject_known_row_prefixes(cases, public.inventory)
        commitments.verify_heldout_reveal(freeze, nonce, record_bytes,
                                          commitment['heldout_commitment'], inventory_snapshot=snapshot)
        sealed = commitments.seal_heldout(freeze, nonce, cases, allocation=allocation,
                                          inventory_snapshot=snapshot,
                                          current_inventory_sha256=snapshot.sha256)
        if sealed.record_bytes != record_bytes or sealed.commitment_sha256 != commitment['heldout_commitment']:
            raise CustodyError('retokenized cases differ from the sealed record')
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        raise CustodyError('held-out verification failed') from error
    return freeze, snapshot, nonce, record_bytes, commitment


def start_owner_once(attempt, commitment):
    """Record an exclusive model-owner start before any model is loaded."""
    if (type(commitment) is not dict
            or type(commitment.get('heldout_commitment')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', commitment['heldout_commitment']) is None):
        raise CustodyError('a verified held-out commitment is required')
    _write_new(Path(attempt) / 'owner-started.json', _json_bytes({
        'schema': 'mlg-natural-heldout-owner-start-v1',
        'heldout_commitment': commitment['heldout_commitment']}))


class _FrozenSourceLoader(importlib.abc.Loader):
    """Compile verified source bytes directly; never consume a cached .pyc."""
    def __init__(self, path, digest):
        self.path, self.digest = path, digest

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        data = _read_private(self.path, 8 * 1024 * 1024)
        if _digest(data) != self.digest:
            raise CustodyError('frozen import differs from the published source inventory')
        exec(compile(data, str(self.path), 'exec', dont_inherit=True), module.__dict__)


class _FrozenSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve study modules only from the protected discovery checkout."""
    def __init__(self, root, source_hashes):
        self.root, self.source_hashes = root, source_hashes

    def _spec(self, fullname, relative, *, package=False):
        digest = self.source_hashes.get(relative)
        if type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ModuleNotFoundError('project source is not in the frozen binding')
        path = self.root / relative
        loader = _FrozenSourceLoader(path, digest)
        return importlib.util.spec_from_file_location(
            fullname, path, loader=loader, submodule_search_locations=[] if package else None)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scripts':
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if fullname.startswith('scripts.'):
            if fullname != 'scripts.v2_source_guards':
                raise ModuleNotFoundError('unapproved project script')
            return self._spec(fullname, 'scripts/v2_source_guards.py')
        if fullname != 'looking_glass' and not fullname.startswith('looking_glass.'):
            return None
        parts = fullname.split('.')
        if any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) is None for part in parts):
            raise ModuleNotFoundError('invalid project module name')
        relative = '/'.join(parts)
        package, source = relative + '/__init__.py', relative + '.py'
        if package in self.source_hashes:
            return self._spec(fullname, package, package=True)
        if source in self.source_hashes:
            return self._spec(fullname, source)
        raise ModuleNotFoundError('project module has no frozen source')


def _load_file_module(name, path, digest):
    spec = importlib.util.spec_from_file_location(name, path, loader=_FrozenSourceLoader(path, digest))
    if spec is None or spec.loader is None or name in sys.modules:
        raise CustodyError('operational module cannot be loaded safely')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _private_config(path, expected_sha=None):
    try:
        raw = _read_private(path, 64 * 1024)
        if expected_sha is not None and (_digest(raw) != expected_sha
                                         or re.fullmatch(r'[0-9a-f]{64}', expected_sha) is None):
            raise CustodyError('private configuration changed between supervisor and child')
        config = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise CustodyError('private configuration is invalid') from error
    fields = {'schema', 'frozen_root', 'cache_dir', 'ledger_path', 'private_root',
              'inventory_path', 'inventory_source_url', 'wrapper_url', 'deviation_url',
              'operational_binding_path', 'operational_binding_url', 'commitment_url'}
    if (type(config) is not dict or set(config) != fields
            or config['schema'] != 'mlg-natural-heldout-private-config-v1'
            or any(type(config[key]) is not str or not config[key] or len(config[key]) > 4096
                   for key in fields - {'schema', 'commitment_url'})
            or (config['commitment_url'] is not None and
                (type(config['commitment_url']) is not str or not config['commitment_url']))):
        raise CustodyError('private configuration has invalid fields')
    return config


def _public_from_config(config, custody_root, v7_inventory):
    result = custody_root / 'results/distilgpt2-natural-agreement-v1-20261028'
    github = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
    return PublicInputs(
        binding=_read_private(result / 'binding-v2.json', 1024 * 1024),
        prediction=_read_private(result / 'natural-prediction-v2.json', 8 * 1024 * 1024),
        aggregate=_read_private(result / 'natural-prediction-freeze.json', 1024 * 1024),
        inventory=_read_private(config['inventory_path'], 1024 * 1024), v7_inventory=v7_inventory,
        inventory_source=_read_private(custody_root / 'looking_glass' /
                                      f'pretrained_exposures_v{CURRENT_INVENTORY_VERSION}.py', 1024 * 1024),
        operational_binding=_read_private(config['operational_binding_path'], 1024 * 1024),
        binding_url=github + 'dea2f2a6f5db8627f7d65100628fe60fd0344668/results/distilgpt2-natural-agreement-v1-20261028/binding-v2.json',
        prediction_url=github + '294e63cf4ffb38d740a5fd7862a61b6cad5b47a1/results/distilgpt2-natural-agreement-v1-20261028/natural-prediction-v2.json',
        aggregate_url=github + 'c8d37acc064ecb80d522fcdd0c98aea7a1d78f87/results/distilgpt2-natural-agreement-v1-20261028/natural-prediction-freeze.json',
        inventory_sha256=CURRENT_INVENTORY_SHA256, inventory_source_url=config['inventory_source_url'],
        operational_binding_url=config['operational_binding_url'], wrapper_url=config['wrapper_url'],
        deviation_url=config['deviation_url'], deviation_bytes=_read_private(
            custody_root / SOURCE_SUFFIXES['deviation'], 1024 * 1024))


def _verify_local_operational_sources(public, custody_root):
    try:
        record = json.loads(public.operational_binding)
        for name, relative in SOURCE_SUFFIXES.items():
            raw = _read_private(custody_root / relative, 8 * 1024 * 1024)
            if _digest(raw) != record['sources'][name]['sha256']:
                raise CustodyError('local operational source differs from its published binding')
    except (ValueError, KeyError, TypeError) as error:
        raise CustodyError('local operational sources are invalid') from error


def _install_frozen_finder(root, source_hashes):
    if any(name == 'looking_glass' or name.startswith('looking_glass.')
           or name == 'scripts' or name.startswith('scripts.') for name in sys.modules):
        raise CustodyError('project modules were loaded before the frozen-source bootstrap')
    index = sys.meta_path.index(importlib.machinery.PathFinder)
    if type(source_hashes) is not dict or not source_hashes:
        raise CustodyError('the frozen source inventory is missing')
    sys.meta_path.insert(index, _FrozenSourceFinder(root, source_hashes))


def _install_dependency_site(site):
    if not site.is_dir() or sys.path.count(str(site)) > 1:
        raise CustodyError('trusted dependency directory is invalid')
    if str(site) not in sys.path:
        sys.path.append(str(site))


def _require_fresh_dependencies():
    names = ('torch', 'numpy', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub', 'scipy')
    if any(name == prefix or name.startswith(prefix + '.')
           for name in sys.modules for prefix in names):
        raise CustodyError('numerical dependencies were loaded before the CPU startup check')


def production_prepare(config_path, *, mode='validate', expected_config_sha=None):
    """Production path: fixed ledger, OS nonce, official observer and frozen code."""
    if mode not in ('validate', 'prepare', 'supervise-prepare', 'observe-prepare'):
        raise CustodyError('unknown preparation mode')
    if mode in ('supervise-prepare', 'observe-prepare'):
        _ignore_console_interrupts()
    if (not sys.flags.isolated or not sys.flags.no_site or sys.flags.optimize != 0 or not sys.dont_write_bytecode
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or os.path.lexists(sys.pycache_prefix) or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1'
            or os.environ.get('OMP_NUM_THREADS') != '1'
            or 'MKL_CBWR' in os.environ or 'ATEN_CPU_CAPABILITY' in os.environ):
        raise CustodyError('requires the frozen isolated CPU startup and unset numerical overrides')
    _require_fresh_dependencies()
    if mode == 'prepare' and expected_config_sha is None:
        raise CustodyError('supervised preparation requires a pinned configuration digest')
    config_bytes = _read_private(config_path, 64 * 1024)
    config_sha = _digest(config_bytes)
    if expected_config_sha is not None and config_sha != expected_config_sha:
        raise CustodyError('private configuration changed between supervisor and child')
    config = _private_config(config_path, config_sha)
    root = _checked_path(config['frozen_root'])
    wrapper_path = _checked_path(Path(__file__))
    custody_root = _checked_path(wrapper_path.parents[1])
    site = _checked_path(os.environ.get('MLG_NATURAL_SITE', ''))
    _install_dependency_site(site)
    inspect_cache_staging(config['cache_dir'])
    inspect_evidence_destination(config['private_root'])
    binding_path = custody_root / 'results/distilgpt2-natural-agreement-v1-20261028/binding-v2.json'
    binding_bytes = _read_private(binding_path, 1024 * 1024)
    if _digest(binding_bytes) != BINDING_SHA256:
        raise CustodyError('local discovery binding differs')
    binding = json.loads(binding_bytes)
    if binding.get('source_commit') != SOURCE_COMMIT:
        raise CustodyError('discovery source identity differs')
    _install_frozen_finder(root, binding['source_sha256'])
    from looking_glass import natural_agreement_controller as control
    from scripts import v2_source_guards as guards
    SourceFileLease = guards.SourceFileLease
    if mode in ('validate', 'prepare', 'supervise-prepare'):
        control.await_start()
    initial = control.verify_source_inventory(root, SOURCE_COMMIT, binding['source_sha256'])
    core_files = [root / name for name in initial[1]] + [root / name for name in (
        'docs/PRETRAINED_NUMBER_PATCH_V1.md', 'docs/PRETRAINED_NUMBER_NATURAL_PREDICTION_CLARIFICATION.md',
        'results/distilgpt2-agreement-preflight-v1-20261028/frozen.json',
        'results/distilgpt2-agreement-preflight-v1-20261028/run/result.json',
        'results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32')]
    operation_files = [custody_root / relative for relative in SOURCE_SUFFIXES.values()]
    with SourceFileLease(root, core_files), SourceFileLease(custody_root, operation_files):
        if control.verify_source_inventory(root, *initial) != initial:
            raise CustodyError('frozen source changed before protection')
        runtime = runpy.run_path(str(root / 'tools/natural_agreement_cpu.py'), run_name='heldout_frozen_runtime')
        runtime['_startup']()
        if mode in ('validate', 'prepare'):
            runtime['_install_owner_audit_hook']()
        runtime['_configure_runtime']()
        if runtime['_bytes'](runtime['build_binding'](config['cache_dir'], expected_source=initial)) != binding_bytes:
            raise CustodyError('current runtime or checkpoint differs from discovery binding')
        from looking_glass import pretrained_exposures_v7 as v7
        from looking_glass import pretrained_number_commitments as commitments
        from looking_glass import agreement_heldout as heldout
        from looking_glass import agreement_preflight_v1 as preflight
        public = _public_from_config(config, custody_root, v7.canonical_inventory_bytes())
        wrapper_bytes = _read_private(wrapper_path, 8 * 1024 * 1024)
        check_public_inputs(public, commitments=commitments, heldout=heldout, wrapper_bytes=wrapper_bytes)
        _verify_local_operational_sources(public, custody_root)
        def observe_with_cache():
            observation = observe_current_inventory()
            observation['cache'] = inspect_cache_staging(config['cache_dir'])
            observation['evidence_destination'] = inspect_evidence_destination(config['private_root'])
            return observation
        observe_with_cache()
        if mode == 'validate':
            return {'status': 'validated_no_draw', 'aggregate_sha256': AGGREGATE_SHA256,
                    'not_checked': ['ledger capacity or phase claim', 'held-out commitment',
                                    'owner execution and scientific audit']}
        operational = json.loads(public.operational_binding)
        if mode in ('supervise-prepare', 'observe-prepare'):
            private_root = _checked_path(config['private_root'])
            attempt = private_root / ('natural-heldout-' + AGGREGATE_SHA256)
            supervisor_dir = private_root / ('natural-heldout-' + AGGREGATE_SHA256 + '-preparation-supervisor')
            if mode == 'observe-prepare':
                outer_command = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + sys.pycache_prefix,
                                 '-c', OUTER_BOOTSTRAP, str(wrapper_path), 'supervise-prepare',
                                 '--config', str(config_path), '--config-sha', config_sha]
                custody_state = [None, None, False, False]
                try:
                    return observe_supervisor_process(outer_command, attempt, supervisor_dir, public, wrapper_bytes,
                                                      config_path=config_path, config_sha256=config_sha,
                                                      job_factory=lambda: _outer_supervisor_job(guards),
                                                      custody_state=custody_state)
                finally:
                    _quarantine_state_if_needed(custody_state)
            supervisor = _load_file_module('_mlg_natural_heldout_supervisor',
                                           custody_root / SOURCE_SUFFIXES['supervisor'],
                                           operational['sources']['supervisor']['sha256'])
            command = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + sys.pycache_prefix,
                       '-c', runtime['BOOTSTRAP'], str(wrapper_path), 'prepare', '--config', str(config_path),
                       '--config-sha', config_sha]
            def audit_result(_directory, _exit_code, _reaped):
                return audit_preparation(attempt, public, wrapper_bytes)
            _supervise_preparation(supervisor, command, supervisor_dir, audit_result)
            verify_preparation_supervisor(attempt, supervisor_dir, public, wrapper_bytes,
                                          require_outer_observation=False)
            commitment = json.loads(_read_private(attempt / 'commitment.json', 8192))
            return {'status': 'supervised_sealed_private', 'heldout_commitment': commitment['heldout_commitment'],
                    'operational_binding_sha256': commitment['operational_binding_sha256'],
                    'preparation_config_sha256': config_sha,
                    'supervisor_record_sha256': _digest(_read_private(supervisor_dir / 'supervisor.json',
                                                                     2 * 1024 * 1024))}
        ledger = _load_file_module('_mlg_natural_heldout_ledger', custody_root / SOURCE_SUFFIXES['ledger'],
                                   operational['sources']['ledger']['sha256'])
        tokenizer = preflight._tokenizer(config['cache_dir'])
        claim = lambda phase, *, output_path: ledger.claim_phase(config['ledger_path'], phase, output_path=output_path)
        _, commitment = prepare_once(public, config['private_root'], commitments=commitments,
                                     heldout=heldout, tokenizer=tokenizer, wrapper_bytes=wrapper_bytes,
                                     claim_phase=claim, observe_inventory=observe_with_cache,
                                     config_bytes=config_bytes, config_path=config_path,
                                     nonce_source=secrets.token_bytes)
        return {'status': 'sealed_private', 'heldout_commitment': commitment['heldout_commitment'],
                'operational_binding_sha256': commitment['operational_binding_sha256']}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare one private, precommitted natural held-out split.')
    parser.add_argument('mode', choices=('validate', 'prepare', 'supervise-prepare', 'observe-prepare'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha')
    args = parser.parse_args()
    try:
        print(json.dumps(production_prepare(args.config, mode=args.mode,
                                            expected_config_sha=args.config_sha), sort_keys=True))
    except BaseException as error:
        # Do not print tracebacks or messages that might contain private case data.
        print(json.dumps({'status': 'incomplete', 'error_type': type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
