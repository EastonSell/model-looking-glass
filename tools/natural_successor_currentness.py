"""Read-only, point-in-time official exposure observation for the successor.

The published operation must independently bind an already reviewed version,
source digest and canonical payload digest. This observer checks those exact
identities on official main and observes the same head at the beginning and
end. It cannot rule out an intervening change followed by a reset. It does not
derive the payload from the source, review inputs,
prove there are no unrecorded exposures, or lock unrelated model users.
"""
import hashlib
import json
import re
import urllib.request


class CurrentnessError(ValueError):
    """The official inventory cannot be observed at the bound identity."""


_API = 'https://api.github.com/repos/EastonSell/model-looking-glass'
_RAW = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_NAME = re.compile(r'pretrained_exposures_v([1-9][0-9]*)\.py\Z')


def _request(url, maximum, *, api=False):
    headers = {'User-Agent': 'Model-Looking-Glass-successor-custodian'}
    if api:
        headers['Accept'] = 'application/vnd.github+json'
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.geturl() != url:
            raise CurrentnessError('official observation redirected')
        raw = response.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise CurrentnessError('official response is empty or oversized')
    return raw


def _api_json(url):
    try:
        return json.loads(_request(url, 2 * 1024 * 1024, api=True))
    except (ValueError, UnicodeError, RecursionError):
        raise CurrentnessError('official observation is invalid JSON') from None


def _source(url):
    return _request(url, 8 * 1024 * 1024)


def _head(value):
    head = value.get('sha') if type(value) is dict else None
    if type(head) is not str or _COMMIT.fullmatch(head) is None:
        raise CurrentnessError('official main identity is unavailable')
    return head


def _pairs(items):
    output = {}
    for key, value in items:
        if key in output:
            raise CurrentnessError('duplicate inventory key')
        output[key] = value
    return output


def observe(*, version, source_sha256, inventory_sha256, inventory_bytes,
            api_reader=_api_json, source_reader=_source):
    """Observe reviewed source and payload identities without loading a model."""
    if (type(version) is not int or not 3 <= version <= 999999
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in (source_sha256, inventory_sha256))
            or type(inventory_bytes) is not bytes or not 1 <= len(inventory_bytes) <= 1000000
            or hashlib.sha256(inventory_bytes).hexdigest() != inventory_sha256):
        raise CurrentnessError('reviewed inventory binding differs')
    try:
        inventory = json.loads(inventory_bytes, object_pairs_hook=_pairs)
        canonical = json.dumps(inventory, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('ascii')
        if (type(inventory) is not dict or canonical != inventory_bytes
                or inventory.get('version') != 'distilgpt2-development-exposures-v' + str(version)):
            raise CurrentnessError('canonical inventory version differs')
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise CurrentnessError('canonical inventory is invalid') from None
    head = _head(api_reader(_API + '/commits/main'))
    listing = api_reader(_API + '/contents/looking_glass?ref=' + head)
    if type(listing) is not list or len(listing) >= 1000:
        raise CurrentnessError('official source listing is incomplete')
    versions = []
    for item in listing:
        if type(item) is not dict or item.get('type') != 'file':
            continue
        name = item.get('name')
        match = _NAME.fullmatch(name) if type(name) is str else None
        if match:
            versions.append(int(match.group(1)))
    if not versions or max(versions) != version:
        raise CurrentnessError('official latest exposure version changed')
    url = _RAW + head + '/looking_glass/pretrained_exposures_v' + str(version) + '.py'
    raw = source_reader(url)
    if (type(raw) is not bytes or not 0 < len(raw) <= 8 * 1024 * 1024
            or hashlib.sha256(raw).hexdigest() != source_sha256):
        raise CurrentnessError('official exposure source differs')
    if _head(api_reader(_API + '/commits/main')) != head:
        raise CurrentnessError('official main changed during observation')
    return {'schema': 'mlg-natural-successor-currentness-observation-v1', 'head': head,
            'version': version, 'source_sha256': source_sha256, 'inventory_sha256': inventory_sha256}
