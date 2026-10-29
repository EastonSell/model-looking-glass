"""Source-verified operator for the shared split-allocation announcement.

Run only from a bootstrap that verifies this published file's SHA-256 before
execution. The operator never draws a nonce, reads private cases or runs a model.
Its digest-only marker coordinates separate tracks; it is not an attestation
against another process using the same account.
"""
import argparse
import hashlib
import os
from pathlib import Path
import stat
import sys
import types
import urllib.request


SOURCE_COMMIT = '02e1c5d5c79813e3d67bbd0468b599926b4b9f83'
GATE_ROOT_SHA256 = '0ede711a758f4cd5a7b3651d12aa9b81ac7f0cac7e6419a309411e24c9f80e70'
GATE_ROOT_IDENTITY = (1219152768, 4503599632054271)
FREEZES = {'natural': '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb',
           'injection': 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'}
SOURCE_HASHES = {
    'pretrained_exposures': '6ec5a00a6950a739ddfab1c0335065609aead275f0f393323ecb22699d93a9cd',
    'pretrained_exposures_v2': 'fa9014fb5cce2c3d8be3497a561754e47115ec07b9be3a896bb3bf1b20f8c474',
    'pretrained_exposures_v3': '2546bb672a461bfea10f0aed5e83c3a2110b2c898b56ff17542cb4718fc32886',
    'pretrained_number_commitments': '635f25553c20dd437f5cc4a8a9ca3a7a1453e7f8a517875fe75eb01ede2d5f35',
    'device_lease': '7f52ad238a1c9e8800c3d1ebf3b6969edd810a02754d1462c34f06cf783b57bf',
    'injection_split_custody': 'c2679ec61b82aca283bf20a5eb5e3e550ee80e2f31c40608f0b559ad66ab8d04',
    'agreement_split_gate': '6cf493062141e842aef2dc1584ce1ddbd2119e5427e867f3dde820fccb313d30',
}


class CallerError(RuntimeError):
    """The shared ordering state cannot be safely used."""


def _ordinary(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise CallerError('an absolute non-traversing path is required')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise CallerError('a path crosses a link or reparse point')
    return path


def _remote(name):
    url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + SOURCE_COMMIT +
           '/looking_glass/' + name + '.py')
    with urllib.request.urlopen(url, timeout=20) as response:
        if response.geturl() != url:
            raise CallerError('published source redirected')
        return response.read(8 * 1024 * 1024 + 1)


def load_sources(source_root, *, reader=_remote):
    """Capture and verify every dependency before any project code executes."""
    root = _ordinary(source_root)
    captured = {}
    for name, expected in SOURCE_HASHES.items():
        path = _ordinary(root / 'looking_glass' / (name + '.py'))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 8 * 1024 * 1024:
            raise CallerError('operational source is unsafe or oversized')
        raw = path.read_bytes()
        remote = reader(name)
        if (type(remote) is not bytes or len(raw) != info.st_size or raw != remote
                or hashlib.sha256(raw).hexdigest() != expected):
            raise CallerError('operational source differs from its published bytes')
        captured[name] = (path, raw)
    if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
        raise CallerError('project modules were already loaded')
    package = types.ModuleType('looking_glass')
    package.__package__ = 'looking_glass'
    package.__path__ = []
    sys.modules['looking_glass'] = package
    for name, (path, raw) in captured.items():
        qualified = 'looking_glass.' + name
        module = types.ModuleType(qualified)
        module.__file__ = str(path)
        module.__package__ = 'looking_glass'
        sys.modules[qualified] = module
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return sys.modules['looking_glass.agreement_split_gate']


def checked_root(path):
    root = _ordinary(path)
    info = root.lstat()
    fingerprint = hashlib.sha256(os.path.normcase(os.path.abspath(str(root))).encode('utf-8')).hexdigest()
    if (fingerprint != GATE_ROOT_SHA256 or not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != GATE_ROOT_IDENTITY):
        raise CallerError('the shared gate root identity differs')
    return root


def operate(action, track, root, source_root, *, reader=_remote):
    if action not in ('inspect', 'allocate') or track not in FREEZES:
        raise CallerError('unsupported action or track')
    root = checked_root(root)
    gate_module = load_sources(source_root, reader=reader)
    root = checked_root(root)
    # The lease checks the root again and guards the marker transition.
    with gate_module.acquire(root, expected_root=root, timeout=30) as gate:
        checked_root(root)
        if action == 'allocate':
            gate.announce_allocation(track, FREEZES[track])
    return {'status': 'inspected' if action == 'inspect' else 'allocation_announced', 'track': track}


def main():
    parser = argparse.ArgumentParser(description='Announce a separately frozen split allocation.')
    parser.add_argument('action', choices=('inspect', 'allocate'))
    parser.add_argument('--track', choices=tuple(FREEZES), required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-root', required=True)
    args = parser.parse_args()
    try:
        result = operate(args.action, args.track, Path(args.root), Path(args.source_root))
        print(result['status'] + ':' + result['track'])
    except BaseException as error:
        print('incomplete:' + type(error).__name__, file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
