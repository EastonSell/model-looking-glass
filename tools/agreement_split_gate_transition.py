"""Observe commitments and announce first forwards with an uncommitted peer.

This caller cannot verify a two-split comparison. It fails closed if the peer
is already committed; a separately reviewed verifier is required for that case.
"""
import hashlib
import json
from pathlib import Path
import re
import stat
import types
import urllib.request

HELPER_SHA256 = 'cab2c5cc652a67ee2437b408eeac7c06feb3b13cd95a2e6c91ec1fdf55cba8df'
HELPER_URL = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
              '479b41fa3ab1b3131644fa14d68d761ce9672f42/tools/agreement_split_gate_allocation.py')
PUBLIC_URL = re.compile(r'https://raw\.githubusercontent\.com/EastonSell/model-looking-glass/'
                        r'[0-9a-f]{40}/results/[A-Za-z0-9_./-]+\.json\Z')


def _read(path, limit):
    path = Path(path)
    info = path.lstat()
    if (not path.is_absolute() or not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or info.st_nlink != 1 or getattr(info, 'st_file_attributes', 0) & 1024
            or info.st_size > limit):
        raise ValueError('unsafe input')
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError('input changed')
    return raw


def _remote(url):
    with urllib.request.urlopen(url, timeout=20) as response:
        if response.geturl() != url:
            raise ValueError('public source redirected')
        return response.read(8 * 1024 * 1024 + 1)


def load_helper(path, *, reader=_remote):
    local = _read(path, 8 * 1024 * 1024)
    remote = reader(HELPER_URL)
    if (type(remote) is not bytes or remote != local
            or hashlib.sha256(local).hexdigest() != HELPER_SHA256):
        raise ValueError('helper source differs')
    helper = types.ModuleType('_mlg_gate_allocation_verified')
    helper.__file__ = str(path)
    exec(compile(local, str(path), 'exec', dont_inherit=True), helper.__dict__)
    return helper


def read_published(path, url, expected_sha, *, reader=_remote):
    if type(url) is not str or PUBLIC_URL.fullmatch(url) is None or '..' in url.split('/'):
        raise ValueError('immutable official result URL required')
    local = _read(path, 8192)
    remote = reader(url)
    if (type(remote) is not bytes or local != remote
            or hashlib.sha256(local).hexdigest() != expected_sha):
        raise ValueError('published result differs')
    return local


def operate(action, track, root, source_root, *, helper, source_reader,
            commitment_bytes=None, publication_check=None, receipt_bytes=None, receipt_check=None):
    if action not in ('commit', 'forward') or track not in helper.FREEZES:
        raise ValueError('unsupported transition')
    if receipt_bytes is not None or receipt_check is not None:
        raise ValueError('two-split clearance requires a separately reviewed verifier')
    root = helper.checked_root(root)
    helper.checked_lock(root)
    gate_module = helper.load_sources(source_root, reader=source_reader)
    root = helper.checked_root(root)
    helper.checked_lock(root)
    with gate_module.acquire(root, expected_root=root, timeout=30) as gate:
        helper.checked_root(root)
        helper.checked_lock(root)
        if action == 'commit':
            public = json.loads(commitment_bytes)
            gate.announce_commitment(track, helper.FREEZES[track], public['heldout_commitment'],
                hashlib.sha256(commitment_bytes).hexdigest(), public['exposure_inventory_sha256'],
                commitment_bytes=commitment_bytes, publication_check=publication_check)
            result = {'status': 'commitment_announced', 'track': track}
        else:
            result = gate.announce_forward(track)
        helper.checked_lock(root)
    helper.checked_root(root)
    helper.checked_lock(root)
    return result


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Coordinate a published commitment or possible first forward.')
    parser.add_argument('action', choices=('commit', 'forward'))
    parser.add_argument('--track', choices=('natural', 'injection'), required=True)
    for name in ('root', 'source-root', 'helper-source'):
        parser.add_argument('--' + name, required=True)
    for name in ('commitment-file', 'commitment-url', 'commitment-sha256',
                 'receipt-file', 'receipt-url', 'receipt-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    try:
        helper = load_helper(Path(args.helper_source), reader=_remote)
        commit_fields = (args.commitment_file, args.commitment_url, args.commitment_sha256)
        receipt_fields = (args.receipt_file, args.receipt_url, args.receipt_sha256)
        if args.action == 'commit':
            if not all(commit_fields) or any(receipt_fields):
                raise ValueError('commitment inputs required')
            public = read_published(*commit_fields, reader=_remote)
            result = operate('commit', args.track, Path(args.root), Path(args.source_root),
                helper=helper, source_reader=helper._remote, commitment_bytes=public,
                publication_check=lambda candidate: candidate == public)
            print(result['status'] + ':' + result['track'])
        else:
            if any(commit_fields) or any(receipt_fields):
                raise ValueError('forward inputs differ')
            result = operate('forward', args.track, Path(args.root), Path(args.source_root),
                helper=helper, source_reader=helper._remote)
            print('forward_announced:' + args.track + ':' + result['comparison'])
        return 0
    except BaseException:
        print('transition incomplete', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
