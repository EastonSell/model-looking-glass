"""Rebuild and verify the published, bounded 32-token GPU evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

INDEX_SHA256 = '664314b390a2a7222ad3118b5bae498ffdd1f44caaec97b95800c0a527ea43d3'
MAX_PART = 850 * 1024
MAX_ARCHIVE = 32 * 1024 * 1024
MAX_MEMBERS = 64


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    destination = args.destination
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError('destination must be a new absolute directory')
    raw = (root / 'archive-index.json').read_bytes()
    if len(raw) > 128 * 1024 or digest(raw) != INDEX_SHA256:
        raise ValueError('archive index differs from its published pin')
    index = json.loads(raw)
    parts, members = index['parts'], index['members']
    if not 0 < len(parts) <= MAX_MEMBERS or not 0 < len(members) <= MAX_MEMBERS:
        raise ValueError('archive inventory is outside the fixed bounds')
    if [entry['path'] for entry in parts] != [f'evidence.zip.{i:03d}' for i in range(1, len(parts) + 1)]:
        raise ValueError('archive parts are not contiguous')
    names = [entry['path'] for entry in members]
    if len(set(names)) != len(names):
        raise ValueError('archive contains repeated paths')
    for name in names:
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '\\' in name or ':' in name
                or any(part in ('.', '..', '') for part in path.parts)):
            raise ValueError('archive contains an unsafe path')
    if type(index['archive']['bytes']) is not int or not 0 < index['archive']['bytes'] <= MAX_ARCHIVE:
        raise ValueError('archive exceeds the fixed bound')
    destination.mkdir()
    archive = destination / 'rebuilt-evidence.zip'
    count = 0
    h = hashlib.sha256()
    with archive.open('xb') as output:
        for entry in parts:
            length = entry['bytes']
            if type(length) is not int or not 0 < length <= MAX_PART:
                raise ValueError('part size is outside the fixed bound')
            with (root / entry['path']).open('rb') as stream:
                data = stream.read(MAX_PART + 1)
            git = hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()
            if len(data) != length or digest(data) != entry['sha256'] or git != entry['git_blob_sha1']:
                raise ValueError('archive part failed integrity checks')
            count += len(data)
            if count > MAX_ARCHIVE:
                raise ValueError('archive exceeds the fixed bound')
            h.update(data)
            if output.write(data) != len(data):
                raise OSError('short archive write')
        output.flush()
        os.fsync(output.fileno())
    if count != index['archive']['bytes'] or h.hexdigest() != index['archive']['sha256']:
        raise ValueError('reconstructed archive differs')
    with zipfile.ZipFile(archive) as source:
        if source.namelist() != names:
            raise ValueError('archive member inventory differs')
        for entry, info in zip(members, source.infolist(), strict=True):
            if type(entry['bytes']) is not int or not 0 <= entry['bytes'] <= 7 * 1024 * 1024:
                raise ValueError('member exceeds its bound')
            if info.file_size != entry['bytes'] or info.is_dir():
                raise ValueError('archive member size differs')
            target = destination.joinpath(*PurePosixPath(entry['path']).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            count = 0
            with source.open(info) as reader, target.open('xb') as output:
                for chunk in iter(lambda: reader.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > entry['bytes']:
                        raise ValueError('archive member grew beyond its bound')
                    h.update(chunk)
                    if output.write(chunk) != len(chunk):
                        raise OSError('short member write')
                output.flush()
                os.fsync(output.fileno())
            if count != entry['bytes'] or h.hexdigest() != entry['sha256']:
                raise ValueError('archive member differs from its pin')
    print(json.dumps({'status': 'verified', 'members': len(members), 'archive_sha256': index['archive']['sha256']}))


if __name__ == '__main__':
    main()
