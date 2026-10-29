"""Reassemble the published study archive after verifying every part."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA256 = 'da204b88f5c209c5c1151a51ccd600b88087cf232dbb9e3de6b32f4b0f0f4a8b'
ARCHIVE_SHA256 = '1111e2630dc755273ff01a4a898b27b75d79171765c26489f47ee4ad513ebcfa'
ARCHIVE_BYTES = 88346152


def sha(path):
    digest, length = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
            length += len(data)
    return digest.hexdigest(), length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts-dir', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('parts manifest differs from the published pin')
    manifest = json.loads(raw)
    if (manifest.get('archive_sha256') != ARCHIVE_SHA256 or manifest.get('archive_bytes') != ARCHIVE_BYTES
            or len(manifest.get('parts', [])) != 11):
        raise ValueError('archive identity differs')
    for number, row in enumerate(manifest['parts'], 1):
        name = 'attempt-v2.tar.gz.part-%03d' % number
        if row['name'] != name or sha(args.parts_dir / name) != (row['sha256'], row['bytes']):
            raise ValueError('part differs: ' + name)
    digest, total = hashlib.sha256(), 0
    with args.output.open('xb') as out:
        for row in manifest['parts']:
            with (args.parts_dir / row['name']).open('rb') as stream:
                while data := stream.read(1024 * 1024):
                    if out.write(data) != len(data):
                        raise OSError('short output write')
                    digest.update(data)
                    total += len(data)
    if digest.hexdigest() != ARCHIVE_SHA256 or total != ARCHIVE_BYTES:
        raise ValueError('joined archive differs; partial output remains')
    print(json.dumps({'sha256': digest.hexdigest(), 'bytes': total}, sort_keys=True))


if __name__ == '__main__':
    main()
