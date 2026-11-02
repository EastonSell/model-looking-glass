"""Verify and join the recorded study archive parts into a new file."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA256 = '25eb3e46af608c31144db5c5231de39ad49267e37c0c0e3f30ed3e78b2d68f04'
ARCHIVE_SHA256 = '14accf1935da546d76214c06436ec7a1d94896fa22201782587d37a20490d688'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.parts_dir / 'parts-manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('part manifest differs from the published pin')
    document = json.loads(raw)
    parts = document.get('parts')
    if (type(parts) is not list or len(parts) != 13 or document.get('archive_bytes') != 104294509
            or document.get('archive_sha256') != ARCHIVE_SHA256):
        raise ValueError('archive part inventory differs')
    overall = hashlib.sha256()
    total = 0
    with args.output.open('xb') as output:
        for index, item in enumerate(parts, 1):
            name = f'study-attempt-v1.tar.gz.part-{index:03d}'
            if item.get('file') != name or type(item.get('size')) is not int or not 0 < item['size'] <= 8 * 1024**2:
                raise ValueError('archive part name or size differs')
            part_hash = hashlib.sha256()
            size = 0
            with (args.parts_dir / name).open('rb') as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > item['size']:
                        raise ValueError('archive part exceeds declared size')
                    part_hash.update(chunk)
                    overall.update(chunk)
                    if output.write(chunk) != len(chunk):
                        raise OSError('short archive write')
            if size != item['size'] or part_hash.hexdigest() != item['sha256']:
                raise ValueError('archive part digest or size differs')
            total += size
    if total != document['archive_bytes'] or overall.hexdigest() != document['archive_sha256']:
        raise ValueError('reassembled archive differs')
    print(json.dumps({'bytes': total, 'sha256': overall.hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
