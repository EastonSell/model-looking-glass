"""Reassemble the indexed discovery archive without overwriting a file."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_bytes())
    if index.get('schema') != 'vision-occlusion-v1-discovery-archive-index':
        raise ValueError('unexpected archive index')
    digest, total = hashlib.sha256(), 0
    with args.output.open('xb') as output:
        for position, item in enumerate(index['parts']):
            if item['file'] != f'discovery-evidence.part-{position:03d}':
                raise ValueError('invalid part order')
            raw = (args.index.parent / item['file']).read_bytes()
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('part size or hash differs')
            output.write(raw)
            total += len(raw)
            digest.update(raw)
        if total != index['archive_bytes'] or digest.hexdigest() != index['archive_sha256']:
            raise ValueError('archive size or hash differs')
    print(f'restored {total} bytes; sha256={digest.hexdigest()}')


if __name__ == '__main__':
    main()
