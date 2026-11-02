"""Reassemble and verify the saved agreement-direction evidence archive."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    index = json.loads((directory / 'archive-index.json').read_bytes())
    if index.get('schema') != 'agreement-direction-evidence-archive-v1':
        raise ValueError('unknown archive index')
    whole = hashlib.sha256()
    size = 0
    with args.output.open('xb') as output:
        for part in index['parts']:
            name = part['path']
            if type(name) is not str or Path(name).name != name:
                raise ValueError('unsafe part name')
            digest = hashlib.sha256()
            count = 0
            with (directory / name).open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
                    digest.update(block)
                    whole.update(block)
                    count += len(block)
                    size += len(block)
            if count != part['bytes'] or digest.hexdigest() != part['sha256']:
                raise ValueError('a part does not match its index')
    if size != index['archive_bytes'] or whole.hexdigest() != index['archive_sha256']:
        raise ValueError('the reconstructed archive does not match its index')
    print(json.dumps({'archive_bytes': size, 'sha256': whole.hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
