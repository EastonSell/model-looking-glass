"""Verify the published parts and rebuild their exact selected-logit gzip."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value, size = hashlib.sha256(), 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(64 * 1024), b''):
            size += len(block)
            value.update(block)
    return size, value.hexdigest()


def checked_write(stream, block):
    if stream.write(block) != len(block):
        raise OSError('short write; output is incomplete')


def rebuild(directory, output):
    with (directory / 'parts.json').open(encoding='utf-8') as source:
        metadata = json.load(source)
    if (metadata.get('schema') != 'bert-negation-selected-logit-parts-v1' or
            metadata.get('combined_bytes') != 24470665 or
            metadata.get('combined_sha256') != '4308884eefd1bd4f69da55d4e5f6e58f4df56381401b2a39cd7e02ea302fdc61' or
            len(metadata.get('parts', [])) != 6):
        raise ValueError('unexpected subset manifest')
    for index, part in enumerate(metadata['parts']):
        if part.get('name') != f'selected.f32.gz.part-{index:03d}':
            raise ValueError('unexpected part name or order')
        if digest(directory / part['name']) != (part.get('bytes'), part.get('sha256')):
            raise ValueError('part bytes or digest differ')
    whole, total = hashlib.sha256(), 0
    with output.open('xb') as result:
        for part in metadata['parts']:
            part_hash, part_size = hashlib.sha256(), 0
            with (directory / part['name']).open('rb') as source:
                for block in iter(lambda: source.read(64 * 1024), b''):
                    checked_write(result, block)
                    whole.update(block)
                    part_hash.update(block)
                    total += len(block)
                    part_size += len(block)
            if (part_size, part_hash.hexdigest()) != (part['bytes'], part['sha256']):
                raise ValueError('part changed while rebuilding; output is incomplete')
    if (total, whole.hexdigest()) != (metadata['combined_bytes'], metadata['combined_sha256']):
        raise ValueError('rebuilt output digest differs')
    if digest(output) != (metadata['combined_bytes'], metadata['combined_sha256']):
        raise ValueError('closed output digest differs')
    return {'bytes': total, 'sha256': whole.hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new file to create; never overwrites')
    args = parser.parse_args()
    print(json.dumps(rebuild(Path(__file__).resolve().parent, args.output), sort_keys=True))
