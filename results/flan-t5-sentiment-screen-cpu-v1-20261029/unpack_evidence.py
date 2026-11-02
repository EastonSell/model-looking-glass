"""Reconstruct the closed screen's child directory from checked ZIP parts.

The caller supplies a trusted output parent and a new, nonexistent directory.
This checks the saved archive manifest, not the authenticity of the manifest
or the process that produced the records. On failure it leaves the partial
directory in place for inspection and never overwrites an existing file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='new child directory in a trusted parent')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / 'parts/manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'flan-sentiment-screen-tensors-v1':
        raise ValueError('unexpected manifest')
    entries = manifest['tensors']
    expected_names = [f'tensors/{i:04d}.f32' for i in range(330)]
    if [entry['name'] for entry in entries] != expected_names:
        raise ValueError('tensor inventory differs')
    inventory = {entry['name']: entry for entry in entries}
    if len(inventory) != 330:
        raise ValueError('duplicate tensor names')
    parts = manifest['parts']
    if len(parts) != len({part['name'] for part in parts}):
        raise ValueError('duplicate part names')
    seen = set()
    args.output.mkdir(exist_ok=False)
    (args.output / 'tensors').mkdir()
    for part in parts:
        name = part['name']
        if not re.fullmatch(r'part-[0-9]{3}\.zip', name):
            raise ValueError('invalid part name')
        data = (package / 'parts' / name).read_bytes()
        if len(data) != part['size'] or digest(data) != part['sha256']:
            raise ValueError('part bytes differ')
        from io import BytesIO
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if archive.namelist() != part['tensors']:
                raise ValueError('part members differ')
            for item in archive.infolist():
                if item.filename not in inventory or item.filename in seen:
                    raise ValueError('unexpected or repeated tensor')
                metadata = inventory[item.filename]
                if item.file_size != metadata['size']:
                    raise ValueError('tensor size differs')
                payload = archive.read(item)
                if len(payload) != metadata['size'] or digest(payload) != metadata['sha256']:
                    raise ValueError('tensor digest differs')
                destination = args.output / 'tensors' / Path(item.filename).name
                with destination.open('xb') as stream:
                    if stream.write(payload) != len(payload):
                        raise OSError('short tensor write')
                seen.add(item.filename)
    if seen != set(expected_names):
        raise ValueError('incomplete tensor inventory')
    for name in ('events.jsonl', 'terminal.json'):
        source = package / 'child-records' / name
        payload = source.read_bytes()
        metadata = manifest['records'][name]
        if len(payload) != metadata['size'] or digest(payload) != metadata['sha256']:
            raise ValueError('record bytes differ')
        with (args.output / name).open('xb') as stream:
            if stream.write(payload) != len(payload):
                raise OSError('short record write')
    print(json.dumps({'status': 'ok', 'parts': len(parts), 'tensors': len(seen)}, sort_keys=True))


if __name__ == '__main__':
    main()
