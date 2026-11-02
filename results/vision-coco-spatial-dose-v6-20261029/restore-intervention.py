"""Reassemble and verify the published v6 intervention records without a model."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


ARCHIVE_SHA256 = 'c796a6af6e0b01d7b491be729cab3072a485a4d81486965707bf43e15b24e216'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='new directory to receive the verified data/ directory')
    args = parser.parse_args()
    index = json.loads((args.package / 'intervention-evidence-index.json').read_bytes())
    require(index['schema'] == 'vision-spatial-dose-v6-intervention-archive-v1' and
            index['archive_sha256'] == ARCHIVE_SHA256 and index['archive_bytes'] <= 256_000_000,
            'archive index identity differs')
    accumulator = io.BytesIO()
    for n, part in enumerate(index['parts']):
        require(part['name'] == f'intervention-evidence.part-{n:03d}' and
                type(part['bytes']) is int and 0 < part['bytes'] <= 850_000, 'invalid part inventory')
        raw = (args.package / part['name']).read_bytes()
        require(len(raw) == part['bytes'] and hashlib.sha256(raw).hexdigest() == part['sha256'],
                'archive part differs')
        accumulator.write(raw)
    payload = accumulator.getvalue()
    require(len(payload) == index['archive_bytes'] and hashlib.sha256(payload).hexdigest() == ARCHIVE_SHA256,
            'reconstructed archive differs')
    members = {r['name']: r for r in index['members']}
    require(len(members) == len(index['members']) == 286 and
            all(re.fullmatch(r'data/[A-Za-z0-9_.-]+', name) for name in members),
            'member inventory differs')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'data').mkdir()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(members) and len({i.filename for i in infos}) == len(infos) and
                {i.filename for i in infos} == set(members), 'archive members differ')
        for info in infos:
            item = members[info.filename]
            require(type(item['bytes']) is int and 0 < item['bytes'] <= 1_000_000 and
                    info.file_size == item['bytes'] and not info.is_dir(), 'invalid member size')
            with archive.open(info) as source:
                raw = source.read(item['bytes'] + 1)
            require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
                    'archive member differs')
            with (args.output / info.filename).open('xb') as out:
                if out.write(raw) != len(raw):
                    raise OSError('short member write')
    print(json.dumps({'archive_sha256': ARCHIVE_SHA256, 'members': len(members),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
