"""Reconstruct the published v4 screen and intervention evidence without a model."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='new directory for screen-data and intervention-data')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    index = json.loads((package / 'intervention-evidence-index.json').read_bytes())
    if index.get('schema') != 'vision-recognized-v4-intervention-archive-v1':
        raise ValueError('unexpected archive index')
    args.output.mkdir(parents=True, exist_ok=False)
    archive_path = args.output / 'intervention-evidence.zip'
    hasher = hashlib.sha256()
    size = 0
    with archive_path.open('xb') as out:
        for number, part in enumerate(index['parts']):
            if part['name'] != f'intervention-evidence.part-{number:03d}':
                raise ValueError('unexpected part order')
            raw = (package / part['name']).read_bytes()
            if len(raw) != part['bytes'] or sha(raw) != part['sha256']:
                raise ValueError('part digest or length mismatch')
            hasher.update(raw)
            size += len(raw)
            out.write(raw)
    if size != index['archive_bytes'] or hasher.hexdigest() != index['archive_sha256']:
        raise ValueError('archive digest or length mismatch')
    expected = {row['name']: row for row in index['members']}
    if len(expected) != len(index['members']) or len(expected) != 210:
        raise ValueError('unexpected intervention member inventory')
    intervention = args.output / 'intervention-data'
    intervention.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected) or {i.filename for i in infos} != set(expected):
            raise ValueError('intervention archive inventory mismatch')
        for info in infos:
            path = PurePosixPath(info.filename)
            if (len(path.parts) != 2 or path.parts[0] != 'data' or info.is_dir()
                    or path.name in ('.', '..') or '\\' in info.filename):
                raise ValueError('unexpected intervention member path')
            entry = expected[info.filename]
            if info.file_size != entry['bytes'] or info.file_size > 2_000_000:
                raise ValueError('intervention member length mismatch')
            raw = archive.read(info)
            if sha(raw) != entry['sha256']:
                raise ValueError('intervention member digest mismatch')
            (intervention / path.name).write_bytes(raw)
    screen_archive = package / 'screen-evidence.zip'
    custody = json.loads((package / 'screen-custody.json').read_bytes())
    if sha(screen_archive.read_bytes()) != custody['screen_archive_sha256']:
        raise ValueError('screen archive digest mismatch')
    screen = args.output / 'screen-data'
    screen.mkdir()
    with zipfile.ZipFile(screen_archive) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(infos) != 131 or len(set(names)) != len(infos):
            raise ValueError('unexpected screen member inventory')
        for info in infos:
            path = PurePosixPath(info.filename)
            if (len(path.parts) != 1 or info.is_dir() or path.name in ('.', '..')
                    or '\\' in info.filename or info.file_size > 2_000_000):
                raise ValueError('unexpected screen member')
            (screen / path.name).write_bytes(archive.read(info))
    print(json.dumps({'screen_files': len(names), 'intervention_files': len(expected),
                      'archive_sha256': hasher.hexdigest()}))


if __name__ == '__main__':
    main()
