"""Verify and optionally extract the bounded public CPU trace-smoke archive."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    info = json.loads((root / 'archive-index.json').read_bytes())
    if (info.get('schema') != 'brokered-cpu-trace-smoke-archive-v1'
            or type(info.get('archive_bytes')) is not int or not 0 < info['archive_bytes'] <= 4 * 1024**2
            or type(info.get('parts')) is not list or not 1 <= len(info['parts']) <= 16
            or type(info.get('members')) is not list or not 1 <= len(info['members']) <= 64):
        raise ValueError('unsupported archive index')
    parts = []
    for index, part in enumerate(info['parts']):
        if (type(part) is not dict or part.get('path') != f'parts/part-{index:03}.txt'
                or type(part.get('bytes')) is not int or not 0 < part['bytes'] <= 400000):
            raise ValueError('invalid archive part')
        raw = (root / part['path']).read_bytes()
        if len(raw) != part['bytes'] or sha(raw) != part['sha256']:
            raise ValueError('archive part differs')
        parts.append(raw)
    archive = base64.b64decode(b''.join(parts), validate=True)
    if len(archive) != info['archive_bytes'] or sha(archive) != info['archive_sha256']:
        raise ValueError('archive differs')
    rows = info['members']
    names = []
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                *(f'LPT{i}' for i in range(1, 10))}
    for row in rows:
        if type(row) is not dict or type(row.get('path')) is not str or type(row.get('bytes')) is not int:
            raise ValueError('invalid member metadata')
        path = PurePosixPath(row['path'])
        if (path.is_absolute() or not path.parts or '\\' in row['path']
                or any(not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9._-]*', part)
                       or part.endswith('.') or part.split('.')[0].upper() in reserved for part in path.parts)
                or not 0 <= row['bytes'] <= 1024**2):
            raise ValueError('invalid member path or size')
        names.append(row['path'])
    if len(set(name.casefold() for name in names)) != len(names) or sum(row['bytes'] for row in rows) > 4 * 1024**2:
        raise ValueError('duplicate or oversized members')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        if len(z.namelist()) != len(names) or sorted(z.namelist()) != sorted(names):
            raise ValueError('archive inventory differs')
        for row in rows:
            entry = z.getinfo(row['path'])
            if entry.is_dir() or entry.file_size != row['bytes'] or entry.compress_size > 4 * 1024**2:
                raise ValueError('archive member size differs')
            with z.open(entry) as stream:
                raw = stream.read(row['bytes'] + 1)
            if len(raw) != row['bytes'] or sha(raw) != row['sha256']:
                raise ValueError('archive member differs')
        if args.extract is not None:
            args.extract.mkdir(parents=False, exist_ok=False)
            for row in rows:
                path = args.extract / row['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(row['path']) as stream:
                    raw = stream.read(row['bytes'] + 1)
                with path.open('xb') as output:
                    if len(raw) != row['bytes'] or output.write(raw) != len(raw):
                        raise OSError('short extraction')
    print(json.dumps({'status': 'verified', 'archive_sha256': info['archive_sha256'], 'members': len(rows)}))


if __name__ == '__main__':
    main()
