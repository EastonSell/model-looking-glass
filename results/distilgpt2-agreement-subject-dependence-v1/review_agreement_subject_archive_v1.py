"""Stream-check an archived public attempt against its closed original and part manifest."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile


def sha_file(path):
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def ordinary(path, directory=False):
    st = path.lstat()
    if (stat.S_ISLNK(st.st_mode) or getattr(st, 'st_file_attributes', 0) & 0x400
            or not (stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode))):
        raise ValueError('unexpected or aliased source entry')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'parts-dir', 'parts-manifest', 'archive-manifest', 'outer', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    for path in (a.archive, a.parts_manifest, a.archive_manifest):
        ordinary(path)
    ordinary(a.parts_dir, True)
    ordinary(a.outer, True)
    parts_raw, manifest_raw = a.parts_manifest.read_bytes(), a.archive_manifest.read_bytes()
    parts, manifest = json.loads(parts_raw), json.loads(manifest_raw)
    expected_archive = '14accf1935da546d76214c06436ec7a1d94896fa22201782587d37a20490d688'
    if parts['archive_sha256'] != expected_archive or parts['archive_bytes'] != 104294509:
        raise ValueError('parts manifest archive pin differs')
    if manifest['archive']['sha256'] != expected_archive or manifest['archive']['size'] != 104294509:
        raise ValueError('archive manifest pin differs')
    if sha_file(a.archive) != (104294509, expected_archive) or len(parts['parts']) != 13:
        raise ValueError('archive bytes or part count differs')
    combined = hashlib.sha256()
    for index, record in enumerate(parts['parts'], 1):
        name = f'study-attempt-v1.tar.gz.part-{index:03d}'
        if record['file'] != name:
            raise ValueError('part order differs')
        path = a.parts_dir / name
        ordinary(path)
        part_hash, size = hashlib.sha256(), 0
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                part_hash.update(chunk)
                combined.update(chunk)
                size += len(chunk)
        if size != record['size'] or part_hash.hexdigest() != record['sha256']:
            raise ValueError('part bytes differ')
    if combined.hexdigest() != expected_archive:
        raise ValueError('concatenated parts differ')
    entries = manifest['entries']
    indexed = {row['path']: row for row in entries}
    if len(indexed) != len(entries) or len(indexed) != 1709:
        raise ValueError('manifest member inventory differs')
    seen = set()
    files = directories = total = 0
    with tarfile.open(a.archive, mode='r|gz') as tar:
        for member in tar:
            name = member.name
            path = PurePosixPath(name)
            if (name in seen or path.is_absolute() or not path.parts
                    or path.parts[0] != 'study-attempt-v1' or any(part in ('.', '..') for part in path.parts)):
                raise ValueError('unexpected archive path')
            seen.add(name)
            row = indexed.get(name)
            if row is None:
                raise ValueError('archive member absent from manifest')
            original = a.outer.joinpath(*path.parts[1:])
            if member.isdir():
                if row != {'path': name, 'type': 'directory'}:
                    raise ValueError('directory declaration differs')
                ordinary(original, True)
                directories += 1
            elif member.isfile():
                ordinary(original)
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError('missing archive file payload')
                h, size = hashlib.sha256(), 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
                    size += len(chunk)
                if (row != {'path': name, 'type': 'file', 'size': size, 'sha256': h.hexdigest()}
                        or member.size != size or sha_file(original) != (size, h.hexdigest())):
                    raise ValueError('archive, manifest or original file differs')
                files += 1
                total += size
            else:
                raise ValueError('archive contains special member')
    if seen != set(indexed) or files != 1704 or directories != 5:
        raise ValueError('archive completeness differs')
    result = {'status': 'pass', 'archive_sha256': expected_archive, 'archive_bytes': 104294509,
              'archive_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
              'parts_manifest_sha256': hashlib.sha256(parts_raw).hexdigest(),
              'parts': 13, 'files': files, 'directories': directories, 'file_bytes': total,
              'scope': 'Public archive, manifests, ordered parts and closed original were byte-compared; no private split was read.'}
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with a.output.open('xb') as f:
        f.write(raw)
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}))


if __name__ == '__main__':
    main()
