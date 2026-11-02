"""Stream a closed attempt to bounded parts and verify every archived file.

The parts are the only archive copy: no full tarball or reconstructed copy is
created. This handles public study evidence, not a privacy certification.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile

PART_SIZE = 8 * 1024 * 1024


class SplitWriter:
    def __init__(self, directory):
        self.directory, self.parts = directory, []
        self.archive_hash, self.archive_bytes = hashlib.sha256(), 0
        self.handle = self.part_hash = None
        self.part_bytes = 0

    def write(self, data):
        view = memoryview(data)
        written = len(view)
        self.archive_hash.update(view)
        self.archive_bytes += written
        while view:
            if self.handle is None:
                name = 'study-attempt-v1.tar.gz.part-%03d' % (len(self.parts) + 1)
                self.handle = (self.directory / name).open('xb')
                self.name, self.part_hash, self.part_bytes = name, hashlib.sha256(), 0
            piece = view[:PART_SIZE - self.part_bytes]
            if self.handle.write(piece) != len(piece):
                raise OSError('short part write')
            self.part_hash.update(piece)
            self.part_bytes += len(piece)
            view = view[len(piece):]
            if self.part_bytes == PART_SIZE:
                self.finish_part()
        return written

    def finish_part(self):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.parts.append({'name': self.name, 'bytes': self.part_bytes,
                               'sha256': self.part_hash.hexdigest()})
            self.handle = None

    def flush(self):
        if self.handle is not None:
            self.handle.flush()


class HashingReader:
    def __init__(self, stream):
        self.stream, self.hash, self.count = stream, hashlib.sha256(), 0

    def read(self, count):
        data = self.stream.read(count)
        self.hash.update(data)
        self.count += len(data)
        return data


class PartReader:
    def __init__(self, directory, parts):
        self.paths = iter(directory / row['name'] for row in parts)
        self.handle = None

    def read(self, count):
        if count < 0:
            raise ValueError('unbounded archive read')
        chunks = []
        while count:
            if self.handle is None:
                try:
                    self.handle = next(self.paths).open('rb')
                except StopIteration:
                    break
            data = self.handle.read(count)
            if not data:
                self.handle.close()
                self.handle = None
                continue
            chunks.append(data)
            count -= len(data)
        return b''.join(chunks)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def ordinary(path):
    info = path.lstat()
    if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('reparse point in attempt')
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise ValueError('hard-linked file in attempt')
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError('nonregular attempt entry')
    return info


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short manifest write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--supervisor-sha256')
    args = parser.parse_args()
    root = args.attempt.resolve(strict=True)
    output = args.output.absolute()
    if not root.is_dir() or output.exists() or output.resolve().is_relative_to(root):
        raise ValueError('attempt or fresh output directory is invalid')
    ordinary(root)
    entries = sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix())
    if len(entries) > 5000:
        raise ValueError('entry count exceeds bound')
    output.mkdir()
    parts_dir = output / 'parts'
    parts_dir.mkdir()
    sink = SplitWriter(parts_dir)
    records, directories = [], []
    with gzip.GzipFile(filename='', mode='wb', fileobj=sink, mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode='w|') as tar:
            for path in entries:
                info = ordinary(path)
                relative = path.relative_to(root).as_posix()
                member = tarfile.TarInfo('study-attempt-v1/' + relative)
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ''
                if stat.S_ISDIR(info.st_mode):
                    member.type, member.mode = tarfile.DIRTYPE, 0o755
                    tar.addfile(member)
                    directories.append(member.name)
                else:
                    member.size, member.mode = info.st_size, 0o644
                    with path.open('rb') as stream:
                        wrapped = HashingReader(stream)
                        tar.addfile(member, wrapped)
                    if wrapped.count != info.st_size or ordinary(path).st_size != info.st_size:
                        raise ValueError('file changed while packaging')
                    records.append({'name': member.name, 'bytes': info.st_size,
                                    'sha256': wrapped.hash.hexdigest()})
    sink.finish_part()
    if sum(x['bytes'] for x in records) > 512 * 1024**2:
        raise ValueError('attempt exceeds apparent-size bound')
    if args.supervisor_sha256 is not None:
        supervisor_raw = (root / 'supervisor.json').read_bytes()
        if hashlib.sha256(supervisor_raw).hexdigest() != args.supervisor_sha256:
            raise ValueError('supervisor receipt changed')
        rows = json.loads(supervisor_raw)['run_files']
        listed = {x['path']: (x['bytes'], x['sha256']) for x in rows}
        prefix = 'study-attempt-v1/run/'
        archived = {x['name'][len(prefix):]: (x['bytes'], x['sha256'])
                    for x in records if x['name'].startswith(prefix)}
        if len(listed) != len(rows) or listed != archived:
            raise ValueError('archive differs from supervisor file inventory')
    manifest = {'schema': 'agreement-substitution-archive-manifest-v1',
                'files': records, 'directories': directories}
    parts = {'schema': 'agreement-substitution-archive-parts-v1',
             'archive_bytes': sink.archive_bytes, 'archive_sha256': sink.archive_hash.hexdigest(),
             'part_limit_bytes': PART_SIZE, 'parts': sink.parts}
    expected = {x['name']: x for x in records}
    seen_files, seen_dirs = set(), set()
    source = PartReader(parts_dir, sink.parts)
    try:
        with tarfile.open(fileobj=source, mode='r|gz') as tar:
            for member in tar:
                if member.isdir():
                    if member.name not in directories or member.name in seen_dirs:
                        raise ValueError('unexpected archived directory')
                    seen_dirs.add(member.name)
                elif member.isfile() and member.name in expected and member.name not in seen_files:
                    row = expected[member.name]
                    digest, total = hashlib.sha256(), 0
                    with tar.extractfile(member) as stream:
                        while data := stream.read(1024 * 1024):
                            digest.update(data)
                            total += len(data)
                    if total != row['bytes'] or digest.hexdigest() != row['sha256']:
                        raise ValueError('archived file differs')
                    seen_files.add(member.name)
                else:
                    raise ValueError('unexpected archived member')
    finally:
        source.close()
    if seen_files != set(expected) or seen_dirs != set(directories):
        raise ValueError('archive omitted an entry')
    check = hashlib.sha256()
    for row in sink.parts:
        digest, length = hashlib.sha256(), 0
        with (parts_dir / row['name']).open('rb') as stream:
            while data := stream.read(1024 * 1024):
                check.update(data)
                digest.update(data)
                length += len(data)
        if length != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise ValueError('part bytes differ')
    if check.hexdigest() != parts['archive_sha256']:
        raise ValueError('joined archive hash differs')
    manifest_sha = write_json(output / 'archive-manifest.json', manifest)
    parts_sha = write_json(output / 'parts-manifest.json', parts)
    print(json.dumps({'archive_sha256': parts['archive_sha256'], 'archive_bytes': sink.archive_bytes,
                      'files': len(records), 'parts': len(sink.parts),
                      'manifest_sha256': manifest_sha, 'parts_manifest_sha256': parts_sha}, sort_keys=True))


if __name__ == '__main__':
    main()
