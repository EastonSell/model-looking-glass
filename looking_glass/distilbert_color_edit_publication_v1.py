"""Create and verify compressed, byte-exact public stage evidence bundles."""
import gzip
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

from .distilbert_color_edit_records_v1 import MAX_BYTES

PART_BYTES = 3 * 1024 * 1024
ALLOWED = {'prelaunch.json', 'claim.json', 'worker-start.json', 'events.jsonl',
           'result.json', 'failure.json', 'exit.json', 'validated.json', 'supervisor-observation.json',
           'supervisor-observation-failure.json'}


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _file_metadata(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _write_exact(path, raw):
    with path.open('xb', buffering=0) as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short evidence write')
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != raw:
        raise OSError('saved evidence differs from intended bytes')


def pack_stage(stage_directory, output_directory, phase):
    """Write a new compressed archive and its index; originals stay untouched."""
    stage, output = Path(stage_directory), Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError('archive output already exists')
    if phase not in ('screen', 'development', 'evaluation') or stage.is_symlink() or not stage.is_dir():
        raise ValueError('invalid stage or phase')
    run_root = stage.parent.resolve(strict=True)
    target = output.parent.resolve(strict=True) / output.name
    if target == run_root or target.is_relative_to(run_root) or run_root.is_relative_to(target):
        raise ValueError('public output must be disjoint from the retained run')
    run_path = run_root / 'run.json'
    if run_path.is_symlink() or not run_path.is_file() or run_path.stat().st_size > 1024 * 1024:
        raise ValueError('invalid retained run record')
    run_raw = run_path.read_bytes()
    paths = list(stage.iterdir())
    if (not paths or any(p.is_symlink() or not p.is_file() or p.name not in ALLOWED for p in paths)
            or not {'claim.json', 'exit.json'}.issubset({p.name for p in paths})):
        raise ValueError('unexpected or incomplete stage artifact inventory')
    files = {p.name: p.read_bytes() for p in paths}
    if sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError('stage exceeds the retained evidence budget')
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            for name, raw in sorted(files.items()):
                header = tarfile.TarInfo(name)
                header.size, header.mode, header.mtime, header.uid, header.gid = len(raw), 0o644, 0, 0, 0
                header.uname = header.gname = ''
                archive.addfile(header, io.BytesIO(raw))
    raw_archive = compressed.getvalue()
    parts = []
    for index in range(0, len(raw_archive), PART_BYTES):
        chunk = raw_archive[index:index + PART_BYTES]
        name = 'part-%04d.gzchunk' % (index // PART_BYTES)
        parts.append({'name': name, **_file_metadata(chunk)})
    result = {'schema': 'distilbert-color-stage-archive-v1', 'phase': phase,
              'files': {name: _file_metadata(raw) for name, raw in sorted(files.items())},
              'run': {'name': 'run.json', **_file_metadata(run_raw)},
              'archive': _file_metadata(raw_archive), 'parts': parts,
              'scope': 'Compressed exact copies of cooperative stage records; not execution attestation.'}
    index_bytes = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if stage.name != phase or not (stage.parent / 'run.json').is_file():
        raise ValueError('stage must be part of its retained run directory')
    claim = strict_json(files['claim.json'])
    prior_indexes = claim.get('prior_public_index_bytes')
    if type(prior_indexes) is not int or prior_indexes < 0:
        raise ValueError('invalid prior archive-index accounting')
    from .distilbert_color_edit_runner_v1 import _inspect_run
    retained, _ = _inspect_run(stage.parent)
    if retained + prior_indexes + len(index_bytes) + len(run_raw) > MAX_BYTES:
        raise ValueError('public archive metadata would exceed the shared retained-byte cap')
    output.mkdir(exist_ok=False)
    for index in range(0, len(raw_archive), PART_BYTES):
        _write_exact(output / ('part-%04d.gzchunk' % (index // PART_BYTES)),
                     raw_archive[index:index + PART_BYTES])
    _write_exact(output / 'run.json', run_raw)
    _write_exact(output / 'index.json', index_bytes)
    if run_path.read_bytes() != run_raw or any((stage / name).read_bytes() != raw for name, raw in files.items()):
        raise RuntimeError('source stage changed during archival')
    return result


def git_blob(repository, commit, relative):
    if (type(commit) is not str or re.fullmatch('[0-9a-f]{40}', commit) is None or
            type(relative) is not str or '\\' in relative):
        raise ValueError('invalid immutable Git reference')
    path = PurePosixPath(relative)
    if path.is_absolute() or path.as_posix() != relative or any(p in ('', '.', '..') for p in path.parts):
        raise ValueError('invalid public evidence path')
    return subprocess.run(['git', '-C', str(repository), 'show', f'{commit}:{relative}'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60).stdout


def verify_publication(repository, commit, prefix, stage_directory, phase):
    """Reconstruct Git-published compressed bytes and compare every local file."""
    path = PurePosixPath(prefix)
    if (type(prefix) is not str or path.is_absolute() or path.as_posix() != prefix or
            '\\' in prefix or any(part in ('', '.', '..') for part in path.parts)):
        raise ValueError('invalid publication prefix')
    index_raw = git_blob(repository, commit, prefix + '/index.json')
    if len(index_raw) > 1024 * 1024:
        raise ValueError('archive index exceeds its budget')
    index = strict_json(index_raw)
    if (type(index) is not dict or index.get('schema') != 'distilbert-color-stage-archive-v1' or
            index.get('phase') != phase or type(index.get('parts')) is not list or
            not 1 <= len(index['parts']) <= 100 or type(index.get('files')) is not dict):
        raise ValueError('invalid public archive index')
    public_run = git_blob(repository, commit, prefix + '/run.json')
    if (len(public_run) > 1024 * 1024 or
            index.get('run') != {'name': 'run.json', **_file_metadata(public_run)} or
            public_run != (Path(stage_directory).parent / 'run.json').read_bytes()):
        raise ValueError('public run sidecar differs')
    chunks = []
    compressed_size = 0
    for number, item in enumerate(index['parts']):
        if type(item) is not dict or item.get('name') != 'part-%04d.gzchunk' % number:
            raise ValueError('archive parts are not in fixed order')
        raw = git_blob(repository, commit, prefix + '/' + item['name'])
        if len(raw) > PART_BYTES or _file_metadata(raw) != {'bytes': item.get('bytes'), 'sha256': item.get('sha256')}:
            raise ValueError('public archive part differs')
        compressed_size += len(raw)
        if compressed_size > MAX_BYTES:
            raise ValueError('public compressed archive exceeds its bound')
        chunks.append(raw)
    compressed = b''.join(chunks)
    if len(compressed) > MAX_BYTES or _file_metadata(compressed) != index.get('archive'):
        raise ValueError('public compressed archive differs')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as stream:
        uncompressed = stream.read(MAX_BYTES + 1024 * 1024 + 1)
    if len(uncompressed) > MAX_BYTES + 1024 * 1024:
        raise ValueError('public archive expands beyond its bound')
    stage = Path(stage_directory)
    local_paths = list(stage.iterdir())
    if (any(p.is_symlink() or not p.is_file() or p.name not in ALLOWED for p in local_paths) or
            set(index['files']) != {p.name for p in local_paths}):
        raise ValueError('local stage inventory differs from public archive')
    with tarfile.open(fileobj=io.BytesIO(uncompressed), mode='r:') as archive:
        members = archive.getmembers()
        if (len(members) != len(index['files']) or len({m.name for m in members}) != len(members)
                or any(not m.isfile() or m.name not in ALLOWED for m in members)):
            raise ValueError('invalid public archive members')
        for member in members:
            raw = archive.extractfile(member).read(MAX_BYTES + 1)
            if (_file_metadata(raw) != index['files'][member.name] or
                    raw != (stage / member.name).read_bytes()):
                raise ValueError('public stage bytes differ from the local original')
    return {'commit': commit, 'prefix': prefix, 'index_sha256': hashlib.sha256(index_raw).hexdigest(),
            'index_bytes': len(index_raw), 'run_sidecar_bytes': len(public_run), 'files': index['files']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('screen', 'development', 'evaluation'), required=True)
    args = parser.parse_args()
    print(json.dumps(pack_stage(args.stage, args.output, args.phase), sort_keys=True))


if __name__ == '__main__':
    main()
