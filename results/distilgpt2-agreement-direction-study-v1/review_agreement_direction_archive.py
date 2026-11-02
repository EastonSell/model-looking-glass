"""Stream-check agreement evidence archive and parts against closed originals."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

WORK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('custody_reader', WORK / 'review_agreement_direction_custody.py')
custody = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custody)
require = custody.require


def review(package, outer, exit_path, boundary_path):
    index_raw = custody.bounded(package / 'archive-index.json', 256 * 1024)
    index = custody.strict_json(index_raw)
    require(index['schema'] == 'agreement-direction-evidence-archive-v1', 'archive index schema differs')
    supervisor_raw = custody.bounded(outer / 'supervisor.json', 16 * 1024 * 1024)
    supervisor = custody.strict_json(supervisor_raw)
    require(custody.digest(supervisor_raw) == index['supervisor_sha256'], 'supervisor binding differs')
    expected = {'run/' + row['path']: (row['bytes'], row['sha256']) for row in supervisor['run_files']}
    require(len(expected) == len(supervisor['run_files']) == index['run_file_count'],
            'duplicate or missing run inventory')
    extras = {'pre-run.json': outer / 'pre-run.json',
              'child-custody.json': outer / 'child-custody.json',
              'supervisor.json': outer / 'supervisor.json',
              'launch-boundary-v1.json': boundary_path,
              'observer-exit-v1.json': exit_path,
              'claim-public.json': package / 'claim-public.json'}
    for name, path in extras.items():
        raw = custody.bounded(path, 16 * 1024 * 1024)
        expected[name] = (len(raw), custody.digest(raw))
    projection = custody.strict_json(custody.bounded(extras['claim-public.json'], 256 * 1024))
    require(projection == index['claim_projection'], 'claim projection differs')
    claims = list(outer.glob('agreement-direction-v1-*.claim'))
    require(len(claims) == 1, 'claim inventory differs')
    claim_raw = custody.bounded(claims[0], 256 * 1024)
    claim = custody.strict_json(claim_raw)
    require(len(claim_raw) == projection['original_claim_bytes']
            and custody.digest(claim_raw) == projection['original_claim_sha256']
            and claim['prelaunch_sha256'] == projection['prelaunch_sha256'] == index['prelaunch_sha256'],
            'claim projection binding differs')

    parts = index['parts']
    require(type(parts) is list and len(parts) == 13, 'part inventory differs')
    archive = package / 'evidence.tar.gz'
    archive_info = custody.ordinary(archive)
    require(archive_info.st_size == index['archive_bytes'], 'archive size differs')
    joined_hash, joined_size = hashlib.sha256(), 0
    with archive.open('rb') as combined:
        for i, part in enumerate(parts):
            require(part['path'] == 'evidence.tar.gz.part-%03d' % i, 'part order differs')
            path = package / part['path']
            info = custody.ordinary(path)
            require(info.st_size == part['bytes'], 'part size differs')
            current, size = hashlib.sha256(), 0
            with path.open('rb') as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    require(combined.read(len(chunk)) == chunk, 'parts differ from archive bytes')
                    size += len(chunk)
                    joined_size += len(chunk)
                    current.update(chunk)
                    joined_hash.update(chunk)
            require(size == part['bytes'] and current.hexdigest() == part['sha256'], 'part hash differs')
        require(combined.read(1) == b'', 'archive has extra bytes after parts')
    require(joined_size == index['archive_bytes'] and joined_hash.hexdigest() == index['archive_sha256'],
            'rejoined archive hash differs')

    observed, path_fields, pid_fields = set(), set(), set()
    path_re = re.compile(r'(?i)(?:^|\s)[a-z]:[\\/]')
    def privacy_scan(value, file_name, field=''):
        if type(value) is dict:
            for key, item in value.items():
                child = field + '.' + key if field else key
                if key.lower() in ('pid', 'owner_pid', 'parent_pid', 'child_pid', 'supervisor_pid'):
                    pid_fields.add(file_name + ':' + child)
                privacy_scan(item, file_name, child)
        elif type(value) is list:
            for item in value:
                privacy_scan(item, file_name, field + '[]')
        elif type(value) is str and path_re.search(value):
            path_fields.add(file_name + ':' + field)

    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            name = member.name
            parts_name = PurePosixPath(name)
            require(member.isfile() and name not in observed and name in expected
                    and not parts_name.is_absolute() and '..' not in parts_name.parts,
                    'unexpected or aliased archive member')
            size, expected_hash = expected[name]
            require(member.size == size, 'archive member size differs')
            data = stream.extractfile(member)
            require(data is not None, 'archive member is unreadable')
            h, total, capture = hashlib.sha256(), 0, bytearray()
            scan_json = name.endswith('.json') and size <= 4 * 1024 * 1024
            while True:
                chunk = data.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                h.update(chunk)
                if scan_json:
                    capture.extend(chunk)
            require(total == size and h.hexdigest() == expected_hash, 'archive member hash differs')
            if scan_json:
                privacy_scan(custody.strict_json(bytes(capture)), name)
            observed.add(name)
    require(observed == set(expected), 'archive is missing an expected member')
    return {'archive_index_sha256': custody.digest(index_raw), 'archive_sha256': joined_hash.hexdigest(),
            'archive_bytes': joined_size, 'verified_parts': len(parts), 'verified_members': len(observed),
            'verified_run_members': index['run_file_count'],
            'json_absolute_path_fields': sorted(path_fields), 'json_pid_fields': sorted(pid_fields),
            'scope': ('Exact saved-byte and selected JSON privacy scan only; '
                      'not proof of execution, complete privacy or scientific validity.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--outer', required=True, type=Path)
    parser.add_argument('--exit-record', required=True, type=Path)
    parser.add_argument('--boundary-record', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = review(args.package, args.outer, args.exit_record, args.boundary_record)
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if args.output is not None:
        with args.output.open('xb') as stream:
            require(stream.write(raw) == len(raw), 'short report write')
            stream.flush()
            os.fsync(stream.fileno())
    print(raw.decode('utf-8'), end='')


if __name__ == '__main__':
    main()
