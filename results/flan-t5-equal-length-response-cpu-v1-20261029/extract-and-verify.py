"""Reconstruct the original run from indexed ZIP parts into a fresh directory."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

if len(sys.argv) != 2:
    raise SystemExit('usage: python extract-and-verify.py <fresh-output-directory|--verify-only>')
root = Path(__file__).resolve().parent
verify_only = sys.argv[1] == '--verify-only'
if not verify_only:
    dest = Path(sys.argv[1])
    if not dest.parent.is_dir():
        raise SystemExit('output parent must exist')
    dest.mkdir(exist_ok=False)
index = json.loads((root / 'source-index.json').read_bytes())
if index.get('schema') != 'mlg-flant5-equal-length-source-index-v1':
    raise ValueError('unknown source index')
files = index['files']
expected = {item['path']: item for item in files}
if len(expected) != len(files):
    raise ValueError('duplicate indexed path')
if expected['run/completed.json']['sha256'] != index['terminal_sha256']:
    raise ValueError('terminal binding differs')
seen = set()
part_names = set()
for part in index['parts']:
    name = part['path']
    if (not name.startswith('parts/') or Path(name).name != name.removeprefix('parts/')
            or name in part_names):
        raise ValueError('invalid archive path')
    part_names.add(name)
    data = (root / name).read_bytes()
    if len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
        raise ValueError('archive bytes differ from index')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            path = member.filename
            if (not path.startswith('run/') or Path(path).name != path.removeprefix('run/')
                    or path in seen or path not in expected or expected[path]['part'] != name):
                raise ValueError('unexpected archive member')
            if member.file_size != expected[path]['bytes']:
                raise ValueError('member length differs from index')
            raw = archive.read(member)
            item = expected[path]
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('member bytes differ from index')
            if not verify_only:
                output = dest / 'run' / Path(path).name
                output.parent.mkdir(exist_ok=True)
                with output.open('xb') as stream:
                    stream.write(raw)
            seen.add(path)
if seen != set(expected):
    raise ValueError('indexed members are missing')
print(json.dumps({'status': 'verified', 'files': len(seen), 'terminal_sha256': index['terminal_sha256']}))
