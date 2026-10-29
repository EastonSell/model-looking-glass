"""Write a package hash index without including the index in itself."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
target = root / 'sha256s.json'
if target.exists():
    raise FileExistsError(target)
files = {}
for path in sorted(root.rglob('*')):
    if not path.is_file() or path == target:
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    files[path.relative_to(root).as_posix()] = {'bytes':path.stat().st_size, 'sha256':digest.hexdigest()}
raw = (json.dumps({'schema':'copy-behavior-intervention-package-sha256-v1','files':files},
                  sort_keys=True, indent=2) + '\n').encode()
with target.open('xb') as stream:
    stream.write(raw)
print(hashlib.sha256(raw).hexdigest())
