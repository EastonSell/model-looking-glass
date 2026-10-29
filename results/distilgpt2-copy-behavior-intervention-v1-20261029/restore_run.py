"""Restore a published copying-intervention run from its checked stream chunks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--package', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
package = a.package.resolve(strict=True)
index = json.loads((package / 'stream-index.json').read_bytes())
if index['schema'] != 'copy-behavior-intervention-public-stream-v1':
    raise ValueError('wrong stream schema')
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def safe_path(value):
    if (type(value) is not str or not value or '\\' in value or ':' in value
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('unsafe package member path')
    p = Path(value)
    if p.is_absolute() or p.drive or p.root:
        raise ValueError('unsafe package member path')
    return p
out = a.output_dir
out.mkdir(parents=True, exist_ok=False)
for member in index['records']:
    compressed = (package / safe_path(member['gzip_path'])).read_bytes()
    if sha(compressed) != member['gzip_sha256']:
        raise ValueError('compressed record digest mismatch')
    raw = gzip.decompress(compressed)
    if len(raw) != member['bytes'] or sha(raw) != member['sha256']:
        raise ValueError('restored record digest mismatch')
    with (out / safe_path(member['path'])).open('xb') as f:
        f.write(raw)

chunks = index['tensor_stream']['chunks']
stream_hash = hashlib.sha256()
offset = 0
for chunk in chunks:
    if chunk['offset'] != offset:
        raise ValueError('noncontiguous chunk offset')
    raw = (package / safe_path(chunk['path'])).read_bytes()
    if len(raw) != chunk['bytes'] or sha(raw) != chunk['sha256']:
        raise ValueError('chunk length or digest mismatch')
    stream_hash.update(raw)
    offset += len(raw)
if offset != index['tensor_stream']['bytes'] or stream_hash.hexdigest() != index['tensor_stream']['sha256']:
    raise ValueError('tensor stream digest mismatch')

chunk_number, chunk_data, within = 0, b'', 0
file_offset = 0
for member in index['tensor_stream']['files']:
    if member['offset'] != file_offset:
        raise ValueError('noncontiguous tensor file offset')
    h, remaining = hashlib.sha256(), member['bytes']
    with (out / safe_path(member['path'])).open('xb') as f:
        while remaining:
            if within == len(chunk_data):
                if chunk_number >= len(chunks):
                    raise ValueError('truncated tensor stream')
                chunk_data = (package / safe_path(chunks[chunk_number]['path'])).read_bytes()
                chunk_number += 1
                within = 0
            count = min(remaining, len(chunk_data) - within)
            block = chunk_data[within:within+count]
            f.write(block)
            h.update(block)
            within += count
            remaining -= count
    if h.hexdigest() != member['sha256']:
        raise ValueError('restored tensor file digest mismatch')
    file_offset += member['bytes']
if file_offset != index['tensor_stream']['bytes']:
    raise ValueError('tensor files do not cover stream')
print(json.dumps({'status':'restored', 'tensor_files':len(index['tensor_stream']['files']),
                  'tensor_bytes':file_offset, 'tensor_stream_sha256':stream_hash.hexdigest()}))
