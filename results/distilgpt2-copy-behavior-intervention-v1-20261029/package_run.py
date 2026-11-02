"""Make deterministic, independently restorable public chunks from a closed run."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--restore-script', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--allow-synthetic-fixture', action='store_true')
a = p.parse_args()
run = a.run.resolve(strict=True)
result_raw, audit_raw = (run / 'result.json').read_bytes(), a.audit.read_bytes()
result, audit = json.loads(result_raw), json.loads(audit_raw)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
if audit['result_sha256'] != sha(result_raw) or bool(audit['synthetic_fixture']) != a.allow_synthetic_fixture:
    raise ValueError('replay does not bind result or synthetic scope')
if not a.allow_synthetic_fixture and (audit['status'] != 'complete_saved_byte_pass' or not audit['checkpoint_rows_verified']):
    raise ValueError('production independent replay is incomplete')
if result['status'] != 'complete' or (run / 'failure.json').exists():
    raise ValueError('run is incomplete')
for name, key in [('manifest.json','manifest_sha256'), ('attempt.jsonl','journal_sha256'),
                  ('projection-weights.f32','weight_sha256')]:
    if sha((run/name).read_bytes()) != audit[key]:
        raise ValueError(f'independent replay does not bind {name}')
case_names = sorted({r['case_id'] + '.f32' for r in result['records']})
if not case_names or any(re.fullmatch(r'copy-behavior-\d{2}-[0-7]\.f32', n) is None for n in case_names):
    raise ValueError('unexpected raw tensor filename')
out = a.output_dir
out.mkdir(parents=True, exist_ok=False)
(out / 'raw').mkdir()
(out / 'records').mkdir()
shutil.copyfile(a.restore_script, out / 'restore_run.py')
records = []
for name, raw in [('manifest.json',(run/'manifest.json').read_bytes()),
                  ('attempt.jsonl',(run/'attempt.jsonl').read_bytes()),
                  ('result.json',result_raw),
                  ('projection-weights.f32',(run/'projection-weights.f32').read_bytes()),
                  ('independent-audit.json',audit_raw)]:
    target = 'records/' + name + '.gz'
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw)
    compressed = buf.getvalue()
    (out/target).write_bytes(compressed)
    records.append({'path':name, 'bytes':len(raw), 'sha256':sha(raw), 'gzip_path':target,
                    'gzip_bytes':len(compressed), 'gzip_sha256':sha(compressed)})

limit = 4824672
chunks, files, stream = [], [], hashlib.sha256()
pending = bytearray()
stream_offset = 0
def save_chunk(data):
    path = f'raw/part-{len(chunks):03d}.bin'
    raw = bytes(data)
    (out/path).write_bytes(raw)
    chunks.append({'path':path, 'offset':sum(c['bytes'] for c in chunks), 'bytes':len(raw),
                   'sha256':sha(raw), 'git_blob_sha1':hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()})
for name in case_names:
    raw = (run/name).read_bytes()
    files.append({'path':name, 'offset':stream_offset, 'bytes':len(raw), 'sha256':sha(raw)})
    stream.update(raw)
    stream_offset += len(raw)
    pending.extend(raw)
    while len(pending) >= limit:
        save_chunk(pending[:limit])
        del pending[:limit]
if pending:
    save_chunk(pending)
if stream_offset != result['counts']['retained'] or stream_offset != audit['action_bytes']:
    raise ValueError('tensor stream length differs from independently checked count')
index = {'schema':'copy-behavior-intervention-public-stream-v1',
         'source_result_sha256':sha(result_raw), 'independent_audit_sha256':sha(audit_raw),
         'synthetic_fixture':bool(audit['synthetic_fixture']), 'records':records,
         'tensor_stream':{'bytes':stream_offset, 'sha256':stream.hexdigest(), 'chunks':chunks, 'files':files}}
raw_index = (json.dumps(index, sort_keys=True, indent=2) + '\n').encode()
(out/'stream-index.json').write_bytes(raw_index)
print(json.dumps({'index_sha256':sha(raw_index), 'chunks':len(chunks), 'tensor_files':len(files),
                  'tensor_bytes':stream_offset, 'tensor_stream_sha256':stream.hexdigest()}))
