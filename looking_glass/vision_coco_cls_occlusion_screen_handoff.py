"""Validate a published complete fresh response screen before token patches.

The caller binds the source files and immutable public commit independently.
This reader assumes a quiescent workspace and does not attest execution.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from looking_glass.vision_coco_cls_occlusion_screen import verify_saved_response_screen


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_screen_handoff(images, data, archive_path, index_raw, result_raw, observer_raw, readback_raw,
                          *, manifest_sha256, public_commit, model_state):
    index, observer, readback = (json.loads(raw) for raw in (index_raw, observer_raw, readback_raw))
    archive_path, data = Path(archive_path), Path(data)
    if (index.get('schema') != 'vision-cls-occlusion-v1-screen-archive-index' or
            type(index.get('archive_bytes')) is not int or not 0 < index['archive_bytes'] <= 150_000_000 or
            archive_path.stat().st_size != index['archive_bytes']):
        raise ValueError('screen archive size differs')
    with archive_path.open('rb') as stream:
        archive = stream.read(index['archive_bytes'] + 1)
    if len(archive) != index['archive_bytes'] or sha(archive) != index.get('archive_sha256'):
        raise ValueError('screen archive hash differs')
    parts = index.get('parts')
    if (type(parts) is not list or not 1 <= len(parts) <= 192 or
            readback.get('schema') != 'vision-cls-occlusion-v1-screen-public-readback' or
            readback.get('status') != 'pass' or readback.get('public_commit') != public_commit or
            readback.get('index_sha256') != sha(index_raw) or
            readback.get('archive_sha256') != index['archive_sha256'] or
            type(readback.get('parts')) is not list or len(readback['parts']) != len(parts)):
        raise ValueError('screen public readback differs')
    offset = 0
    for i, part in enumerate(parts):
        if (type(part) is not dict or part.get('file') != f'screen-evidence.part-{i:03d}' or
                type(part.get('bytes')) is not int or not 0 < part['bytes'] <= 10_000_000):
            raise ValueError('screen part differs')
        chunk = archive[offset:offset + part['bytes']]
        blob = hashlib.sha1(b'blob ' + str(len(chunk)).encode() + b'\0' + chunk).hexdigest()
        if (len(chunk) != part['bytes'] or part.get('sha256') != sha(chunk) or
                readback['parts'][i] != {**part, 'git_blob': blob}):
            raise ValueError('screen public part differs')
        offset += len(chunk)
    if offset != len(archive):
        raise ValueError('screen part lengths differ')
    members = index.get('members')
    if (type(members) is not list or not 1 <= len(members) <= 512 or
            any(type(row) is not dict or type(row.get('file')) is not str or
                Path(row['file']).name != row['file'] for row in members) or
            len({row['file'] for row in members}) != len(members)):
        raise ValueError('screen member inventory differs')
    names = {row['file'] for row in members}
    if {p.name for p in data.iterdir() if p.is_file()} != names or any(not p.is_file() for p in data.iterdir()):
        raise ValueError('screen data inventory differs')
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if len(zipped.namelist()) != len(names) or set(zipped.namelist()) != names:
            raise ValueError('screen archive inventory differs')
        for row in members:
            info = zipped.getinfo(row['file'])
            if type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= 5_000_000 or info.file_size != row['bytes']:
                raise ValueError('screen member size differs')
            raw = zipped.read(row['file'])
            if sha(raw) != row.get('sha256') or raw != (data / row['file']).read_bytes():
                raise ValueError('screen member differs')
    started_raw, terminal_raw = (data / 'started.json').read_bytes(), (data / 'terminal.json').read_bytes()
    started, terminal = json.loads(started_raw), json.loads(terminal_raw)
    if (started.get('schema') != 'vision-cls-occlusion-v1-screen-start' or started.get('mode') != 'screen' or
            terminal.get('schema') != 'vision-cls-occlusion-v1-screen-terminal' or
            terminal.get('status') != 'complete' or terminal.get('mode') != 'screen' or
            started.get('manifest_sha256') != manifest_sha256 or terminal.get('manifest_sha256') != manifest_sha256 or
            started.get('image_ids') != [row['image_id'] for row in images] or
            started.get('model_state_sha256') != model_state or terminal.get('model_state_sha256_before_after') != model_state or
            type(started.get('source')) is not dict or type(started['source'].get('revision')) is not str or
            not re.fullmatch('[0-9a-f]{40}', started['source']['revision']) or
            terminal.get('source_revision') != started['source']['revision'] or
            type(started.get('runtime')) is not dict or terminal.get('runtime') != started['runtime'] or
            type(terminal.get('submissions')) is not int or type(terminal.get('saved_forwards')) is not int or
            terminal['submissions'] != terminal['saved_forwards'] or terminal.get('screen_result_sha256') != sha(result_raw)):
        raise ValueError('screen phase records differ')
    if (observer.get('schema') != 'vision-cls-occlusion-v1-screen-direct-observer' or
            observer.get('status') != 'complete' or observer.get('mode') != 'screen' or
            observer.get('source_revision') != started['source']['revision'] or
            type(observer.get('exit_code')) is not int or observer['exit_code'] != 0 or
            observer.get('reaped') is not True or observer.get('timed_out') is not False or
            observer.get('forced_stop') is not False or 'error_type' not in observer or observer['error_type'] is not None or
            observer.get('terminal_sha256') != sha(terminal_raw)):
        raise ValueError('screen direct observer differs')
    reconstructed = verify_saved_response_screen(images, data, expected_result=result_raw,
                                                  expected_manifest_sha256=manifest_sha256)
    selected = [row['image_id'] for row in reconstructed['selected']]
    expected_decision = 'pending' if selected else 'inconclusive'
    if (terminal['submissions'] != reconstructed['forwards'] or terminal.get('selected_ids') != selected or
            terminal.get('counts') != reconstructed['counts'] or
            terminal.get('decision') != expected_decision or
            json.loads(result_raw).get('decision') != expected_decision):
        raise ValueError('screen terminal selection differs')
    expected = {'started.json', 'terminal.json', 'screen-result.json'}
    expected.update(f'forward-{number:03d}.{suffix}' for number in range(1, reconstructed['forwards'] + 1)
                    for suffix in ('json', 'bin'))
    for ident in reconstructed['inputs']:
        expected.update(f'screen-input-{ident}-{name}.bin' for name in ('clean', 'animal', 'background'))
    if names != expected:
        raise ValueError('screen complete evidence inventory differs')
    return {**reconstructed, 'runtime': started['runtime'], 'source': started['source'],
            'terminal_sha256': sha(terminal_raw), 'archive_sha256': index['archive_sha256']}
