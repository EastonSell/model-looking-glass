"""Bind a published exposed discovery archive before the fresh screen."""
import hashlib
import io
import json
from pathlib import Path
import zipfile


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_discovery_handoff(data, archive_path, index_raw, freeze_raw, observer_raw, audit_raw, readback_raw,
                             *, expected_manifest_sha256, expected_public_commit, expected_model_state):
    """Check saved archive, indexed parts, independent audit and direct exit.

    The public-readback record is cooperative; it is not execution or timing
    attestation. The production owner fixes the expected immutable commit.
    """
    index, freeze, observer, audit, readback = (json.loads(raw) for raw in
                                               (index_raw, freeze_raw, observer_raw, audit_raw, readback_raw))
    ids = [404350, 151497, 470798, 135116]
    if (freeze.get('schema') != 'vision-cls-occlusion-v1-discovery-freeze' or
            freeze.get('status') != 'complete' or freeze.get('manifest_sha256') != expected_manifest_sha256 or
            freeze.get('selected_ids') != ids or freeze.get('index_sha256') != _sha(index_raw) or
            freeze.get('observer_sha256') != _sha(observer_raw) or freeze.get('audit_sha256') != _sha(audit_raw)):
        raise ValueError('discovery freeze differs')
    if (observer.get('schema') != 'vision-cls-occlusion-v1-direct-observer' or
            observer.get('status') != 'complete' or observer.get('mode') != 'discovery' or
            observer.get('source_revision') != freeze.get('source_revision') or
            type(observer.get('exit_code')) is not int or observer['exit_code'] != 0 or
            observer.get('reaped') is not True or observer.get('timed_out') is not False or
            observer.get('forced_stop') is not False or 'error_type' not in observer or
            observer['error_type'] is not None or observer.get('terminal_sha256') != freeze.get('terminal_sha256')):
        raise ValueError('discovery observer differs')
    audit_result = audit.get('result', {})
    if (audit.get('schema') != 'vision-cls-occlusion-independent-discovery-audit-v1' or
            audit.get('synthetic') is not False or audit_result.get('manifest_sha256') != expected_manifest_sha256 or
            audit_result.get('selected') != ids or audit_result.get('phase_terminal_sha256') != freeze.get('terminal_sha256')):
        raise ValueError('discovery scientific audit differs')
    archive_path = Path(archive_path)
    if (index.get('schema') != 'vision-cls-occlusion-v1-discovery-archive-index' or
            type(index.get('archive_bytes')) is not int or not 0 < index['archive_bytes'] <= 100_000_000 or
            archive_path.stat().st_size != index['archive_bytes']):
        raise ValueError('discovery archive size differs')
    with archive_path.open('rb') as stream:
        raw = stream.read(index['archive_bytes'] + 1)
    if (len(raw) != index['archive_bytes'] or _sha(raw) != index.get('archive_sha256') or
            freeze.get('archive_sha256') != index['archive_sha256']):
        raise ValueError('discovery archive hash differs')
    parts = index.get('parts')
    if (type(parts) is not list or not 1 <= len(parts) <= 128 or
            readback.get('schema') != 'vision-cls-occlusion-v1-discovery-public-readback' or
            readback.get('status') != 'pass' or readback.get('public_commit') != expected_public_commit or
            readback.get('index_sha256') != _sha(index_raw) or
            readback.get('archive_sha256') != index['archive_sha256'] or
            type(readback.get('parts')) is not list or len(readback['parts']) != len(parts)):
        raise ValueError('discovery public readback differs')
    offset = 0
    for number, part in enumerate(parts):
        if (type(part) is not dict or part.get('file') != f'discovery-evidence.part-{number:03d}' or
                type(part.get('bytes')) is not int or not 0 < part['bytes'] <= 10_000_000):
            raise ValueError('discovery part differs')
        chunk = raw[offset:offset + part['bytes']]
        blob = hashlib.sha1(b'blob ' + str(len(chunk)).encode() + b'\0' + chunk).hexdigest()
        if (len(chunk) != part['bytes'] or part.get('sha256') != _sha(chunk) or
                readback['parts'][number] != {**part, 'git_blob': blob}):
            raise ValueError('discovery public readback part differs')
        offset += len(chunk)
    if offset != len(raw):
        raise ValueError('discovery part lengths differ')
    members = index.get('members')
    if (type(members) is not list or not 1 <= len(members) <= 512 or
            any(type(row) is not dict or type(row.get('file')) is not str or
                Path(row['file']).name != row['file'] for row in members) or
            len({row['file'] for row in members}) != len(members)):
        raise ValueError('discovery member inventory differs')
    data = Path(data)
    names = {row['file'] for row in members}
    if {path.name for path in data.iterdir() if path.is_file()} != names or any(not path.is_file() for path in data.iterdir()):
        raise ValueError('discovery evidence inventory differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        if len(zipped.namelist()) != len(names) or set(zipped.namelist()) != names:
            raise ValueError('discovery archive inventory differs')
        for row in members:
            info = zipped.getinfo(row['file'])
            if type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= 5_000_000 or info.file_size != row['bytes']:
                raise ValueError('discovery member size differs')
            member = zipped.read(row['file'])
            if _sha(member) != row.get('sha256') or member != (data / row['file']).read_bytes():
                raise ValueError('discovery archive member differs')
        started_raw, terminal_raw = zipped.read('started.json'), zipped.read('terminal.json')
    started, terminal = json.loads(started_raw), json.loads(terminal_raw)
    if (_sha(terminal_raw) != freeze.get('terminal_sha256') or
            started.get('schema') != 'vision-cls-occlusion-v1-start' or started.get('mode') != 'discovery' or
            started.get('selected_ids') != ids or started.get('manifest_sha256') != expected_manifest_sha256 or
            started.get('source', {}).get('revision') != freeze.get('source_revision') or
            started.get('model_state_sha256') != expected_model_state or
            terminal.get('schema') != 'vision-cls-occlusion-v1-terminal' or terminal.get('status') != 'complete' or
            terminal.get('mode') != 'discovery' or terminal.get('selected') != 4 or
            terminal.get('manifest_sha256') != expected_manifest_sha256 or
            terminal.get('source_revision') != freeze.get('source_revision') or
            terminal.get('runtime') != started.get('runtime') or
            terminal.get('model_state_sha256_before_after') != expected_model_state or
            type(terminal.get('cases')) is not list or [row.get('image_id') for row in terminal['cases']] != ids):
        raise ValueError('discovery terminal differs')
    sizes = {'matchable': 23, 'limited-dose': 23, 'dose-ineligible': 16, 'dose-unmatchable': 16}
    try:
        count = sum(sizes[row['dose']['status']] for row in terminal['cases'])
    except (KeyError, TypeError) as error:
        raise ValueError('discovery dose status differs') from error
    if (type(terminal.get('submissions')) is not int or terminal['submissions'] != count or
            type(terminal.get('saved_forwards')) is not int or terminal['saved_forwards'] != count or
            audit_result.get('forwards') != count):
        raise ValueError('discovery forward count differs')
    expected = {'started.json', 'terminal.json'}
    for case in terminal['cases']:
        ident = case['image_id']
        expected.add(f'case-{ident}.json')
        expected.update(f'initial-input-{ident}-{name}.bin' for name in ('clean', 'animal', 'background'))
    expected.update(f'forward-{number:03d}.{suffix}' for number in range(1, count + 1)
                    for suffix in ('json', 'bin'))
    if names != expected:
        raise ValueError('discovery complete evidence inventory differs')
    if audit_result.get('phase_start_sha256') != _sha(started_raw):
        raise ValueError('discovery audit start differs')
    for case in terminal['cases']:
        ident = case['image_id']
        if json.loads((data / f'case-{ident}.json').read_bytes()) != case:
            raise ValueError('discovery case record differs')
        for name in ('clean', 'animal', 'background'):
            if (data / f'initial-input-{ident}-{name}.bin').stat().st_size != 602112:
                raise ValueError('discovery input size differs')
    hashes = audit_result.get('forward_hashes')
    if type(hashes) is not list or len(hashes) != count:
        raise ValueError('discovery audited forward inventory differs')
    for number, expected_hash in enumerate(hashes, 1):
        meta_raw = (data / f'forward-{number:03d}.json').read_bytes()
        payload = (data / f'forward-{number:03d}.bin').read_bytes()
        metadata = json.loads(meta_raw)
        if (type(expected_hash) is not dict or type(expected_hash.get('submission')) is not int or
                expected_hash['submission'] != number or expected_hash.get('json_sha256') != _sha(meta_raw) or
                expected_hash.get('binary_sha256') != _sha(payload) or type(metadata.get('submission')) is not int or
                metadata['submission'] != number or metadata.get('sha256') != _sha(payload)):
            raise ValueError('discovery audited forward differs')
    return {'selected_ids': ids, 'runtime': started['runtime'], 'source': started['source'],
            'terminal_sha256': freeze['terminal_sha256'], 'archive_sha256': index['archive_sha256']}
