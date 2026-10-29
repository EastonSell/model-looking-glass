"""Fabricated checks for the published discovery-to-screen handoff."""
import hashlib
import json
import zipfile

import pytest


def test_discovery_archive_and_public_readback_must_match_closed_bytes(tmp_path):
    from looking_glass.vision_coco_cls_occlusion_discovery_handoff import verify_discovery_handoff

    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    canonical = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    ids, manifest, revision, commit = [404350, 151497, 470798, 135116], 'a' * 64, 'b' * 40, 'c' * 40
    started = {'schema': 'vision-cls-occlusion-v1-start', 'mode': 'discovery', 'selected_ids': ids,
               'manifest_sha256': manifest, 'source': {'revision': revision}, 'runtime': {'fake': True},
               'model_state_sha256': 'd' * 64}
    terminal = {'schema': 'vision-cls-occlusion-v1-terminal', 'status': 'complete', 'mode': 'discovery',
                'manifest_sha256': manifest, 'source_revision': revision, 'selected': 4,
                'cases': [{'image_id': value, 'dose': {'status': 'limited-dose'}} for value in ids],
                'runtime': {'fake': True}, 'model_state_sha256_before_after': 'd' * 64,
                'submissions': 92, 'saved_forwards': 92}
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'started.json').write_bytes(canonical(started))
    (data / 'terminal.json').write_bytes(canonical(terminal))
    archive = tmp_path / 'archive.zip'

    def envelope(forward_hashes):
        with zipfile.ZipFile(archive, 'w') as z:
            for path in sorted(data.iterdir()):
                z.writestr(path.name, path.read_bytes())
        blob = archive.read_bytes()
        part = {'file': 'discovery-evidence.part-000', 'bytes': len(blob), 'sha256': sha(blob)}
        members = [{'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}
                   for path in sorted(data.iterdir())]
        index = {'schema': 'vision-cls-occlusion-v1-discovery-archive-index', 'archive_bytes': len(blob),
                 'archive_sha256': sha(blob), 'parts': [part], 'members': members}
        index_raw = canonical(index)
        observer = {'schema': 'vision-cls-occlusion-v1-direct-observer', 'mode': 'discovery', 'status': 'complete',
                    'source_revision': revision, 'exit_code': 0, 'reaped': True, 'timed_out': False,
                    'forced_stop': False, 'error_type': None, 'terminal_sha256': sha(canonical(terminal))}
        observer_raw = canonical(observer)
        audit_raw = canonical({'schema': 'vision-cls-occlusion-independent-discovery-audit-v1', 'synthetic': False,
                               'result': {'manifest_sha256': manifest, 'selected': ids, 'forward_hashes': forward_hashes,
                                          'phase_start_sha256': sha(canonical(started)),
                                          'phase_terminal_sha256': sha(canonical(terminal)), 'forwards': 92}})
        freeze = {'schema': 'vision-cls-occlusion-v1-discovery-freeze', 'status': 'complete',
                  'manifest_sha256': manifest, 'selected_ids': ids, 'source_revision': revision,
                  'terminal_sha256': sha(canonical(terminal)), 'index_sha256': sha(index_raw),
                  'archive_sha256': sha(blob), 'observer_sha256': sha(observer_raw), 'audit_sha256': sha(audit_raw)}
        git_blob = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
        readback = {'schema': 'vision-cls-occlusion-v1-discovery-public-readback', 'status': 'pass',
                    'public_commit': commit, 'archive_sha256': sha(blob), 'index_sha256': sha(index_raw),
                    'parts': [{**part, 'git_blob': git_blob}]}
        return data, archive, index_raw, canonical(freeze), observer_raw, audit_raw, canonical(readback)

    kwargs = {'expected_manifest_sha256': manifest, 'expected_public_commit': commit, 'expected_model_state': 'd' * 64}
    with pytest.raises(ValueError, match='inventory'):
        verify_discovery_handoff(*envelope([]), **kwargs)

    for case in terminal['cases']:
        ident = case['image_id']
        (data / f'case-{ident}.json').write_bytes(canonical(case))
        for name in ('clean', 'animal', 'background'):
            (data / f'initial-input-{ident}-{name}.bin').write_bytes(b'\0' * 602112)
    hashes = []
    for number in range(1, 93):
        raw = f'fabricated-{number}'.encode()
        metadata = canonical({'submission': number, 'sha256': sha(raw)})
        (data / f'forward-{number:03d}.bin').write_bytes(raw)
        (data / f'forward-{number:03d}.json').write_bytes(metadata)
        hashes.append({'submission': number, 'binary_sha256': sha(raw), 'json_sha256': sha(metadata)})
    args = envelope(hashes)
    checked = verify_discovery_handoff(*args, **kwargs)
    assert checked['selected_ids'] == ids and checked['runtime'] == {'fake': True}
    readback = json.loads(args[-1])
    readback['parts'][0]['git_blob'] = '0' * 40
    with pytest.raises(ValueError, match='readback|part'):
        verify_discovery_handoff(*args[:-1], canonical(readback), **kwargs)
    (data / 'terminal.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='member|evidence'):
        verify_discovery_handoff(*args, **kwargs)
