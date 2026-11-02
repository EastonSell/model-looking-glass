"""Stream-check a public selected-logit ZIP against the closed full evidence."""
if not __debug__:
    raise RuntimeError('run without Python optimization')

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

WORK = Path(r'E:\Codex\model-looking-glass\work')
ATTEMPT = WORK / 'bert-multisuffix-cpu-v1-20261029-01'
CONTROLLER = WORK / 'bert-multisuffix-controller-v1-20261029-01'
ZIP = WORK / 'bert-multisuffix-v1/results/bert-multisuffix-agreement-cpu-v1-20261029/selected-evidence-v1.zip'
OUTPUT = WORK / 'review-multisuffix-selected-archive-v1.json'
ZIP_SHA = '47b8106c4d71ce91b618f27f03a441af70f42911dcea75dddded70dd78194bb9'
FULL_SHA = '623c173a2c0244a3b3aac7978a2fb83a41cf8d1d9de892945b3f8b111d5d2149'
IDS = (2003, 2024, 2001, 2020)


def path_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    assert path_sha(ZIP) == ZIP_SHA
    with ZipFile(ZIP) as z:
        names = [i.filename for i in z.infolist()]
        assert len(names) == len(set(names)) == 9 and 'manifest.json' in names
        manifest = json.loads(z.read('manifest.json'))
        files = manifest['files']
        assert set(files) == set(names) - {'manifest.json'}
        assert manifest['schema'] == 'bert-multisuffix-selected-evidence-v1'
        assert manifest['rows'] == 1088 and manifest['shape'] == [1088, 4]
        assert [x['id'] for x in manifest['token_ids']] == list(IDS)
        assert manifest['dtype'] == 'little-endian float32' and manifest['full_logits_sha256'] == FULL_SHA
        selected = z.read('selected-logits.f32')
        for name, info in files.items():
            data = z.read(name)
            assert (len(data), hashlib.sha256(data).hexdigest()) == (info['bytes'], info['sha256'])
            if name != 'selected-logits.f32':
                source = (CONTROLLER if name == 'exit-observation.json' else ATTEMPT) / name
                assert data == source.read_bytes(), name
    assert len(selected) == 1088 * len(IDS) * 4
    full = hashlib.sha256()
    with (ATTEMPT / 'logits.f32').open('rb') as f:
        for row in range(1088):
            data = f.read(30522 * 4)
            assert len(data) == 30522 * 4
            full.update(data)
            expected = b''.join(data[index * 4:index * 4 + 4] for index in IDS)
            assert selected[row * 16:(row + 1) * 16] == expected, row
        assert not f.read(1)
    assert full.hexdigest() == FULL_SHA and path_sha(ZIP) == ZIP_SHA
    report = {'schema': 'independent-multisuffix-selected-archive-v1', 'status': 'saved_byte_consistency',
              'zip_sha256': ZIP_SHA, 'full_logits_sha256': FULL_SHA,
              'selected_logits_sha256': hashlib.sha256(selected).hexdigest(), 'rows': 1088,
              'token_ids': list(IDS), 'float_values_checked': 1088 * len(IDS),
              'scope': 'All selected float32 bytes and original included records match; the compact archive excludes full logits, hidden witnesses and independent process attestation.'}
    data = (json.dumps(report, sort_keys=True, indent=2) + '\n').encode()
    with OUTPUT.open('xb') as f:
        assert f.write(data) == len(data)
    print('selected_archive_review', hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
