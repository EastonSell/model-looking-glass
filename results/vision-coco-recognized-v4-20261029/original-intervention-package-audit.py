"""Verify the staged, split v4 intervention archive against closed files."""
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work'
PACKAGE = WORK / 'teams/vision_recognized_v4_result/results/vision-coco-recognized-v4-20261029'
RUN = WORK / 'vision-recognized-v4-intervention-01'
OUTER = WORK / 'vision-recognized-v4-intervention-outer-01'
SCREEN = WORK / 'vision-recognized-v4-screen-stage-01'
SCREEN_RUN = WORK / 'vision-recognized-v4-screen-01'
SCREEN_OUTER = WORK / 'vision-recognized-v4-screen-outer-01'


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return path.read_bytes()


index = json.loads(read(PACKAGE / 'intervention-evidence-index.json'))
require(index['schema'] == 'vision-recognized-v4-intervention-archive-v1'
        and index['archive_bytes'] == 31812134
        and index['archive_sha256'] == '143507effc970e35d11a4aa2aa48dd1bb8b2ecbcbeba93fd6ff6e2f71d2a58f1'
        and len(index['parts']) == 38 and len(index['members']) == 210, 'index identity')
parts = []
for number, entry in enumerate(index['parts']):
    require(entry['name'] == f'intervention-evidence.part-{number:03}', 'part order')
    raw = read(PACKAGE / entry['name'])
    require(type(entry['bytes']) is int and len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], 'part digest')
    parts.append(raw)
joined = b''.join(parts)
require(len(joined) == index['archive_bytes'] and sha(joined) == index['archive_sha256'], 'archive digest')
expected_names = {'data/started.json', 'data/terminal.json'} | {f'data/pair-{i:02}.json' for i in range(4)}
expected_names |= {f'data/forward-{i:03}.{ext}' for i in range(1, 103) for ext in ('bin', 'json')}
entries = index['members']
require(len({entry['name'] for entry in entries}) == len(entries) and {entry['name'] for entry in entries} == expected_names,
        'indexed member inventory')
by_name = {entry['name']: entry for entry in entries}
path_pattern = re.compile(rb'(?:[A-Za-z]:\\{1,2}Users\\{1,2}|/home/|GPU-[0-9a-fA-F-]{8,})')
with tempfile.TemporaryDirectory(prefix='vision-v4-package-audit-', dir=WORK) as scratch:
    scratch = Path(scratch)
    with zipfile.ZipFile(io.BytesIO(joined)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == expected_names, 'ZIP member inventory')
        for name in names:
            raw = archive.read(name)
            entry = by_name[name]
            require(type(entry['bytes']) is int and len(raw) == entry['bytes'] and sha(raw) == entry['sha256']
                    and raw == read(RUN / name), 'member/source bytes ' + name)
            if name.endswith('.json'):
                require(not path_pattern.search(raw), 'path/device identifier in archived JSON ' + name)
            target = scratch.joinpath(*name.split('/'))
            target.parent.mkdir(exist_ok=True, parents=True)
            with target.open('xb') as stream:
                stream.write(raw)
            require(read(target) == raw, 'fresh extraction bytes ' + name)

copies = {
    'intervention-observer.json': OUTER / 'receipt.json',
    'intervention-controller-stderr.txt': OUTER / 'stderr.txt',
    'intervention-controller-stdout.txt': OUTER / 'stdout.txt',
    'intervention-source.json': RUN / 'source.json',
    'intervention-supervisor.json': RUN / 'supervisor.json',
    'screen-observer.json': SCREEN_OUTER / 'receipt.json',
    'screen-controller-stderr.txt': SCREEN_OUTER / 'stderr.txt',
    'screen-controller-stdout.txt': SCREEN_OUTER / 'stdout.txt',
    'screen-supervisor.json': SCREEN_RUN / 'supervisor.json',
    'screen-custody.json': SCREEN / 'screen-custody.json',
    'screen-evidence.zip': SCREEN / 'screen-evidence.zip',
    'screen-result.json': SCREEN / 'screen-result.json',
    'screen-independent-audit.json': WORK / 'vision-recognized-v4-screen-independent-audit-v1.json',
    'intervention-independent-audit.json': WORK / 'vision-recognized-v4-intervention-independent-audit-v1.json',
}
for name, original in copies.items():
    require(read(PACKAGE / name) == read(original), 'supporting evidence ' + name)
require(sha(read(PACKAGE / 'independent-scientific-replay.py')) == 'c3794990b2049ecd916f8e010d1fcae6d5b42c18a5cb196cd48039ecc778a598',
        'scientific replay digest')
require(sha(read(PACKAGE / 'observer-controller.py')) == '8b701fc82f19037085fcf27bcd3ffb6206f18b058f63915ead4cc79714352a02',
        'observer source digest')
for path in PACKAGE.iterdir():
    if path.is_file() and path.suffix.lower() in ('.json', '.txt', '.py', '.md'):
        require(not path_pattern.search(read(path)), 'absolute path/device identifier in package ' + path.name)
report = {'status': 'staged-archive-byte-consistency-verified', 'archive_sha256': index['archive_sha256'],
          'archive_bytes': index['archive_bytes'], 'parts': len(index['parts']), 'members': len(entries),
          'supporting_file_copies_verified': len(copies), 'fresh_extraction_matches_closed_data': True,
          'scope': ['actual supervisor receipts intentionally contain numerical process IDs',
                    'CPU-only byte/path check; no model or GPU replay',
                    'public Git blobs and final result note require a separate final-tree check']}
payload = (json.dumps(report, sort_keys=True, separators=(',', ':')) + '\n').encode()
output = WORK / 'vision-recognized-v4-intervention-package-independent-audit.json'
with output.open('xb') as stream:
    stream.write(payload)
print('package audit SHA-256:', sha(payload))
