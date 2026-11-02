"""Restore a replayable closed screen from the published package."""
import argparse, gzip, hashlib, json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    package=args.package
    index=json.loads((package/'sha256s.json').read_bytes())
    if index.get('schema')!='copy-behavior-screen-package-sha256-v1':
        raise ValueError('unexpected package hash index')
    files=index['files']
    def checked(name):
        data=(package/name).read_bytes()
        if digest(data)!=files[name]: raise ValueError('published file hash differs: '+name)
        return data
    raw_index=json.loads(checked('raw/index.json'))
    chunks=raw_index['chunks']
    if (raw_index.get('schema')!='copy-behavior-screen-raw-chunks-v1' or raw_index.get('rows')!=192
            or raw_index.get('row_bytes')!=201028 or len(chunks)!=8
            or [v['first_row'] for v in chunks]!=list(range(0,192,24))):
        raise ValueError('invalid raw chunk geometry')
    args.out.mkdir(parents=True,exist_ok=False)
    matrix=hashlib.sha256()
    with (args.out/'screen.f32').open('xb') as out:
        for item in chunks:
            name=item['path']
            if Path(name).name!=name or item['row_count']!=24 or item['bytes']!=24*201028:
                raise ValueError('invalid raw chunk identity')
            data=checked('raw/'+name)
            if len(data)!=item['bytes'] or digest(data)!=item['sha256']:
                raise ValueError('raw chunk digest differs')
            if out.write(data)!=len(data): raise OSError('short raw matrix write')
            matrix.update(data)
    expected=index['original_uncompressed']['screen.f32']
    if matrix.hexdigest()!=expected['sha256'] or matrix.hexdigest()!=raw_index['matrix_sha256']:
        raise ValueError('reassembled matrix digest differs')
    for packed,name in [('attempt.jsonl.gz','attempt.jsonl'),('result.json.gz','result.json')]:
        data=gzip.decompress(checked(packed))
        if digest(data)!=index['original_uncompressed'][name]['sha256']:
            raise ValueError('uncompressed digest differs: '+name)
        with (args.out/name).open('xb') as out: out.write(data)
    manifest=checked('run-manifest.json')
    with (args.out/'manifest.json').open('xb') as out: out.write(manifest)
    print(json.dumps({'status':'restored','rows':192,'matrix_sha256':matrix.hexdigest()}))


if __name__ == '__main__':
    main()