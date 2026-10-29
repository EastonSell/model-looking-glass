"""Reassemble and verify the eight published copying-screen raw chunks."""
import argparse, hashlib, json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    index=json.loads(args.index.read_bytes())
    if index.get('schema')!='copy-behavior-screen-raw-chunks-v1' or index.get('rows')!=192 or index.get('row_bytes')!=201028:
        raise ValueError('unexpected raw-index schema')
    chunks=index['chunks']
    if len(chunks)!=8 or [x['first_row'] for x in chunks]!=list(range(0,192,24)):
        raise ValueError('unexpected raw chunk order')
    total=hashlib.sha256()
    with args.out.open('xb') as out:
        for item in chunks:
            name=item['path']
            if Path(name).name!=name or item['row_count']!=24 or item['bytes']!=24*201028:
                raise ValueError('invalid chunk identity or geometry')
            data=(args.index.parent/name).read_bytes()
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise ValueError('chunk length or SHA-256 differs')
            if out.write(data)!=len(data): raise OSError('short matrix write')
            total.update(data)
    if total.hexdigest()!=index['matrix_sha256']:
        raise ValueError('reassembled matrix SHA-256 differs')
    print(json.dumps({'bytes':192*201028,'sha256':total.hexdigest(),'rows':192}))


if __name__ == '__main__':
    main()