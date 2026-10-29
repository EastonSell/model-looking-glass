"""Read verified DistilGPT2 projection rows directly from a pinned safetensors file."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

CHECKPOINT_SHA256="e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
KEY="transformer.h.3.attn.c_proj.weight"
HEADS=(2,10,7)
ROW_BYTES=768*4
HEAD_BYTES=64*ROW_BYTES

def verified_rows(path):
    with Path(path).open("rb") as stream:
        headlen_raw=stream.read(8)
        if len(headlen_raw)!=8:raise ValueError("truncated safetensors header")
        headlen=struct.unpack("<Q",headlen_raw)[0]
        if not 0<headlen<=16*1024*1024:raise ValueError("unexpected header size")
        header_raw=stream.read(headlen)
        if len(header_raw)!=headlen:raise ValueError("truncated safetensors header")
        header=json.loads(header_raw)
        desc=header[KEY]
        if desc["dtype"]!="F32" or desc["shape"]!=[768,768]:raise ValueError("unexpected projection weight layout")
        start,end=desc["data_offsets"]
        if type(start) is not int or type(end) is not int or start<0 or end-start!=768*ROW_BYTES:
            raise ValueError("invalid tensor offsets")
        base=8+headlen+start
        ranges={h:(base+h*HEAD_BYTES,base+(h+1)*HEAD_BYTES) for h in HEADS}
        captured={h:bytearray() for h in HEADS}
        prefix=bytearray();hasher=hashlib.sha256();offset=0
        stream.seek(0)
        while True:
            block=stream.read(1024*1024)
            if not block:break
            stop=offset+len(block);hasher.update(block)
            if offset<8+headlen:
                prefix.extend(block[:min(len(block),8+headlen-offset)])
            for h,(left,right) in ranges.items():
                low,high=max(left,offset),min(right,stop)
                if low<high:captured[h].extend(block[low-offset:high-offset])
            offset=stop
        if bytes(prefix)!=headlen_raw+header_raw or hasher.hexdigest()!=CHECKPOINT_SHA256:
            raise ValueError("checkpoint bytes do not match the pinned hash")
        if offset<8+headlen+end or any(len(captured[h])!=HEAD_BYTES for h in HEADS):
            raise ValueError("checkpoint projection slices are incomplete")
    return {h:bytes(captured[h]) for h in HEADS}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    for h in HEADS:parser.add_argument("--head-"+str(h),type=Path)
    args=parser.parse_args();rows=verified_rows(args.checkpoint)
    paths={h:getattr(args,"head_"+str(h)) for h in HEADS}
    if any(p is not None for p in paths.values()):
        if any(p is None for p in paths.values()):raise ValueError("provide all three retained slice files")
        for h in HEADS:
            if paths[h].read_bytes()!=rows[h]:raise ValueError(f"head {h} does not match pinned checkpoint")
    print(json.dumps({"status":"pass","checkpoint_sha256":CHECKPOINT_SHA256,
        "slice_sha256":{str(h):hashlib.sha256(rows[h]).hexdigest() for h in HEADS},
        "retained_slices_checked":all(p is not None for p in paths.values()),
        "scope":"Checkpoint byte consistency, not execution or interpreter attestation."},sort_keys=True))

if __name__=="__main__":main()
