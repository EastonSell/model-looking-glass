"""Reconstruct a closed per-prompt copying run from its saved public bytes.

No model is loaded. This reader checks consistency of a complete saved run;
it does not attest execution, process cleanup, source custody or exposure.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

if not __debug__ or sys.byteorder != "little":
    raise RuntimeError("requires assertions and a little-endian host")
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run",required=True,type=Path)
parser.add_argument("--cases",required=True,type=Path)
parser.add_argument("--journal-sha256",required=True)
parser.add_argument("--result-sha256",required=True)
parser.add_argument("--synthetic-no-manifest",action="store_true",
                    help="test-only: accept an explicitly marked fabricated fixture without a manifest")
args=parser.parse_args()

def sha(b):return hashlib.sha256(b).hexdigest()
def load_pinned(p,d):
    raw=p.read_bytes()
    assert sha(raw)==d
    return raw
def f32(bits):
    sign=-1.0 if bits>>31 else 1.0
    exponent=(bits>>23)&255; fraction=bits&0x7fffff
    assert exponent!=255
    sig=fraction if exponent==0 else (1<<23)|fraction
    return sign*math.ldexp(float(sig),-149 if exponent==0 else exponent-150)
def values(raw):
    assert len(raw)%4==0
    return tuple(f32(n[0]) for n in struct.iter_unpack("<I",raw))
def norm(xs):return math.sqrt(math.fsum(x**2 for x in xs))
def close(a,b):
    if isinstance(a,float) or isinstance(b,float):
        return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    return a==b

def round_f32(x):
    """Binary search adjacent f32 values and use nearest-even ties."""
    assert math.isfinite(x)
    sign=0x80000000 if math.copysign(1.0,x)<0 else 0
    x=abs(x)
    if x==0:return sign
    lo,hi=0,0x7f7fffff
    assert x<=f32(hi)
    while lo<=hi:
        mid=(lo+hi)//2
        if f32(mid)<=x:lo=mid+1
        else:hi=mid-1
    lower=hi
    if f32(lower)==x:return sign|lower
    upper=lower+1
    dl=x-f32(lower);du=f32(upper)-x
    chosen=lower if dl<du or (dl==du and lower%2==0) else upper
    return sign|chosen

def stats(raw,target):
    v=values(raw);assert len(v)==50257 and 0<=target<len(v)
    maximum=max(v);z=math.log(math.fsum(math.exp(x-maximum) for x in v))
    ties=sum(x==maximum for x in v);first=v.index(maximum)
    return v,{"maximum":maximum,"logz_shift":z,"target_logit":v[target],
      "target_log_probability":(v[target]-maximum)-z,"target_rank":1+sum(x>v[target] for x in v),
      "top_unique_id":first if ties==1 else None,"top_representative_id":first,"top_tie_count":ties}

def compare(c,cs,a,ars):
    zc,za=cs["logz_shift"],ars["logz_shift"]
    mc,ma=cs["maximum"],ars["maximum"]
    target_delta=math.fsum((ars["target_logit"],-cs["target_logit"],mc,-ma,zc,-za))
    kl=[];tv=[]
    for x,y in zip(c,a):
        p=math.exp((x-mc)-zc);q=math.exp((y-ma)-za)
        kl.append(p*math.fsum((x,-y,-mc,ma,-zc,za)))
        tv.append(abs(p-q))
    top=("tied/indeterminate" if cs["top_unique_id"] is None or ars["top_unique_id"] is None else
         "changed" if cs["top_unique_id"]!=ars["top_unique_id"] else "unchanged")
    return {"target_logit_delta":ars["target_logit"]-cs["target_logit"],
            "target_log_probability_delta":target_delta,"kl_clean_to_arm":math.fsum(kl),
            "total_variation":math.fsum(tv)/2,"top_change":top}

case_raw=load_pinned(args.cases,"b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69")
cases=json.loads(case_raw)["cases"]
assert len(cases)==32 and len({c["case_id"] for c in cases})==32
result_raw=load_pinned(args.run/"result.json",args.result_sha256);result=json.loads(result_raw)
journal=load_pinned(args.run/"attempt.jsonl",args.journal_sha256)
events=[json.loads(x) for x in journal.splitlines()]
assert result.get("schema")=="distilgpt2-copy-head-prompt-dose-result-v1"
if args.synthetic_no_manifest:
    assert result.get("synthetic_fixture") is True and not (args.run/"manifest.json").exists()
    manifest_raw=None
else:
    manifest_raw=(args.run/"manifest.json").read_bytes();manifest=json.loads(manifest_raw)
    assert manifest.get("schema")=="distilgpt2-copy-head-prompt-dose-run-v1"
    assert manifest["cases_sha256"]==sha(case_raw) and result["manifest_sha256"]==sha(manifest_raw)
assert result["status"]=="complete" and not (args.run/"failure.json").exists()
offsets={c["case_id"]:0 for c in cases}
assert {p.name for p in args.run.glob("*.f32")}=={c["case_id"]+".f32" for c in cases}
position=0;records=[];summaries=[]
assert events[position]["event"]=="runtime_ready";position+=1
heads=(2,10,7);arms=("h2","h10","h7");wbytes=11*768*4

def action(case,kind):
    global position
    start,event,done=events[position:position+3];position+=3
    assert [(x["event"],x["case_id"],x["action"]) for x in (start,event,done)]==[(x,case["case_id"],kind) for x in ("start","raw_recorded","complete")]
    record={k:v for k,v in done.items() if k!="event"}
    assert event["raw"]==record["raw"]
    payload={}
    with (args.run/(case["case_id"]+".f32")).open("rb") as f:
        for name,segment in sorted(record["raw"].items(),key=lambda x:x[1]["offset"]):
            assert segment["offset"]==offsets[case["case_id"]]
            f.seek(segment["offset"]);b=f.read(segment["bytes"])
            assert len(b)==segment["bytes"] and sha(b)==segment["sha256"]
            offsets[case["case_id"]]+=len(b);payload[name]=b
    expected={"logits"} if kind in ("clean","clean_after") else {"logits","cproj_before","cproj_after","cproj_output"}
    assert set(payload)==expected and len(payload["logits"])==50257*4
    for name in expected-{"logits"}:assert len(payload[name])==wbytes
    v,s=stats(payload["logits"],case["target_id"])
    assert close(s,record["score"])
    records.append(record)
    return payload,v,s,record

for ci,case in enumerate(cases):
    clean,cv,cs,_=action(case,"clean")
    no,nv,ns,_=action(case,"noop")
    assert clean["logits"]==no["logits"] and no["cproj_before"]==no["cproj_after"]
    witness=no["cproj_before"];values(witness);noop_output=values(no["cproj_output"])
    source={h:witness[(10*768+64*h)*4:(10*768+64*(h+1))*4] for h in heads}
    original={h:values(source[h]) for h in heads}
    norms={h:norm(original[h]) for h in heads};q=min(norms.values())
    plans={}
    if q>2**-20:
        for h in heads:
            fraction=q/norms[h];beforebits=tuple(x[0] for x in struct.iter_unpack("<I",source[h]))
            afterbits=tuple(round_f32((1-fraction)*x) for x in original[h]);after=struct.pack("<64I",*afterbits)
            achieved=norm(x-y for x,y in zip(values(after),original[h]))
            plans[h]={"fraction":fraction,"delivered":after,"achieved":achieved,
                      "changed_elements":sum(x!=y for x,y in zip(beforebits,afterbits)),
                      "changed_bits":sum((x^y).bit_count() for x,y in zip(beforebits,afterbits))}
    ds=[plans[h]["achieved"] for h in heads] if plans else []
    eligible=bool(plans) and all(abs(x-q)<=1e-5*q for x in ds) and max(ds)-min(ds)<=1e-5*q
    reason=None if eligible else "realized dose mismatch" if plans else "dose floor"
    hs={}
    for h in heads:
        hs[str(h)]={"source_l2":norms[h],"source_bits_hex":source[h].hex()}
        if plans:
            hs[str(h)].update({k:plans[h][k] for k in ("fraction","achieved","changed_elements","changed_bits")})
            hs[str(h)]["delivered_bits_hex"]=plans[h]["delivered"].hex()
    desc={"status":"eligible" if eligible else "dose-ineligible","reason":reason,"q":q,"heads":hs,
          "plan_unavailable":None if plans else "plan unavailable: dose floor"}
    event=events[position];position+=1
    assert event["event"]=="dose_plan" and event["case_id"]==case["case_id"]
    assert close(desc,{k:v for k,v in event.items() if k not in ("event","case_id")})
    comparisons={}
    if eligible:
        shift=(4*(case["group"]-1)+case["variant"])%3
        for kind in arms[shift:]+arms[:shift]:
            head=int(kind[1:]);p,v,s,r=action(case,kind)
            assert p["cproj_before"]==witness
            begin=(10*768+64*head)*4
            assert p["cproj_after"]==witness[:begin]+plans[head]["delivered"]+witness[begin+256:]
            comp=compare(cv,cs,v,s);assert close(comp,r["comparison"])
            output1=noop_output[10*768:];output2=values(p["cproj_output"])[10*768:]
            assert close(norm(b-a for a,b in zip(output1,output2)),r["projection_output_delta_l2"])
            comparisons[kind]=comp
    end,_,_,_=action(case,"clean_after")
    assert end["logits"]==clean["logits"]
    summary={"case_id":case["case_id"],"group":case["group"],"variant":case["variant"],
             "status":desc["status"],"q":q,"plan":desc,"comparisons":comparisons}
    assert close(summary,result["cases"][ci]);summaries.append(summary)

assert position==len(events) and records==result["records"]
assert all((args.run/(k+".f32")).stat().st_size==size for k,size in offsets.items())
count=len(records);assert count==result["submitted_forwards"]==result["model_call_returns_observed"]==result["returned_forwards"]
assert sum(offsets.values())==result["raw_payload_bytes"]<=51573504
families=[]
for gi in range(8):
    rows=summaries[4*gi:4*gi+4]
    if any(row["status"]!="eligible" for row in rows):
        families.append({"status":"dose-ineligible","contrast":None});continue
    d={};d_logit={};case_deltas={}
    for kind in arms:
        delta=[row["comparisons"][kind]["target_log_probability_delta"] for row in rows]
        logit_delta=[row["comparisons"][kind]["target_logit_delta"] for row in rows]
        d[kind]=math.fsum((delta[0],-math.fsum(delta[1:])/3))
        d_logit[kind]=math.fsum((logit_delta[0],-math.fsum(logit_delta[1:])/3))
        case_deltas[kind]={"target_log_probability":delta,"target_logit":logit_delta}
    weighted=[(row["comparisons"]["h10"]["target_log_probability_delta"]-row["comparisons"]["h2"]["target_log_probability_delta"])/row["q"] for row in rows]
    families.append({"status":"evaluable","D":d,"target_logit_D":d_logit,"case_deltas":case_deltas,
                     "contrast":d["h10"]-d["h2"],"target_logit_contrast":d_logit["h10"]-d_logit["h2"],
                     "h10_minus_h7":d["h10"]-d["h7"],"h2_minus_h7":d["h2"]-d["h7"],
                     "dose_normalized_contrast":math.fsum((weighted[0],-math.fsum(weighted[1:])/3))})
assert close(families,result["families"])
evals=[f for f in families if f["status"]=="evaluable"]
n=len(evals);pos=sum(f["contrast"]>0 for f in evals)
mean=math.fsum(f["contrast"] for f in evals)/n if n else None
decision={"status":"ineligible" if n<6 else "criterion-met" if mean>0 and 4*pos>=3*n else "criterion-not-met",
          "evaluable_families":n,"strictly_positive":pos,"mean_contrast":mean}
assert close(decision,result["decision"])
print(json.dumps({"status":"complete_saved_byte_pass","case_sha256":sha(case_raw),"journal_sha256":sha(journal),
                  "result_sha256":sha(result_raw),"manifest_sha256":sha(manifest_raw) if manifest_raw is not None else None,
                  "synthetic_no_manifest":args.synthetic_no_manifest,"forwards":count,
                  "raw_bytes":sum(offsets.values()),"cases":summaries,"families":families,"decision":decision,
                  "scope":"Saved-byte arithmetic and landing consistency; not independent execution, process, source-custody or exposure attestation."},sort_keys=True,allow_nan=False))
